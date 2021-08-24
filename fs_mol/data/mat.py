from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from dpu_utils.utils import RichPath

from fs_mol.data import (
    FSMolBatcher,
    FSMolTask,
    MoleculeDatapoint,
    default_reader_fn,
)

# Assumes that MAT is in the python lib path:
from featurization.data_utils import construct_dataset, load_data_from_smiles, mol_collate_func


@dataclass(frozen=True)
class FSMolMATBatch:
    node_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    distance_matrix: torch.Tensor


@dataclass(frozen=True)
class MATMoleculeDatapoint(MoleculeDatapoint):
    mat_features: np.ndarray


def mat_process_samples(samples: List[MoleculeDatapoint]) -> List[MATMoleculeDatapoint]:
    # Set `one_hot_formal_charge` for compatibilitiy with pretrained weights (see README.md in MAT).
    all_features, _ = load_data_from_smiles(
        x_smiles=[sample.smiles for sample in samples],
        labels=[sample.bool_label for sample in samples],
        one_hot_formal_charge=True,
    )

    # MAT can internally decide that there is something wrong with a sample and reject it. Our
    # dataset is clean, so this shouldn't happen (or at least shouldn't happen silently!).
    if len(all_features) < len(samples):
        raise ValueError("MAT rejected some samples; can't continue, as that may skew results.")

    # Note that `sample.__dict__` is almost like `dataclasses.asdict(sample)`, but shallow, i.e. it
    # doesn't dict-ify the inner dataclass describing molecular graph.
    return [
        MATMoleculeDatapoint(mat_features=features, **sample.__dict__)
        for sample, features in zip(samples, all_features)
    ]


def mat_batcher_init_fn(batch_data: Dict[str, Any]):
    batch_data["mat_features"] = []


def mat_batcher_add_sample_fn(
    batch_data: Dict[str, Any], sample_id: int, sample: MATMoleculeDatapoint
):
    batch_data["mat_features"].append(sample.mat_features)


def mat_batcher_finalizer_fn(batch_data: Dict[str, Any]) -> Tuple[FSMolMATBatch, np.ndarray]:
    adjacency_matrix, node_features, distance_matrix, labels = mol_collate_func(
        construct_dataset(
            batch_data["mat_features"], [[label] for label in batch_data["bool_labels"]]
        )
    )

    batch = FSMolMATBatch(
        node_features=node_features,
        adjacency_matrix=adjacency_matrix,
        distance_matrix=distance_matrix,
    )

    return batch, labels.squeeze(dim=-1).cpu().detach().numpy()


def mat_task_reader_fn(paths: List[RichPath], idx: int) -> List[FSMolTask]:
    [task] = default_reader_fn(paths, idx)
    return [FSMolTask(name=task.name, samples=mat_process_samples(task.samples))]


def get_mat_batcher(max_num_graphs: int):
    return FSMolBatcher(
        max_num_graphs=max_num_graphs,
        init_callback=mat_batcher_init_fn,
        per_datapoint_callback=mat_batcher_add_sample_fn,
        finalizer_callback=mat_batcher_finalizer_fn,
    )
