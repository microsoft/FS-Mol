from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple, Dict, Any
import dataclasses
import logging

import numpy as np
from dpu_utils.utils.richpath import RichPath

from fs_mol.data import (
    DataFold,
    FSMolDataset,
    FSMolTask,
    FSMolTaskSample,
    FSMolBatch,
    StratifiedTaskSampler,
    FSMolBatcher,
    MoleculeDatapoint,
    fsmol_batch_finalizer,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoleculeProtoNetFeatures(FSMolBatch):
    fingerprints: np.ndarray  # [num_samples, FP_DIM]
    descriptors: np.ndarray  # [num_samples, DESC_DIM]


@dataclass(frozen=True)
class ProtoNetBatch:
    support_features: MoleculeProtoNetFeatures
    support_labels: np.ndarray
    query_features: MoleculeProtoNetFeatures

    @property
    def num_support_samples(self) -> int:
        return self.support_features.num_graphs

    @property
    def num_query_samples(self) -> int:
        return self.query_features.num_graphs


@dataclass(frozen=True)
class FeaturisedPNTaskSample:
    task_name: str
    num_support_samples: int
    num_positive_support_samples: int
    num_query_samples: int
    num_positive_query_samples: int
    batches: List[ProtoNetBatch]
    batch_labels: List[np.ndarray]


def batcher_init_fn(batch_data: Dict[str, Any]):
    batch_data["fingerprints"] = []
    batch_data["descriptors"] = []


def batcher_add_sample_fn(batch_data: Dict[str, Any], sample_id: int, sample: MoleculeDatapoint):
    batch_data["fingerprints"].append(sample.get_fingerprint())
    batch_data["descriptors"].append(sample.get_descriptors())


def batcher_finalizer_fn(batch_data: Dict[str, Any]) -> Tuple[MoleculeProtoNetFeatures, np.ndarray]:
    plain_batch = fsmol_batch_finalizer(batch_data)
    return (
        MoleculeProtoNetFeatures(
            fingerprints=np.stack(batch_data["fingerprints"], axis=0),
            descriptors=np.stack(batch_data["descriptors"], axis=0),
            **dataclasses.asdict(plain_batch),
        ),
        np.stack(batch_data["bool_labels"], axis=0),
    )


def task_sample_to_pn_task_sample(
    task_sample: FSMolTaskSample, batcher: FSMolBatcher[MoleculeProtoNetFeatures, np.ndarray]
) -> FeaturisedPNTaskSample:
    support_batches = list(batcher.batch(task_sample.train_samples))
    if len(support_batches) > 1:
        raise ValueError("Support set too large to fit into a single batch!")
    support_features, support_labels = support_batches[0]

    # We need to do some hackery here to establish a stable batch size, as each
    # batch is the sum of support and query batches. To this end, we reset the
    # batch size now, and will restore that in the finally block:
    try:
        orig_max_num_graphs = batcher._max_num_graphs
        max_num_query_graphs = orig_max_num_graphs - support_features.num_graphs
        batcher._max_num_graphs = max_num_query_graphs
        sample_batches = []
        batch_labels: List[np.ndarray] = []
        for query_features, query_labels in batcher.batch(task_sample.test_samples):
            sample_batches.append(
                ProtoNetBatch(
                    support_features=support_features,
                    support_labels=support_labels,
                    query_features=query_features,
                )
            )
            batch_labels.append(query_labels)
    finally:
        batcher._max_num_graphs = orig_max_num_graphs

    return FeaturisedPNTaskSample(
        task_name=task_sample.name,
        num_support_samples=len(task_sample.train_samples),
        num_positive_support_samples=sum(s.bool_label for s in task_sample.train_samples),
        num_query_samples=len(task_sample.test_samples),
        num_positive_query_samples=sum(s.bool_label for s in task_sample.test_samples),
        batches=sample_batches,
        batch_labels=batch_labels,
    )


def get_protonet_batcher(
    max_num_graphs: Optional[int] = None,
    max_num_nodes: Optional[int] = None,
    max_num_edges: Optional[int] = None,
) -> FSMolBatcher[MoleculeProtoNetFeatures, np.ndarray]:
    return FSMolBatcher(
        max_num_graphs,
        max_num_nodes,
        max_num_edges,
        init_callback=batcher_init_fn,
        per_datapoint_callback=batcher_add_sample_fn,
        finalizer_callback=batcher_finalizer_fn,
    )


def get_protonet_task_sample_iterable(
    dataset: FSMolDataset,
    data_fold: DataFold,
    num_samples: int,
    support_size: int,
    query_size: Optional[int],
    max_num_graphs: Optional[int] = None,
    max_num_nodes: Optional[int] = None,
    max_num_edges: Optional[int] = None,
    repeat: bool = False,
) -> Iterable[FeaturisedPNTaskSample]:
    task_sampler = StratifiedTaskSampler(
        train_size_or_ratio=support_size, test_size_or_ratio=query_size
    )
    batcher = get_protonet_batcher(
        max_num_graphs=max_num_graphs,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
    )

    def path_to_batches_pipeline(paths: List[RichPath], idx: int):
        if len(paths) > 1:
            raise ValueError()
        task = FSMolTask.load_from_file(paths[0])
        num_task_samples = 0
        for _ in range(num_samples):
            try:
                task_sample = task_sampler.sample(task, seed=idx + num_task_samples)
                num_task_samples += 1
            except Exception as e:
                logger.debug(f"{task.name}: Sampling failed: {e}")
                continue

            yield task_sample_to_pn_task_sample(task_sample, batcher)

    return dataset.get_task_reading_iterable(
        data_fold=data_fold,
        task_reader_fn=path_to_batches_pipeline,
        repeat=repeat,
    )
