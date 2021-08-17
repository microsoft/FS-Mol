from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dpu_utils.utils import RichPath
from rdkit import RDLogger
from pyprojroot import here as project_root

sys.path.insert(0, "./MAT/src")
sys.path.insert(0, str(project_root()))

from metamol.data import (
    DataFold,
    MetamolBatcher,
    MetamolTask,
    MoleculeDatapoint,
    default_reader_fn,
)
from metamol.models.interface import AbstractTorchModel
from metamol.molfilm_train import eval_model_by_finetuning_on_task
from metamol.utils.molfilm_utils import resolve_starting_model_file
from metamol.utils.test_utils import add_eval_cli_args, set_up_test_run, write_csv_summary

from featurization.data_utils import construct_dataset, load_data_from_smiles, mol_collate_func
from transformer import GraphTransformer, make_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetamolMATBatch:
    node_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    distance_matrix: torch.Tensor


class MATModel(GraphTransformer, AbstractTorchModel[MetamolMATBatch]):
    def forward(self, batch: MetamolMATBatch) -> Any:
        mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0

        return super().forward(
            batch.node_features, mask, batch.adjacency_matrix, batch.distance_matrix, None
        )

    def get_model_state(self) -> Dict[str, Any]:
        return {"model_state_dict": self.state_dict()}

    def is_param_task_specific(self, param_name: str) -> bool:
        return param_name.startswith("generator")

    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        pretrained_state_dict = torch.load(path, map_location=device)

        # Checkpoints saved by us are richer than the original pre-trained checkpoint, as they also
        # contain optimizer state. For now we only want the weights, so throw out the rest.
        if "model_state_dict" in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict["model_state_dict"]

        for name, param in pretrained_state_dict.items():
            if not load_task_specific_weights and self.is_param_task_specific(name):
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> MATModel:
        # Parameters used for pretraining the original MAT model.
        model_params = {
            "d_atom": 28,
            "d_model": 1024,
            "N": 8,
            "h": 16,
            "N_dense": 1,
            "lambda_attention": 0.33,
            "lambda_distance": 0.33,
            "leaky_relu_slope": 0.1,
            "dense_output_nonlinearity": "relu",
            "distance_matrix_kernel": "exp",
            "dropout": 0.0,
            "aggregation_type": "mean",
        }

        model = make_model(**model_params)
        model.to(device)

        # Cast to a subclass, which is valid because `MATModel` only adds a bunch of methods.
        model.__class__ = MATModel

        return model


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


def mat_batcher_finalizer_fn(batch_data: Dict[str, Any]) -> Tuple[MetamolMATBatch, np.ndarray]:
    adjacency_matrix, node_features, distance_matrix, labels = mol_collate_func(
        construct_dataset(
            batch_data["mat_features"], [[label] for label in batch_data["bool_labels"]]
        )
    )

    batch = MetamolMATBatch(
        node_features=node_features,
        adjacency_matrix=adjacency_matrix,
        distance_matrix=distance_matrix,
    )

    return batch, labels.squeeze(dim=-1).cpu().detach().numpy()


def turn_off_warnings():
    # Ignore rdkit warnings.
    RDLogger.DisableLog("rdApp.*")

    # Ignore one specific pytorch warning about tensor copy.
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test finetuning a Molecule Attention Transformer model on a new task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture and initialisation).",
    )

    add_eval_cli_args(parser)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--use-fresh-param-init",
        action="store_true",
        help="Do not use trained weights, but start from a fresh, random initialisation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.000005,
        help="Learning rate for shared model components.",
    )
    parser.add_argument(
        "--task-specific-lr",
        type=float,
        default=0.00001,
        help="Learning rate for shared model components.",
    )

    return parser.parse_args()


def main():
    turn_off_warnings()

    args = parse_command_line()
    out_dir, dataset = set_up_test_run("MAT", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=MATModel,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        device=device,
    )

    def task_reader_fn(paths: List[RichPath], idx: int) -> List[MetamolTask]:
        [task] = default_reader_fn(paths, idx)
        return [MetamolTask(name=task.name, samples=mat_process_samples(task.samples))]

    for task in dataset.get_task_reading_iterable(DataFold.TEST, task_reader_fn=task_reader_fn):
        batcher = MetamolBatcher(
            max_num_graphs=args.batch_size,
            init_callback=mat_batcher_init_fn,
            per_datapoint_callback=mat_batcher_add_sample_fn,
            finalizer_callback=mat_batcher_finalizer_fn,
        )

        test_results = eval_model_by_finetuning_on_task(
            model_weights_file,
            model_cls=MATModel,
            task=task,
            batcher=batcher,
            train_set_sample_sizes=args.train_sizes,
            test_set_size=args.test_size,
            num_samples=args.num_runs,
            learning_rate=args.learning_rate,
            task_specific_learning_rate=args.task_specific_lr,
            metric_to_use="avg_precision",
            seed=args.seed,
            quiet=True,
            device=device,
        )

        write_csv_summary(
            os.path.join(args.save_dir, f"{task.name}_eval_results.csv"), test_results
        )


if __name__ == "__main__":
    main()
