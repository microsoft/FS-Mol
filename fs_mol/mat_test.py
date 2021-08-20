from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from typing import Any, Dict, Optional

import torch
from rdkit import RDLogger
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))
sys.path.insert(0, os.path.join(str(project_root()), "third_party", "MAT", "src"))

from fs_mol.data import FSMolTaskSample
from fs_mol.data.mat import FSMolMATBatch, get_mat_batcher, mat_task_reader_fn
from fs_mol.models.interface import AbstractTorchModel
from fs_mol.multitask_train import eval_model_by_finetuning_on_task
from fs_mol.utils.metrics import BinaryEvalMetrics
from fs_mol.utils.multitask_utils import resolve_starting_model_file
from fs_mol.utils.test_utils import add_eval_cli_args, eval_model, set_up_test_run

# Assumes that MAT is in the python lib path:
from transformer import GraphTransformer, make_model


logger = logging.getLogger(__name__)


class MATModel(GraphTransformer, AbstractTorchModel[FSMolMATBatch]):
    def forward(self, batch: FSMolMATBatch) -> Any:
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

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> BinaryEvalMetrics:
        return eval_model_by_finetuning_on_task(
            model_weights_file,
            model_cls=MATModel,
            task_sample=task_sample,
            temp_out_folder=temp_out_folder,
            batcher=get_mat_batcher(args.batch_size),
            learning_rate=args.learning_rate,
            task_specific_learning_rate=args.task_specific_lr,
            metric_to_use="avg_precision",
            seed=seed,
            quiet=True,
            device=device,
        )

    eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=args.train_sizes,
        out_dir=args.save_dir,
        num_samples=args.num_runs,
        valid_size_or_ratio=0.2,
        task_reader_fn=mat_task_reader_fn,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
