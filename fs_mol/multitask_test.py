import argparse
import logging
import pdb
import sys
import traceback

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.data.multitask import get_multitask_inference_batcher
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.gnn_multitask import GNNMultitaskModel
from fs_mol.multitask_train import eval_model_by_finetuning_on_task
from fs_mol.utils.metrics import BinaryEvalMetrics
from fs_mol.utils.test_utils import add_eval_cli_args, eval_model, set_up_test_run


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test finetuning a GNN Multitask model on tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture).",
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of molecules per batch.",
    )
    parser.add_argument(
        "--use-fresh-param-init",
        action="store_true",
        help="Do not use trained weights, but start from a fresh, random initialisation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
        help="Learning rate for shared model components.",
    )
    parser.add_argument(
        "--task-specific-lr",
        type=float,
        default=0.0001,
        help="Learning rate for shared model components.",
    )

    return parser.parse_args()


def main():
    args = parse_command_line()
    out_dir, dataset = set_up_test_run("Multitask", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=GNNMultitaskModel,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        config_overrides={"num_tasks": 1},
        device=device,
    )

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> BinaryEvalMetrics:
        return eval_model_by_finetuning_on_task(
            model_weights_file,
            model_cls=GNNMultitaskModel,
            task_sample=task_sample,
            batcher=get_multitask_inference_batcher(max_num_graphs=args.batch_size, device=device),
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
        out_dir=out_dir,
        num_samples=args.num_runs,
        valid_size_or_ratio=0.2,
        seed=args.seed,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
