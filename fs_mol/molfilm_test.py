import argparse
import logging
import os
import pdb
import sys
import traceback

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from metamol.data import DataFold
from metamol.data.molfilm import get_molfilm_inference_batcher
from metamol.models.mol_pred_model import MolPredModel
from metamol.molfilm_train import eval_model_by_finetuning_on_task
from metamol.utils.molfilm_utils import resolve_starting_model_file
from metamol.utils.test_utils import add_eval_cli_args, set_up_test_run, write_csv_summary


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test finetuning a MolFiLM GNN model on tasks.",
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
    out_dir, dataset = set_up_test_run("MolFiLM", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=MolPredModel,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        config_overrides={"num_tasks": 1},
        device=device,
    )

    for task in dataset.get_task_reading_iterable(DataFold.TEST):
        test_results = eval_model_by_finetuning_on_task(
            model_weights_file,
            model_cls=MolPredModel,
            task=task,
            batcher=get_molfilm_inference_batcher(max_num_graphs=args.batch_size),
            train_set_sample_sizes=args.train_sizes,
            test_set_size=None,
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
    try:
        main()
    except Exception:
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
