import argparse
import logging
import sys

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run
from fs_mol.utils.protonet_utils import (
    PrototypicalNetworkTrainerConfig,
    PrototypicalNetworkTrainer,
)


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Train a Prototypical Network model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_cli_args(parser)

    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "gnn",
            "ecfp",
            "pc-descs",
            "ecfp+fc",
            "pc-descs+fc",
            "gnn+ecfp+fc",
            "gnn+ecfp+pc-descs+fc",
        ],
        default="gnn+ecfp+fc",
        help="Choice of features to use",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["mahalanobis", "euclidean"],
        default="mahalanobis",
        help="Choice of distance to use.",
    )
    parser.add_argument("--support_set_size", type=int, default=16, help="Size of support set")
    parser.add_argument(
        "--query_set_size",
        type=int,
        default=256,
        help="Size of target set. If -1, use everything but train examples.",
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=16,
        help="Number of tasks to accumulate gradients for.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Number of examples per batch.")
    parser.add_argument(
        "--num_train_steps", type=int, default=10000, help="Number of training steps."
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=50,
        help="Number of training steps between model validations.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--clip-value", type=float, default=1.0, help="Gradient norm clipping value"
    )
    args = parser.parse_args()
    return args


def make_trainer_config(args: argparse.Namespace) -> PrototypicalNetworkTrainerConfig:
    return PrototypicalNetworkTrainerConfig(
        used_features=args.features,
        distance_metric=args.distance_metric,
        batch_size=args.batch_size,
        tasks_per_batch=args.tasks_per_batch,
        support_set_size=args.support_set_size,
        query_set_size=args.query_set_size,
        validate_every_num_steps=args.validate_every,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
    )


def main():
    args = parse_command_line()
    config = make_trainer_config(args)

    out_dir, dataset, aml_run = set_up_train_run(
        f"ProtoNet_{config.used_features}", args, torch=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = PrototypicalNetworkTrainer(config=config).to(device)

    logger.info(f"\tDevice: {device}")
    logger.info(f"\tNum parameters {sum(p.numel() for p in model_trainer.parameters())}")
    logger.info(f"\tModel:\n{model_trainer}")

    model_trainer.train_loop(out_dir, dataset, aml_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
