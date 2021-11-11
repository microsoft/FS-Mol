import argparse
import logging
import os
import sys
import random
import time
from typing import Any, Optional, Tuple, Union

import numpy as np
from dpu_utils.utils import RichPath

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_dataset import FSMolDataset
from fs_mol.utils.logging import set_up_logging


logger = logging.getLogger(__name__)


def add_train_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "DATA_PATH",
        type=str,
        help="Directory containing the task data in train/valid/test subdirectories.",
    )

    parser.add_argument(
        "--task-list-file",
        default="datasets/fsmol-0.1.json",
        type=str,
        help=("JSON file containing the lists of tasks to be used in training/test/valid splits."),
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs",
        help="Path in which to store the test results and log of their computation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use.",
    )

    parser.add_argument(
        "--azureml_logging", action="store_true", help="Log results using AML run context."
    )


def set_up_train_run(
    model_name: str, args: argparse.Namespace, torch: bool = False, tf: bool = False
) -> Tuple[str, FSMolDataset, Optional[Any]]:
    logger.info(f"Setting random seed {args.seed}.")
    set_seed(args.seed, torch=torch, tf=tf)

    run_name = f"FSMol_{model_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    set_up_logging(os.path.join(out_dir, f"train.log"))

    logger.info(f"Starting train run {run_name}.")
    logger.info(f"\tArguments: {args}")
    logger.info(f"\tOutput dir: {out_dir}")
    logger.info(f"\tData path: {args.DATA_PATH}")

    fsmol_dataset = FSMolDataset.from_directory(
        directory=RichPath.create(args.DATA_PATH),
        task_list_file=RichPath.create(args.task_list_file),
    )

    if args.azureml_logging:
        from azureml.core.run import Run

        aml_run = Run.get_context()
    else:
        aml_run = None

    return out_dir, fsmol_dataset, aml_run


def str2bool(v: Union[str, bool]) -> bool:
    import argparse

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int, torch: bool = False, tf: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if tf:
        import tensorflow

        tensorflow.random.set_seed(seed)

    if torch:
        import torch as t

        t.manual_seed(seed)
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = True
