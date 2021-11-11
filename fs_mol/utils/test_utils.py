import argparse
import csv
import dataclasses
import tempfile
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from dpu_utils.utils.richpath import RichPath

from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset
from fs_mol.data.fsmol_task import FSMolTask, FSMolTaskSample
from fs_mol.data.fsmol_task_sampler import (
    DatasetClassTooSmallException,
    DatasetTooSmallException,
    FoldTooSmallException,
    StratifiedTaskSampler,
)
from fs_mol.utils.cli_utils import set_seed
from fs_mol.utils.logging import prefix_log_msgs, set_up_logging
from fs_mol.utils.metrics import BinaryEvalMetrics


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FSMolTaskSampleEvalResults(BinaryEvalMetrics):
    task_name: str
    seed: int
    num_train: int
    num_test: int
    fraction_pos_train: float
    fraction_pos_test: float


def add_data_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "DATA_PATH",
        type=str,
        nargs="+",
        help=(
            "File(s) containing the test data."
            " If this is a directory, the --task-list-file argument will be used to determine what files to test on."
            " Otherwise, it is the data file(s) on which testing is done."
        ),
    )

    parser.add_argument(
        "--task-list-file",
        default="datasets/fsmol-0.1.json",
        type=str,
        help=("JSON file containing the lists of tasks to be used in training/test/valid splits."),
    )


def add_eval_cli_args(parser: argparse.ArgumentParser) -> None:
    add_data_cli_args(parser)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs",
        help="Path in which to store the test results and log of their computation.",
    )

    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of runs with different data splits to do."
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")

    parser.add_argument(
        "--train-sizes",
        type=json.loads,
        default=[16, 32, 64, 128, 256],
        help="JSON list of number of training points to sample.",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Number of test samples to take, default is take all remaining after splitting out train.",
    )


def set_up_dataset(args: argparse.Namespace, **kwargs):
    # Handle the different task entry methods.
    # Permit a directory or a list of files
    if len(args.DATA_PATH) == 1 and RichPath.create(args.DATA_PATH[0]).is_dir():
        assert (
            RichPath.create(args.DATA_PATH[0]).join("test").exists()
        ), "If DATA_PATH is a directory it must contain test/ sub-directory for evaluation."

        return FSMolDataset.from_directory(
            args.DATA_PATH[0], task_list_file=RichPath.create(args.task_list_file), **kwargs
        )
    else:
        return FSMolDataset(test_data_paths=[RichPath.create(p) for p in args.DATA_PATH], **kwargs)


def set_up_test_run(
    model_name: str, args: argparse.Namespace, torch: bool = False, tf: bool = False
) -> Tuple[str, FSMolDataset]:
    set_seed(args.seed, torch=torch, tf=tf)
    run_name = f"FSMol_Eval_{model_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    set_up_logging(os.path.join(out_dir, f"{run_name}.log"))

    dataset = set_up_dataset(args)
    logger.info(
        f"Starting test run {run_name} on {len(dataset.get_task_names(DataFold.TEST))} assays"
    )
    logger.info(f"\tArguments: {args}")
    logger.info(f"\tOutput dir: {out_dir}")

    return out_dir, dataset


def write_csv_summary(output_csv_file: str, test_results: Iterable[FSMolTaskSampleEvalResults]):
    with open(output_csv_file, "w", newline="") as csv_file:
        fieldnames = [
            "num_train_requested",
            "num_train",
            "fraction_positive_train",
            "num_test",
            "fraction_positive_test",
            "seed",
            "valid_score",
            "average_precision_score",
            "roc_auc",
            "acc",
            "balanced_acc",
            "precision",
            "recall",
            "f1_score",
            "delta_auprc",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for test_result in test_results:
            csv_writer.writerow(
                {
                    "num_train_requested": test_result.num_train,
                    "num_train": test_result.num_train,
                    "num_test": test_result.num_test,
                    "fraction_positive_train": test_result.fraction_pos_train,
                    "fraction_positive_test": test_result.fraction_pos_test,
                    "seed": test_result.seed,
                    "average_precision_score": test_result.avg_precision,
                    "roc_auc": test_result.roc_auc,
                    "acc": test_result.acc,
                    "balanced_acc": test_result.balanced_acc,
                    "precision": test_result.prec,
                    "recall": test_result.recall,
                    "f1_score": test_result.f1,
                    "delta_auprc": test_result.avg_precision - test_result.fraction_pos_test,
                }
            )


def eval_model(
    test_model_fn: Callable[[FSMolTaskSample, str, int], BinaryEvalMetrics],
    dataset: FSMolDataset,
    train_set_sample_sizes: List[int],
    out_dir: Optional[str] = None,
    num_samples: int = 10,
    valid_size_or_ratio: Union[int, float] = 0.0,
    test_size_or_ratio: Optional[Union[int, float, Tuple[int, int]]] = None,
    fold: DataFold = DataFold.TEST,
    task_reader_fn: Optional[Callable[[List[RichPath], int], Iterable[FSMolTask]]] = None,
    seed: int = 0,
) -> Dict[str, List[FSMolTaskSampleEvalResults]]:
    """Evaluate a model on the FSMolDataset passed.

    Args:
        test_model_fn: A callable directly evaluating the model of interest on a single task
            sample in the form of an FSMolTaskSample. The test_model_fn should act on the task
            sample with the model, using a temporary output folder and seed. All other required
            variables should be defined in the same context as the callable. The function should
            return a BinaryEvalMetrics object from the task.
        dataset: An FSMolDataset with paths to the data to be evaluated supplied.
        train_set_samples_sizes: List[int], a list of the support set sizes at which to evaluate,
            this is the train_samples size in a TaskSample.
        out_dir: final output directory for evaluation results.
        num_samples: number of repeated draws from the task's data on which to evaluate the model.
        valid_size_or_ratio: size of validation set in a TaskSample.
        test_size_or_ratio: size of the test set in a TaskSample.
        fold: the fold of FSMolDataset on which to perform evaluation, typically will be the test fold.
        task_reader_fn: Callable allowing additional transformations on the data prior to its batching
            and passing through a model.
        seed: an base external seed value. Repeated runs vary from this seed.
    """
    task_reading_kwargs = {"task_reader_fn": task_reader_fn} if task_reader_fn is not None else {}
    task_to_results: Dict[str, List[FSMolTaskSampleEvalResults]] = {}

    for task in dataset.get_task_reading_iterable(fold, **task_reading_kwargs):
        test_results: List[FSMolTaskSampleEvalResults] = []
        for train_size in train_set_sample_sizes:
            task_sampler = StratifiedTaskSampler(
                train_size_or_ratio=train_size,
                valid_size_or_ratio=valid_size_or_ratio,
                test_size_or_ratio=test_size_or_ratio,
                allow_smaller_test=True,
            )

            for run_idx in range(num_samples):
                logger.info(f"=== Evaluating on {task.name}, #train {train_size}, run {run_idx}")
                with prefix_log_msgs(
                    f" Test - Task {task.name} - Size {train_size:3d} - Run {run_idx}"
                ), tempfile.TemporaryDirectory() as temp_out_folder:
                    local_seed = seed + run_idx

                    try:
                        task_sample = task_sampler.sample(task, seed=local_seed)
                    except (
                        DatasetTooSmallException,
                        DatasetClassTooSmallException,
                        FoldTooSmallException,
                        ValueError,
                    ) as e:
                        logger.debug(
                            f"Failed to draw sample with {train_size} train points for {task.name}. Skipping."
                        )
                        logger.debug("Sampling error: " + str(e))
                        continue

                    test_metrics = test_model_fn(task_sample, temp_out_folder, local_seed)

                    test_results.append(
                        FSMolTaskSampleEvalResults(
                            task_name=task.name,
                            seed=local_seed,
                            num_train=train_size,
                            num_test=len(task_sample.test_samples),
                            fraction_pos_train=task_sample.train_pos_label_ratio,
                            fraction_pos_test=task_sample.test_pos_label_ratio,
                            **dataclasses.asdict(test_metrics),
                        )
                    )

        task_to_results[task.name] = test_results

        if out_dir is not None:
            write_csv_summary(os.path.join(out_dir, f"{task.name}_eval_results.csv"), test_results)

    logger.info(f"=== Completed evaluation on all tasks.")

    return task_to_results
