import logging
import os
from enum import Enum
from typing import List, Dict, Iterable, Optional, Union, Callable, TypeVar

from dpu_utils.utils import RichPath

from metamol.data.file_reader_iterable import (
    BufferedFileReaderIterable,
    SequentialFileReaderIterable,
)
from metamol.data.metamol_task import MetamolTask, get_task_name_from_path


logger = logging.getLogger(__name__)


TaskReaderResultType = TypeVar("TaskReaderResultType")


NUM_EDGE_TYPES = 3  # Single, Double, Triple
NUM_NODE_FEATURES = 32  # comes from data preprocessing


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def default_reader_fn(paths: List[RichPath], idx: int) -> List[MetamolTask]:
    if len(paths) > 1:
        raise ValueError()

    return [MetamolTask.load_from_file(paths[0])]


class MetamolDataset:
    """Dataset of related tasks, provided as individual files split into meta-train, meta-valid and
    meta-test sets."""

    def __init__(
        self,
        train_data_paths: List[RichPath] = [],
        valid_data_paths: List[RichPath] = [],
        test_data_paths: List[RichPath] = [],
        num_workers: Optional[int] = None,
    ):
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths,
            DataFold.VALIDATION: valid_data_paths,
            DataFold.TEST: test_data_paths,
        }
        self._num_workers = num_workers if num_workers is not None else os.cpu_count() or 1
        logger.info(f"Identified {len(self._fold_to_data_paths[DataFold.TRAIN])} training tasks.")
        logger.info(
            f"Identified {len(self._fold_to_data_paths[DataFold.VALIDATION])} validation tasks."
        )
        logger.info(f"Identified {len(self._fold_to_data_paths[DataFold.TEST])} test tasks.")

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        return len(self._fold_to_data_paths[fold])

    @staticmethod
    def from_task_split_file(
        data_path: Union[str, RichPath], task_split_path: Union[str, RichPath], **kwargs
    ) -> "MetamolDataset":
        """Create a new MetamolDataset object from a data path (containing the pre-processed tasks)
        and a JSON file describing how the tasks are split across train/valid/test.

        Args:
            data_path: Path containing .jsonl.gz files representing the pre-processed tasks.
            task_split_path: Path to a JSON file containing a dictionary listing the names of the
                train, validation, and test tasks.
            **kwargs: remaining arguments are forwarded to the MetamolDataset constructor.
        """
        if isinstance(data_path, str):
            data_rp = RichPath.create(data_path)
        else:
            data_rp = data_path

        if isinstance(task_split_path, str):
            task_split_file_rp = RichPath.create(task_split_path)
        else:
            task_split_file_rp = task_split_path

        all_task_file_names = data_rp.get_filtered_files_in_dir("*.jsonl.gz")
        fold_to_task_name_list = task_split_file_rp.read_by_file_suffix()

        def get_fold_file_names(data_fold_name: str):
            return [
                file_name
                for file_name in all_task_file_names
                if any(
                    file_name.basename() == f"{task_name}.jsonl.gz"
                    for task_name in fold_to_task_name_list[data_fold_name]
                )
            ]

        return MetamolDataset(
            train_data_paths=get_fold_file_names("train"),
            valid_data_paths=sorted(get_fold_file_names("valid")),
            test_data_paths=sorted(get_fold_file_names("test")),
            **kwargs,
        )

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        return [get_task_name_from_path(path) for path in self._fold_to_data_paths[data_fold]]

    def get_task_reading_iterable(
        self,
        data_fold: DataFold,
        task_reader_fn: Callable[
            [List[RichPath], int], Iterable[TaskReaderResultType]
        ] = default_reader_fn,
        repeat: bool = False,
        reader_chunk_size: int = 1,
    ) -> Iterable[TaskReaderResultType]:
        if self._num_workers == 0:
            return SequentialFileReaderIterable(
                reader_fn=task_reader_fn,
                data_paths=self._fold_to_data_paths[data_fold],
                shuffle_data=data_fold == DataFold.TRAIN,
                repeat=repeat,
                reader_chunk_size=reader_chunk_size,
            )
        else:
            return BufferedFileReaderIterable(
                reader_fn=task_reader_fn,
                data_paths=self._fold_to_data_paths[data_fold],
                shuffle_data=data_fold == DataFold.TRAIN,
                repeat=repeat,
                reader_chunk_size=reader_chunk_size,
                num_workers=self._num_workers,
            )
