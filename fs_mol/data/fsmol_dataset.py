import logging
import os
from enum import Enum
from typing import List, Dict, Iterable, Optional, Union, Callable, TypeVar

from dpu_utils.utils import RichPath

from fs_mol.data.file_reader_iterable import (
    BufferedFileReaderIterable,
    SequentialFileReaderIterable,
)
from fs_mol.data.fsmol_task import FSMolTask, get_task_name_from_path


logger = logging.getLogger(__name__)


TaskReaderResultType = TypeVar("TaskReaderResultType")


NUM_EDGE_TYPES = 3  # Single, Double, Triple
NUM_NODE_FEATURES = 32  # comes from data preprocessing


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def default_reader_fn(paths: List[RichPath], idx: int) -> List[FSMolTask]:
    if len(paths) > 1:
        raise ValueError()

    return [FSMolTask.load_from_file(paths[0])]


class FSMolDataset:
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
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        **kwargs,
    ) -> "FSMolDataset":
        """Create a new FSMolDataset object from a directory containing the pre-processed
        files (*.jsonl.gz) split in to train/valid/test subdirectories.

        Args:
            directory: Path containing .jsonl.gz files representing the pre-processed tasks.
            task_list_file: (Optional) path of the .json file that stores which assays are to be
            used in each fold. Used for subset selection.
            **kwargs: remaining arguments are forwarded to the FSMolDataset constructor.
        """
        if isinstance(directory, str):
            data_rp = RichPath.create(directory)
        else:
            data_rp = directory

        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            else:
                task_list_file = task_list_file
            task_list = task_list_file.read_by_file_suffix()
        else:
            task_list = None

        def get_fold_file_names(data_fold_name: str):
            fold_dir = data_rp.join(data_fold_name)
            if task_list is None:
                return fold_dir.get_filtered_files_in_dir("*.jsonl.gz")
            else:
                return [
                    file_name
                    for file_name in fold_dir.get_filtered_files_in_dir("*.jsonl.gz")
                    if any(
                        file_name.basename() == f"{task_name}.jsonl.gz"
                        for task_name in task_list[data_fold_name]
                    )
                ]

        return FSMolDataset(
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
