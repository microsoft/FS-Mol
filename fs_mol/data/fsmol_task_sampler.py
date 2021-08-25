import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from fs_mol.data.fsmol_task import FSMolTask, FSMolTaskSample


logger = logging.getLogger(__name__)


class SamplingException(Exception):
    def __init__(
        self,
        task_name: str,
        num_samples: int,
        num_train: Optional[int],
        num_valid: Optional[int],
        num_test: Optional[int],
    ):
        super().__init__()
        self._task_name = task_name
        self._num_samples = num_samples
        self._num_train = num_train
        self._num_valid = num_valid
        self._num_test = num_test


class DatasetTooSmallException(SamplingException):
    def __init__(
        self, task_name: str, num_samples: int, num_train: int, num_valid: int, num_test: int
    ):
        super().__init__(task_name, num_samples, num_train, num_valid, num_test)

    def __str__(self):
        return (
            f"Cannot satisfy request to split dataset as specified because the dataset is too small.\n"
            f"  Task name: {self._task_name}\n"
            f"  Number of samples: {self._num_samples}\n"
            f"  Requested sample: {self._num_train} train, {self._num_valid} valid, {self._num_test} test samples."
        )


class DatasetClassTooSmallException(SamplingException):
    def __init__(
        self,
        task_name: str,
        num_samples: int,
        num_train: int,
        num_valid: int,
        num_test: int,
        label_class: bool,
        num_class_samples: int,
    ):
        super().__init__(task_name, num_samples, num_train, num_valid, num_test)
        self._label_class = label_class
        self._num_class_samples = num_class_samples

    def __str__(self):
        return (
            f"Cannot satisfy request to split dataset as specified because the number of {self._label_class}-labelled samples is too small.\n"
            f"  Task name: {self._task_name}\n"
            f"  Number of {self._label_class} samples: {self._num_class_samples}\n"
            f"  Requested sample: {self._num_train} train, {self._num_valid} valid, {self._num_test} test samples."
        )


class FoldTooSmallException(SamplingException):
    def __init__(
        self,
        task_name: str,
        num_samples: int,
        fold_name: str,
        num_train: Optional[int] = None,
        num_valid: Optional[int] = None,
        num_test: Optional[int] = None,
    ):
        super().__init__(task_name, num_samples, num_train, num_valid, num_test)
        self._fold_name = fold_name

    def __str__(self):
        return (
            f"Cannot satisfy request to split dataset as the {self._fold_name} fold would be too small (wouldn't contain both true and false labels).\n"
            f"  Task name: {self._task_name}\n"
            f"  Number of samples: {self._num_samples}\n"
            f"  Allocated sample sizes: {self._num_train} train, {self._num_valid} valid, {self._num_test} test samples."
        )


class TaskSampler(ABC):
    """Abstract superclass providing common interface for different implementations
    of task samplers. Configuration of the task samplers happens through their
    respective constructors.
    """

    @abstractmethod
    def sample(self, task: FSMolTask, seed: int = 0) -> FSMolTaskSample:
        """Split the given task into train/valid/test, with the contract that results
        remain the same if you provide the same seed.
        """
        pass


def _compute_per_fold_nums(
    num_samples: int,
    train_size_or_ratio: Union[int, float],
    valid_size_or_ratio: Union[int, float],
    test_size_or_ratio: Optional[Union[int, float, Tuple[int, int]]] = 256,
) -> Tuple[int, int, int]:
    # We first try to take as much for the train fold as requested:
    if isinstance(train_size_or_ratio, float):
        num_train = int(num_samples * train_size_or_ratio)
    else:
        num_train = min(num_samples, train_size_or_ratio)

    # If required we may need to split something off of train for valid:
    if isinstance(valid_size_or_ratio, int):
        num_valid = valid_size_or_ratio
    else:
        if valid_size_or_ratio > 0:
            num_valid = int(num_train * valid_size_or_ratio)
            num_train -= num_valid
        else:
            num_valid = 0
    num_remaining = num_samples - num_train - num_valid

    if test_size_or_ratio is None:
        num_test = num_remaining
    elif isinstance(test_size_or_ratio, int):
        num_test = test_size_or_ratio
    elif isinstance(test_size_or_ratio, tuple):
        min_num, target_num = test_size_or_ratio
        num_test = max(min_num, min(target_num, num_remaining))
    else:
        num_test = int(num_samples * test_size_or_ratio)

    return num_train, num_valid, num_test


class RandomTaskSampler(TaskSampler):
    def __init__(
        self,
        train_size_or_ratio: Union[int, float] = 128,
        valid_size_or_ratio: Union[int, float] = 0,
        test_size_or_ratio: Optional[Union[int, float, Tuple[int, int]]] = 256,
        allow_smaller_test: bool = True,
    ):
        """Create a random sampler object. When applied to a dataset, it will draw random samples
        of the specified sizes from the full set of datapoints, which means that the results may
        have a substantially skewed distribution.

        Sampling can only fail if a fixed size is requested for any of the folds that cannot be
        satisfied.

        Args:
            train_size_or_ratio: If a float, this is interpreted as a fraction of the overall
                dataset size to be used as training data.
                If an integer, it is interpreted as the total number of samples.
            valid_size_or_ratio: If a float, this is interpreted as a fraction of the TRAINING
                fold (specified above) to be used for validation.
                If an integer, it is interpreted as the total number of samples.
                In both cases, these examples are taken from the training data, not from the overall
                dataset.
            test_size_or_ratio: If a float, this is interpreted as a fraction of the overall
                dataset size to be used as test data.
                If an integer, it is interpreted as the total number of samples.
                If a tuple of integers, it is interpreted as (min_num, target_num) of
                samples; we then provide at least min_num samples, at most target_num samples,
                and only throw an exception if we there are not even min_num samples available.
        """
        self._train_size_or_ratio = train_size_or_ratio
        self._valid_size_or_ratio = valid_size_or_ratio
        self._test_size_or_ratio = test_size_or_ratio
        self._allow_smaller_test = allow_smaller_test

    def sample(self, task: FSMolTask, seed: int = 0) -> FSMolTaskSample:
        rng = np.random.Generator(np.random.PCG64(seed=seed))
        # Make a copy of the samples that we can permute:
        samples = list(task.samples)
        num_samples = len(samples)
        rng.shuffle(samples)

        num_train, num_valid, num_test = _compute_per_fold_nums(
            num_samples=num_samples,
            train_size_or_ratio=self._train_size_or_ratio,
            valid_size_or_ratio=self._valid_size_or_ratio,
            test_size_or_ratio=self._test_size_or_ratio,
        )

        num_remaining = num_samples - num_train - num_valid
        if num_test > num_remaining and self._allow_smaller_test:
            num_test = num_remaining

        if num_train + num_valid + num_test > num_samples:
            raise DatasetTooSmallException(
                task.name,
                num_samples=num_samples,
                num_train=num_train,
                num_valid=num_valid,
                num_test=num_test,
            )

        return FSMolTaskSample(
            name=task.name,
            train_samples=samples[:num_train],
            valid_samples=samples[num_train : num_train + num_valid],
            test_samples=samples[-num_test:],
        )


class BalancedTaskSampler(TaskSampler):
    def __init__(
        self,
        train_size_or_ratio: Union[int, float] = 128,
        valid_size_or_ratio: Union[int, float] = 0.0,
        test_size_or_ratio: Optional[Union[int, float, Tuple[int, int]]] = 256,
        allow_smaller_test: bool = True,
    ):
        """Create a balanced sampler object. When applied to a dataset, it will draw
        samples of the specified sizes from the full set of datapoints for each fold,
        each with a balanced number of false/true labels.

        Sampling can fail for two reasons:
         * A fixed size is requested for any of the folds that cannot be satisfied.
         * The resulting folds do not contain representative of both false and true labels.

        Args:
            train_size_or_ratio: If a float, this is interpreted as a fraction of the overall
                dataset size to be used as training data.
                If an integer, it is interpreted as the total number of samples.
            valid_size_or_ratio: If a float, this is interpreted as a fraction of the TRAINING
                fold (specified above) to be used for validation.
                If an integer, it is interpreted as the total number of samples.
                In both cases, these examples are taken from the training data, not from the overall
                dataset.
            test_size_or_ratio: If a float, this is interpreted as a fraction of the overall
                dataset size to be used as test data.
                If an integer, it is interpreted as the total number of samples.
                If a tuple of integers, it is interpreted as (min_num, target_num) of
                samples; we then provide at least min_num samples, at most target_num samples,
                and only throw an exception if we there are not even min_num samples available.
        """
        self._train_size_or_ratio = train_size_or_ratio
        self._valid_size_or_ratio = valid_size_or_ratio
        self._test_size_or_ratio = test_size_or_ratio
        self._allow_smaller_test = allow_smaller_test

    def sample(self, task: FSMolTask, seed: int = 0) -> FSMolTaskSample:
        rng = np.random.Generator(np.random.PCG64(seed=seed))
        pos_samples, neg_samples = task.get_pos_neg_separated()

        rng.shuffle(pos_samples)
        rng.shuffle(neg_samples)

        num_train, num_valid, num_test = _compute_per_fold_nums(
            num_samples=len(task.samples),
            train_size_or_ratio=self._train_size_or_ratio,
            valid_size_or_ratio=self._valid_size_or_ratio,
            test_size_or_ratio=self._test_size_or_ratio,
        )

        num_remaining = len(task.samples) - num_train - num_valid
        if num_test > num_remaining and self._allow_smaller_test:
            num_test = num_remaining

        if len(pos_samples) < num_train // 2 + num_valid // 2 + num_test // 2:
            raise DatasetClassTooSmallException(
                task.name,
                num_samples=len(task.samples),
                num_train=num_train,
                num_valid=num_valid,
                num_test=num_test,
                label_class=True,
                num_class_samples=len(pos_samples),
            )
        if len(neg_samples) < num_train // 2 + num_valid // 2 + num_test // 2:
            raise DatasetClassTooSmallException(
                task.name,
                num_samples=len(task.samples),
                num_train=num_train,
                num_valid=num_valid,
                num_test=num_test,
                label_class=False,
                num_class_samples=len(neg_samples),
            )

        return FSMolTaskSample(
            train_samples=pos_samples[: num_train // 2] + neg_samples[: num_train // 2],
            valid_samples=(
                pos_samples[num_train // 2 : num_train // 2 + num_valid // 2]
                + neg_samples[num_train // 2 : num_train // 2 + num_valid // 2]
            ),
            test_samples=pos_samples[-num_test // 2 :] + neg_samples[-num_test // 2 :],
        )


class StratifiedTaskSampler(TaskSampler):
    def __init__(
        self,
        train_size_or_ratio: Union[int, float] = 128,
        valid_size_or_ratio: Union[int, float] = 0.0,
        test_size_or_ratio: Optional[Union[int, float, Tuple[int, int]]] = 256,
        allow_smaller_test: bool = True,
    ):
        """Create a stratified sampler object. When applied to a dataset, it will draw stratified
        samples of the specified sizes from the full set of datapoints for each fold.

        Sampling can fail for two reasons:
         * A fixed size is requested for any of the folds that cannot be satisfied.
         * The resulting folds do not contain representative of both false and true labels.

        Args:
            train_size_or_ratio: If a float, this is interpreted as a fraction of the overall
                dataset size to be used as training data.
                If an integer, it is interpreted as the total number of samples.
            valid_size_or_ratio: If a float, this is interpreted as a fraction of the TRAINING
                fold (specified above) to be used for validation.
                If an integer, it is interpreted as the total number of samples.
                In both cases, these examples are taken from the training data, not from the overall
                dataset.
            test_size_or_ratio: If a float, this is interpreted as a fraction of the overall
                dataset size to be used as test data.
                If an integer, it is interpreted as the total number of samples.
                If a tuple of integers, it is interpreted as (min_num, target_num) of
                samples; we then provide at least min_num samples, at most target_num samples,
                and only throw an exception if we there are not even min_num samples available.
        """
        self._train_size_or_ratio = train_size_or_ratio
        self._valid_size_or_ratio = valid_size_or_ratio
        self._test_size_or_ratio = test_size_or_ratio
        self._allow_smaller_test = allow_smaller_test

    def sample(self, task: FSMolTask, seed: int = 0) -> FSMolTaskSample:
        # Just defer to the sklearn splitter:
        pos_samples, neg_samples = task.get_pos_neg_separated()
        num_neg_samples = len(neg_samples)
        num_pos_samples = len(pos_samples)
        num_samples = num_neg_samples + num_pos_samples
        samples = neg_samples + pos_samples
        labels = np.concatenate((np.zeros(num_neg_samples), np.ones(num_pos_samples)), axis=0)
        indices = np.arange(num_samples)

        if isinstance(self._train_size_or_ratio, int):
            possible_test_size = num_samples - self._train_size_or_ratio
        else:
            possible_test_size = num_samples - int(num_samples * self._train_size_or_ratio)

        if self._test_size_or_ratio is None:
            num_test = possible_test_size
        else:
            if isinstance(self._test_size_or_ratio, int):
                num_test = self._test_size_or_ratio
            elif isinstance(self._test_size_or_ratio, tuple):
                min_num, target_num = self._test_size_or_ratio
                num_test = max(min_num, min(target_num, possible_test_size))
            else:
                num_test = int(self._test_size_or_ratio * num_samples)
            if self._allow_smaller_test:
                num_test = min(num_test, possible_test_size)

        if num_test < 2:
            raise DatasetTooSmallException(
                task.name,
                num_samples=num_samples,
                num_train=self._train_size_or_ratio,
                num_valid=0,
                num_test=num_test,
            )
        train_test_splitter_obj = StratifiedShuffleSplit(
            n_splits=1, train_size=self._train_size_or_ratio, test_size=num_test, random_state=seed
        )
        train_valid_idxs, test_idxs = next(iter(train_test_splitter_obj.split(X=indices, y=labels)))

        train_valid_samples = [samples[i] for i in train_valid_idxs]
        test_samples = [samples[i] for i in test_idxs]

        if len(test_samples) < 2:
            raise FoldTooSmallException(
                task_name=task.name,
                num_samples=num_samples,
                fold_name="test",
                num_train=len(train_valid_samples),
            )

        if self._valid_size_or_ratio > 0:
            train_valid_splitter_obj = StratifiedShuffleSplit(
                n_splits=1, test_size=self._valid_size_or_ratio, random_state=seed
            )
            train_idxs, valid_idxs = next(
                iter(
                    train_valid_splitter_obj.split(
                        X=np.arange(len(train_valid_samples)),
                        y=[s.bool_label for s in train_valid_samples],
                    )
                )
            )
            train_samples = [train_valid_samples[i] for i in train_idxs]
            valid_samples = [train_valid_samples[i] for i in valid_idxs]

            num_pos_valid_samples = sum(s.bool_label for s in valid_samples)
            if not (0 < num_pos_valid_samples < len(valid_samples)):
                raise FoldTooSmallException(
                    task_name=task.name,
                    num_samples=num_samples,
                    fold_name="valid",
                    num_train=len(train_samples),
                    num_test=len(test_samples),
                )
        else:
            train_samples = train_valid_samples
            valid_samples = []

        num_pos_train_samples = sum(s.bool_label for s in train_samples)
        if not (0 < num_pos_train_samples < len(train_samples)):
            raise FoldTooSmallException(
                task_name=task.name,
                num_samples=num_samples,
                fold_name="train",
                num_train=len(train_samples),
                num_test=len(test_samples),
            )

        num_pos_test_samples = sum(s.bool_label for s in test_samples)
        if not (0 < num_pos_test_samples < len(test_samples)):
            raise FoldTooSmallException(
                task_name=task.name,
                num_samples=num_samples,
                fold_name="test",
                num_train=len(train_samples),
                num_test=len(test_samples),
            )

        return FSMolTaskSample(
            name=task.name,
            train_samples=train_samples,
            valid_samples=valid_samples,
            test_samples=test_samples,
        )
