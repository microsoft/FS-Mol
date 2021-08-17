#!/usr/bin/env python3
import dataclasses
import json
import logging
import os
import sys
from typing import Dict, Optional, List, Any

import numpy as np
import sklearn.ensemble
import sklearn.neighbors
from dpu_utils.utils import run_and_debug
from pyprojroot import here as project_root
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(project_root()))

from metamol.data import (
    DataFold,
    MetamolTask,
    DatasetClassTooSmallException,
    DatasetTooSmallException,
    FoldTooSmallException,
    StratifiedTaskSampler,
)
from metamol.utils.cli_utils import str2bool
from metamol.utils.logging import prefix_log_msgs
from metamol.utils.metrics import compute_binary_task_metrics
from metamol.utils.test_utils import (
    MetamolTaskSampleEvalResults,
    write_csv_summary,
    add_eval_cli_args,
    set_up_test_run,
)

logger = logging.getLogger(__name__)

# TODO: extend to whichever models seem useful.
# hyperparam search params
DEFAULT_GRID_SEARCH: Dict[str, Dict[str, List[Any]]] = {
    "randomForest": {
        "n_estimators": [10, 100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_leaf": [1, 2, 5],
    },
    "kNN": {
        "n_neighbors": [8, 16, 32, 64, 128],
        "metric": [
            "minkowski",
            "mahalanobis",
        ],
    },
}

NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "randomForest": sklearn.ensemble.RandomForestClassifier,
    "kNN": sklearn.neighbors.KNeighborsClassifier,
}


def test(
    model_name: str,
    task: MetamolTask,
    train_set_sample_sizes: List[int],
    num_samples: int,
    test_set_size: Optional[int] = None,
    use_grid_search: bool = True,
    grid_search_parameters: Optional[Dict[str, Any]] = None,
    model_params: Dict[str, Any] = {},
    seed: int = 0,
) -> List[MetamolTaskSampleEvalResults]:
    test_results = []
    for train_size in train_set_sample_sizes:
        task_sampler = StratifiedTaskSampler(
            train_size_or_ratio=train_size,
            valid_size_or_ratio=0.0,
            test_size_or_ratio=test_set_size,
            allow_smaller_test=True,
        )
        for run_idx in range(num_samples):
            logger.info(f"=== Evaluating on {task.name}, #train {train_size}, run {run_idx}")
            with prefix_log_msgs(f" Inner - {task.name} - Size {train_size:3d} - Run {run_idx}"):
                try:
                    task_sample = task_sampler.sample(task, seed=seed + run_idx)
                except (
                    DatasetTooSmallException,
                    DatasetClassTooSmallException,
                    FoldTooSmallException,
                    ValueError,
                ) as e:
                    logger.info(
                        f" Failed to draw sample with {train_size} train points for {task.name}. Skipping."
                    )
                    logger.debug(" Sampling error: " + str(e))
                    continue

                train_data = task_sample.train_samples
                test_data = task_sample.test_samples

                # get data in to form for sklearn
                X_train = np.array([x.get_fingerprint() for x in train_data])
                X_test = np.array([x.get_fingerprint() for x in test_data])
                logger.info(f" Training with {X_train.shape[0]} datapoints.")
                y_train = [float(x.bool_label) for x in train_data]
                y_test = [float(x.bool_label) for x in test_data]

                # use the train data to train a baseline model with CV grid search
                # reinstantiate model for each seed.
                if use_grid_search:
                    if grid_search_parameters is None:
                        grid_search_parameters = DEFAULT_GRID_SEARCH[model_name]
                        grid_search = GridSearchCV(
                            NAME_TO_MODEL_CLS[model_name](),
                            grid_search_parameters,
                        )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                else:
                    model = NAME_TO_MODEL_CLS[model_name]()
                    params = model.get_params()
                    params.update(model_params)
                    model.set_params()
                    model.fit(X_train, y_train)

                # Compute test results:
                y_predicted_true_probs = model.predict_proba(X_test)[:, 1]
                test_metrics = compute_binary_task_metrics(y_predicted_true_probs, y_test)

                logger.info(f" Test metrics: {test_metrics}")
                logger.info(
                    f" Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data."
                )

                test_results.append(
                    MetamolTaskSampleEvalResults(
                        task_name=task.name,
                        seed=seed + run_idx,
                        num_train=train_size,
                        num_test=len(task_sample.test_samples),
                        fraction_pos_train=task_sample.train_pos_label_ratio,
                        fraction_pos_test=task_sample.test_pos_label_ratio,
                        **dataclasses.asdict(test_metrics),
                    )
                )

    return test_results


def run_from_args(args) -> None:
    out_dir, dataset = set_up_test_run(args.model, args)

    for task in dataset.get_task_reading_iterable(DataFold.TEST):
        test_results = test(
            model_name=args.model,
            task=task,
            train_set_sample_sizes=args.train_sizes,
            num_samples=args.num_runs,
            use_grid_search=args.grid_search,
            model_params=args.model_params,
            seed=args.seed,
        )
        write_csv_summary(os.path.join(out_dir, f"{task.name}_eval_results.csv"), test_results)


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test sklearn models on tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        default="randomForest",
        choices=["randomForest", "kNN"],
        help="The model to use.",
    )
    parser.add_argument(
        "--grid-search",
        type=str2bool,
        default=True,
        help="Perform grid search over hyperparameter space rather than use defaults/passed parameters.",
    )
    parser.add_argument(
        "--model-params",
        type=lambda s: json.loads(s),
        default={},
        help=(
            "JSON dictionary containing model hyperparameters, if not using grid search these will"
            " be used."
        ),
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    args = parser.parse_args()

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
