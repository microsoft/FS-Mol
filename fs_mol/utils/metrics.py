import dataclasses
from typing import Dict, Tuple, List
from typing_extensions import Literal
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)


@dataclass(frozen=True)
class BinaryEvalMetrics:
    size: int
    acc: float
    balanced_acc: float
    f1: float
    prec: float
    recall: float
    roc_auc: float
    avg_precision: float
    kappa: float


BinaryMetricType = Literal[
    "acc", "balanced_acc", "f1", "prec", "recall", "roc_auc", "avg_precision", "kappa"
]


def compute_binary_task_metrics(predictions: List[float], labels: List[float]) -> BinaryEvalMetrics:
    normalized_predictions = [
        pred >= 0.5 for pred in predictions
    ]  # Normalise probabilities to bool values

    if np.sum(labels) == len(labels) or np.sum(labels) == 0:
        roc_auc = 0.0
    else:
        roc_auc = roc_auc_score(labels, predictions)

    return BinaryEvalMetrics(
        size=len(predictions),
        acc=accuracy_score(labels, normalized_predictions),
        balanced_acc=balanced_accuracy_score(labels, normalized_predictions),
        f1=f1_score(labels, normalized_predictions, zero_division=1),
        prec=precision_score(labels, normalized_predictions, zero_division=1),
        recall=recall_score(labels, normalized_predictions, zero_division=1),
        roc_auc=roc_auc,
        avg_precision=average_precision_score(labels, predictions),
        kappa=cohen_kappa_score(labels, normalized_predictions),
    )


def avg_metrics_over_tasks(
    task_results: Dict[str, List[BinaryEvalMetrics]],
) -> Dict[str, Tuple[float, float]]:
    # average results over all tasks in input dictionary
    # the average over each task is first created
    # technically input is Dict[str, FSMolTaskSampleEvalResults], but everything
    # not in BinaryEvalMetrics is unused here.
    aggregated_metrics = {}
    for (task, results) in task_results.items():
        # this returns, for each task, a dictionary of aggregated results
        aggregated_metrics[task] = avg_task_metrics_list(results)

    # compute the mean and std across tasks by going through values (drop within task stds)
    aggregated_over_tasks = {}
    for metric_field in dataclasses.fields(BinaryEvalMetrics):
        metric_values = [x.get(metric_field.name)[0] for _, x in aggregated_metrics.items()]
        aggregated_over_tasks[metric_field.name] = (np.mean(metric_values), np.std(metric_values))

    return aggregated_over_tasks


def avg_task_metrics_list(
    results: List[BinaryEvalMetrics],
) -> Dict[str, Tuple[float, float]]:
    aggregated_metrics = {}

    # Compute mean/std:
    for metric_field in dataclasses.fields(BinaryEvalMetrics):
        metric_values = [getattr(task_metrics, metric_field.name) for task_metrics in results]
        aggregated_metrics[metric_field.name] = (np.mean(metric_values), np.std(metric_values))

    return aggregated_metrics


def compute_metrics(
    task_to_predictions: Dict[int, List[float]],
    task_to_labels: Dict[int, List[float]],
) -> Dict[int, BinaryEvalMetrics]:
    per_task_results: Dict[int, BinaryEvalMetrics] = {}
    for task_id in task_to_predictions.keys():
        per_task_results[task_id] = compute_binary_task_metrics(
            task_to_predictions[task_id], labels=task_to_labels[task_id]
        )

    return per_task_results
