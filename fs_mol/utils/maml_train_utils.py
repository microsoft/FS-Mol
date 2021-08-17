import dataclasses
import logging
import itertools
from fs_mol.data.fsmol_task import MoleculeDatapoint
import os
import pickle
import tempfile
from functools import partial
from typing import Dict, Any, Iterable, List, Optional, Tuple, Callable, Union
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from tf2_gnn.cli_utils.model_utils import _get_name_to_variable_map, load_weights_verbosely
from tf2_gnn.cli_utils.dataset_utils import get_model_file_path

from fs_mol.data import (
    FSMolTask,
    DatasetClassTooSmallException,
    DatasetTooSmallException,
    FoldTooSmallException,
    StratifiedTaskSampler,
)
from fs_mol.models.split_lr_graph_binary_classification import SplitLRGraphBinaryClassificationTask
from fs_mol.utils.logging import PROGRESS_LOG_LEVEL, prefix_log_msgs, restrict_console_log_level
from fs_mol.utils.metrics import (
    BinaryEvalMetrics,
    BinaryMetricType,
    avg_metrics_list,
    compute_binary_task_metrics,
)
from fs_mol.utils.maml_data_utils import TFGraphBatchIterable
from fs_mol.utils.test_utils import FSMolTaskSampleEvalResults


logger = logging.getLogger(__name__)


MetricType = Union[BinaryMetricType, Literal["loss"]]


def save_model(
    save_file: str,
    model: SplitLRGraphBinaryClassificationTask,
    extra_data_to_store: Dict[str, Any] = {},
    quiet: bool = True,
) -> None:

    data_to_store = {
        "model_class": model.__class__,
        "model_params": model._params,
    }

    var_name_to_variable = _get_name_to_variable_map(model)
    var_name_to_weights = {name: var.value().numpy() for name, var in var_name_to_variable.items()}
    data_to_store["model_weights"] = var_name_to_weights

    data_to_store.update(extra_data_to_store)

    pkl_file = get_model_file_path(save_file, "pkl")
    with open(pkl_file, "wb") as out_file:
        pickle.dump(data_to_store, out_file, pickle.HIGHEST_PROTOCOL)

    if not quiet:
        logger.info(f" Stored model metadata and weights to {pkl_file}.")


def __metrics_from_batch_results(task_results: List[Dict[str, Any]]):
    predictions, labels = [], []
    for task_result in task_results:
        predictions.append(task_result["predictions"].numpy())
        labels.append(task_result["labels"])
    return compute_binary_task_metrics(
        predictions=np.concatenate(predictions, axis=0), labels=np.concatenate(labels, axis=0)
    )


def train_loop(
    model: SplitLRGraphBinaryClassificationTask,
    train_data: Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]],
    valid_fn: Callable[[SplitLRGraphBinaryClassificationTask], float],
    model_save_file: str,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 100,
    patience: int = 5,
    quiet: bool = False,
):
    logger.info("== Running validation on initial model")
    initial_valid_metric = valid_fn(model)
    best_valid_metric = initial_valid_metric
    logger.info(f"  Initial validation metric: {best_valid_metric:.5f}")
    save_model(model_save_file, model, quiet=quiet)

    epochs_since_best = 0
    for epoch in range(0, max_num_epochs):
        logger.info(f"== Epoch {epoch}")
        logger.info(f"  = Training")
        train_loss, _, train_results = model.run_one_epoch(train_data, training=True, quiet=True)
        train_epoch_metrics = __metrics_from_batch_results(train_results)
        if metric_to_use == "loss":
            mean_train_metric = -train_loss
        else:
            mean_train_metric = getattr(train_epoch_metrics, metric_to_use)
        logger.log(PROGRESS_LOG_LEVEL, f"  Mean train loss: {train_loss:.5f}")
        logger.info(f"  Mean train {metric_to_use}: {mean_train_metric:.5f}")
        logger.info(f"  = Validation")
        valid_metric = valid_fn(model)
        logger.log(PROGRESS_LOG_LEVEL, f"  Validation metric: {valid_metric:.5f}")

        if valid_metric > best_valid_metric:
            logger.info(
                f"   New best validation result {valid_metric:.5f} (increased from {best_valid_metric:.5f})."
            )
            best_valid_metric = valid_metric
            epochs_since_best = 0
            save_model(model_save_file, model, quiet=quiet)
        else:
            epochs_since_best += 1
            logger.log(
                PROGRESS_LOG_LEVEL, f"   Now had {epochs_since_best} epochs since best result."
            )
            if epochs_since_best >= patience:
                break

    return best_valid_metric


def validate_on_data_iterable(
    model: SplitLRGraphBinaryClassificationTask,
    data_iterable: Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]],
    metric_to_use: MetricType = "avg_precision",
    quiet: bool = False,
) -> float:
    valid_loss, _, valid_results = model.run_one_epoch(data_iterable, training=False, quiet=quiet)
    valid_metrics = __metrics_from_batch_results(valid_results)
    logger.info(f"  Validation loss: {valid_loss:.5f}")
    if metric_to_use == "loss":
        return -valid_loss  # We are maximising things, so flip the sign on the loss
    else:
        return getattr(valid_metrics, metric_to_use)


def finetune_and_eval_on_task(
    model: SplitLRGraphBinaryClassificationTask,
    model_weights: Dict[str, tf.Tensor],
    train_samples: List[MoleculeDatapoint],
    valid_samples: List[MoleculeDatapoint],
    test_samples: List[MoleculeDatapoint],
    out_folder: str,
    max_num_nodes_in_batch: int,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 50,
    patience: int = 10,
    quiet: bool = False,
) -> Tuple[float, BinaryEvalMetrics]:
    model_save_file = os.path.join(out_folder, f"best_model.pkl")
    # We now need to set the parameters to their current values in the training model:
    for var in model.trainable_variables:
        # Note that the validation model is created under tf.name_scope("valid"), and so
        # variable "valid/foo/bar" corresponds to "foo/bar" in the full (metatraining) model:
        if var.name.startswith("valid/"):
            model_var_name = var.name.split("/", 1)[1]
        else:
            model_var_name = var.name
        var.assign(model_weights[model_var_name])
    model.reset_optimizer_state_to_initial()
    with restrict_console_log_level(logging.WARN):
        best_valid_metric = train_loop(
            model=model,
            train_data=TFGraphBatchIterable(
                samples=train_samples, max_num_nodes=max_num_nodes_in_batch
            ),
            valid_fn=partial(
                validate_on_data_iterable,
                data_iterable=TFGraphBatchIterable(
                    samples=valid_samples,
                    max_num_nodes=max_num_nodes_in_batch,
                ),
                metric_to_use="loss",
                quiet=True,
            ),
            model_save_file=model_save_file,
            metric_to_use=metric_to_use,
            max_num_epochs=max_num_epochs,
            patience=patience,
            quiet=True,
        )

    logger.log(
        PROGRESS_LOG_LEVEL,
        f" Best validation loss:        {float(best_valid_metric):.5f}",
    )
    # Load best model state and eval on test data:
    load_weights_verbosely(model_save_file, model)

    test_loss, _, test_model_results = model.run_one_epoch(
        TFGraphBatchIterable(samples=test_samples, max_num_nodes=max_num_nodes_in_batch),
        training=False,
        quiet=quiet,
    )
    test_metrics = __metrics_from_batch_results(test_model_results)
    logger.log(PROGRESS_LOG_LEVEL, f" Test loss:                   {float(test_loss):.5f}")
    logger.log(PROGRESS_LOG_LEVEL, f" Test metrics: {test_metrics}")

    return test_loss, test_metrics


def eval_model_by_finetuning_on_task(
    model: SplitLRGraphBinaryClassificationTask,
    model_weights: Dict[str, tf.Tensor],
    task: FSMolTask,
    train_set_sample_sizes: List[int],
    test_set_size: Optional[int],
    num_samples: int,
    max_num_nodes_in_batch: int,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 50,
    patience: int = 10,
    seed: int = 0,
    quiet: bool = False,
) -> Tuple[List[float], List[FSMolTaskSampleEvalResults]]:
    test_losses: List[float] = []
    test_results: List[FSMolTaskSampleEvalResults] = []
    for train_size in train_set_sample_sizes:
        task_sampler = StratifiedTaskSampler(
            train_size_or_ratio=train_size,
            valid_size_or_ratio=0.2,
            test_size_or_ratio=test_set_size,
            allow_smaller_test=True,
        )
        for run_idx in range(num_samples):
            logger.info(f"=== Evaluating on {task.name}, #train {train_size}, run {run_idx}")
            with tempfile.TemporaryDirectory() as temp_out_folder, prefix_log_msgs(
                f" Inner - {task.name} - Size {train_size:3d} - Run {run_idx}"
            ):
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

                test_loss, test_metrics = finetune_and_eval_on_task(
                    model,
                    model_weights,
                    train_samples=task_sample.train_samples,
                    valid_samples=task_sample.valid_samples,
                    test_samples=task_sample.test_samples,
                    out_folder=temp_out_folder,
                    max_num_nodes_in_batch=max_num_nodes_in_batch,
                    metric_to_use=metric_to_use,
                    max_num_epochs=max_num_epochs,
                    patience=patience,
                    quiet=quiet,
                )

                logger.info(
                    f" Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data."
                )
                logger.info(
                    f" Dataset sample test {metric_to_use}: {getattr(test_metrics, metric_to_use):.4f}"
                )
                test_losses.append(test_loss)
                test_results.append(
                    FSMolTaskSampleEvalResults(
                        task_name=task.name,
                        seed=seed + run_idx,
                        num_train=train_size,
                        num_test=len(task_sample.test_samples),
                        fraction_pos_train=task_sample.train_pos_label_ratio,
                        fraction_pos_test=task_sample.test_pos_label_ratio,
                        **dataclasses.asdict(test_metrics),
                    )
                )

    return test_losses, test_results


def eval_model_by_finetuning_on_tasks(
    model: SplitLRGraphBinaryClassificationTask,
    model_weights: Dict[str, tf.Tensor],
    tasks: Iterable[FSMolTask],
    max_num_nodes_in_batch: int,
    metric_to_use: MetricType = "avg_precision",
    seed: int = 0,
    train_set_sample_sizes: List[int] = [16, 128],
    test_set_size: Optional[int] = 512,
    num_samples: int = 5,
    aml_run=None,
) -> float:
    task_to_losses: Dict[str, List[float]] = {}
    task_to_results: Dict[str, List[BinaryEvalMetrics]] = {}
    for task in tasks:
        task_losses, task_metrics = eval_model_by_finetuning_on_task(
            model=model,
            model_weights=model_weights,
            task=task,
            train_set_sample_sizes=train_set_sample_sizes,
            test_set_size=test_set_size,
            num_samples=num_samples,
            max_num_nodes_in_batch=max_num_nodes_in_batch,
            metric_to_use=metric_to_use,
            seed=seed,
            quiet=True,
        )
        task_to_results[task.name] = task_metrics
        task_to_losses[task.name] = task_losses

    mean_metrics = avg_metrics_list(list(itertools.chain(*task_to_results.values())))
    if aml_run is not None:
        for metric_name, (metric_mean, _) in mean_metrics.items():
            aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

    if metric_to_use == "loss":
        return -np.mean(itertools.chain(*task_to_results.values()))
    else:
        return mean_metrics[metric_to_use][0]
