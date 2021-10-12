import logging
import itertools
import os
import pickle
from functools import partial
from typing import Dict, Any, Iterable, List, Optional, Tuple, Callable, Union
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from tf2_gnn.cli_utils.model_utils import _get_name_to_variable_map, load_weights_verbosely
from tf2_gnn.cli_utils.dataset_utils import get_model_file_path

from fs_mol.data import DataFold, FSMolDataset, FSMolTaskSample
from fs_mol.data.maml import TFGraphBatchIterable
from fs_mol.models.metalearning_graph_binary_classification import (
    MetalearningGraphBinaryClassificationTask,
)
from fs_mol.utils.logging import PROGRESS_LOG_LEVEL, restrict_console_log_level
from fs_mol.utils.metrics import (
    BinaryEvalMetrics,
    BinaryMetricType,
    avg_metrics_over_tasks,
    compute_binary_task_metrics,
)
from fs_mol.utils.test_utils import eval_model


logger = logging.getLogger(__name__)


MetricType = Union[BinaryMetricType, Literal["loss"]]


def save_model(
    save_file: str,
    model: MetalearningGraphBinaryClassificationTask,
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
    model: MetalearningGraphBinaryClassificationTask,
    train_data: Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]],
    valid_fn: Callable[[MetalearningGraphBinaryClassificationTask], float],
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
    model: MetalearningGraphBinaryClassificationTask,
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


def eval_model_by_finetuning_on_task(
    model: MetalearningGraphBinaryClassificationTask,
    model_weights: Dict[str, tf.Tensor],
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    max_num_nodes_in_batch: int,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 50,
    patience: int = 10,
    quiet: bool = False,
) -> BinaryEvalMetrics:
    model_save_file = os.path.join(temp_out_folder, f"best_model.pkl")
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
                samples=task_sample.train_samples, max_num_nodes=max_num_nodes_in_batch
            ),
            valid_fn=partial(
                validate_on_data_iterable,
                data_iterable=TFGraphBatchIterable(
                    samples=task_sample.valid_samples, max_num_nodes=max_num_nodes_in_batch
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

    logger.log(PROGRESS_LOG_LEVEL, f" Best validation loss:        {float(best_valid_metric):.5f}")
    # Load best model state and eval on test data:
    load_weights_verbosely(model_save_file, model)

    test_loss, _, test_model_results = model.run_one_epoch(
        TFGraphBatchIterable(
            samples=task_sample.test_samples, max_num_nodes=max_num_nodes_in_batch
        ),
        training=False,
        quiet=quiet,
    )
    test_metrics = __metrics_from_batch_results(test_model_results)
    logger.log(PROGRESS_LOG_LEVEL, f" Test loss:                   {float(test_loss):.5f}")
    logger.log(PROGRESS_LOG_LEVEL, f" Test metrics: {test_metrics}")

    logger.info(
        f" Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data."
    )
    logger.info(f" Dataset sample test {metric_to_use}: {getattr(test_metrics, metric_to_use):.4f}")

    return test_metrics


def eval_model_by_finetuning_on_tasks(
    model: MetalearningGraphBinaryClassificationTask,
    model_weights: Dict[str, tf.Tensor],
    dataset: FSMolDataset,
    max_num_nodes_in_batch: int,
    metric_to_use: MetricType = "avg_precision",
    seed: int = 0,
    train_set_sample_sizes: List[int] = [16, 128],
    test_set_size: Optional[int] = 512,
    num_samples: int = 5,
    aml_run=None,
) -> float:
    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> BinaryEvalMetrics:
        return eval_model_by_finetuning_on_task(
            model=model,
            model_weights=model_weights,
            task_sample=task_sample,
            temp_out_folder=temp_out_folder,
            max_num_nodes_in_batch=max_num_nodes_in_batch,
            metric_to_use=metric_to_use,
            quiet=True,
        )

    task_to_results = eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=train_set_sample_sizes,
        num_samples=num_samples,
        valid_size_or_ratio=0.2,
        test_size_or_ratio=test_set_size,
        fold=DataFold.VALIDATION,
        seed=seed,
    )

    mean_metrics = avg_metrics_over_tasks(task_to_results)
    if aml_run is not None:
        for metric_name, (metric_mean, _) in mean_metrics.items():
            aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

    if metric_to_use == "loss":
        return -np.mean(itertools.chain(*task_to_results.values()))
    else:
        return mean_metrics[metric_to_use][0]
