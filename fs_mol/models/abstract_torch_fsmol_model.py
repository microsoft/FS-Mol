from __future__ import annotations

import logging
import os
import sys
import time
from abc import abstractclassmethod, abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Tuple,
    Dict,
    List,
    Optional,
    DefaultDict,
    Callable,
    Iterable,
    Union,
    Type,
    Any,
    Generic,
    TypeVar,
)
from typing_extensions import Literal

import numpy as np
import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import (
    FSMolBatcher,
    FSMolBatchIterable,
    FSMolTaskSample,
)
from fs_mol.utils.logging import PROGRESS_LOG_LEVEL
from fs_mol.utils.metric_logger import MetricLogger
from fs_mol.utils.metrics import (
    avg_metrics_list,
    compute_metrics,
    BinaryEvalMetrics,
    BinaryMetricType,
)

logger = logging.getLogger(__name__)


BatchFeaturesType = TypeVar("BatchFeaturesType")
MetricType = Union[BinaryMetricType, Literal["loss"]]


class AbstractTorchFSMolModel(Generic[BatchFeaturesType], torch.nn.Module):
    @abstractmethod
    def forward(self, batch: BatchFeaturesType) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def is_param_task_specific(self, param_name: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load model weights from a saved checkpoint."""
        raise NotImplementedError()

    @abstractclassmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> AbstractTorchFSMolModel[BatchFeaturesType]:
        """Build the model architecture based on a saved checkpoint."""
        raise NotImplementedError()


def create_optimizer(
    model: AbstractTorchFSMolModel[BatchFeaturesType],
    lr: float = 0.001,
    task_specific_lr: float = 0.005,
    warmup_steps: int = 1000,
    task_specific_warmup_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    # Split parameters into shared and task-specific ones:
    shared_parameters, task_spec_parameters = [], []
    for param_name, param in model.named_parameters():
        if model.is_param_task_specific(param_name):
            task_spec_parameters.append(param)
        else:
            shared_parameters.append(param)

    opt = torch.optim.Adam(
        [
            {"params": task_spec_parameters, "lr": task_specific_lr},
            {"params": shared_parameters, "lr": lr},
        ],
    )

    def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
        if cur_step >= warmup_steps:
            return 1.0
        return cur_step / warmup_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=[
            partial(
                linear_warmup, warmup_steps=task_specific_warmup_steps
            ),  # for task specific paramters
            partial(linear_warmup, warmup_steps=warmup_steps),  # for shared paramters
        ],
    )

    return opt, scheduler


def save_model(
    path: str,
    model: AbstractTorchFSMolModel[BatchFeaturesType],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
):
    data = model.get_model_state()

    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch

    torch.save(data, path)


def resolve_starting_model_file(
    model_file: str,
    model_cls: Type[AbstractTorchFSMolModel[BatchFeaturesType]],
    out_dir: str,
    use_fresh_param_init: bool,
    config_overrides: Dict[str, Any] = {},
    device: Optional[torch.device] = None,
):
    # If we start from a fresh init, create a model, do a random init, and store that away somewhere:
    if use_fresh_param_init:
        logger.info("Using fresh model init.")
        model = model_cls.build_from_model_file(
            model_file=model_file, config_overrides=config_overrides, device=device
        )

        resolved_model_file = os.path.join(out_dir, f"fresh_init.pkl")
        save_model(resolved_model_file, model)

        # Hack to give AML some time to actually save.
        time.sleep(1)
    else:
        resolved_model_file = model_file
        logger.info(f"Using model weights loaded from {resolved_model_file}.")

    return resolved_model_file


def run_on_data_iterable(
    model: AbstractTorchFSMolModel[BatchFeaturesType],
    data_iterable: Iterable[Tuple[BatchFeaturesType, np.ndarray]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    max_num_steps: Optional[int] = None,
    quiet: bool = False,
    metric_name_prefix: str = "",
    aml_run=None,
) -> Tuple[float, Dict[int, BinaryEvalMetrics]]:
    """Run the given model on the provided data loader.

    Args:
        model: Model to run things on.
        data_iterable: Iterable that provides the data we run on; data has been batched
            by an appropriate batcher.
        optimizer: Optional optimizer. If present, the given model will be trained.
        lr_scheduler: Optional learning rate scheduler around optimizer.
        max_num_steps: Optional number of steps. If not provided, will run until end of data loader.
    """

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    if optimizer is None:
        model.eval()
    else:
        model.train()

    per_task_preds: DefaultDict[int, List[float]] = defaultdict(list)
    per_task_labels: DefaultDict[int, List[float]] = defaultdict(list)

    metric_logger = MetricLogger(
        log_fn=lambda msg: logger.log(PROGRESS_LOG_LEVEL, msg),
        aml_run=aml_run,
        quiet=quiet,
        metric_name_prefix=metric_name_prefix,
    )
    for batch_idx, (batch, labels) in enumerate(iter(data_iterable)):
        if max_num_steps is not None and batch_idx >= max_num_steps:
            break

        if optimizer is not None:
            optimizer.zero_grad()

        predictions = model(batch)
        predictions = predictions.squeeze(dim=-1)

        # Compute loss and weigh it:
        loss = torch.mean(
            criterion(
                predictions, torch.tensor(labels, device=predictions.device, dtype=torch.float)
            )
        )
        metric_logger.log_metrics(loss=loss.detach().cpu().item())

        # Training step:
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # === Finally, collect per-task results to be used for further eval:
        sample_to_task_id: Dict[int, int] = {}
        if hasattr(batch, "sample_to_task_id"):
            sample_to_task_id = batch.sample_to_task_id
        else:
            # If we don't have a sample task information, just use 0 as default task ID:
            sample_to_task_id = defaultdict(lambda: 0)

        # Apply sigmoid to have predictions in appropriate range for computing (scikit) scores.
        num_samples = labels.shape[0]
        predictions = torch.sigmoid(predictions).detach().cpu()
        for i in range(num_samples):
            task_id = sample_to_task_id[i]
            per_task_preds[task_id].append(predictions[i].item())
            per_task_labels[task_id].append(labels[i])

    metrics = compute_metrics(per_task_preds, per_task_labels)

    return metric_logger.get_mean_metric_value("loss"), metrics


def validate_on_data_iterable(
    model: AbstractTorchFSMolModel[BatchFeaturesType],
    data_iterable: Iterable[Tuple[BatchFeaturesType, np.ndarray]],
    metric_to_use: MetricType = "avg_precision",
    quiet: bool = False,
) -> float:
    valid_loss, valid_metrics = run_on_data_iterable(
        model, data_iterable=data_iterable, quiet=quiet
    )
    if not quiet:
        logger.info(f"  Validation loss: {valid_loss:.5f}")
    # If our data_iterable had more than one task, we'll have one result per task - average them:
    mean_valid_metrics = avg_metrics_list(list(valid_metrics.values()))
    if metric_to_use == "loss":
        return -valid_loss  # We are maximising things elsewhere, so flip the sign on the loss
    else:
        return mean_valid_metrics[metric_to_use][0]


def train_loop(
    model: AbstractTorchFSMolModel[BatchFeaturesType],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_data: Iterable[Tuple[BatchFeaturesType, np.ndarray]],
    valid_fn: Callable[[AbstractTorchFSMolModel[BatchFeaturesType]], float],
    output_folder: str,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 100,
    patience: int = 5,
    aml_run=None,
    quiet: bool = False,
):
    if quiet:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    initial_valid_metric = float("-inf")
    best_valid_metric = initial_valid_metric
    logger.log(log_level, f"  Initial validation metric: {best_valid_metric:.5f}")

    save_model(os.path.join(output_folder, "best_model.pt"), model, optimizer, -1)

    epochs_since_best = 0
    for epoch in range(0, max_num_epochs):
        logger.log(log_level, f"== Epoch {epoch}")
        logger.log(log_level, f"  = Training")
        train_loss, train_metrics = run_on_data_iterable(
            model,
            data_iterable=train_data,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            quiet=quiet,
            metric_name_prefix="train_",
            aml_run=aml_run,
        )
        mean_train_metric = np.mean(
            [getattr(task_metrics, metric_to_use) for task_metrics in train_metrics.values()]
        )
        logger.log(log_level, f"  Mean train loss: {train_loss:.5f}")
        logger.log(log_level, f"  Mean train {metric_to_use}: {mean_train_metric:.5f}")
        logger.log(log_level, f"  = Validation")
        valid_metric = valid_fn(model)
        logger.log(log_level, f"  Validation metric: {valid_metric:.5f}")

        if valid_metric > best_valid_metric:
            logger.log(
                log_level,
                f"   New best validation result {valid_metric:.5f} (increased from {best_valid_metric:.5f}).",
            )
            best_valid_metric = valid_metric
            epochs_since_best = 0

            save_model(os.path.join(output_folder, "best_model.pt"), model, optimizer, epoch)
        else:
            epochs_since_best += 1
            logger.log(log_level, f"   Now had {epochs_since_best} epochs since best result.")
            if epochs_since_best >= patience:
                break

    return best_valid_metric


def eval_model_by_finetuning_on_task(
    model_weights_file: str,
    model_cls: Type[AbstractTorchFSMolModel[BatchFeaturesType]],
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    batcher: FSMolBatcher[BatchFeaturesType, np.ndarray],
    learning_rate: float,
    task_specific_learning_rate: float,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 50,
    patience: int = 10,
    seed: int = 0,
    quiet: bool = False,
    device: Optional[torch.device] = None,
) -> BinaryEvalMetrics:
    # Build the model afresh and load the shared weights.
    model: AbstractTorchFSMolModel[BatchFeaturesType] = model_cls.build_from_model_file(
        model_weights_file, quiet=quiet, device=device, config_overrides={"num_tasks": 1}
    )
    model.load_model_weights(model_weights_file, load_task_specific_weights=False)

    (optimizer, lr_scheduler) = create_optimizer(
        model,
        lr=learning_rate,
        task_specific_lr=task_specific_learning_rate,
        warmup_steps=2,
        task_specific_warmup_steps=2,
    )

    best_valid_metric = train_loop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=FSMolBatchIterable(task_sample.train_samples, batcher, shuffle=True, seed=seed),
        valid_fn=partial(
            validate_on_data_iterable,
            data_iterable=FSMolBatchIterable(task_sample.valid_samples, batcher),
            metric_to_use="loss",
            quiet=quiet,
        ),
        output_folder=temp_out_folder,
        metric_to_use=metric_to_use,
        max_num_epochs=max_num_epochs,
        patience=patience,
        quiet=True,
    )

    logger.log(PROGRESS_LOG_LEVEL, f" Final validation loss:       {float(best_valid_metric):.5f}")
    # Load best model state and eval on test data:
    best_trained_model_file = os.path.join(temp_out_folder, "best_model.pt")
    model.load_model_weights(best_trained_model_file, load_task_specific_weights=True)
    test_loss, _test_metrics = run_on_data_iterable(
        model, data_iterable=FSMolBatchIterable(task_sample.test_samples, batcher), quiet=quiet
    )
    test_metrics = next(iter(_test_metrics.values()))
    logger.log(PROGRESS_LOG_LEVEL, f" Test loss:                   {float(test_loss):.5f}")
    logger.info(f" Test metrics: {test_metrics}")
    logger.info(
        f"Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data.",
    )
    logger.log(
        PROGRESS_LOG_LEVEL,
        f"Dataset sample test {metric_to_use}: {getattr(test_metrics, metric_to_use):.4f}",
    )

    return test_metrics
