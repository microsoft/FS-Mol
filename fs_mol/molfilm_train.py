import argparse
import dataclasses
import itertools
import logging
import os
import pdb
import sys
import tempfile
import traceback
from collections import defaultdict
from functools import partial
from typing import Tuple, Dict, List, Optional, DefaultDict, Callable, Iterable, Union, Type
from typing_extensions import Literal

import numpy as np
import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import (
    NUM_EDGE_TYPES,
    NUM_NODE_FEATURES,
    DataFold,
    FSMolBatcher,
    FSMolTask,
    FSMolBatchIterable,
    StratifiedTaskSampler,
    DatasetClassTooSmallException,
    DatasetTooSmallException,
    FoldTooSmallException,
)
from fs_mol.data.molfilm import (
    FSMolMolFiLMBatch,
    MolFiLMTaskSampleBatchIterable,
    get_molfilm_inference_batcher,
)
from fs_mol.models.interface import AbstractTorchModel
from fs_mol.models.mol_pred_model import MolPredConfig, MolPredModel, ThickGNNConfig, create_model
from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run, str2bool
from fs_mol.utils.logging import PROGRESS_LOG_LEVEL, prefix_log_msgs
from fs_mol.utils.metric_logger import MetricLogger
from fs_mol.utils.metrics import (
    avg_metrics_list,
    compute_metrics,
    BinaryEvalMetrics,
    BinaryMetricType,
)
from fs_mol.utils.molfilm_utils import create_optimizer, save_model
from fs_mol.utils.test_utils import FSMolTaskSampleEvalResults


SMALL_NUMBER = 1e-7


logger = logging.getLogger(__name__)


MetricType = Union[BinaryMetricType, Literal["loss"]]


def run_on_data_iterable(
    model: AbstractTorchModel,
    data_iterable: Iterable[Tuple[FSMolMolFiLMBatch, np.ndarray]],
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
        data_iterable: Iterable that provides the data we run on.
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

        # Apply sigmoid to have predictions in appropriate range for computing (scikit) scores.
        predictions = torch.sigmoid(predictions).detach().cpu()

        if hasattr(batch, "sample_to_task_id"):

            def get_task_id(idx: int) -> int:
                return batch.sample_to_task_id[idx]

        else:

            def get_task_id(idx: int) -> int:
                return 0

        num_samples = labels.shape[0]

        for i in range(num_samples):
            task_id = get_task_id(i)

            per_task_preds[task_id].append(predictions[i].item())
            per_task_labels[task_id].append(labels[i])

    metrics = compute_metrics(per_task_preds, per_task_labels)

    return metric_logger.get_mean_metric_value("loss"), metrics


def validate_on_data_iterable(
    model: AbstractTorchModel,
    data_iterable: Iterable[Tuple[FSMolMolFiLMBatch, np.ndarray]],
    metric_to_use: MetricType = "avg_precision",
    quiet: bool = False,
) -> float:
    valid_loss, valid_metrics = run_on_data_iterable(
        model,
        data_iterable=data_iterable,
        quiet=quiet,
    )
    if not quiet:
        logger.info(f"  Validation loss: {valid_loss:.5f}")
    # If our data_iterable had more than one task, we'll have one result per task - average them:
    mean_valid_metrics = avg_metrics_list(list(valid_metrics.values()))
    if metric_to_use == "loss":
        return -valid_loss  # We are maximising things, so flip the sign on the loss
    else:
        return mean_valid_metrics[metric_to_use][0]


def eval_model_by_finetuning_on_task(
    model_weights_file: str,
    model_cls: Type[AbstractTorchModel],
    task: FSMolTask,
    batcher: FSMolBatcher,
    train_set_sample_sizes: List[int],
    test_set_size: Optional[int],
    num_samples: int,
    learning_rate: float,
    task_specific_learning_rate: float,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 50,
    patience: int = 10,
    seed: int = 0,
    quiet: bool = False,
    device: Optional[torch.device] = None,
) -> List[FSMolTaskSampleEvalResults]:
    test_results: List[FSMolTaskSampleEvalResults] = []
    for train_size in train_set_sample_sizes:
        task_sampler = StratifiedTaskSampler(
            train_size_or_ratio=train_size,
            valid_size_or_ratio=0.2,
            test_size_or_ratio=test_set_size,
            allow_smaller_test=True,
        )
        for run_idx in range(num_samples):
            logger.info(
                f"=== Evaluating on {task.name}, #train {train_size}, run {run_idx}",
            )
            with prefix_log_msgs(
                f" Inner - {task.name} - Size {train_size:3d} - Run {run_idx}"
            ), tempfile.TemporaryDirectory() as temp_out_folder:
                try:
                    task_sample = task_sampler.sample(task, seed=seed + run_idx)
                except (
                    DatasetTooSmallException,
                    DatasetClassTooSmallException,
                    FoldTooSmallException,
                    ValueError,
                ) as e:
                    logger.warning(
                        f"Failed to draw sample with {train_size} train points for {task.name}. Skipping."
                    )
                    logger.debug("Sampling error: " + str(e))
                    continue

                # Build the model afresh and load the shared weights.
                model = model_cls.build_from_model_file(
                    model_weights_file,
                    quiet=quiet,
                    device=device,
                    config_overrides={
                        "num_tasks": 1,
                    },
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
                    train_data=FSMolBatchIterable(
                        task_sample.train_samples, batcher, shuffle=True, seed=seed + run_idx
                    ),
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

                logger.log(
                    PROGRESS_LOG_LEVEL,
                    f" Final validation loss:       {float(best_valid_metric):.5f}",
                )
                # Load best model state and eval on test data:
                best_trained_model_file = os.path.join(temp_out_folder, "best_model.pt")
                model.load_model_weights(best_trained_model_file, load_task_specific_weights=True)
                test_loss, _test_metrics = run_on_data_iterable(
                    model,
                    data_iterable=FSMolBatchIterable(task_sample.test_samples, batcher),
                    quiet=quiet,
                )
                test_metrics = next(iter(_test_metrics.values()))
                logger.log(
                    PROGRESS_LOG_LEVEL, f" Test loss:                   {float(test_loss):.5f}"
                )
                logger.info(f" Test metrics: {test_metrics}")
                logger.info(
                    f"Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data.",
                )
                logger.log(
                    PROGRESS_LOG_LEVEL,
                    f"Dataset sample test {metric_to_use}: {getattr(test_metrics, metric_to_use):.4f}",
                )
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

    return test_results


def validate_by_finetuning_on_tasks(
    model: AbstractTorchModel,
    tasks: Iterable[FSMolTask],
    learning_rate: float,
    task_specific_learning_rate: float,
    batch_size: int = 128,
    metric_to_use: MetricType = "avg_precision",
    seed: int = 0,
    aml_run=None,
) -> float:
    with tempfile.TemporaryDirectory() as tempdir:
        # First, store the current state of the model, so that we can just load it back in
        # repeatedly as starting point during finetuning:
        current_model_path = os.path.join(tempdir, "cur_model.pt")
        save_model(current_model_path, model)

        # Move model off GPU to make space for validation model:
        model_device = model.device
        model = model.to(torch.device("cpu"))

        task_to_results: Dict[str, List[FSMolTaskSampleEvalResults]] = {}
        for task in tasks:
            task_metrics = eval_model_by_finetuning_on_task(
                current_model_path,
                model_cls=MolPredModel,
                task=task,
                batcher=get_molfilm_inference_batcher(max_num_graphs=batch_size),
                train_set_sample_sizes=[16, 128],
                test_set_size=512,
                num_samples=3,
                learning_rate=learning_rate,
                task_specific_learning_rate=task_specific_learning_rate,
                metric_to_use=metric_to_use,
                seed=seed,
                quiet=True,
                device=model_device,
            )
            task_to_results[task.name] = task_metrics

        mean_metrics = avg_metrics_list(list(itertools.chain(*task_to_results.values())))
        if aml_run is not None:
            for metric_name, (metric_mean, _) in mean_metrics.items():
                aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

        model = model.to(model_device)

        return mean_metrics[metric_to_use][0]


def train_loop(
    model: AbstractTorchModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_data: Iterable[Tuple[FSMolMolFiLMBatch, np.ndarray]],
    valid_fn: Callable[[AbstractTorchModel], float],
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
    # initial_valid_metric = float("-inf")
    initial_valid_metric = valid_fn(model)
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


def add_model_arguments(parser: argparse.ArgumentParser):
    # GNN parameters.
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="PNA",
        choices=["MultiHeadAttention", "MultiAggr", "PNA", "Plain"],
        help="Type of GNN architecture to use.",
    )
    parser.add_argument(
        "--num_gnn_layers", type=int, default=10, help="Number of GNN layers to use."
    )
    parser.add_argument(
        "--node_embed_dim", type=int, default=128, help="Size of GNN node representations."
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of heads used in each GNN message propagation step. Relevant in MultiHeadAttention.",
    )
    parser.add_argument(
        "--per_head_dim",
        type=int,
        default=64,
        help="Size of message representation in each attention head.",
    )
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=1024,
        help="Size of intermediate representation used in BOOM layer. Set to 0 to deactivate BOOM layer.",
    )
    parser.add_argument("--message_function_depth", type=int, default=1)
    parser.add_argument(
        "--use_msg_film",
        type=str2bool,
        default=True,
        help="Toggle task-dependent modulation of GNN messages.",
    )
    parser.add_argument(
        "--use_msg_att_film",
        type=str2bool,
        default=False,
        help="Toggle task-dependent modulation of GNN message attention scores.",
    )

    parser.add_argument(
        "--readout_type",
        type=str,
        default="combined_task",
        choices=[
            "sum",
            "min",
            "max",
            "mean",
            "weighted_sum",
            "weighted_mean",
            "task_weighted_sum",
            "task_weighted_mean",
            "combined",
            "combined_task",
        ],
        help="Readout used to summarise atoms into a molecule",
    )
    parser.add_argument(
        "--readout_use_all_states",
        type=str2bool,
        default=True,
        help="Indicates if all intermediate GNN activations or only the final ones should be used when computing a graph-level representation.",
    )
    parser.add_argument(
        "--use_init_film",
        type=str2bool,
        default=True,
        help="Toggle task-dependent modulation of initial node representations.",
    )
    parser.add_argument(
        "--use_tail_task_emb",
        type=str2bool,
        default=False,
        help="Toggle use of an additional task embedding as input to output MLP.",
    )
    parser.add_argument(
        "--use_output_masking",
        type=str2bool,
        default=True,
        help="Produce predictions for all tasks for each input, but only compute loss on correct one.",
    )
    parser.add_argument("--num_tail_layers", type=int, default=2)
    parser.add_argument(
        "--task_embedding_dim",
        type=int,
        help="Size of task embeddings. If unspecified, size is automatically determined by usage. If specified, appropriate projections for all usages are instantiated.",
    )


def make_model_from_args(
    num_tasks: int, args: argparse.Namespace, device: Optional[torch.device] = None
):
    model_config = MolPredConfig(
        num_tasks=num_tasks,
        node_feature_dim=NUM_NODE_FEATURES,
        gnn_config=ThickGNNConfig(
            type=args.gnn_type,
            hidden_dim=args.node_embed_dim,
            num_edge_types=NUM_EDGE_TYPES,
            num_heads=args.num_heads,
            per_head_dim=args.per_head_dim,
            intermediate_dim=args.intermediate_dim,
            message_function_depth=args.message_function_depth,
            num_layers=args.num_gnn_layers,
            use_msg_film=args.use_msg_film,
            use_msg_att_film=args.use_msg_att_film,
        ),
        num_outputs=1,
        readout_type=args.readout_type,
        readout_use_only_last_timestep=not args.readout_use_all_states,
        num_tail_layers=args.num_tail_layers,
        use_init_film=args.use_init_film,
        use_tail_task_emb=args.use_tail_task_emb,
        use_output_masking=args.use_output_masking,
        task_embedding_dim=args.task_embedding_dim,
    )
    model = create_model(model_config, device=device)
    return model


def add_train_loop_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
        help="Learning rate for shared model components.",
    )
    parser.add_argument(
        "--metric-to-use",
        type=str,
        choices=[
            "acc",
            "balanced_acc",
            "f1",
            "prec",
            "recall",
            "roc_auc",
            "avg_precision",
            "kappa",
        ],
        default="avg_precision",
        help="Metric to evaluate on validation data.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a Multitask GNN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_cli_args(parser)

    add_model_arguments(parser)

    # Training parameters:
    add_train_loop_arguments(parser)
    parser.add_argument(
        "--task-specific-lr",
        type=float,
        default=0.0001,
        help="Learning rate for shared model components. By default, 10x core learning rate.",
    )
    parser.add_argument(
        "--finetune-lr-scale",
        type=float,
        default=1.0,
        help="Scaling factor for LRs used in finetuning eval.",
    )

    args = parser.parse_args()

    out_dir, fsmol_dataset, aml_run = set_up_train_run("MolFiLM", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model_from_args(
        num_tasks=fsmol_dataset.get_num_fold_tasks(DataFold.TRAIN), args=args, device=device
    )
    logger.info(f"\tNum parameters {sum(p.numel() for p in model.parameters())}")
    logger.info(f"\tDevice: {device}")
    logger.info(f"\tModel:\n{model}")

    train_task_name_to_id = {
        name: i for i, name in enumerate(fsmol_dataset.get_task_names(data_fold=DataFold.TRAIN))
    }
    if args.task_specific_lr is not None:
        task_specific_lr = args.task_specific_lr
    else:
        task_specific_lr = 10 * args.learning_rate

    optimizer, lr_scheduler = create_optimizer(
        model,
        lr=args.learning_rate,
        task_specific_lr=task_specific_lr,
        warmup_steps=100,
        task_specific_warmup_steps=100,
    )

    # Validation is done by finetuning on a bunch of tasks:
    valid_fn = partial(
        validate_by_finetuning_on_tasks,
        tasks=fsmol_dataset.get_task_reading_iterable(DataFold.VALIDATION),
        learning_rate=args.finetune_lr_scale * args.learning_rate,
        task_specific_learning_rate=args.finetune_lr_scale * task_specific_lr,
        batch_size=args.batch_size,
        metric_to_use=args.metric_to_use,
        seed=args.seed,
        aml_run=aml_run,
    )

    train_loop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=MolFiLMTaskSampleBatchIterable(
            fsmol_dataset,
            data_fold=DataFold.TRAIN,
            task_name_to_id=train_task_name_to_id,
            max_num_graphs=args.batch_size,
        ),
        valid_fn=valid_fn,
        output_folder=out_dir,
        metric_to_use=args.metric_to_use,
        max_num_epochs=args.num_epochs,
        patience=args.patience,
        aml_run=aml_run,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
