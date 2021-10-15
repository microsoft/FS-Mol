import argparse
import logging
import os
import pdb
import sys
import tempfile
import traceback
from functools import partial
from typing import Optional

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import DataFold, FSMolDataset, FSMolTaskSample
from fs_mol.data.multitask import MultitaskTaskSampleBatchIterable, get_multitask_inference_batcher
from fs_mol.models.abstract_torch_fsmol_model import (
    MetricType,
    eval_model_by_finetuning_on_task,
    train_loop,
    create_optimizer,
    save_model,
)
from fs_mol.models.gnn_multitask import GNNMultitaskConfig, GNNMultitaskModel, create_model
from fs_mol.modules.graph_feature_extractor import (
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)
from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run
from fs_mol.utils.metrics import avg_metrics_over_tasks, BinaryEvalMetrics
from fs_mol.utils.test_utils import eval_model


SMALL_NUMBER = 1e-7


logger = logging.getLogger(__name__)


def validate_by_finetuning_on_tasks(
    model: GNNMultitaskModel,
    dataset: FSMolDataset,
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

        def test_model_fn(
            task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
        ) -> BinaryEvalMetrics:
            return eval_model_by_finetuning_on_task(
                current_model_path,
                model_cls=GNNMultitaskModel,
                task_sample=task_sample,
                batcher=get_multitask_inference_batcher(
                    max_num_graphs=batch_size, device=model_device
                ),
                learning_rate=learning_rate,
                task_specific_learning_rate=task_specific_learning_rate,
                metric_to_use=metric_to_use,
                seed=seed,
                quiet=True,
                device=model_device,
            )

        task_to_results = eval_model(
            test_model_fn=test_model_fn,
            dataset=dataset,
            train_set_sample_sizes=[16, 128],
            num_samples=3,
            valid_size_or_ratio=0.2,
            test_size_or_ratio=512,
            fold=DataFold.VALIDATION,
            seed=seed,
        )

        mean_metrics = avg_metrics_over_tasks(task_to_results)
        if aml_run is not None:
            for metric_name, (metric_mean, _) in mean_metrics.items():
                aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

        model = model.to(model_device)

        return mean_metrics[metric_to_use][0]


def add_model_arguments(parser: argparse.ArgumentParser):
    add_graph_feature_extractor_arguments(parser)
    parser.add_argument("--num_tail_layers", type=int, default=2)


def make_model_from_args(
    num_tasks: int, args: argparse.Namespace, device: Optional[torch.device] = None
):
    model_config = GNNMultitaskConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        num_tasks=num_tasks,
        num_tail_layers=args.num_tail_layers,
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

    out_dir, fsmol_dataset, aml_run = set_up_train_run("Multitask", args, torch=True)

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
        dataset=fsmol_dataset,
        learning_rate=args.finetune_lr_scale * args.learning_rate,
        task_specific_learning_rate=args.finetune_lr_scale * task_specific_lr,
        batch_size=args.batch_size,
        metric_to_use=args.metric_to_use,
        seed=args.seed,
        aml_run=aml_run,
    )

    _, best_model_state = train_loop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=MultitaskTaskSampleBatchIterable(
            fsmol_dataset,
            data_fold=DataFold.TRAIN,
            task_name_to_id=train_task_name_to_id,
            max_num_graphs=args.batch_size,
            device=device,
        ),
        valid_fn=valid_fn,
        metric_to_use=args.metric_to_use,
        max_num_epochs=args.num_epochs,
        patience=args.patience,
        aml_run=aml_run,
    )

    torch.save(best_model_state, os.path.join(out_dir, "best_model.pt"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
