import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset
from fs_mol.data.protonet import ProtoNetBatch, get_protonet_task_sample_iterable
from fs_mol.models.protonet import PrototypicalNetwork, PrototypicalNetworkConfig
from fs_mol.utils.metrics import BinaryEvalMetrics, compute_binary_task_metrics, avg_metrics_list
from fs_mol.utils.metric_logger import MetricLogger


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrototypicalNetworkTrainerConfig(PrototypicalNetworkConfig):
    batch_size: int = 256
    tasks_per_batch: int = 16
    support_set_size: int = 16
    query_set_size: int = 256

    num_train_steps: int = 10000
    validate_every_num_steps: int = 50

    learning_rate: float = 0.001
    clip_value: Optional[float] = None


def run_on_batches(
    model: PrototypicalNetwork,
    batches: List[ProtoNetBatch],
    batch_labels: List[np.ndarray],
    train: bool = False,
    tasks_per_batch: int = 1,
) -> Tuple[float, BinaryEvalMetrics]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_num_samples = 0.0, 0
    task_preds: List[np.ndarray] = []
    task_labels: List[np.ndarray] = []

    num_gradient_accumulation_steps = len(batches) * tasks_per_batch
    for batch_features, batch_labels in zip(batches, batch_labels):
        # Compute task loss
        batch_logits = model(batch_features)
        batch_loss = model.compute_loss(batch_logits, batch_labels)
        # divide this batch loss by the total number of accumulation steps
        batch_loss = batch_loss / num_gradient_accumulation_steps
        if train:
            batch_loss.backward()
        total_loss += (
            batch_loss.detach() * batch_features.num_query_samples * num_gradient_accumulation_steps
        )
        total_num_samples += batch_features.num_query_samples
        batch_preds = torch.nn.functional.softmax(batch_logits, dim=1).detach().cpu().numpy()
        task_preds.append(batch_preds[:, 1])
        task_labels.append(batch_labels)

    metrics = compute_binary_task_metrics(
        predictions=np.concatenate(task_preds, axis=0), labels=np.concatenate(task_labels, axis=0)
    )

    # we will report loss per sample as before.
    return total_loss.cpu().item() / total_num_samples, metrics


class PrototypicalNetworkTrainer(PrototypicalNetwork):
    def __init__(self, config: PrototypicalNetworkTrainerConfig):
        super().__init__(config)
        self.config = config
        self.optimizer = torch.optim.Adam(self.parameters(), config.learning_rate)

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def save_model(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ):
        data = self.get_model_state()

        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            data["epoch"] = epoch

        torch.save(data, path)

    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        for name, param in pretrained_state_dict["model_state_dict"].items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

        optimizer_weights = pretrained_state_dict.get("optimizer_state_dict")
        if optimizer_weights is not None:
            for name, param in optimizer_weights.items():
                self.optimizer.state_dict()[name].copy_(param)

    def load_model_gnn_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        gnn_model_state_dict = pretrained_state_dict["model_state_dict"]
        our_state_dict = self.state_dict()

        # This is somewhat specialised to the case of using weights from the GNNMultitask model:
        for name, param in gnn_model_state_dict.items():
            if name.startswith("gnn.gnn_blocks") or name.startswith("init_node_proj"):
                our_name = f"graph_feature_extractor.{name}"
                our_state_dict[our_name].copy_(param)
            else:
                logger.debug(f"I: Not loading parameter {name}")


    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> "PrototypicalNetworkTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        if not quiet:
            logger.info(f" Loading model configuration from {model_file}.")

        model = PrototypicalNetworkTrainer(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            load_task_specific_weights=True,
            device=device,
        )
        return model

    def train_loop(self, out_dir: str, dataset: FSMolDataset, aml_run=None):
        self.save_model(os.path.join(out_dir, "best_validation.pt"))

        train_task_sample_iterator = iter(
            get_protonet_task_sample_iterable(
                dataset=dataset,
                data_fold=DataFold.TRAIN,
                num_samples=1,
                max_num_graphs=self.config.batch_size,
                support_size=self.config.support_set_size,
                query_size=self.config.query_set_size,
                repeat=True,
            )
        )

        best_validation_avg_prec = 0.0
        metric_logger = MetricLogger(
            log_fn=lambda msg: logger.info(msg),
            aml_run=aml_run,
            window_size=max(10, self.config.validate_every_num_steps / 5),
        )

        for step in range(1, self.config.num_train_steps + 1):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            task_batch_losses: List[float] = []
            task_batch_metrics: List[BinaryEvalMetrics] = []
            for _ in range(self.config.tasks_per_batch):
                train_task_sample = next(train_task_sample_iterator)
                task_loss, task_metrics = run_on_batches(
                    self,
                    batches=train_task_sample.batches,
                    batch_labels=train_task_sample.batch_labels,
                    train=True,
                    tasks_per_batch=self.config.tasks_per_batch,
                )
                task_batch_losses.append(task_loss)
                task_batch_metrics.append(task_metrics)

            # Now do a training step - run_on_batches will have accumulated gradients
            if self.config.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_value)
            self.optimizer.step()

            task_batch_mean_loss = np.mean(task_batch_losses)
            task_batch_avg_metrics = avg_metrics_list(task_batch_metrics)
            metric_logger.log_metrics(
                loss=task_batch_mean_loss,
                avg_prec=task_batch_avg_metrics["avg_precision"][0],
                kappa=task_batch_avg_metrics["kappa"][0],
                acc=task_batch_avg_metrics["acc"][0],
            )

            if step % self.config.validate_every_num_steps == 0:
                valid_task_sample_iterable = get_protonet_task_sample_iterable(
                    dataset=dataset,
                    data_fold=DataFold.VALIDATION,
                    num_samples=1,
                    max_num_graphs=self.config.batch_size,
                    support_size=self.config.support_set_size,
                    query_size=self.config.query_set_size,
                )

                valid_losses: List[float] = []
                valid_metrics: List[BinaryEvalMetrics] = []
                for task_sample in valid_task_sample_iterable:
                    task_loss, task_metrics = run_on_batches(
                        self,
                        task_sample.batches,
                        batch_labels=task_sample.batch_labels,
                        train=False,
                    )
                    valid_losses.append(task_loss)
                    valid_metrics.append(task_metrics)
                mean_valid_loss = np.mean(valid_losses)
                avg_valid_metrics = avg_metrics_list(valid_metrics)

                valid_avg_prec_mean, valid_avg_prec_std = avg_valid_metrics["avg_precision"]

                if aml_run:
                    # printing some measure of loss on all validation tasks.
                    aml_run.log(f"valid_mean_loss", mean_valid_loss)
                    aml_run.log(f"valid_mean_avg_prec", valid_avg_prec_mean)
                    aml_run.log(f"valid_mean_kappa", avg_valid_metrics["kappa"][0])

                logger.info(
                    f"Validated at train step [{step}/{self.config.num_train_steps}],"
                    f" Valid Loss: {mean_valid_loss:.7f},"
                    f" Valid Avg. Prec.: {valid_avg_prec_mean:.3f}+/-{valid_avg_prec_std:.3f}",
                )

                # save model if validation avg prec is the best so far
                if valid_avg_prec_mean > best_validation_avg_prec:
                    best_validation_avg_prec = valid_avg_prec_mean
                    model_path = os.path.join(out_dir, "best_validation.pt")
                    self.save_model(model_path)
                    logger.info(f"Updated {model_path} to new best model at train step {step}")

        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained.pt"))
