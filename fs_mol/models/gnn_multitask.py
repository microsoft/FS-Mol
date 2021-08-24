from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from fs_mol.data.multitask import FSMolMultitaskBatch
from fs_mol.models.abstract_torch_fsmol_model import AbstractTorchFSMolModel
from fs_mol.modules.mlp import MLP
from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class GNNMultitaskConfig:
    graph_feature_extractor_config: GraphFeatureExtractorConfig
    num_tasks: int
    num_tail_layers: int = 1


class GNNMultitaskModel(AbstractTorchFSMolModel[FSMolMultitaskBatch]):
    def __init__(self, config: GNNMultitaskConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")  # Default, we'll override where appropriate

        self.graph_feature_extractor = GraphFeatureExtractor(config.graph_feature_extractor_config)

        self.tail_mlp = self.__create_tail_MLP(self.config.num_tasks)

    def __create_tail_MLP(self, num_tasks: int) -> MLP:
        return MLP(
            input_dim=self.config.graph_feature_extractor_config.readout_output_dim,
            hidden_layer_dims=[self.config.graph_feature_extractor_config.readout_output_dim]
            * (self.config.num_tail_layers - 1),
            out_dim=num_tasks,
        )

    def to(self, device):
        self.device = device
        return super().to(device)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None):
        self.tail_mlp = self.__create_tail_MLP(new_num_tasks or self.config.num_tasks)

    def forward(self, batch: FSMolMultitaskBatch):
        """Predicts a float (unbounded), representing binding affinity for each input molecule.

        Args:
            batch: A batch of featurized molecules.

        Returns:
            List of per-molecule predictions of length `batch.num_graphs`.
        """
        mol_representations = self.graph_feature_extractor(batch)
        mol_predictions = self.tail_mlp(mol_representations)
        mol_predictions = torch.gather(mol_predictions, 1, batch.sample_to_task_id.unsqueeze(-1))
        return mol_predictions

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def is_param_task_specific(self, param_name: str) -> bool:
        return param_name.startswith("tail_mlp.")

    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load model weights from a saved checkpoint."""
        load_model(
            path,
            model=self,
            reinit_task_specific_weights=not load_task_specific_weights,
            quiet=quiet,
            device=device,
        )

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> GNNMultitaskModel:
        """Build the model architecture based on a saved checkpoint."""
        model, _ = load_model(
            model_file,
            config_overrides=config_overrides,
            load_model_weights=False,
            quiet=quiet,
            device=device,
        )

        return model


def create_model(
    config: GNNMultitaskConfig, device: Optional[torch.device] = None
) -> GNNMultitaskModel:
    model = GNNMultitaskModel(config)
    if device is not None:
        model = model.to(device)
    return model


def load_model(
    model_file: str,
    model: Optional[GNNMultitaskModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config_overrides: Dict[str, Any] = {},
    load_model_weights: bool = True,
    reinit_task_specific_weights: bool = True,
    quiet: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[GNNMultitaskModel, Optional[torch.optim.Optimizer]]:
    """Load weights from file, either into an existing model, or a fresh model
    created following the loaded configuration."""
    checkpoint = torch.load(model_file, map_location=device)
    config = checkpoint["model_config"]

    if not quiet:
        logger.info(f" Loading model configuration from {model_file}.")
    if model is None:
        for key, val in config_overrides.items():
            if not quiet:
                logger.info(
                    f"  I: Overriding model config parameter {key} from {getattr(config, key)} to {val}!"
                )
            setattr(config, key, val)
        model = create_model(config, device)

    if load_model_weights:
        # Filter down to parts of the model we want to re-use:
        params_to_load = {
            param_name: param_value
            for param_name, param_value in checkpoint["model_state_dict"].items()
            if (
                not model.is_param_task_specific(param_name)
                or (model.is_param_task_specific(param_name) and not reinit_task_specific_weights)
            )
        }

        missing, unexpected = model.load_state_dict(params_to_load, strict=False)
        missing = [
            param_name for param_name in missing if not model.is_param_task_specific(param_name)
        ]

        if len(missing) > 0:
            logger.error("  E: Had no values for the following parameters:")
            for m in missing:
                logger.error(f"    {m}")
            raise ValueError(f"Trying to load model, but was missing parameters!")
        if len(unexpected) > 0:
            logger.error("  E: Found unexpected parameters:")
            for u in unexpected:
                logger.error(f"    {u}")
            raise ValueError(f"Trying to load model, but found unexpected parameters!")

        if reinit_task_specific_weights:
            if not quiet:
                logger.info(f"  Re-initialising task-specific parameters.")
            model.reinitialize_task_parameters()

        model = model.to(model.device)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer
