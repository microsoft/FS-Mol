import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from fs_mol.data.multitask import FSMolMultitaskBatch
from fs_mol.models.abstract_torch_fsmol_model import (
    AbstractTorchFSMolModel,
    ModelStateType,
    TorchFSMolModelLoss,
    TorchFSMolModelOutput,
)
from fs_mol.modules.mlp import MLP
from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)


logger = logging.getLogger(__name__)


@dataclass
class GNNMultitaskConfig:
    graph_feature_extractor_config: GraphFeatureExtractorConfig
    num_tasks: int
    num_tail_layers: int = 1


def add_gnn_multitask_model_arguments(parser: argparse.ArgumentParser):
    add_graph_feature_extractor_arguments(parser)
    parser.add_argument("--num_tail_layers", type=int, default=3)


def make_gnn_multitask_model_from_args(
    num_tasks: int, args: argparse.Namespace, device: Optional[torch.device] = None
):
    model_config = GNNMultitaskConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        num_tasks=num_tasks,
        num_tail_layers=args.num_tail_layers,
    )
    return create_model(model_config, device=device)


class GNNMultitaskModel(
    AbstractTorchFSMolModel[FSMolMultitaskBatch, TorchFSMolModelOutput, TorchFSMolModelLoss]
):
    def __init__(self, config: GNNMultitaskConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")  # Default, we'll override where appropriate

        self.graph_feature_extractor = GraphFeatureExtractor(config.graph_feature_extractor_config)

        self.tail_mlp = self.__create_tail_MLP(self.config.num_tasks)

    def __create_tail_MLP(self, num_tasks: int) -> MLP:
        return MLP(
            input_dim=self.config.graph_feature_extractor_config.readout_config.output_dim,
            hidden_layer_dims=[self.config.graph_feature_extractor_config.readout_config.output_dim]
            * (self.config.num_tail_layers - 1),
            out_dim=num_tasks,
        )

    def to(self, device):
        self.device = device
        return super().to(device)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None):
        self.tail_mlp = self.__create_tail_MLP(new_num_tasks or self.config.num_tasks)

    def forward(self, batch: FSMolMultitaskBatch) -> TorchFSMolModelOutput:
        """Predicts a float (unbounded), representing binding affinity for each input molecule.

        Args:
            batch: A batch of featurized molecules.

        Returns:
            List of per-molecule predictions of length `batch.num_graphs`.
        """
        mol_representations = self.graph_feature_extractor(batch)
        mol_predictions = self.tail_mlp(mol_representations)
        mol_predictions = torch.gather(mol_predictions, 1, batch.sample_to_task_id.unsqueeze(-1))
        return TorchFSMolModelOutput(molecule_binary_label=mol_predictions)

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def is_param_task_specific(self, param_name: str) -> bool:
        return param_name.startswith("tail_mlp.")

    def load_model_state(
        self,
        model_state: ModelStateType,
        load_task_specific_weights: bool,
        quiet: bool = False,
    ) -> None:
        # Filter down to parts of the model we want to re-use:
        params_to_load = {
            param_name: param_value
            for param_name, param_value in model_state["model_state_dict"].items()
            if (
                not self.is_param_task_specific(param_name)
                or (self.is_param_task_specific(param_name) and load_task_specific_weights)
            )
        }

        missing, unexpected = self.load_state_dict(params_to_load, strict=False)
        missing = [
            param_name for param_name in missing if not self.is_param_task_specific(param_name)
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

        if not load_task_specific_weights:
            if not quiet:
                logger.info(f"  Re-initialising task-specific parameters.")
            self.reinitialize_task_parameters()

        # This should be a no-op, but may be required after a re-init:
        self.to(self.device)

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> "GNNMultitaskModel":
        """Load weights from file, either into an existing model, or a fresh model
        created following the loaded configuration."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        if not quiet:
            logger.info(f" Loading model configuration from {model_file}.")
        for key, val in config_overrides.items():
            if not quiet:
                logger.info(
                    f"  I: Overriding model config parameter {key} from {getattr(config, key)} to {val}!"
                )
            setattr(config, key, val)

        model = create_model(config, device)

        return model


def create_model(
    config: GNNMultitaskConfig, device: Optional[torch.device] = None
) -> GNNMultitaskModel:
    model = GNNMultitaskModel(config)
    if device is not None:
        model = model.to(device)
    return model
