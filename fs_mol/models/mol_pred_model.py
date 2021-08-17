from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fs_mol.data.molfilm import FSMolMolFiLMBatch
from fs_mol.models.interface import AbstractTorchModel
from fs_mol.modules.thick_gnn import ThickGNN, ThickGNNConfig
from fs_mol.modules.graph_readout import (
    GraphReadout,
    CombinedGraphReadout,
    MultiHeadWeightedGraphReadout,
    UnweightedGraphReadout,
)
from fs_mol.modules.mlp import MLP
from fs_mol.modules.task_specific_models import (
    ProjectedTaskEmbeddingLayerProvider,
    TaskEmbeddingFiLMLayer,
    LearnedTaskEmbeddingLayerProvider,
)


logger = logging.getLogger(__name__)


@dataclass
class MolPredConfig:
    num_tasks: int
    gnn_config: ThickGNNConfig
    node_feature_dim: int = 32
    num_outputs: int = 1
    readout_type: str = "sum"
    readout_use_only_last_timestep: bool = False
    readout_dim: Optional[int] = None
    readout_num_heads: int = 12
    readout_head_dim: int = 64
    num_tail_layers: int = 1
    use_init_film: bool = False
    use_tail_task_emb: bool = False
    use_output_masking: bool = False
    task_embedding_dim: Optional[int] = None


class MolPredModel(AbstractTorchModel[FSMolMolFiLMBatch]):
    def __init__(self, config: MolPredConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")  # Default, we'll override where appropriate

        # Set some values / defaults:
        if config.use_output_masking:
            config.num_outputs = config.num_tasks
        else:
            config.num_outputs = 1

        if config.readout_dim is None:
            config.readout_dim = 4 * config.gnn_config.hidden_dim
        else:
            config.readout_dim

        if config.task_embedding_dim is None:
            self.task_embedding_provider = LearnedTaskEmbeddingLayerProvider(
                num_tasks=config.num_tasks
            )
        else:
            self.task_embedding_provider = ProjectedTaskEmbeddingLayerProvider(
                task_embedding_dim=config.task_embedding_dim, num_tasks=config.num_tasks
            )

        # Initial (per-node) layers:
        self.init_node_proj = nn.Linear(
            config.node_feature_dim, config.gnn_config.hidden_dim, bias=False
        )
        if config.use_init_film:
            self.init_node_film_layer: Optional[nn.Module] = TaskEmbeddingFiLMLayer(
                self.task_embedding_provider, config.gnn_config.hidden_dim
            )
        else:
            self.init_node_film_layer = None

        self.gnn = ThickGNN(self.config.gnn_config, self.task_embedding_provider)

        if config.readout_use_only_last_timestep:
            readout_node_dim = config.gnn_config.hidden_dim
        else:
            readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim

        # Readout layers:
        if config.readout_type.startswith("combined"):
            self.readout: GraphReadout = CombinedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_dim,
                task_embedding_provider=self.task_embedding_provider,
                num_heads=config.readout_num_heads,
                head_dim=config.readout_head_dim,
                use_task_specific_scores=config.readout_type.endswith("_task"),
            )
        elif "weighted" in config.readout_type:
            self.readout = MultiHeadWeightedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_dim,
                task_embedding_provider=self.task_embedding_provider,
                num_heads=config.readout_num_heads,
                head_dim=config.readout_head_dim,
                weighting_type=config.readout_type,
            )
        else:
            self.readout = UnweightedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_dim,
                pooling_type=config.readout_type,
            )

        # Final prediction layers:
        if config.use_tail_task_emb:
            # This will be concatenated to the output:
            self.tail_task_emb: Optional[
                nn.Module
            ] = self.task_embedding_provider.get_task_embedding_layer(
                embedding_dim=config.readout_dim
            )
            self.mol_representation_dim = 2 * config.readout_dim
        else:
            self.tail_task_emb = None
            self.mol_representation_dim = config.readout_dim

        self.tail_mlp = self.__create_tail_MLP()

    def __create_tail_MLP(self) -> MLP:
        return MLP(
            input_dim=self.mol_representation_dim,
            hidden_layer_dims=[self.mol_representation_dim] * (self.config.num_tail_layers - 1),
            out_dim=self.config.num_outputs,
        )

    def to(self, device):
        self.device = device
        return super().to(device)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None):
        self.task_embedding_provider.reinitialize_task_parameters(new_num_tasks)

        # TODO: Reinitialise the entire tail MLP - maybe we want to keep more bits here?
        self.tail_mlp = self.__create_tail_MLP()

    def forward(self, batch: FSMolMolFiLMBatch):
        """Predicts a float (unbounded), representing binding affinity for each input molecule.

        Args:
            batch: A batch of featurized molecules.

        Returns:
            List of per-molecule predictions of length `batch.num_graphs`.
        """
        # First check if we got what we needed:
        if batch.sample_to_task_id is None:
            if (
                self.config.use_init_film
                or self.config.gnn_config.use_msg_film
                or self.config.gnn_config.use_msg_att_film
                or self.config.use_tail_task_emb
            ):
                raise ValueError(
                    f"Using FiLM/task embeddings requires passing in the sample_to_task map!"
                )

        # Our inputs are numpy arrays, so we will need to move everything into torch.Tensor on
        # the right device.
        node_features = torch.tensor(batch.node_features, dtype=torch.float, device=self.device)
        node_to_graph = torch.tensor(batch.node_to_graph, dtype=torch.long, device=self.device)

        if batch.sample_to_task_id is not None:
            graph_to_task = torch.tensor(
                batch.sample_to_task_id, dtype=torch.long, device=self.device
            )

            node_to_task: Optional[torch.Tensor] = graph_to_task[node_to_graph]
        else:
            # the below always needs node_to_task, so won't work without it
            # might want to change some of the node_to_task to node_to_graph
            raise NotImplementedError

        # ----- Update the per-task embeddings (if necessary)
        self.task_embedding_provider.update_task_embeddings()

        # ----- Initial (per-node) layers:
        initial_node_features = self.init_node_proj(node_features)
        if self.init_node_film_layer is not None:
            initial_node_features = self.init_node_film_layer(initial_node_features, node_to_task)

        # ----- Message passing layers:
        all_node_representations = self.gnn(
            initial_node_features, batch.adjacency_lists, node_to_task
        )

        # ----- Readout phase:
        if self.config.readout_use_only_last_timestep:
            readout_node_reprs = all_node_representations[-1]
        else:
            readout_node_reprs = torch.cat(all_node_representations, dim=-1)

        mol_representations = self.readout(
            readout_node_reprs, node_to_graph, batch.num_graphs, node_to_task
        )

        # ----- Tail phase
        if self.tail_task_emb is not None:
            mol_representations = torch.cat(
                (mol_representations, self.tail_task_emb(graph_to_task)), axis=1
            )

        mol_predictions = self.tail_mlp(mol_representations)

        # If we use masking, we throw away all predictions that have nothing to do with
        # our target task:
        if self.config.use_output_masking:
            mol_predictions = torch.gather(mol_predictions, 1, graph_to_task.unsqueeze(-1))

        return mol_predictions

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def is_param_task_specific(self, param_name: str) -> bool:
        return (
            param_name.endswith(".task_emb.weight")
            or param_name.endswith("._per_task_matrices")
            or (self.config.use_output_masking and param_name.startswith("tail_mlp."))
        )

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
    ) -> MolPredModel:
        """Build the model architecture based on a saved checkpoint."""
        model, _ = load_model(
            model_file,
            config_overrides=config_overrides,
            load_model_weights=False,
            quiet=quiet,
            device=device,
        )

        return model


def create_model(config: MolPredConfig, device: Optional[torch.device] = None) -> MolPredModel:
    model = MolPredModel(config)
    if device is not None:
        model = model.to(device)
    return model


def load_model(
    model_file: str,
    model: Optional[MolPredModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config_overrides: Dict[str, Any] = {},
    load_model_weights: bool = True,
    reinit_task_specific_weights: bool = True,
    quiet: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[MolPredModel, Optional[torch.optim.Optimizer]]:
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
