import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fs_mol.data.multitask import FSMolMultitaskBatch
from fs_mol.models.abstract_torch_fsmol_model import AbstractTorchFSMolModel, ModelStateType
from fs_mol.modules.film_gnn import FiLMGNN, FiLMGNNConfig
from fs_mol.modules.graph_readout import (
    GraphReadout,
    CombinedGraphReadout,
    MultiHeadWeightedGraphReadout,
    UnweightedGraphReadout,
)
from fs_mol.modules.mlp import MLP
from fs_mol.modules.task_specific_modules import (
    ProjectedTaskEmbeddingLayerProvider,
    TaskEmbeddingFiLMLayer,
    LearnedTaskEmbeddingLayerProvider,
)


logger = logging.getLogger(__name__)


@dataclass
class GNNMolFiLMConfig:
    num_tasks: int
    gnn_config: FiLMGNNConfig
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
    task_embedding_dim: Optional[int] = None


class GNNMolFiLMModel(AbstractTorchFSMolModel[FSMolMultitaskBatch]):
    def __init__(self, config: GNNMolFiLMConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")  # Default, we'll override where appropriate

        # Set some values / defaults:
        config.num_outputs = config.num_tasks

        if config.readout_dim is None:
            config.readout_dim = 4 * config.gnn_config.hidden_dim
        else:
            config.readout_dim

        # Initial (per-node) layers:
        self.init_node_proj = nn.Linear(
            config.node_feature_dim, config.gnn_config.hidden_dim, bias=False
        )

        # Set up FiLM layer providers:
        # (providers act to manufacture layers)
        if config.task_embedding_dim is None:
            self.task_embedding_provider = LearnedTaskEmbeddingLayerProvider(
                num_tasks=config.num_tasks
            )
        else:
            self.task_embedding_provider = ProjectedTaskEmbeddingLayerProvider(
                task_embedding_dim=config.task_embedding_dim, num_tasks=config.num_tasks
            )

        # Initial FiLM layer if called for:
        if config.use_init_film:
            self.init_node_film_layer: Optional[nn.Module] = TaskEmbeddingFiLMLayer(
                self.task_embedding_provider, config.gnn_config.hidden_dim
            )
        else:
            self.init_node_film_layer = None

        # Central GNN component
        self.gnn = FiLMGNN(
            self.config.gnn_config, task_embedding_provider=self.task_embedding_provider
        )

        # Set up for readout last or intermediate timesteps in GNN
        if config.readout_use_only_last_timestep:
            readout_node_dim = config.gnn_config.hidden_dim
        else:
            readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim

        # Readout layers:
        if config.readout_type.startswith("combined"):
            self.readout: GraphReadout = CombinedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_dim,
                num_heads=config.readout_num_heads,
                head_dim=config.readout_head_dim,
            )
        elif "weighted" in config.readout_type:
            self.readout = MultiHeadWeightedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_dim,
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

        # Option to concatenate the embedding of the tasks to the mol representations
        # prior to passing through the tail mlp.
        if config.use_tail_task_emb:
            self.tail_task_embedding: Optional[
                nn.Module
            ] = self.task_embedding_provider.get_task_embedding_layer(
                embedding_dim=config.readout_dim
            )
            # the tail MLP needs to accept twice the inputs now
            # (note, config.readout_dim is updated to 4 * gnn.hidden_dim if it was not set in defaults)
            self.mol_representation_dim = 2 * config.readout_dim
        else:
            self.tail_task_embedding = None
            self.mol_representation_dim = config.readout_dim

        self.tail_mlp = self.__create_tail_MLP(self.config.num_outputs)

    # TODO: fix the sizes of this layer
    def __create_tail_MLP(self, num_outputs: int) -> MLP:
        return MLP(
            input_dim=self.mol_representation_dim,
            hidden_layer_dims=[self.mol_representation_dim] * (self.config.num_tail_layers - 1),
            out_dim=num_outputs,
        )

    def to(self, device):
        self.device = device
        return super().to(device)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None):
        self.task_embedding_provider.reinitialize_task_parameters(new_num_tasks)

        self.tail_mlp = self.__create_tail_MLP(new_num_tasks or self.config.num_outputs)

    def forward(self, batch: FSMolMultitaskBatch):
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
                    f"Using FiLM/task embeddings requires the batch has a sample_to_task_id map!"
                )

        # Our inputs are numpy arrays, so we will need to move everything into torch.Tensor on
        # the right device.
        node_features = torch.tensor(batch.node_features, dtype=torch.float, device=self.device)
        node_to_graph = torch.tensor(batch.node_to_graph, dtype=torch.long, device=self.device)

        # set up inputs specifically necessary for FiLM layers
        if batch.sample_to_task_id is not None:
            sample_to_task = torch.tensor(
                batch.sample_to_task_id, dtype=torch.long, device=self.device
            )
            # make a single node to task map (TODO: change FiLM components to accept batch.node_to_graph map)
            node_to_task: Optional[torch.Tensor] = sample_to_task[batch.node_to_graph]
        else:
            # TODO: change this so that if FiLM layers are turned off you don't require the sample_to_task_id
            # below.
            raise NotImplementedError

        # ----- Update the per-task embeddings (if necessary)
        self.task_embedding_provider.update_task_embeddings()

        # ----- Initial (per-node) layers:
        initial_node_features = self.init_node_proj(node_features)

        # ----- Apply initial embedding FiLM layer if required:
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

        mol_representations = self.readout(readout_node_reprs, node_to_graph, batch.num_graphs)

        # ----- Tail phase
        # ----- If we are using the task embeddings, concatenate with mol representations
        if self.tail_task_embedding is not None:
            mol_representations = torch.cat(
                (mol_representations, self.tail_task_embedding(sample_to_task)), axis=1
            )

        # ----- Then apply the tail MLP
        mol_predictions = self.tail_mlp(mol_representations)

        # If we use masking, we throw away all predictions that have nothing to do with
        # our target task:
        mol_predictions = torch.gather(mol_predictions, 1, sample_to_task.unsqueeze(-1))

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
            or param_name.startswith("tail_mlp.")
        )

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
    ) -> "GNNMolFiLMModel":
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
    config: GNNMolFiLMConfig, device: Optional[torch.device] = None
) -> "GNNMolFiLMModel":
    model = GNNMolFiLMModel(config)
    if device is not None:
        model = model.to(device)
    return model
