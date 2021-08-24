import argparse
from dataclasses import dataclass
from fs_mol.utils.cli_utils import str2bool
from typing import Optional
from typing_extensions import Literal

import torch
import torch.nn as nn

from fs_mol.data.fsmol_batcher import FSMolBatch
from fs_mol.data.fsmol_dataset import NUM_NODE_FEATURES
from fs_mol.modules.gnn import GNN, GNNConfig, add_gnn_model_arguments, make_gnn_config_from_args
from fs_mol.modules.graph_readout import (
    CombinedGraphReadout,
    GraphReadout,
    MultiHeadWeightedGraphReadout,
    UnweightedGraphReadout,
)


@dataclass(frozen=True)
class GraphFeatureExtractorConfig:
    gnn_config: GNNConfig = GNNConfig()
    initial_node_feature_dim: int = NUM_NODE_FEATURES
    readout_type: Literal[
        "sum", "min", "max", "mean", "weighted_sum", "weighted_mean", "combined"
    ] = "combined"
    readout_use_all_states: bool = True
    readout_dim: Optional[int] = None
    readout_num_heads: int = 12
    readout_head_dim: int = 64
    readout_output_dim: int = 512


def add_graph_feature_extractor_arguments(parser: argparse.ArgumentParser):
    add_gnn_model_arguments(parser)

    parser.add_argument(
        "--readout_type",
        type=str,
        default="combined",
        choices=["sum", "min", "max", "mean", "weighted_sum", "weighted_mean", "combined"],
        help="Readout used to summarise atoms into a molecule",
    )
    parser.add_argument(
        "--readout_use_all_states",
        type=str2bool,
        default=True,
        help="Indicates if all intermediate GNN activations or only the final ones should be used when computing a graph-level representation.",
    )
    parser.add_argument(
        "--readout_num_heads",
        type=int,
        default=12,
        help="Number of heads used in the readout heads.",
    )
    parser.add_argument(
        "--readout_head_dim",
        type=int,
        default=64,
        help="Dimensionality of each readout head.",
    )
    parser.add_argument(
        "--readout_output_dim",
        type=int,
        default=256,
        help="Dimensionality of the readout result.",
    )


def make_graph_feature_extractor_config_from_args(args: argparse.Namespace, initial_node_feature_dim: int = NUM_NODE_FEATURES) -> GraphFeatureExtractorConfig:
    return GraphFeatureExtractorConfig(
        gnn_config=make_gnn_config_from_args(args),
        initial_node_feature_dim=initial_node_feature_dim,
        readout_type=args.readout_type,
        readout_use_all_states=args.readout_use_all_states,
        readout_num_heads=args.readout_num_heads,
        readout_head_dim=args.readout_head_dim,
        readout_output_dim=args.readout_output_dim,
    )


class GraphFeatureExtractor(nn.Module):
    def __init__(self, config: GraphFeatureExtractorConfig):
        super().__init__()
        self.config = config

        # Initial (per-node) layers:
        self.init_node_proj = nn.Linear(
            config.initial_node_feature_dim, config.gnn_config.hidden_dim, bias=False
        )

        self.gnn = GNN(self.config.gnn_config)

        if config.readout_use_all_states:
            readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim
        else:
            readout_node_dim = config.gnn_config.hidden_dim

        # Readout layers:
        if config.readout_type.startswith("combined"):
            self.readout: GraphReadout = CombinedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_output_dim,
                num_heads=config.readout_num_heads,
                head_dim=config.readout_head_dim,
            )
        elif "weighted" in config.readout_type:
            self.readout = MultiHeadWeightedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_output_dim,
                num_heads=config.readout_num_heads,
                head_dim=config.readout_head_dim,
                weighting_type=config.readout_type,
            )
        else:
            self.readout = UnweightedGraphReadout(
                node_dim=readout_node_dim,
                out_dim=config.readout_output_dim,
                pooling_type=config.readout_type,
            )

        readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim
        self.init_node_proj = nn.Linear(
            in_features=NUM_NODE_FEATURES,  # This is fixed in our dataset
            out_features=config.gnn_config.hidden_dim,
            bias=False,
        )

        self.final_norm_layer = nn.BatchNorm1d(num_features=config.readout_output_dim)

    def forward(self, input: FSMolBatch) -> torch.Tensor:
        # ----- Initial (per-node) layer:
        initial_node_features = self.init_node_proj(input.node_features)

        # ----- Message passing layers:
        all_node_representations = self.gnn(initial_node_features, input.adjacency_lists)

        # ----- Readout phase:
        if self.config.readout_use_all_states:
            readout_node_reprs = torch.cat(all_node_representations, dim=-1)
        else:
            readout_node_reprs = all_node_representations[-1]

        mol_representations = self.readout(
            node_embeddings=readout_node_reprs,
            node_to_graph_id=input.node_to_graph,
            num_graphs=input.num_graphs,
        )

        return self.final_norm_layer(mol_representations)
