import argparse
from dataclasses import dataclass
from typing import Optional
from typing_extensions import Literal

import torch
import torch.nn as nn

from fs_mol.data.fsmol_batcher import FSMolBatch
from fs_mol.data.fsmol_dataset import NUM_NODE_FEATURES
from fs_mol.modules.gnn import GNN, GNNConfig, add_gnn_model_arguments, make_gnn_config_from_args
from fs_mol.modules.graph_readout import (
    GraphReadoutConfig,
    add_graph_readout_arguments,
    make_graph_readout_config_from_args,
    make_readout_model,
)


@dataclass(frozen=True)
class GraphFeatureExtractorConfig:
    initial_node_feature_dim: int = NUM_NODE_FEATURES
    gnn_config: GNNConfig = GNNConfig()
    readout_config: GraphReadoutConfig = GraphReadoutConfig()
    output_norm: Literal["off", "layer", "batch"] = "off"


def add_graph_feature_extractor_arguments(parser: argparse.ArgumentParser):
    add_gnn_model_arguments(parser)
    add_graph_readout_arguments(parser)


def make_graph_feature_extractor_config_from_args(
    args: argparse.Namespace, initial_node_feature_dim: int = NUM_NODE_FEATURES
) -> GraphFeatureExtractorConfig:
    return GraphFeatureExtractorConfig(
        initial_node_feature_dim=initial_node_feature_dim,
        gnn_config=make_gnn_config_from_args(args),
        readout_config=make_graph_readout_config_from_args(args),
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

        if config.readout_config.use_all_states:
            readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim
        else:
            readout_node_dim = config.gnn_config.hidden_dim

        self.readout = make_readout_model(
            self.config.readout_config,
            readout_node_dim,
        )

        if self.config.output_norm == "off":
            self.final_norm_layer: Optional[torch.nn.Module] = None
        elif self.config.output_norm == "layer":
            self.final_norm_layer = nn.LayerNorm(
                normalized_shape=self.config.readout_config.output_dim
            )
        elif self.config.output_norm == "batch":
            self.final_norm_layer = nn.BatchNorm1d(
                num_features=self.config.readout_config.output_dim
            )

    def forward(self, input: FSMolBatch) -> torch.Tensor:
        # ----- Initial (per-node) layer:
        initial_node_features = self.init_node_proj(input.node_features)

        # ----- Message passing layers:
        all_node_representations = self.gnn(initial_node_features, input.adjacency_lists)

        # ----- Readout phase:
        if self.config.readout_config.use_all_states:
            readout_node_reprs = torch.cat(all_node_representations, dim=-1)
        else:
            readout_node_reprs = all_node_representations[-1]

        mol_representations = self.readout(
            node_embeddings=readout_node_reprs,
            node_to_graph_id=input.node_to_graph,
            num_graphs=input.num_graphs,
        )

        if self.final_norm_layer is not None:
            mol_representations = self.final_norm_layer(mol_representations)

        return mol_representations
