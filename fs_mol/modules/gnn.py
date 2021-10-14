import argparse
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_log_softmax, scatter_mean, scatter_max

from fs_mol.data.fsmol_dataset import NUM_EDGE_TYPES
from fs_mol.modules.mlp import MLP


SMALL_NUMBER = 1e-7


@dataclass
class GNNConfig:
    type: str = "PNA"
    num_edge_types: int = NUM_EDGE_TYPES
    hidden_dim: int = 128
    num_heads: int = 4
    per_head_dim: int = 32
    intermediate_dim: int = 512
    message_function_depth: int = 1
    num_layers: int = 8
    dropout_rate: float = 0.0
    use_rezero_scaling: bool = True
    make_edges_bidirectional: bool = True


def add_gnn_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="PNA",
        choices=["MultiHeadAttention", "MultiAggr", "PNA", "Plain"],
        help="Type of GNN architecture to use.",
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
        "--num_gnn_layers", type=int, default=10, help="Number of GNN layers to use."
    )


def make_gnn_config_from_args(args: argparse.Namespace) -> GNNConfig:
    return GNNConfig(
        type=args.gnn_type,
        hidden_dim=args.node_embed_dim,
        num_edge_types=NUM_EDGE_TYPES,
        num_heads=args.num_heads,
        per_head_dim=args.per_head_dim,
        intermediate_dim=args.intermediate_dim,
        message_function_depth=args.message_function_depth,
        num_layers=args.num_gnn_layers,
    )


class BOOMLayer(nn.Module):
    """Shallow MLP with large intermediate layer.

    Named in Sect. 3 of https://arxiv.org/pdf/1911.11423.pdf:
    'Why Boom? We take a vector from small (1024) to big (4096) to small (1024). It’s really not
     that hard to visualize - use your hands if you need to whilst shouting "boooOOOOmmm".'
    """

    def __init__(self, inout_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(inout_dim, intermediate_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(intermediate_dim, inout_dim)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class RelationalMP(nn.Module):
    """Relational message passing, using different message functions for each relation/edge
    type."""

    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int,
        num_edge_types: int,
        message_function_depth: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.num_edge_types = num_edge_types

        self.message_fns = nn.ModuleList()

        for _ in range(num_edge_types):
            self.message_fns.append(
                MLP(
                    input_dim=2 * hidden_dim,
                    out_dim=msg_dim,
                    hidden_layer_dims=[2 * hidden_dim] * (message_function_depth - 1),
                )
            )

    @property
    def message_size(self) -> int:
        return self.msg_dim

    def forward(
        self,
        x: torch.Tensor,
        adj_lists: List[torch.Tensor],
    ):
        all_msg_list: List[torch.Tensor] = []  # all messages exchanged between nodes
        all_tgts_list: List[torch.Tensor] = []  # [E] - list of targets for all messages

        for edge_type, adj_list in enumerate(adj_lists):
            srcs = adj_list[:, 0]
            tgts = adj_list[:, 1]

            messages = self.message_fns[edge_type](torch.cat((x[srcs], x[tgts]), dim=1))
            messages = nn.functional.relu(messages)

            all_msg_list.append(messages)
            all_tgts_list.append(tgts)

        all_messages = torch.cat(all_msg_list, dim=0)  # [E, msg_dim]
        all_targets = torch.cat(all_tgts_list, dim=0)  # [E]

        return self._aggregate_messages(all_messages, all_targets, num_nodes=x.shape[0])

    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        targets: torch.Tensor,
        num_nodes: Union[torch.Tensor, int],
    ):
        aggregated_messages = scatter_sum(
            src=messages,
            index=targets,
            dim=0,
            dim_size=num_nodes,
        )  # [V, msg_dim]

        return aggregated_messages


class RelationalMultiAggrMP(RelationalMP):
    """Relational message passing, but using three different aggregation strategies (sum, mean, stdev, max).
    Optionally, also includes different scalers (as in https://arxiv.org/abs/2004.05718)."""

    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int,
        num_edge_types: int,
        message_function_depth: int = 1,
        use_pna_scalers: bool = False,
    ):
        # We create 3 messages per edge (or one msg of 3 times the size), and aggregate them differently:
        super().__init__(
            hidden_dim,
            3 * msg_dim,
            num_edge_types,
            message_function_depth,
        )
        self.partial_msg_dim = msg_dim
        self.use_pna_scalers = use_pna_scalers

    @property
    def message_size(self) -> int:
        message_size = 4 * self.partial_msg_dim  # Aggregated as sum, mean, stdev, max
        if self.use_pna_scalers:
            message_size = 3 * message_size  # Each scaled by identity, amplifier, attenuator
        return message_size

    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        targets: torch.Tensor,
        num_nodes: Union[torch.Tensor, int],
    ):
        sum_aggregated_messages = scatter_sum(
            src=messages[:, : self.partial_msg_dim],
            index=targets,
            dim=0,
            dim_size=num_nodes,
        )  # [V, partial_msg_dim]
        mean_messages = messages[:, self.partial_msg_dim : 2 * self.partial_msg_dim]
        mean_aggregated_messages = scatter_mean(
            src=mean_messages,
            index=targets,
            dim=0,
            dim_size=num_nodes,
        )  # [V, partial_msg_dim]
        per_node_message_stdev = (
            nn.functional.relu(mean_messages.pow(2) - mean_aggregated_messages[targets].pow(2))
            + SMALL_NUMBER
        )
        std_aggregated_messages = torch.sqrt(
            scatter_sum(src=per_node_message_stdev, index=targets, dim=0, dim_size=num_nodes)
        )  # [V, partial_msg_dim]
        max_aggregated_messages = scatter_max(
            src=messages[:, 2 * self.partial_msg_dim : 3 * self.partial_msg_dim],
            index=targets,
            dim=0,
            dim_size=num_nodes,
        )[
            0
        ]  # [V, partial_msg_dim]  # [0] needed to strip out the indices of the max.

        messages = torch.cat(
            (
                sum_aggregated_messages,
                mean_aggregated_messages,
                std_aggregated_messages,
                max_aggregated_messages,
            ),
            dim=1,
        )

        if self.use_pna_scalers:
            # First, compute degrees of nodes:
            node_degrees = scatter_sum(
                torch.ones_like(targets),
                index=targets,
                dim_size=num_nodes,
            )
            delta = 1.1515  # Computed over LSC dataset

            log_node_degrees = torch.log(node_degrees.float() + 1).unsqueeze(-1)

            amplification_scale_factor = log_node_degrees / delta
            attenuation_scale_factor = delta / (log_node_degrees + SMALL_NUMBER)

            messages = torch.cat(
                (
                    messages,
                    amplification_scale_factor * messages,
                    attenuation_scale_factor * messages,
                ),
                dim=1,
            )

        return messages


class RelationalMultiHeadAttentionMP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        per_head_dim: int,
        num_edge_types: int,
        message_function_depth: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.per_head_dim = per_head_dim
        self.num_edge_types = num_edge_types

        self.message_fns = nn.ModuleList()
        self.target_node_query_projs = nn.ModuleList()
        self.neighbour_node_key_projs = nn.ModuleList()
        self.query_scaling_factor = self.per_head_dim ** -0.5

        for _ in range(num_edge_types):
            self.target_node_query_projs.append(
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=self.num_heads * self.per_head_dim,
                    bias=False,
                )
            )
            self.neighbour_node_key_projs.append(
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=self.num_heads * self.per_head_dim,
                    bias=False,
                )
            )
            self.message_fns.append(
                MLP(
                    input_dim=2 * hidden_dim,
                    out_dim=per_head_dim * num_heads,
                    hidden_layer_dims=[2 * hidden_dim] * (message_function_depth - 1),
                )
            )

    @property
    def message_size(self) -> int:
        return self.num_heads * self.per_head_dim

    def forward(
        self,
        x: torch.Tensor,
        adj_lists: List[torch.Tensor],
    ):
        all_msg_list: List[torch.Tensor] = []  # all messages exchanged between nodes
        all_scores_list: List[torch.Tensor] = []  # attention scores for all messages
        all_tgts_list: List[torch.Tensor] = []  # [E] - list of targets for all messages

        for edge_type, adj_list in enumerate(adj_lists):
            srcs = adj_list[:, 0]
            tgts = adj_list[:, 1]

            # Compute messages
            src_node_reprs = x[srcs]
            tgt_node_reprs = x[tgts]
            messages = self.message_fns[edge_type](
                torch.cat((src_node_reprs, tgt_node_reprs), dim=1)
            )
            messages = nn.functional.relu(messages).view(-1, self.num_heads, self.per_head_dim)

            # Compute weights by doing a query projection of node targets, a key project of node keys,
            # and then computing their alignment:
            edge_queries = self.target_node_query_projs[edge_type](
                tgt_node_reprs
            )  # [E, num_heads * head_dim]
            edge_queries = self.query_scaling_factor * edge_queries
            edge_keys = self.neighbour_node_key_projs[edge_type](
                src_node_reprs
            )  # [E, num_heads * head_dim]

            # We sometimes don't have any edges of a type in the batch, and einsum doesn't
            # like that:
            if edge_queries.shape[0] > 0:
                edge_scores = torch.einsum(
                    "ehd,ehd->eh",
                    edge_queries.view(-1, self.num_heads, self.per_head_dim),
                    edge_keys.view(-1, self.num_heads, self.per_head_dim),
                )
            else:
                edge_scores = torch.zeros(
                    size=(0, self.num_heads),
                    dtype=edge_queries.dtype,
                    device=edge_queries.device,
                )

            all_msg_list.append(messages)  # [E_i, num_heads, head_dim]
            all_scores_list.append(edge_scores)  # [E_i, num_heads]
            all_tgts_list.append(tgts)

        all_messages = torch.cat(all_msg_list, dim=0)  # [E, num_heads, head_dim]
        all_scores = torch.cat(all_scores_list, dim=0)  # [E, num_heads]
        all_targets = torch.cat(all_tgts_list, dim=0)  # [E]

        # Compute attention scores per head:
        all_probs = torch.exp(
            scatter_log_softmax(
                src=all_scores,
                index=all_targets,
                dim=0,
            )
        )  # [E, num_heads]

        all_weighted_messages = all_probs.unsqueeze(-1) * all_messages  # [E, num_heads, head_dim]

        aggregated_messages = scatter_sum(
            src=all_weighted_messages,
            index=all_targets,
            dim=0,
            dim_size=x.shape[0],
        )  # [V, num_heads, head_dim]

        return aggregated_messages.view(-1, self.num_heads * self.per_head_dim)


class GNNBlock(nn.Module):
    """Block in a GNN, following a Transformer-like residual structure, using the "Pre-Norm" style
    and ReZero weighting using \alpha:
      v' = v + \alpha * Dropout(NeighbourHoodAtt(LN(v))))
      v = v' + \alpha * Linear(Dropout(Act(Linear(LN(v'))))))

    Pre-Norm reference: https://arxiv.org/pdf/2002.04745v1.pdf
    ReZero reference: https://arxiv.org/pdf/2003.04887v1.pdf
    ReZero' (with \alpha a vector instead of scalar): https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        if config.use_rezero_scaling:
            self.alpha = nn.Parameter(torch.full(size=(1,), fill_value=SMALL_NUMBER))

        self.mp_layers: nn.ModuleList = nn.ModuleList()
        if config.type.lower() == "MultiHeadAttention".lower():
            self.mp_layer_in_dim = config.hidden_dim
            self.mp_layers.append(
                RelationalMultiHeadAttentionMP(
                    hidden_dim=self.mp_layer_in_dim,
                    num_heads=config.num_heads,
                    per_head_dim=config.per_head_dim,
                    num_edge_types=config.num_edge_types,
                    message_function_depth=config.message_function_depth,
                )
            )
        else:
            # We hijack the "num_heads" parameter for the "towers" trick introduced by Gilmer et al.,
            # in which several message passing mechanisms work in parallel on subsets. We slice the
            # overall node representations into a part for each of these:
            self.mp_layer_in_dim = self.config.hidden_dim // self.config.num_heads
            assert (
                self.config.hidden_dim % self.config.num_heads == 0
            ), "Number of heads needs to divide GNN hidden dim."
            for _ in range(config.num_heads):
                if (
                    config.type.lower() == "MultiAggr".lower()
                    or config.type.lower() == "PNA".lower()
                ):
                    self.mp_layers.append(
                        RelationalMultiAggrMP(
                            hidden_dim=self.mp_layer_in_dim,
                            msg_dim=config.per_head_dim,
                            num_edge_types=config.num_edge_types,
                            message_function_depth=config.message_function_depth,
                            use_pna_scalers=config.type.lower() == "PNA".lower(),
                        )
                    )
                elif config.type.lower() == "Plain".lower():
                    self.mp_layers.append(
                        RelationalMP(
                            hidden_dim=self.mp_layer_in_dim,
                            msg_dim=config.per_head_dim,
                            num_edge_types=config.num_edge_types,
                            message_function_depth=config.message_function_depth,
                        )
                    )
                else:
                    raise ValueError(f"Unknown GNN type {config.type}.")

        total_msg_dim = sum(mp_layer.message_size for mp_layer in self.mp_layers)
        self.msg_out_projection = nn.Linear(
            in_features=total_msg_dim, out_features=config.hidden_dim
        )

        self.mp_norm_layer = nn.LayerNorm(normalized_shape=config.hidden_dim)

        if config.intermediate_dim > 0:
            self.boom_layer: Optional[BOOMLayer] = BOOMLayer(
                inout_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                dropout=config.dropout_rate,
            )
            self.boom_norm_layer: Optional[nn.Module] = nn.LayerNorm(
                normalized_shape=config.hidden_dim
            )
        else:
            self.boom_layer = None
            self.boom_norm_layer = None

        # We will use this one dropout layer everywhere, as it's stateless:
        self.dropout_layer = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        node_representations,
        adj_lists,
    ):
        """
        Args:
            node_representations: float tensor of shape (num_nodes, config.hidden_dim)
            adj_lists: List of (num_edges, 2) tensors (one per edge-type)
        Returns:
            node_representations: float (num_graphs, config.hidden_dim) tensor
        """
        aggregated_messages = []
        for i, mp_layer in enumerate(self.mp_layers):
            sliced_node_representations = node_representations[
                :, i * self.mp_layer_in_dim : (i + 1) * self.mp_layer_in_dim
            ]
            aggregated_messages.append(
                mp_layer(
                    x=sliced_node_representations,
                    adj_lists=adj_lists,
                )
            )

        new_representations = self.msg_out_projection(torch.cat(aggregated_messages, dim=-1))
        new_representations = self.dropout_layer(new_representations)
        if self.config.use_rezero_scaling:
            new_representations = self.alpha * new_representations
        node_representations = node_representations + new_representations

        if self.boom_layer is not None and self.boom_norm_layer is not None:
            boomed_representations = self.dropout_layer(
                self.boom_layer(self.boom_norm_layer(node_representations))
            )
            if self.config.use_rezero_scaling:
                boomed_representations = self.alpha * boomed_representations
            node_representations = node_representations + boomed_representations

        return node_representations


class GNN(nn.Module):
    def __init__(
        self,
        config: GNNConfig,
    ):
        super().__init__()
        self.config = config

        self.gnn_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.gnn_blocks.append(GNNBlock(config))

    def forward(self, node_features, adj_lists) -> List[torch.Tensor]:
        """
        args:
            node_representations: float tensor of shape (num_nodes, config.hidden_dim)
            adj_lists: List of (num_edges, 2) tensors (one per edge-type)
        output:
            all_node_representations: list of float32 (num_graphs, config.hidden_dim) tensors,
                one for the result of each timestep of the GNN (and the initial one)
        """
        # We may need to introduce additional edges to make everything bidirectional:
        if self.config.make_edges_bidirectional:
            adj_lists = [
                torch.cat((adj_list, torch.flip(adj_list, dims=(1,))), dim=0)
                for adj_list in adj_lists
            ]

        # Actually do message passing:
        cur_node_representations = node_features
        all_node_representations: List[torch.Tensor] = [cur_node_representations]
        for gnn_block in self.gnn_blocks:
            cur_node_representations = gnn_block(
                node_representations=cur_node_representations,
                adj_lists=adj_lists,
            )
            all_node_representations.append(cur_node_representations)

        return all_node_representations
