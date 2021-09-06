from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_log_softmax, scatter_mean, scatter_max

from fs_mol.modules.mlp import MLP
from fs_mol.modules.gnn import (
    BOOMLayer,
)

from fs_mol.modules.task_specific_modules import (
    TaskEmbeddingLayerProvider,
    TaskEmbeddingFiLMLayer,
)


SMALL_NUMBER = 1e-7


class FiLMRelationalMP(nn.Module):
    """Relational message passing, using different message functions for each relation/edge
    type."""

    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int,
        num_edge_types: int,
        message_function_depth: int = 1,
        task_embedding_provider: Optional[TaskEmbeddingLayerProvider] = None,
        use_msg_film: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.num_edge_types = num_edge_types

        self.message_fns = nn.ModuleList()
        self.message_film_layers = nn.ModuleList()

        self.use_msg_film = use_msg_film

        for _ in range(num_edge_types):
            self.message_fns.append(
                MLP(
                    input_dim=2 * hidden_dim,
                    out_dim=msg_dim,
                    hidden_layer_dims=[2 * hidden_dim] * (message_function_depth - 1),
                )
            )
            if self.use_msg_film:
                assert (
                    task_embedding_provider is not None
                ), "A task embedding provider must be available for using message-passing FiLM layers"
                self.message_film_layers.append(
                    TaskEmbeddingFiLMLayer(task_embedding_provider, msg_dim)
                )

    @property
    def message_size(self) -> int:
        return self.msg_dim

    def forward(
        self,
        x: torch.Tensor,
        adj_lists: List[torch.Tensor],
        node_to_task: Optional[torch.Tensor] = None,
        edge_to_task: Optional[List[torch.Tensor]] = None,
    ):
        all_msg_list: List[torch.Tensor] = []  # all messages exchanged between nodes
        all_tgts_list: List[torch.Tensor] = []  # [E] - list of targets for all messages

        for edge_type, adj_list in enumerate(adj_lists):
            srcs = adj_list[:, 0]
            tgts = adj_list[:, 1]

            messages = self.message_fns[edge_type](torch.cat((x[srcs], x[tgts]), dim=1))
            if self.use_msg_film and edge_to_task is not None:
                # edge_to_task[edge_type] here is the long tensor of the relevant task ids
                messages = self.message_film_layers[edge_type](messages, edge_to_task[edge_type])
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


class FiLMRelationalMultiAggrMP(FiLMRelationalMP):
    """Relational message passing, but using three different aggregation strategies (sum, mean, stdev, max).
    Optionally, also includes different scalers (as in https://arxiv.org/abs/2004.05718)."""

    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int,
        num_edge_types: int,
        message_function_depth: int = 1,
        use_pna_scalers: bool = False,
        task_embedding_provider: Optional[TaskEmbeddingLayerProvider] = None,
        use_msg_film: bool = False,
    ):
        # We create 3 messages per edge (or one msg of 3 times the size), and aggregate them differently:
        super().__init__(
            hidden_dim,
            3 * msg_dim,
            num_edge_types,
            message_function_depth,
            task_embedding_provider=task_embedding_provider,
            use_msg_film=use_msg_film,
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


class FiLMRelationalMultiHeadAttentionMP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        per_head_dim: int,
        num_edge_types: int,
        message_function_depth: int = 1,
        task_embedding_provider: Optional[TaskEmbeddingLayerProvider] = None,
        use_msg_film: bool = False,
        use_msg_att_film: bool = False,
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

        self.message_film_layers = nn.ModuleList()

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
            if use_msg_film:
                assert (
                    task_embedding_provider is not None
                ), "A task embedding provider must be available for using message-passing FiLM layers"
                self.message_film_layers.append(
                    TaskEmbeddingFiLMLayer(task_embedding_provider, per_head_dim * num_heads)
                )

        if use_msg_att_film:
            assert (
                task_embedding_provider is not None
            ), "A task embedding provider must be available for using FiLM on query layers."
            self.query_node_films: Optional[nn.ModuleList] = nn.ModuleList()
            for _ in range(num_edge_types):
                self.query_node_films.append(
                    TaskEmbeddingFiLMLayer(
                        task_embedding_provider, self.num_heads * self.per_head_dim
                    )
                )
        else:
            self.query_node_films = None

    @property
    def message_size(self) -> int:
        return self.num_heads * self.per_head_dim

    def forward(
        self,
        x: torch.Tensor,
        adj_lists: List[torch.Tensor],
        node_to_task: Optional[torch.Tensor] = None,
        edge_to_task: Optional[List[torch.Tensor]] = None,
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
            if self.query_node_films is not None:
                edge_queries = self.query_node_films[edge_type](
                    edge_queries, task_ids=edge_to_task[edge_type]
                )
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


@dataclass
class FiLMGNNConfig:
    type: str = "MultiHeadAttention"
    num_edge_types: int = 3
    hidden_dim: int = 128
    num_heads: int = 4
    per_head_dim: int = 32
    intermediate_dim: int = 512
    message_function_depth: int = 1
    num_layers: int = 8
    dropout_rate: float = 0.0
    use_rezero_scaling: bool = True
    make_edges_bidirectional: bool = True
    use_msg_film: bool = False
    use_msg_att_film: bool = False


class FiLMGNNBlock(nn.Module):
    """Block in a GNN, following a Transformer-like residual structure, using the "Pre-Norm" style
    and ReZero weighting using \alpha:
      v' = v + \alpha * Dropout(NeighbourHoodAtt(LN(v))))
      v = v' + \alpha * Linear(Dropout(Act(Linear(LN(v'))))))

    Pre-Norm reference: https://arxiv.org/pdf/2002.04745v1.pdf
    ReZero reference: https://arxiv.org/pdf/2003.04887v1.pdf
    ReZero' (with \alpha a vector instead of scalar): https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(
        self, config: FiLMGNNConfig, task_embedding_provider: Optional[TaskEmbeddingLayerProvider]
    ):
        super().__init__()
        self.config = config

        if config.use_rezero_scaling:
            self.alpha = nn.Parameter(torch.full(size=(1,), fill_value=SMALL_NUMBER))

        self.mp_layers: nn.ModuleList = nn.ModuleList()
        if config.type.lower() == "MultiHeadAttention".lower():
            self.mp_layer_in_dim = config.hidden_dim
            self.mp_layers.append(
                FiLMRelationalMultiHeadAttentionMP(
                    hidden_dim=self.mp_layer_in_dim,
                    num_heads=config.num_heads,
                    per_head_dim=config.per_head_dim,
                    num_edge_types=config.num_edge_types,
                    message_function_depth=config.message_function_depth,
                    task_embedding_provider=task_embedding_provider,
                    use_msg_film=config.use_msg_film,
                    use_msg_attn_film=config.use_msg_att_film,
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
                        FiLMRelationalMultiAggrMP(
                            hidden_dim=self.mp_layer_in_dim,
                            msg_dim=config.per_head_dim,
                            num_edge_types=config.num_edge_types,
                            message_function_depth=config.message_function_depth,
                            use_pna_scalers=config.type.lower() == "PNA".lower(),
                            task_embedding_provider=task_embedding_provider,
                            use_msg_film=config.use_msg_film,
                        )
                    )
                elif config.type.lower() == "Plain".lower():
                    self.mp_layers.append(
                        FiLMRelationalMP(
                            hidden_dim=self.mp_layer_in_dim,
                            msg_dim=config.per_head_dim,
                            num_edge_types=config.num_edge_types,
                            message_function_depth=config.message_function_depth,
                            task_embedding_provider=task_embedding_provider,
                            use_msg_film=config.use_msg_film,
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
        node_to_task: Optional[torch.Tensor] = None,
        edge_to_task: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            node_representations: float tensor of shape (num_nodes, config.hidden_dim)
            adj_lists: List of (num_edges, 2) tensors (one per edge-type)
            node_to_task: long tensor of shape (num_nodes,)
            edge_to_task: optional list of long tensors of shape (num_edges,), mapping each edge
                to the task ID of the enclosing graph (one tensor per edge type)
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
                    node_to_task=node_to_task,
                    edge_to_task=edge_to_task,
                )
            )

        # ----- Pass through linear to get back to config.hidden_dim
        new_representations = self.msg_out_projection(torch.cat(aggregated_messages, dim=-1))
        # ----- Pass through dropout
        new_representations = self.dropout_layer(new_representations)
        # ----- Apply rezero scaling if required
        if self.config.use_rezero_scaling:
            new_representations = self.alpha * new_representations
        # ----- Sum over all representations (same dimensions)
        node_representations = node_representations + new_representations

        if self.boom_layer is not None and self.boom_norm_layer is not None:
            boomed_representations = self.dropout_layer(
                self.boom_layer(self.boom_norm_layer(node_representations))
            )
            if self.config.use_rezero_scaling:
                boomed_representations = self.alpha * boomed_representations
            node_representations = node_representations + boomed_representations

        return node_representations


class FiLMGNN(nn.Module):
    def __init__(
        self,
        config: FiLMGNNConfig,
        task_embedding_provider: Optional[TaskEmbeddingLayerProvider] = None,
    ):
        super().__init__()
        self.config = config

        self.gnn_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.gnn_blocks.append(FiLMGNNBlock(config, task_embedding_provider))

    def forward(self, node_features, adj_lists, node_to_task) -> List[torch.Tensor]:
        """
        args:
            node_representations: float tensor of shape (num_nodes, config.hidden_dim)
            adj_lists: List of (num_edges, 2) tensors (one per edge-type)
            node_to_task: long tensor of shape (num_nodes,)
        output:
            all_node_representations: list of float32 (num_graphs, config.hidden_dim) tensors,
                one for the result of each timestep of the GNN (and the initial one)
        """

        # First check if we got what we needed:
        if node_to_task is None and (self.config.use_msg_film or self.config.use_msg_att_film):
            raise ValueError(
                f"Using FiLM/task embeddings requires passing in the node_to_task map!"
            )

        # We may need to introduce additional edges to make everything bidirectional:
        if self.config.make_edges_bidirectional:
            adj_lists = [
                np.concatenate((adj_list, np.flip(adj_list, axis=1)), axis=0)
                for adj_list in adj_lists
            ]

        # We need to make the adjacency lists appropriate tensors:
        # If node_to_task also exists we can use this to assign each edge
        # to a task.
        torch_adj_lists, edge_to_task = [], []
        for adj_list in adj_lists:
            torch_adj_list = torch.tensor(adj_list, dtype=torch.long, device=node_features.device)
            torch_adj_lists.append(torch_adj_list)
            if node_to_task is not None:
                # grab the task id of each of the first nodes of each edge of this type
                edge_to_task.append(node_to_task[torch_adj_list[:, 0]])

        # Actually do message passing:
        cur_node_representations = node_features
        all_node_representations: List[torch.Tensor] = [cur_node_representations]
        for gnn_block in self.gnn_blocks:
            cur_node_representations = gnn_block(
                node_representations=cur_node_representations,
                adj_lists=torch_adj_lists,
                node_to_task=node_to_task,
                edge_to_task=edge_to_task,
            )
            all_node_representations.append(cur_node_representations)

        return all_node_representations
