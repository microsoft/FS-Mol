from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from typing_extensions import Literal

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter

from fs_mol.modules.mlp import MLP
from fs_mol.utils.cli_utils import str2bool


@dataclass(frozen=True)
class GraphReadoutConfig:
    readout_type: Literal[
        "sum",
        "min",
        "max",
        "mean",
        "weighted_sum",
        "weighted_mean",
        "combined",
    ] = "combined"
    use_all_states: bool = True
    num_heads: int = 12
    head_dim: int = 64
    output_dim: int = 512


def add_graph_readout_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--readout_type",
        type=str,
        default="combined",
        choices=[
            "sum",
            "min",
            "max",
            "mean",
            "weighted_sum",
            "weighted_mean",
            "combined",
        ],
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
        default=512,
        help="Dimensionality of the readout result.",
    )


def make_graph_readout_config_from_args(args: argparse.Namespace) -> GraphReadoutConfig:
    return GraphReadoutConfig(
        readout_type=args.readout_type,
        use_all_states=args.readout_use_all_states,
        num_heads=args.readout_num_heads,
        head_dim=args.readout_head_dim,
        output_dim=args.readout_output_dim,
    )


class GraphReadout(nn.Module, ABC):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
    ):
        """
        Args:
            node_dim: Dimension of each node node representation.
            out_dim: Dimension of the graph repersentation to produce.
        """
        super().__init__()
        self._node_dim = node_dim
        self._out_dim = out_dim

    @abstractmethod
    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_to_graph_id: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: representations of individual graph nodes. A float tensor
                of shape [num_nodes, self.node_dim].
            node_to_graph_id: int tensor of shape [num_nodes], assigning a graph_id to each
                node.
            num_graphs: int scalar, giving the number of graphs in the batch.

        Returns:
            float tensor of shape [num_graphs, out_dim]
        """
        pass


class CombinedGraphReadout(GraphReadout):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        num_heads: int,
        head_dim: int,
    ):
        """
        See superclass for first few parameters.

        Args:
            num_heads: Number of independent heads to use for independent weights.
            head_dim: Size of the result of each independent head.
            num_mlp_layers: Number of layers in the MLPs used to compute per-head weights and
                outputs.
        """
        super().__init__(node_dim, out_dim)
        self._num_heads = num_heads
        self._head_dim = head_dim

        # Create weighted_mean, weighted_sum, max pooling layers:
        self._weighted_mean_pooler = MultiHeadWeightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            weighting_type="weighted_mean",
        )
        self._weighted_sum_pooler = MultiHeadWeightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            weighting_type="weighted_sum",
        )
        self._max_pooler = UnweightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            pooling_type="max",
        )

        # Single linear layer to combine results:
        self._combination_layer = nn.Linear(3 * out_dim, out_dim, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_to_graph_id: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        mean_graph_repr = self._weighted_mean_pooler(node_embeddings, node_to_graph_id, num_graphs)
        sum_graph_repr = self._weighted_sum_pooler(node_embeddings, node_to_graph_id, num_graphs)
        max_graph_repr = self._max_pooler(node_embeddings, node_to_graph_id, num_graphs)

        # concat & non-linearity & combine:
        raw_graph_repr = torch.cat((mean_graph_repr, sum_graph_repr, max_graph_repr), dim=1)

        return self._combination_layer(nn.functional.relu(raw_graph_repr))


class MultiHeadWeightedGraphReadout(GraphReadout):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        num_heads: int,
        head_dim: int,
        weighting_type: Literal["weighted_sum", "weighted_mean"],
        num_mlp_layers: int = 1,
    ):
        """
        See superclass for first few parameters.

        Args:
            num_heads: Number of independent heads to use for independent weights.
            head_dim: Size of the result of each independent head.
            weighting_type: Type of weighting to use, either "weighted_sum" (weights
                are in [0, 1], obtained through a logistic sigmoid) or "weighted_mean" (weights
                are in [0, 1] and sum up to 1 for each graph, obtained through a softmax).
            num_mlp_layers: Number of layers in the MLPs used to compute per-head weights and
                outputs.
        """
        super().__init__(node_dim, out_dim)
        self._num_heads = num_heads
        self._head_dim = head_dim

        if weighting_type not in (
            "weighted_sum",
            "weighted_mean",
        ):
            raise ValueError(f"Unknown weighting type {weighting_type}!")
        self._weighting_type = weighting_type

        self._scoring_module = MLP(
            input_dim=self._node_dim,
            hidden_layer_dims=[self._head_dim * num_heads] * num_mlp_layers,
            out_dim=num_heads,
        )

        self._transformation_mlp = MLP(
            input_dim=self._node_dim,
            hidden_layer_dims=[self._head_dim * num_heads] * num_mlp_layers,
            out_dim=num_heads * head_dim,
        )
        self._combination_layer = nn.Linear(num_heads * head_dim, out_dim, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_to_graph_id: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        # Step 1: compute scores, then normalise them according to config:
        scores = self._scoring_module(node_embeddings)  # [V, num_heads]

        if self._weighting_type == "weighted_sum":
            weights = torch.sigmoid(scores)  # [V, num_heads]
        elif self._weighting_type == "weighted_mean":
            weights = scatter_softmax(scores, index=node_to_graph_id, dim=0)  # [V, num_heads]
        else:
            raise ValueError(f"Unknown weighting type {self._weighting_type}!")

        # Step 2: compute transformed node representations:
        values = self._transformation_mlp(node_embeddings)  # [V, num_heads * head_dim]
        values = values.view(-1, self._num_heads, self._head_dim)  # [V, num_heads, head_dim]

        # Step 3: apply weights and sum up per graph:
        weighted_values = weights.unsqueeze(-1) * values  # [V, num_heads, head_dim]
        per_graph_values = torch.zeros(
            (num_graphs, self._num_heads * self._head_dim),
            device=node_embeddings.device,
        )
        per_graph_values.index_add_(
            0,
            node_to_graph_id,
            weighted_values.view(-1, self._num_heads * self._head_dim),
        )  # [num_graphs, num_heads * head_dim]

        # Step 4: go to output size:
        return self._combination_layer(per_graph_values)  # [num_graphs, out_dim]


class UnweightedGraphReadout(GraphReadout):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        pooling_type: Literal["min", "max", "sum", "mean"],
    ):
        """
        See superclass for first few parameters.

        Args:
            pooling_type: Type of pooling to use. One of "min", "max", "sum" and "mean".
        """
        super().__init__(node_dim, out_dim)
        self._pooling_type = pooling_type

        if pooling_type not in ("min", "max", "sum", "mean"):
            raise ValueError(f"Unknown weighting type {self.pooling_type}!")

        self._combination_layer = nn.Linear(self._node_dim, out_dim, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_to_graph_id: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        per_graph_values = scatter(
            src=node_embeddings,
            index=node_to_graph_id,
            dim=0,
            dim_size=num_graphs,
            reduce=self._pooling_type,
        )  # [num_graphs, self.pooling_input_dim]
        return self._combination_layer(per_graph_values)  # [num_graphs, out_dim]


def make_readout_model(
    readout_config: GraphReadoutConfig,
    readout_node_dim: int,
) -> GraphReadout:
    if readout_config.readout_type.startswith("combined"):
        return CombinedGraphReadout(
            node_dim=readout_node_dim,
            out_dim=readout_config.output_dim,
            num_heads=readout_config.num_heads,
            head_dim=readout_config.head_dim,
        )
    elif "weighted" in readout_config.readout_type:
        return MultiHeadWeightedGraphReadout(
            node_dim=readout_node_dim,
            out_dim=readout_config.output_dim,
            num_heads=readout_config.num_heads,
            head_dim=readout_config.head_dim,
            weighting_type=readout_config.readout_type,
        )
    else:
        return UnweightedGraphReadout(
            node_dim=readout_node_dim,
            out_dim=readout_config.output_dim,
            pooling_type=readout_config.readout_type,
        )
