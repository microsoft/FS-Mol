from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter

from fs_mol.modules.mlp import MLP
from fs_mol.modules.task_specific_modules import TaskEmbeddingLayerProvider


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
        node_to_task: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: representations of individual graph nodes. A float tensor
                of shape [num_nodes, self.node_dim].
            node_to_graph_id: int tensor of shape [num_nodes], assigning a graph_id to each
                node.
            num_graphs: int scalar, giving the number of graphs in the batch.
            node_to_task: long tensor of shape (num_nodes,) assigning a task id to each node.
            Optional if no task-dependent readout is configured.

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
        task_embedding_provider: Optional[TaskEmbeddingLayerProvider] = None,
        use_task_specific_scores: bool = False,
    ):
        """
        See superclass for first few parameters.

        Args:
            num_heads: Number of independent heads to use for independent weights.
            head_dim: Size of the result of each independent head.
            num_mlp_layers: Number of layers in the MLPs used to compute per-head weights and
                outputs.
            task_embedding_layer_provider: provider of layer to use task embedding at this point.
            use_task_specific_scores: Use task-specific scores when performing the final pooling
                operation.
        """
        super().__init__(node_dim, out_dim)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._use_task_specific_scores = use_task_specific_scores

        # Create weighted_mean, weighted_sum, max pooling layers:
        self._weighted_mean_pooler = MultiHeadWeightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            task_embedding_provider=task_embedding_provider,
            weighting_type=f"{'task_' if use_task_specific_scores else ''}weighted_mean",
        )
        self._weighted_sum_pooler = MultiHeadWeightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            task_embedding_provider=task_embedding_provider,
            weighting_type=f"{'task_' if use_task_specific_scores else ''}weighted_sum",
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
        node_to_task: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean_graph_repr = self._weighted_mean_pooler(
            node_embeddings, node_to_graph_id, num_graphs, node_to_task
        )
        sum_graph_repr = self._weighted_sum_pooler(
            node_embeddings, node_to_graph_id, num_graphs, node_to_task
        )
        max_graph_repr = self._max_pooler(
            node_embeddings, node_to_graph_id, num_graphs, node_to_task
        )

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
        weighting_type: str,
        num_mlp_layers: int = 1,
        task_embedding_provider: Optional[TaskEmbeddingLayerProvider] = None,
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
            task_embedding_provider: Optional task embedding layer provider, should the task embedding
                need to be used here.
        """
        super().__init__(node_dim, out_dim)
        self._num_heads = num_heads
        self._head_dim = head_dim

        if weighting_type not in (
            "weighted_sum",
            "weighted_mean",
            "task_weighted_sum",
            "task_weighted_mean",
        ):
            raise ValueError(f"Unknown weighting type {weighting_type}!")
        self._weighting_type = weighting_type

        # the task specific component only affects via the use_task_specific_scores
        if weighting_type.startswith("task_"):
            self._use_task_specific_scores = True
            self._weighting_type = weighting_type[5:]
        else:
            self._use_task_specific_scores = False
            self._weighting_type = weighting_type

        if self._use_task_specific_scores:
            assert (
                task_embedding_provider is not None
            ), "Using task specific scores requires an embedding provider!"
            self._scoring_projection = torch.nn.Linear(self._node_dim, self._head_dim * num_heads)
            self._scoring_module = task_embedding_provider.get_task_linear_layer(
                in_size=self._head_dim * num_heads,
                out_size=num_heads,
            )
        else:
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
        node_to_task: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Step 1: compute scores, then normalise them according to config:
        if self._use_task_specific_scores:
            projected_node_embeddings = self._scoring_projection(
                node_embeddings
            )  # [V, node_dim * num_heads]
            scores = self._scoring_module(
                projected_node_embeddings,
                node_to_task,
            )  # [V, num_heads]
        else:
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
        pooling_type: str,
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
        node_to_task: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_graph_values = scatter(
            src=node_embeddings,
            index=node_to_graph_id,
            dim=0,
            dim_size=num_graphs,
            reduce=self._pooling_type,
        )  # [num_graphs, self.pooling_input_dim]
        return self._combination_layer(per_graph_values)  # [num_graphs, out_dim]
