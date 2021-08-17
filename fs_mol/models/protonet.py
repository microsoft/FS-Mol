from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal
import torch
import torch.nn as nn

from metamol.data.metamol_dataset import NUM_EDGE_TYPES, NUM_NODE_FEATURES
from metamol.modules.thick_gnn import ThickGNN, ThickGNNConfig
from metamol.modules.graph_readout import CombinedGraphReadout
from metamol.data.protonet import ProtoNetBatch, MoleculeProtoNetFeatures


FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class PrototypicalNetworkConfig:
    # Model configuration:
    gnn_config: ThickGNNConfig = ThickGNNConfig(
        type="PNA",
        hidden_dim=64,
        num_edge_types=NUM_EDGE_TYPES,
        num_heads=2,
        per_head_dim=32,
        intermediate_dim=512,
        message_function_depth=1,
        num_layers=8,
        use_msg_film=False,
        use_msg_att_film=False,
    )
    gnn_feature_dim: int = 512
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class GraphFeatureExtractor(nn.Module):
    def __init__(self, gnn_config: ThickGNNConfig, embedding_dim: int):
        super().__init__()

        readout_node_dim = (gnn_config.num_layers + 1) * gnn_config.hidden_dim
        self.init_node_proj = nn.Linear(
            in_features=NUM_NODE_FEATURES,  # This is fixed in our dataset
            out_features=gnn_config.hidden_dim,
            bias=False,
        )
        self.gnn = ThickGNN(gnn_config)
        self.readout_layer = CombinedGraphReadout(
            node_dim=readout_node_dim,
            out_dim=embedding_dim,
            num_heads=12,
            head_dim=64,
            use_task_specific_scores=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input: MoleculeProtoNetFeatures) -> torch.Tensor:
        initial_node_features = self.init_node_proj(
            torch.tensor(input.node_features, device=self.device)
        )
        all_node_representations = self.gnn(initial_node_features, input.adjacency_lists)
        readout_node_reprs = torch.cat(all_node_representations, dim=-1)
        mol_representations = self.readout_layer(
            node_embeddings=readout_node_reprs,
            node_to_graph_id=torch.tensor(
                input.node_to_graph, device=self.device, dtype=torch.long
            ),
            num_graphs=input.num_graphs,
        )
        return mol_representations


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        config: PrototypicalNetworkConfig,
    ):
        super().__init__()
        self.config = config

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                gnn_config=config.gnn_config,
                embedding_dim=config.gnn_feature_dim,
            )

        # Create MLP if needed:
        if self.config.used_features.endswith("+fc"):
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.gnn_feature_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: ProtoNetBatch):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
        if "ecfp" in self.config.used_features:
            support_features.append(
                torch.tensor(input_batch.support_features.fingerprints, device=self.device)
            )
            query_features.append(
                torch.tensor(input_batch.query_features.fingerprints, device=self.device)
            )
        if "pc-descs" in self.config.used_features:
            support_features.append(
                torch.tensor(input_batch.support_features.descriptors, device=self.device)
            )
            query_features.append(
                torch.tensor(input_batch.query_features.descriptors, device=self.device)
            )

        support_features_flat = torch.cat(support_features, dim=1)
        query_features_flat = torch.cat(query_features, dim=1)

        if self.config.distance_metric == "mahalanobis":
            class_means, class_precision_matrices = self.compute_class_means_and_precisions(
                support_features_flat, torch.tensor(input_batch.support_labels, device=self.device)
            )

            # grabbing the number of classes and query examples for easier use later
            number_of_classes = class_means.size(0)
            number_of_targets = query_features_flat.size(0)

            """
            Calculating the Mahalanobis distance between query examples and the class means
            including the class precision estimates in the calculations, reshaping the distances
            and multiplying by -1 to produce the sample logits
            """
            repeated_target = query_features_flat.repeat(1, number_of_classes).view(
                -1, class_means.size(1)
            )
            repeated_class_means = class_means.repeat(number_of_targets, 1)
            repeated_difference = repeated_class_means - repeated_target
            repeated_difference = repeated_difference.view(
                number_of_targets, number_of_classes, repeated_difference.size(1)
            ).permute(1, 0, 2)
            first_half = torch.matmul(repeated_difference, class_precision_matrices)
            logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1
        else:  # euclidean
            logits = self._protonets_euclidean_classifier(
                support_features_flat,
                query_features_flat,
                torch.tensor(input_batch.support_labels, device=self.device),
            )

        return logits

    def compute_class_means_and_precisions(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        precisions = []
        task_covariance_estimate = self._estimate_cov(features)
        for c in torch.unique(labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(features, 0, self._extract_class_indices(labels, c))
            # mean pooling examples to form class means
            means.append(torch.mean(class_features, dim=0, keepdim=True).squeeze())
            lambda_k_tau = class_features.size(0) / (class_features.size(0) + 1)
            lambda_k_tau = min(lambda_k_tau, 0.1)
            precisions.append(
                torch.inverse(
                    (lambda_k_tau * self._estimate_cov(class_features))
                    + ((1 - lambda_k_tau) * task_covariance_estimate)
                    + 0.1
                    * torch.eye(class_features.size(1), class_features.size(1)).to(self.device)
                )
            )

        means = torch.stack(means)
        precisions = torch.stack(precisions)

        return means, precisions

    @staticmethod
    def _estimate_cov(
        examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
    ) -> torch.Tensor:
        """
        SCM: Function based on the suggested implementation of Modar Tensai
        and his answer as noted in:
        https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError("m has more than 2 dimensions")
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.tensor(labels, dtype=torch.long, device=logits.device)
        )

    def _protonets_euclidean_classifier(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        class_prototypes = self._compute_class_prototypes(support_features, support_labels)
        logits = self._euclidean_distances(query_features, class_prototypes)
        return logits

    def _compute_class_prototypes(
        self, support_features: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        means = []
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(
                support_features, 0, self._extract_class_indices(support_labels, c)
            )
            means.append(torch.mean(class_features, dim=0))
        return torch.stack(means)

    def _euclidean_distances(
        self, query_features: torch.Tensor, class_prototypes: torch.Tensor
    ) -> torch.Tensor:
        num_query_features = query_features.shape[0]
        num_prototypes = class_prototypes.shape[0]

        distances = (
            (
                query_features.unsqueeze(1).expand(num_query_features, num_prototypes, -1)
                - class_prototypes.unsqueeze(0).expand(num_query_features, num_prototypes, -1)
            )
            .pow(2)
            .sum(dim=2)
        )

        return -distances
