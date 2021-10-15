from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))
sys.path.insert(0, os.path.join(str(project_root()), "third_party", "MAT", "src"))

from fs_mol.data.mat import FSMolMATBatch
from fs_mol.models.abstract_torch_fsmol_model import (
    AbstractTorchFSMolModel,
    ModelStateType,
    TorchFSMolModelOutput,
    TorchFSMolModelLoss,
)

# Assumes that MAT is in the python lib path:
from transformer import GraphTransformer, make_model


logger = logging.getLogger(__name__)


class MATModel(
    GraphTransformer,
    AbstractTorchFSMolModel[FSMolMATBatch, TorchFSMolModelOutput, TorchFSMolModelLoss],
):
    def forward(self, batch: FSMolMATBatch) -> Any:
        mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0

        return TorchFSMolModelOutput(
            molecule_binary_label=super().forward(
                batch.node_features, mask, batch.adjacency_matrix, batch.distance_matrix, None
            )
        )

    def get_model_state(self) -> Dict[str, Any]:
        return {"model_state_dict": self.state_dict()}

    def is_param_task_specific(self, param_name: str) -> bool:
        return param_name.startswith("generator")

    def load_model_state(
        self,
        model_state: ModelStateType,
        load_task_specific_weights: bool,
        quiet: bool = False,
    ) -> None:
        # Checkpoints saved by us are richer than the original pre-trained checkpoint, as they also
        # contain optimizer state. For now we only want the weights, so throw out the rest.
        if "model_state_dict" in model_state:
            pretrained_state_dict = model_state["model_state_dict"]

        for name, param in pretrained_state_dict.items():
            if not load_task_specific_weights and self.is_param_task_specific(name):
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> MATModel:
        # Parameters used for pretraining the original MAT model.
        model_params = {
            "d_atom": 28,
            "d_model": 1024,
            "N": 8,
            "h": 16,
            "N_dense": 1,
            "lambda_attention": 0.33,
            "lambda_distance": 0.33,
            "leaky_relu_slope": 0.1,
            "dense_output_nonlinearity": "relu",
            "distance_matrix_kernel": "exp",
            "dropout": 0.0,
            "aggregation_type": "mean",
        }

        model = make_model(**model_params)
        model.to(device)

        # Cast to a subclass, which is valid because `MATModel` only adds a bunch of methods.
        model.__class__ = MATModel

        return model
