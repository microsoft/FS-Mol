from __future__ import annotations

from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

import torch
from torch import nn


BatchType = TypeVar("BatchType")


class AbstractTorchModel(Generic[BatchType], nn.Module):
    @abstractmethod
    def forward(self, batch: BatchType) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def is_param_task_specific(self, param_name: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load model weights from a saved checkpoint."""
        raise NotImplementedError()

    @abstractclassmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> AbstractTorchModel:
        """Build the model architecture based on a saved checkpoint."""
        raise NotImplementedError()
