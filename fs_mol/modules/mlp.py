from typing import List

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, out_dim: int, hidden_layer_dims: List[int], activation=nn.ReLU()
    ):
        super().__init__()

        layers = []
        cur_hidden_dim = input_dim
        for hidden_layer_dim in hidden_layer_dims:
            layers.append(nn.Linear(cur_hidden_dim, hidden_layer_dim))
            layers.append(activation)
            cur_hidden_dim = hidden_layer_dim
        layers.append(nn.Linear(cur_hidden_dim, out_dim))
        self._layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self._layers(inputs)
