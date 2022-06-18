from __future__ import annotations
from abc import abstractmethod
from typing import Sequence, Optional, Dict, Any

import torch
from torch_geometric.data import Data
from torch.optim import Optimizer, Adam
from torch.nn import Module, Sequential, Parameter, Linear, ReLU, Dropout, Tanh


class LinearLayer(Sequential):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0):
        super().__init__(
            Linear(input_size, output_size), ReLU(), Dropout(dropout)
        )


class FeedForward(Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_size: Sequence[int],
        output_size: int,
        output_activation: Optional[Module] = None,
        dropout: float = 0,
    ):
        layers_dim = [input_size] + list(hidden_size) + [output_size]
        layers: list = [
            LinearLayer(layers_dim[i], layers_dim[i + 1], dropout)
            for i in range(len(layers_dim) - 2)
        ] + [
            Linear(layers_dim[-2], layers_dim[-1]),
        ]
        if output_activation is not None:
            layers.append(output_activation)
        super().__init__(*layers)


class VariationalEncoder(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Sequence[int],
        emb_size: int,
        dropout: float = 0,
    ):
        super().__init__()
        layers_dim = [input_size] + list(hidden_size) + [emb_size]
        layers = [
            LinearLayer(layers_dim[i], layers_dim[i + 1], dropout)
            for i in range(len(layers_dim) - 2)
        ]
        self.hidden = Sequential(*layers,)
        self.mu = Linear(layers_dim[-2], layers_dim[-1])
        self.log_std = Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x: torch.Tensor):
        hidden = self.hidden(x)
        mu = self.mu(hidden)
        log_std: torch.Tensor = self.log_std(hidden)
        std = log_std.exp()
        return mu, std


class VariationalDecoder(Module):
    def __init__(
        self,
        emb_size: int,
        hidden_size: Sequence[int],
        output_size: int,
        output_activation: Module = Tanh(),
        dropout: float = 0,
    ):
        super().__init__()
        self.decoder = FeedForward(
            emb_size, hidden_size, output_size, output_activation, dropout
        )
        self.log_std = Parameter(torch.zeros(1, output_size))

    def forward(self, z: torch.Tensor):
        mu = self.decoder(z)
        std = self.log_std.expand(z.size(0), self.log_std.size(1)).exp()
        return mu, std


class ModelBase(Module):
    def __init__(self):
        super().__init__()

    def get_optimizer(self, param: dict) -> Optimizer:
        optim = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param.get("lr", 0.0001),
            weight_decay=param.get("l2_reg", 0.0),
        )
        return optim

    @abstractmethod
    def train_step(
        self,
        device: torch.device,
        labeled_data: Data,
        unlabeled_data: Optional[Data],
        optimizer: Optimizer,
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def test_step(
        self, device: torch.device, test_data: Data
    ) -> Dict[str, float]:
        raise NotImplementedError
