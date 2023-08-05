from __future__ import annotations
from typing import OrderedDict, Sequence, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from captum.attr import IntegratedGradients
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

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


class SaliencyScoreForward(ABC):
    @abstractmethod
    def ss_forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_baselines_inputs(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = data.x
        y: torch.Tensor = data.y
        baselines = x[y == 0].mean(dim=0).view(1, -1)
        inputs = x[y == 1]
        return baselines, inputs

    def saliency_score(self, data: Data) -> torch.Tensor:
        baselines, inputs = self.get_baselines_inputs(data)
        ig = IntegratedGradients(self.ss_forward, True)
        scores: torch.Tensor = ig.attribute(
            inputs=inputs, baselines=baselines, target=1
        )

        scores = scores.detach().cpu().numpy()
        print(scores[0].shape)
        scores = np.array([squareform(score) for score in scores])
        return scores

class LatentSpaceEncoding(ABC):
    @abstractmethod
    def ls_forward(self, data: Data) -> torch.Tensor:
        """
        return z
        """
        raise NotImplementedError

    @abstractmethod
    def is_forward(self, data: Data) -> torch.Tensor:
        """
        return z
        """
        raise NotImplementedError

    @abstractmethod
    def get_surface(self, z: torch.Tensor) -> torch.Tensor:
        """
        return y value for each z
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_surface(self, x: torch.Tensor) -> torch.Tensor:
        """
        return y value for each x
        """
        raise NotImplementedError

    @staticmethod
    def _prepare_grid(
        x: np.ndarray, pipeline: Pipeline, grid_points_dist: float = 0.1
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        min1, max1 = x[:, 0].min() - 1, x[:, 0].max() + 1
        min2, max2 = x[:, 1].min() - 1, x[:, 1].max() + 1
        x1grid = np.arange(min1, max1, grid_points_dist)
        x2grid = np.arange(min2, max2, grid_points_dist)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid_xt = np.hstack((r1, r2))
        emb_grid = pipeline.inverse_transform(grid_xt)
        return (xx, yy), emb_grid

    def get_latent_space_encoding(self, data: Data) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            z = self.ls_forward(data).detach().numpy()
            pipeline = make_pipeline(StandardScaler(), TruncatedSVD(2, random_state=0))
            x = pipeline.fit_transform(z)

            surface, emb_grid = self._prepare_grid(x, pipeline)
            emb_grid = torch.tensor(emb_grid, dtype=data.x.dtype)
            zz: np.ndarray = self.get_surface(emb_grid)[:, 1].detach().numpy()

            xx: np.ndarray = surface[0]
            yy: np.ndarray = surface[1]
            zz = zz.reshape(xx.shape)
        return {"x": x, "xx": xx, "yy": yy, "zz": zz}

    def get_input_space_encoding(self, data: Data) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = self.is_forward(data).detach().numpy()
            pipeline = make_pipeline(StandardScaler(), PCA(2, random_state=0))
            x = pipeline.fit_transform(x)

            surface, emb_grid = self._prepare_grid(
                x, pipeline, grid_points_dist=1.5
            )
            emb_grid = torch.tensor(emb_grid, dtype=data.x.dtype)
            zz: np.ndarray = self.get_input_surface(emb_grid)[
                :, 1
            ].detach().numpy()

            xx: np.ndarray = surface[0]
            yy: np.ndarray = surface[1]
            zz = zz.reshape(xx.shape)
        return {"x": x, "xx": xx, "yy": yy, "zz": zz}


class ModelBase(Module, SaliencyScoreForward):
    def __init__(self):
        super().__init__()

    def get_optimizer(self, param: dict) -> Optimizer:
        optim = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param.get("lr", 0.0001),
            weight_decay=param.get("l2_reg", 0.0),
        )
        return optim

    @classmethod
    def load_from_state_dict(
        cls,
        path: str,
        model_params: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ) -> ModelBase:
        state_dict: OrderedDict[str, torch.Tensor] = torch.load(
            path, map_location=device
        )
        model = cls(**model_params)
        model.load_state_dict(state_dict)
        return model

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
