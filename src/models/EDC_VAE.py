import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from torch.nn import Softmax, Tanh
from torch.optim import Optimizer
from torch.distributions import Normal
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import kl_divergence_loss
from utils.metrics import ClassificationMetrics as CM
from models.base import (
    ModelBase,
    FeedForward,
    VariationalDecoder,
    VariationalEncoder,
)


class EDC_VAE(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        emb_size: int,
        clf_hidden_1: int,
        clf_hidden_2: int,
        clf_output_size: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.encoder = VariationalEncoder(
            input_size,
            [hidden_size] if hidden_size > 0 else [],
            emb_size,
            dropout=dropout,
        )
        self.decoder = VariationalDecoder(
            emb_size,
            [hidden_size] if hidden_size > 0 else [],
            input_size,
            Tanh(),
            dropout=dropout,
        )
        self.classifier = FeedForward(
            emb_size,
            [h for h in [clf_hidden_1, clf_hidden_2] if h > 0],
            clf_output_size,
            Softmax(dim=1),
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.decoder(F.relu(z))

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(F.relu(z))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_mu, z_std = self.encode(x)
        if self.training:
            q = Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu
        x_mu, x_std = self.decode(z)
        y = self.classify(z)
        return {
            "y": y,
            "x_mu": x_mu,
            "x_std": x_std,
            "z": z,
            "z_mu": z_mu,
            "z_std": z_std,
        }

    def train_step(
        self,
        device: torch.device,
        labeled_data: Data,
        unlabeled_data: Optional[Data],
        optimizer: Optimizer,
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:

        self.to(device)
        self.train()

        labeled_x: torch.Tensor = labeled_data.x
        real_y: torch.Tensor = labeled_data.y
        labeled_x, real_y = (
            labeled_x.to(device),
            real_y.to(device),
        )

        if unlabeled_data is not None:
            unlabeled_x: torch.Tensor = unlabeled_data.x
            unlabeled_x = unlabeled_x.to(device)

        with torch.enable_grad():
            optimizer.zero_grad()

            labeled_res = self(labeled_x)
            pred_y = labeled_res["y"]
            labeled_x_mu = labeled_res["x_mu"]
            labeled_x_std = labeled_res["x_std"]
            labeled_z_mu = labeled_res["z_mu"]
            labeled_z_std = labeled_res["z_std"]
            if unlabeled_data is not None:
                unlabeled_res = self(unlabeled_x)
                unlabeled_x_mu = unlabeled_res["x_mu"]
                unlabeled_x_std = unlabeled_res["x_std"]
                unlabeled_z_mu = unlabeled_res["z_mu"]
                unlabeled_z_std = unlabeled_res["z_std"]
                x = torch.cat((labeled_x, unlabeled_x), dim=0)
                x_mu = torch.cat((labeled_x_mu, unlabeled_x_mu), dim=0)
                x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
                z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
                z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
            else:
                x = labeled_x
                x_mu = labeled_x_mu
                x_std = labeled_x_std
                z_mu = labeled_z_mu
                z_std = labeled_z_std

            ce_loss = F.cross_entropy(pred_y, real_y)
            ll_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )

            gamma1 = hyperparameters.get("ll_loss", 1)
            gamma2 = hyperparameters.get("kl_loss", 1)
            total_loss = ce_loss + gamma1 * ll_loss + gamma2 * kl_loss
            total_loss.backward()
            optimizer.step()

        accuracy = CM.accuracy(real_y, pred_y)
        sensitivity = CM.tpr(real_y, pred_y)
        specificity = CM.tnr(real_y, pred_y)
        precision = CM.ppv(real_y, pred_y)
        f1_score = CM.f1_score(real_y, pred_y)
        metrics = {
            "ce_loss": ce_loss.item(),
            "ll_loss": ll_loss.item(),
            "kl_loss": kl_loss.item(),
            "accuracy": accuracy.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "f1": f1_score.item(),
            "precision": precision.item(),
        }
        return metrics

    def test_step(
        self, device: torch.device, test_data: Data
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()

        with torch.no_grad():
            x: torch.Tensor = test_data.x
            real_y: torch.Tensor = test_data.y
            x, real_y = x.to(device), real_y.to(device)

            res = self(x)
            pred_y = res["y"]
            x_mu = res["x_mu"]
            x_std = res["x_std"]
            z_mu = res["z_mu"]
            z_std = res["z_std"]

            ce_loss = F.cross_entropy(pred_y, real_y)
            ll_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)

            metrics = {
                "ce_loss": ce_loss.item(),
                "ll_loss": ll_loss.item(),
                "kl_loss": kl_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics
