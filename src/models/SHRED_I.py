import os
import sys
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Tuple, Optional, Any
from torch.optim import Optimizer
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from models.SHRED import SHRED
from utils.loss import kl_divergence_loss
from utils.metrics import ClassificationMetrics as CM


class SHRED_I(SHRED):
    
    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        site: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        ch_res = self.ch(x, age, gender, site)
        vae_res = self.vae_ffn(ch_res["x_ch"])

        x_ch_mu = vae_res["x_mu"]
        vae_res["x_ch_mu"] = x_ch_mu
        vae_res["x_mu"] = self.ch.inverse(
            x_ch_mu,
            ch_res["age"],
            ch_res["gender"],
            ch_res["gamma"],
            ch_res["delta"],
        )
        return {**ch_res, **vae_res}

    def ss_forward(self, x_ch: torch.Tensor) -> torch.Tensor:
        y = self.vae_ffn.ss_forward(x_ch)
        return y

    def get_baselines_inputs(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = data.x, data.y
        age, gender, site = data.age, data.gender, data.d
        ch_res = self.combat(x, age, gender, site)
        x_ch: torch.Tensor = ch_res["x_ch"]
        baselines = x_ch[y == 0].mean(dim=0).view(1, -1)
        inputs = x_ch[y == 1]
        return baselines, inputs

    def ls_forward(self, data: Data) -> torch.Tensor:
        x, age, gender, site = data.x, data.age, data.gender, data.d
        ch_res = self.combat(x, age, gender, site)
        x_ch: torch.Tensor = ch_res["x_ch"]
        z_mu, _ = self.vae_ffn.encode(x_ch)
        return z_mu

    def is_forward(self, data: Data) -> torch.Tensor:
        x, age, gender, site = data.x, data.age, data.gender, data.d
        ch_res = self.combat(x, age, gender, site)
        x_ch: torch.Tensor = ch_res["x_ch"]
        return x_ch

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
        labeled_age: torch.Tensor = labeled_data.age
        labeled_gender: torch.Tensor = labeled_data.gender
        labeled_site: torch.Tensor = labeled_data.d
        real_y: torch.Tensor = labeled_data.y
        labeled_x, labeled_age, labeled_gender, labeled_site, real_y = (
            labeled_x.to(device),
            labeled_age.to(device),
            labeled_gender.to(device),
            labeled_site.to(device),
            real_y.to(device),
        )

        if unlabeled_data is not None:
            unlabeled_x: torch.Tensor = unlabeled_data.x
            unlabeled_age: torch.Tensor = unlabeled_data.age
            unlabeled_gender: torch.Tensor = unlabeled_data.gender
            unlabeled_site: torch.Tensor = unlabeled_data.d
            unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site = (
                unlabeled_x.to(device),
                unlabeled_age.to(device),
                unlabeled_gender.to(device),
                unlabeled_site.to(device),
            )

        with torch.enable_grad():
            optimizer.zero_grad()

            labeled_res = self(
                labeled_x, labeled_age, labeled_gender, labeled_site
            )
            alpha: torch.Tensor = labeled_res["alpha"]
            labeled_age_x = labeled_res["age"]
            labeled_gender_x = labeled_res["gender"]
            pred_y = labeled_res["y"]
            labeled_x_mu = labeled_res["x_mu"]
            labeled_x_std = labeled_res["x_std"]
            labeled_z_mu = labeled_res["z_mu"]
            labeled_z_std = labeled_res["z_std"]
            labeled_eps = labeled_res["eps"]
            if unlabeled_data is not None:
                unlabeled_res = self(
                    unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site
                )
                unlabeled_age_x = unlabeled_res["age"]
                unlabeled_gender_x = unlabeled_res["gender"]
                unlabeled_x_mu = unlabeled_res["x_mu"]
                unlabeled_x_std = unlabeled_res["x_std"]
                unlabeled_z_mu = unlabeled_res["z_mu"]
                unlabeled_z_std = unlabeled_res["z_std"]
                unlabeled_eps = unlabeled_res["eps"]
                age_x = torch.cat((labeled_age_x, unlabeled_age_x), dim=0)
                gender_x = torch.cat(
                    (labeled_gender_x, unlabeled_gender_x), dim=0
                )
                x = torch.cat((labeled_x, unlabeled_x), dim=0)
                x_mu = torch.cat((labeled_x_mu, unlabeled_x_mu), dim=0)
                x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
                z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
                z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
                eps = torch.cat((labeled_eps, unlabeled_eps), dim=0)
            else:
                age_x = labeled_age_x
                gender_x = labeled_gender_x
                x = labeled_x
                x_mu = labeled_x_mu
                x_std = labeled_x_std
                z_mu = labeled_z_mu
                z_std = labeled_z_std
                eps = labeled_eps

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            ch_loss = (eps ** 2).sum(dim=1).mean()
            
            alpha_loss = (
                F.mse_loss(
                    alpha.expand(age_x.size()),
                    x,
                    reduction="none",
                )
                .sum(dim=1)
                .mean()
            )

            gamma1 = hyperparameters.get("rc_loss", 1)
            gamma2 = hyperparameters.get("kl_loss", 1)
            gamma3 = hyperparameters.get("ch_loss", 1)
            use_alpha_loss = hyperparameters.get("alpha_loss", True)

            if use_alpha_loss:
                total_loss = (
                    ce_loss
                    + gamma1 * rc_loss
                    + gamma2 * kl_loss
                    + gamma3 * (ch_loss + alpha_loss)
                )
            else:
                total_loss = (
                    ce_loss
                    + gamma1 * rc_loss
                    + gamma2 * kl_loss
                    + gamma3 * ch_loss
                )
            total_loss.backward()
            clip_grad_norm_(self.parameters(), 0.1)
            optimizer.step()

        accuracy = CM.accuracy(real_y, pred_y)
        sensitivity = CM.tpr(real_y, pred_y)
        specificity = CM.tnr(real_y, pred_y)
        precision = CM.ppv(real_y, pred_y)
        f1_score = CM.f1_score(real_y, pred_y)
        metrics = {
            "ce_loss": ce_loss.item(),
            "rc_loss": rc_loss.item(),
            "kl_loss": kl_loss.item(),
            "ch_loss": ch_loss.item(),
            "alpha_loss": alpha_loss.item(),
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
            age: torch.Tensor = test_data.age
            gender: torch.Tensor = test_data.gender
            site: torch.Tensor = test_data.d
            real_y: torch.Tensor = test_data.y
            x, age, gender, site, real_y = (
                x.to(device),
                age.to(device),
                gender.to(device),
                site.to(device),
                real_y.to(device),
            )

            res = self(x, age, gender, site)
            pred_y = res["y"]
            x_mu = res["x_mu"]
            x_std = res["x_std"]
            z_mu = res["z_mu"]
            z_std = res["z_std"]
            alpha: torch.Tensor = res["alpha"]
            age_x: torch.Tensor = res["age"]
            gender_x = res["gender"]
            eps: torch.Tensor = res["eps"]

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            ch_loss = (eps ** 2).sum(dim=1).mean()
            
            alpha_loss = (
                F.mse_loss(
                    alpha.expand(age_x.size()),
                    x,
                    reduction="none",
                )
                .sum(dim=1)
                .mean()
            )

        accuracy = CM.accuracy(real_y, pred_y)
        sensitivity = CM.tpr(real_y, pred_y)
        specificity = CM.tnr(real_y, pred_y)
        precision = CM.ppv(real_y, pred_y)
        f1_score = CM.f1_score(real_y, pred_y)
        metrics = {
            "ce_loss": ce_loss.item(),
            "rc_loss": rc_loss.item(),
            "kl_loss": kl_loss.item(),
            "ch_loss": ch_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "accuracy": accuracy.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "f1": f1_score.item(),
            "precision": precision.item(),
        }
        return metrics
