from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import os
import json
import time
import copy
import logging
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from data import Dataset
from models.EDC_VAE import EDC_VAE
from utils.misc import (
    get_device,
    get_pbar,
    mkdir,
    on_error,
    seed_torch,
    count_parameters,
)


@dataclass(frozen=True)
class TrainerParams:
    output_directory: str
    model_name: str
    model_params: Dict[str, Any]
    optim_params: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    dataset_path: str
    dataset_name: str
    seeds: Sequence[int]
    num_fold: int
    ssl: bool
    harmonize: bool
    labeled_sites: Optional[Union[str, Sequence[str]]] = field(default=None)
    device: int = field(default=-1)
    verbose: bool = field(default=False)
    patience: int = field(default=np.inf)
    max_epoch: int = field(default=1000)
    save_model: bool = field(default=False)
    time_id: bool = field(init=False, default_factory=lambda: int(time.time()))

    @property
    def dataset(self) -> Dataset:
        return Dataset(self.dataset_path, self.dataset_name, self.harmonize)

    def to_dict(self, seed: int, fold: int) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": str(self.model_params),
            "optim_params": str(self.optim_params),
            "hyperparameters": str(self.hyperparameters),
            "dataset": self.dataset_name,
            "seed": seed,
            "fold": fold,
            "ssl": self.ssl,
            "harmonize": self.harmonize,
            "labeled_sites": self.labeled_sites,
            "device": self.device,
            "epochs_log_path": self.epochs_log_path(seed, fold),
        }

    def model_path(self, seed: int, fold: int) -> str:
        return os.path.join(
            os.path.abspath(self.output_directory),
            "models",
            "{}_{}_{}_{}_{}.pt".format(
                self.dataset_name, self.model_name, seed, fold, self.time_id,
            ),
        )

    def epochs_log_path(self, seed: int, fold: int) -> str:
        return os.path.join(
            os.path.abspath(self.output_directory),
            "epochs_log",
            "{}_{}_{}_{}_{}.log".format(
                self.dataset_name, self.model_name, seed, fold, self.time_id,
            ),
        )


@dataclass(frozen=True)
class TrainerResults:
    trainer_params_dict: Dict[str, Any]
    num_labeled_train: int
    num_unlabeled_train: int
    num_valid: int
    baseline_accuracy: float
    best_metrics: Dict[str, float]
    best_epoch: int
    time_taken: int
    model_size: int
    model_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.trainer_params_dict,
            "num_labeled_train": self.num_labeled_train,
            "num_unlabeled_train": self.num_unlabeled_train,
            "num_valid": self.num_valid,
            "baseline_accuracy": self.baseline_accuracy,
            **self.best_metrics,
            "best_epoch": self.best_epoch,
            "time_taken": self.time_taken,
            "model_size": self.model_size,
            "model_path": self.model_path,
        }


class Trainer(ABC):
    def __init__(
        self, trainer_params: TrainerParams,
    ):
        super().__init__()
        self.trainer_params = trainer_params
        self.dataset = trainer_params.dataset
        self.__called = False

    def _set_called(self):
        if self.__called:
            raise Exception("Trainer.run() can only be called once")
        self.__called = True

    @staticmethod
    def verbose_info(train_metrics: dict, valid_metrics: dict) -> str:
        all_metrics = []
        for k, v in train_metrics.items():
            all_metrics.append("train_{}: {:.4f}".format(k, v))
        for k, v in valid_metrics.items():
            all_metrics.append("valid_{}: {:.4f}".format(k, v))
        return " ".join(all_metrics)

    @staticmethod
    def _get_baseline_accuracy(data: Union[Data, Sequence[Data]]) -> float:
        if not isinstance(data, Data):
            y = torch.cat([d.y for d in data], dim=0)
        else:
            y = data.y
        _, counts = y.unique(return_counts=True)
        return (counts.max() / y.size(0)).item()

    @abstractmethod
    def run(self):
        raise NotImplementedError


class EDC_VAE_Trainer(Trainer):
    @on_error(None, True)
    def _run_single_seed_fold(
        self, seed: int, fold: int, data_dict: Dict[str, Union[Data, int]]
    ) -> TrainerResults:
        seed_torch()
        device = get_device(self.trainer_params.device)
        verbose = self.trainer_params.verbose

        start = time.time()
        num_labeled_train = data_dict.get("num_labeled_train", 0)
        num_unlabeled_train = data_dict.get("num_unlabeled_train", 0)
        num_valid = data_dict.get("num_test", 0)
        baseline_accuracy = self._get_baseline_accuracy(data_dict.get("test"))

        self.trainer_params.model_params["input_size"] = data_dict["input_size"]
        self.trainer_params.model_params["num_sites"] = data_dict["num_sites"]
        model = EDC_VAE(**self.trainer_params.model_params)
        model_size = count_parameters(model)
        optimizer = model.get_optimizer(self.trainer_params.optim_params)

        patience = self.trainer_params.patience
        cur_patience = 0
        max_epoch = self.trainer_params.max_epoch
        best_epoch = 0
        best_metrics = {
            "ce_loss": np.inf,
            "accuracy": 0,
        }
        save_model = self.trainer_params.save_model
        best_model_state_dict = None

        epochs_log_path = self.trainer_params.epochs_log_path(seed, fold)
        mkdir(os.path.dirname(epochs_log_path))
        with open(epochs_log_path, "w") as f:
            f.write("")

        pbar = get_pbar(max_epoch, verbose)
        for epoch in pbar:
            try:
                train_metrics = model.train_step(
                    device,
                    data_dict.get("labeled_train", None),
                    data_dict.get("unlabeled_train", None),
                    optimizer,
                    self.trainer_params.hyperparameters,
                )
                valid_metrics = model.test_step(
                    device, data_dict.get("test", None)
                )
            except Exception as e:
                logging.error(e)
            with open(epochs_log_path, "a") as f:
                f.write(
                    json.dumps(
                        {"train": train_metrics, "valid": valid_metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )

            """
            save priority:
            1. accuracy
            2. ce_loss
            """
            save = valid_metrics["accuracy"] > best_metrics["accuracy"] or (
                valid_metrics["accuracy"] == best_metrics["accuracy"]
                and valid_metrics["ce_loss"] < best_metrics["ce_loss"]
            )
            if save:
                best_epoch = epoch
                best_metrics = valid_metrics.copy()
                if save_model:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                cur_patience = 0
            else:
                cur_patience += 1

            if verbose:
                pbar.set_postfix_str(
                    self.verbose_info(train_metrics, valid_metrics)
                )
            if cur_patience == patience:
                break

        if save_model and best_model_state_dict is not None:
            try:
                model_path = self.trainer_params.model_path(seed, fold)
                mkdir(os.path.dirname(model_path))
                torch.save(best_model_state_dict, model_path)
            except Exception as e:
                logging.error(str(e))
                model_path = None
        else:
            model_path = None

        end = time.time()
        return TrainerResults(
            trainer_params_dict=self.trainer_params.to_dict(seed, fold),
            num_labeled_train=num_labeled_train,
            num_unlabeled_train=num_unlabeled_train,
            num_valid=num_valid,
            baseline_accuracy=baseline_accuracy,
            best_metrics=best_metrics,
            best_epoch=best_epoch,
            time_taken=end - start,
            model_size=model_size,
            model_path=model_path,
        )

    def run(self, results_csv_path: str) -> List[Dict[str, Any]]:
        self._set_called()
        all_seed_fold_results = list()

        for seed in self.trainer_params.seeds:
            data_dict_generator = self.dataset.load_split_data(
                seed=seed,
                num_fold=self.trainer_params.num_fold,
                ssl=self.trainer_params.ssl,
                labeled_sites=self.trainer_params.labeled_sites,
            )

            for fold, data_dict in enumerate(data_dict_generator):
                fold_result: Optional[
                    TrainerResults
                ] = self._run_single_seed_fold(seed, fold, data_dict)
                if fold_result is None:
                    continue

                all_seed_fold_results.append(fold_result.to_dict())
                logging.info(
                    "RESULT:\n{}".format(
                        json.dumps(all_seed_fold_results[-1], indent=4)
                    )
                )

                df = pd.DataFrame(all_seed_fold_results).dropna(how="all")
                if not df.empty:
                    df.to_csv(results_csv_path, index=False)

        return all_seed_fold_results
