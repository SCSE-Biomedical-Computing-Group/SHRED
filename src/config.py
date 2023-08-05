from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

import os.path as osp
from itertools import product

__dir__ = osp.dirname(osp.dirname(osp.abspath(__file__)))
EXPERIMENT_DIR = osp.join(__dir__, "experiments")


@dataclass(frozen=True)
class RangeGenerator:
    min: int
    max: int

    def generate(self):
        return list(range(self.min, self.max))

    @staticmethod
    def parse(range_cfg: Dict[str, Any]) -> RangeGenerator:
        return RangeGenerator(**range_cfg)


@dataclass(frozen=True)
class ModelConfig:
    all_models: Sequence[single_model]

    @dataclass(frozen=True)
    class single_model:
        model_name: str
        model_params: Dict[str, Any]
        optim_params: Dict[str, Any] = field(default_factory=dict)
        hyperparameters: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return dict(
                model_name=self.model_name,
                model_params=self.model_params,
                optim_params=self.optim_params,
                hyperparameters=self.hyperparameters,
            )

    def generate(self):
        return [cfg.to_dict() for cfg in self.all_models]

    @staticmethod
    def parse(model_configs: Sequence[Dict[str, Any]]) -> ModelConfig:
        all_models = [ModelConfig.single_model(**cfg) for cfg in model_configs]
        return ModelConfig(all_models)


@dataclass(frozen=True)
class DataConfig:
    all_data: Sequence[single_data]

    @dataclass(frozen=True)
    class single_data:
        dataset: str
        labeled_sites: Sequence[Optional[Union[str, Sequence[str]]]]
        unlabeled_sites: Sequence[Optional[Union[str, Sequence[str]]]] = field(
            default=(None,)
        )
        num_unlabeled: Sequence[Optional[Union[str, Sequence[str]]]] = field(
            default=(None,)
        )
        output_directory: Optional[str] = field(default=None)

    def generate(self):
        return [
            dict(
                dataset=cfg.dataset,
                labeled_sites=labeled_sites,
                unlabeled_sites=unlabeled_sites,
                num_unlabeled=num_unlabeled,
                output_directory=cfg.output_directory,
            )
            for cfg in self.all_data
            for labeled_sites in cfg.labeled_sites
            for unlabeled_sites in cfg.unlabeled_sites
            for num_unlabeled in cfg.num_unlabeled
        ]

    @staticmethod
    def parse(data_configs: Sequence[Dict[str, Any]]) -> DataConfig:
        return DataConfig(
            [DataConfig.single_data(**cfg) for cfg in data_configs]
        )


@dataclass(frozen=True)
class ExperimentSettings:
    all_settings: Sequence[single_setting]

    @dataclass(frozen=True)
    class single_setting:
        ssl: bool = field(default=False)
        harmonize: bool = field(default=False)
        validation: bool = field(default=False)

        def to_dict(self) -> Dict[str, bool]:
            return dict(
                ssl=self.ssl,
                harmonize=self.harmonize,
                validation=self.validation,
            )

    def generate(self):
        return [cfg.to_dict() for cfg in self.all_settings]

    @staticmethod
    def parse(exp_settings: Sequence[Dict[str, bool]]) -> ExperimentSettings:
        return ExperimentSettings(
            [ExperimentSettings.single_setting(**cfg) for cfg in exp_settings]
        )


@dataclass(frozen=True)
class ProcessConfig:
    device: int = field(default=-1)
    verbose: bool = field(default=0)
    max_epoch: int = field(default=1000)
    patience: int = field(default=1000)
    dataloader_num_process: int = field(default=1)
    save_model_condition: Sequence[Dict[str, Any]] = field(default_factory=list)

    def match_save_model_condition(self, config: Dict[str, Any]):
        if not self.save_model_condition:
            return True
        for condition in self.save_model_condition:
            matched = True
            for key, value in condition.items():
                if key not in config:
                    matched = False
                elif value != config[key]:
                    matched = False
                if not matched:
                    break
            if matched:
                return True
        return False

    def update(self, config: Dict[str, Any]):
        config["device"] = self.device
        config["verbose"] = self.verbose
        config["max_epoch"] = self.max_epoch
        config["patience"] = self.patience
        config["dataloader_num_process"] = self.dataloader_num_process
        config["save_model"] = self.match_save_model_condition(config)
        return config


@dataclass(frozen=True)
class FrameworkConfigParser:
    seed: RangeGenerator
    fold: RangeGenerator
    model: ModelConfig
    data: DataConfig
    experiment_settings: ExperimentSettings
    process: ProcessConfig

    def generate(self):
        for model, data, exp_setting in product(
            self.model.generate(),
            self.data.generate(),
            self.experiment_settings.generate(),
        ):
            config = {
                "seed": self.seed.generate(),
                "fold": self.fold.generate(),
                **model,
                **data,
                **exp_setting,
            }
            config = self.process.update(config)
            yield config

    @staticmethod
    def parse(
        seed: Dict[str, int],
        fold: Dict[str, Any],
        model: Sequence[Dict[str, Any]],
        data: Sequence[Dict[str, Any]],
        experiment_settings: Sequence[Dict[str, bool]],
        process: Dict[str, Any],
    ) -> FrameworkConfigParser:
        return FrameworkConfigParser(
            seed=RangeGenerator.parse(seed),
            fold=RangeGenerator.parse(fold),
            model=ModelConfig.parse(model),
            data=DataConfig.parse(data),
            experiment_settings=ExperimentSettings.parse(experiment_settings),
            process=ProcessConfig(**process),
        )
