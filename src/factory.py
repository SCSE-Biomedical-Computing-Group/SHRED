from abc import ABC, abstractclassmethod
from dataclasses import dataclass

import os
import sys
from typing import Any, Dict, Tuple, Type, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.base import ModelBase
from models import (
    FFN,
    VAE_FFN,
    SHRED,
    SHRED_I,
    SHRED_III,
)

from data import DataloaderBase, ModelBaseDataloader


class FrameworkFactory(ABC):
    @abstractclassmethod
    def load_model(
        cls, model_name: str, model_param: Dict[str, Any]
    ) -> Union[ModelBase, Tuple[ModelBase, ModelBase]]:
        raise NotImplementedError

    @abstractclassmethod
    def load_dataloader(
        cls, model_name: str, dataloader_param: Dict[str, Any]
    ) -> DataloaderBase:
        raise NotImplementedError


class SingleStageFrameworkFactory(FrameworkFactory):
    @dataclass
    class Mapping:
        model_cls: Type[ModelBase]
        dataloader_cls: Type[DataloaderBase]

    mapping = {
        "FFN": Mapping(FFN, ModelBaseDataloader),
        "VAE-FFN": Mapping(VAE_FFN, ModelBaseDataloader),
        "SHRED": Mapping(SHRED, ModelBaseDataloader),
        "SHRED-I": Mapping(SHRED_I, ModelBaseDataloader),
        "SHRED-III": Mapping(SHRED_III, ModelBaseDataloader),
    }

    @classmethod
    def get_model_class(cls, model_name: str) -> ModelBase:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        return model_mapping.model_cls

    @classmethod
    def load_model(
        cls, model_name: str, model_param: Dict[str, Any]
    ) -> ModelBase:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        return model_mapping.model_cls(**model_param)

    @classmethod
    def load_dataloader(
        cls, model_name: str, dataloader_param: Dict[str, Any]
    ) -> DataloaderBase:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        return model_mapping.dataloader_cls(**dataloader_param)

