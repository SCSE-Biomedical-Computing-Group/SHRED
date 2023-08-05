from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch_geometric.data import Data

from utils.data import corr_mx_flatten


class Dataset(Enum):
    Schiz = "Schiz"


class DataloaderBase(ABC):
    def __init__(self, dataset: Dataset, harmonize: bool = False):
        self.dataset = dataset
        self.harmonize = harmonize
        self._init_dataset_()

    def _init_dataset_(self) -> Data:
        if self.dataset == Dataset.Schiz:
            from Schiz import load_data_fmri, get_ages_and_genders, get_sites
        else:
            raise NotImplementedError

        data: Tuple[np.ndarray] = load_data_fmri(harmonized=self.harmonize)
        self.X: np.ndarray = data[0]
        self.Y: np.ndarray = data[1].argmax(axis=1)
        self.X_flattened: np.ndarray = corr_mx_flatten(self.X)

        age_gender: Tuple[np.ndarray, np.ndarray] = get_ages_and_genders()
        age, gender = age_gender

        mean_age = np.nanmean(age)
        age = np.where(np.isnan(age), mean_age, age)
        age = np.expand_dims(age, axis=1)

        assert np.all(np.isnan(gender) | (gender >= 0) | (gender <= 1))
        gender = np.where(np.isnan(gender), np.nanmean(gender), gender)
        gender = np.expand_dims(gender, axis=1)

        self.age: np.ndarray = age
        self.gender: np.ndarray = gender
        self.sites: np.ndarray = get_sites()

    def _get_indices(
        self,
        seed: int = 0,
        fold: int = 0,
        ssl: bool = False,
        validation: bool = False,
        labeled_sites: Optional[Union[str, Sequence[str]]] = None,
        unlabeled_sites: Optional[Union[str, Sequence[str]]] = None,
        num_unlabeled: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if self.dataset == Dataset.Schiz:
            from Schiz import get_splits
        else:
            raise NotImplementedError

        indices_list = defaultdict(list)
        if labeled_sites is None or isinstance(labeled_sites, str):
            labeled_sites = [labeled_sites]
        for site_id in labeled_sites:
            splits = get_splits(site_id, test=validation)
            if validation:
                test_indices = splits[seed][0]
                labeled_train_indices, val_indices = splits[seed][1][fold]
                indices_list["labeled_train"].append(labeled_train_indices)
                indices_list["valid"].append(val_indices)
                indices_list["test"].append(test_indices)
            else:
                labeled_train_indices, test_indices = splits[seed][1][fold]
                indices_list["labeled_train"].append(labeled_train_indices)
                indices_list["test"].append(test_indices)

        indices = dict()
        for k, v in indices_list.items():
            if len(v) == 1:
                indices[k] = v[0]
            else:
                indices[k] = np.concatenate(v, axis=0)

        if ssl:
            if isinstance(unlabeled_sites, str):
                unlabeled_sites = [unlabeled_sites]
            unlabeled_indices = np.arange(len(self.X))
            if unlabeled_sites is not None:
                unlabeled_indices = unlabeled_indices[
                    np.isin(self.sites, unlabeled_sites)
                ]
            for idx in indices.values():
                unlabeled_indices = np.setdiff1d(unlabeled_indices, idx)
            if (
                num_unlabeled is not None
                and len(unlabeled_indices) > num_unlabeled
            ):
                unlabeled_indices = np.random.choice(
                    unlabeled_indices, num_unlabeled
                )
            indices["unlabeled_train"] = unlabeled_indices

        keys = list(indices.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert (
                    np.intersect1d(indices[keys[i]], indices[keys[j]]).size == 0
                )
        return indices

    @abstractmethod
    def load_split_data(
        self,
        seed: int = 0,
        fold: int = 0,
        ssl: bool = False,
        validation: bool = False,
        labeled_sites: Optional[Union[str, Sequence[str]]] = None,
        unlabeled_sites: Optional[Union[str, Sequence[str]]] = None,
        num_unlabeled: Optional[int] = None,
        num_process: int = 1,
    ) -> Union[
        Dict[str, Union[int, Data]], Dict[str, Union[int, Sequence[Data]]]
    ]:
        raise NotImplementedError

    @abstractmethod
    def load_all_data(
        self,
        sites: Optional[Union[str, Sequence[str]]] = None,
        num_process: int = 1,
    ) -> Union[
        Dict[str, Union[int, Data]], Dict[str, Union[int, Sequence[Data]]]
    ]:
        raise NotImplementedError


class ModelBaseDataloader(DataloaderBase):
    @staticmethod
    def make_dataset(
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        age: np.ndarray,
        gender: np.ndarray,
    ) -> Data:
        graph = Data()
        graph.x = torch.tensor(x).float()
        graph.y = torch.tensor(y)
        graph.d = torch.tensor(d)
        graph.age = torch.tensor(age).float()
        graph.gender = torch.tensor(gender).float()
        return graph

    def load_split_data(
        self,
        seed: int = 0,
        fold: int = 0,
        ssl: bool = False,
        validation: bool = False,
        labeled_sites: Optional[Union[str, Sequence[str]]] = None,
        unlabeled_sites: Optional[Union[str, Sequence[str]]] = None,
        num_unlabeled: Optional[int] = None,
        num_process: int = 1,
    ) -> Dict[str, Union[int, Data]]:
        indices = self._get_indices(
            seed,
            fold,
            ssl,
            validation,
            labeled_sites,
            unlabeled_sites,
            num_unlabeled,
        )

        if ssl:
            all_train_indices = np.concatenate(
                (indices["labeled_train"], indices["unlabeled_train"])
            )
        else:
            all_train_indices = indices["labeled_train"]
        le = LabelEncoder()
        le.fit(self.sites[all_train_indices])

        all_data: Dict[str, Data] = dict()
        for name, idx in indices.items():
            all_data[name] = self.make_dataset(
                x=self.X_flattened[idx],
                y=self.Y[idx],
                d=le.transform(self.sites[idx]),
                age=self.age[idx],
                gender=self.gender[idx],
            )

        all_data["input_size"] = int(self.X_flattened.shape[1])
        all_data["num_sites"] = int(len(le.classes_))

        empty = Data(x=torch.tensor([]))
        all_data["num_labeled_train"] = all_data.get(
            "labeled_train", empty
        ).x.size(0)
        all_data["num_unlabeled_train"] = all_data.get(
            "unlabeled_train", empty
        ).x.size(0)
        all_data["num_valid"] = all_data.get("valid", empty).x.size(0)
        all_data["num_test"] = all_data.get("test", empty).x.size(0)
        return all_data

    def load_all_data(
        self,
        sites: Optional[Union[str, Sequence[str]]] = None,
        num_process: int = 1,
    ) -> Dict[str, Union[int, Data]]:
        if isinstance(sites, str):
            sites = [sites]
        all_indices = np.arange(len(self.X))
        if sites is not None:
            all_indices = all_indices[np.isin(self.sites, sites)]

        le = LabelEncoder()
        le.fit(self.sites[all_indices])

        return {
            "data": self.make_dataset(
                x=self.X_flattened[all_indices],
                y=self.Y[all_indices],
                d=le.transform(self.sites[all_indices]),
                age=self.age[all_indices],
                gender=self.gender[all_indices],
            ),
            "input_size": int(self.X_flattened.shape[1]),
            "num_sites": int(len(le.classes_)),
        }
