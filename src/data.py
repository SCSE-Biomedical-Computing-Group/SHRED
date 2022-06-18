import os
import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Generator, Optional, Sequence, Union

import torch
from torch_geometric.data import Data


def corr_mx_flatten(X: np.ndarray) -> np.ndarray:
    """
    returns upper triangluar matrix of each sample in X

    option 1:
    X.shape == (num_sample, num_feature, num_feature)
    X_flattened.shape == (num_sample, num_feature * (num_feature - 1) / 2)

    option 2:
    X.shape == (num_feature, num_feature)
    X_flattend.shape == (num_feature * (num_feature - 1) / 2,)
    """
    upper_triangular_idx = np.triu_indices(X.shape[1], 1)
    if len(X.shape) == 3:
        X = X[:, upper_triangular_idx[0], upper_triangular_idx[1]]
    else:
        X = X[upper_triangular_idx[0], upper_triangular_idx[1]]
    return X


def combat_harmonization(X: np.ndarray, meta_df: pd.DataFrame) -> np.ndarray:
    covars = meta_df[["SITE", "AGE", "GENDER"]]
    categorical_cols = ["GENDER"]
    continuous_cols = ["AGE"]
    batch_col = "SITE"
    combat = neuroCombat(
        dat=X.T,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
    )
    harmonized_X = combat["data"].T
    harmonized_X = np.clip(harmonized_X, -1, 1)
    return harmonized_X


def split_kfoldcv_sbj(
    y: np.ndarray, subjects: np.ndarray, num_fold: int, seed: int
):
    unique_subjects, first_subject_index = np.unique(
        subjects, return_index=True
    )
    subject_y = y[first_subject_index]
    subject_X = np.zeros_like(subject_y)
    skfold = StratifiedKFold(n_splits=num_fold, random_state=seed, shuffle=True)

    result = []
    for train_subject_index, test_subject_index in skfold.split(
        subject_X, subject_y
    ):
        train_subjects = unique_subjects[train_subject_index]
        test_subjects = unique_subjects[test_subject_index]
        train_index = np.argwhere(np.isin(subjects, train_subjects)).flatten()
        test_index = np.argwhere(np.isin(subjects, test_subjects)).flatten()
        assert len(np.intersect1d(train_index, test_index)) == 0
        assert (
            len(np.intersect1d(subjects[train_index], subjects[test_index]))
            == 0
        )
        result.append((train_index, test_index))
    return result


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


class Dataset:
    """
    required columns in meta.csv
    - SUBJECT
    - AGE
    - GENDER
    - SITE
    - DISEASE_LABEL
    - FC_MATRIX_PATH
    """

    def __init__(self, data_csv_path: str, name: str, harmonize: bool = False):
        self.data_csv_path = os.path.abspath(data_csv_path)
        self.data_folder = os.path.dirname(self.data_csv_path)
        self.name = name
        self.harmonize = harmonize
        self._init_properties_()

    def _init_properties_(self):
        meta_df = pd.read_csv(self.data_csv_path)
        X = np.array(
            [
                np.load(os.path.join(self.data_folder, path))
                for path in meta_df["FC_MATRIX_PATH"]
            ]
        )
        X = corr_mx_flatten(np.nan_to_num(X))
        if self.harmonize:
            self.X = combat_harmonization(X, meta_df)
        else:
            self.X = X

        self.subjects = meta_df["SUBJECT"].values
        self.ages = meta_df["AGE"].values
        self.genders = meta_df["GENDER"].values
        self.sites = meta_df["SITE"].values
        self.y = meta_df["DISEASE_LABEL"].values

    def _get_indices(
        self,
        seed: int = 0,
        num_fold: int = 5,
        ssl: bool = False,
        labeled_sites: Optional[Union[str, Sequence[str]]] = None,
    ) -> Generator[Dict[str, np.ndarray], None, None]:

        if labeled_sites is None:
            is_labeled = np.ones(len(self.sites), dtype=bool)
        else:
            if isinstance(labeled_sites, str):
                labeled_sites = [labeled_sites]
            is_labeled = np.isin(self.sites, labeled_sites)

        unlabeled_indices = np.argwhere(~is_labeled).flatten()
        labeled_indices = np.argwhere(is_labeled).flatten()
        for train, test in split_kfoldcv_sbj(
            self.y[is_labeled], self.subjects[is_labeled], num_fold, seed
        ):
            result = {
                "labeled_train": labeled_indices[train],
                "test": labeled_indices[test],
            }
            if ssl:
                result["unlabeled_train"] = unlabeled_indices
            yield result

    def load_split_data(
        self,
        seed: int = 0,
        num_fold: int = 5,
        ssl: bool = False,
        labeled_sites: Optional[Union[str, Sequence[str]]] = None,
    ) -> Generator[Dict[str, Union[int, Data]], None, None]:
        for indices in self._get_indices(seed, num_fold, ssl, labeled_sites):
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
                all_data[name] = make_dataset(
                    x=self.X[idx],
                    y=self.y[idx],
                    d=le.transform(self.sites[idx]),
                    age=self.ages[idx],
                    gender=self.genders[idx],
                )

            all_data["input_size"] = int(self.X.shape[1])
            all_data["num_sites"] = int(len(le.classes_))

            empty = Data(x=torch.tensor([]))
            all_data["num_labeled_train"] = all_data.get(
                "labeled_train", empty
            ).x.size(0)
            all_data["num_unlabeled_train"] = all_data.get(
                "unlabeled_train", empty
            ).x.size(0)
            all_data["num_test"] = all_data.get("test", empty).x.size(0)
            yield all_data

    def load_all_data(
        self, sites: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, Union[int, Data]]:
        if isinstance(sites, str):
            sites = [sites]
        all_indices = np.arange(len(self.X))
        if sites is not None:
            all_indices = all_indices[np.isin(self.sites, sites)]

        le = LabelEncoder()
        le.fit(self.sites[all_indices])

        return {
            "data": make_dataset(
                x=self.X[all_indices],
                y=self.y[all_indices],
                d=le.transform(self.sites[all_indices]),
                age=self.ages[all_indices],
                gender=self.genders[all_indices],
            ),
            "input_size": int(self.X.shape[1]),
            "num_sites": int(len(le.classes_)),
        }
