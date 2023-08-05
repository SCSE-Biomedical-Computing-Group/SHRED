import torch
import numpy as np

from tqdm import tqdm
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from torch_sparse.tensor import SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def corr_mx_flatten(X):
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
