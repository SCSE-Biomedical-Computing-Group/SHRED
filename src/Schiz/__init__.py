import os
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schiz_config import *


def load_data_fmri(harmonized=False, time_series=False):
    """
    Inputs
    site_id: str or None
        - if specified, splits will only contains index of subjects from the specified site
    harmonized: bool
        - whether or not to return combat harmonized X
    
    Returns
    X: np.ndarray with N subject samples, each sample consists of a ROI x ROI correlation matrix
    Y: np.ndarray with N subject samples, each sample has a one-hot encoded label (0: Normal, 1: Diseased)
    """
    if harmonized:
        X = np.load(HARMONIZED_X_PATH)
    else:
        X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    if not time_series:
        return X, Y
    raise NotImplementedError


def get_splits(site_id=None, test=False):
    if site_id is None:
        path = SPLIT_TEST_PATH if test else SPLIT_CV_PATH
    else:
        path = "{}_test.npy".format(site_id) if test else "{}_cv.npy".format(site_id)
        path = os.path.join(SSL_SPLITS_DIR, path)
    splits = np.load(path, allow_pickle=True)
    return splits


def get_ages_and_genders():
    """
    ages: np.array of float representing the age of subject when the scan is obtained
    gender: np.array of int representing the subject's gender
        - 0: Female
        - 1: Male
    """
    meta_df = pd.read_csv(META_CSV_PATH)
    ages = np.array(meta_df["age"])
    genders = np.array(meta_df["sex"])
    return ages, genders


def get_sites():
    meta_df = pd.read_csv(META_CSV_PATH)
    sites = np.array(meta_df["study"])
    return sites


def load_meta_df():
    df =  pd.read_csv(META_CSV_PATH)
    df["FILE_PATH"] = df["FILE_PATH"].apply(eval)
    return df

