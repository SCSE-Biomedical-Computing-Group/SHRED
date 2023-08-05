import os.path as osp


__dir__ = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


MAIN_DIR = osp.abspath(osp.join(__dir__, "data", "Schiz"))
META_CSV_PATH = osp.join(MAIN_DIR, "meta.csv")
X_PATH = osp.join(MAIN_DIR, "X.npy")
X_TS_PATH = osp.join(MAIN_DIR, "X_ts.npy")
Y_PATH = osp.join(MAIN_DIR, "Y.npy")
SPLIT_TEST_PATH = osp.join(MAIN_DIR, "split_test.npy")
SPLIT_CV_PATH = osp.join(MAIN_DIR, "split_cv.npy")
SSL_SPLITS_DIR = osp.join(MAIN_DIR, "ssl_splits")
HARMONIZED_X_PATH = osp.join(MAIN_DIR, "harmonized_X.npy")