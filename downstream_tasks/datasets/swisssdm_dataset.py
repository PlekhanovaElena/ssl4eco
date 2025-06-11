import numpy as np

from downstream_tasks.utils.root import get_root
from downstream_tasks.datasets.base import BaseClassificationDataset


class SwissSDMBaseDataset(BaseClassificationDataset):
    index_file = None
    image_dir = None
    num_dim = 2
    patch_size = 108

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        unique_labels = np.unique(list(df["label"]))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_index[label] for label in list(df["label"])]
        return labels


class SwissSDMccDataset(SwissSDMBaseDataset):
    index_file = get_root() + "/index_files/sdm/cc_labels.csv"
    image_dir = get_root() + "/sdm/cc/imgs"


class SwissSDMciDataset(SwissSDMBaseDataset):
    index_file = get_root() + "/index_files/sdm/ci_labels.csv"
    image_dir = get_root() + "/sdm/ci/imgs"


class SwissSDMldDataset(SwissSDMBaseDataset):
    index_file = get_root() + "/index_files/sdm/ld_labels.csv"
    image_dir = get_root() + "/sdm/ld/imgs"
