import numpy as np
import os

from downstream_tasks.utils.root import get_root
from downstream_tasks.datasets.base import BaseClassificationDataset


class ArcticDataset(BaseClassificationDataset):
    index_file = get_root() + "/index_files/arctic50k_labels.csv"
    image_dir = get_root() + "/arctic50k/imgs"
    num_dim = 5
    patch_size = 264

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        unique_labels = np.unique(list(df["five_types"]))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_index[label] for label in list(df["five_types"])]
        return labels
