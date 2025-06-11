import numpy as np

from downstream_tasks.utils.root import get_root
from downstream_tasks.datasets.base import BaseClassificationDataset


class BiomesDataset(BaseClassificationDataset):
    index_file = get_root() + "/index_files/biomes_labels_50k.csv"
    image_dir = get_root() + "/biomes50k/imgs"
    num_dim = 15
    patch_size = 264

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        unique_labels = np.unique(list(df["biome"]))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_index[label] for label in list(df["biome"])]
        return labels


class BiomesWinterDataset(BiomesDataset):
    image_dir = get_root() + "/biomes_winter50k/imgs"
