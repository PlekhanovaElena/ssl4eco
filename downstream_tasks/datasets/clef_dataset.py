import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC
from torchmetrics import MetricCollection

from downstream_tasks.utils.root import get_root
from downstream_tasks.datasets.base import BaseMultiLabelClassificationDataset


class CLEFDataset(BaseMultiLabelClassificationDataset):
    index_file = get_root() + "/index_files/sdm/clef_labels.csv"
    image_dir = get_root() + "/sdm/clef/imgs"
    index_key = 'patchID'
    num_dim = 2174
    patch_size = 108

    def _index_to_sample_path(self, index):
        if len(str(index)) < 7:
            path = Path(self.image_dir) / np.char.zfill(str(index), 6)
        else:
            path = Path(self.image_dir) / str(index)
        return path

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        return [np.array(row[1][3:]).astype(np.float32) for row in df.iterrows()]

    def _init_loss(self):
        """Return the loss used for the downstream task."""
        # Alternative worth trying: torch.nn.BCEWithLogitsLoss()
        # return  torch.nn.MultiLabelSoftMarginLoss()
        pos_weight = 12 # 4 8 16 32 64 128 256
        print("Using non-default BCEWithLogitsLoss; Pos weight:", pos_weight)
        # torch.nn.BCEWithLogitsLoss() BCELoss
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([2174]) * pos_weight)


class CLEFBlindDataset(CLEFDataset):
    index_file = get_root() + "/index_files/sdm/test_blind.csv"
    image_dir = get_root() + "/sdm/clef_blind/imgs"
    replace_missing_sample_with_first_sample = True

    def _read_index_file(self):
        return pd.read_csv(self.index_file, sep=";", header='infer', low_memory=False)

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        return [1.0 for _ in df.iterrows()]
