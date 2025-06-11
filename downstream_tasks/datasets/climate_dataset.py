import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics import MetricCollection

from downstream_tasks.utils.root import get_root
from downstream_tasks.metrics import UnnormalizedMetric, PerDimMetric
from downstream_tasks.datasets.base import BaseRegressionDataset


class Climate4Dataset(BaseRegressionDataset):
    index_file = get_root() + "/index_files/climate_labels_50k.csv"
    image_dir = get_root() + "/biomes50k/imgs"
    num_dim = 4
    patch_size = 264
    chelsa_channels = ["temp", "prec", "evap", "swb"]

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        labels = df[self.chelsa_channels].to_numpy()

        # Gaussian normalization
        mean = labels.mean(axis=0)
        std = labels.std(axis=0)
        labels = (labels - mean) / std

        # Store the mean and variance in the attributes, to be used for
        # metrics construction
        self.mean = mean
        self.std = std

        return labels

    def _init_metric(self):
        """Return the metric used for the downstream task. Expects a
        torchmetric-like object.
        """
        return MetricCollection({
            'mae': UnnormalizedMetric(
                PerDimMetric(MeanAbsoluteError, self.chelsa_channels, own_name="mae"),
                self.mean,
                self.std),
            'rmse': UnnormalizedMetric(
                PerDimMetric(MeanSquaredError, self.chelsa_channels, squared=False, own_name="rmse"),
                self.mean,
                self.std),
            'r2': UnnormalizedMetric(
                PerDimMetric(R2Score, self.chelsa_channels, own_name="r2"),
                self.mean,
                self.std),
            'total normalized mae': MeanAbsoluteError(),
            'total normalized rmse': MeanSquaredError(squared=False, num_outputs=len(self.chelsa_channels)),
            'total normalized r2': R2Score(),
        })


class Climate4WinterDataset(Climate4Dataset):
    image_dir = get_root() + "/biomes_winter50k/imgs"
