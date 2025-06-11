import os
import torch
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import cv2
from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score, \
    MultilabelAUROC, MultilabelAccuracy, MulticlassAccuracy, MulticlassAUROC
from torchmetrics import MetricCollection


class BaseDataset(torch.utils.data.Dataset, ABC):
    index_file = None
    image_dir = None
    num_dim = None
    splits = False
    split_idx = None #expects [len(train), len(test)]
    index_key = 'ind'
    latitude_key = 'lat'
    longitude_key = 'lon'
    bands = [
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11',
        'B12', 'NDVI']
    patch_size = None
    replace_missing_sample_with_first_sample = False

    def __init__(self, transform):
        self.transform = transform
        self.samples, df = self._parse_image_dir()
        self.labels = self._read_labels(df)
        self.locations = self._read_location(df)
        self.metric = self._init_metric()
        self.loss_fn = self._init_loss()

    def _read_index_file(self):
        return pd.read_csv(self.index_file)

    def _index_to_sample_path(self, index):
        if len(str(index)) < 7:
            path = Path(self.image_dir) / np.char.zfill(str(index), 6)
        else:
            path = Path(self.image_dir) / str(index)
        return path

    def _parse_image_dir(self):
        """Returns the list of image patch directories and the
        corresponding DataFrame holding metadata.
        """
        # Read the data files
        df = self._read_index_file()
        indices = list(df[self.index_key])
        samples = []
        for index in indices:
            if len(str(index)) < 7:
                samples.append(Path(self.image_dir) / np.char.zfill(str(index), 6))
            else:
                samples.append(Path(self.image_dir) / str(index))
        samples = [self._index_to_sample_path(index) for index in indices]

        # Check all expected images are found
        valid_samples = []
        valid_indices = []
        invalid_count = 0
        for sample, index in zip(samples, indices):
            if os.path.isdir(sample):
                valid_samples.append(sample)
                valid_indices.append(index)
                continue
            invalid_count += 1
            if self.replace_missing_sample_with_first_sample:
                valid_samples.append(valid_samples[0])
                valid_indices.append(index)
        # Proceed with whatever is found and replace the missing images
        # with as many valid images
        if invalid_count > 0:
            print(
                f"Could not find all images listed in {self.index_file}. "
                f"Missing {invalid_count}/{len(indices)} images.")
        if self.replace_missing_sample_with_first_sample:
            print(
                f"Replaced missing images with as many copies of the first"
                f"image of the dataset. Total images: {len(valid_samples)}")

        return valid_samples, df[df[self.index_key].isin(valid_indices)]

    @property
    @abstractmethod
    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        pass

    def _read_location(self, df):
        """Extract longitude-latitude coordinates from a DataFrame."""
        return df[[self.longitude_key, self.latitude_key]].to_numpy()

    def _load_image(self, path):
        channels = []
        for b in self.bands:
            img_path = self._image_tif_path(path, b)

            # Dirty little move to fallback for sometimes-missing
            # patches for CLEFBlind. In this case we just take the
            # first image of the dataset
            try:
                ch = rasterio.open(img_path).read(1)
            except:
                if self.replace_missing_sample_with_first_sample:
                    first_img_path = self._image_tif_path(self.samples[0], b)
                    print(
                        f"WARNING: could not read {img_path}. Falling back to "
                        f"{first_img_path}.")
                    ch = rasterio.open(first_img_path).read(1)
                else:
                    raise ValueError(f"Could not read {img_path}.")

            ch = cv2.resize(
                ch,
                dsize=(self.patch_size, self.patch_size),
                interpolation=cv2.INTER_CUBIC)
            channels.append(ch.astype(np.float32))
        img = np.dstack(channels)
        return img

    def _image_tif_path(self, path, band):
        """Return the path to an image tif based on an image directory
        and the band name.
        """
        return Path(path) / f'{band}.tif'

    @abstractmethod
    def _init_metric(self):
        """Return the metric used for the downstream task. Expects a
        torchmetric-like object.
        """
        pass

    @abstractmethod
    def _init_loss(self):
        """Return the loss used for the downstream task."""
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self._load_image(self.samples[idx])
        if self.transform:
            image = self.transform(image)
        
        return image, self.locations[idx], self.labels[idx]


class BaseClassificationDataset(BaseDataset):

    @property
    def num_classes(self):
        """Alias for num_dim for classification datasets."""
        return self.num_dim

    def _init_metric(self):
        """Return the metric used for the downstream task. Expects a
        torchmetric-like object.
        """
        return MetricCollection({
            'macro_f1': MulticlassF1Score(
                self.num_dim,
                top_k=1,
                average='macro'),
            'micro_f1': MulticlassF1Score(
                self.num_dim,
                top_k=1,
                average='micro'),
            'weighted_f1': MulticlassF1Score(
                self.num_dim,
                top_k=1,
                average='weighted'),
            'macro_auroc': MulticlassAUROC(
                self.num_dim,
                average='macro'),
            'weighted_auroc': MulticlassAUROC(
                self.num_dim,
                average='weighted'),
            'macro_acc': MulticlassAccuracy(
                self.num_dim,
                top_k=1,
                average='macro'),
            'micro_acc': MulticlassAccuracy(
                self.num_dim,
                top_k=1,
                average='micro'),
            'weighted_acc': MulticlassAccuracy(
                self.num_dim,
                top_k=1,
                average='weighted'),
        })

    def _init_loss(self):
        """Return the loss used for the downstream task."""
        return torch.nn.CrossEntropyLoss()


class BaseMultiLabelClassificationDataset(BaseDataset):

    def _init_metric(self):
        """Return the metric used for the downstream task. Expects a
        torchmetric-like object.
        """
        return MetricCollection({
            'macro_f1': MultilabelF1Score(
                self.num_dim,
                threshold=0.5,
                average='macro'),
            'micro_f1': MultilabelF1Score(
                self.num_dim,
                threshold=0.5,
                average='micro'),
            'weighted_f1': MultilabelF1Score(
                self.num_dim,
                threshold=0.5,
                average='weighted'),
            'macro_auroc': MultilabelAUROC(
                self.num_dim,
                average='macro'),
            'micro_auroc': MultilabelAUROC(
                self.num_dim,
                average='micro'),
            'weighted_auroc': MultilabelAUROC(
                self.num_dim,
                average='weighted'),
        })

    def _init_loss(self):
        """Return the loss used for the downstream task."""
        # Alternative worth trying: torch.nn.BCEWithLogitsLoss()
        return  torch.nn.MultiLabelSoftMarginLoss()


class BaseRegressionDataset(BaseDataset):

    def __getitem__(self, idx):
        image, locations, labels = super().__getitem__(idx)
        return image, locations, labels.astype(np.float32)

    def _init_loss(self):
        """Return the loss used for the downstream task."""
        return torch.nn.MSELoss()
