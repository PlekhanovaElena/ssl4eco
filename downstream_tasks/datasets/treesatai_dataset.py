import numpy as np
from pathlib import Path
import os
import json
import rasterio
from pyproj import Proj as pp_proj
from pyproj import transform as pp_transform
from torchmetrics.classification import MultilabelAveragePrecision, \
    MultilabelF1Score
from torchmetrics import MetricCollection

from downstream_tasks.utils.root import get_root
from downstream_tasks.datasets.base import BaseMultiLabelClassificationDataset


LABELS = ["Abies",
          "Acer",
          "Alnus",
          "Betula",
          "Cleared",
          "Fagus",
          "Fraxinus",
          "Larix",
          "Picea",
          "Pinus",
          "Populus",
          "Prunus",
          "Pseudotsuga",
          "Quercus",
          "Tilia"
          ]



class TreeSatAIDataset(BaseMultiLabelClassificationDataset):
    index_file_train = get_root() + "/TreeSatAI_orig/train_filenames.lst"
    index_file_test =  get_root() + "/TreeSatAI_orig/test_filenames.lst"
    image_dir = get_root()  + "/TreeSatAI_orig/s2/60m"
    label_file = get_root() + "/TreeSatAI_orig/labels/TreeSatBA_v9_60m_multi_labels.json"
    location_file =  get_root() + "/TreeSatAI_orig/geojson/p.GeoJSON"
    splits = True
    num_dim = 15
    patch_size = 6

    def _parse_image_dir(self):
        """Returns the list of image patch directories and the
        corresponding DataFrame holding metadata.
        """
        self.samples, self.split_idx = self.merge_index_files([self.index_file_train, self.index_file_test])
        return self.samples, None

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        #NOTE Code taken from Omnisat
        with open(self.label_file) as file:
            data = json.load(file)
            data_dict = {os.path.basename(file): data[os.path.basename(file)] for file in self.samples}
            labels = self.filter_labels_by_threshold(data_dict, 0.00)
            lines = list(labels.keys())
            y = np.zeros((len(lines), self.num_dim))
            for i, line in enumerate(lines):
                for u in labels[line]:                    
                    y[i][LABELS.index(u)] = 1
        return y.astype(np.int32)

    def _read_location(self, df):
        """Extract longitude-latitude coordinates from a DataFrame."""
        with open(self.location_file) as file:
            data = json.load(file)
            extracted_order = {str(feat["properties"]["IMG_ID"]) + ".tif": i for i, feat in enumerate(data["features"])}
            locations = np.array([feat["geometry"]["coordinates"] for feat in data["features"]])
            sorted_indices = np.array([extracted_order[os.path.basename(name)] for name in self.samples])
            locations = locations[sorted_indices]
        return locations


    def _init_metric(self):
        """Return the metric used for the downstream task. Expects a
        torchmetric-like object.
        """
        return MetricCollection({
            'micro_map': MultilabelAveragePrecision(
                self.num_dim, average='micro'),
            'macro_map': MultilabelAveragePrecision(
                self.num_dim, average='macro'),
            'macro_f1': MultilabelF1Score(
                self.num_dim, threshold=0.5, average='macro'),
            'micro_f1': MultilabelF1Score(
                self.num_dim, threshold=0.5, average='micro'),
        })

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = self._load_image(path)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        location = self.locations[idx]
        return image, location, label
    
    def _load_image(self, path):
        img = rasterio.open(path).read().astype(np.float32)
        ndvi = rasterio.open(path[:-4]+"_NDVI.tif").read().astype(np.float32)

        img = np.concatenate([img, ndvi], axis=0)
        img = np.transpose(img, (1,2,0))
        return img


    @staticmethod
    def merge_index_files(files):
        samples = []
        split_idx = []
        for i, file in enumerate(files):
            with open(file, 'r') as f:
                samples.append(np.array([os.path.join(os.path.dirname(file),"s2/60m/",line.strip()) for line in f.readlines()]))
                split_idx.append(len(samples[i]))
        return np.concatenate(samples), split_idx
    
    @staticmethod
    def filter_labels_by_threshold(labels_dict, area_threshold = 0.07):
        """
        Parameters
        ----------
        labels_dict: dict, {filename1: [(label, area)],
                            filename2: [(label, area), (label, area)],
                            ...
                            filenameN: [(label, area), (label, area)]}
        area_threshold: float
        
        Returns
        -------
        filtered: dict, {filename1: [label],
                        filename2: [label, label],
                        ...
                        filenameN: [label, label]}
        """
        filtered = {}
        for img in labels_dict:
            for lbl, area in labels_dict[img]:
                # if area greater than threshold we keep the label
                if area > area_threshold:
                    # init the list of labels for the image
                    if img not in filtered:
                        filtered[img] = []
                    # add only the label, since we won't use area information further
                    filtered[img].append(lbl)   
        return filtered



