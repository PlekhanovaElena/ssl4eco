import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from pyproj import Proj as pp_proj
from pyproj import transform as pp_transform
from pyproj import Transformer
import glob

from torchmetrics.classification import MultilabelAveragePrecision, \
    MultilabelF1Score
from torchmetrics import MetricCollection
import os
from downstream_tasks.datasets.base import BaseMultiLabelClassificationDataset
from downstream_tasks.utils.root import get_root



LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


class BE10Dataset(BaseMultiLabelClassificationDataset):
    index_file_train = get_root() + "/BE_full/bigearthnet-train.csv"
    index_file_test = get_root() + "/BE_full/bigearthnet-test.csv"
    
    image_dir = get_root() + "/BE_full/BigEarthNet-v1.0"
    index_file_snow = get_root()+ "/index_files/be10/patches_with_seasonal_snow.csv"
    index_file_cloud = get_root() + "/index_files/be10/patches_with_cloud_and_shadow.csv"
    
    splits = True
    num_dim = 19
    bands = [
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09',
        'B11', 'B12', 'NDVI']
    patch_size = 128

    def _parse_image_dir(self):
        """Returns the list of image patch directories and the
        corresponding DataFrame holding metadata.
        """

        samples, self.split_idx = self.merge_index_files([self.index_file_train, self.index_file_test])
        return samples, None

    def _read_labels(self, df):
        """Extract labels from a DataFrame (optimized single-threaded)."""
        print("Reading labels")

        labels = [None] * len(self.samples)  # Preallocate list size

        for i, file in enumerate(tqdm(self.samples)):
            path = f"{file}/{os.path.basename(file)}_labels_metadata.json"  # Avoid os.path.join overhead

            with open(path, 'r') as f:
                labels[i] = self.get_multihot_new(json.load(f)['labels'])  # Inline function call

        return labels
  

    def _read_location(self, df):
        """Extract longitude-latitude coordinates from a DataFrame (optimized)."""
        print("Reading locations")

        locations = np.empty((len(self.samples), 2), dtype=np.float64)  # Preallocate memory
        transformers = {}  # Cache transformers

        for i, file in enumerate(tqdm(self.samples)):
            path = f"{file}/{os.path.basename(file)}_labels_metadata.json"

            with open(path, 'rb') as f:  # Use binary mode for orjson
                metadata = json.loads(f.read())
            zone = int(metadata['projection'].split("zone ")[1].split("N")[0])
            if zone not in transformers:
                transformers[zone] = Transformer.from_crs(f"EPSG:326{zone}", "EPSG:4326", always_xy=True)
            locations[i] = transformers[zone].transform(metadata['coordinates']['ulx'], metadata['coordinates']['uly'])

        return locations


    def _image_tif_path(self, path, band):
        """Return the path to an image tif based on an image directory
        and the band name.
        """
        path = Path(path)
        return path / f'{path.name}_{band}.tif'

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

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.int64)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target

    @staticmethod
    def merge_index_files(files):
        samples = []
        split_idx = []
        for i, file in enumerate(files):
            with open(file, 'r') as f:
                dir_list = np.array([os.path.join(os.path.dirname(file),"BigEarthNet-v1.0/",line.split(",")[0].strip()) for line in f.readlines()])
                samples.append(dir_list)
                split_idx.append(len(samples[i]))
        return np.concatenate(samples), split_idx
    

