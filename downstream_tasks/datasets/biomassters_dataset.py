import torch
import numpy as np
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
from os import listdir

from downstream_tasks.utils.root import get_root
from downstream_tasks.loss import NormalizingKLDivLoss
from downstream_tasks.metrics import JSDivergence

from torchmetrics import MetricCollection
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from downstream_tasks.metrics import LogitMetric, PerDimMetric

class BioMasstersDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, mode="summer", split="train"):
        self.transforms = transforms

        self.agbm_bin_bounds = [0., 4.57000017, 21.78000069, 45.63999939, 66.72000122, 87.45999908, 113.27999878, 156.8999939, 500.]
        #self.metric = JSDivergence()
        self.metric = MetricCollection({
            'mae': LogitMetric(
                PerDimMetric(MeanAbsoluteError, self.agbm_bin_bounds[1:], own_name="mae")),
            'rmse': LogitMetric(
                PerDimMetric(MeanSquaredError, self.agbm_bin_bounds[1:], own_name="rmse", squared=False)),
            'r2': LogitMetric(
                PerDimMetric(R2Score, self.agbm_bin_bounds[1:], own_name="r2")),
            'total mae': LogitMetric(MeanAbsoluteError()),
            'total rmse': LogitMetric(MeanSquaredError(squared=False, num_outputs=len(self.agbm_bin_bounds)-1)),
            'total r2': LogitMetric(R2Score()),
        })
        self.loss_fn = NormalizingKLDivLoss(reduction= "batchmean")

        self.mode = mode

        if mode == "july":
            months = ["10"]
        elif mode == "january":
            months = ["04"]
        elif mode == "summer":
            months = ["09", "10", "11"]
        elif mode == "autumn":
            months = ["12", "02", "01"]
        elif mode == "winter":
            months = ["03", "04", "05"]
        elif mode == "spring":
            months = ["06", "07", "08"]
        elif mode == "all":
            months = [""]
        else:
            raise NotImplementedError("Unknown mode: " + mode)

        root = get_root() + "/BioMassters/"
        self.features_dir = root + split + "_features_processed/"
        self.agbm_dir = root + split + "_agbm/"

        self.files = []
        for m in months:
            self.files += [chip_id for chip_id in listdir(self.features_dir) if ("S2_" + m) in chip_id]
        
        if split == "train":
            print("Using", mode, "images:", len(self.files))

        # Decision how to deal with the values
        self.agbm_clip = 500
        self.num_dim = len(self.agbm_bin_bounds) - 1

        self.splits = None
        self.split_idx = None

    def __len__(self):
        return len(self.files)

    def _load_image(self, file):
        img = rasterio.open(self.features_dir + file).read().astype(np.float32)
        return img

    def _load_label(self, chip_id):
        agbm = rasterio.open(self.agbm_dir + chip_id + "_agbm.tif").read(1)
        agbm[agbm > self.agbm_clip] = self.agbm_clip
        label = np.histogram(agbm.flatten(), self.agbm_bin_bounds)[0] / len(agbm.flatten())
        label = label + 1e-5 # Need this to avoid zero case
        return label

    def __getitem__(self, idx):
        file = self.files[idx]
        image = np.transpose(self._load_image(file), (1, 2, 0))
        if self.transforms:
            image = self.transforms(image)
        label = self._load_label(file.split("_")[0])

        return image, torch.tensor([float('nan')]), label.astype(np.float32)


class BioMasstersWinterDataset(BioMasstersDataset):
    def __init__(self, transforms, split="train"):
        super().__init__(transforms, mode="winter", split=split)
