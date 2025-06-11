import torch
import numpy as np
from cvtorchvision import cvtransforms


class SelectBandsTransform(torch.nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.bands = bands

    def forward(self, img):
        return img[:,:, self.bands]


class SetBandToZero(torch.nn.Module):
    def __init__(self, band):
        super().__init__()
        self.band = band

    def forward(self, img):
        img[:,:, self.band] = 0
        return img


class NormTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, img):
        img = (img.astype(float) / 10000.0) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SetPatchSizeToPretraingSize(torch.nn.Module):
    def __init__(self, patch_size, interpolation='BICUBIC'):
        super().__init__()
        self.patch_size = patch_size
        self.interpolation = interpolation

    def forward(self, img):
        h, w, _ = img.shape
        if h < self.patch_size or w < self.patch_size:
            transform = cvtransforms.Resize(
                self.patch_size,
                interpolation=self.interpolation)
        else:
            transform = cvtransforms.CenterCrop(self.patch_size)
        return transform(img)


class MeanStdTransform(torch.nn.Module):
    # Taken from CROMA / SatMAE
    def __init__(self):
        super().__init__()
    def forward(self, img):

        _,_,c = img.shape
       
        imgs = []

        for channel in range(c):

            min_value = img[:, :, channel].mean() - 2 * img[:, :, channel].std()
            max_value = img[:, :, channel].mean() + 2 * img[:, :, channel].std() +1e-6

            transformed_img = (img[:, :, channel] - min_value) / (max_value - min_value) * 255.0
            transformed_img = np.clip(transformed_img, 0, 255)
            transformed_img = np.expand_dims(transformed_img, axis=-1)
            imgs.append(transformed_img)

        imgs = np.concatenate(imgs, axis=-1)
        return imgs


class SatlasNormRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, img):
        img[:,:,:3] = NormTransform()(img[:,:,:3])
        img[:,:,:3] = img[:,:,:3].astype(float) / 255.0

        return img


class SatlasNormOther(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, img):
        img[:,:,3:] = (img[:,:,3:].astype(float)) / 8160
        img[:,:,3:] = np.clip(img[:,:,3:], 0, 1)

        return img
    

class SentinelNormalize:
    """
    Taken from rshf SatMAE
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
