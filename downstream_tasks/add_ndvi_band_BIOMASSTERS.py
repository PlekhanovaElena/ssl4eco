from os import listdir, makedirs
import numpy as np
import rasterio
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def add_ndvi(image_dir, file):
    if "S1" in file:
        return
    ras = rasterio.open(image_dir + file)
    meta = ras.meta.copy()
    ras = ras.read()
    zero_layer = np.zeros_like(ras[0])
    # B4 is the third layer (index 2)
    # B8 is the seventh layer (index 6)
    red_band = ras[2].astype(np.float32)
    nir_band = ras[6].astype(np.float32)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 0.00001)
    ndvi[ndvi < 0] = 0
    ndvi = (ndvi * 10000).astype(np.uint16)
    
    ras = np.stack([zero_layer, *ras[:8], zero_layer, *ras[8:-1], ndvi]) # Adding in B1 and B9 as zero each; Dropping the cloud probability; Adding ndvi at the end
    res_dir = image_dir[:-1] + "_processed/"
    meta["count"] = 13
    with rasterio.open(res_dir + file, "w", **meta) as f:
        f.write(ras)


def process_biomassters_images(image_directory):
    makedirs(image_directory[:-1] + "_processed/", exist_ok=True)
    files = listdir(image_directory)
    with ProcessPoolExecutor(max_workers=16) as executor:
        for ind in range(len(files)):
            executor.submit(add_ndvi, image_directory, files[ind])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_ndvi.py <biomassters_directory>")
        sys.exit(1)
    
    img_dir = sys.argv[1]

    # Validate directory
    if not os.path.isdir(img_dir):
        print(f"Error: The directory {img_dir} does not exist.")
        sys.exit(1)

    # Process images
    train_dir = img_dir + "train_features/"
    process_biomassters_images(train_dir)
    test_dir = img_dir + "test_features/"
    process_biomassters_images(test_dir)
