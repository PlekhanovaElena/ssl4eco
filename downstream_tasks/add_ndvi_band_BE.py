import numpy as np
import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import rasterio

def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]
    
def calculate_ndvi(nir_band, red_band):
    nir_band = nir_band.astype(np.float32)
    red_band = red_band.astype(np.float32)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 0.00001)
    return ndvi

def make_path_band(dirname_loc, bandname_loc):
    return os.path.join(dirname_loc, os.path.basename(dirname_loc) + "_" + bandname_loc + ".tif")
    
def add_ndvi_band(dirname):
    filename_red = make_path_band(dirname, "B04")
    filename_nir = make_path_band(dirname, "B08")
    with rasterio.open(filename_red) as src:
         red_band = src.read(1)
    with rasterio.open(filename_nir) as src:
         nir_band = src.read(1)
    ndvi = calculate_ndvi(nir_band, red_band)
    ndvi[ndvi < 0] = 0
    ndvi = (ndvi * 10000).astype(np.uint16)
    
    # Write the NDVI band in separate file with same metadata
    meta = src.meta.copy()
    output_file = make_path_band(dirname, "NDVI")
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(ndvi,1)
        
def process_images(img_dir):
    # Get the list of image paths
    dirs = list_full_paths(img_dir)
    dirs.reverse()
    dirs = dirs[:450000]
    
    # Use ProcessPoolExecutor to run jobs in parallel
    with ProcessPoolExecutor(max_workers=16) as executor:
        # Use tqdm to display a progress bar
        list(tqdm(executor.map(add_ndvi_band, dirs), total=len(dirs), desc="Processing images"))
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_ndvi.py <image_directory>")
        sys.exit(1)
    
    img_dir = sys.argv[1]

    # Validate directory
    if not os.path.isdir(img_dir):
        print(f"Error: The directory {img_dir} does not exist.")
        sys.exit(1)

    # Process images
    process_images(img_dir)
