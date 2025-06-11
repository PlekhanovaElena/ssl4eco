# Downloads one of the downstream tasks: biomes, arctic CAVM or EU-forest
import ee
import os
import time
import pandas as pd
import csv
import rasterio
from rasterio.transform import Affine
from collections import OrderedDict
import numpy as np
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json


def center_crop(img, out_size):
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = (image_height - crop_height + 1) // 2
    crop_left = (image_width - crop_width + 1) // 2
    return img[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]


def get_properties(imagee):
    return imagee.getInfo()


def get_patch(image_id, center_coord, radius, band_names, crop, dtype: str = "float32"):
    image = ee.Image(image_id)
    region = (ee.Geometry.Point(center_coord).buffer(radius).bounds())

    # Resample all bands to 10m resolution
    patch = image.select(*band_names).resample('bilinear').reproject(crs=image.select('B2').projection(), scale=10).sampleRectangle(region, defaultValue=0)
    features = patch.getInfo()  # the actual download

    raster = OrderedDict()
    for ib,band in enumerate(band_names):
        img = np.atleast_3d(features["properties"][band])
        crop = [int(ci) for ci in crop]
        img = center_crop(img, out_size=[crop[ib],crop[ib]])
        raster[band] = img.astype(dtype)

    coords0 = np.array(features["geometry"]["coordinates"][0])
    coords = [[coords0[:, 0].min(), coords0[:, 1].max()],
              [coords0[:, 0].max(), coords0[:, 1].min()]]

    return {"raster": raster, "coords": coords, "metadata": get_properties(image)}


def save_geotiff(img, coords, filename):
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(
        coords[0][0] - xres / 2, coords[0][1] + yres / 2
    ) * Affine.scale(xres, -yres)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": channels,
        "crs": "+proj=latlong",
        "transform": transform,
        "dtype": img.dtype,
        "compress": "lzw",
        "predictor": 2,
    }
    with rasterio.open(filename, "w", **profile) as f:
        f.write(img.transpose(2, 0, 1))


def calculate_ndvi(nir_band, red_band):
    nir_band = nir_band.astype(np.float32)
    red_band = red_band.astype(np.float32)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 0.00001)
    return ndvi


def calc_ndvi_save_patch(raster, coords, metadata, patch_id, path, meta_path, dtype):
    nir_band = raster["B8"]
    red_band = raster["B4"]
    ndvi = calculate_ndvi(nir_band, red_band)
    ndvi[ndvi < 0] = 0
    ndvi = (ndvi * 10000).astype(dtype)
    
    with open(os.path.join(meta_path, f"{patch_id}.json"), "w") as f:
            json.dump(metadata, f)

    for band, img in raster.items():
        save_geotiff(img, coords, os.path.join(path, f"{band}.tif"))
    save_geotiff(ndvi, coords, os.path.join(path, "NDVI.tif"))


def process_row(row, save_path, radius, band_names, crop, dtype, verbose, ext_path):
    if verbose:
        print(row.iloc[0])
    center_coord = (row["lon"], row["lat"])
    mfolder = str(row.iloc[0]).zfill(6)
    location_path = os.path.join(save_path, "imgs", mfolder)
    meta_path = os.path.join(save_path, "metadata", mfolder)
    os.makedirs(location_path, exist_ok=True)
    os.makedirs(meta_path, exist_ok=True)

    image_id_column = 'image_id_1'
    image_id = "COPERNICUS/S2_SR_HARMONIZED/"+row[image_id_column]
    patch_id = os.path.basename(image_id)[:8] + image_id[-7:]

    patch = get_patch(image_id, center_coord, radius, band_names, crop, dtype)
        
    calc_ndvi_save_patch(
        raster=patch["raster"],
        coords=patch["coords"],
        metadata=patch["metadata"],
        patch_id=patch_id,
        path=location_path,
        meta_path=meta_path,
        dtype=dtype
        )
            
    try:
        with open(ext_path, "a", newline='') as f:
            writer = csv.writer(f)
            data = [row.iloc[0], center_coord[0], center_coord[1], 1]
            if verbose:
                print(data)
            writer.writerow(data)
    except Exception as e:
        print(f"An error occurred: {e}")


def process_images_parallel(df, save_path, radius, band_names, dtype, crop, ncores, resume_file, verbose):
    # Initialize dictionaries to store processed coordinates and their status
    ext_coords = {}
    ext_flags = {}

    # Load previously processed data if resume_file is provided
    ext_path = resume_file
    if resume_file and os.path.exists(resume_file):
        print("Resuming from the provided file")
        with open(resume_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
                ext_flags[key] = int(row[3])  # success or not
    else:
        ext_path = os.path.join(save_path, "checked_locations.csv")

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {}
        for i, row in df.iterrows():
            # Skip already processed rows based on resume data
            if str(row.iloc[0]) in ext_coords.keys() and ext_flags[str(row.iloc[0])] != 0: 
                continue
            
            futures[executor.submit(process_row, row, save_path, radius, 
                                    band_names, crop, dtype, verbose, ext_path)] = i

        start_time = time.time()
        for fi, future in enumerate(as_completed(futures)):
            if fi % 50 == 0:
                print(f"Downloaded {fi} locations in {time.time() - start_time:.3f}s.")
            future.result()


def main(csv_index_path, save_path, ncores, resume, crop_size, verbose):
    band_names = ['B1', 'B2', 'B3', 'B4', 
                  'B5', 'B6', 'B7', 'B8', 
                  'B8A', 'B9', 'B11', 'B12']
    
    dtype = "uint16"
    ee.Initialize()
    df = pd.read_csv(csv_index_path, low_memory=False)
    # Process the first 1000 rows
    #df = df.loc[:1000,]
    if round(crop_size / 6) != crop_size / 6:
        raise ValueError("Crop size should be divisible by 6")
    
    crop=[crop_size/6, crop_size, crop_size, crop_size, 
          crop_size/2, crop_size/2, crop_size/2, crop_size, 
          crop_size/2, crop_size/6, crop_size/2, crop_size/2]
    radius = crop_size*10/2 # 10m per pixel gives diameter, devided by 2 gives radius
    
    # Process images in parallel
    process_images_parallel(df, save_path, radius, band_names, dtype, crop, ncores, resume, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images from a CSV index.')
    parser.add_argument('--csv_index_path', type=str, help='Path to the CSV index file.')
    parser.add_argument('--save_path', type=str, help='Path to save the images')
    parser.add_argument('--ncores', type=int, help='Number of cores to use')
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument('--crop_size', type=int, help='Crop size of the image in pixels')
    parser.add_argument('--verbose', type=bool, default=False, help='Print image index')

    args = parser.parse_args()
    
    main(args.csv_index_path, args.save_path, args.ncores, args.resume, args.crop_size, args.verbose)
