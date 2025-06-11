# Data Download 
We provide instructions for downloading our SSL4Eco dataset, as well as 
the datasets used for downstream evaluation.

First, obtain Google Earth Engine (GEE) authentication credentials by 
following the [installation instructions](https://developers.google.com/earth-engine/guides/python_install). Then, in your 
environment, authenticate to GEE:

```bash
earthengine authenticate --auth_mode=notebook
```

### Downloading SSL4Eco pretraining dataset 

Once authenticated to GEE, you can use the `download_ssl4eco.py` script 
to download SSL4Eco:

```bash
python download_ssl4eco.py \
    --csv_index_path ../index_files/ssl4eco_index_with_image_ids.csv \
    --save_path ./ssl4eco_dataset \
    --ncores 64
```

The script has a `--resume` option that resumes downloading 
based on a log of processed image ids. Ny default, such log is generated
as `checked_locations.csv` created in the `--save_path` folder.

💾 The resulting dataset comprises patches of 256 × 256 with 12-band 
Sentinel 2A and NDVI. The total size of the dataset is 1.5 TB.

⏳ On our machines, complete download takes 11.2h on 64 cores.


### Downloading downstream datasets
To download our arctic, biomes, or euforest datasets, use the following:

```bash
# arctic
python downstream_download_from_metadata.py  \
    --csv_index_path ../index_files/arctic_labels_with_image_ids.csv \
    --save_path ./arctic \
    --ncores 64 \
    --crop 264

# biomes
python downstream_download_from_metadata.py \
    --csv_index_path ../index_files/biomes_labels_with_image_ids.csv \
    --save_path ./biomes \
    --ncores 64 \
    --crop 264

# euforest
python downstream_download_from_metadata.py \
    --csv_index_path ../index_files/euforest_labels_with_image_ids.csv \
    --save_path ./euforest \
    --ncores 64 \
    --crop 264
```

⏳ On our machines, complete download of arctic/biomes datasets takes 
about 1h on 64 cores, and euforest takes 20min on 64 cores.
