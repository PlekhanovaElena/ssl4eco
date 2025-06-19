# Downstream tasks
We provide scripts for reproducing our the pretraining of our SeCo-Eco 
and MoCo-Eco models on our SSL4Eco dataset.

> **Note**: If you have not done it already, make sure you downloaded 
the desired downstream datasets. See the 
[data download section](../data_download/README.md) for this. 

> **Note**: If you intend to use our pretrained SeCo-Eco model, make 
sure you downloaded the weights from 
[huggingface](https://huggingface.co/eplekh/secoeco) 🤗.

We use hydra for managing our script's arguments here. You can find the 
default config in [default.yaml](conf/default.yaml). The syntax for 
running the downstream evaluation of a model is simple: 

```bash
# Set the SECOECO_ROOT variable to indicate where the datasets can be 
# found and where the outputs will be saved
export SECOECO_ROOT=/root/path/to/your/datasets

# Evaluate SeCo-Eco on the biomes dataset, with linear probing
python main.py MODEL=secoeco DATASET=biomes PROBE=linear

# Evaluate Satlas on the arctic dataset, with k-NN
python main.py MODEL=satlas DATASET=arctic PROBE=knn
```

The complete list of supported models can be found [here](test_modules/__init__.py):

```bash
'ablcalendar'
'croma'
'dofabase'
'dofalarge'
'mocoeco'
'satlas'
'satmae'
'seco'
'secoeco'
'ssl4eo'
```

The complete list of supported datasets can be found [here](datasets/__init__.py):

```bash
'arctic'
'bigearthnet'
'biomassters'
'biomassterswinter'
'biomes'
'clef'
'clefblind'
'climate'
'euforest'
'euforestsp'
'treesatai'
```

## Downstream tasks preparation

### Biomes, Arctic, Climate, EUForest, EUForestSp

Please follow the instructions of [data download section](../data_download/README.md) for image download. The corresponding index iles are in the `index_files` folder. The datasets are ready to use.

### BigEarthNet 10%

To prepare the BigEarthNet 10% dataset, please download the images from [here](https://bigearth.net/v1.0.html) and place the `BigEarthNet-v1.0` folder into the `BE_full` folder. 

Copy the train-test split into the BE_full folder:

```bash
cp ./index_files/be10/bigearthnet-train.csv ./BE_full/bigearthnet-train.csv
cp ./index_files/be10/bigearthnet-test.csv ./BE_full/bigearthnet-test.csv
```

To add NDVI band to BigEarthNet imagery, run

```bash
python ./downstream_tasks/add_ndvi_band_BE.py ./BE_full/imgs
```

### CLEF, CLEF blind

The images for CLEF should be downloaded similar to [other datasets](../data_download/README.md) to the folders `sdm/clef/imgs` based on coordinates from `/index_files/sdm/clef_labels.csv` index file. The blind part of the competition should be downloaded into `sdm/clef_blind/imgs` based on coordinates from `/index_files/sdm/test_blind.csv` index file.

The final score can be obtained by submitting the resulting prediction into the official [Leaderboard](https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/leaderboard).

### Biomassters

Please download original dataset from [here](https://huggingface.co/datasets/nascetti-a/BioMassters) Then  add NDVI band with the `add_ndvi_band_BE.py` script. 












