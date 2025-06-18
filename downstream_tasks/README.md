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
'ablb9'
'ablb9weights'
'ablb12'
'ablb12n'
'ablb12weights'
'ablb12nweights'
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
'biomeswinter'
'clef'
'clefblind'
'climate'
'climatewinter'
'euforest'
'euforestsp'
'euforestwinter'
'treesatai'
```
