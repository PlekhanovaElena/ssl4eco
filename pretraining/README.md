# Pretraining
We provide scripts for reproducing our the pretraining of our SeCo-Eco 
and MoCo-Eco models on our SSL4Eco dataset.

⏳ For the record, on our A100 GPU, the full pretraining of SeCo-Eco 
takes about 7-8 days, while MoCo-Eco takes about 4 days.

### SeCo-Eco pretraining
Make sure you adjust `--data` on your machine, based on where you 
[downloaded SSL4Eco](../data_download/README.md).

```bash
python -u pretrain_seco_3heads.py \
    --data /path/to/ssl4eco/images \  # adjust to your needs
    --workers 12 \
    --bands B9 \
    --batch-size 256 \
    --epochs 100 \
    --log seco \
    --seed 0 \
    --dtype uint16 \
    --moco-t 0.2 \
    --moco-k 65536
```

### MoCo-Eco pretraining
Make sure you adjust `--data` on your machine, based on where you 
[downloaded SSL4Eco](../data_download/README.md).

```bash
python -u pretrain_moco_contrast.py \
    --data /path/to/ssl4eco/images \  # adjust to your needs
    --checkpoints moco \
    --gpu 0 \
    --bands B9 \
    --arch resnet50 \
    --workers 12 \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.03 \
    --mlp \
    --moco-t 0.2 \
    --aug-plus \
    --cos \
    --seed 0 \
    --mode s2c \
    --dtype uint16 \
    --season augment \
    --in_size 224 \
    --log moco
```
