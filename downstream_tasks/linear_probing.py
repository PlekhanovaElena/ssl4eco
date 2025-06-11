import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.random import sample_without_replacement
from tqdm import tqdm
import numpy as np


from utils import (
    load_model, load_dataset, get_embeddings, calculate_metric_means,
    EmbeddingDataset, calculate_clef_blind_submission, create_tsne)

from datasets.biomassters_dataset import BioMasstersDataset


def run_linear_probing(cfg, log):
    # LOAD MODEL
    model, transform, probe_input_dim = load_model(cfg.MODEL)

    # LOAD DATASET
    image_ds, num_dim, metric, loss_fn, task, main_metric, optimal_k = (
        load_dataset(cfg.DATASET, transform))
    loss_fn = loss_fn.to(cfg.DEVICE)
    print("Using metric:", metric)

    #TODO move this to the laod_dataset func once merged
    splits, split_idx = image_ds.splits, image_ds.split_idx
    
    # GET EMBEDDINGS (This can also be redone inside each iteration if
    # the embedding_retrieval is random)
    embeddings, labels = get_embeddings(
        image_ds,
        model,
        cfg.NUM_WORKERS,
        cfg.SAVE_NAME,
        cfg.DEVICE,
        cfg.USE_PRECOMPUTED_EMBEDDINGS)

    # Load the entire
    embedding_ds = EmbeddingDataset(embeddings, labels)

    # Define k-fold cross-validation
    torch.manual_seed(cfg.SEED)
    if splits:
        if cfg.DATASET == "bigearthnet":
            #pick a random 20% subset of the indices within the training set
            train_subsets = np.array([sample_without_replacement(
                                      split_idx[0],
                                      round(split_idx[0] * 0.11),
                                      random_state=cfg.SEED + i)
                                        for i in range(cfg.K_FOLDS)])

            kf_idx = zip(train_subsets, 
                        [np.arange(split_idx[0], split_idx[0]+split_idx[1])]*cfg.K_FOLDS)
        else:
            kf_idx = zip([np.arange(split_idx[0])]*cfg.K_FOLDS, 
                        [np.arange(split_idx[0], split_idx[0]+split_idx[1])]*cfg.K_FOLDS)

    else:
        if (
                task == "multi_label_classification"
                or task == "regression"
                or task == "distribution_regression"):
            kf = KFold(
                n_splits=cfg.K_FOLDS,
                shuffle=True,
                random_state=cfg.SEED)
        else:
            kf = StratifiedKFold(
                n_splits=cfg.K_FOLDS,
                shuffle=True,
                random_state=cfg.SEED)
        kf_idx = kf.split(embedding_ds, embedding_ds.labels)

    # K-FOLD CROSS-VALIDATION TRAINING LOGIC
    metric_values = []  # Adapt if multiple metrics are used
    for it, (train_idx, test_idx) in enumerate(kf_idx):

        # In case we only want to iterate over the first few folds, we
        # can exit here using cfg.NUM_USED_FOLDS
        if it >= cfg.NUM_USED_FOLDS:
            break

        # Prepare paths for storing predictions
        os.makedirs(cfg.SAVE_NAME, exist_ok=True)
        y_hat_path = f"{cfg.SAVE_NAME}/y_hat_{it}.npy"
        y_path = f"{cfg.SAVE_NAME}/y_{it}.npy"
        clef_submission_path = f"{cfg.SAVE_NAME}/blind/submission_{it}.csv"

        # Iterate multiple times. First seed torch and reset the probe
        if (
                cfg.USE_PRECOMPUTED_TEST_PREDICTIONS
                and os.path.isfile(y_hat_path)
                and os.path.isfile(y_path)):
            y_hat = torch.tensor(np.load(y_hat_path)).to(cfg.DEVICE)
            y = torch.tensor(np.load(y_path)).to(cfg.DEVICE)
            if it == 0:
                print("Using precomputed test predictions:")
                print("  Shape of y:", y.shape)
                print("  Shape of y_hat:", y_hat.shape)
            metric.reset()
            if cfg.DATASET == "clef":
                calculate_clef_blind_submission(clef_submission_path, y_hat)
                # Skip metric update since no labels available
                continue
            metric.update(y_hat.cpu(), y.cpu())
        else:
            print("Iteration:", it)

            torch.manual_seed(cfg.SEED + it)  # Set a new seed for each iteration
            probe = torch.nn.Linear(probe_input_dim, num_dim).to(cfg.DEVICE)
            optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=cfg.LEARNING_RATE)

            # Recover the kth test fold and split the remaining data
            # into train and val in a reproducible fashion
            torch.manual_seed(cfg.SEED)
            if "biomassters" in cfg.DATASET:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    embedding_ds, cfg.TRAIN_VAL_SPLIT)
                test_ds = BioMasstersDataset(transform, image_ds.mode, "test")
                test_embs, bl = get_embeddings(
                    test_ds,
                    model,
                    cfg.NUM_WORKERS,
                    f"{cfg.SAVE_NAME}/test",
                    cfg.DEVICE,
                    cfg.USE_PRECOMPUTED_EMBEDDINGS)
                test_dataset = EmbeddingDataset(test_embs, bl)
            elif cfg.DATASET == "clef":
                # We load CLEFBlind as test set, if the DATASET is CLEF
                train_dataset, val_dataset = torch.utils.data.random_split(
                    embedding_ds, cfg.TRAIN_VAL_SPLIT)  # Manually only keep a small subset as val
                blind_ds = load_dataset("clefblind", transform)[0]
                blind_embs, bl = get_embeddings(
                    blind_ds,
                    model,
                    cfg.NUM_WORKERS,
                    f"{cfg.SAVE_NAME}/blind",
                    cfg.DEVICE,
                    cfg.USE_PRECOMPUTED_EMBEDDINGS)
                test_dataset = EmbeddingDataset(blind_embs, bl)
            else:
                test_dataset = Subset(embedding_ds, test_idx)
                train_dataset, val_dataset = torch.utils.data.random_split(
                    Subset(embedding_ds, train_idx),
                    cfg.TRAIN_VAL_SPLIT)

            best_val_loss = 10000  # Set to high starting value
            early_stopping_counter = 0

            progress_bar = tqdm(range(cfg.EPOCHS))
            for epoch in progress_bar:

                # Shuffle the train data each epoch
                train_dl = DataLoader(
                    dataset=train_dataset,
                    batch_size=cfg.BATCH_SIZE,
                    num_workers=cfg.NUM_WORKERS,
                    shuffle=True)

                for x, y in train_dl:
                    x = x.to(cfg.DEVICE).to(torch.float32)
                    y = y.to(cfg.DEVICE)

                    optimizer.zero_grad()
                    y_hat = probe(x)
                    if y_hat.shape[-1] == 1:
                        y_hat = y_hat.squeeze(dim=-1)
                    loss = loss_fn(y_hat, y)
                    loss.backward()
                    optimizer.step()

                # Now calculate val_loss
                val_dl = DataLoader(
                    dataset=val_dataset,
                    batch_size=cfg.BATCH_SIZE,
                    num_workers=cfg.NUM_WORKERS,
                    shuffle=False)
                acc_loss = 0
                for x, y in val_dl:
                    x = x.to(cfg.DEVICE).to(torch.float32)
                    y = y.to(cfg.DEVICE)
                    with torch.no_grad():
                        y_hat = probe(x)
                    if y_hat.shape[-1] == 1:
                        y_hat = y_hat.squeeze(dim=-1)
                    acc_loss += loss_fn(y_hat, y).detach().cpu().item()
                acc_loss /= len(val_dl)

                # Calculate early stopping
                if best_val_loss - acc_loss > cfg.EARLY_STOPPING_SENSITIVITY:
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if best_val_loss > acc_loss:
                    best_val_loss = acc_loss
                progress_bar.set_description(
                    f"Val-loss: {acc_loss:.8f} (Best: {best_val_loss:.8f} - "
                    f"Patience: {early_stopping_counter})")

                # Check if over patience
                if early_stopping_counter >= cfg.EARLY_STOPPING_PATIENCE:
                    break

            # Now calculate the test metric
            test_dl = DataLoader(
                dataset=test_dataset,
                batch_size=cfg.BATCH_SIZE,
                num_workers=cfg.NUM_WORKERS,
                shuffle=False)

            metric.reset()
            acc_y = []
            acc_y_hat = []
            for x, y in test_dl:
                x = x.to(cfg.DEVICE).to(torch.float32)
                y = y.to(cfg.DEVICE)
                with torch.no_grad():
                    y_hat = probe(x)
                if y_hat.shape[-1] == 1:
                    y_hat = y_hat.squeeze(dim=-1)
                acc_y.append(y)
                acc_y_hat.append(y_hat)

            acc_y = torch.concat(acc_y, dim=0)
            acc_y_hat = torch.concat(acc_y_hat, dim=0)
            np.save(y_hat_path, acc_y_hat.detach().cpu())
            np.save(y_path, acc_y.detach().cpu())
            if cfg.DATASET == "clef":
                calculate_clef_blind_submission(clef_submission_path, acc_y_hat)
                # Skip metric update since no labels available
                continue
            metric.update(acc_y_hat.cpu(), acc_y.cpu())
        
        current_metric_value = metric.compute()
        metric_values.append(current_metric_value)
        print(f"Metrics for iteration {it}: {current_metric_value}")

    calculate_metric_means(metric_values, cfg, log)

    if cfg.SAVE_TSNE:
        create_tsne(cfg, embeddings, labels)
