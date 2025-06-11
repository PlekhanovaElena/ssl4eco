import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.random import sample_without_replacement
from torch.nn.functional import one_hot
import numpy as np

from downstream_tasks.utils import load_model, load_dataset, get_embeddings, \
    calculate_metric_means, EmbeddingDataset, calculate_clef_blind_submission
from downstream_tasks.datasets.biomassters_dataset import BioMasstersDataset


def run_knn_probing(cfg, log):
    if not isinstance(cfg.K_PROBE, int):
        raise ValueError(
            f"Cannot do k-NN with k={cfg.K_PROBE}. Expected an int. If "
            f"K_PROBE<1 is passed, the known optimal value of k for the "
            f"dataset will be used")
    if cfg.TEMP_PROBE <= 0:
        raise ValueError(f"Cannot do a softmax with temperature={cfg.TEMP_PROBE}")

    # LOAD MODEL
    model, transform, probe_input_dim = load_model(cfg.MODEL)
    # LOAD DATASET
    image_ds, num_dim, metric, loss_fn, task, main_metric, optimal_k = (
        load_dataset(cfg.DATASET, transform))
    loss_fn = loss_fn.to(cfg.DEVICE)
    print("Using metric:", metric)

    # Update k for k-fold if need be
    if cfg.K_PROBE < 1:
        cfg.K_PROBE = optimal_k

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

            # Recover the kth test fold and use the remaining data for
            # knn search
            torch.manual_seed(cfg.SEED)
            if "biomassters" in cfg.DATASET:
                x_train = embedding_ds.embeddings
                y_train = embedding_ds.labels
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
                x_train = embedding_ds.embeddings
                y_train = embedding_ds.labels
                blind_ds = load_dataset("CLEFBlind", transform)[0]
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
                x_train = embedding_ds.embeddings[train_idx]
                y_train = embedding_ds.labels[train_idx]

            # Move al the train embeddings to device [N_train, C]
            x_train = torch.as_tensor(x_train).to(cfg.DEVICE)
            y_train = torch.as_tensor(y_train).to(cfg.DEVICE)

            # Normalize the embeddings to prepare for cosine similarity
            # [N_train, C]
            x_train = x_train / x_train.norm(dim=1).view(-1, 1)

            # Loop over the test embeddings, to search for the k-nearest
            # train embeddings and aggregate associated labels
            test_dl = DataLoader(
                dataset=test_dataset,
                batch_size=cfg.BATCH_SIZE,
                num_workers=cfg.NUM_WORKERS,
                shuffle=False)

            metric.reset()
            acc_y = []
            acc_y_hat = []
            for x, y in test_dl:
                # Move embeddings to working device
                x = x.to(cfg.DEVICE)  # [N_test, C]
                y = y.to(cfg.DEVICE)  # shape depends on task

                # Normalize the embeddings [N_test, C]
                x = x / x.norm(dim=1).view(-1, 1)

                # Bruteforce pairwise similarities [N_test, N_train]
                sim = x @ x_train.T

                # Get the top-k nearest neighbors [N_test, K]
                nn = sim.topk(k=cfg.K_PROBE, dim=1).indices

                # Compute the weights for the neighbors [N_test, K]
                weights = (sim.gather(1, nn) / cfg.TEMP_PROBE).softmax(dim=1)

                # Aggregate the labels based on the weights and the
                # task at hand
                if task == 'multi_label_classification':
                    # For multi-label classification, the labels are
                    # expected as one-hot encodings, we weighted-sum
                    # them across the neighbors to produce logits. The
                    # fact that the weights sum to 1 ensures the
                    # per-class outputs are also in [0, 1]
                    y_hat = (y_train[nn] * weights.unsqueeze(-1)).sum(dim=1)
                elif task == 'classification':
                    # For single-label classification, we convert the
                    # labels to one-hot encodings, weighted-sum them
                    # across the neighbors to produce logits
                    if y_train.ndim == 1:
                        y_train = one_hot(y_train, num_classes=num_dim)
                    y_hat = (y_train[nn] * weights.unsqueeze(-1)).sum(dim=1)
                elif task == 'regression':
                    # For regression tasks, we simply take the weighted
                    # sum of the target values
                    y_hat = (y_train[nn] * weights.unsqueeze(-1)).sum(dim=1)
                elif task == 'distribution_regression':
                    # For distribution regression tasks, we simply take
                    # the weighted sum of the target distributions
                    y_hat = (y_train[nn] * weights.unsqueeze(-1)).sum(dim=1)
                else:
                    raise NotImplementedError(
                        f"Label aggregation for {task=} is not defined.")

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
