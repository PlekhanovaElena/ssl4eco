import torch
from tqdm import tqdm
import os
import os.path
import numpy as np
import pandas as pd

from downstream_tasks.utils.root import get_root
from downstream_tasks.test_modules import *
from downstream_tasks.datasets import *


def load_model(model_name):
    model_name = model_name.lower()
    if model_name not in MODEL_LIST:
        raise NotImplementedError(
            f"Model '{model_name}' does not exist. Available models: "
            f"{MODEL_LIST}")
    print("Using model:", model_name)
    model = MODEL_DICT[model_name]()
    return model, model.transform, model.probe_input_dim


def load_dataset(dataset_name, transform):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_LIST:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' does not exist. Available datasets: "
            f"{DATASET_LIST}")
    print("Using dataset:", dataset_name)
    dataset = DATASET_DICT[dataset_name][0](transform)
    task = DATASET_DICT[dataset_name][1]
    main_metric = DATASET_DICT[dataset_name][2]
    optimal_k = DATASET_DICT[dataset_name][3]
    return dataset, dataset.num_dim, dataset.metric, dataset.loss_fn, task, main_metric, optimal_k


def get_embeddings(
        image_ds,
        model,
        NUM_WORKERS,
        SAVE_NAME,
        DEVICE,
        USE_PRECOMPUTED_EMBEDDINGS):
    save_name = SAVE_NAME + "/" + str(len(image_ds))
    if os.path.isfile(save_name + "_embeddings.npy") and USE_PRECOMPUTED_EMBEDDINGS:
        embeddings = np.load(save_name + "_embeddings.npy")
        labels = np.load(save_name + "_labels.npy")
        print("Embedding and label shapes:", embeddings.shape, labels.shape)
        return embeddings, labels

    image_dl = torch.utils.data.DataLoader(
        image_ds,
        num_workers=NUM_WORKERS,
        batch_size=16)
    model = model.to(DEVICE)
    model.eval()

    embeddings = []
    labels = []
    for image, location, label in tqdm(image_dl):
        labels.append(label)
        image = image.to(DEVICE)
        location = location.to(DEVICE)
        with torch.no_grad():
            embeddings.append(model(image, location).cpu())

    embeddings = torch.concat(embeddings, dim=0)
    labels = torch.concat(labels, dim=0)
    print("Embedding and label shapes:", embeddings.shape, labels.shape)

    os.makedirs(SAVE_NAME, exist_ok=True)
    np.save(save_name + "_embeddings.npy", embeddings)
    np.save(save_name + "_labels.npy", labels)

    return embeddings, labels


def calculate_metric_means(metric_values, cfg, log):
    if len(metric_values) == 0:
        return
    csv_path = cfg.SAVE_NAME[:-len(cfg.SAVE_NAME.split("/")[-1])] + "res.csv"
    if os.path.exists(csv_path):
        res_csv = pd.read_csv(csv_path, index_col=0)
        res_csv.reindex(MODEL_LIST)
    else:
        res_csv = pd.DataFrame(index=MODEL_LIST)

    for i, md in enumerate(metric_values):
        log.info("Iteration " + str(i) + " metrics: " + str(md))

    for metric in metric_values[0].keys():
        # Iterate over all metrics in the first dict and create list
        # for each metric across all iterations
        vals = [mv[metric] for mv in metric_values]
        mean = np.array(vals).mean(axis=0)
        std = np.array(vals).std(axis=0)
        probe_info = f"{cfg.K_PROBE}-NN" if cfg.PROBE == 'knn' else cfg.PROBE
        log.info(f"{probe_info} {metric}: {mean} mean {std} std")
        res_csv.loc[cfg.MODEL, f"{cfg.DATASET} {probe_info} {metric} Mean"] = mean.mean()
        res_csv.loc[cfg.MODEL, f"{cfg.DATASET} {probe_info} {metric} STD"] = std.mean()

    res_csv.to_csv(csv_path)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def calculate_clef_blind_submission(save_path, y_hat):
    ocPa = pd.read_csv(
        get_root() + "/index_files/Presences_Absences_train.csv", header='infer', sep=';')
    ocPa = ocPa.groupby(['patchID',
                         'lon',
                         'lat',
                         'speciesId']).first().reset_index()
    dat_ocpa = ocPa[['patchID', 'lon', 'lat', 'speciesId']]
    dat_ocpa['label'] = 1
    multi_label_ocpa = pd.pivot(dat_ocpa,
                                index=['lat', 'lon', 'patchID'],
                                columns='speciesId',
                                values='label') \
        .reset_index() \
        .fillna(0)
    species_id_list = multi_label_ocpa.columns.tolist()[3:]

    threshold = 0.5
    predicted_present = (y_hat >= threshold)
    num_predicted = predicted_present.sum(dim=1)
    per_class_pred = predicted_present.sum(dim=0)
    print(
        f"Number of predicted on blind: "
        f"min={num_predicted.min().item()}, "
        f"max={num_predicted.max().item()}, "
        f"mean={num_predicted.float().mean().item():0.1f}")
    print(
        f"Number of predicted pre class: "
        f"min={per_class_pred.min().item()}, "
        f"max={per_class_pred.max().item()}, "
        f"mean={per_class_pred.float().mean().item():0.1f}")
    sub = pd.DataFrame(columns=["Predicted"])
    sub.index.names = ["Id"]
    for i in range(len(predicted_present)):
        # sub.loc[i + 1] = " ".join([str(species_id_list[val]) for val in
        #                            torch.topk(predicted_present[i].int(), 30)[1].sort()[0].cpu().numpy()])
        pp = predicted_present[i].nonzero().squeeze(dim=1).cpu().numpy()
        if len(pp) > 0:
            sub.loc[i + 1] = " ".join([
                str(species_id_list[val])
                for val in pp])
        else:
            # TODO: fix this with proper heuristic
            sub.loc[i + 1] = "5"

    assert len(sub) == 22404
    sub.to_csv(save_path)
    print("Saved to:", save_path)


def create_tsne(cfg, embeddings, labels):
    from bhtsne import tsne
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    sample = torch.randperm(len(embeddings), generator=g_cpu)[:1024]
    embs = embeddings[sample]
    lab = labels[sample]

    red = tsne(np.float64(embs))

    vals = np.unique(lab)
    print("Unique labels:", vals)
    colors = [
        "xkcd:forest green", "darkgoldenrod", "red", "orange", "darkcyan",
        "indigo", "xkcd:kelly green", "slategrey", "xkcd:blue green",
        "xkcd:carmine", "greenyellow", "lightcoral", "xkcd:sand", "m", "k",
        "k", "k"]
    val_to_color = {vals[i]: colors[i] for i in range(len(vals))}

    fig, ax = plt.subplots(figsize=(12, 11))

    ax.scatter(
        red[:, 0],
        red[:, 1],
        marker="o",
        c=[val_to_color[int(i)] for i in lab],
        alpha=1,
        s=40)

    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")

    handles = []
    for i in range(len(vals)):
        handles.append(mpatches.Patch(color=colors[i], label=vals[i]))
        fig.legend(handles=handles, framealpha=1)

    plt.title("TSNE " + cfg.DATASET + "/" + cfg.MODEL)
    fig.savefig(cfg.SAVE_NAME + "/" + cfg.DATASET + "_" + cfg.MODEL + "_tsne.png")

    plt.close()
