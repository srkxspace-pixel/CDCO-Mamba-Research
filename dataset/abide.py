# Reconstructed from the repo's dataset/__pycache__/abide.cpython-38.pyc
# (the .py was never committed).
import os
import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abide_data(cfg: DictConfig):
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    smri = data["smri"]
    site = data["site"]

    scaler = StandardScaler(mean=np.mean(final_timeseires), std=np.std(final_timeseires))
    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson)]

    labels = torch.Tensor(labels)

    # prepare_abide.py stores sMRI as a path to a .npy; copy-on-write mmap keeps 1 GB+ out of RAM.
    if isinstance(smri, str):  # filename relative to the dataset file
        smri = torch.from_numpy(np.load(os.path.join(os.path.dirname(cfg.dataset.path), smri), mmap_mode='c'))
    else:
        smri = torch.Tensor(smri)

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels, smri
