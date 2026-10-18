import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict
# import pymetis as metis  # unused; pymetis has no Windows build
import networkx as nx

def load_adni_data(cfg: DictConfig):
    
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    smri = data['smri']
    
    class_indices = []
    for c in np.unique(labels):
        indices = np.where(labels == c)[0]
        np.random.shuffle(indices)
        class_indices.append(indices)

    balanced_indices = []
    max_len = max(len(indices) for indices in class_indices)
    for i in range(max_len):
        for indices in class_indices:
            if i < len(indices):
                balanced_indices.append(indices[i])
    balanced_indices = np.array(balanced_indices)

    final_pearson = final_pearson[balanced_indices]
    final_timeseires = final_timeseires[balanced_indices]
    labels = labels[balanced_indices]
    smri = smri[balanced_indices]

    scaler = StandardScaler(mean=np.mean(final_timeseires), std=np.std(final_timeseires))
    final_timeseires = scaler.transform(final_timeseires)
    
    final_timeseires, final_pearson = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson)]

    labels = torch.Tensor(labels)
    smri = torch.Tensor(smri)
    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels, smri

