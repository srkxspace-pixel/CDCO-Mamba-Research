"""Runs the trained seed-0 models (baseline and full) on their 28 test subjects, on CPU.

    python tools/demo.py          # clean, fMRI noise 1.0, sMRI missing
    python tools/demo.py --all    # all 7 test conditions

Uses the same evaluation code as training (training.training.robustness) and prints, next to each
clean result, the value the Kaggle run logged for the same model and subjects.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import open_dict  # noqa: E402
from models.CDCo_MambaNet import CDCo_MambaNet  # noqa: E402
from training.training import CONDITIONS, robustness  # noqa: E402

DEMO = ROOT / 'demo'
VARIANTS = {'baseline': [], 'full': ['model.reliability_fusion=true', 'model.dynamic_dim=true']}
SHORT = ['clean', 'fmri_noise_1.0', 'smri_missing']


def load_model(variant, d):
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        cfg = compose('config', overrides=['dataset=ABIDE', 'model=bnt', 'model.cls_num=2'] + VARIANTS[variant])
    with open_dict(cfg):  # what dataset/abide.py sets from the data
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = d['corr'].shape[1:]
        cfg.dataset.timeseries_sz = d['timeseries'].shape[2]
    model = CDCo_MambaNet(cfg)
    w = DEMO / f'{variant}_seed0.pt'
    if w.stat().st_size < 1e6:
        sys.exit(f'{w.name} is a Git LFS placeholder: install Git LFS and run "git lfs pull" (clone with git, not the ZIP).')
    state = torch.load(w, map_location='cpu', weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='all 7 test conditions')
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = np.load(DEMO / 'abide_seed0_test.npz', allow_pickle=True)
    ref = json.load(open(DEMO / 'kaggle_seed0_reference.json'))
    y = torch.from_numpy(d['label'])
    loader = DataLoader(TensorDataset(torch.from_numpy(d['timeseries']), torch.from_numpy(d['corr']),
                                      F.one_hot(y, 2), torch.from_numpy(d['smri'])), batch_size=16)
    conds = {k: v for k, v in CONDITIONS.items() if a.all or k in SHORT}
    print(f'Seed-0 test set: {len(y)} ABIDE subjects ({int((y == 0).sum())} controls, {int(y.sum())} autism), '
          f'{len(set(d["site"].tolist()))} sites, CPU')

    res = {}
    for v in VARIANTS:
        t0 = time.time()
        model = load_model(v, d)
        torch.manual_seed(0)  # the second head adds random noise even at test time (mamba_fuse.fuse_method1)
        res[v] = robustness(model, loader, conds)
        print(f'  {v:8s} model: {len(conds)} conditions in {time.time() - t0:.0f} s')

    print('\nAccuracy / AUC (%)         baseline       full (gate)')
    for c in conds:
        b, f = res['baseline'][c], res['full'][c]
        line = f'  {c:16s}  {100 * b["ACC"]:5.1f} / {100 * b["AUC"]:5.1f}   {100 * f["ACC"]:5.1f} / {100 * f["AUC"]:5.1f}'
        if c == 'clean':
            kb, kf = ref['baseline']['clean'], ref['full']['clean']
            line += (f'   Kaggle: {100 * kb["ACC"]:.1f} / {100 * kb["AUC"]:.1f}'
                     f'   {100 * kf["ACC"]:.1f} / {100 * kf["AUC"]:.1f}')
        print(line)
    probs = res['baseline']['clean']['probs'] + res['full']['clean']['probs']
    flips = sum((p > 0.5) != (k > 0.5) for p, k in zip(res['full']['clean']['probs'], ref['full']['clean']['probs']))
    print(f'  Every P(autism) lies between {min(probs):.2f} and {max(probs):.2f}, barely more decisive than a coin. The original\n'
          f'  code also adds random noise inside the model at test time, so each evaluation is one random draw: Kaggle\'s\n'
          f'  own log scored the same full-model weights {100 * ref["test_last_epoch_ACC"]["full"]:.1f}% and '
          f'{100 * ref["full"]["clean"]["ACC"]:.1f}% in two passes, and {flips} of {len(y)} predictions\n'
          f'  here differ from Kaggle\'s (3.6 points each). This demo fixes the random seed, so every run prints the same.')

    print('\nGate in the full model (mean over subjects): weight sMRI / fMRI, features kept sMRI / fMRI (of 48)')
    for c in conds:
        r_s, r_f, w_s, w_f, k_s, k_f = res['full'][c]['mean_r_w_k']
        print(f'  {c:16s}  {w_s:4.2f} / {w_f:4.2f}    {k_s:4.1f} / {k_f:4.1f}')

    print('\nPer patient, clean scans: P(autism) from the full model here vs logged on Kaggle')
    print('  subject               site        truth    here   Kaggle  weight sMRI/fMRI')
    for s, fid, site, p_k in zip(res['full']['clean']['subjects'], d['file_id'], d['site'], ref['full']['clean']['probs']):
        truth = 'autism' if s['label'] else 'control'
        print(f'  {fid:20s}  {site:10s}  {truth:7s}  {s["prob"]:.3f}  {p_k:.3f}   {s["r_w_k"][2]:.2f} / {s["r_w_k"][3]:.2f}')


if __name__ == '__main__':
    main()
