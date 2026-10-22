"""Exports the seed-0 test split, the 28 subjects the seed-0 models were evaluated on, for tools/demo.py.

    python tools/export_demo_data.py --data ../../data/abide/abide_284.npy

Builds the split with the training code itself (same config, same seed as __main__.py), so the exported
subjects are exactly the ones the reported seed-0 runs were tested on. Writes demo/abide_seed0_test.npz.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import dataset_factory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path, help='abide_284.npy built by tools/prepare_abide.py')
    ap.add_argument('--out', type=Path, default=ROOT / 'demo' / 'abide_seed0_test.npz')
    a = ap.parse_args()

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        cfg = compose('config', overrides=[f'dataset.path={a.data.resolve()}', 'dataset.expand=false'])
    random.seed(0); np.random.seed(0); torch.manual_seed(0)  # as __main__.py does for repeat 0
    f, c, y, s = dataset_factory(cfg)[2].dataset.tensors

    full = np.load(a.data, allow_pickle=True).item()
    raw = torch.from_numpy(full['timeseires']).float()
    raw = (raw - raw.mean()) / raw.std()  # the loader z-scores with the cohort mean/std
    idx = [int(torch.cdist(x.flatten()[None], raw.flatten(1)).argmin()) for x in f]  # each test subject's cohort row
    a.out.parent.mkdir(exist_ok=True)
    np.savez_compressed(a.out, timeseries=f.numpy(), corr=c.numpy(), label=y.argmax(1).numpy(),
                        smri=s.numpy(), file_id=full['file_id'][idx], site=full['site'][idx])
    print(f'wrote {a.out}: {len(idx)} subjects ({int((y.argmax(1) == 0).sum())} controls, '
          f'{int(y.argmax(1).sum())} autism), {a.out.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
