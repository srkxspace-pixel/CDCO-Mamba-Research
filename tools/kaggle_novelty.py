"""Kaggle runner: paired comparison of the repaired baseline vs the full novelty model on ABIDE.

    python tools/kaggle_novelty.py            # from the repo root, inside a Kaggle GPU notebook

Data: uses an attached dataset containing abide_284.npy (+ abide_smri96.npy) if there is one,
otherwise rebuilds it from public ABIDE with tools/prepare_abide.py (~40 min of CPU).
Runs every (seed, variant) pair, one process per GPU at a time, seed by seed so finished results
stay paired; nothing new starts after LAUNCH_UNTIL_H (counted from script start, data rebuild included)
so the commit ends inside Kaggle's 12 h.
Writes tables, a paired t-test and a robustness plot to /kaggle/working.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CORRUPT = ['training.corrupt_p_missing=0.15', 'training.corrupt_p_noise=0.3']
VARIANTS = {
    'baseline': [],
    'full': ['model.reliability_fusion=true', 'model.dynamic_dim=true'] + CORRUPT,
    # ablations for a later session: 'corrupt_only': CORRUPT,
    # 'raf': ['model.reliability_fusion=true'] + CORRUPT, 'dfd': ['model.dynamic_dim=true'] + CORRUPT,
}
SEEDS = [0, 1, 2, 3, 4]
LAUNCH_UNTIL_H, HARD_STOP_H = 9.3, 11.4  # hours since this script started (Kaggle kills the session at 12 h)
T0 = time.time()
CODE, OUT = Path(__file__).resolve().parents[1], Path('/kaggle/working')
M = ['ACC', 'SPE', 'SEN', 'F1', 'AUC']
ORDER = ['clean', 'fmri_noise_0.5', 'fmri_noise_1.0', 'smri_noise_0.5', 'smri_noise_1.0', 'fmri_missing', 'smri_missing']


def sh(*cmd):
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_data():
    hits = glob.glob('/kaggle/input/**/abide_284.npy', recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    print('no attached ABIDE dataset: rebuilding it from public ABIDE (about 40 min)', flush=True)
    sh(sys.executable, '-m', 'pip', 'install', '-q', 'SimpleITK', 'nilearn')
    sh(sys.executable, '-u', str(CODE / 'tools/prepare_abide.py'), '--root', '/tmp/abide', '--workers', str(os.cpu_count() or 4))
    return '/tmp/abide'


def train(data, gpus):
    (OUT / 'logs').mkdir(exist_ok=True)

    def launch(tag, seed, overrides, g):
        cmd = [sys.executable, '-u', '-W', 'ignore', '__main__.py', 'dataset=ABIDE', 'model=bnt', 'model.cls_num=2',
               'repeat_time=1', f'seed={seed}', f'dataset.path={data}/abide_284.npy', f'dataset.aug_cache=/tmp/aug_gpu{g}.npy',
               f'log_path=result/{tag}', f'hydra.run.dir=outputs/{tag}'] + overrides
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), WANDB_MODE='disabled')
        return subprocess.Popen(cmd, cwd=CODE, env=env, stdout=open(OUT / f'logs/{tag}.log', 'w'), stderr=subprocess.STDOUT)

    def last_epoch(tag):
        lines = [l for l in open(OUT / f'logs/{tag}.log', errors='ignore') if 'Epoch[' in l]
        return lines[-1].split('INFO] ')[-1][:120] if lines else 'preparing data / augmentation'

    last_print = 0
    queue = [(f'{v}_seed{s}', s, VARIANTS[v]) for s in SEEDS for v in VARIANTS]
    running = {}
    while queue or running:
        h = (time.time() - T0) / 3600
        for g in gpus:
            if g not in running and queue and h < LAUNCH_UNTIL_H:
                tag, s, ov = queue.pop(0)
                running[g] = (tag, time.time(), launch(tag, s, ov, g))
                print(f'[{h:.2f} h] {tag} started on GPU {g}', flush=True)
        for g, (tag, t0, p) in list(running.items()):
            if p.poll() is not None:
                print(f'[{h:.2f} h] {tag} finished (exit {p.returncode}) in {(time.time() - t0) / 3600:.2f} h', flush=True)
                del running[g]
            elif h > HARD_STOP_H:
                p.terminate()
                print(f'{tag} stopped at the time limit', flush=True)
                del running[g]
        if h >= LAUNCH_UNTIL_H and queue:
            print('time budget reached; not started:', [q[0] for q in queue], flush=True)
            queue = []
        if time.time() - last_print > 900:
            for tag, _, _ in running.values():
                print(f'  [{h:.2f} h] {tag}: {last_epoch(tag)}', flush=True)
            last_print = time.time()
        time.sleep(30)


def report():
    rows = []
    for f in glob.glob(str(CODE / 'result/*_seed*/*/final.json')):
        v, s = Path(f).parent.parent.name.rsplit('_seed', 1)
        for cond, m in json.load(open(f))['robustness'].items():
            rows.append(dict(variant=v, seed=int(s), condition=cond, **{k: m[k] for k in M},
                             **dict(zip(['r_s', 'r_f', 'w_s', 'w_f', 'k_s', 'k_f'], m.get('mean_r_w_k', [np.nan] * 6)))))
    if not rows:
        print('no finished runs; see /kaggle/working/logs')
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'all_results.csv', index=False)
    print('\ncompleted runs:', df[df.condition == 'clean'].groupby('variant').seed.apply(sorted).to_dict())

    ms = lambda x: f'{100 * x.mean():.2f} ± {100 * x.std(ddof=1):.2f}'
    print('\nClean test set, last epoch (%; mean ± std over seeds)')
    print(df[df.condition == 'clean'].groupby('variant')[M].agg(ms).to_string())
    for k in ('ACC', 'AUC'):
        t = df.pivot_table(index='condition', columns='variant', values=k, aggfunc=ms).reindex(ORDER)
        print(f'\nRobustness, {k} (%)')
        print(t.to_string())
        t.to_csv(OUT / f'robustness_{k}.csv')

    if {'baseline', 'full'} <= set(df.variant):
        from scipy.stats import ttest_rel
        out = []
        for cond in ORDER:
            a = df[(df.condition == cond) & (df.variant == 'full')].set_index('seed')
            b = df[(df.condition == cond) & (df.variant == 'baseline')].set_index('seed')
            seeds = a.index.intersection(b.index)
            for k in ('ACC', 'AUC'):
                d = 100 * (a.loc[seeds, k] - b.loc[seeds, k])
                p = ttest_rel(a.loc[seeds, k], b.loc[seeds, k]).pvalue if len(seeds) > 1 else np.nan
                out.append(dict(condition=cond, metric=k, seeds=len(seeds), mean_diff=round(d.mean(), 2),
                                full_better=int((d > 0).sum()), p_paired_t=round(p, 3)))
        paired = pd.DataFrame(out)
        paired.to_csv(OUT / 'paired_full_vs_baseline.csv', index=False)
        print('\nfull - baseline, paired by seed (percentage points)')
        print(paired.to_string(index=False))

    gate = df[df.r_s.notna()]
    if len(gate):
        rel = gate.groupby(['variant', 'condition'])[['r_s', 'r_f', 'w_s', 'w_f', 'k_s', 'k_f']].mean().round(2)
        print('\nReliability r, fusion weight w and kept dims k (of 48) by test condition')
        print(rel.to_string())
        rel.to_csv(OUT / 'reliability_by_condition.csv')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    acc = df.groupby(['condition', 'variant']).ACC.agg(['mean', 'std']).reset_index()
    vs = list(dict.fromkeys(df.variant))
    x, w = np.arange(len(ORDER)), 0.8 / max(1, len(vs))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, v in enumerate(vs):
        a = acc[acc.variant == v].set_index('condition').reindex(ORDER)
        ax.bar(x + i * w - 0.4 + w / 2, 100 * a['mean'], w, yerr=100 * a['std'], label=v, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', ' ') for o in ORDER], rotation=20)
    ax.set_ylabel('test accuracy (%)')
    ax.axhline(50, color='gray', lw=0.8, ls='--')
    ax.legend()
    ax.set_title('ABIDE ASD vs NC: robustness')
    plt.tight_layout()
    plt.savefig(OUT / 'robustness_acc.png', dpi=120)
    print('\nwrote', sorted(p.name for p in OUT.glob('*.csv')), 'and robustness_acc.png')


if __name__ == '__main__':
    if sys.argv[1:] == ['--report']:
        report()
        sys.exit()
    gpus = list(range(torch.cuda.device_count()))
    assert gpus, 'Turn on a GPU accelerator (Settings -> Accelerator -> GPU T4 x2)'
    print('torch', torch.__version__, '| GPUs:', [torch.cuda.get_device_name(i) for i in gpus], flush=True)
    train(find_data(), gpus)
    report()
