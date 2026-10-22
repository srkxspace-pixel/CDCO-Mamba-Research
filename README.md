# CDCo-MambaNet on ABIDE: Reproduction and Reliability-Gated Fusion

This project reproduces **CDCo-MambaNet**, a multimodal (sMRI + fMRI) brain-disorder classifier, on public
ABIDE I data and extends it with a **reliability-gated fusion** meant to keep working when one scan is missing
or corrupted. It builds on the authors' code ([NewWr/CDCo-MambaNet](https://github.com/NewWr/CDCo-MambaNet),
commit `7b9adcf`), whose history is kept in this repository.

**Result in one line:** the repaired model reaches 55.0 ± 7.8% accuracy on 284 public ABIDE subjects (the paper
reports 77.1 ± 5.4% on its own data); the reliability gate does not improve on it, and the analysis below shows why.

## What this repository adds

1. **A runnable original.** The released code did not run, and once running it collapsed to predicting one
   class. All repairs are listed under [Bugs fixed](#bugs-found-and-fixed-in-the-released-code).
2. **A public-data pipeline.** `tools/prepare_abide.py` builds a 284-subject ABIDE I cohort
   (144 controls, 140 autism; 17 sites) from public sources, because the authors' data is not publicly reachable.
3. **The novelty: reliability-gated fusion** (`models/CDCo_MambaNet/encoder/reliability.py`).
4. **A robustness protocol.** Both models are tested clean, with Gaussian noise on one scan, and with one scan
   missing (7 conditions), over 5 paired seeds with paired t-tests.

## Method: reliability-gated fusion

A gate sits between the two encoders and the cross-modal Mamba fusion block. For each patient it scores each
scan's reliability, then down-weights and compresses the less reliable one.

- **Reliability score.** An evidential head per scan outputs Dirichlet evidence over the classes
  (Han et al., ICLR 2021); reliability is one minus its uncertainty. A missing scan gets r = 0.
- **Reliability-aware fusion (RAF).** The more reliable scan passes unchanged; the other is blended toward a
  learned stand-in token in proportion to its relative reliability.
- **Dynamic feature dimension (DFD).** Each scan keeps only its leading k of 48 features, k from 12 to 48 set by
  relative reliability; nested dropout orders the features by importance.
- **Corruption training.** Each scan is dropped with probability 0.15 (never both) or gets Gaussian noise with
  probability 0.3 during training.

$$
r_m = 1 - \frac{K}{\sum_k \alpha_{m,k}}, \qquad a_m = \frac{r_m}{\max(r_{\mathrm{sMRI}}, r_{\mathrm{fMRI}})}, \qquad
H_m \leftarrow a_m H_m + (1 - a_m)\, t_m, \qquad k_m = 12 + 36\, a_m
$$

All of it is switched by config; with every switch off the model is the repaired original.

| Switch | Default | Meaning |
|---|---|---|
| `model.reliability_fusion` | `false` | RAF |
| `model.dynamic_dim` | `false` | DFD |
| `training.corrupt_p_missing` | `0.0` | probability a scan is dropped during training |
| `training.corrupt_p_noise` | `0.0` | probability a scan gets noise during training |
| `model.distill_detach` | `true` | stop-gradient on distillation targets (`false` = original behaviour, which collapses) |
| `model.grad_checkpoint` | `false` | lets batch 16 fit a 4 GB GPU |

## Results

Kaggle, 2 × NVIDIA T4, 5 seeds, 100 epochs, last epoch reported, 28 test subjects per seed (stratified
80/10/10 split). Full log: [`results/kaggle_v3_run.log`](results/kaggle_v3_run.log).

**Clean test set**

| Metric (%) | Paper | Baseline (repaired original) | Full (with gate) | Paired p |
|---|---|---|---|---|
| Accuracy | 77.14 ± 5.42 | 55.00 ± 7.82 | 53.57 ± 9.78 | 0.48 |
| F1 | 76.02 ± 6.63 | 53.00 ± 7.60 | 51.79 ± 10.40 | |
| AUC | 77.14 ± 5.42 | 55.41 ± 13.85 | 56.33 ± 13.83 | 0.77 |

**Accuracy under noise and missing scans (%)**

| Condition | Baseline | Full | Difference | Paired p |
|---|---|---|---|---|
| fMRI noise 0.5 | 55.00 | 50.71 | −4.29 | 0.11 |
| fMRI noise 1.0 | 55.00 | 50.00 | −5.00 | 0.30 |
| sMRI noise 0.5 | 57.14 | 48.57 | −8.57 | 0.18 |
| sMRI noise 1.0 | 52.86 | 48.57 | −4.29 | 0.24 |
| fMRI missing | 54.29 | 49.29 | −5.00 | 0.40 |
| sMRI missing | 54.29 | 47.14 | −7.14 | 0.19 |

No difference is statistically significant. The gate's weights stayed at 0.93–0.99 under every noise level and
moved only when a scan was missing.

**Why the gate did not help**

1. **Weak signal.** A logistic regression on the same connectivity reaches about 56%; the baseline's validation
   loss stayed near 2 ln 2 (50/50 predictions) for all 100 epochs.
2. **Reliability is measured as confidence.** On weak-signal data both evidential heads stay uncertain
   (reliability about 0.3), so noise cannot lower it further.
3. **Confound.** Only the full model was trained with corruption; a corruption-only baseline was not run.
4. **Small test sets.** One subject moves accuracy by 3.6 points; with 5 seeds only gaps of roughly 5–15 points
   could reach p < 0.05.

Next steps: supervise the gate with the known corruption, add the corruption-only baseline, use stratified
5-fold cross-validation, and correct for scanner site.

## Bugs found and fixed in the released code

| Problem | Effect | Fix |
|---|---|---|
| Distillation targets not detached | Every model collapsed to one class (10/10 runs at exactly 50%) | Stop-gradient on the peer prediction (`model.distill_detach`) |
| Sensitivity/specificity swapped FP and FN and averaged both classes | For two classes SPE = SEN = macro precision | Per-class recall; self-check in `training/metrics.py` |
| AUC computed from hard labels | AUC equals balanced accuracy (the paper's ABIDE AUC equals its accuracy in every row) | AUC from probabilities |
| Mixup mixed only an input the model ignores | Pure label noise | Mix the inputs the model uses |
| Augmentation zeroed the original scans in place | Damaged training data | Augment copies |
| All-zero scan breaks the fMRI encoder's normalisation | NaN gradients for a missing scan | Missing scan = N(0, 1) values |
| Steps per epoch counted before augmentation; schedule could divide by zero | Wrong learning-rate schedule | Fixed count and guard |
| No seeds | Runs not repeatable or paired | Seeded splits and initialisation |
| Hyphenated package folder, two dataset files only as bytecode, one layer's source missing | Code could not be imported | Renamed, recovered, authors' `nn.Conv1d` alternative used |
| Mamba CUDA kernel needs a Linux build | No run on Windows | Pure-PyTorch selective-scan fallback |
| `.cuda()` calls inside the fusion block | No run on a CPU-only machine | Removed (the tensors are already on the right device) |
| T1 registration sampled with a wall-clock seed | Every data build aligns the scans slightly differently | Fixed seed in `tools/prepare_abide.py` |
| The second classifier head adds random noise to its input even at test time (`fuse_method1`) | Each evaluation is a random draw: Kaggle scored the same seed-0 weights 57.1% and 53.6% in two passes | Kept, as it is part of the evaluated model; the demo fixes the seed |

The run in `results/` was made before the sensitivity/specificity fix, so its SPE and SEN columns are macro
precision; accuracy, F1 and AUC are unaffected.

## Repository layout

```
__main__.py                  training entry point (Hydra)
conf/                        configs: dataset, model (incl. novelty switches), training
dataset/                     loaders and augmentation
models/CDCo_MambaNet/        the model; encoder/reliability.py is the reliability gate
training/                    training loop and metrics
tools/prepare_abide.py       builds the 284-subject ABIDE cohort from public data
tools/kaggle_novelty.py      paired baseline vs full experiment, one model per GPU
tools/demo.py                runs the trained seed-0 models on their 28 test patients, on CPU
tools/export_demo_data.py    rebuilds demo/abide_seed0_test.npz with the training code's own split
demo/                        trained seed-0 weights (Git LFS), the 28 test patients, Kaggle's logged values
notebooks/kaggle_run.ipynb   Kaggle notebook that runs the whole experiment
results/kaggle_v3_run.log    log of the reported 10.3-hour run
docs/                        the original authors' README and data notes
```

## Quick start

**1. Install** (Python 3.10 or 3.11; a CPU is enough for the checks)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**2. Self-checks** (seconds each)

```bash
python training/metrics.py                            # correct sensitivity/specificity on a known example
python models/CDCo_MambaNet/encoder/reliability.py    # gate: missing scan -> weight 0 and 12 kept features
```

**3. Live demo** (CPU, about 1 minute; clone with git so Git LFS fetches the 111 MB weight files)

The two trained seed-0 models from the reported Kaggle run, and the 28 real ABIDE patients they were tested
on, are in `demo/`. The demo runs both models with clean scans, with noise on the fMRI, and with the sMRI missing.

```bash
python tools/demo.py          # clean, fMRI noise 1.0, sMRI missing
python tools/demo.py --all    # all 7 test conditions (about 2 minutes)
```

```
Accuracy / AUC (%)         baseline       full (gate)
  clean              53.6 /  46.9    64.3 /  62.2   Kaggle: 53.6 / 53.6   53.6 / 57.7
  fmri_noise_1.0     53.6 /  48.5    60.7 /  65.8
  smri_missing       46.4 /  41.8    50.0 /  46.9

Gate in the full model (mean over subjects): weight sMRI / fMRI, features kept sMRI / fMRI (of 48)
  clean             1.00 / 0.97    48.0 / 47.1
  fmri_noise_1.0    0.98 / 0.98    47.5 / 47.4
  smri_missing      0.00 / 1.00    12.0 / 48.0
```

It then lists each patient (ABIDE subject ID, site, diagnosis), the full model's P(autism), the value logged on
Kaggle, and the gate's weights. What to read from it:

- **The gate works as designed for a missing scan:** its weight drops to 0.00 and it keeps 12 of 48 features.
  Noise barely moves the weights, which is the main finding of this project.
- **The models are near chance:** every P(autism) lies between 0.39 and 0.57.
- **One draw is not a result.** The original code adds random noise inside the model at test time (see Bugs), so
  each evaluation of these 28 patients is a random draw; the demo fixes the seed so it prints the same numbers on
  every run. The 5-seed results above are the evidence; the gap in this single draw is not.

**4. Build the data** (public ABIDE I; about 45 minutes on 4 cores; needs about 4 GB of disk, raw downloads included)

```bash
python tools/prepare_abide.py --root data/abide --workers 4
```

**5. Train one model** (GPU; about 1.8 hours on a T4; use an absolute data path)

```bash
WANDB_MODE=disabled python __main__.py dataset=ABIDE model=bnt model.cls_num=2 repeat_time=1 seed=0 \
    dataset.path=/abs/path/to/data/abide/abide_284.npy
# full model: add  model.reliability_fusion=true model.dynamic_dim=true \
#                  training.corrupt_p_missing=0.15 training.corrupt_p_noise=0.3
# 4 GB GPU:   add  model.grad_checkpoint=true
```

**6. Full experiment**: import `notebooks/kaggle_run.ipynb` into Kaggle, set *GPU T4 x2* and *Internet on*,
then *Save & Run All* (about 10.5 hours, unattended).

## Data

ABIDE I via the [Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/)
(DPARSF, band-pass 0.01–0.1 Hz, no global signal regression, AAL-116 regions, first 115 volumes) and raw T1
images (N4, registration to MNI152, 96³). The data is not redistributed here; `tools/prepare_abide.py` downloads
it. Use is subject to the ABIDE data terms.

## Credits

- Original model and code: Z. Wang, J. Xie, D. Tang, J. Deng, Y. Liang, Q. Huang, J. Wang, "Contrastive
  distillation and cross-modal Mamba fusion for multimodal brain disease diagnosis", *Biomedical Signal
  Processing and Control*; code at [NewWr/CDCo-MambaNet](https://github.com/NewWr/CDCo-MambaNet). The original
  repository states no license; this repository is for academic research.
- Evidential uncertainty: Han et al., "Trusted Multi-View Classification", ICLR 2021.
- Data: ABIDE I and the Preprocessed Connectomes Project.

## Acknowledgements

AI assistance (Claude) was used for debugging and running experiments.
