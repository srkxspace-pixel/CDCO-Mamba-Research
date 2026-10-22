# Notebooks

| Notebook | What it is |
|---|---|
| `kaggle_v3_executed.ipynb` | **The notebook that produced the reported results**, downloaded from Kaggle with its output: version 3, 2 × NVIDIA T4, 10.3 hours, 10 models (baseline and full, seeds 0–4). |
| `kaggle_run.ipynb` | The same experiment for a new run: it clones this repository instead of carrying a patch. |

Both are thin launchers. The experiment itself is `tools/kaggle_novelty.py`: it builds the ABIDE data if none is
attached, trains every (variant, seed) pair one per GPU, runs the robustness test and prints the tables.
The model, training loop and evaluation are the repository code it runs.

## The executed notebook, cell by cell

1. **Cell 1** writes the code as a compressed patch (`novelty.patch.gz`) and checks its MD5. The code was not on
   GitHub when the run was made, so the notebook carried it this way. Output: `novelty.patch.gz: OK`.
2. **Cell 2** clones the authors' repository at commit `7b9adcf`, applies the patch, installs the packages and runs
   `tools/kaggle_novelty.py`. Its output is the full run log: data build, training progress every 15 minutes,
   then the result tables. The same log is in `results/kaggle_v3_run.log`.

The patch is exactly this repository's code at the time of the run. To check it (needs a git clone):

```bash
python tools/verify_kaggle_notebook.py
```

```
notebook patch: 58785 bytes, md5 30234e9270819495cc7aefcb09320830
repository    : 58785 bytes, md5 30234e9270819495cc7aefcb09320830
IDENTICAL: the Kaggle run used this code
```

## Earlier runs

Version 2 of the same Kaggle notebook ran the original training code for 10 hours: all 10 models collapsed to
predicting one class (training accuracy fixed at 0.50661, test accuracy 50.00%). That led to the stop-gradient fix
in commit `177a62f`, which version 3 includes. Both versions are in the Kaggle notebook's version history.
