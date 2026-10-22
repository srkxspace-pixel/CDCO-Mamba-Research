"""Checks that the executed Kaggle notebook ran exactly this repository's code.

    python tools/verify_kaggle_notebook.py      # needs a git clone (the ZIP has no history)

Cell 1 of notebooks/kaggle_v3_executed.ipynb carries the code as a gzipped, base64-encoded patch on top of
the authors' commit 7b9adcf. This decodes it and compares it byte for byte with `git diff 7b9adcf 177a62f`
(this repository's code at the time of the run, compiled cache files excluded).
"""
import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.load(open(ROOT / 'notebooks' / 'kaggle_v3_executed.ipynb', encoding='utf-8'))
patch = gzip.decompress(base64.b64decode(''.join(nb['cells'][0]['source']).split()[1]))
repo = subprocess.run(['git', 'diff', '7b9adcf', '177a62f', '--', '.', ':(exclude)*__pycache__*'],
                      cwd=ROOT, capture_output=True, check=True).stdout
md5 = lambda b: hashlib.md5(b).hexdigest()
print(f'notebook patch: {len(patch)} bytes, md5 {md5(patch)}')
print(f'repository    : {len(repo)} bytes, md5 {md5(repo)}')
print('IDENTICAL: the Kaggle run used this code' if patch == repo else 'DIFFERENT')
