"""Build the ABIDE file read by dataset/abide.py from public ABIDE I data (no application needed).

fMRI  ABIDE Preprocessed (PCP) DPARSF outputs, band-pass 0.01-0.1 Hz, no GSR, AAL-116 ROI
      signals: the DPABI/DPARSF -> ROISignals route of data_preprocessing.md, run by PCP
      (DPABI itself needs MATLAB). Truncated to the first 115 volumes, the shortest common
      length, which gives the 64-wide map conf/model/bnt.yaml expects for ABIDE.
sMRI  Raw T1w from ABIDE RawDataBIDS, processed as data_preprocessing.md describes, with
      SimpleITK: N4 -> rigid+affine to MNI152 (ICBM 2009a) -> MNI brain mask -> z-score in
      mask -> 96x96x96 over the mask's bounding box.
Cohort Paper's ABIDE size and balance (144 NC / 140 ASD); age <= 18 to match its age profile
      (13.9 / 13.2 y), OHSU dropped (77 volumes < 115), any PCP QC rater "fail" dropped.
      Random sample per class (seed 0); failed downloads/preprocessing are replaced in order.

    python tools/prepare_abide.py --root ../../data/abide
"""
import argparse
import concurrent.futures as cf
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

S3 = "https://s3.amazonaws.com/fcp-indi"
PRJ = "data/Projects/ABIDE_Initiative"
T = 115
GRID = 96


def candidates(pheno, n_tc, n_asd, spare, seed):
    p = pd.read_csv(pheno)
    p = p[(p.FILE_ID != "no_filename") & (p.AGE_AT_SCAN <= 18) & (p.SITE_ID != "OHSU")]
    qc = [c for c in p.columns if c.startswith("qc_") and "notes" not in c]
    p = p[~(p[qc] == "fail").any(axis=1)]
    order = {dx: p[p.DX_GROUP == dx].sample(frac=1, random_state=seed) for dx in (1, 2)}
    return order[2].head(n_tc + spare), order[1].head(n_asd + spare)


def t1_keys():
    """sub-XXXXXXX -> S3 key of its first T1w in RawDataBIDS."""
    keys, token = {}, None
    while True:
        q = {"list-type": "2", "prefix": f"{PRJ}/RawDataBIDS/"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(urllib.request.urlopen(f"{S3}?{urllib.parse.urlencode(q)}", timeout=120).read())
        ns = {"s": root.tag.split("}")[0][1:]}
        for k in root.iterfind("s:Contents/s:Key", ns):
            if k.text.endswith("_T1w.nii.gz"):
                keys.setdefault(k.text.split("/")[-3], k.text)
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not token:
            return keys


def fetch(url, dst):
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
    return dst


def output_grid(mask_path):
    import SimpleITK as sitk
    mask = sitk.ReadImage(str(mask_path)) > 0
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    bb = stats.GetBoundingBox(1)
    start, size = np.array(bb[:3], float), np.array(bb[3:], float)
    ref = sitk.Image([GRID] * 3, sitk.sitkFloat32)
    ref.SetSpacing(tuple(size * np.array(mask.GetSpacing()) / GRID))
    ref.SetOrigin(mask.TransformContinuousIndexToPhysicalPoint(tuple(start - 0.5 + 0.5 * size / GRID)))
    ref.SetDirection(mask.GetDirection())
    mask96 = sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    return ref, sitk.GetArrayFromImage(mask96) > 0


def _register(fixed, moving, tx):
    import SimpleITK as sitk
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(32)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.05, 0)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.5)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2])
    r.SetSmoothingSigmasPerLevel([2, 1])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(tx, inPlace=True)
    r.Execute(fixed, moving)
    return tx


def preprocess_t1(src, dst, tmpl_path, mask_path):
    """Returns NCC with the template inside the brain mask (registration QC)."""
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    tmpl = sitk.ReadImage(str(tmpl_path), sitk.sitkFloat32)
    ref, m = output_grid(mask_path)
    t96 = sitk.GetArrayFromImage(sitk.Resample(tmpl, ref, sitk.Transform(), sitk.sitkLinear, 0.0))
    if not dst.exists():
        img = sitk.ReadImage(str(src), sitk.sitkFloat32)
        head = sitk.OtsuThreshold(img, 0, 1, 200)
        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.SetMaximumNumberOfIterations([50, 50, 30])
        n4.Execute(sitk.Shrink(img, [4] * 3), sitk.Shrink(head, [4] * 3))
        img = sitk.Cast(img / sitk.Exp(n4.GetLogBiasFieldAsImage(img)), sitk.sitkFloat32)

        rigid = sitk.CenteredTransformInitializer(
            tmpl, img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
        rigid = _register(tmpl, img, rigid)
        affine = sitk.AffineTransform(3)
        affine.SetCenter(rigid.GetCenter())
        affine.SetMatrix(rigid.GetMatrix())
        affine.SetTranslation(rigid.GetTranslation())
        affine = _register(tmpl, img, affine)

        a = sitk.GetArrayFromImage(sitk.Resample(img, ref, affine, sitk.sitkLinear, 0.0))
        v = a[m]
        np.save(dst, np.where(m, (a - v.mean()) / (v.std() + 1e-8), 0).astype(np.float32))
    a = np.load(dst)
    return float(np.corrcoef(a[m], t96[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--n-tc", type=int, default=144)
    ap.add_argument("--n-asd", type=int, default=140)
    ap.add_argument("--spare", type=int, default=10, help="extra candidates per class to replace failures")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-ncc", type=float, default=0.0,
                    help="NCC is logged for QC only by default: on ABIDE even 0.19-scoring scans were well aligned")
    a = ap.parse_args()
    raw, t1dir, roidir, sdir = (a.root / d for d in ("raw", "raw/t1", "raw/rois_aal", "smri96"))
    for d in (raw, t1dir, roidir, sdir):
        d.mkdir(parents=True, exist_ok=True)

    pheno = fetch(f"{S3}/{PRJ}/Phenotypic_V1_0b_preprocessed1.csv", raw / "Phenotypic_V1_0b_preprocessed1.csv")
    tc, asd = candidates(pheno, a.n_tc, a.n_asd, a.spare, seed=0)
    cand = pd.concat([tc, asd])
    keys = t1_keys()
    cand = cand[cand.SUB_ID.map(lambda s: f"sub-{int(s):07d}" in keys)]
    print(f"{len(cand)} candidates with a T1w")

    from nilearn.datasets import fetch_icbm152_2009
    icbm = fetch_icbm152_2009(data_dir=str(a.root / "templates"))

    def download(row):
        sub = f"sub-{int(row.SUB_ID):07d}"
        fetch(f"{S3}/{PRJ}/Outputs/dparsf/filt_noglobal/rois_aal/{row.FILE_ID}_rois_aal.1D",
              roidir / f"{row.FILE_ID}_rois_aal.1D")
        fetch(f"{S3}/{keys[sub]}", t1dir / f"{row.FILE_ID}_T1w.nii.gz")
        return row.FILE_ID

    ok = set()
    with cf.ThreadPoolExecutor(8) as ex:
        for f in cf.as_completed([ex.submit(download, r) for r in cand.itertuples()]):
            try:
                ok.add(f.result())
            except Exception as e:  # keep going; a replacement candidate fills the slot
                print("download failed:", e)
    print(f"downloaded {len(ok)}/{len(cand)}")

    ncc = {}
    with cf.ProcessPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(preprocess_t1, t1dir / f"{fid}_T1w.nii.gz", sdir / f"{fid}.npy", icbm.t1, icbm.mask): fid
                for fid in cand.FILE_ID if fid in ok}
        for i, f in enumerate(cf.as_completed(jobs), 1):
            fid = jobs[f]
            try:
                ncc[fid] = f.result()
            except Exception as e:
                print("preprocess failed:", fid, e)
            if i % 20 == 0:
                print(f"sMRI preprocessed {i}/{len(jobs)}", flush=True)
    cand = cand.assign(ncc=cand.FILE_ID.map(ncc))
    cand.to_csv(a.root / "candidates_qc.csv", index=False)

    good = cand[cand.ncc >= a.min_ncc]
    cohort = pd.concat([good[good.DX_GROUP == 2].head(a.n_tc), good[good.DX_GROUP == 1].head(a.n_asd)])
    cohort.to_csv(a.root / "cohort.csv", index=False)
    print(f"cohort: {len(cohort)} (TC {(cohort.DX_GROUP == 2).sum()}, ASD {(cohort.DX_GROUP == 1).sum()}), "
          f"age {cohort.AGE_AT_SCAN.mean():.2f}+-{cohort.AGE_AT_SCAN.std():.2f}, "
          f"min NCC {cohort.ncc.min():.3f}, sites {cohort.SITE_ID.nunique()}")

    ts = np.stack([np.loadtxt(roidir / f"{f}_rois_aal.1D")[:T].T for f in cohort.FILE_ID]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.stack([np.corrcoef(x) for x in ts])).astype(np.float32)
    smri_path = a.root / "abide_smri96.npy"
    smri = np.lib.format.open_memmap(smri_path, mode="w+", dtype=np.float32, shape=(len(cohort), 1, GRID, GRID, GRID))
    for i, f in enumerate(cohort.FILE_ID):
        smri[i, 0] = np.load(sdir / f"{f}.npy")
    smri.flush()
    np.save(a.root / "abide_284.npy", {
        "timeseires": ts, "corr": corr,
        "label": (cohort.DX_GROUP == 1).astype(np.int64).to_numpy(),  # 0 = NC (TC), 1 = ASD
        "site": cohort.SITE_ID.to_numpy(), "file_id": cohort.FILE_ID.to_numpy(),
        "smri": smri_path.name,  # relative to this file; memory-mapped by dataset/abide.py
    }, allow_pickle=True)
    print("wrote", a.root / "abide_284.npy", ts.shape, corr.shape, smri.shape)


if __name__ == "__main__":
    main()
