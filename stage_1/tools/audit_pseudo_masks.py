"""Audit of the nnU-Net pseudo-masks against the case-level classification labels.

The case-level labels (MBH_label_case/splits/*_split.csv) are true annotations, the
pseudo-masks are nnU-Net predictions. Crossing them gives, per volume and subtype:

    label absent  + pseudo-mask present -> the pseudo-mask is provably wrong (FP)
    label present + pseudo-mask absent  -> the pseudo-mask is provably incomplete (FN)

The valid_<SUBTYPE> and n_fn columns feed the certified loss (models/pseudo_loss.py).

Usage (from the repository root):
    ICH_DATA_ROOT=/path/to/datasets python tools/audit_pseudo_masks.py
    # writes pseudo_mask_audit.csv and pseudo_mask_audit_summary.csv in $ICH_DATA_ROOT/Pseudo_Masks,
    # where data/dataset.py expects them
"""
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
import config  # noqa: E402

# Same order as the nnU-Net labels of the pseudo-masks and as config.CLASS_NAMES.
COLS = config.CLASS_NAMES
SHORT = ["any", "EDH", "IPH", "IVH", "SAH", "SDH"]
SUBTYPES = SHORT[1:]
OUT_DIR = Path(config.PSEUDO_MASKS_DIR)


def audit() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        csvp = Path(config.CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
        if not csvp.exists():
            continue
        df = pd.read_csv(csvp)
        for _, r in df.iterrows():
            pid = r["patientID_studyID"]
            mp = Path(config.PSEUDO_MASKS_DIR) / f"{pid}.nii.gz"
            if not mp.exists():
                continue  # no pseudo-mask: data/dataset.py drops this volume too
            m = np.asarray(nib.load(str(mp)).dataobj).astype(np.int64)
            counts = np.bincount(m.ravel(), minlength=6)[:6]

            rec = {"pid": pid, "split": split, "vox_total": int(counts[1:].sum()),
                   "gt_any": int(r["any"])}
            for c, name in enumerate(SUBTYPES, start=1):
                gt = int(r[COLS[c]])
                ps = int(counts[c] > 0)
                rec[f"vox_{name}"] = int(counts[c])
                rec[f"gt_{name}"] = gt
                rec[f"pseudo_{name}"] = ps
                # valid = the channel can be a dense target for this volume. Invalid only
                # for a FN: the lesion exists but the pseudo-mask does not locate it.
                # FPs are handled voxel-wise by the loss (certified absence), not here.
                rec[f"valid_{name}"] = int(not (gt == 1 and ps == 0))
            rec["n_fn"] = sum(1 - rec[f"valid_{n}"] for n in SUBTYPES)
            rec["n_fp"] = sum(int(rec[f"gt_{n}"] == 0 and rec[f"pseudo_{n}"] == 1) for n in SUBTYPES)
            rec["mask_empty"] = int(rec["vox_total"] == 0)
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in SUBTYPES:
        gt, ps = d[f"gt_{name}"].values, d[f"pseudo_{name}"].values
        tp = int(((gt == 1) & (ps == 1)).sum())
        fn = int(((gt == 1) & (ps == 0)).sum())
        fp = int(((gt == 0) & (ps == 1)).sum())
        tn = int(((gt == 0) & (ps == 0)).sum())
        vox = d.loc[d[f"vox_{name}"] > 0, f"vox_{name}"]
        out.append({
            "scope": "subtype", "item": name, "n_volumes": len(d),
            "gt_positives": int(gt.sum()), "pseudo_positives": int(ps.sum()),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall_pct": round(100 * tp / max(tp + fn, 1), 1),
            "precision_pct": round(100 * tp / max(tp + fp, 1), 1),
            "median_vox_if_predicted": float(vox.median()) if len(vox) else 0.0,
        })

    for scope, sub in [("all", d)] + [(s, d[d.split == s]) for s in sorted(d.split.unique())]:
        g = sub[[f"gt_{c}" for c in SUBTYPES]].values
        p = sub[[f"pseudo_{c}" for c in SUBTYPES]].values
        exact = (g == p).all(axis=1)
        out.append({
            "scope": f"volume/{scope}", "item": "presence_vector", "n_volumes": len(sub),
            "gt_positives": "", "pseudo_positives": "",
            "TP": int(exact.sum()), "FN": int((sub.n_fn > 0).sum()),
            "FP": int((sub.n_fp > 0).sum()), "TN": int(sub.mask_empty.sum()),
            "recall_pct": round(100 * exact.mean(), 1) if len(sub) else 0.0,
            "precision_pct": "", "median_vox_if_predicted": "",
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = audit()
    per_vol = OUT_DIR / "pseudo_mask_audit.csv"
    summary = OUT_DIR / "pseudo_mask_audit_summary.csv"
    d.to_csv(per_vol, index=False)
    summarize(d).to_csv(summary, index=False)
    print(f"{len(d)} volumes audited")
    print(f"  per volume: {per_vol}")
    print(f"  summary:    {summary}")
    print(summarize(d).to_string(index=False))
