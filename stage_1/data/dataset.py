"""Data lists for every dataset and split.

Each getter returns a list of dicts consumed by the MONAI transforms:
    image        path to the CT volume (.nii.gz)
    label        path to the mask, or None
    class_label  (6,) float tensor, multi-label, ordered as config.CLASS_NAMES
    task         routes the sample to its loss in models/lightning_module.py

To plug in a new segmentation dataset, copy get_instance_segmentation_data and
give it its own `task`. The expected directory layout is described in docs/DATA.md.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config

NO_LABEL = torch.zeros(len(config.CLASS_NAMES), dtype=torch.float32)


def _img_seg_pairs(root, split):
    """Sorted (image, mask) paths from <root>/<split>/{img,seg}/*.nii.gz, paired by sort order."""
    images = sorted((Path(root) / split / "img").glob("*.nii.gz"))
    labels = sorted((Path(root) / split / "seg").glob("*.nii.gz"))
    assert len(images) == len(labels), f"{root}/{split}: {len(images)} images vs {len(labels)} masks"
    return list(zip(images, labels))


def _seg_fname_to_pid(fname: str) -> str:
    """MBH-Seg file name -> patientID_studyID, the key of the classification CSVs.

    MBH_SEG_2025_LLG_<date>_ID_xxx_ID_yyy.nii.gz -> ID_xxx_ID_yyy
    any other name                               -> name without .nii.gz
    """
    base = fname.replace(".nii.gz", "")
    if base.startswith("MBH_SEG_2025_LLG_"):
        parts = base.split("_")
        for i, p in enumerate(parts):
            if p == "ID" and i > 0:
                return "_".join(parts[i:])
    return base


# ── Segmentation datasets ─────────────────────────────────────────────────────

def get_segmentation_data(split="train"):
    """MBH-Seg (6 classes).

    Volumes listed in MBH_label_case/splits/classif_annotation_seg<split>.csv also
    get their case-level labels (task "seg_multi": seg + cls losses); the others
    are segmentation only (task "seg_orig").
    """
    csv_path = Path(config.CLASSIFICATION_DATA_DIR) / "splits" / f"classif_annotation_seg{split}.csv"
    id_to_cls = {}
    if csv_path.exists():
        for _, row in pd.read_csv(csv_path).iterrows():
            id_to_cls[str(row["patientID_studyID"])] = torch.tensor(
                [float(row[c]) for c in config.CLASS_NAMES], dtype=torch.float32)

    data = []
    for img, lbl in _img_seg_pairs(config.SEG_DIR, split):
        pid = _seg_fname_to_pid(img.name)
        data.append({
            "image":       str(img),
            "label":       str(lbl),
            "class_label": id_to_cls.get(pid, NO_LABEL).clone(),
            "task":        "seg_multi" if pid in id_to_cls else "seg_orig",
        })

    n_multi = sum(d["task"] == "seg_multi" for d in data)
    print(f"{split.upper()} seg: seg_multi {n_multi} | seg_orig {len(data) - n_multi} | total {len(data)}")
    return data


def get_inhouse_segmentation_data(split="train"):
    """Private in-house dataset (4 classes).

    Returns [] when the directory is absent: the code runs without this dataset, and
    any dataset with the same layout and label space can take its place.
    """
    return [{"image": str(img), "label": str(lbl), "class_label": NO_LABEL.clone(), "task": "seg_inhouse"}
            for img, lbl in _img_seg_pairs(config.IN_HOUSE_DIR, split)]


def get_instance_segmentation_data(split="train"):
    """INSTANCE 2022 (2 classes: background / hemorrhage)."""
    return [{"image": str(img), "label": str(lbl), "class_label": NO_LABEL.clone(), "task": "seg_instance"}
            for img, lbl in _img_seg_pairs(config.INSTANCE_DIR, split)]


def get_physionet_segmentation_data(split="test"):
    """PhysioNet CT-ICH (Hssayeni et al.), the 36 hemorrhage-positive scans: external test set.

    Never used for training or model selection, so every split other than "test" is empty.
    Layout: <PHYSIONET_DIR>/ct_scans/<pid>.nii and masks/<pid>.nii, paired by file name.
    The masks are coded 0/255: evaluate.py binarizes them (> 0).
    """
    root = Path(config.PHYSIONET_DIR)
    if split != "test":
        return []
    return [{"image": str(img), "label": str(root / "masks" / img.name), "class_label": NO_LABEL.clone(),
             "task": "seg_instance"}
            for img in sorted((root / "ct_scans").glob("*.nii"))]


# ── Classification dataset (case-level labels) ────────────────────────────────

def _load_pseudo_validity() -> dict:
    """patientID_studyID -> (6,) flags of the usable pseudo-mask channels.

    Only used by the certified pseudo-mask loss (models/pseudo_loss.py). Read from the CSV
    written by tools/audit_pseudo_masks.py:
        [0]   1 if no subtype was missed by nnU-Net (background usable)
        [1:6] 0 for a subtype present in the case label but absent from the pseudo-mask
    Missing CSV -> {} (every channel treated as valid).
    """
    csv_path = Path(config.PSEUDO_MASKS_DIR) / "pseudo_mask_audit.csv"
    if not csv_path.exists():
        print(f"[pseudo] {csv_path} not found: every channel treated as valid. "
              f"Run `python tools/audit_pseudo_masks.py` first.")
        return {}
    out = {}
    for _, r in pd.read_csv(csv_path).iterrows():
        v = [1.0 if int(r["n_fn"]) == 0 else 0.0]
        v += [float(r[f"valid_{n}"]) for n in ["EDH", "IPH", "IVH", "SAH", "SDH"]]
        out[str(r["pid"])] = torch.tensor(v, dtype=torch.float32)
    return out


def get_classification_data(split="train", cls_crop="pseudo", certified=False):
    """MBH case-level labels: positives (with an nnU-Net pseudo-mask) + healthy volumes.

    Positives come from splits/<split>_split.csv and healthy volumes from
    splits/<split>_healthy_split.csv; images are MBH_label_case/<patientID_studyID>.nii.gz.
    Positives without a pseudo-mask (Pseudo_Masks/<patientID_studyID>.nii.gz) are dropped:
    these are the MBH-Seg volumes, already used through get_segmentation_data.

    cls_crop="pseudo": the pseudo-mask guides the 96^3 crop (task "cls_pseudo").
    cls_crop="random": in training only, positives become plain "classification" items
        with a random crop; validation always keeps the guided crop so that val
        metrics are comparable across settings.
    certified: attach the audit flags used by the label-gated loss (models/pseudo_loss.py).
    """
    root = Path(config.CLASSIFICATION_DATA_DIR)
    guided = not (split == "train" and cls_crop == "random")
    validity = _load_pseudo_validity() if guided and certified else {}

    df = pd.read_csv(root / "splits" / f"{split}_split.csv")
    positives, n_unmasked = [], 0
    for _, row in df.iterrows():
        pid = row["patientID_studyID"]
        mask_path = str(Path(config.PSEUDO_MASKS_DIR) / f"{pid}.nii.gz")
        if not os.path.exists(mask_path):
            n_unmasked += 1
            continue
        item = {
            "image":       str(root / f"{pid}.nii.gz"),
            "label":       mask_path if guided else None,
            "class_label": torch.tensor(np.array([row[c] for c in config.CLASS_NAMES], dtype=np.float32)),
            "has_mask":    guided,
            # Sampling group, independent of cls_crop so that the sampler masses
            # (cls_mask / cls_nomask in train.py) are the same in every setting.
            "cls_group":   "mask",
            "task":        "cls_pseudo" if guided else "classification",
        }
        if guided:
            item["valid_channels"] = validity.get(str(pid), torch.ones(6)).clone()
        positives.append(item)

    healthy_df = pd.read_csv(root / "splits" / f"{split}_healthy_split.csv")
    negatives = [{
        "image":       str(root / f"{row['patientID_studyID']}.nii.gz"),
        "label":       None,
        "class_label": NO_LABEL.clone(),
        "has_mask":    False,
        "cls_group":   "nomask",
        "task":        "classification",
    } for _, row in healthy_df.iterrows()]

    if not positives and not df.empty:
        print(f"[WARNING] {split}: no classification positive kept. Pseudo-masks are expected in "
              f"{config.PSEUDO_MASKS_DIR}/<patientID_studyID>.nii.gz (see docs/DATA.md).")

    data = positives + negatives
    np.random.default_rng(seed=42).shuffle(data)
    print(f"{split.upper()} cls: positives {len(positives)} | healthy {len(negatives)} | "
          f"crop {'pseudo-mask' if guided else 'random'} | "
          f"positives without pseudo-mask (dropped) {n_unmasked}")
    return data


def compute_pos_weights(split="train"):
    """Per-class n_neg / n_pos over the classification data, used as BCEWithLogitsLoss pos_weight."""
    labels = np.array([d["class_label"].numpy() for d in get_classification_data(split)])  # (N, 6)
    n_pos = labels.sum(axis=0)
    pos_weight = (len(labels) - n_pos) / np.maximum(n_pos, 1)
    print("pos_weight: " + ", ".join(f"{n} {w:.2f}" for n, w in zip(config.CLASS_NAMES, pos_weight)))
    return torch.tensor(pos_weight, dtype=torch.float32)
