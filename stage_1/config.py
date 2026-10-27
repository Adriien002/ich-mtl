"""Paths, dataset constants and loading of the run configurations (configs/*.yaml).

Paths come from environment variables:
    ICH_DATA_ROOT   root of the datasets (default: ./datasets), layout in docs/DATA.md
    ICH_OUTPUT_DIR  checkpoints, caches and evaluation outputs (default: ./outputs)
"""
import ast
import copy
import os
from pathlib import Path

import yaml

# ======================
# PATHS
# ======================
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("ICH_DATA_ROOT", os.path.join(_HERE, "datasets"))
PROJECT_ROOT = os.environ.get("ICH_OUTPUT_DIR", os.path.join(_HERE, "outputs"))

SEG_DIR = os.path.join(DATA_ROOT, "mbh")                             # MBH-Seg: voxel masks (6 classes)
CLASSIFICATION_DATA_DIR = os.path.join(DATA_ROOT, "MBH_label_case")  # MBH case-level labels
PSEUDO_MASKS_DIR = os.path.join(DATA_ROOT, "Pseudo_Masks")          # nnU-Net pseudo-masks of the case-level positives
IN_HOUSE_DIR = os.path.join(DATA_ROOT, "in_house")                  # optional private dataset (4 classes)
INSTANCE_DIR = os.path.join(DATA_ROOT, "instance")                  # INSTANCE 2022 (2 classes)
PHYSIONET_DIR = os.path.join(DATA_ROOT, "physionet")                # PhysioNet CT-ICH, external test only
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_CONFIG = os.path.join(_HERE, "configs", "final.yaml")

# ======================
# DATASET CONSTANTS
# ======================
NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

img_size = [96, 96, 96]
patch_per_case = 2
non_zero_norm = True

# Class sampling ratios of RandCropByLabelClassesd (background first), per segmentation dataset.
class_ratios_per_dataset = {
    "mbh": [0.1, 0.18, 0.18, 0.18, 0.18, 0.18],
    "instance": [0.1, 0.9],
    "in_house": [0.1, 0.3, 0.3, 0.3],
}


# ======================
# RUN CONFIGURATION
# ======================
def _merge(base, override):
    """Recursive dict merge: values of `override` win."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=DEFAULT_CONFIG, overrides=()):
    """Read a YAML config, resolving its `_base_` chain, then apply "a.b=value" overrides."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = _merge(load_config(path.parent / base), cfg)

    for item in overrides:
        key, _, raw = item.partition("=")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw                         # plain string, e.g. decoder_sharing=full
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = value
    _check(cfg)
    return cfg


def _check(cfg):
    choices = {
        ("model", "decoder_sharing"): ("none", "last_block", "full"),
        ("data", "cls_crop"): ("pseudo", "random"),
        ("loss", "cls_target"): ("volume", "patch"),
        ("loss", "pseudo_mode"): ("dice", "certified"),
        ("loss", "seg_weighting"): ("learned", "fixed"),
    }
    for (section, key), allowed in choices.items():
        if cfg[section][key] not in allowed:
            raise ValueError(f"{section}.{key}={cfg[section][key]!r}, expected one of {allowed}")
    if cfg["data"]["cls_crop"] == "random" and cfg["loss"]["pseudo_weight"] != 0:
        raise ValueError("data.cls_crop=random requires loss.pseudo_weight=0: "
                         "without pseudo-masks there is no dense target.")
