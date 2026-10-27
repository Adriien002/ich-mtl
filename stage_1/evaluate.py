"""Test-set Dice of one checkpoint on the segmentation datasets.

Inference protocol:
  - same preprocessing as training (RAS, 1 mm spacing, CropForeground,
    ScaleIntensityRange -10/140 HU, SpatialPad, NormalizeIntensity);
  - sliding window 96^3, overlap 0.5, gaussian blending;
  - Invertd then argmax: the Dice is computed in the ORIGINAL image space,
    against the raw mask read with nibabel;
  - MBH only: post-processing of postprocess_mbh.py (EDH/SDH relabelling, removal of
    EDH, SAH or SDH when the class totals fewer than 25 voxels), as in the paper and for
    every baseline; --no-postproc gives the raw Dice;
  - PhysioNet (external test, INSTANCE head): masks coded 0/255, binarized (> 0);
  - DiceHelper(include_background=False) on one-hot maps, NaN for a class absent
    from the ground truth.

Averaging: the score of a class is the mean over the volumes where it is PRESENT
in the ground truth (NaN ignored); the column n_<class> gives this support.

The model is rebuilt from the config.yaml of the run (next to the checkpoint), or from
--config; the weights must match it exactly.

Printed scores follow Table 2 of the paper: EDH excluded, each dataset as mean ± std across
its classes, Overall = mean ± std (ddof=1) of INSTANCE, In-house and MBH. PhysioNet is
printed apart. The CSV keeps every class, EDH included.

Usage:
  python evaluate.py --ckpt path/to/best.ckpt [--config configs/final.yaml] \
      [--name my_run] [--datasets mbh instance physionet] [--save-nifti] [--no-postproc]
"""
import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import monai.transforms as T
from monai.data import DataLoader, PersistentDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceHelper
from monai.networks.utils import one_hot
from tqdm import tqdm

import config
import data.dataset as dataset
from models.lightning_module import load_trained_model
from postprocess_mbh import postprocess_mbh

OUT_LONG = Path(config.PROJECT_ROOT) / "dice_table_long.csv"
CACHE_ROOT = Path(os.environ.get("ICH_CACHE_DIR", Path(config.PROJECT_ROOT) / "cache")) / "test"
ROI_SIZE = (96, 96, 96)
# EDH is left out of every aggregate of the paper (Sec. 4.3); its rows stay in the CSV.
EXCLUDED_CLASSES = {"EDH"}
IN_DOMAIN = ("instance", "in_house", "mbh")

# The PersistentDataset cache depends only on the dataset and the transforms, not on
# the checkpoint: it is shared by all evaluations.
DATASETS = {
    "instance": dict(task="seg_instance", get_data=dataset.get_instance_segmentation_data,
                     num_classes=2, class_names=["Hematoma"]),
    "in_house": dict(task="seg_inhouse", get_data=dataset.get_inhouse_segmentation_data,
                     num_classes=4, class_names=["ICH", "IVH", "PHE"]),
    "mbh":      dict(task="seg_orig", get_data=dataset.get_segmentation_data,
                     num_classes=6, class_names=["EDH", "IPH", "IVH", "SAH", "SDH"], postproc=True),
    "physionet": dict(task="seg_instance", get_data=dataset.get_physionet_segmentation_data,
                      num_classes=2, class_names=["Hematoma"], binarize_label=True),
}

TRANSFORMS = T.Compose([
    T.LoadImaged(keys=["image", "label"]),
    T.EnsureChannelFirstd(keys=["image", "label"]),
    T.Orientationd(keys=["image", "label"], axcodes="RAS"),
    T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["trilinear", "nearest"]),
    T.CropForegroundd(keys=["image", "label"], source_key="image"),
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    T.SpatialPadd(keys=["image", "label"], spatial_size=config.img_size, mode="constant", value=0.0),
    T.NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=config.non_zero_norm),
])


def eval_dataset(model, device, ds_key, save_dir=None, postproc=True):
    """Dice per volume and per class, in the original image space."""
    cfg = DATASETS[ds_key]
    num_classes, class_names = cfg["num_classes"], cfg["class_names"]
    postproc = postproc and cfg.get("postproc", False)

    def seg_predictor(x):
        return model(x, task=cfg["task"])[0]

    cache_dir = CACHE_ROOT / ds_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_list = cfg["get_data"](split="test")
    if not data_list:
        print(f"  {ds_key}: no test volume found, skipped", flush=True)
        return []
    ds = PersistentDataset(data_list, transform=TRANSFORMS, cache_dir=str(cache_dir))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    inverter = T.Invertd(keys=["pred"], transform=TRANSFORMS, orig_keys=["image"],
                         nearest_interp=False, to_tensor=False)
    dice_helper = DiceHelper(include_background=False, softmax=False, reduction="none")

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=ds_key, file=sys.stdout):
            images = batch["image"].to(device)
            batch["pred"] = sliding_window_inference(
                images, roi_size=ROI_SIZE, sw_batch_size=2,
                predictor=seg_predictor, overlap=0.5, mode="gaussian")

            decollated = decollate_batch(batch)
            pred = inverter(decollated[0])["pred"]          # [C, H, W, D] original space
            argmax_map = torch.argmax(pred, dim=0).cpu().numpy().astype(np.uint8)
            if postproc:
                argmax_map = postprocess_mbh(argmax_map)
            filename = Path(decollated[0]["image"].meta["filename_or_obj"]).name

            orig_seg = nib.load(decollated[0]["label"].meta["filename_or_obj"]).get_fdata()
            orig_seg = (orig_seg > 0) if cfg.get("binarize_label") else orig_seg
            orig_seg = orig_seg.astype(np.uint8)
            pred_t = torch.from_numpy(argmax_map).unsqueeze(0).unsqueeze(0).to(device)
            target_t = torch.from_numpy(orig_seg).unsqueeze(0).unsqueeze(0).to(device)
            scores, _ = dice_helper(one_hot(pred_t, num_classes=num_classes),
                                    one_hot(target_t, num_classes=num_classes))

            for i, name in enumerate(class_names):
                rows.append({"dataset": ds_key, "class": name, "filename": filename,
                             "postproc": postproc, "dice": float(scores[0, i]) * 100})

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(argmax_map, pred.meta["affine"]),
                         str(save_dir / f"PRED_{filename}"))
            del images, pred_t, target_t
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", type=Path, help="run config (default: config.yaml next to the checkpoint)")
    ap.add_argument("--name", default="run", help="name of the run (CSV column)")
    ap.add_argument("--datasets", nargs="+", default=["instance", "in_house", "mbh", "physionet"],
                    choices=list(DATASETS), help="datasets to evaluate (missing ones are skipped)")
    ap.add_argument("--out", type=Path, default=OUT_LONG,
                    help="long-format output CSV, appended to (default: $ICH_OUTPUT_DIR/dice_table_long.csv)")
    ap.add_argument("--save-nifti", action="store_true",
                    help="also write the predicted masks as NIfTI")
    ap.add_argument("--no-postproc", action="store_true",
                    help="raw Dice on MBH, without the post-processing of the paper")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.ckpt.exists():
        sys.exit(f"checkpoint not found: {args.ckpt}")
    model = load_trained_model(args.ckpt, args.config, map_location=device)
    model.eval().to(device)

    rows = []
    for ds_key in args.datasets:
        save_dir = (Path(config.PROJECT_ROOT) / "predictions" / args.name / args.ckpt.stem / ds_key
                    if args.save_nifti else None)
        rows += eval_dataset(model, device, ds_key, save_dir, postproc=not args.no_postproc)

    if not rows:
        sys.exit("no test volume found in any dataset (check ICH_DATA_ROOT)")
    df = pd.DataFrame(rows)
    df.insert(0, "run", args.name)
    df.insert(1, "decoder_sharing", model.cfg["model"]["decoder_sharing"])
    df.insert(2, "checkpoint", args.ckpt.name)

    # Paper protocol (Table 2): EDH excluded; dataset score = mean ± std across its classes;
    # Overall = mean ± std across the three in-domain datasets (PhysioNet reported apart).
    print("\n  ----- results of this checkpoint (EDH excluded) -----")
    scores = {}
    for ds_key in args.datasets:
        sub = df[(df.dataset == ds_key) & ~df["class"].isin(EXCLUDED_CLASSES)]
        if sub.empty:
            continue
        # per-class mean over the volumes where the class is present (NaN ignored),
        # then mean over classes.
        per_class = sub.groupby("class", sort=False)["dice"].mean()
        support = sub.groupby("class", sort=False)["dice"].count()
        scores[ds_key] = per_class.mean()
        std = f"±{per_class.std():5.2f}" if len(per_class) > 1 else " " * 6
        detail = " | ".join(f"{c}={v:.2f} (n={support[c]})" for c, v in per_class.items())
        print(f"  {ds_key:<9} {scores[ds_key]:6.2f} {std}   {detail}")
    in_domain = pd.Series({k: scores[k] for k in IN_DOMAIN if k in scores})
    if len(in_domain) == len(IN_DOMAIN):
        print(f"  {'overall':<9} {in_domain.mean():6.2f} ±{in_domain.std():5.2f}   (mean of {', '.join(IN_DOMAIN)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = not args.out.exists()
    df.to_csv(args.out, mode="a", header=header, index=False, float_format="%.4f")
    print(f"\n  -> {args.out} (+{len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
