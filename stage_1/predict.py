"""Inference on new CT scans: hemorrhage segmentation mask + subtype probabilities.

For every input NIfTI volume:
  - segmentation: sliding window 96^3 (overlap 0.5, gaussian), resampled back to
    the original image space, written as <name>_seg.nii.gz.
    MBH head labels: 0 background, 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH.
  - classification (MBH head only): sigmoid probabilities of the 6 classes
    (any, EDH, IPH, IVH, SAH, SDH), max over all 96^3 patches of the volume,
    written to predictions.csv.

Usage:
  python predict.py --ckpt path/to/best.ckpt --input scan.nii.gz --output out_dir
  python predict.py --ckpt path/to/best.ckpt --input folder_of_nifti/ --output out_dir

The model is rebuilt from the config.yaml next to the checkpoint, or from --config
(default: configs/final.yaml).
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import monai.transforms as T
from monai.data.utils import dense_patch_slices
from monai.inferers import sliding_window_inference

import config
from models.lightning_module import load_trained_model

ROI_SIZE = tuple(config.img_size)
OVERLAP = 0.5
HEAD_TO_TASK = {"mbh": "seg_orig", "instance": "seg_instance", "in_house": "seg_inhouse"}
CLS_COLUMNS = ["any", "EDH", "IPH", "IVH", "SAH", "SDH"]

TRANSFORMS = T.Compose([
    T.LoadImaged(keys=["image"]),
    T.EnsureChannelFirstd(keys=["image"]),
    T.Orientationd(keys=["image"], axcodes="RAS"),
    T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="trilinear"),
    T.CropForegroundd(keys=["image"], source_key="image"),
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    T.SpatialPadd(keys=["image"], spatial_size=config.img_size, mode="constant", value=0.0),
    T.NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=config.non_zero_norm),
])


@torch.no_grad()
def sliding_window_cls(volume, model, device, sw_batch=4):
    """volume: [1, 1, H, W, D] -> [NUM_CLASSES] probabilities (max over all patches)."""
    step = tuple(max(1, int(r * (1 - OVERLAP))) for r in ROI_SIZE)
    probs_max = torch.zeros(config.NUM_CLASSES)
    buffer = []

    def flush(buf, pmax):
        _, cls_logits = model(torch.cat(buf, dim=0).to(device))
        return torch.max(pmax, torch.sigmoid(cls_logits).float().cpu().max(dim=0).values)

    for sl in dense_patch_slices(volume.shape[2:], ROI_SIZE, step):
        patch = volume[:, :, sl[0], sl[1], sl[2]]
        pads = []
        for p_sz, r_sz in zip(reversed(patch.shape[2:]), reversed(ROI_SIZE)):
            pads += [0, max(0, r_sz - p_sz)]
        if any(p > 0 for p in pads):
            patch = F.pad(patch, pads)
        buffer.append(patch)
        if len(buffer) == sw_batch:
            probs_max = flush(buffer, probs_max)
            buffer = []
    if buffer:
        probs_max = flush(buffer, probs_max)
    return probs_max


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", type=Path, help="run config (default: config.yaml next to the checkpoint)")
    ap.add_argument("--input", required=True, type=Path, help="a .nii.gz file or a folder of .nii.gz files")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--head", default="mbh", choices=list(HEAD_TO_TASK),
                    help="segmentation head to use (default: mbh, 5 subtypes)")
    ap.add_argument("--no-cls", action="store_true", help="skip the classification output")
    args = ap.parse_args()

    files = sorted(args.input.glob("*.nii.gz")) if args.input.is_dir() else [args.input]
    if not files:
        sys.exit(f"no .nii.gz file found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.ckpt, args.config, map_location=device)
    model.eval().to(device)
    task = HEAD_TO_TASK[args.head]
    do_cls = (args.head == "mbh") and not args.no_cls

    inverter = T.Invertd(keys=["pred"], transform=TRANSFORMS, orig_keys=["image"],
                         nearest_interp=False, to_tensor=True)
    rows = []
    for f in files:
        data = TRANSFORMS({"image": str(f)})
        volume = data["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            seg = sliding_window_inference(
                volume, roi_size=ROI_SIZE, sw_batch_size=2,
                predictor=lambda x: model(x, task=task)[0], overlap=OVERLAP, mode="gaussian")
        data["pred"] = seg[0].cpu()
        pred = inverter(data)["pred"]                                   # [C, H, W, D] original space
        mask = torch.argmax(pred, dim=0).numpy().astype(np.uint8)
        ref = nib.load(str(f))
        name = f.name.replace(".nii.gz", "")
        out = nib.Nifti1Image(mask, ref.affine)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(args.output / f"{name}_seg.nii.gz"))

        row = {"volume": f.name}
        if do_cls:
            img = volume.as_tensor() if hasattr(volume, "as_tensor") else volume
            probs = sliding_window_cls(img, model, device)
            row.update({c: float(p) for c, p in zip(CLS_COLUMNS, probs)})
        rows.append(row)
        print(f"{f.name}: done" + (" | " + " ".join(f"{c}={row[c]:.3f}" for c in CLS_COLUMNS) if do_cls else ""),
              flush=True)

    if do_cls:
        pd.DataFrame(rows).to_csv(args.output / "predictions.csv", index=False, float_format="%.4f")
    print(f"outputs written to {args.output}")


if __name__ == "__main__":
    main()
