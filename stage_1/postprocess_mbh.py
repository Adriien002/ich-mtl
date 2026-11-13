"""MBH post-processing, from the MBH-SEG 2024 winning solution (submission.py).

Labels: 0 background, 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH.
  1. Every connected component of EDH ∪ SDH that holds both classes is relabelled
     with its majority class.
  2. EDH, SAH or SDH is removed when the class totals fewer than 25 voxels.

Applied to MBH predictions in evaluate.py, as in the paper and for every baseline.

Usage on a folder of predictions (writes a copy, the input is left untouched):
  python postprocess_mbh.py --input preds/ --output preds_postproc/
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label

EDH, SAH, SDH = 1, 4, 5
MIN_VOXELS = 25


def postprocess_mbh(segmentation):
    """Post-process one MBH label map (numpy array); returns a new array of the same dtype."""
    seg = segmentation.astype(np.int32)

    if np.any(seg == EDH) and np.any(seg == SDH):
        labeled_mask, num_features = label((seg == EDH) | (seg == SDH))
        for i in range(1, num_features + 1):
            component = labeled_mask == i
            classes, counts = np.unique(seg[component], return_counts=True)
            if EDH in classes and SDH in classes:
                seg[component] = classes[np.argmax(counts)]

    for c in (EDH, SAH, SDH):
        n = np.sum(seg == c)
        if 0 < n < MIN_VOXELS:
            seg[seg == c] = 0

    return seg.astype(segmentation.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of predicted .nii.gz label maps")
    ap.add_argument("--output", required=True, help="folder for the post-processed maps")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input).glob("*.nii.gz"))
    for f in files:
        img = nib.load(f)
        seg = postprocess_mbh(img.get_fdata().astype(np.int32))
        nib.save(nib.Nifti1Image(seg, img.affine, img.header), out_dir / f.name)
    print(f"{len(files)} files written to {out_dir}")


if __name__ == "__main__":
    main()
