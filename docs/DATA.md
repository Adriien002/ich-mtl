# Data

The code reads every dataset under one root, set by `ICH_DATA_ROOT` (default: `stage_1/datasets`).
The splits used in the paper are in [`splits/`](../splits).

```
$ICH_DATA_ROOT/
├── mbh/{train,val,test}/{img,seg}/*.nii.gz        MBH segmentation        splits/mbh_seg.csv
├── instance/{train,val,test}/{img,seg}/*.nii.gz   INSTANCE 2022           splits/instance.csv
├── in_house/{train,val,test}/{img,seg}/*.nii.gz   In-house (see below)
├── MBH_label_case/<patientID_studyID>.nii.gz      MBH classification
│   └── splits/*.csv                               copy of splits/MBH_label_case/
├── Pseudo_Masks/<patientID_studyID>.nii.gz        pseudo-masks (see below)
└── physionet/{ct_scans,masks}/*.nii               PhysioNet CT-ICH, external test only
```

In `img/` and `seg/`, a mask must have the same file name as its image.

| Dataset | Source | Notes |
|---|---|---|
| MBH (segmentation and classification) | [Zenodo](https://doi.org/10.5281/zenodo.15094748) | Several annotators labeled the segmentation volumes: we use the masks of **Laurent Létourneau-Guillon** (files `MBH_SEG_2025_LLG_*`, co-author of the paper). |
| INSTANCE 2022 | [instance-challenge.org](https://instance-challenge.org/) | binary masks |
| In-house | not yet public | 359 RSNA 2019 scans annotated for ICH, IVH and PHE (Wu et al.). Will be released through the RSNA Medical Imaging and Resource Archive (MIRA), pending RSNA approval. Optional: training runs without it. |
| PhysioNet CT-ICH | [PhysioNet](https://doi.org/10.13026/G1KQ-SC84) | keep the original layout; masks coded 0/255 |

**MBH labels.** Segmentation masks and pseudo-masks use 0 background, 1 EDH, 2 IPH, 3 IVH,
4 SAH, 5 SDH, the order of `config.CLASS_NAMES`. Remap masks stored in another order: a
wrong order raises no error, it silently swaps the classes.

**Pseudo-masks.** Each classification positive needs a pseudo-mask, used to center its
crop; positives without one are dropped. Ours come from an nnU-Net trained on the 165 MBH
training volumes. A neuroradiologist is reviewing them, and the revised masks will be
released. The certified pseudo-mask loss also needs `Pseudo_Masks/pseudo_mask_audit.csv`,
written by `python stage_1/tools/audit_pseudo_masks.py`.
