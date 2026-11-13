# Data

The code reads every dataset under one root, set by `ICH_DATA_ROOT` (default: `stage_1/datasets`).
The splits used in the paper are in [`splits/`](../splits):

- `splits/mbh_seg.csv` and `splits/instance.csv` (`file,split`): put each image in
  `<dataset>/<split>/img/` and its mask, under the same file name, in `<dataset>/<split>/seg/`.
- `splits/MBH_label_case/*.csv`: copy them as is into `$ICH_DATA_ROOT/MBH_label_case/splits/`.
- `splits/in_house.csv` (`file,patientID,rsna_image_id,split`): RSNA 2019 patient IDs.
  When a file name carries `_instUid_<id>`, `<id>` is the RSNA image ID of one slice of the
  scan and tells apart the studies of a patient who has several.

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
| MBH (segmentation and classification) | [Zenodo](https://doi.org/10.5281/zenodo.15094748) | Several annotators labeled the segmentation volumes. The 238 segmentation volumes come from two sources: 192 use the masks of **Laurent Létourneau-Guillon** (co-author of the paper), stored as `MBH_SEG_2025_LLG_2025_06_12_<patientID_studyID>.nii.gz` (in the Zenodo release, the LLG mask is `label_annot_1.nii.gz` when it exists, otherwise `label_annot_2.nii.gz`); 46 come from the MBH challenge validation set, where each case folder holds `image.nii.gz` and one mask per annotator (`label_annot_1.nii.gz`, `label_annot_2.nii.gz`, ...): we use only `label_annot_5.nii.gz`, stored as `MBH_v2_val_5_<patientID_studyID>.nii.gz`; the other annotators' masks are not used. `splits/mbh_seg.csv` gives the file name and split of each volume. Inter-annotator variability on MBH is high: training or evaluating with masks from other annotators will change the segmentation results. |
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
