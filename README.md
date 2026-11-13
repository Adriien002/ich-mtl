# ich-mtl

Code for the paper **"A Unified 3D Multi-Task Network for Intracranial Hemorrhage Segmentation,
Subtyping and Expansion Prediction from Heterogeneously Annotated CT"**, currently under review
at the *Journal of Medical Imaging* (SPIE JMI).

## Repository layout

| Folder | Content | Status |
|---|---|---|
| [`stage_1/`](stage_1) | Multi-task network: 3D hemorrhage segmentation and subtype classification | available |
| `stage_2/` | Hematoma expansion prediction (pseudo-mask preprocessing, linear probe) | coming soon |
| [`splits/`](splits) | Train/val/test splits used in the paper | available |
| [`docs/DATA.md`](docs/DATA.md) | Dataset download and folder layout | available |

## Reproducibility

The paper uses public datasets (MBH, INSTANCE 2022 and PhysioNet CT-ICH) and an in-house
dataset that will be released through the RSNA Medical Imaging and Resource Archive (MIRA),
pending RSNA approval. Training runs without the in-house dataset.

The exact splits are in [`splits/`](splits). Follow [`docs/DATA.md`](docs/DATA.md) to download
the datasets and arrange them under `$ICH_DATA_ROOT`.

## Installation

Python 3.10. Install the PyTorch build that matches your CUDA version first
(<https://pytorch.org/get-started/locally/>), then:

```bash
pip install -r requirements.txt
```

## Stage 1: segmentation and subtype classification

Run the scripts from `stage_1/`. Two environment variables set the paths:

- `ICH_DATA_ROOT`: root of the datasets (default: `stage_1/datasets`), layout in [`docs/DATA.md`](docs/DATA.md);
- `ICH_OUTPUT_DIR`: checkpoints, caches and evaluation outputs (default: `stage_1/outputs`).

```bash
cd stage_1
export ICH_DATA_ROOT=/path/to/datasets
export ICH_OUTPUT_DIR=/path/to/outputs

# Train the final model of the paper
python train.py --config configs/final.yaml

# Ablations (configs/ablations/): seg_only, no_mta, no_pseudo_mask, equal_weights, ...
python train.py --config configs/ablations/no_mta.yaml

# Test-set Dice (Table 2) and subtype AUC (Table 3)
python evaluate.py --ckpt $ICH_OUTPUT_DIR/checkpoints/final/<best>.ckpt --cls

# Inference on new CT scans (NIfTI file or folder)
python predict.py --ckpt path/to/best.ckpt --input scan.nii.gz --output out_dir
```

Each script lists all its options with `--help`.

**Pretrained weights.** The checkpoint of the model reported in the paper will be released here.

## Stage 2: hematoma expansion prediction

Coming soon. The expansion cohort is private, so this stage will mainly document the method:
how the pseudo-masks are preprocessed and how the linear probe is trained on the
features of the Stage 1 encoder.

## Citation

The paper is under review; the citation will be added after publication.
