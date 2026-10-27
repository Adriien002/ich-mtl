"""Stage 1 training: 3D segmentation (MBH, INSTANCE, optional in-house) + subtype classification (MBH).

Usage:
    ICH_DATA_ROOT=/path/to/datasets python train.py --config configs/final.yaml
    python train.py --config configs/ablations/no_mta.yaml --name no_mta
    python train.py --config configs/final.yaml --set seed=42 train.epochs=10

Checkpoints, logs and the resolved config.yaml go to $ICH_OUTPUT_DIR/checkpoints/<name>.
"""
import argparse
import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from monai.data import DataLoader, PersistentDataset
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset

import config
import data.dataset as dataset
import utils
from data.samplers import build_task_balanced_sampler
from data.transform import get_classification_transforms, get_segmentation_transforms
from models.lightning_module import MultiTaskHemorrhageModule

warnings.filterwarnings("ignore", message=".*no available indices of class.*")
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT_CONFIG, help="YAML run configuration")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. loss.mta_weight=0 seed=42")
    ap.add_argument("--name", help="run name (default: name of the config file)")
    ap.add_argument("--resume", action="store_true", help="resume from <run dir>/last.ckpt if it exists")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases instead of CSV")
    return ap.parse_args()


def build_datasets(cfg, cache_dir):
    """[(name, PersistentDataset)] for train and val, in the order of their ConcatDataset."""
    with_cls = cfg["data"]["classification"]
    cls_crop = cfg["data"]["cls_crop"]
    certified = cfg["loss"]["pseudo_mode"] == "certified"

    def entries(split):
        seg = [
            ("seg_hemo", dataset.get_segmentation_data(split), get_segmentation_transforms(split, "mbh")),
            ("in_house", dataset.get_inhouse_segmentation_data(split), get_segmentation_transforms(split, "in_house")),
            ("instance", dataset.get_instance_segmentation_data(split), get_segmentation_transforms(split, "instance")),
        ]
        if not with_cls:
            return seg
        cls = dataset.get_classification_data(split, cls_crop=cls_crop, certified=certified)
        # Split by sampling group, not by has_mask: with cls_crop=random the positives
        # have no mask but stay their own sub-dataset, with the same sampling mass.
        positives = [d for d in cls if d["cls_group"] == "mask"]
        healthy = [d for d in cls if d["cls_group"] == "nomask"]
        guided = split != "train" or cls_crop != "random"
        # A random crop draws 2 patches per positive, like the guided crop.
        pos_tf = get_classification_transforms(split, guided, num_samples=1 if guided else config.patch_per_case)
        return seg + [
            ("cls_mask", positives, pos_tf),
            ("cls_nomask", healthy, get_classification_transforms(split, False)),
        ]

    out = {}
    for split in ("train", "val"):
        named = []
        for name, data, tf in entries(split):
            if not data:  # e.g. the private in-house dataset when it is absent
                print(f"[{split}] empty dataset ignored: {name}", flush=True)
                continue
            # The cache key depends on the crop of the positives, whose transforms differ.
            tag = f"{name}_{cls_crop}" if (name == "cls_mask" and split == "train") else name
            named.append((name, PersistentDataset(data, transform=tf, cache_dir=os.path.join(cache_dir, f"{tag}_{split}"))))
        out[split] = named
    return out["train"], out["val"]


def main():
    args = parse_args()
    cfg = config.load_config(args.config, args.set)
    name = args.name or Path(args.config).stem
    run_dir = Path(config.SAVE_DIR) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    pl.seed_everything(cfg["seed"], workers=True)
    set_determinism(seed=cfg["seed"])
    train_cfg = cfg["train"]

    # PersistentDataset cache: $SLURM_TMPDIR on a cluster node, otherwise $ICH_CACHE_DIR or ./outputs/cache.
    cache_dir = os.environ.get("ICH_CACHE_DIR", os.path.join(os.environ.get("SLURM_TMPDIR", config.PROJECT_ROOT), "cache"))
    named_train, named_val = build_datasets(cfg, cache_dir)

    # Fill the cache sequentially before starting multi-worker loaders: with a sampler
    # drawing with replacement, two workers could otherwise write the same cache file.
    for ds_name, ds in named_train + named_val:
        for i in range(len(ds)):
            ds[i]
        print(f"[cache] {ds_name}: {len(ds)} volumes", flush=True)

    loader_kw = dict(num_workers=train_cfg["num_workers"], persistent_workers=train_cfg["num_workers"] > 0,
                     pin_memory=torch.cuda.is_available(), collate_fn=utils.multitask_collate_fn)
    train_loader = DataLoader(
        ConcatDataset([d for _, d in named_train]), batch_size=train_cfg["batch_size"],
        sampler=build_task_balanced_sampler(named_train, alpha=cfg["data"]["sampling_alpha"]), **loader_kw)
    val_loader = DataLoader(ConcatDataset([d for _, d in named_val]), batch_size=1, shuffle=False, **loader_kw)

    num_steps = len(train_loader) * train_cfg["epochs"]
    model = MultiTaskHemorrhageModule(cfg, num_steps=num_steps)
    print(f"{name}: {num_steps} steps | {cfg}", flush=True)

    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", "ich-mtl"), name=name, config=cfg)
    else:
        logger = CSVLogger(str(run_dir), name="logs")

    # Best checkpoint = highest mean validation Dice over the segmentation datasets.
    callbacks = [ModelCheckpoint(dirpath=run_dir, filename="best_{epoch:03d}_{val_mean_dice_segds:.4f}",
                                 monitor="val_mean_dice_segds", mode="max", save_top_k=3, save_last=True)]
    if cfg["data"]["classification"]:
        # Classification overfits early: its best checkpoint is kept separately.
        callbacks += [
            ModelCheckpoint(dirpath=run_dir, filename="best_cls_{epoch:03d}_{val_cls_loss:.4f}",
                            monitor="val_cls_loss", mode="min", save_top_k=1),
            ModelCheckpoint(dirpath=run_dir, filename="best_clsall_{epoch:03d}_{val_cls_loss_all:.4f}",
                            monitor="val_cls_loss_all", mode="min", save_top_k=1),
        ]

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        default_root_dir=run_dir,
        logger=logger,
        gradient_clip_val=train_cfg["grad_clip"],
        log_every_n_steps=50,
        check_val_every_n_epoch=train_cfg["val_every"],
        precision="16-mixed" if use_gpu else "32-true",
        callbacks=callbacks,
    )
    last = run_dir / "last.ckpt"
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(last) if args.resume and last.exists() else None)


if __name__ == "__main__":
    main()
