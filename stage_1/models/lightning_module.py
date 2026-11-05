"""Training and validation logic: losses of Eq. (5) and checkpoint-selection metrics."""
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper, DiceMetric
from monai.networks.utils import one_hot
from torch.optim import SGD
from torchmetrics.classification import MultilabelAUROC, MultilabelPrecision, MultilabelRecall

import config
import data.dataset as dataset
from models.architecture import BasicUNetWithClassification
from models.pseudo_loss import CertifiedPseudoLoss

# Bounds of the learned log-variances s_t (Eq. 12): weight 1/(2 exp(s)) in [0.30, 1.50].
UW_LOG_VAR_MIN, UW_LOG_VAR_MAX = -1.1, 0.5
ROI_SIZE = (96, 96, 96)


class MultiTaskHemorrhageModule(pl.LightningModule):
    """cfg: run configuration (configs/*.yaml). num_steps: total optimizer steps, for the LR decay."""

    def __init__(self, cfg: dict, num_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_steps = num_steps
        loss = cfg["loss"]
        self.cls_weight = loss["cls_weight"]
        self.cls_target = loss["cls_target"]
        self.mta_weight = loss["mta_weight"]
        self.pseudo_weight = loss["pseudo_weight"]
        self.pseudo_mode = loss["pseudo_mode"]
        self.pseudo_loss_fn = CertifiedPseudoLoss(w_bg=loss["pseudo_bg_weight"]) if self.pseudo_mode == "certified" else None
        self.seg_weighting = loss["seg_weighting"]
        self.seg_fixed_weight = loss["seg_fixed_weight"]

        self.model = BasicUNetWithClassification(
            spatial_dims=3, in_channels=1, out_channels_orig=6, out_channels_inhouse=4,
            num_cls_classes=config.NUM_CLASSES, decoder_sharing=cfg["model"]["decoder_sharing"],
        )
        print(f"decoder_sharing={cfg['model']['decoder_sharing']} | "
              f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        # BCE pos_weight = n_neg / n_pos on the training labels, clipped to [1, 10] (Eq. 10).
        # Without training data (inference), ones; the checkpoint then restores the value.
        try:
            pos_weights = torch.clip(dataset.compute_pos_weights(split="train"), min=1.0, max=10.0)
        except (FileNotFoundError, OSError):
            pos_weights = torch.ones(config.NUM_CLASSES)
        self.register_buffer("pos_weights", pos_weights)

        self.seg_loss_fn_mbh = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.seg_loss_fn_inhouse = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.seg_loss_fn_instance = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        # Log-variances of the 3 segmentation losses (0 = MBH, 1 = in-house, 2 = INSTANCE).
        # Classification keeps the fixed weight cls_weight (Sec. 3.2.4). In "fixed" mode the
        # parameter is frozen, but kept so that the state_dict is the same in both modes.
        self.uw_log_var = nn.Parameter(torch.tensor([-0.5, -0.5, -0.5]))
        if self.seg_weighting == "fixed":
            self.uw_log_var.requires_grad_(False)

        # Per-volume Dice (logged per class) and epoch Dice (checkpoint selection)
        self.seg_orig_dice = DiceHelper(include_background=False, softmax=True, num_classes=6, reduction='none')
        self.seg_inhouse_dice = DiceHelper(include_background=False, softmax=True, num_classes=4, reduction='none')
        self.seg_instance_dice = DiceHelper(include_background=False, softmax=True, num_classes=2, reduction='none')
        self.val_dice = {
            "mbh": DiceMetric(include_background=False, reduction="mean"),
            "inhouse": DiceMetric(include_background=False, reduction="mean"),
            "instance": DiceMetric(include_background=False, reduction="mean"),
        }
        self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)

    def forward(self, x, task="seg_orig"):
        return self.model(x, task=task)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _seg_weights(self):
        """Effective weights of the 3 segmentation losses."""
        if self.seg_weighting == "fixed":
            return torch.full((3,), self.seg_fixed_weight, device=self.device)
        s = torch.clamp(self.uw_log_var, min=UW_LOG_VAR_MIN, max=UW_LOG_VAR_MAX)
        return 1.0 / (2.0 * s.exp())

    def _uw(self, loss, seg_idx):
        """Weighted segmentation loss: Kendall et al. with clamped s (Eq. 11-12), or a constant."""
        if self.seg_weighting == "fixed":
            return self.seg_fixed_weight * loss
        s = torch.clamp(self.uw_log_var[seg_idx], min=UW_LOG_VAR_MIN, max=UW_LOG_VAR_MAX)
        return loss / (2.0 * s.exp()) + s / 2.0

    @staticmethod
    def _patch_cls_target(y_seg):
        """(B,1,H,W,D) mask of the sampled patch -> (B,6) presence vector ('any' + 5 subtypes).

        The case-level label describes the whole volume, while a 96^3 patch often contains
        only part of its subtypes. Used for MBH volumes with a manual mask (cls_target=patch).
        """
        sub = torch.stack([(y_seg == c).flatten(1).any(dim=1) for c in range(1, 6)], dim=1)
        return torch.cat([sub.any(dim=1, keepdim=True), sub], dim=1).float()

    @staticmethod
    def _consist_loss(cls_logits, seg_logits, eps=1e-6, p=3.0):
        """Presence-consistency loss L_MTA (Eqs. 1-3): MSE between sigmoid(cls) and the
        GeM-pooled (p=3) presence of each label in the MBH segmentation of the same patch.
        No detach: the gradient reaches both heads."""
        soft = torch.softmax(seg_logits, dim=1)
        seg_presence = torch.cat([1.0 - soft[:, 0:1], soft[:, 1:]], dim=1)   # 'any' = 1 - background
        pooled = F.adaptive_avg_pool3d(seg_presence.clamp(min=eps).pow(p), 1).pow(1.0 / p).flatten(1)
        return F.mse_loss(torch.sigmoid(cls_logits), pooled)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        # Classification patches without dense target (healthy volumes, or positives
        # with cls_crop=random). L_MTA on the healthy ones pushes the segmentation to empty.
        if batch.get("classification") is not None:
            x, y = batch["classification"]["image"], batch["classification"]["class_label"]
            seg_logits, cls_logits = self.model(x)
            cls_loss = self.cls_loss_fn(cls_logits, y)
            total_loss += self.cls_weight * cls_loss
            self.log("train_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)
            if self.mta_weight > 0:
                healthy = (y.sum(dim=1) == 0)
                if healthy.any():
                    lmta = self._consist_loss(cls_logits[healthy], seg_logits[healthy])
                    total_loss += self.mta_weight * lmta
                    self.log("train_lmta_cls", lmta, batch_size=int(healthy.sum()), on_epoch=True)

        # Classification positives with a pseudo-mask: BCE + weak segmentation target.
        # No L_MTA: the case-level label is not reliable at the patch level.
        if batch.get("cls_pseudo") is not None:
            x = batch["cls_pseudo"]["image"]
            y_cls = batch["cls_pseudo"]["class_label"]
            y_seg = batch["cls_pseudo"]["label"]
            seg_logits, cls_logits = self.model(x)
            cls_loss = self.cls_loss_fn(cls_logits, y_cls)
            if self.pseudo_mode == "certified":
                valid = batch["cls_pseudo"].get("valid_channels")
                if valid is None:
                    valid = torch.ones_like(y_cls)
                seg_loss, parts = self.pseudo_loss_fn(seg_logits, y_seg, y_cls, valid)
                for k, v in parts.items():
                    self.log(f"train_{k}", v, batch_size=x.shape[0], on_epoch=True)
            else:
                seg_loss = self.seg_loss_fn_mbh(seg_logits, y_seg)
            total_loss += self.cls_weight * cls_loss + self.pseudo_weight * seg_loss
            self.log("train_cls_pseudo_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("train_cls_pseudo_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        # MBH segmentation, volumes without case-level label
        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            seg_logits, cls_logits = self.model(x)
            seg_loss = self.seg_loss_fn_mbh(seg_logits, y)
            total_loss += self._uw(seg_loss, 0)
            self.log("train_seg_orig_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            if self.mta_weight > 0:
                lmta = self._consist_loss(cls_logits, seg_logits)
                total_loss += self.mta_weight * lmta
                self.log("train_lmta_seg", lmta, batch_size=x.shape[0], on_epoch=True)

        # MBH segmentation + classification (manual mask and case-level label)
        if batch.get("seg_multi") is not None:
            x = batch["seg_multi"]["image"]
            y_seg = batch["seg_multi"]["label"]
            y_cls = batch["seg_multi"]["class_label"]
            if self.cls_target == "patch":
                y_cls = self._patch_cls_target(y_seg).to(y_cls.dtype)
            seg_logits, cls_logits = self.model(x)
            seg_loss = self.seg_loss_fn_mbh(seg_logits, y_seg)
            cls_loss = self.cls_loss_fn(cls_logits, y_cls)
            total_loss += self._uw(seg_loss, 0) + self.cls_weight * cls_loss
            self.log("train_seg_multi_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("train_seg_multi_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)
            if self.mta_weight > 0:
                lmta = self._consist_loss(cls_logits, seg_logits)
                total_loss += self.mta_weight * lmta
                self.log("train_lmta_multi", lmta, batch_size=x.shape[0], on_epoch=True)

        # In-house segmentation
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            seg_logits, _ = self.model(x, task="seg_inhouse")
            seg_loss = self.seg_loss_fn_inhouse(seg_logits, y)
            total_loss += self._uw(seg_loss, 1)
            self.log("train_seg_inhouse_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        # INSTANCE segmentation
        if batch.get("seg_instance") is not None:
            x, y = batch["seg_instance"]["image"], batch["seg_instance"]["label"]
            seg_logits, _ = self.model(x, task="seg_instance")
            seg_loss = self.seg_loss_fn_instance(seg_logits, y)
            total_loss += self._uw(seg_loss, 2)
            self.log("train_seg_instance_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            w = self._seg_weights()
        self.log_dict({"uw_seg_orig": w[0], "uw_seg_inhouse": w[1], "uw_seg_instance": w[2],
                       "uw_cls": self.cls_weight}, on_epoch=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _update_cls_metrics(self, cls_logits, y):
        y_pred = torch.sigmoid(cls_logits)
        self.cls_auc.update(y_pred, y.int())
        self.cls_mean_auc.update(y_pred, y.int())

    def _validate_seg(self, x, y, task, key, num_classes, dice_helper, prefix):
        """Sliding-window segmentation of a whole validation volume; returns (logits, loss)."""
        y_hat = sliding_window_inference(x, roi_size=ROI_SIZE, sw_batch_size=2,
                                         predictor=lambda img: self.model(img, task=task)[0])
        loss_fn = {"mbh": self.seg_loss_fn_mbh, "inhouse": self.seg_loss_fn_inhouse,
                   "instance": self.seg_loss_fn_instance}[key]
        loss = loss_fn(y_hat, y)
        pred = torch.softmax(y_hat, dim=1).argmax(dim=1, keepdim=True)
        self.val_dice[key](one_hot(pred, num_classes, dim=1), one_hot(y, num_classes, dim=1))
        scores, _ = dice_helper(y_hat, y)
        present = y.unique().long().tolist()[1:]
        if present:
            self.log_dict({f"{prefix}{c}": scores[0][c - 1].item() for c in present},
                          on_epoch=True, prog_bar=(prefix != "dice_mbh_multi_c"))
        return y_hat, loss

    def validation_step(self, batch, batch_idx):
        total_loss = 0.0

        # Classification patches (healthy volumes and pseudo-masked positives, guided crop)
        for task in ("classification", "cls_pseudo"):
            if batch.get(task) is not None:
                x, y = batch[task]["image"], batch[task]["class_label"]
                _, cls_logits = self.model(x)
                loss = self.cls_loss_fn(cls_logits, y)
                self._update_cls_metrics(cls_logits, y)
                total_loss += self.cls_weight * loss
                self.log("val_cls_loss", loss, batch_size=x.shape[0], on_epoch=True)
                self.log("val_cls_loss_all", loss, batch_size=x.shape[0], on_epoch=True)

        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            _, loss = self._validate_seg(x, y, "seg_orig", "mbh", 6, self.seg_orig_dice, "dice_mbh_c")
            total_loss += self._uw(loss, 0)
            self.log("val_seg_loss", loss, batch_size=x.shape[0])

        # MBH volumes with a manual mask: Dice on the whole volume, BCE on the whole volume.
        # They enter val_cls_loss_all only, so that val_cls_loss covers the pseudo-masked positives.
        if batch.get("seg_multi") is not None:
            x, y_seg, y_cls = batch["seg_multi"]["image"], batch["seg_multi"]["label"], batch["seg_multi"]["class_label"]
            _, seg_loss = self._validate_seg(x, y_seg, "seg_orig", "mbh", 6, self.seg_orig_dice, "dice_mbh_multi_c")
            _, cls_logits = self.model(x)
            cls_loss = self.cls_loss_fn(cls_logits, y_cls)
            self._update_cls_metrics(cls_logits, y_cls)
            total_loss += self._uw(seg_loss, 0) + self.cls_weight * cls_loss
            self.log("val_seg_multi_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("val_seg_multi_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("val_cls_loss_all", cls_loss, batch_size=x.shape[0], on_epoch=True)

        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            _, loss = self._validate_seg(x, y, "seg_inhouse", "inhouse", 4, self.seg_inhouse_dice, "dice_inhouse_c")
            total_loss += self._uw(loss, 1)
            self.log("val_seg_inhouse_loss", loss, batch_size=x.shape[0])

        if batch.get("seg_instance") is not None:
            x, y = batch["seg_instance"]["image"], batch["seg_instance"]["label"]
            _, loss = self._validate_seg(x, y, "seg_instance", "instance", 2, self.seg_instance_dice, "dice_instance_c")
            total_loss += self._uw(loss, 2)
            self.log("val_seg_instance_loss", loss, batch_size=x.shape[0])

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if len(self.cls_auc.preds) > 0:
            self.log_dict({
                'val_mean_auc': self.cls_mean_auc.compute(),
                'val_mean_precision': self.cls_mean_precision.compute(),
                'val_mean_recall': self.cls_mean_recall.compute(),
            }, on_epoch=True, prog_bar=True)
            per_class = self.cls_auc.compute()
            for i in range(config.NUM_CLASSES):
                self.log(f'val_auc_class_{i}', per_class[i].item(), on_epoch=True)
            for metric in (self.cls_auc, self.cls_mean_auc, self.cls_mean_precision, self.cls_mean_recall):
                metric.reset()

        # Checkpoint selection: mean over the segmentation datasets of their epoch Dice
        dice_means = {}
        for name, metric in self.val_dice.items():
            try:
                agg = metric.aggregate()
                val = agg.item() if torch.is_tensor(agg) else float(agg)
            except ValueError:
                continue
            finally:
                metric.reset()
            if val == val:  # skip NaN
                dice_means[name] = val
                self.log(f"val_dice_{name}", val, on_epoch=True, prog_bar=True)
        if dice_means:
            self.log("val_mean_dice_segds", sum(dice_means.values()) / len(dice_means), on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        train = self.cfg["train"]
        uw_params = [p for n, p in self.named_parameters() if "uw_log_var" in n]
        net_params = [p for n, p in self.named_parameters() if "uw_log_var" not in n]

        # Network: SGD Nesterov, linear decay to 0. Log-variances: own group, constant LR.
        param_groups = [{"params": net_params, "lr": train["lr"], "momentum": 0.99,
                         "weight_decay": train["weight_decay"], "nesterov": True}]
        lr_lambdas = [lambda step: max(0.0, float(self.num_steps - step) / float(max(1, self.num_steps)))]
        if self.seg_weighting == "learned":
            param_groups.append({"params": uw_params, "lr": 1e-2, "momentum": 0.0, "weight_decay": 0.0, "nesterov": False})
            lr_lambdas.append(lambda step: 1.0)

        optimizer = SGD(param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}


def load_trained_model(ckpt_path, config_path=None, map_location="cpu"):
    """Rebuild the model of a run and load its checkpoint weights (strict).

    config_path defaults to the config.yaml written by train.py next to the checkpoint,
    then to configs/final.yaml. A decoder_sharing that does not match the checkpoint
    raises an error instead of silently loading part of the weights.
    """
    ckpt_path = Path(ckpt_path)
    if config_path is None:
        run_cfg = ckpt_path.parent / "config.yaml"
        config_path = run_cfg if run_cfg.exists() else config.DEFAULT_CONFIG
    cfg = config.load_config(config_path)
    model = MultiTaskHemorrhageModule(cfg, num_steps=1)
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)["state_dict"]
    model.load_state_dict(state)
    print(f"loaded {ckpt_path.name} with {config_path}", flush=True)
    return model
