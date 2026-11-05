"""3D multi-task U-Net: shared encoder, (partially) shared decoder, one segmentation head
per dataset and a classification head on the bottleneck (Sec. 3.2.1, Fig. 2)."""
from collections.abc import Sequence

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep


class BasicUNetWithClassification(nn.Module):
    """MONAI BasicUNet with per-dataset segmentation heads and a subtype classification head.

    task selects the segmentation head: "seg_inhouse", "seg_instance", or any other
    value for the MBH head (6 classes), which is also the one used by the classification
    tasks and the presence-consistency loss.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels_orig: int = 6,       # MBH: background + 5 subtypes
        out_channels_inhouse: int = 4,    # in-house: background, ICH, IVH, PHE
        out_channels_instance: int = 2,   # INSTANCE: background, hemorrhage
        num_cls_classes: int = 6,         # any + 5 subtypes
        decoder_sharing: str = "last_block",  # "none" | "last_block" | "full"
        features: Sequence[int] = (32, 64, 128, 256, 512, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        if decoder_sharing not in ("none", "last_block", "full"):
            raise ValueError(f"invalid decoder_sharing: {decoder_sharing!r}")
        self.decoder_sharing = decoder_sharing
        fea = ensure_tuple_rep(features, 6)

        # Shared encoder
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        def upcat(i):  # decoder block i (4 = deepest), same as MONAI BasicUNet
            if i == 1:
                return UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
            return UpCat(spatial_dims, fea[i], fea[i - 1], fea[i - 1], act, norm, bias, dropout, upsample)

        def conv1x1(out_channels):
            return Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        outs = {"orig": out_channels_orig, "inhouse": out_channels_inhouse, "instance": out_channels_instance}
        if decoder_sharing == "none":
            # One full decoder per dataset
            for name, out in outs.items():
                setattr(self, f"dec_{name}", nn.ModuleDict({
                    "upcat_4": upcat(4), "upcat_3": upcat(3), "upcat_2": upcat(2), "upcat_1": upcat(1),
                    "final_conv": conv1x1(out),
                }))
        else:
            # Shared decoder trunk (upcat_4 -> upcat_2)
            self.upcat_4, self.upcat_3, self.upcat_2 = upcat(4), upcat(3), upcat(2)
            if decoder_sharing == "last_block":
                # Dataset-specific last block + 1x1 conv
                for name, out in outs.items():
                    setattr(self, f"head_{name}", nn.ModuleDict({"upcat_1": upcat(1), "final_conv": conv1x1(out)}))
            else:
                # Shared last block, dataset-specific 1x1 conv only
                self.upcat_1 = upcat(1)
                for name, out in outs.items():
                    setattr(self, f"head_{name}", conv1x1(out))

        # Classification head on the bottleneck (down_4)
        self.cls_head = nn.Sequential(
            nn.Conv3d(fea[4], fea[4], kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.AdaptiveMaxPool3d((2, 2, 2)),
            nn.Flatten(),
            nn.Linear(fea[4] * 8, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_cls_classes),
        )
        # Unused in the forward pass; kept so that the released checkpoints load with strict=True.
        self.cls_spatial_proj = Conv["conv", spatial_dims](fea[4], out_channels_orig, kernel_size=1)

    def _decode(self, task, x0, x1, x2, x3, x4):
        name = {"seg_inhouse": "inhouse", "seg_instance": "instance"}.get(task, "orig")
        if self.decoder_sharing == "none":
            dec = getattr(self, f"dec_{name}")
            u2 = dec["upcat_2"](dec["upcat_3"](dec["upcat_4"](x4, x3), x2), x1)
            return dec["final_conv"](dec["upcat_1"](u2, x0))

        u2 = self.upcat_2(self.upcat_3(self.upcat_4(x4, x3), x2), x1)
        head = getattr(self, f"head_{name}")
        if self.decoder_sharing == "full":
            return head(self.upcat_1(u2, x0))
        return head["final_conv"](head["upcat_1"](u2, x0))

    def forward(self, x: torch.Tensor, task: str = "seg_orig"):
        """Return (seg_logits, cls_logits); cls_logits is None for the in-house and INSTANCE heads."""
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        cls_logits = None if task in ("seg_inhouse", "seg_instance") else self.cls_head(x4)
        return self._decode(task, x0, x1, x2, x3, x4), cls_logits
