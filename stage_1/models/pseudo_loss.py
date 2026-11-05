"""Label-gated loss on the nnU-Net pseudo-masks of the classification positives (Sec. 3.2.2, Fig. 3).

The pseudo-masks are noisy (their subtype content matches the case-level label for only
44% of the volumes), whereas the case-level labels are true annotations. The loss only
keeps what the case-level label can certify, channel by channel:

  A. Subtype present in the label and in the patch: soft Dice on that channel. The
     pseudo-mask is trusted for WHERE the lesion is, not for whether it exists.
     A subtype present in the label but absent from the patch is not supervised:
     the lesion may lie outside the patch.
  B. Subtype absent from the label: every voxel is a certified negative. Penalises
     the generalized mean (p=3) of the channel probability over the patch, which
     corrects the false positives of the pseudo-mask.
  C. Background, at a reduced weight, only on volumes where the pseudo-mask misses
     none of the labelled subtypes (valid[:, 0] = 1).

The (B, 6) `valid` flags come from tools/audit_pseudo_masks.py, through data/dataset.py:
    valid[:, 0]   1 if no labelled subtype is missing from the pseudo-mask
    valid[:, 1:6] 0 for a subtype present in the label but absent from the pseudo-mask

Channels: seg 0 = background, 1..5 = EDH/IPH/IVH/SAH/SDH; class_label 0 = 'any', 1..5 = subtypes.
"""
import torch
import torch.nn as nn


class CertifiedPseudoLoss(nn.Module):
    """w_bg: weight of term C (0 makes the loss fully certified). gem_p: exponent of term B."""

    def __init__(self, w_bg: float = 0.5, gem_p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.w_bg, self.gem_p, self.eps = w_bg, gem_p, eps

    def forward(self, seg_logits, y_seg, y_cls, valid):
        """
        seg_logits: (B, 6, H, W, D) logits of the MBH head
        y_seg:      (B, 1, H, W, D) pseudo-mask, integer labels 0..5
        y_cls:      (B, 6) case-level label, 0/1
        valid:      (B, 6) audit flags
        Returns (loss, parts), parts being detached scalars for logging.
        """
        p = torch.softmax(seg_logits, dim=1)
        tgt = y_seg.squeeze(1).long()
        B = p.shape[0]
        gt = y_cls[:, 1:6]
        val = valid[:, 1:6]

        onehot = torch.stack([(tgt == c) for c in range(1, 6)], dim=1).float()   # (B,5,H,W,D)
        in_patch = onehot.flatten(2).sum(-1) > 0                                  # (B,5)
        m_loc = (gt > 0.5) & (val > 0.5) & in_patch                               # term A
        m_abs = gt < 0.5                                                          # term B
        pc = p[:, 1:6]

        # A. soft Dice per channel, averaged over the selected channels
        inter = (pc * onehot).flatten(2).sum(-1)
        denom = pc.flatten(2).sum(-1) + onehot.flatten(2).sum(-1)
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        n_loc = m_loc.sum()
        loss_loc = (dice * m_loc).sum() / n_loc.clamp(min=1) if n_loc > 0 else pc.sum() * 0.0

        # B. certified absence: BCE towards 0 of the GeM-pooled channel probability
        pooled = pc.clamp(min=self.eps).pow(self.gem_p).flatten(2).mean(-1).pow(1.0 / self.gem_p)
        bce0 = -torch.log((1.0 - pooled).clamp(min=self.eps))
        n_abs = m_abs.sum()
        loss_abs = (bce0 * m_abs).sum() / n_abs.clamp(min=1) if n_abs > 0 else bce0.sum() * 0.0

        # C. background NLL, volumes without a missed subtype only
        bg_ok = valid[:, 0] > 0.5
        if self.w_bg > 0 and bg_ok.any():
            is_bg = (tgt == 0).float()
            nll_bg = -torch.log(p[:, 0].clamp(min=self.eps)) * is_bg
            per_sample = nll_bg.flatten(1).sum(-1) / is_bg.flatten(1).sum(-1).clamp(min=1.0)
            loss_bg = (per_sample * bg_ok.float()).sum() / bg_ok.sum().clamp(min=1)
        else:
            loss_bg = seg_logits.sum() * 0.0

        loss = loss_loc + loss_abs + self.w_bg * loss_bg
        parts = {
            "pseudo_loc": loss_loc.detach(),
            "pseudo_abs": loss_abs.detach(),
            "pseudo_bg": loss_bg.detach(),
            "pseudo_n_loc_per_sample": n_loc.float().detach() / max(B, 1),
            "pseudo_n_abs_per_sample": n_abs.float().detach() / max(B, 1),
            "pseudo_frac_bg_used": bg_ok.float().mean().detach(),
        }
        return loss, parts
