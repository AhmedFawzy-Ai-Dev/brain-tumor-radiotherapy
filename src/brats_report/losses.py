"""Loss and metric for multi-label segmentation (WT / TC / ET)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    """1 - Dice, averaged over channels, on sigmoid probabilities."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.w = bce_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.w * bce + (1 - self.w) * soft_dice_loss(logits, target)


@torch.no_grad()
def dice_per_channel(logits, target, thresh=0.5, eps=1e-6):
    """Hard Dice per channel -> tensor of shape (C,)."""
    probs = (torch.sigmoid(logits) > thresh).float()
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (denom + eps)
