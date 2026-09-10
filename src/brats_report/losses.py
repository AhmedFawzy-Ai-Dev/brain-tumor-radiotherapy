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


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    """Tversky loss: alpha weights false positives, beta false negatives.
    beta>alpha penalises misses more - better recall on small tumours."""
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    tp = (p * target).sum(dims)
    fp = (p * (1 - target)).sum(dims)
    fn = ((1 - p) * target).sum(dims)
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - t.mean()


class TverskyBCELoss(nn.Module):
    """BCE + Tversky - stronger on small-lesion recall than Dice+BCE."""

    def __init__(self, bce_weight=0.3, alpha=0.3, beta=0.7):
        super().__init__()
        self.w, self.alpha, self.beta = bce_weight, alpha, beta

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.w * bce + (1 - self.w) * tversky_loss(logits, target, self.alpha, self.beta)


@torch.no_grad()
def dice_per_channel(logits, target, thresh=0.5, eps=1e-6):
    """Hard Dice per channel -> tensor of shape (C,)."""
    probs = (torch.sigmoid(logits) > thresh).float()
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (denom + eps)
