"""Run a trained U-Net over a whole volume and assemble a 3-D label map.

Predicts each axial slice, resizes predictions back to the scan's native
resolution (so measurements stay in true millimetres), enforces the nested
WT>=TC>=ET relationship, and keeps the largest connected tumour component.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.transform import resize

from .dataset import _normalize
from .io import Scan
from .model import UNet2D


def load_model(path: str | Path, device="cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = UNet2D(in_ch=ckpt["in_ch"], out_ch=ckpt["out_ch"], base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_label(scan: Scan, model, ckpt, device="cpu", thresh=0.5,
                  keep_largest=True, batch=16) -> np.ndarray:
    if not scan.is_multimodal or scan.n_modalities != ckpt["in_ch"]:
        raise ValueError(f"expected {ckpt['in_ch']}-channel image, got {scan.data.shape}")
    size = ckpt["size"]
    vol = _normalize(scan.data.astype(np.float32))          # (H,W,D,C)
    H, W, D, C = vol.shape
    probs = np.zeros((3, H, W, D), np.float32)

    ks = list(range(D))
    for s in range(0, D, batch):
        chunk = ks[s:s + batch]
        stack = np.stack([resize(vol[:, :, k, :], (size, size, C), order=1,
                                 mode="constant", anti_aliasing=True) for k in chunk])
        x = torch.from_numpy(stack.transpose(0, 3, 1, 2).astype(np.float32)).to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()           # (n,3,size,size)
        for bi, k in enumerate(chunk):
            for c in range(3):
                probs[c, :, :, k] = resize(p[bi, c], (H, W), order=1,
                                           mode="constant", anti_aliasing=False)

    wt = probs[0] > thresh
    tc = (probs[1] > thresh) & wt
    et = (probs[2] > thresh) & tc

    if keep_largest and wt.any():
        lab, n = ndimage.label(wt)
        if n > 1:
            biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            keep = lab == biggest
            wt &= keep
            tc &= keep
            et &= keep

    label = np.zeros((H, W, D), np.uint8)
    label[wt] = 1          # edema / whole-tumour shell
    label[tc] = 2          # tumour core (non-enhancing)
    label[et] = 3          # enhancing tumour
    return label
