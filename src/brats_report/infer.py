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

from . import DEFAULT_POST, POST
from .dataset import _normalize
from .io import Scan
from .model import UNet2D


def load_model(path: str | Path, device="cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = UNet2D(in_ch=ckpt["in_ch"], out_ch=ckpt["out_ch"], base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _predict_slices(vol, model, size, device, batch, flip_axis=None):
    """Sigmoid maps (3, H, W, D) for a normalised (H, W, D, C) volume, one axial
    slice at a time, resized back to native resolution. ``flip_axis`` (0 or 1, an
    in-plane axis) mirrors each slice along it before the model and un-mirrors
    the output."""
    H, W, D, C = vol.shape
    probs = np.zeros((3, H, W, D), np.float32)
    for s in range(0, D, batch):
        chunk = list(range(s, min(s + batch, D)))
        stack = np.stack([resize(vol[:, :, k, :], (size, size, C), order=1,
                                 mode="constant", anti_aliasing=True) for k in chunk])
        if flip_axis is not None:
            stack = np.flip(stack, axis=1 + flip_axis)      # (n, size, size, C)
        x = torch.from_numpy(np.ascontiguousarray(
            stack.transpose(0, 3, 1, 2), dtype=np.float32)).to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()           # (n,3,size,size)
        if flip_axis is not None:
            p = np.flip(p, axis=2 + flip_axis)
        for bi, k in enumerate(chunk):
            for c in range(3):
                probs[c, :, :, k] = resize(p[bi, c], (H, W), order=1,
                                           mode="constant", anti_aliasing=False)
    return probs


@torch.no_grad()
def predict_probs(scan: Scan, model, ckpt, device="cpu", tta=False, batch=16) -> np.ndarray:
    """Per-voxel WT/TC/ET probabilities, shape (3, H, W, D).

    ``tta`` averages the prediction with that of the left-right mirrored scan
    (brains are roughly left-right symmetric, so the mirror image is a realistic
    second view). The left-right axis is read from the scan's orientation."""
    if not scan.is_multimodal or scan.n_modalities != ckpt["in_ch"]:
        raise ValueError(f"expected {ckpt['in_ch']}-channel image, got {scan.data.shape}")
    vol = _normalize(scan.data.astype(np.float32))          # (H,W,D,C)
    probs = _predict_slices(vol, model, ckpt["size"], device, batch)
    if tta:
        lr = next((a for a in (0, 1) if scan.axcodes[a] in ("L", "R")), 0)
        probs += _predict_slices(vol, model, ckpt["size"], device, batch, flip_axis=lr)
        probs /= 2
    return probs


def probs_to_label(probs: np.ndarray, thresh=0.5, post="largest",
                   spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Threshold WT/TC/ET maps into a nested 3-D label map (0 bg, 1 edema,
    2 core, 3 enhancing), then clean up separate predicted pieces:

    * ``largest`` keeps only the largest tumour piece,
    * ``min1cm3`` drops pieces under 1 cm3 (specks) but keeps real separate regions,
    * ``all`` keeps everything.
    """
    if post not in POST:
        raise ValueError(f"post must be one of {POST}, got {post!r}")
    wt = probs[0] > thresh
    tc = (probs[1] > thresh) & wt
    et = (probs[2] > thresh) & tc

    if post != "all" and wt.any():
        lab, n = ndimage.label(wt)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            if post == "largest":
                keep = lab == np.argmax(sizes) + 1
            else:
                cm3 = sizes * float(np.prod(spacing)) / 1000.0
                big = np.flatnonzero((cm3 >= 1.0) | (sizes == sizes.max())) + 1
                keep = np.isin(lab, big)
            wt &= keep
            tc &= keep
            et &= keep

    label = np.zeros(wt.shape, np.uint8)
    label[wt] = 1          # edema / whole-tumour shell
    label[tc] = 2          # tumour core (non-enhancing)
    label[et] = 3          # enhancing tumour
    return label


def predict_label(scan: Scan, model, ckpt, device="cpu", thresh=0.5,
                  post=DEFAULT_POST, tta=False, batch=16) -> np.ndarray:
    probs = predict_probs(scan, model, ckpt, device=device, tta=tta, batch=batch)
    return probs_to_label(probs, thresh=thresh, post=post, spacing=scan.spacing)
