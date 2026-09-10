"""2-D single-image tumour segmentation.

Unlike the 3-D BraTS segmenter, this runs on ONE ordinary 2-D MRI image (what a
clinic usually has), producing a tumour outline so the tool can measure the
lesion on 2-D inputs. Trained on the LGG dataset (grayscale + binary masks).
Measurements come out in pixels, or in mm when the image carries pixel spacing
(e.g. DICOM).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.transform import resize

from .dataset import normalize_2d
from .model import UNet2D


def load_seg2d(path: str | Path, device="cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = UNet2D(in_ch=1, out_ch=1, base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_mask2d(image2d: np.ndarray, model, ckpt, device="cpu", thresh=0.5,
                   keep_largest=True) -> np.ndarray:
    """Segment a 2-D grayscale image -> binary mask at native resolution."""
    a = image2d.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    H, W = a.shape
    size = ckpt["size"]
    x = resize(normalize_2d(a), (size, size), order=1, mode="constant",
               anti_aliasing=True).astype(np.float32)
    t = torch.from_numpy(x[None, None]).to(device)
    p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    p = resize(p, (H, W), order=1, mode="constant", anti_aliasing=False)
    mask = p > thresh
    if keep_largest and mask.any():
        lab, n = ndimage.label(mask)
        if n > 1:
            biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            mask = lab == biggest
    return mask.astype(np.uint8)
