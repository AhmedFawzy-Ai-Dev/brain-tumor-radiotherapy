"""2-D axial-slice dataset for BraTS / MSD Task01 (and the synthetic phantoms).

Each training example is one axial slice: a 4-channel image (FLAIR/T1w/T1gd/T2w)
and a 3-channel multi-label target (WT/TC/ET). Volumes are z-scored per modality
over the brain region, then slices are resized to a fixed square for the CPU
U-Net. Axial = last spatial axis (true for MSD Task01 and the phantoms).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from . import REGIONS
from . import io as bio


def list_cases(root: str | Path) -> list[tuple[Path, Path]]:
    """Find matched imagesTr/labelsTr pairs under a dataset root (recursively)."""
    root = Path(root)
    pairs = []
    for img in sorted(root.rglob("imagesTr/*.nii*")):
        lbl = img.parent.parent / "labelsTr" / img.name
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def label_to_multilabel(label: np.ndarray) -> np.ndarray:
    """(H,W[,D]) integer label -> (...,3) float {WT,TC,ET}."""
    out = np.stack([np.isin(label, REGIONS[r]) for r in ("WT", "TC", "ET")], axis=-1)
    return out.astype(np.float32)


def _normalize(volume: np.ndarray) -> np.ndarray:
    """Per-modality z-score over nonzero (brain) voxels; background stays 0."""
    vol = volume.astype(np.float32)
    for c in range(vol.shape[-1]):
        ch = vol[..., c]
        brain = ch > 0
        if brain.sum() > 0:
            mu, sd = ch[brain].mean(), ch[brain].std() + 1e-6
            ch = (ch - mu) / sd
            ch[~brain] = 0.0
        vol[..., c] = ch
    return vol


class BratsSlices(Dataset):
    def __init__(self, pairs, size=128, empty_frac=0.15, augment=False, seed=0):
        self.pairs = list(pairs)
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        # Materialise every kept slice once, resized + normalised, held in RAM as
        # float16. Each (large) volume is loaded and z-scored exactly once - this
        # avoids re-normalising 3-D volumes on every shuffled batch access.
        self.imgs: list[np.ndarray] = []   # each (size, size, 4) float16
        self.masks: list[np.ndarray] = []  # each (size, size, 3) float16
        s = self.size
        for img_p, lbl_p in self.pairs:
            image = _normalize(np.asanyarray(bio.load_nifti(img_p).data).astype(np.float32))
            label = np.asanyarray(bio.load_nifti(lbl_p).data)
            n = label.shape[2]
            tumor = [k for k in range(n) if label[:, :, k].any()]
            empt = [k for k in range(n) if k not in tumor]
            keep = min(len(empt), int(len(tumor) * empty_frac) + 1)
            keep_empt = list(self.rng.choice(empt, size=keep, replace=False)) if empt else []
            for k in tumor + keep_empt:
                img = resize(image[:, :, k, :], (s, s, image.shape[-1]), order=1,
                             mode="constant", anti_aliasing=True).astype(np.float16)
                ml = resize(label_to_multilabel(label[:, :, k]), (s, s, 3), order=0,
                            mode="constant", anti_aliasing=False).astype(np.float16)
                self.imgs.append(img)
                self.masks.append(ml)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32)
        ml = self.masks[i].astype(np.float32)
        if self.augment and self.rng.random() < 0.5:
            img = img[:, ::-1, :].copy()
            ml = ml[:, ::-1, :].copy()
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        ml = torch.from_numpy(np.ascontiguousarray(ml.transpose(2, 0, 1)))
        return img, ml
