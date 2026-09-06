"""Synthetic BraTS-format phantoms.

Two uses:
  * unit tests - build a tumour of *known* size and confirm the measurement
    engine recovers it (see tests/test_measure.py);
  * a tiny, instantly-available dataset so the whole train -> measure -> report
    pipeline is runnable and testable without the 7.6 GB download.

The generated volumes imitate the MSD Task01 layout: a 4-channel "MRI" image
(FLAIR/T1w/T1gd/T2w) and an integer label with values {0:bg, 1:edema,
2:non-enhancing, 3:enhancing}.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from . import MODALITIES


def ellipsoid(shape, center, semi_axes) -> np.ndarray:
    """Boolean ellipsoid mask with given voxel center and semi-axes."""
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    az, ay, ax = semi_axes
    d = ((zz - cz) / az) ** 2 + ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2
    return d <= 1.0


def make_label(shape=(96, 96, 64), center=None, semi_axes=(14, 10, 8),
               seed=0) -> np.ndarray:
    """A nested tumour: edema shell (1), non-enhancing (2), enhancing core (3)."""
    if center is None:
        center = (shape[0] // 2 + 8, shape[1] // 2 - 6, shape[2] // 2)
    label = np.zeros(shape, np.uint8)
    sa = np.array(semi_axes, float)
    label[ellipsoid(shape, center, sa)] = 1              # whole tumour / edema
    label[ellipsoid(shape, center, sa * 0.65)] = 2       # tumour core
    label[ellipsoid(shape, center, sa * 0.35)] = 3       # enhancing core
    return label


def make_image(label: np.ndarray, seed=0) -> np.ndarray:
    """A 4-channel pseudo-MRI whose intensities correlate with the label so a
    model can actually learn something on the synthetic set."""
    rng = np.random.default_rng(seed)
    shape = label.shape
    # a faint "brain" ellipsoid so background isn't uniform
    brain = ellipsoid(shape, tuple(s // 2 for s in shape),
                      tuple(s * 0.45 for s in shape))
    chans = []
    # per-modality mean intensity for each label value (bg, edema, non-enh, enh)
    profiles = {
        "FLAIR": [0.1, 0.9, 0.7, 0.6],
        "T1w":   [0.1, 0.3, 0.4, 0.5],
        "T1gd":  [0.1, 0.3, 0.5, 0.95],
        "T2w":   [0.1, 0.85, 0.75, 0.65],
    }
    for m in MODALITIES:
        img = np.where(brain, 0.25, 0.0).astype(np.float32)
        for val, mean in enumerate(profiles[m]):
            if val == 0:
                continue
            img[label == val] = mean
        img = img + rng.normal(0, 0.05, shape).astype(np.float32)
        chans.append(np.clip(img, 0, 1))
    return np.stack(chans, axis=-1)


def save_case(out_dir: Path, case_id: str, spacing=(1.0, 1.0, 1.0), **kw):
    """Write one imagesTr/labelsTr NIfTI pair in MSD layout."""
    out_dir = Path(out_dir)
    (out_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    label = make_label(**kw)
    image = make_image(label, seed=kw.get("seed", 0))
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(image, affine),
             str(out_dir / "imagesTr" / f"{case_id}.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine),
             str(out_dir / "labelsTr" / f"{case_id}.nii.gz"))


def make_dataset(out_dir="data/synthetic", n=12, spacing=(1.0, 1.0, 1.0),
                 shape=(96, 96, 64)):
    """Generate `n` varied phantoms so a model has something to train on."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(1234)
    ids = []
    for i in range(n):
        cid = f"SYN_{i:03d}"
        semi = tuple(rng.integers(low, high) for low, high in [(9, 18), (7, 14), (5, 11)])
        center = (shape[0] // 2 + int(rng.integers(-12, 12)),
                  shape[1] // 2 + int(rng.integers(-12, 12)),
                  shape[2] // 2 + int(rng.integers(-8, 8)))
        save_case(out_dir, cid, spacing=spacing, shape=shape,
                  center=center, semi_axes=semi, seed=i)
        ids.append(cid)
    print(f"wrote {n} synthetic cases to {out_dir}")
    return ids


if __name__ == "__main__":
    make_dataset()
