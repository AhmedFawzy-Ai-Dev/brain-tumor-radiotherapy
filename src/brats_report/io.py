"""Reading medical images into (array, spacing, orientation).

Supports NIfTI (``.nii`` / ``.nii.gz``, the BraTS/MSD format) directly, and a
folder of DICOM slices (what a real scanner exports) via pydicom. Both return a
numpy array plus the physical voxel spacing in millimetres and the anatomical
axis codes, which is everything the measurement engine needs to report real-world
dimensions.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


class Scan:
    """A loaded volume with the metadata needed for physical measurement."""

    def __init__(self, data: np.ndarray, spacing: tuple, axcodes: tuple,
                 affine: np.ndarray, source: str = ""):
        self.data = data                    # (H, W, D) or (H, W, D, C)
        self.spacing = spacing              # (sx, sy, sz) in mm
        self.axcodes = axcodes              # e.g. ('R', 'A', 'S')
        self.affine = affine
        self.source = source

    @property
    def is_multimodal(self) -> bool:
        return self.data.ndim == 4

    @property
    def n_modalities(self) -> int:
        return self.data.shape[3] if self.is_multimodal else 1

    def __repr__(self) -> str:
        return (f"Scan(shape={self.data.shape}, spacing={self.spacing}, "
                f"axcodes={self.axcodes})")


def load_nifti(path: str | Path) -> Scan:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()
    spacing = tuple(float(z) for z in zooms[:3])
    axcodes = nib.aff2axcodes(img.affine)
    return Scan(data, spacing, axcodes, img.affine, source=str(path))


def load_dicom_series(folder: str | Path) -> Scan:
    """Load a directory of single-slice DICOM files into a 3D volume.

    Slices are ordered by ImagePositionPatient along the slice normal, and
    spacing is taken from PixelSpacing + slice separation. This is the path a
    real radiotherapy workflow uses; for the BraTS demo we use NIfTI.
    """
    import pydicom

    folder = Path(folder)
    files = [pydicom.dcmread(str(p)) for p in folder.glob("*") if p.is_file()]
    files = [f for f in files if hasattr(f, "ImagePositionPatient")]
    if not files:
        raise ValueError(f"no DICOM slices with position found in {folder}")
    files.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    volume = np.stack([f.pixel_array.astype(np.float32) for f in files], axis=-1)

    py, px = (float(v) for v in files[0].PixelSpacing)
    if len(files) > 1:
        z0 = float(files[0].ImagePositionPatient[2])
        z1 = float(files[1].ImagePositionPatient[2])
        sz = abs(z1 - z0) or float(getattr(files[0], "SliceThickness", 1.0))
    else:
        sz = float(getattr(files[0], "SliceThickness", 1.0))
    spacing = (px, py, sz)
    # DICOM axial slices are conventionally L->R, P->A, I->S in patient space.
    return Scan(volume, spacing, ("L", "P", "S"), np.eye(4), source=str(folder))


def load_scan(path: str | Path) -> Scan:
    p = Path(path)
    if p.is_dir():
        return load_dicom_series(p)
    return load_nifti(p)
