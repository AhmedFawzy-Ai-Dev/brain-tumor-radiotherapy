"""Turn a segmentation mask + voxel spacing into real-world tumour measurements.

Everything here is geometry on the predicted mask - no learning - so it is exact
and unit-testable against phantoms of known size (see ``synthetic.py`` / tests).

For each reporting region (whole tumour, tumour core, enhancing tumour) we report:
  * volume in cm3 (millilitres),
  * anatomical bounding extents LR / AP / SI in mm,
  * the maximum 3-D calliper diameter in mm,
  * the RECIST-style longest axial diameter and its perpendicular in mm,
  * an approximate anatomical location of the centroid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import REGIONS

# --- axis <-> anatomy helpers ------------------------------------------------

def _dim_name(code: str) -> str:
    return {"R": "LR", "L": "LR", "A": "AP", "P": "AP", "S": "SI", "I": "SI"}[code]


def _slice_axis(axcodes: tuple) -> int:
    """Index of the superior-inferior axis (the one we stack axial slices on)."""
    for i, c in enumerate(axcodes):
        if c in ("S", "I"):
            return i
    return 2


# --- calliper geometry -------------------------------------------------------

def _max_caliper(coords_mm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Largest distance between any two points, in the points' own units.

    Uses the convex hull so it is exact and fast even for large masks. Returns
    (distance, point_a, point_b).
    """
    pts = np.unique(coords_mm, axis=0)
    if len(pts) < 2:
        return 0.0, pts[0] if len(pts) else np.zeros(coords_mm.shape[1]), \
            pts[0] if len(pts) else np.zeros(coords_mm.shape[1])
    try:
        hull = ConvexHull(pts)
        cand = pts[hull.vertices]
    except (QhullError, ValueError):
        cand = pts  # degenerate (flat / collinear) -> brute force on all points
    if len(cand) > 3000:  # keep the O(n^2) safe on huge flat regions
        idx = np.random.default_rng(0).choice(len(cand), 3000, replace=False)
        cand = cand[idx]
    d2 = np.sum((cand[:, None, :] - cand[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return float(np.sqrt(d2[i, j])), cand[i], cand[j]


def _recist(slice_mask: np.ndarray, sx: float, sy: float):
    """Longest in-plane diameter and its perpendicular, in mm, on one slice."""
    ys, xs = np.nonzero(slice_mask)
    if len(xs) < 2:
        return 0.0, 0.0
    coords = np.column_stack([xs * sx, ys * sy]).astype(float)
    longest, a, b = _max_caliper(coords)
    if longest == 0:
        return 0.0, 0.0
    d = (b - a) / longest
    perp = np.array([-d[1], d[0]])            # unit vector perpendicular to long axis
    proj = coords @ perp
    return longest, float(proj.max() - proj.min())


# --- region measurement ------------------------------------------------------

@dataclass
class RegionMeasurement:
    region: str
    voxels: int
    volume_cm3: float
    extent_mm: dict = field(default_factory=dict)      # {'LR':.., 'AP':.., 'SI':..}
    max_diameter_mm: float = 0.0                        # 3-D calliper
    recist_long_mm: float = 0.0                         # longest axial diameter
    recist_short_mm: float = 0.0                        # perpendicular to it
    recist_slice: int = -1
    location: str = ""
    present: bool = True


def region_mask(label: np.ndarray, region: str) -> np.ndarray:
    return np.isin(label, REGIONS[region])


def _location(mask: np.ndarray, axcodes: tuple) -> str:
    """Coarse description of where the centroid sits (honest, atlas-free)."""
    cen = np.array(np.nonzero(mask)).mean(axis=1)
    mid = (np.array(mask.shape) - 1) / 2
    words = []
    for i, code in enumerate(axcodes):
        pos = cen[i] - mid[i]
        # positive index direction points toward `code`
        toward = code if pos > 0 else {"R": "L", "L": "R", "A": "P",
                                       "P": "A", "S": "I", "I": "S"}[code]
        if abs(pos) < mask.shape[i] * 0.08:
            continue  # near midline on this axis
        words.append({"R": "right", "L": "left", "A": "anterior",
                      "P": "posterior", "S": "superior", "I": "inferior"}[toward])
    return ", ".join(words) if words else "central / midline"


def measure_region(label: np.ndarray, region: str, spacing: tuple,
                   axcodes: tuple) -> RegionMeasurement:
    mask = region_mask(label, region)
    n = int(mask.sum())
    sx, sy, sz = spacing
    if n == 0:
        return RegionMeasurement(region, 0, 0.0, present=False)

    vol_cm3 = n * (sx * sy * sz) / 1000.0

    # anatomical extents from the bounding box
    coords = np.array(np.nonzero(mask))
    extent = {}
    for axis in range(3):
        span = (coords[axis].max() - coords[axis].min() + 1) * spacing[axis]
        extent[_dim_name(axcodes[axis])] = round(float(span), 1)

    # 3-D max calliper diameter
    surf = np.column_stack([coords[0] * sx, coords[1] * sy, coords[2] * sz])
    max_diam, _, _ = _max_caliper(surf)

    # RECIST: scan axial slices, keep the largest longest-diameter
    zax = _slice_axis(axcodes)
    inplane = [a for a in range(3) if a != zax]
    ps = [spacing[inplane[0]], spacing[inplane[1]]]
    best = (0.0, 0.0, -1)
    for k in range(mask.shape[zax]):
        sl = np.take(mask, k, axis=zax)
        if sl.sum() < 2:
            continue
        long_mm, short_mm = _recist(sl.T if inplane[0] > inplane[1] else sl, ps[1], ps[0])
        if long_mm > best[0]:
            best = (long_mm, short_mm, k)

    return RegionMeasurement(
        region=region, voxels=n, volume_cm3=round(vol_cm3, 2), extent_mm=extent,
        max_diameter_mm=round(max_diam, 1), recist_long_mm=round(best[0], 1),
        recist_short_mm=round(best[1], 1), recist_slice=best[2],
        location=_location(mask, axcodes), present=True,
    )


def measure_all(label: np.ndarray, spacing: tuple, axcodes: tuple) -> dict:
    """Measure every reporting region. Returns {region: RegionMeasurement}."""
    return {r: measure_region(label, r, spacing, axcodes) for r in REGIONS}


def measure_mask_2d(mask: np.ndarray, spacing=(1.0, 1.0), unit="px") -> dict:
    """Measure a single 2-D binary tumour mask.

    `spacing` is (sx, sy) per pixel; `unit` is "px" (JPG/PNG, no physical scale)
    or "mm" (e.g. from a DICOM PixelSpacing). Reports area, RECIST-style longest
    diameter + perpendicular, and bounding box - all in the given unit.
    """
    m = np.asarray(mask) > 0
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {"present": False, "unit": unit}
    sx, sy = float(spacing[0]), float(spacing[1])
    coords = np.column_stack([xs * sx, ys * sy]).astype(float)
    long_d, a, b = _max_caliper(coords)
    if long_d > 0:
        d = (b - a) / long_d
        proj = coords @ np.array([-d[1], d[0]])
        short_d = float(proj.max() - proj.min())
    else:
        short_d = 0.0
    area = float(len(xs) * sx * sy)
    w = float((xs.max() - xs.min() + 1) * sx)
    h = float((ys.max() - ys.min() + 1) * sy)
    out = {"present": True, "unit": unit, "pixels": int(len(xs)),
           "area": round(area, 1), "recist_long": round(long_d, 1),
           "recist_short": round(short_d, 1), "bbox_w": round(w, 1),
           "bbox_h": round(h, 1)}
    if unit == "mm":  # add cm / cm2 conveniences
        out["area_cm2"] = round(area / 100.0, 2)
    return out
