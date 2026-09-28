"""Before/after picture of the stray-island bug, on one patient's EXPERT mask.

Left: the old RECIST diameter, measured over every voxel of the whole-tumour
region, reaching a speck outside the lesion. Right: the current one, on the
lesion itself.

    python scripts/islands_figure.py --case BRATS_131 --out docs/images/islands_before_after.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402

from brats_report import io as bio  # noqa: E402
from brats_report.measure import _max_caliper, pieces, region_mask  # noqa: E402

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def longest_axial(mask, spacing):
    """(length mm, slice, endpoints) of the longest in-plane calliper over slices."""
    best = (0.0, -1, None, None)
    for k in range(mask.shape[2]):
        sl = mask[:, :, k]
        if sl.sum() < 2:
            continue
        pts = np.column_stack(np.nonzero(sl)) * np.array(spacing[:2])
        d, a, b = _max_caliper(pts.astype(float))
        if d > best[0]:
            best = (d, k, a / np.array(spacing[:2]), b / np.array(spacing[:2]))
    return best


def panel(ax, flair, mask_slice, a, b, length, title, color, islands=None):
    g = flair.astype(np.float32)
    lo, hi = np.percentile(g[g > 0], [1, 99]) if (g > 0).any() else (0, 1)
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.stack([g, g, g], -1)
    rgb[mask_slice] = rgb[mask_slice] * 0.45 + np.array([1.0, 0.85, 0.1]) * 0.55
    n0, n1 = mask_slice.shape
    # radiological display: rows run against axis 1 (anterior up), columns
    # against axis 0 (patient's right on the image's left) for RAS data
    disp = rgb.transpose(1, 0, 2)[::-1, ::-1]

    def xy(p):
        return n0 - 1 - p[0], n1 - 1 - p[1]

    ax.imshow(disp)
    (xa, ya), (xb, yb) = xy(a), xy(b)
    ax.plot([xa, xb], [ya, yb], "-", color=color, lw=2.2, marker="o", ms=4.5)
    ax.annotate(f"{length:.0f} mm", ((xa + xb) / 2, (ya + yb) / 2), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=13, fontweight="bold",
                color=color, path_effects=[patheffects.withStroke(linewidth=3, foreground="black")])
    for (px, py) in islands or []:
        x, y = xy((px, py))
        ax.add_patch(plt.Circle((x, y), 7, fill=False, color=color, lw=1.8))
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")
    ax.set_xlim(20, n0 - 20)
    ax.set_ylim(n1 - 15, 15)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/brats_subset/Task01_BrainTumour")
    ap.add_argument("--case", default="BRATS_131")
    ap.add_argument("--out", default="docs/images/islands_before_after.png")
    args = ap.parse_args()

    img = bio.load_nifti(Path(args.data) / "imagesTr" / f"{args.case}.nii.gz")
    lab = bio.load_nifti(Path(args.data) / "labelsTr" / f"{args.case}.nii.gz")
    if tuple(img.axcodes) != ("R", "A", "S"):
        raise SystemExit(f"expected RAS orientation, got {img.axcodes}")
    wt = region_mask(np.asanyarray(lab.data), "WT")
    piece_map, sizes = pieces(wt)
    lesion = piece_map == (np.argmax(sizes) + 1)
    sp = img.spacing
    flair = np.asanyarray(img.data)[..., 0]

    old = longest_axial(wt, sp)
    new = longest_axial(lesion, sp)
    # the end of the old line that isn't on the lesion is the island it reached
    islands = [p for p in (old[2], old[3])
               if not lesion[int(round(p[0])), int(round(p[1])), old[1]]]
    speck_cm3 = [sizes[piece_map[int(round(p[0])), int(round(p[1])), old[1]] - 1]
                 * float(np.prod(sp)) / 1000 for p in islands]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], flair[:, :, old[1]], wt[:, :, old[1]], old[2], old[3], old[0],
          f"Before: whole region, slice {old[1]}", "#ff5a4f", islands)
    panel(axes[1], flair[:, :, new[1]], lesion[:, :, new[1]], new[2], new[3], new[0],
          f"After: largest lesion, slice {new[1]}", "#39d0ff")
    fig.suptitle(f"Longest axial diameter on an expert mask ({args.case}, a held-out patient)",
                 x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK)
    speck = f" of {speck_cm3[0]:.2f} cm³" if speck_cm3 else ""
    fig.text(0.012, 0.9, f"The mask has {len(sizes)} separate pieces. Measured over every voxel, "
             f"the diameter runs to a speck{speck} outside the lesion (circled).",
             fontsize=10, color=INK_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.03)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"{args.case}: {old[0]:.1f} mm (slice {old[1]}) -> {new[0]:.1f} mm (slice {new[1]}); "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
