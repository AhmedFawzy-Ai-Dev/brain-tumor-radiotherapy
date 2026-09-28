"""How much did separate pieces inflate the diameters? Measured on EXPERT masks.

Before v0.2 the calliper diameters were taken over every voxel of a region, so a
piece far from the tumour stretched them across the gap. This compares that old
way with the current one (largest connected lesion) on the test patients' expert
masks - no model involved, so it isolates the measurement code - and says what
the far end of each inflated diameter landed in:

  * a **speck** (a piece under 1 cm3): the old number was simply wrong;
  * a **separate region** of 1 cm3 or more: the old number spanned two regions;
    the report now measures the largest and flags the others (``large_pieces``).

    python scripts/islands_effect.py --json docs/eval/islands.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brats_report import io as bio
from brats_report.dataset import list_cases
from brats_report.evaluate import case_id, load_split
from brats_report.measure import _max_caliper, _slice_axis, measure_region, pieces, region_mask


def old_diameters(mask, spacing, axcodes):
    """The pre-fix measurement over every voxel: (max 3-D mm, its two ends,
    RECIST mm, its two ends), ends as voxel indices."""
    sp = np.array(spacing, float)
    coords = np.array(np.nonzero(mask)).T
    d3, a3, b3 = _max_caliper(coords * sp)
    zax = _slice_axis(axcodes)
    inplane = [a for a in range(3) if a != zax]
    best = (0.0, None, None)
    for k in range(mask.shape[zax]):
        sl = np.take(mask, k, axis=zax)
        if sl.sum() < 2:
            continue
        pts = np.array(np.nonzero(sl)).T * sp[inplane]
        d, a, b = _max_caliper(pts.astype(float))
        if d > best[0]:
            full = []
            for p in (a, b):
                v = np.zeros(3)
                v[inplane] = p / sp[inplane]
                v[zax] = k
                full.append(v)
            best = (d, full[0], full[1])
    return d3, a3 / sp, b3 / sp, best[0], best[1], best[2]


def far_piece_cm3(lab, cm3, biggest, ends):
    """Size (cm3) of the piece an old calliper end sits in, if not the main lesion."""
    ids = [int(lab[tuple(np.round(p).astype(int))]) for p in ends]
    other = [cm3[i - 1] for i in ids if i not in (0, biggest)]
    return round(float(min(other)), 3) if other else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/brats_subset")
    ap.add_argument("--split", default="data/brats_split.json")
    ap.add_argument("--region", default="WT")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    for _, lbl_p in load_split(args.split, list_cases(args.data))["test"]:
        scan = bio.load_nifti(lbl_p)
        label = np.asanyarray(scan.data)
        mask = region_mask(label, args.region)
        lab, sizes = pieces(mask)
        cm3 = sizes * float(np.prod(scan.spacing)) / 1000.0
        biggest = int(np.argmax(sizes)) + 1
        d3, a3, b3, rec, ar, br = old_diameters(mask, scan.spacing, scan.axcodes)
        m = measure_region(label, args.region, scan.spacing, scan.axcodes)

        row = {"case": case_id(lbl_p), "components": m.components,
               "large_pieces": m.large_pieces,
               "recist_old_mm": round(rec, 1), "recist_new_mm": m.recist_long_mm,
               "max3d_old_mm": round(d3, 1), "max3d_new_mm": m.max_diameter_mm,
               "recist_far_piece_cm3": (far_piece_cm3(lab, cm3, biggest, [ar, br])
                                        if ar is not None else None),
               "max3d_far_piece_cm3": far_piece_cm3(lab, cm3, biggest, [a3, b3])}
        rows.append(row)
        print(f"{row['case']}: {row['components']:3d} pieces ({row['large_pieces']} >= 1 cm3)  "
              f"RECIST {row['recist_old_mm']:6.1f} -> {row['recist_new_mm']:6.1f} mm  "
              f"max 3-D {row['max3d_old_mm']:6.1f} -> {row['max3d_new_mm']:6.1f} mm  "
              f"far piece {row['recist_far_piece_cm3']} / {row['max3d_far_piece_cm3']} cm3")

    def inflation(kind, speck):
        out = []
        for r in rows:
            far = r[f"{kind}_far_piece_cm3"]
            d = r[f"{kind}_old_mm"] - r[f"{kind}_new_mm"]
            if far is not None and d > 1 and (far < 1.0) == speck:
                out.append((r["case"], round(d, 1)))
        return out

    summary = {"region": args.region, "n_cases": len(rows),
               "cases_with_more_than_one_piece": int(sum(r["components"] > 1 for r in rows)),
               "cases_with_two_or_more_pieces_over_1cm3": int(
                   sum(r["large_pieces"] > 1 for r in rows))}
    for speck, name in ((True, "speck"), (False, "separate_region")):
        for kind in ("recist", "max3d"):
            hits = inflation(kind, speck)
            summary[f"{name}_{kind}_inflated_cases"] = len(hits)
            summary[f"{name}_{kind}_inflation_max_mm"] = max((d for _, d in hits), default=0.0)
            summary[f"{name}_{kind}_cases"] = [c for c, _ in hits]
    speck_cases = set(summary["speck_recist_cases"]) | set(summary["speck_max3d_cases"])
    summary["speck_inflated_cases_any"] = len(speck_cases)
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
