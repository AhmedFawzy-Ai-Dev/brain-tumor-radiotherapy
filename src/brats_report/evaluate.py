"""Evaluate the 3-D pipeline the way a report is used: per patient, on whole scans.

Training reports a slice-level Dice (tumour slices only). That flatters the model
and says nothing about the numbers that end up in a report. This script runs the
full inference pipeline on held-out patients and compares against the expert mask:

  * **Dice** per region (WT / TC / ET) on the whole 3-D volume,
  * **HD95** — 95th-percentile surface distance in mm (boundary quality),
  * **measurement error** — the report's volume (cm3), max 3-D diameter (mm) and
    RECIST longest axial diameter (mm) from the predicted mask vs. the same
    measurements taken from the expert mask with the same code.

Usage::

    python -m brats_report.evaluate split --data data/brats_subset --out data/brats_split.json
    python -m brats_report.evaluate run --data data/brats_subset --split data/brats_split.json \\
        --model models/brats_unet.pt --tta --name "baseline + TTA" --json docs/eval/tta.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import REGIONS
from . import io as bio
from .dataset import list_cases
from .measure import measure_region

REGION_ORDER = ("WT", "TC", "ET")


# --- split -------------------------------------------------------------------

def case_id(path: Path) -> str:
    return path.name.split(".")[0]


def make_split(ids, n_test, n_val, seed=0, not_in_test=()):
    """Deterministic train/val/test split of case ids. Cases in ``not_in_test``
    (e.g. ones an earlier model was trained on) are never put in the test set."""
    rng = np.random.default_rng(seed)
    ids = sorted(ids)
    blocked = set(not_in_test)
    eligible = [i for i in ids if i not in blocked]
    if len(eligible) < n_test:
        raise ValueError(f"only {len(eligible)} cases eligible for test, need {n_test}")
    test = sorted(rng.choice(eligible, n_test, replace=False).tolist())
    rest = [i for i in ids if i not in set(test)]
    val = sorted(rng.choice(rest, n_val, replace=False).tolist())
    train = sorted(i for i in rest if i not in set(val))
    return {"seed": seed, "train": train, "val": val, "test": test}


def load_split(path, pairs):
    """Map a split JSON onto (image, label) pairs: {'train': [...], ...}."""
    split = json.loads(Path(path).read_text())
    by_id = {case_id(img): (img, lbl) for img, lbl in pairs}
    missing = [i for part in ("train", "val", "test") for i in split[part] if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} split cases not found under data, e.g. {missing[:3]}")
    return {part: [by_id[i] for i in split[part]] for part in ("train", "val", "test")}


# --- metrics -----------------------------------------------------------------

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice of two boolean masks; 1.0 when both are empty (a correct 'no tumour')."""
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(pred, gt).sum() / s)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """Symmetric 95th-percentile Hausdorff distance in mm.

    NaN when exactly one mask is empty (the distance is undefined), 0 when both are.
    """
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("nan")
    # crop to both masks plus a background margin: exact (every surface voxel is
    # inside, none touches the border) and far faster than the whole scan
    idx = np.argwhere(pred | gt)
    box = tuple(slice(max(lo - 2, 0), hi + 3)
                for lo, hi in zip(idx.min(0), idx.max(0), strict=True))
    pred, gt = np.pad(pred[box], 1), np.pad(gt[box], 1)
    sp, sg = _surface(pred), _surface(gt)
    to_gt = ndimage.distance_transform_edt(~sg, sampling=spacing)[sp]
    to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(np.percentile(np.concatenate([to_gt, to_pred]), 95))


def _measure(label, region, spacing, axcodes):
    m = measure_region(label, region, spacing, axcodes)
    return {"volume_cm3": m.volume_cm3, "max_diameter_mm": m.max_diameter_mm,
            "recist_long_mm": m.recist_long_mm}


def score_case(pred_label, gt_label, spacing, axcodes) -> dict:
    out = {}
    for r in REGION_ORDER:
        p, g = np.isin(pred_label, REGIONS[r]), np.isin(gt_label, REGIONS[r])
        out[r] = {"dice": round(dice(p, g), 4), "hd95_mm": round(hd95(p, g, spacing), 2),
                  "pred": _measure(pred_label, r, spacing, axcodes),
                  "gt": _measure(gt_label, r, spacing, axcodes)}
    return out


def summarize(cases: list[dict]) -> dict:
    """Aggregate per-case scores into the numbers the README reports."""
    summary = {}
    for r in REGION_ORDER:
        d = np.array([c["scores"][r]["dice"] for c in cases])
        h = np.array([c["scores"][r]["hd95_mm"] for c in cases], float)
        pv = np.array([c["scores"][r]["pred"]["volume_cm3"] for c in cases])
        gv = np.array([c["scores"][r]["gt"]["volume_cm3"] for c in cases])
        pd_ = np.array([c["scores"][r]["pred"]["max_diameter_mm"] for c in cases])
        gd = np.array([c["scores"][r]["gt"]["max_diameter_mm"] for c in cases])
        pr = np.array([c["scores"][r]["pred"]["recist_long_mm"] for c in cases])
        gr = np.array([c["scores"][r]["gt"]["recist_long_mm"] for c in cases])
        has = gv > 0
        rel = np.abs(pv[has] - gv[has]) / gv[has] * 100
        summary[r] = {
            "dice_mean": round(float(d.mean()), 4),
            "dice_median": round(float(np.median(d)), 4),
            "hd95_median_mm": round(float(np.nanmedian(h)), 2) if np.isfinite(h).any() else None,
            "volume_mae_cm3": round(float(np.abs(pv - gv).mean()), 2),
            "volume_median_abs_pct": round(float(np.median(rel)), 1) if has.any() else None,
            "volume_bias_cm3": round(float((pv - gv).mean()), 2),
            "volume_r": (round(float(np.corrcoef(pv, gv)[0, 1]), 3)
                         if pv.std() > 0 and gv.std() > 0 else None),
            "max_diameter_mae_mm": round(float(np.abs(pd_ - gd).mean()), 1),
            "recist_mae_mm": round(float(np.abs(pr - gr).mean()), 1),
        }
    summary["dice_mean_all"] = round(float(np.mean(
        [summary[r]["dice_mean"] for r in REGION_ORDER])), 4)
    return summary


# --- CLI ---------------------------------------------------------------------

def _cmd_split(args):
    ids = [case_id(img) for img, _ in list_cases(args.data)]
    blocked = []
    if args.not_in_test:
        blocked = [ln.strip() for ln in Path(args.not_in_test).read_text().splitlines()
                   if ln.strip()]
    split = make_split(ids, args.test, args.val, args.seed, blocked)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, indent=2) + "\n")
    print(f"{len(ids)} cases -> {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test  ->  {args.out}")


def _print_summary(name, summary, n):
    print(f"\n{name}: mean Dice {summary['dice_mean_all']:.3f} on {n} patients")
    for r in REGION_ORDER:
        s = summary[r]
        print(f"  {r}: Dice {s['dice_mean']:.3f}  HD95 {s['hd95_median_mm']} mm  "
              f"vol err {s['volume_mae_cm3']} cm3 (median {s['volume_median_abs_pct']}%, "
              f"bias {s['volume_bias_cm3']:+})  max-diam err {s['max_diameter_mae_mm']} mm  "
              f"RECIST err {s['recist_mae_mm']} mm")


def _cmd_run(args):
    import torch

    from .infer import load_model, predict_probs, probs_to_label

    if args.threads:
        torch.set_num_threads(args.threads)
    posts = args.post.split(",")
    test = load_split(args.split, list_cases(args.data))[args.part]
    if args.limit:
        test = test[:args.limit]
    model, ckpt = load_model(args.model)
    cases = {p: [] for p in posts}
    t_total = 0.0
    for i, (img_p, lbl_p) in enumerate(test, 1):
        scan = bio.load_nifti(img_p)
        gt = np.asanyarray(bio.load_nifti(lbl_p).data).astype(np.uint8)
        t0 = time.time()
        probs = predict_probs(scan, model, ckpt, tta=args.tta)   # one pass, every variant
        dt = time.time() - t0
        t_total += dt
        line = []
        for post in posts:
            pred = probs_to_label(probs, post=post, spacing=scan.spacing)
            sc = score_case(pred, gt, scan.spacing, scan.axcodes)
            cases[post].append({"case": case_id(img_p), "seconds": round(dt, 1), "scores": sc})
            line.append(f"{post}: Dice WT/TC/ET {sc['WT']['dice']:.3f}/{sc['TC']['dice']:.3f}/"
                        f"{sc['ET']['dice']:.3f} WT vol {sc['WT']['pred']['volume_cm3']:.1f}")
        print(f"[{i:2d}/{len(test)}] {case_id(img_p)}  "
              f"(expert WT {sc['WT']['gt']['volume_cm3']:.1f} cm3, {dt:.0f}s)  " + " | ".join(line))

    for post in posts:
        summary = summarize(cases[post])
        name = args.name or Path(args.model).stem
        if len(posts) > 1:
            name = f"{name} [{post}]"
        result = {"name": name, "model": str(args.model), "tta": args.tta, "post": post,
                  "part": args.part, "n_cases": len(cases[post]),
                  "seconds_per_case": round(t_total / len(test), 1),
                  "train_config": ckpt.get("train_config"), "summary": summary,
                  "cases": cases[post]}
        _print_summary(name, summary, len(test))
        if args.json:
            out = Path(args.json)
            if len(posts) > 1:
                out = out.with_name(f"{out.stem}_{post}{out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(f"-> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="write a fixed train/val/test split")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out", default="data/brats_split.json")
    sp.add_argument("--test", type=int, default=30)
    sp.add_argument("--val", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--not-in-test", help="text file of case ids to keep out of the test set")
    sp.set_defaults(fn=_cmd_split)

    rp = sub.add_parser("run", help="evaluate a model on the held-out patients")
    rp.add_argument("--data", required=True)
    rp.add_argument("--split", default="data/brats_split.json")
    rp.add_argument("--part", default="test", choices=["train", "val", "test"])
    rp.add_argument("--model", required=True)
    rp.add_argument("--name", default="")
    rp.add_argument("--tta", action="store_true", help="average with the left-right flip")
    rp.add_argument("--post", default="largest",
                    help="clean-up of separate predicted pieces: largest, min1cm3 or all; "
                         "a comma list scores several from one inference pass "
                         "(one JSON each, suffixed with the variant)")
    rp.add_argument("--limit", type=int, default=0)
    rp.add_argument("--threads", type=int, default=0)
    rp.add_argument("--json", default="")
    rp.set_defaults(fn=_cmd_run)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
