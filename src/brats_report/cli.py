"""Command-line entry point: scan -> (segmentation + measurement) + (type
classification) -> combined report.

Examples::

    # full: segment + measure + classify a 3-D scan
    brats-report --image case.nii.gz --model models/brats_unet.pt \
        --classifier models/tumour_clf.pt --out reports/case

    # report from an existing mask, plus type classification
    brats-report --image case.nii.gz --label seg.nii.gz \
        --classifier models/tumour_clf.pt --out reports/case

    # classify a single 2-D MRI image (jpg/png) - type only, no 3-D measurements
    brats-report --image slice.jpg --classifier models/tumour_clf.pt --out reports/case

    # synthetic phantom demo (no data/model needed)
    brats-report --demo --out reports/demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import MODALITIES
from .io import Scan, load_nifti, load_scan
from .measure import _slice_axis, measure_all
from .report import _key_slice, generate_report

IMG2D_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_2d_as_scan(path: Path) -> Scan:
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return Scan(arr[:, :, None], (1.0, 1.0, 1.0), ("R", "A", "S"), np.eye(4), source=str(path))


def _classify(scan: Scan, label, clf_path: str):
    from .classify import load_classifier, predict_type
    clf, ck = load_classifier(clf_path)
    zax = _slice_axis(scan.axcodes)
    if label is not None and np.asanyarray(label).any():
        k = _key_slice(np.asanyarray(label), zax)
    else:
        k = scan.data.shape[zax] // 2
    if scan.is_multimodal:
        mod = MODALITIES.index("T1gd")
        img2d = np.take(scan.data[..., mod], k, axis=zax)
    else:
        img2d = np.take(scan.data, k, axis=zax)
    tt = predict_type(clf, img2d, ck)
    tt["model"] = f"{ck.get('arch','resnet18')} (frozen backbone + head)"
    tt["accuracy"] = ck.get("test_accuracy", "n/a")
    # the classifier is trained on clinical 2-D MRI (skull present); a skull-stripped
    # multimodal research volume (BraTS) is out-of-distribution for it -> flag it.
    if scan.is_multimodal and scan.n_modalities >= 4:
        tt["ood"] = True
    return tt


def _run(scan, label, out, patient_meta, model_info, tumour_type=None):
    meas = measure_all(np.asanyarray(label), scan.spacing, scan.axcodes)
    paths = generate_report(scan, label, meas, out, patient_meta, model_info,
                            tumour_type=tumour_type)
    print(f"\nReport written to {out}:")
    for k, p in paths.items():
        print(f"  {k:9s} {p}")
    if tumour_type:
        print(f"\nPredicted type: {tumour_type['name']} "
              f"({tumour_type['confidence']*100:.0f}% confidence)")
    wt = meas["WT"]
    if wt.present:
        print(f"Whole tumour: {wt.volume_cm3:.1f} cm3, longest axial "
              f"{wt.recist_long_mm:.0f} mm, max 3D {wt.max_diameter_mm:.0f} mm, "
              f"location {wt.location}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Brain tumour report (DRAFT).")
    ap.add_argument("--image", help="4-channel NIfTI, a DICOM folder, or a 2-D image")
    ap.add_argument("--label", help="existing integer segmentation NIfTI (skip model)")
    ap.add_argument("--model", help="trained segmentation U-Net checkpoint (.pt)")
    ap.add_argument("--classifier", help="trained tumour-type classifier (.pt)")
    ap.add_argument("--out", default="reports/case", help="output directory")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic phantom")
    ap.add_argument("--patient-id", default="")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args(argv)

    meta = {"Patient ID": args.patient_id or "N/A (de-identified)"}

    if args.demo:
        from . import synthetic
        tmp = Path(args.out) / "_synthetic"
        synthetic.save_case(tmp, "DEMO_000", spacing=(1.0, 1.0, 1.0),
                            shape=(96, 96, 64), semi_axes=(15, 11, 9), seed=7)
        scan = load_nifti(tmp / "imagesTr" / "DEMO_000.nii.gz")
        label = np.asanyarray(load_nifti(tmp / "labelsTr" / "DEMO_000.nii.gz").data)
        tt = _classify(scan, label, args.classifier) if args.classifier else None
        return _run(scan, label, args.out, meta,
                    {"name": "ground-truth (synthetic demo)", "val_dice_mean": "n/a"}, tt)

    if not args.image:
        ap.error("--image is required (or use --demo)")
    path = Path(args.image)

    # 2-D image -> classification only (no 3-D measurements possible)
    if path.suffix.lower() in IMG2D_EXT:
        if not args.classifier:
            ap.error("a 2-D image needs --classifier (no 3-D data to measure)")
        scan = _load_2d_as_scan(path)
        label = np.zeros(scan.data.shape[:3], np.uint8)
        tt = _classify(scan, label, args.classifier)
        return _run(scan, label, args.out, meta,
                    {"name": "n/a (2-D image; no segmentation)", "val_dice_mean": "n/a"}, tt)

    scan = load_scan(args.image)
    if args.label:
        label = np.asanyarray(load_nifti(args.label).data)
        info = {"name": "ground-truth label", "val_dice_mean": "n/a"}
    elif args.model:
        from .infer import load_model, predict_label
        model, ckpt = load_model(args.model)
        label = predict_label(scan, model, ckpt, thresh=args.thresh)
        info = {"name": f"UNet2D base={ckpt['base']}",
                "val_dice_mean": ckpt.get("val_dice_mean", "n/a")}
    else:
        ap.error("provide --model to segment, or --label to use an existing mask")

    tt = _classify(scan, label, args.classifier) if args.classifier else None
    return _run(scan, label, args.out, meta, info, tt)


if __name__ == "__main__":
    main()
