"""Train the tumour-type classifier on a folder dataset (Training/ + Testing/,
each with glioma/ meningioma/ notumor/ pituitary/ subfolders - the Kaggle
"Brain Tumor MRI Dataset" layout).

For CPU speed we extract ImageNet-backbone features **once** (frozen), cache them,
then train the light head for many epochs in seconds. Reports honest test metrics
(accuracy, macro precision/recall/F1, per-class) - never hand-written.

Usage::

    python -m brats_report.train_classifier --data data/tumor_cls --epochs 40 \
        --out models/tumour_clf.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .classify import IMAGENET_MEAN, IMAGENET_STD, TUMOR_CLASSES, TumourClassifier


def _tf(size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])


@torch.no_grad()
def extract_features(model, ds, device, batch=32, max_n=None):
    idx = list(range(len(ds)))
    if max_n and len(idx) > max_n:
        idx = list(np.random.default_rng(0).choice(idx, max_n, replace=False))
    dl = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch, shuffle=False)
    feats, labels, done = [], [], 0
    for x, y in dl:
        feats.append(model.features(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
        done += len(y)
        print(f"\r  features {done}/{len(idx)}", end="", flush=True)
    print()
    return np.concatenate(feats), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with Training/ and Testing/")
    ap.add_argument("--out", default="models/tumour_clf.pt")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = use all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    train_dir = root / "Training"
    test_dir = root / "Testing"
    if not train_dir.exists():
        raise SystemExit(f"expected {train_dir} (Kaggle layout). Got: {list(root.iterdir())}")

    tf = _tf(args.size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=tf)
    classes = train_ds.classes
    print(f"classes: {classes}  (train {len(train_ds)}, test {len(test_ds)})")
    if classes != TUMOR_CLASSES:
        print(f"WARNING: class order {classes} != expected {TUMOR_CLASSES}")

    model = TumourClassifier(arch=args.arch, n_classes=len(classes), pretrained=True).to(device)

    print("Extracting features (one-off, frozen backbone)...")
    t0 = time.time()
    max_n = args.max_per_class * len(classes) if args.max_per_class else None
    Xtr, ytr = extract_features(model, train_ds, device, max_n=max_n)
    Xte, yte = extract_features(model, test_ds, device)
    print(f"  done in {time.time()-t0:.0f}s. train feats {Xtr.shape}, test feats {Xte.shape}")

    # train/val split on cached train features
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    nval = int(len(Xtr) * args.val_frac)
    vi, ti = perm[:nval], perm[nval:]
    Xt, yt = torch.tensor(Xtr[ti]), torch.tensor(ytr[ti])
    Xv, yv = torch.tensor(Xtr[vi]), torch.tensor(ytr[vi])

    counts = np.bincount(ytr[ti], minlength=len(classes))
    cw = torch.tensor((counts.sum() / (len(classes) * np.maximum(counts, 1))).astype(np.float32))
    loss_fn = nn.CrossEntropyLoss(weight=cw.to(device))
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.head.train()
        perm2 = torch.randperm(len(Xt))
        run = 0.0
        for s in range(0, len(Xt), 128):
            b = perm2[s:s + 128]
            opt.zero_grad()
            loss = loss_fn(model.head(Xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()
            run += loss.item()
        model.head.eval()
        with torch.no_grad():
            vp = model.head(Xv.to(device)).argmax(1).cpu().numpy()
        vacc = accuracy_score(yv.numpy(), vp)
        if vacc > best:
            best = vacc
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{args.epochs}  loss {run:.3f}  val_acc {vacc:.4f}")

    model.head.load_state_dict(best_state)

    # test metrics
    model.eval()
    with torch.no_grad():
        te_pred = model.head(torch.tensor(Xte).to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(yte, te_pred)
    prec = precision_score(yte, te_pred, average="macro", zero_division=0)
    rec = recall_score(yte, te_pred, average="macro", zero_division=0)
    f1 = f1_score(yte, te_pred, average="macro", zero_division=0)
    print(f"\nTEST  acc {acc:.4f}  macro-P {prec:.4f}  macro-R {rec:.4f}  macro-F1 {f1:.4f}")
    print(classification_report(yte, te_pred, target_names=classes, zero_division=0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                "classes": classes, "size": args.size,
                "test_accuracy": round(float(acc), 4),
                "test_macro_f1": round(float(f1), 4)}, out)
    metrics = {"arch": args.arch, "classes": classes, "best_val_acc": round(float(best), 4),
               "test": {"accuracy": round(float(acc), 4), "macro_precision": round(float(prec), 4),
                        "macro_recall": round(float(rec), 4), "macro_f1": round(float(f1), 4)},
               "confusion_matrix": confusion_matrix(yte, te_pred).tolist(),
               "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
