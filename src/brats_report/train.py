"""Train the compact 2-D U-Net on a BraTS/MSD (or synthetic) dataset root.

Usage::

    python -m brats_report.train --data data/brats_subset --epochs 15 --out models/brats_unet.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BratsSlices, list_cases
from .evaluate import case_id, load_split
from .losses import DiceBCELoss, TverskyBCELoss, dice_per_channel
from .model import UNet2D, count_params

REGION_ORDER = ("WT", "TC", "ET")


def _dice_dict(val):
    return {r: round(v, 4) for r, v in zip(REGION_ORDER, val, strict=True)}


def split_cases(pairs, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_frac))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = torch.zeros(3)
    n = 0
    for x, y in loader:
        d = dice_per_channel(model(x.to(device)), y.to(device))
        tot += d.cpu()
        n += 1
    return (tot / max(n, 1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/brats_unet.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="",
                    help="split JSON from `brats_report.evaluate split` (its test cases are "
                         "never trained on); default: random val-frac split")
    ap.add_argument("--cases", default="",
                    help="text file of case ids to restrict training to (with the random "
                         "val-frac split); reproduces the baseline from data/baseline_cases.txt")
    ap.add_argument("--loss", choices=["dicebce", "tversky"], default="dicebce")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list_cases(args.data)
    if args.cases:
        keep = set(Path(args.cases).read_text().split())
        pairs = [p for p in pairs if case_id(p[0]) in keep]
    if not pairs:
        raise SystemExit(f"no imagesTr/labelsTr pairs found under {args.data}")
    if args.split:
        parts = load_split(args.split, pairs)
        train_p, val_p = parts["train"], parts["val"]
        print(f"{len(pairs)} cases, split {args.split}: {len(train_p)} train / "
              f"{len(val_p)} val (+{len(parts['test'])} test, not used here)")
    else:
        train_p, val_p = split_cases(pairs, args.val_frac, args.seed)
        print(f"{len(pairs)} cases -> {len(train_p)} train / {len(val_p)} val")

    train_ds = BratsSlices(train_p, size=args.size, augment=True, seed=args.seed)
    val_ds = BratsSlices(val_p, size=args.size, augment=False, seed=args.seed)
    print(f"slices: {len(train_ds)} train / {len(val_ds)} val")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet2D(in_ch=4, out_ch=3, base=args.base).to(device)
    print(f"model: UNet2D base={args.base}, {count_params(model)/1e6:.2f}M params, device={device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = DiceBCELoss() if args.loss == "dicebce" else TverskyBCELoss()

    history, best = [], -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        val = evaluate(model, val_dl, device)
        mean_val = float(np.mean(val))
        history.append({"epoch": ep, "train_loss": run / len(train_dl),
                        "val_dice": _dice_dict(val),
                        "val_dice_mean": round(mean_val, 4),
                        "seconds": round(time.time() - t0)})
        print(f"epoch {ep:2d}/{args.epochs}  loss {run/len(train_dl):.4f}  "
              f"val Dice WT/TC/ET {val[0]:.3f}/{val[1]:.3f}/{val[2]:.3f}  "
              f"mean {mean_val:.3f}  ({time.time()-t0:.0f}s)")
        if mean_val > best:
            best = mean_val
            torch.save({"state_dict": model.state_dict(), "base": args.base,
                        "in_ch": 4, "out_ch": 3, "size": args.size,
                        "regions": REGION_ORDER, "val_dice_mean": best,
                        "val_dice": _dice_dict(val), "epoch": ep,
                        "n_train_cases": len(train_p), "train_config": vars(args)},
                       out)
        # rewritten every epoch, so an interrupted run still leaves its history
        metrics = {"best_val_dice_mean": round(best, 4), "n_cases": len(pairs),
                   "n_train_cases": len(train_p), "n_val_cases": len(val_p),
                   "n_train_slices": len(train_ds), "history": history,
                   "config": vars(args)}
        out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nbest mean val Dice {best:.3f}  ->  {out}")


if __name__ == "__main__":
    main()
