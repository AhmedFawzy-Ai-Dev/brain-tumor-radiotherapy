"""Train the 2-D single-image tumour segmenter on the LGG dataset.

Split is by PATIENT (no slice leakage). Reports binary Dice on a held-out set.

    python -m brats_report.train_seg2d --data data/lgg_raw --epochs 15 --out models/seg2d.pt
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LGGSlices, list_lgg_pairs
from .losses import DiceBCELoss, TverskyBCELoss, dice_per_channel
from .model import UNet2D, count_params


def split_by_patient(triples, val_frac=0.2, seed=0):
    by_pat = defaultdict(list)
    for t in triples:
        by_pat[t[0]].append(t)
    pats = list(by_pat)
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n_val = max(1, int(len(pats) * val_frac))
    val_p, train_p = pats[:n_val], pats[n_val:]
    tr = [t for p in train_p for t in by_pat[p]]
    va = [t for p in val_p for t in by_pat[p]]
    return tr, va, len(train_p), len(val_p)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        tot += float(dice_per_channel(model(x.to(device)), y.to(device))[0])
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="LGG root (contains *_mask.tif)")
    ap.add_argument("--out", default="models/seg2d.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--loss", choices=["tversky", "dicebce"], default="tversky")
    ap.add_argument("--empty-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    triples = list_lgg_pairs(args.data)
    if not triples:
        raise SystemExit(f"no *_mask.tif pairs under {args.data}")
    tr, va, npt, npv = split_by_patient(triples, args.val_frac, args.seed)
    print(f"{len(triples)} slices, {npt} train / {npv} val patients")
    train_ds = LGGSlices(tr, size=args.size, empty_frac=args.empty_frac,
                         augment=True, seed=args.seed)
    val_ds = LGGSlices(va, size=args.size, empty_frac=0.0, augment=False, seed=args.seed)
    print(f"materialised slices: {len(train_ds)} train / {len(val_ds)} val")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet2D(in_ch=1, out_ch=1, base=args.base).to(device)
    print(f"UNet2D(1->1) base={args.base}, {count_params(model)/1e6:.2f}M params, {device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = TverskyBCELoss() if args.loss == "tversky" else DiceBCELoss()
    print(f"loss={args.loss}, train empty_frac={args.empty_frac}; val Dice = tumour slices only")

    best, hist = -1.0, []
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
        dice = evaluate(model, val_dl, device)
        hist.append({"epoch": ep, "train_loss": run / len(train_dl), "val_dice": round(dice, 4)})
        print(f"epoch {ep:2d}/{args.epochs}  loss {run/len(train_dl):.4f}  "
              f"val Dice {dice:.4f}  ({time.time()-t0:.0f}s)")
        if dice > best:
            best = dice
            torch.save({"state_dict": model.state_dict(), "base": args.base,
                        "in_ch": 1, "out_ch": 1, "size": args.size,
                        "val_dice": round(best, 4)}, out)

    out.with_suffix(".metrics.json").write_text(json.dumps(
        {"best_val_dice": round(best, 4), "history": hist,
         "n_slices": len(triples)}, indent=2))
    print(f"\nbest val Dice {best:.4f}  ->  {out}")


if __name__ == "__main__":
    main()
