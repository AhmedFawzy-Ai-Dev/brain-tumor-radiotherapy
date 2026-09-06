"""Download the 4-class Brain Tumour MRI classification dataset and lay it out as
ImageFolder (Training/ + Testing/ with glioma/ meningioma/ notumor/ pituitary/).

Source: the public HuggingFace mirror `Simezu/brain-tumour-MRI-scan` (the standard
Kaggle 7023-image set: 5712 train / 1311 test). We pull the auto-generated
**parquet** (one file per split) rather than 7000 individual images - two robust,
resumable downloads - then decode the embedded JPEGs into folders.

Usage::

    python scripts/download_tumor_cls.py --out data/tumor_cls
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

REPO = "Simezu/brain-tumour-MRI-scan"
API = f"https://huggingface.co/api/datasets/{REPO}/parquet"


def fetch(url: str, dest: Path, label: str, retries: int = 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        pos = dest.stat().st_size if dest.exists() else 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    print(f"\r  [{label}] {dest.stat().st_size/1e6:.1f} MB", end="", flush=True)
            print()
            return
        except Exception as e:  # noqa: BLE001 - resume from current size
            if attempt >= retries:
                raise
            print(f"\n  [{label}] retry {attempt+1} after {e}")
            time.sleep(min(2 * (attempt + 1), 20))


def canonical(name: str) -> str:
    """'1-notumor' / '2-glioma' -> 'notumor' / 'glioma'."""
    return name.split("-")[-1].strip().lower().replace(" ", "")


def label_names(table) -> list[str] | None:
    meta = table.schema.metadata or {}
    if b"huggingface" in meta:
        try:
            info = json.loads(meta[b"huggingface"])
            return info["info"]["features"]["label"]["names"]
        except Exception:  # noqa: BLE001
            return None
    return None


def export_split(parquet_path: Path, out_dir: Path, split: str, max_per_class=None):
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path)
    names = label_names(t) or ["1-notumor", "2-glioma", "3-meningioma", "4-pituitary"]
    canon = [canonical(n) for n in names]
    df = t.to_pandas()
    per = {}
    for i, row in df.iterrows():
        lab = canon[int(row["label"])]
        if max_per_class and per.get(lab, 0) >= max_per_class:
            continue
        img = row["image"]
        b = img["bytes"] if isinstance(img, dict) else img
        d = out_dir / split / lab
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{split.lower()}_{i:05d}.jpg").write_bytes(io.BytesIO(b).getvalue())
        per[lab] = per.get(lab, 0) + 1
    print(f"  {split}: wrote {sum(per.values())} images {dict(per)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/tumor_cls"))
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all")
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()

    with urllib.request.urlopen(API, timeout=60) as r:
        urls = json.load(r)["default"]
    tmp = args.out / "_parquet"
    max_pc = args.max_per_class or None
    split_map = {"train": "Training", "test": "Testing"}
    for split, folder in split_map.items():
        url = urls[split][0]
        pqf = tmp / f"{split}.parquet"
        print(f"[{split}] {url}")
        fetch(url, pqf, split)
        export_split(pqf, args.out, folder, max_per_class=max_pc)

    if not args.keep_parquet:
        for p in tmp.glob("*.parquet"):
            p.unlink()
        try:
            tmp.rmdir()
        except OSError:
            pass
    print(f"\nDone -> {args.out}  (Training/ + Testing/)")


if __name__ == "__main__":
    main()
