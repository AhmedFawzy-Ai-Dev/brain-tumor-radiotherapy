"""Download the LGG brain-MRI segmentation dataset (2-D images + tumour masks).

Pulls the single public zip mirror from HuggingFace (the Buda "kaggle_3m" set:
~110 patients, FLAIR-style 2-D slices with binary masks) and extracts it. Used to
train the 2-D single-image tumour segmenter (`brats_report.train_seg2d`).

    python scripts/download_lgg.py --out data/lgg_raw
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = ("https://huggingface.co/datasets/gymprathap/Brain-MRI-LGG-Segmentation/"
       "resolve/main/Brain-MRI-LGG-Segmentation.zip")


def download(url: str, dest: Path, retries: int = 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        pos = dest.stat().st_size if dest.exists() else 0
        try:
            req = Request(url, headers={"Range": f"bytes={pos}-"})
            with urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    print(f"\r  {dest.stat().st_size/1e6:.0f} MB", end="", flush=True)
            print()
            return
        except Exception as e:  # noqa: BLE001 - flaky link, resume from offset
            if attempt >= retries:
                raise
            print(f"\n  retry {attempt+1} after {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/lgg_raw"))
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    zip_path = args.out.parent / "lgg.zip"
    print(f"Downloading LGG dataset -> {zip_path}")
    download(URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    n = len(list(args.out.rglob("*_mask.tif")))
    print(f"Done: {n} image/mask pairs under {args.out}")
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
