"""Download the trained models from the GitHub release into models/.

    python scripts/download_models.py           # what the app uses (~80 MB)
    python scripts/download_models.py --all     # + the 24-patient baseline U-Net (~98 MB)

Every file is checked against its SHA-256 below before it is kept. Checkpoints
are read with ``torch.load`` (pickle), so never load one whose hash doesn't match.
Files already present with the right hash are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = ("https://github.com/AhmedFawzy-Ai-Dev/brain-tumor-radiotherapy/"
           "releases/download/v0.2.0")

MODELS = {   # file -> (sha256, what it is)
    "brats_unet.pt": ("21e83150a577ce9e5f93888cc3295c9e66f0d02837423b3028a7031dca9404db",
                      "3-D segmentation U-Net, 80 training patients (the app's default)"),
    "tumour_clf.pt": ("f4e82dc99a3c2182716b9b3e2b7fdf33656ebe7e1dff1c98106b6e89cfa5e72a",
                      "tumour-type classifier for 2-D clinical images"),
    "seg2d.pt": ("aa6c828be594d7289a2c2d29cd4cc7746e2cc0751f7ef810226666d7deae38e3",
                 "2-D single-image segmenter (LGG, FLAIR)"),
}
BASELINE = {
    "brats_unet_baseline.pt": ("efa0af00506b018e98e417ba56ae9204feb255353f907d48ee5f593fad0e2a9f",
                               "the first U-Net, 24 training patients (for `make eval`)"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, digest: str, out: Path) -> None:
    dest = out / name
    if dest.exists() and sha256(dest) == digest:
        print(f"  {name}: already there")
        return
    part = dest.with_name(dest.name + ".part")
    h, got = hashlib.sha256(), 0
    with urllib.request.urlopen(f"{RELEASE}/{name}", timeout=60) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {name}: {got / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if h.hexdigest() != digest:
        part.unlink()
        raise SystemExit(f"{name}: SHA-256 mismatch, file discarded")
    part.replace(dest)
    print(f"  {name}: ok (SHA-256 verified)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    ap.add_argument("--all", action="store_true", help="also the 24-patient baseline U-Net")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {**MODELS, **(BASELINE if args.all else {})}
    print(f"Downloading {len(wanted)} model(s) into {args.out}")
    for name, (digest, what) in wanted.items():
        print(f"- {what}")
        fetch(name, digest, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
