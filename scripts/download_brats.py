"""Download a small labeled subset of the BraTS / Medical Segmentation Decathlon
`Task01_BrainTumour` dataset **without** pulling the full 7.6 GB archive.

The dataset is hosted as one big ``.tar`` on a public S3 bucket that supports HTTP
range requests. Inside the tar the members are laid out alphabetically::

    Task01_BrainTumour/dataset.json
    Task01_BrainTumour/imagesTr/BRATS_XXX.nii.gz   (4D: FLAIR, T1w, T1gd, T2w)
    Task01_BrainTumour/imagesTs/BRATS_XXX.nii.gz   (no labels -> we skip these)
    Task01_BrainTumour/labelsTr/BRATS_XXX.nii.gz   (3D integer masks)

So a modest **prefix** of the file contains the first N training images, and a
modest **suffix** contains every training label. We grab exactly those two byte
ranges (a few hundred MB total), parse each as a partial tar in memory, and write
out matched image/label pairs. Both ranges are downloaded with a resume loop so a
dropped connection continues instead of restarting.

Usage::

    python scripts/download_brats.py --n 40 --out data/brats_subset
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
ROOT = "Task01_BrainTumour"
BLOCK = 512


def _open(url: str, start: int, end: int | None):
    rng = f"bytes={start}-" + ("" if end is None else str(end))
    req = urllib.request.Request(url, headers={"Range": rng})
    return urllib.request.urlopen(req, timeout=60)


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int | None, label: str,
                retries: int = 20) -> bytes:
    """Download bytes [start, end] (end inclusive; None = to EOF) with resume."""
    total = (end - start + 1) if end is not None else None
    buf = bytearray()
    attempt = 0
    while True:
        cur = start + len(buf)
        if end is not None and cur > end:
            break
        try:
            with _open(url, cur, end) as resp:
                while True:
                    want = 1 << 20
                    if total is not None:
                        remaining = total - len(buf)
                        if remaining <= 0:
                            break
                        want = min(want, remaining)
                    chunk = resp.read(want)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    got = len(buf) / (1 << 20)
                    tot = f"/{total / (1 << 20):.0f}" if total else ""
                    print(f"\r  [{label}] {got:.1f}{tot} MB", end="", flush=True)
            if total is None or len(buf) >= total:
                break
        except Exception as e:  # noqa: BLE001 - flaky network, resume from offset
            attempt += 1
            if attempt > retries:
                raise
            print(f"\n  [{label}] retry {attempt}/{retries} after {e}")
            time.sleep(min(2 * attempt, 20))
    print()
    return bytes(buf)


def extract_tar_fragment(data: bytes, want_dir: str, out_dir: Path,
                         limit: int | None = None) -> list[str]:
    """Parse `data` (a valid tar starting at a member header) as a stream and
    write out real files under ``ROOT/<want_dir>/``. Stops cleanly at truncation.
    Skips macOS AppleDouble ``._`` companions. Returns saved BRATS ids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    tf = tarfile.open(fileobj=io.BytesIO(data), mode="r|")
    try:
        for m in tf:
            name = m.name.split("/")[-1]
            if not m.isfile() or name.startswith("._"):
                continue
            if f"/{want_dir}/" not in m.name:
                continue
            if not name.startswith("BRATS_") or not name.endswith(".nii.gz"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            content = f.read()
            if len(content) != m.size:  # truncated tail -> stop
                break
            (out_dir / name).write_bytes(content)
            saved.append(name.replace(".nii.gz", ""))
            print(f"  saved {want_dir}/{name} ({m.size/1e6:.1f} MB)")
            if limit is not None and len(saved) >= limit:
                break
    except (tarfile.ReadError, tarfile.StreamError, EOFError):
        pass  # ran off the end of the partial buffer -> keep what we have
    return saved


def find_labels_anchor(suffix: bytes) -> int | None:
    """Find the 512-aligned offset of the first `labelsTr` tar header."""
    for off in range(0, len(suffix) - BLOCK, BLOCK):
        hdr = suffix[off:off + BLOCK]
        if hdr[257:262] != b"ustar":
            continue
        nm = hdr[0:100].split(b"\x00")[0]
        if b"labelsTr/" in nm:
            return off
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40, help="number of training patients")
    ap.add_argument("--out", type=Path, default=Path("data/brats_subset"))
    ap.add_argument("--prefix-mb", type=int, default=650,
                    help="bytes to pull for images (bigger -> more patients)")
    ap.add_argument("--suffix-mb", type=int, default=300,
                    help="bytes to pull for labels (from end of file)")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    out = args.out
    img_dir = out / ROOT / "imagesTr"
    lbl_dir = out / ROOT / "labelsTr"

    print(f"Target: {args.n} patients -> {out}")
    clen = content_length(args.url)
    print(f"Archive size: {clen/1e9:.2f} GB (range requests supported)")

    # 1) images from the prefix
    print(f"\n[1/2] Downloading first ~{args.prefix_mb} MB for images...")
    prefix = fetch_range(args.url, 0, args.prefix_mb * (1 << 20) - 1, "images")
    imgs = extract_tar_fragment(prefix, "imagesTr", img_dir, limit=args.n)
    print(f"  -> {len(imgs)} image volumes")

    # 2) labels from the suffix (all labels live at the tail)
    suffix_mb = args.suffix_mb
    labels: list[str] = []
    for _ in range(3):
        print(f"\n[2/2] Downloading last ~{suffix_mb} MB for labels...")
        start = ((clen - suffix_mb * (1 << 20)) // BLOCK) * BLOCK
        start = max(start, 0)
        suffix = fetch_range(args.url, start, None, "labels")
        anchor = find_labels_anchor(suffix)
        if anchor is None:
            suffix_mb *= 2
            print("  labelsTr not found in suffix; enlarging window...")
            continue
        labels = extract_tar_fragment(suffix[anchor:], "labelsTr", lbl_dir)
        break
    print(f"  -> {len(labels)} label volumes")

    # keep only matched pairs
    common = sorted(set(imgs) & set(labels))
    for stray in set(imgs) - set(common):
        (img_dir / f"{stray}.nii.gz").unlink(missing_ok=True)
    for stray in set(labels) - set(common):
        (lbl_dir / f"{stray}.nii.gz").unlink(missing_ok=True)

    print(f"\nDone. {len(common)} matched image/label pairs in {out}")
    if not common:
        print("WARNING: no matched pairs - try a larger --prefix-mb/--suffix-mb")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
