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
ranges (a few hundred MB total), parse each as a partial tar, and write out matched
image/label pairs. Each range is fetched as parallel chunks (``--workers``
connections; a single S3 connection is often slow) cached on disk, so a dropped
connection or an interrupted run resumes instead of restarting.

Usage::

    python scripts/download_brats.py --n 40 --out data/brats_subset
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
                retries: int = 20, progress=None) -> bytes:
    """Download bytes [start, end] (end inclusive; None = to EOF) with resume.

    ``progress(n)`` is called with each chunk size; without it progress is printed."""
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
                    if progress is not None:
                        progress(len(chunk))
                    else:
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
    if progress is None:
        print()
    return bytes(buf)


def fetch_to_file(url: str, start: int, end: int, dest: Path, label: str,
                  workers: int = 8, chunk_mb: int = 8) -> Path:
    """Download bytes [start, end] (inclusive) into ``dest`` with parallel range
    requests. Finished chunks are kept in ``<dest>.parts/`` until the whole range
    is assembled, so re-running after an interruption skips them."""
    parts = dest.with_name(dest.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    size = chunk_mb << 20
    ranges = [(i, s, min(s + size - 1, end)) for i, s in enumerate(range(start, end + 1, size))]
    total = end - start + 1
    lock, got = threading.Lock(), [0]
    t0 = time.time()

    def bump(n):
        with lock:
            got[0] += n
            mb, secs = got[0] / (1 << 20), max(time.time() - t0, 1e-6)
            print(f"\r  [{label}] {mb:.0f}/{total / (1 << 20):.0f} MB  "
                  f"({mb / secs:.1f} MB/s, {workers} connections)", end="", flush=True)

    def get(i, s, e):
        path = parts / f"{i:05d}"
        if path.exists() and path.stat().st_size == e - s + 1:
            bump(e - s + 1)
            return
        data = fetch_range(url, s, e, label, progress=bump)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in [ex.submit(get, *r) for r in ranges]:
            fut.result()
    print()
    with open(dest, "wb") as out:
        for i, _, _ in ranges:
            with open(parts / f"{i:05d}", "rb") as f:
                shutil.copyfileobj(f, out)
    shutil.rmtree(parts)
    return dest


def extract_tar_fragment(data, want_dir: str, out_dir: Path,
                         limit: int | None = None) -> list[str]:
    """Parse `data` (bytes or a binary file object holding a valid tar that starts
    at a member header, possibly truncated) as a stream and
    write out real files under ``ROOT/<want_dir>/``. Stops cleanly at truncation.
    Skips macOS AppleDouble ``._`` companions. Returns saved BRATS ids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    fileobj = io.BytesIO(data) if isinstance(data, bytes | bytearray) else data
    tf = tarfile.open(fileobj=fileobj, mode="r|")
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
    ap.add_argument("--suffix-mb", type=int, default=32,
                    help="bytes to pull for labels (from end of file; the ~480 labels "
                         "are ~30 KB each, so 32 MB holds all of them)")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--workers", type=int, default=8, help="parallel connections")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the downloaded tar fragments in <out>/_download")
    args = ap.parse_args()

    out = args.out
    img_dir = out / ROOT / "imagesTr"
    lbl_dir = out / ROOT / "labelsTr"

    print(f"Target: {args.n} patients -> {out}")
    clen = content_length(args.url)
    print(f"Archive size: {clen/1e9:.2f} GB (range requests supported)")

    # 1) images from the prefix
    cache = out / "_download"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/2] Downloading first ~{args.prefix_mb} MB for images...")
    end = min(args.prefix_mb * (1 << 20), clen) - 1
    prefix = fetch_to_file(args.url, 0, end, cache / f"prefix_{args.prefix_mb}MB.tar",
                           "images", workers=args.workers)
    with open(prefix, "rb") as f:
        imgs = extract_tar_fragment(f, "imagesTr", img_dir, limit=args.n)
    print(f"  -> {len(imgs)} image volumes")

    # 2) labels from the suffix (all labels live at the tail)
    suffix_mb = args.suffix_mb
    labels: list[str] = []
    for _ in range(3):
        print(f"\n[2/2] Downloading last ~{suffix_mb} MB for labels...")
        start = ((clen - suffix_mb * (1 << 20)) // BLOCK) * BLOCK
        start = max(start, 0)
        suffix = fetch_to_file(args.url, start, clen - 1, cache / f"suffix_{suffix_mb}MB.tar",
                               "labels", workers=args.workers).read_bytes()
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

    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
    print(f"\nDone. {len(common)} matched image/label pairs in {out}")
    if not common:
        print("WARNING: no matched pairs - try a larger --prefix-mb/--suffix-mb")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
