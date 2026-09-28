"""Copy a few held-out cases into data/examples/ for the web UI's example list.

    python scripts/make_examples.py

Takes the first two BraTS **test-split** patients (never trained on) and one
image per class from the classification **test** set, if those datasets have been
downloaded. Nothing is copied from the training data.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "examples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    split = DATA / "brats_split.json"
    images = DATA / "brats_subset" / "Task01_BrainTumour" / "imagesTr"
    if split.exists() and images.is_dir():
        for case in json.loads(split.read_text())["test"][:2]:
            src = images / f"{case}.nii.gz"
            if src.exists():
                shutil.copy2(src, OUT / src.name)
                copied.append(src.name)
    testing = DATA / "tumor_cls" / "Testing"
    for cls in ("glioma", "meningioma", "pituitary", "notumor"):
        folder = testing / cls
        if folder.is_dir():
            first = sorted(folder.iterdir())[0]
            dst = OUT / f"{cls}_{first.name}"
            shutil.copy2(first, dst)
            copied.append(dst.name)
    print(f"{len(copied)} examples in {OUT}: {', '.join(copied) or 'none (download data first)'}")


if __name__ == "__main__":
    main()
