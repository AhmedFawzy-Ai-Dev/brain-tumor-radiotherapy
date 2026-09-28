# Data

This folder holds datasets locally; the large files are **git-ignored**. Two small
files are committed because the results depend on them:

- `brats_split.json` — the fixed train / val / test split (80 / 10 / 30 patients)
- `baseline_cases.txt` — the 30 patients the first (baseline) model was trained on;
  none of them is in the test set, so every model is scored on unseen patients

## Real data - BraTS / Medical Segmentation Decathlon `Task01_BrainTumour`

4-modality brain MRI (FLAIR, T1w, T1gd, T2w) with expert tumour masks, NIfTI with
real millimetre voxel spacing. Fetch 120 labelled patients (~1.3 GB, not the full
7.6 GB) with:

```bash
python scripts/download_brats.py --n 120 --prefix-mb 1300 --out data/brats_subset
```

The archive is one `.tar`; training images sit at its start and all labels at its
end (~30 KB each), so the script range-requests only those two slices, in parallel
chunks cached on disk (an interrupted download resumes). This lands as:

```
data/brats_subset/Task01_BrainTumour/
  imagesTr/BRATS_XXX.nii.gz   # 4D: (H, W, D, 4)
  labelsTr/BRATS_XXX.nii.gz   # 3D integer mask {0 bg, 1 edema, 2 non-enh, 3 enh}
```

Then recreate the split (deterministic, seed 0 — it reproduces `brats_split.json`):

```bash
python -m brats_report.evaluate split --data data/brats_subset --test 30 --val 10 \
    --not-in-test data/baseline_cases.txt --out data/brats_split.json
```

Source: Medical Segmentation Decathlon (http://medicaldecathlon.com/), CC-BY-SA 4.0.
Please cite the MSD / BraTS challenges if you use this data.

## Tumour-type classification (2-D clinical MRI)

`python scripts/download_tumor_cls.py --out data/tumor_cls` — the Kaggle 4-class brain
MRI set (glioma / meningioma / pituitary / no tumour), with its own `Testing/` split.

## 2-D segmentation (LGG)

`python scripts/download_lgg.py --out data/lgg_raw` — ~110 patients of 2-D FLAIR
slices with binary tumour masks, used for the single-image segmenter.

## Synthetic data (for tests / instant demo)

```bash
python -c "from brats_report.synthetic import make_dataset; make_dataset('data/synthetic', n=12)"
```

Phantoms in the same layout, generated in seconds - used by the test suite and by
`brats-report --demo`.
