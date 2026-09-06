# Data

This folder holds datasets locally; the large files are **git-ignored**.

## Real data - BraTS / Medical Segmentation Decathlon `Task01_BrainTumour`

4-modality brain MRI (FLAIR, T1w, T1gd, T2w) with expert tumour masks, NIfTI with
real millimetre voxel spacing. Fetch a small labelled subset (a few hundred MB,
not the full 7.6 GB) with:

```bash
python scripts/download_brats.py --n 30 --out data/brats_subset
```

This lands as:

```
data/brats_subset/Task01_BrainTumour/
  imagesTr/BRATS_XXX.nii.gz   # 4D: (H, W, D, 4)
  labelsTr/BRATS_XXX.nii.gz   # 3D integer mask {0 bg, 1 edema, 2 non-enh, 3 enh}
```

Source: Medical Segmentation Decathlon (http://medicaldecathlon.com/), CC-BY-SA 4.0.
Please cite the MSD / BraTS challenges if you use this data.

## Synthetic data (for tests / instant demo)

```bash
python -c "from brats_report.synthetic import make_dataset; make_dataset('data/synthetic', n=12)"
```

Phantoms in the same layout, generated in seconds - used by the test suite and by
`brats-report --demo`.
