# Evaluation results

Every number in the main README comes from a file here. All segmentation files
are per patient, on the same 30 held-out BraTS patients (`data/brats_split.json`,
test part), written by `python -m brats_report.evaluate run`.

| file | what |
|---|---|
| `01_baseline_<post>.json` | model trained on 24 patients (the original prototype), no TTA |
| `02_tta_<post>.json` | the same model with left-right flip TTA |
| `03_more_data_<post>.json` | the same recipe trained on the 80 training patients of the split |
| `04_more_data_tta_<post>.json` | that model with flip TTA (the app's default) |
| `islands.json` | on the **expert** masks: diameters over every voxel (the old code) vs. on the largest lesion, and what each inflated diameter reached — `scripts/islands_effect.py` |
| `classifier.json` | tumour-type classifier on the 1,311-image test set (confusion matrix included) |

`<post>` is how separate predicted pieces were cleaned up — `largest` (keep only
the largest), `min1cm3` (drop pieces under 1 cm³), `all` (keep everything);
each run scores all three from one inference pass.

Each segmentation file holds, per patient and per region (WT / TC / ET): Dice,
HD95 in mm, and the volume, max 3-D diameter and longest axial diameter measured
on the prediction and on the expert mask. `summary` aggregates them (mean Dice,
median HD95, volume error and bias, diameter errors).
