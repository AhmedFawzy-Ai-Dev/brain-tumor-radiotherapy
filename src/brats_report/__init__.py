"""brats_report - a research prototype that segments a brain tumor from an MRI
scan, measures it in real millimetres, and drafts a structured radiotherapy
report for clinician review.

NOT A MEDICAL DEVICE. Every output is an AI-generated draft and must be verified
by a qualified clinician before any clinical use.
"""

__version__ = "0.1.0"

# MSD Task01_BrainTumour integer label meanings (verified against dataset.json).
LABELS = {0: "background", 1: "edema", 2: "non-enhancing tumor", 3: "enhancing tumor"}

# Standard BraTS reporting regions, expressed as sets of the labels above.
REGIONS = {
    "WT": (1, 2, 3),   # whole tumor  (edema + non-enhancing + enhancing)
    "TC": (2, 3),      # tumor core   (non-enhancing + enhancing)
    "ET": (3,),        # enhancing tumor
}
REGION_NAMES = {
    "WT": "Whole tumour (edema + core)",
    "TC": "Tumour core (non-enhancing + enhancing)",
    "ET": "Enhancing tumour",
}

# The 4 MRI channels stacked in each imagesTr volume, in order.
MODALITIES = ("FLAIR", "T1w", "T1gd", "T2w")

DISCLAIMER = (
    "AI-GENERATED DRAFT - NOT A DIAGNOSIS. This report was produced automatically "
    "by a research prototype and has not been validated for clinical use. All "
    "findings, measurements and locations must be independently verified by a "
    "qualified radiologist / radiation oncologist before any clinical decision."
)
