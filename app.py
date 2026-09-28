"""Gradio web UI for the brain-tumour reporting prototype.

A clinician uploads a scan and gets, on one page: the annotated slice with the
measured diameter, the measurements, the tumour type (for 2-D clinical images),
and a downloadable bilingual PDF + JSON.

    python app.py            # then open http://127.0.0.1:7860

- 4-channel BraTS NIfTI (.nii/.nii.gz) -> 3-D segmentation + measurement
- 2-D MRI image (.jpg/.png) or single .dcm -> 2-D segmentation + measurement + type

NOT A MEDICAL DEVICE - every output is an AI-generated draft for clinician review.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from brats_report import DISCLAIMER, i18n  # noqa: E402
from brats_report.cli import IMG2D_EXT, process_2d  # noqa: E402
from brats_report.io import load_scan  # noqa: E402
from brats_report.measure import measure_all  # noqa: E402
from brats_report.report import generate_report  # noqa: E402

MODELS = ROOT / "models"
SEG_MODEL = MODELS / "brats_unet.pt"
SEG2D_MODEL = MODELS / "seg2d.pt"
CLF_MODEL = MODELS / "tumour_clf.pt"
OUT = ROOT / "reports" / "_ui"
EXAMPLES = ROOT / "data" / "examples"      # optional: `make examples` fills it

DISCLAIMER_HTML = (
    "<div class='disclaimer'>⚠️ <b>Research prototype — not a medical device.</b> "
    f"{DISCLAIMER}<br><span dir='rtl'>{i18n.DISCLAIMER_AR}</span></div>"
)
CSS = """
.disclaimer {background:#fdecea;color:#7a1c14;border:1px solid #f5c2bd;padding:10px 12px;
             border-radius:8px;font-size:0.9em;line-height:1.45}
.timing {color:#52514e;font-size:0.85em}
"""
TYPE_LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour",
               "pituitary": "Pituitary"}
SHORT_REGION = {"WT": "Whole tumour", "TC": "Tumour core", "ET": "Enhancing"}
TABLE_HEAD = ["Region", "Volume cm³", "Extent LR×AP×SI mm", "Longest axial mm", "Max 3-D mm"]

_cache: dict = {}


def _model(kind: str):
    """Load each model once per process."""
    if kind not in _cache:
        if kind == "seg3d":
            from brats_report.infer import load_model
            _cache[kind] = load_model(str(SEG_MODEL))
    return _cache[kind]


def _table_3d(meas) -> list[list]:
    rows = []
    for r, m in meas.items():
        if not m.present:
            rows.append([SHORT_REGION[r], "not detected", "", "", ""])
            continue
        e = m.extent_mm
        rows.append([SHORT_REGION[r], f"{m.volume_cm3:.1f}",
                     f"{e.get('LR', 0):.0f} × {e.get('AP', 0):.0f} × {e.get('SI', 0):.0f}",
                     f"{m.recist_long_mm:.0f} × {m.recist_short_mm:.0f}",
                     f"{m.max_diameter_mm:.0f}"])
    return rows


def _findings_3d(meas) -> str:
    from brats_report.report import _findings_bilingual
    lines = ["#### Findings / النتائج"]
    for en, arb in _findings_bilingual(meas, None):
        lines += [f"- {en}", f"- <span dir='rtl'>{arb}</span>"]
    return "\n".join(lines)


def _findings_2d(m2d, tumour_type) -> str:
    from brats_report.report import _findings_2d as f2d
    lines = ["#### Findings / النتائج"]
    for en, arb in f2d(m2d, tumour_type):
        lines += [f"- {en}", f"- <span dir='rtl'>{arb}</span>"]
    return "\n".join(lines)


def _type_scores(tt):
    """The type panel: shown with the class probabilities, hidden when there are none."""
    if not tt:
        return gr.update(value=None, visible=False)
    return gr.update(value={TYPE_LABELS.get(c, c): p for c, p in tt["probs"].items()},
                     visible=True)


def process(file, run_classifier, tta):
    empty = (None, _type_scores(None), None, "Please upload a scan. / برجاء رفع فحص.",
             None, None, "")
    if not file:
        return empty
    t0 = time.time()
    path = Path(file)
    suffix = path.suffix.lower()

    if suffix in IMG2D_EXT or suffix == ".dcm":
        clf = str(CLF_MODEL) if (run_classifier and CLF_MODEL.exists()) else None
        seg2d = str(SEG2D_MODEL) if SEG2D_MODEL.exists() else None
        paths, tt, m2d = process_2d(path, OUT, classifier=clf, seg2d=seg2d)
        table = None
        if m2d.get("present"):
            u = m2d["unit"]
            table = [["Tumour (2-D)", f"area {m2d['area']:.0f} {u}²", "",
                      f"{m2d['recist_long']:.0f} × {m2d['recist_short']:.0f} {u}", ""]]
        note = ("\n\n_The 2-D outline comes from a segmenter trained on FLAIR slices (LGG "
                "dataset); on other sequences, e.g. contrast-enhanced T1, it is unreliable. "
                "Sizes are in pixels unless the input is a DICOM with pixel spacing._")
        timing = f"2-D pipeline · {time.time() - t0:.1f} s on CPU"
        return (str(paths["image"]), _type_scores(tt), table, _findings_2d(m2d, tt) + note,
                str(paths["pdf"]), str(paths["json"]), timing)

    if not SEG_MODEL.exists():
        return (None, _type_scores(None), None, "3-D segmentation model not found: train "
                "it first (`make train`).", None, None, "")
    from brats_report.infer import predict_label
    scan = load_scan(path)
    model, ckpt = _model("seg3d")
    label = predict_label(scan, model, ckpt, tta=tta)
    model_info = {"name": f"UNet2D base={ckpt['base']}" + (" + flip TTA" if tta else ""),
                  "val_dice_mean": ckpt.get("val_dice_mean", "n/a")}
    meas = measure_all(np.asanyarray(label), scan.spacing, scan.axcodes)
    paths = generate_report(scan, label, meas, OUT, model_info=model_info)
    note = ("\n\n_Tumour type is only predicted for clinical 2-D images: the classifier "
            "was not trained on skull-stripped multimodal volumes like this one._")
    timing = (f"3-D pipeline · {scan.data.shape[2]} slices · {time.time() - t0:.1f} s on CPU")
    return (str(paths["image"]), _type_scores(None), _table_3d(meas), _findings_3d(meas) + note,
            str(paths["pdf"]), str(paths["json"]), timing)


def _examples() -> list[list]:
    if not EXAMPLES.is_dir():
        return []
    files = sorted(p for p in EXAMPLES.iterdir()
                   if p.suffix.lower() in IMG2D_EXT | {".gz", ".nii", ".dcm"})
    return [[str(p)] for p in files]


with gr.Blocks(title="Brain Tumour Report") as demo:
    gr.Markdown("# 🧠 Brain Tumour Report · تقرير ورم الدماغ\n"
                "Upload an MRI scan: the tumour is segmented, **measured in millimetres**, "
                "and a bilingual (English/Arabic) report draft is written for a clinician "
                "to review.")
    gr.HTML(DISCLAIMER_HTML)
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            _ftypes = [".gz", ".nii", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".dcm"]
            inp = gr.File(label="Scan: BraTS NIfTI, 2-D MRI image, or DICOM / الفحص",
                          file_types=_ftypes)
            clf = gr.Checkbox(value=True, label="Classify tumour type (2-D images) / تصنيف النوع")
            tta = gr.Checkbox(value=False,
                              label="Flip test-time augmentation (3-D) / تعزيز وقت الاختبار")
            btn = gr.Button("Generate report / أنشئ التقرير", variant="primary")
            timing = gr.Markdown(elem_classes="timing")
            ex = _examples()
            if ex:
                gr.Examples(ex, [inp], label="Examples: held-out patients / images")
            pdf = gr.File(label="PDF report / التقرير")
            js = gr.File(label="JSON")
        with gr.Column(scale=3):
            with gr.Row():
                img = gr.Image(label="Annotated slice / الشريحة الموضّحة", height=540, scale=3,
                               show_label=False)
                types = gr.Label(label="Tumour type / نوع الورم", num_top_classes=4,
                                 visible=False, scale=2)
            table = gr.Dataframe(headers=TABLE_HEAD, label="Measurements / القياسات",
                                 interactive=False, wrap=True)
            md = gr.Markdown()
    btn.click(process, [inp, clf, tta], [img, types, table, md, pdf, js, timing])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False,
                theme=gr.themes.Soft(primary_hue="blue"), css=CSS)
