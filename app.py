"""Gradio web UI for the brain-tumour reporting prototype.

A clinician uploads a scan and gets, on one page: the annotated key slice, the
tumour type, the measurements, and a downloadable bilingual PDF report.

    python app.py            # then open the printed local URL

Accepts a 4-channel BraTS NIfTI (.nii/.nii.gz) -> segmentation + measurement +
type; or a 2-D MRI image (.jpg/.png) -> type classification only.

NOT A MEDICAL DEVICE - every output is an AI-generated draft for clinician review.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from brats_report import DISCLAIMER, i18n  # noqa: E402
from brats_report.cli import IMG2D_EXT, _classify, _load_2d_as_scan  # noqa: E402
from brats_report.io import load_scan  # noqa: E402
from brats_report.measure import measure_all  # noqa: E402
from brats_report.report import generate_report  # noqa: E402

MODELS = Path(__file__).parent / "models"
SEG_MODEL = MODELS / "brats_unet.pt"
CLF_MODEL = MODELS / "tumour_clf.pt"
OUT = Path(__file__).parent / "reports" / "_ui"

DISCLAIMER_HTML = (
    f"<div style='background:#b00020;color:#fff;padding:10px;border-radius:6px'>"
    f"⚠️ {DISCLAIMER}<br><span dir='rtl'>{i18n.DISCLAIMER_AR}</span></div>"
)


def _summary_md(meas, tumour_type) -> str:
    lines = ["### Findings / النتائج"]
    from brats_report.report import _findings_bilingual
    for en, arb in _findings_bilingual(meas, tumour_type):
        lines.append(f"- {en}")
        lines.append(f"- <span dir='rtl'>{arb}</span>")
    wt = meas["WT"]
    if wt.present:
        lines += ["", "| Region | Vol (cm³) | LR | AP | SI | Max3D | RECIST |",
                  "|---|---|---|---|---|---|---|"]
        for r, m in meas.items():
            if not m.present:
                continue
            ee = m.extent_mm
            lines.append(f"| {r} | {m.volume_cm3:.1f} | {ee.get('LR','-')} | "
                         f"{ee.get('AP','-')} | {ee.get('SI','-')} | {m.max_diameter_mm:.0f} | "
                         f"{m.recist_long_mm:.0f}×{m.recist_short_mm:.0f} |")
    return "\n".join(lines)


def process(file, run_classifier):
    if not file:
        return None, "Please upload a scan. / برجاء رفع فحص.", None
    path = Path(file)
    is2d = path.suffix.lower() in IMG2D_EXT

    if is2d:
        scan = _load_2d_as_scan(path)
        label = np.zeros(scan.data.shape[:3], np.uint8)
        model_info = {"name": "n/a (2-D image)", "val_dice_mean": "n/a"}
    else:
        scan = load_scan(path)
        if not SEG_MODEL.exists():
            return None, "Segmentation model not found (train it first).", None
        from brats_report.infer import load_model, predict_label
        model, ckpt = load_model(str(SEG_MODEL))
        label = predict_label(scan, model, ckpt)
        model_info = {"name": f"UNet2D base={ckpt['base']}",
                      "val_dice_mean": ckpt.get("val_dice_mean", "n/a")}

    tumour_type = None
    if run_classifier and CLF_MODEL.exists():
        tumour_type = _classify(scan, label, str(CLF_MODEL))

    meas = measure_all(np.asanyarray(label), scan.spacing, scan.axcodes)
    paths = generate_report(scan, label, meas, OUT, model_info=model_info,
                            tumour_type=tumour_type, stem="ui_report")
    return str(paths["image"]), _summary_md(meas, tumour_type), str(paths["pdf"])


with gr.Blocks(title="Brain Tumour Report") as demo:
    gr.Markdown("# 🧠 Brain Tumour Report / تقرير ورم الدماغ\n"
                "Upload a scan to get a draft report. / ارفع فحصاً للحصول على مسودة تقرير.")
    gr.HTML(DISCLAIMER_HTML)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.File(label="Scan (NIfTI .nii.gz or 2-D image) / الفحص",
                          file_types=[".gz", ".nii", ".jpg", ".jpeg", ".png", ".bmp"])
            clf = gr.Checkbox(value=True, label="Classify tumour type / تصنيف النوع")
            btn = gr.Button("Generate report / أنشئ التقرير", variant="primary")
        with gr.Column(scale=2):
            img = gr.Image(label="Annotated key slice / الشريحة الموضّحة")
            md = gr.Markdown()
            pdf = gr.File(label="Download PDF report / حمّل التقرير")
    btn.click(process, [inp, clf], [img, md, pdf])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
