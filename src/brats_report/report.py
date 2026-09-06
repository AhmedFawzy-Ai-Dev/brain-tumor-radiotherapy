"""Render measurements + classification into a **bilingual (English + Arabic)**
annotated image and a structured PDF / JSON / Markdown radiotherapy report draft.

Arabic in the PDF is shaped and bidi-reordered (ReportLab can't do it natively) and
drawn in an Arabic-capable TTF; Markdown keeps raw Arabic. JSON stays English-keyed.
"""
# ruff: noqa: E501  (bilingual strings make some lines long)
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import DISCLAIMER, MODALITIES, REGION_NAMES, __version__, i18n  # noqa: E402
from .io import Scan  # noqa: E402
from .measure import _max_caliper, _slice_axis  # noqa: E402

REGION_RGB = {"WT": (1.0, 0.85, 0.1), "TC": (1.0, 0.5, 0.0), "ET": (0.9, 0.1, 0.1)}


# --- annotated slice ---------------------------------------------------------

def _key_slice(label: np.ndarray, zax: int) -> int:
    wt = label > 0
    areas = [np.take(wt, k, axis=zax).sum() for k in range(label.shape[zax])]
    return int(np.argmax(areas)) if any(areas) else label.shape[zax] // 2


def render_slice(scan: Scan, label: np.ndarray, out_png: Path,
                 display_modality: str = "FLAIR") -> int:
    zax = _slice_axis(scan.axcodes)
    k = _key_slice(label, zax)
    mod = MODALITIES.index(display_modality) if scan.is_multimodal else 0

    img = np.take(scan.data[..., mod] if scan.is_multimodal else scan.data, k, axis=zax)
    lab = np.take(label, k, axis=zax)
    g = img.astype(np.float32)
    lo, hi = np.percentile(g, [1, 99]) if g.max() > g.min() else (0, 1)
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    for r in ("WT", "TC", "ET"):
        m = lab > 0 if r == "WT" else (lab >= 2 if r == "TC" else lab == 3)
        rgb[m] = rgb[m] * 0.45 + np.array(REGION_RGB[r]) * 0.55

    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    ax.imshow(np.rot90(rgb))
    ax.axis("off")
    ys, xs = np.nonzero(lab > 0)
    if len(xs) > 1:
        pts = np.column_stack([xs, ys]).astype(float)
        _, a, b = _max_caliper(pts)
        h = rgb.shape[0]
        ax.plot([a[1], b[1]], [h - 1 - a[0], h - 1 - b[0]], "-", color="cyan",
                lw=1.4, marker="o", ms=3)

    title = f"{display_modality}  |  axial slice {k}" if scan.is_multimodal else "MRI (2-D input)"
    ax.set_title(title, fontsize=9)
    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=9,
                          markerfacecolor=REGION_RGB[r], markeredgecolor="none",
                          label=REGION_NAMES[r]) for r in ("WT", "TC", "ET")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=1, fontsize=6, frameon=False)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return k


# --- findings (bilingual) ----------------------------------------------------

def _type_lines(tumour_type: dict | None) -> tuple[str, str] | None:
    if not tumour_type:
        return None
    conf = tumour_type.get("confidence", 0) * 100
    name = tumour_type.get("name", "?")
    name_ar = i18n.TUMOUR_AR.get(name, name)
    en = (f"Predicted tumour type: {name} (classifier confidence {conf:.0f}%) "
          f"- indicative, verify against imaging.")
    arb = (f"نوع الورم المتوقّع: {name_ar} (ثقة المُصنِّف {conf:.0f}٪) — استرشادي، يجب التحقق منه.")
    if tumour_type.get("ood"):
        en += (" NOTE: this classifier was validated on clinical 2-D MRI; this input is "
               "skull-stripped multimodal (BraTS-style), out-of-distribution, so the type "
               "here is unreliable and shown for demonstration only.")
        arb += (" ملاحظة: دُرِّب المُصنِّف على صور رنين إكلينيكية ثنائية الأبعاد؛ هذه الصورة "
                "منزوعة الجمجمة ومتعددة الأنماط (خارج نطاق التدريب)، لذا التصنيف هنا غير موثوق "
                "ويُعرض للتوضيح فقط.")
    return en, arb


def _findings_bilingual(meas: dict, tumour_type: dict | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    tl = _type_lines(tumour_type)
    if tl:
        out.append(tl)
    wt = meas["WT"]
    if not wt.present:
        out.append(("No tumour segmented above threshold (no 3-D measurements).",
                    "لم يُكتشف ورم فوق العتبة (لا توجد قياسات ثلاثية الأبعاد)."))
        return out
    e = wt.extent_mm
    out.append((
        f"Whole tumour volume {wt.volume_cm3:.1f} cm3; extents "
        f"{e.get('LR','?')} (LR) x {e.get('AP','?')} (AP) x {e.get('SI','?')} (SI) mm; "
        f"longest axial diameter {wt.recist_long_mm:.0f} mm (perpendicular "
        f"{wt.recist_short_mm:.0f} mm); maximum 3-D diameter {wt.max_diameter_mm:.0f} mm.",
        f"حجم الورم الكامل {wt.volume_cm3:.1f} سم³؛ الأبعاد {e.get('LR','?')} (يمين-يسار) × "
        f"{e.get('AP','?')} (أمام-خلف) × {e.get('SI','?')} (علوي-سفلي) مم؛ أطول قطر محوري "
        f"{wt.recist_long_mm:.0f} مم (العمودي عليه {wt.recist_short_mm:.0f} مم)؛ أقصى قطر "
        f"ثلاثي الأبعاد {wt.max_diameter_mm:.0f} مم."))
    out.append((f"Approximate location: {wt.location}.",
                f"الموقع التقريبي: {i18n.loc_ar(wt.location)}."))
    for r in ("TC", "ET"):
        m = meas[r]
        if m.present:
            out.append((f"{REGION_NAMES[r]}: {m.volume_cm3:.1f} cm3, longest axial diameter "
                        f"{m.recist_long_mm:.0f} mm.",
                        f"{i18n.REGION_AR[r]}: {m.volume_cm3:.1f} سم³، أطول قطر محوري "
                        f"{m.recist_long_mm:.0f} مم."))
    return out


def measurements_to_dict(meas: dict) -> dict:
    return {r: {"present": m.present, "volume_cm3": m.volume_cm3, "extent_mm": m.extent_mm,
                "max_diameter_mm": m.max_diameter_mm, "recist_long_mm": m.recist_long_mm,
                "recist_short_mm": m.recist_short_mm, "recist_slice": m.recist_slice,
                "location": m.location} for r, m in meas.items()}


# --- Markdown (bilingual) ----------------------------------------------------

def _h(en: str) -> str:
    return f"## {en} / {i18n.HEAD.get(en, en)}"


def write_markdown(meas, meta, model_info, out_md, tumour_type=None):
    L = [f"# {i18n.TITLE_EN} / {i18n.TITLE_AR}", "",
         f"> {DISCLAIMER}", ">", f"> {i18n.DISCLAIMER_AR}", "", _h("Scan metadata"), ""]
    for k, v in meta.items():
        L.append(f"- **{k} / {i18n.FIELD.get(k, k)}**: {v}")
    L += ["", _h("Findings"), ""]
    for en, arb in _findings_bilingual(meas, tumour_type):
        L.append(f"- {en}")
        L.append(f"- {arb}")
    if tumour_type and tumour_type.get("probs"):
        L += ["", _h("Classification (tumour type)"), "",
              "| Class / الفئة | Probability / الاحتمال |", "|---|---|"]
        for c, p in sorted(tumour_type["probs"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {c} | {p*100:.1f}% |")
    cols = ["Region", "Volume (cm3)", "LR (mm)", "AP (mm)", "SI (mm)",
            "Max 3D (mm)", "RECIST long x short (mm)", "Location"]
    L += ["", _h("Measurements"), "",
          "| " + " | ".join(f"{c} / {i18n.COL_AR[c]}" for c in cols) + " |",
          "|" + "---|" * len(cols)]
    for r, m in meas.items():
        rn = f"{r} - {i18n.REGION_AR[r]}"
        if not m.present:
            L.append(f"| {rn} | not detected / غير مكتشف | - | - | - | - | - | - |")
            continue
        e = m.extent_mm
        loc = f"{m.location} / {i18n.loc_ar(m.location)}"
        L.append(f"| {rn} | {m.volume_cm3:.1f} | {e.get('LR','-')} | {e.get('AP','-')} | "
                 f"{e.get('SI','-')} | {m.max_diameter_mm:.0f} | "
                 f"{m.recist_long_mm:.0f} x {m.recist_short_mm:.0f} | {loc} |")
    L += ["", _h("Model"), "",
          f"- Segmentation / التجزئة: {model_info.get('name','UNet2D')} "
          f"(val Dice {model_info.get('val_dice_mean','n/a')})"]
    if tumour_type:
        L.append(f"- Classifier / المُصنِّف: {tumour_type.get('model','CNN')} "
                 f"(test acc {tumour_type.get('accuracy','n/a')})")
    L += [f"- Generated / تاريخ الإنشاء: {meta.get('Generated')}", "",
          f"_{DISCLAIMER}_", "", f"_{i18n.DISCLAIMER_AR}_"]
    Path(out_md).write_text("\n".join(L), encoding="utf-8")


# --- PDF (bilingual) ---------------------------------------------------------

def write_pdf(meas, meta, model_info, image_png, out_pdf, tumour_type=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm as MM
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    AR, ARB = i18n.register_fonts()
    ar = i18n.ar
    styles = getSampleStyleSheet()
    small = ParagraphStyle("s", parent=styles["Normal"], fontSize=8, leading=10)
    # NB: text is pre-shaped + bidi-reordered (visual order), so draw LTR and
    # right-align - do NOT set wordWrap="RTL" or it double-reverses.
    small_ar = ParagraphStyle("sar", parent=small, fontName=AR, alignment=TA_RIGHT)
    band = ParagraphStyle("band", fontSize=8.5, leading=11, textColor=colors.white)
    band_ar = ParagraphStyle("bandar", fontName=AR, fontSize=8.5, leading=12,
                             textColor=colors.white, alignment=TA_RIGHT)
    navy, red, zebra = colors.HexColor("#1f3a5f"), colors.HexColor("#b00020"), colors.HexColor("#eef2f7")

    def cell(en, arb=None):
        txt = en if arb is None else f"{en}\n{ar(arb)}"
        return Paragraph(txt.replace("\n", "<br/>"),
                         ParagraphStyle("c", fontName=AR, fontSize=7.3, leading=8.6))

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, topMargin=13 * MM,
                            bottomMargin=13 * MM, leftMargin=15 * MM, rightMargin=15 * MM)
    E = [Paragraph(f"{i18n.TITLE_EN}", styles["Title"]),
         Paragraph(ar(i18n.TITLE_AR), ParagraphStyle("tar", parent=styles["Title"],
                                                     fontName=ARB, alignment=TA_RIGHT))]

    dis = Table([[Paragraph("&#9888; " + DISCLAIMER, band)],
                 [Paragraph(ar(i18n.DISCLAIMER_AR), band_ar)]], colWidths=[180 * MM])
    dis.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), red),
                             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    E += [Spacer(1, 6), dis, Spacer(1, 8)]

    if tumour_type:
        conf = tumour_type.get("confidence", 0) * 100
        name = tumour_type.get("name", "?")
        name_ar = i18n.TUMOUR_AR.get(name, name)
        big = ParagraphStyle("big", fontSize=11, leading=14, textColor=colors.white)
        big_ar = ParagraphStyle("bigar", fontName=ARB, fontSize=11, leading=15,
                                textColor=colors.white, alignment=TA_RIGHT)
        tb = Table([[Paragraph(f"Predicted tumour type: <b>{name}</b> (confidence {conf:.0f}%)", big)],
                    [Paragraph(ar(f"نوع الورم المتوقّع: {name_ar} (الثقة {conf:.0f}٪)"), big_ar)]],
                   colWidths=[180 * MM])
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), navy),
                                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        E += [tb, Spacer(1, 8)]

    # scan metadata
    rows = [[cell("Scan metadata", "بيانات الفحص"), ""]]
    for k, v in meta.items():
        rows.append([cell(k, i18n.FIELD.get(k, k)), cell(str(v))])
    mt = Table(rows, colWidths=[55 * MM, 125 * MM])
    mt.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (-1, 0), navy),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    E += [mt, Spacer(1, 8)]

    # findings
    E.append(Paragraph("Findings / " + ar(i18n.HEAD["Findings"]), styles["Heading2"]))
    for en, arb in _findings_bilingual(meas, tumour_type):
        E.append(Paragraph("&bull; " + en, small))
        E.append(Paragraph(ar("• " + arb), small_ar))
        E.append(Spacer(1, 2))
    E.append(Spacer(1, 4))

    # classification probabilities
    if tumour_type and tumour_type.get("probs"):
        E.append(Paragraph("Classification / " + ar(i18n.HEAD["Classification (tumour type)"]),
                           styles["Heading2"]))
        prow = [[cell("Class", "الفئة"), cell("Probability", "الاحتمال")]]
        for c, p in sorted(tumour_type["probs"].items(), key=lambda kv: -kv[1]):
            prow.append([cell(c), cell(f"{p*100:.1f}%")])
        pt = Table(prow, colWidths=[60 * MM, 40 * MM])
        pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra])]))
        E += [pt, Spacer(1, 8)]

    # measurements
    E.append(Paragraph("Measurements / " + ar(i18n.HEAD["Measurements"]), styles["Heading2"]))
    cols = ["Region", "Volume (cm3)", "LR (mm)", "AP (mm)", "SI (mm)",
            "Max 3D (mm)", "RECIST long x short (mm)", "Location"]
    head = [cell(c.replace(" x ", " x\n"), i18n.COL_AR[c]) for c in cols]
    trows = [head]
    for r, m in meas.items():
        if not m.present:
            trows.append([cell(r, i18n.REGION_AR[r])] + [cell("-")] * 7)
            continue
        e = m.extent_mm
        trows.append([
            cell(r, i18n.REGION_AR[r]), cell(f"{m.volume_cm3:.1f}"),
            cell(str(e.get("LR", "-"))), cell(str(e.get("AP", "-"))), cell(str(e.get("SI", "-"))),
            cell(f"{m.max_diameter_mm:.0f}"), cell(f"{m.recist_long_mm:.0f} x {m.recist_short_mm:.0f}"),
            cell(m.location, i18n.loc_ar(m.location))])
    mtab = Table(trows, colWidths=[26 * MM, 20 * MM, 18 * MM, 18 * MM, 18 * MM,
                                   20 * MM, 26 * MM, 34 * MM])
    mtab.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy),
                              ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                              ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra])]))
    E += [mtab, Spacer(1, 8)]

    if Path(image_png).exists():
        E.append(Paragraph("Annotated key slice / " + ar(i18n.HEAD["Annotated key slice"]),
                           styles["Heading2"]))
        E.append(Image(str(image_png), width=92 * MM, height=92 * MM))
    E.append(Spacer(1, 6))
    foot = (f"Segmentation: {model_info.get('name','UNet2D')} "
            f"(val Dice {model_info.get('val_dice_mean','n/a')}). ")
    if tumour_type:
        foot += f"Classifier: {tumour_type.get('model','CNN')} (test acc {tumour_type.get('accuracy','n/a')}). "
    foot += f"brats_report v{__version__}. {meta.get('Generated')}."
    E.append(Paragraph(foot, small))
    E.append(Paragraph("<i>" + DISCLAIMER + "</i>", small))
    E.append(Paragraph(ar(i18n.DISCLAIMER_AR), small_ar))
    doc.build(E)


def generate_report(scan: Scan, label: np.ndarray, meas: dict, out_dir,
                    patient_meta=None, model_info=None, stem="report", tumour_type=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_info = model_info or {}
    png = out_dir / f"{stem}_slice.png"
    key = render_slice(scan, label, png)

    meta = {"Source": Path(scan.source).name or scan.source,
            "Volume shape": " x ".join(map(str, scan.data.shape)),
            "Voxel spacing (mm)": " x ".join(f"{s:.2f}" for s in scan.spacing),
            "Orientation": "".join(scan.axcodes), "Key slice": key,
            "Generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if tumour_type:
        conf = tumour_type.get("confidence", 0) * 100
        meta = {**meta, "Predicted tumour type": f"{tumour_type.get('name','?')} ({conf:.0f}%)"}
    if patient_meta:
        meta = {**patient_meta, **meta}

    pdf, md, js = (out_dir / f"{stem}.{x}" for x in ("pdf", "md", "json"))
    write_pdf(meas, meta, model_info, png, pdf, tumour_type)
    write_markdown(meas, meta, model_info, md, tumour_type)
    js.write_text(json.dumps({"metadata": meta, "measurements": measurements_to_dict(meas),
                              "classification": tumour_type, "model": model_info,
                              "disclaimer": DISCLAIMER}, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    return {"pdf": pdf, "markdown": md, "json": js, "image": png}
