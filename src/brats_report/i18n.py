"""Bilingual (English + Arabic) helpers for the report.

Markdown keeps raw Arabic (viewers shape/RTL it themselves). ReportLab cannot
shape Arabic, so for the PDF we reshape + bidi-reorder with `arabic_reshaper` /
`python-bidi`, drawn in an Arabic-capable TTF (Arial on Windows).
"""
# ruff: noqa: E501  (bilingual strings make some lines long)
from __future__ import annotations

from pathlib import Path

# --- Arabic shaping for the PDF ---------------------------------------------

_FONT_NAME = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_registered = False


def register_fonts() -> tuple[str, str]:
    """Register an Arabic-capable TTF for ReportLab; return (regular, bold) names."""
    global _FONT_NAME, _FONT_BOLD, _registered
    if _registered:
        return _FONT_NAME, _FONT_BOLD
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        ("ArabicFont", "ArabicFont-Bold", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("ArabicFont", "ArabicFont-Bold", "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
        ("ArabicFont", "ArabicFont-Bold", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ]
    for reg, bold, rpath, bpath in candidates:
        if Path(rpath).exists():
            pdfmetrics.registerFont(TTFont(reg, rpath))
            pdfmetrics.registerFont(TTFont(bold, bpath if Path(bpath).exists() else rpath))
            _FONT_NAME, _FONT_BOLD = reg, bold
            break
    _registered = True
    return _FONT_NAME, _FONT_BOLD


def ar(text: str) -> str:
    """Reshape + bidi-reorder Arabic text for correct ReportLab rendering."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:  # noqa: BLE001 - if libs missing, fall back to raw text
        return text


# --- translations (raw Arabic; wrap with ar() only for the PDF) --------------

TITLE_EN = "Brain Tumour Report (DRAFT)"
TITLE_AR = "تقرير ورم الدماغ (مسودة)"

DISCLAIMER_AR = (
    "مسودة مولّدة بالذكاء الاصطناعي — ليست تشخيصاً. أُنتج هذا التقرير آلياً بواسطة نموذج بحثي "
    "ولم يُعتمد للاستخدام السريري. يجب أن يتحقق طبيب أشعة/أورام مختص من جميع النتائج والقياسات "
    "والمواقع قبل اتخاذ أي قرار سريري."
)

HEAD = {
    "Scan metadata": "بيانات الفحص",
    "Findings": "النتائج",
    "Classification (tumour type)": "التصنيف (نوع الورم)",
    "Measurements": "القياسات",
    "Annotated key slice": "الشريحة المميّزة الموضّحة",
    "Model": "النموذج",
    "Class": "الفئة",
    "Probability": "الاحتمال",
}

FIELD = {
    "Patient ID": "رقم المريض",
    "Source": "المصدر",
    "Volume shape": "أبعاد الحجم",
    "Voxel spacing (mm)": "مسافة الفوكسل (مم)",
    "Orientation": "الاتجاه",
    "Key slice": "الشريحة المميّزة",
    "Generated": "تاريخ الإنشاء",
    "Predicted tumour type": "نوع الورم المتوقّع",
}

REGION_AR = {
    "WT": "الورم الكامل (وذمة + نواة)",
    "TC": "نواة الورم (غير معزّز + معزّز)",
    "ET": "الورم المعزّز",
}

COL_AR = {
    "Region": "المنطقة",
    "Volume (cm3)": "الحجم (سم³)",
    "LR (mm)": "يمين-يسار (مم)",
    "AP (mm)": "أمام-خلف (مم)",
    "SI (mm)": "علوي-سفلي (مم)",
    "Max 3D (mm)": "أقصى قطر ثلاثي (مم)",
    "RECIST long x short (mm)": "قطر RECIST طولي×عرضي (مم)",
    "Location": "الموقع",
}

DIM_AR = {"LR": "يمين-يسار", "AP": "أمام-خلف", "SI": "علوي-سفلي"}

LOC_AR = {
    "left": "يسار", "right": "يمين", "anterior": "أمامي", "posterior": "خلفي",
    "superior": "علوي", "inferior": "سفلي", "central / midline": "مركزي / منتصف",
}

TUMOUR_AR = {
    "Glioma": "ورم دبقي (جليوما)", "Meningioma": "ورم سحائي",
    "No tumour": "لا يوجد ورم", "Pituitary": "ورم نخامي",
}


def loc_ar(location: str) -> str:
    if location in LOC_AR:
        return LOC_AR[location]
    parts = [p.strip() for p in location.split(",")]
    return "، ".join(LOC_AR.get(p, p) for p in parts)
