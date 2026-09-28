"""Tests for bilingual (English + Arabic) report output."""
import numpy as np

from brats_report import i18n
from brats_report import io as bio
from brats_report.measure import measure_all
from brats_report.report import generate_report
from brats_report.synthetic import save_case


def test_ar_reshapes_to_nonempty():
    s = i18n.ar("الورم الكامل")
    assert isinstance(s, str) and len(s) > 0


def test_wrapped_arabic_keeps_line_order():
    # bidi-reordering a whole paragraph and then wrapping it puts the END of the
    # sentence on the first line; ar_para must wrap first, so the first word of
    # the text is on the first line
    import arabic_reshaper

    font, _ = i18n.register_fonts()
    text = i18n.DISCLAIMER_AR
    lines = i18n.ar_para(text, font, 8.5, 250).split("<br/>")
    assert len(lines) >= 3
    first = arabic_reshaper.reshape(text).split()[0][::-1]   # visual (RTL) form
    last = arabic_reshaper.reshape(text).split()[-1][::-1]
    assert first in lines[0] and first not in lines[-1]
    assert last in lines[-1]


def test_location_translation():
    assert i18n.loc_ar("left") == "يسار"
    out = i18n.loc_ar("left, posterior, superior")
    assert "يسار" in out and "خلفي" in out and "علوي" in out


def test_report_is_bilingual(tmp_path):
    save_case(tmp_path, "C1", shape=(48, 48, 32), semi_axes=(9, 7, 5))
    scan = bio.load_nifti(tmp_path / "imagesTr" / "C1.nii.gz")
    label = np.asanyarray(bio.load_nifti(tmp_path / "labelsTr" / "C1.nii.gz").data)
    meas = measure_all(label, scan.spacing, scan.axcodes)
    paths = generate_report(scan, label, meas, tmp_path / "rep",
                            model_info={"name": "gt", "val_dice_mean": "n/a"})
    md = paths["markdown"].read_text(encoding="utf-8")
    # both languages present
    assert "Findings" in md and "النتائج" in md
    assert "الورم الكامل" in md
    assert paths["pdf"].stat().st_size > 0
