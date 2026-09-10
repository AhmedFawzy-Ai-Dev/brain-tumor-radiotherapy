"""Tests for the 2-D single-image segmentation + measurement + report path."""
import numpy as np

from brats_report.measure import measure_mask_2d
from brats_report.model import UNet2D
from brats_report.report import generate_report_2d
from brats_report.seg2d import predict_mask2d


def test_measure_mask_2d_rectangle():
    mask = np.zeros((100, 100), np.uint8)
    mask[35:65, 30:70] = 1               # 30 tall x 40 wide
    m = measure_mask_2d(mask, (1.0, 1.0), "px")
    assert m["present"]
    assert m["bbox_w"] == 40 and m["bbox_h"] == 30
    assert abs(m["area"] - 1200) < 1
    assert 47 <= m["recist_long"] <= 50   # rectangle diagonal ~48.6


def test_measure_mask_2d_mm_scale():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 1               # 10x10 px
    m = measure_mask_2d(mask, (2.0, 2.0), "mm")   # 2mm pixels
    assert m["unit"] == "mm"
    assert m["bbox_w"] == 20 and m["bbox_h"] == 20   # 10 px * 2 mm
    assert "area_cm2" in m


def test_measure_mask_2d_empty():
    m = measure_mask_2d(np.zeros((20, 20), np.uint8))
    assert m["present"] is False


def test_predict_mask2d_shape():
    model = UNet2D(in_ch=1, out_ch=1, base=8).eval()
    ckpt = {"in_ch": 1, "out_ch": 1, "base": 8, "size": 64}
    mask = predict_mask2d(np.random.rand(120, 90).astype(np.float32) * 255, model, ckpt)
    assert mask.shape == (120, 90)
    assert mask.dtype == np.uint8


def test_generate_report_2d_writes_files(tmp_path):
    img = (np.random.rand(100, 100).astype(np.float32) * 255)
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:70] = 1
    m2d = measure_mask_2d(mask, (1.0, 1.0), "px")
    tt = {"class": "glioma", "name": "Glioma", "confidence": 0.9,
          "probs": {"glioma": 0.9, "meningioma": 0.05, "notumor": 0.03, "pituitary": 0.02}}
    paths = generate_report_2d(img, mask, m2d, tmp_path / "rep",
                               model_info={"name": "test", "val_dice": "n/a"}, tumour_type=tt)
    for key in ("pdf", "markdown", "json", "image"):
        assert paths[key].exists() and paths[key].stat().st_size > 0
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "Glioma" in md and "مساحة الورم" in md
