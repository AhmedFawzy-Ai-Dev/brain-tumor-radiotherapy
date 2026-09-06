"""Tests for the tumour-type classifier and its report integration.

Uses pretrained=False so no ImageNet weights are downloaded during testing.
"""
import numpy as np

from brats_report.classify import TUMOR_CLASSES, TumourClassifier, predict_type, prep_image


def test_prep_image_shape():
    arr = np.random.rand(180, 220).astype(np.float32) * 255
    x = prep_image(arr, size=224)
    assert tuple(x.shape) == (1, 3, 224, 224)


def test_prep_image_accepts_rgb():
    arr = np.random.rand(64, 64, 3).astype(np.float32)
    x = prep_image(arr, size=128)
    assert tuple(x.shape) == (1, 3, 128, 128)


def test_classifier_forward_and_predict():
    model = TumourClassifier(arch="resnet18", n_classes=4, pretrained=False).eval()
    out = predict_type(model, np.random.rand(120, 120).astype(np.float32) * 255)
    assert out["class"] in TUMOR_CLASSES
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probs"].values()) - 1.0) < 1e-3


def test_report_includes_classification(tmp_path):
    import json

    from brats_report import io as bio
    from brats_report.measure import measure_all
    from brats_report.report import generate_report
    from brats_report.synthetic import save_case

    save_case(tmp_path, "C1", shape=(48, 48, 32), semi_axes=(8, 6, 5))
    scan = bio.load_nifti(tmp_path / "imagesTr" / "C1.nii.gz")
    label = np.asanyarray(bio.load_nifti(tmp_path / "labelsTr" / "C1.nii.gz").data)
    meas = measure_all(label, scan.spacing, scan.axcodes)
    tt = {"class": "glioma", "name": "Glioma", "confidence": 0.88,
          "probs": {"glioma": 0.88, "meningioma": 0.05, "notumor": 0.02, "pituitary": 0.05}}
    paths = generate_report(scan, label, meas, tmp_path / "rep",
                            model_info={"name": "gt", "val_dice_mean": "n/a"}, tumour_type=tt)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["classification"]["name"] == "Glioma"
    assert "Glioma" in paths["markdown"].read_text(encoding="utf-8")
