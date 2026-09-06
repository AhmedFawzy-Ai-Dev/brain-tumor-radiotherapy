"""End-to-end smoke tests for model, dataset, inference and reporting."""
import numpy as np
import torch

from brats_report import io as bio
from brats_report.dataset import BratsSlices, label_to_multilabel, list_cases
from brats_report.infer import predict_label
from brats_report.losses import dice_per_channel
from brats_report.measure import measure_all
from brats_report.model import UNet2D, count_params
from brats_report.report import generate_report
from brats_report.synthetic import make_dataset, save_case


def test_model_forward_shape():
    model = UNet2D(in_ch=4, out_ch=3, base=8)
    y = model(torch.randn(2, 4, 64, 64))
    assert y.shape == (2, 3, 64, 64)
    assert count_params(model) > 0


def test_dice_perfect_prediction():
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, 4:10, 4:10] = 1
    logits = torch.where(target > 0, 10.0, -10.0)  # confident correct
    d = dice_per_channel(logits, target)
    assert torch.all(d > 0.99)


def test_multilabel_encoding():
    lab = np.array([[0, 1], [2, 3]])
    ml = label_to_multilabel(lab)
    assert ml.shape == (2, 2, 3)
    # value 3 is in all three regions (WT, TC, ET)
    assert ml[1, 1].tolist() == [1.0, 1.0, 1.0]
    # value 1 (edema) is only in WT
    assert ml[0, 1].tolist() == [1.0, 0.0, 0.0]


def test_dataset_item_shapes(tmp_path):
    save_case(tmp_path, "T_000", shape=(48, 48, 32), semi_axes=(8, 6, 5))
    pairs = list_cases(tmp_path)
    ds = BratsSlices(pairs, size=64)
    x, y = ds[0]
    assert x.shape == (4, 64, 64)
    assert y.shape == (3, 64, 64)
    assert len(ds) > 0


def test_infer_returns_label(tmp_path):
    save_case(tmp_path, "T_000", shape=(48, 48, 32), semi_axes=(8, 6, 5))
    scan = bio.load_nifti(tmp_path / "imagesTr" / "T_000.nii.gz")
    ckpt = {"in_ch": 4, "out_ch": 3, "base": 8, "size": 64}
    model = UNet2D(in_ch=4, out_ch=3, base=8).eval()
    label = predict_label(scan, model, ckpt)
    assert label.shape == scan.data.shape[:3]
    assert label.dtype == np.uint8


def test_generate_report_writes_files(tmp_path):
    save_case(tmp_path, "T_000", shape=(64, 64, 40), semi_axes=(12, 9, 7))
    scan = bio.load_nifti(tmp_path / "imagesTr" / "T_000.nii.gz")
    label = np.asanyarray(bio.load_nifti(tmp_path / "labelsTr" / "T_000.nii.gz").data)
    meas = measure_all(label, scan.spacing, scan.axcodes)
    paths = generate_report(scan, label, meas, tmp_path / "report",
                            model_info={"name": "gt", "val_dice_mean": "n/a"})
    for key in ("pdf", "markdown", "json", "image"):
        assert paths[key].exists() and paths[key].stat().st_size > 0


def test_make_dataset(tmp_path):
    ids = make_dataset(tmp_path / "syn", n=3, shape=(48, 48, 32))
    assert len(ids) == 3
    assert len(list_cases(tmp_path / "syn")) == 3
