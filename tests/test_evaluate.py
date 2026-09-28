"""The evaluation metrics decide what the README claims, so they are tested too."""
import numpy as np

from brats_report.evaluate import dice, hd95, make_split, score_case, summarize
from brats_report.synthetic import ellipsoid, make_label

AX = ("R", "A", "S")


def _ball(shape=(64, 64, 48), center=(32, 32, 24), r=10):
    return ellipsoid(shape, center, (r, r, r))


def test_dice_bounds():
    a = _ball()
    assert dice(a, a) == 1.0
    assert dice(a, np.zeros_like(a)) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0   # both empty: correct


def test_hd95_shifted_ball_in_mm():
    a = _ball(center=(32, 32, 24))
    b = _ball(center=(35, 32, 24))                 # shifted 3 voxels
    assert abs(hd95(a, b, (1.0, 1.0, 1.0)) - 3.0) <= 1.0
    assert abs(hd95(a, b, (2.0, 1.0, 1.0)) - 6.0) <= 2.0   # spacing is honoured
    assert hd95(a, a, (1.0, 1.0, 1.0)) == 0.0
    assert np.isnan(hd95(a, np.zeros_like(a), (1.0, 1.0, 1.0)))


def test_split_is_disjoint_and_respects_blocklist():
    ids = [f"BRATS_{i:03d}" for i in range(50)]
    blocked = ids[:20]
    s = make_split(ids, n_test=10, n_val=5, seed=0, not_in_test=blocked)
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == 50 and len(te) == 10 and len(va) == 5
    assert not te & set(blocked)
    assert make_split(ids, 10, 5, 0, blocked) == s          # deterministic


def test_evaluate_cli_end_to_end(tmp_path):
    # split -> run with every post-processing variant -> one JSON per variant
    import json

    import torch

    from brats_report.evaluate import main
    from brats_report.model import UNet2D
    from brats_report.synthetic import save_case

    for i in range(4):
        save_case(tmp_path / "data", f"S_{i:03d}", shape=(40, 40, 24), semi_axes=(7, 6, 4),
                  seed=i)
    split = tmp_path / "split.json"
    main(["split", "--data", str(tmp_path / "data"), "--test", "2", "--val", "1",
          "--out", str(split)])
    ckpt = tmp_path / "tiny.pt"
    torch.save({"state_dict": UNet2D(4, 3, base=4).state_dict(), "base": 4, "in_ch": 4,
                "out_ch": 3, "size": 32}, ckpt)
    main(["run", "--data", str(tmp_path / "data"), "--split", str(split), "--model", str(ckpt),
          "--post", "largest,min1cm3,all", "--json", str(tmp_path / "res.json")])
    for post in ("largest", "min1cm3", "all"):
        res = json.loads((tmp_path / f"res_{post}.json").read_text())
        assert res["post"] == post and res["n_cases"] == 2
        assert set(res["summary"]) >= {"WT", "TC", "ET", "dice_mean_all"}


def test_perfect_prediction_scores_zero_error():
    gt = make_label(shape=(64, 64, 48), center=(32, 32, 24), semi_axes=(12, 9, 7))
    sc = score_case(gt, gt, (1.0, 1.0, 1.0), AX)
    for r in ("WT", "TC", "ET"):
        assert sc[r]["dice"] == 1.0
        assert sc[r]["pred"] == sc[r]["gt"]
    summary = summarize([{"scores": sc}, {"scores": sc}])
    assert summary["WT"]["volume_mae_cm3"] == 0.0
    assert summary["WT"]["volume_median_abs_pct"] == 0.0
    assert summary["dice_mean_all"] == 1.0
