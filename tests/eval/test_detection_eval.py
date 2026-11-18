"""Bộ chấm nhận diện (phase 09 bước 5): CPU, dữ liệu đồ chơi dựng trong tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xla.eval import detection as det

W, H = det.IMG_W, det.IMG_H
# Hộp 20×20 px (small) ở giữa từng khung test có người
BOX = (100.0, 100.0, 120.0, 120.0)


def _frames() -> list[dict]:
    """4 khung test (2 có người, 2 không) ở 2 khối; 2 khung dev-val; 2 khung neg ở 2 khối."""
    rows = []
    for i, (blk, alt) in enumerate([(0, 5.0), (0, 5.0), (1, 10.0), (1, 10.0)]):
        rows.append({"frame": f"r_001_{i:05d}.jpg", "t_unix": 1000.0 + i, "alt_m": alt, "block": blk, "split": "test"})
    for i in range(2):
        rows.append({"frame": f"r_001_{10 + i:05d}.jpg", "t_unix": 2000.0 + i, "alt_m": 6.0, "block": 2, "split": "dev"})
    for i in range(2):
        rows.append({"frame": f"n_001_{i:05d}.jpg", "t_unix": 5000.0 + 100 * i, "alt_m": None, "block": 3 + i,
                     "split": "neg"})
    return rows


TEST_POS = ["r_001_00000.jpg", "r_001_00002.jpg"]
TEST_NEG = ["r_001_00001.jpg", "r_001_00003.jpg"]
DEVVAL = ["r_001_00010.jpg", "r_001_00011.jpg"]
NEG = ["n_001_00000.jpg", "n_001_00001.jpg"]


def _yolo_line(b: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = b
    return f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"


@pytest.fixture
def toy(tmp_path: Path) -> dict[str, Path]:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"seed": 1, "block_s": 30, "frames": _frames()}), encoding="utf-8")
    labels = tmp_path / "labels"
    (labels / "test").mkdir(parents=True)
    (labels / "devval").mkdir()
    for f in TEST_POS:
        (labels / "test" / f.replace(".jpg", ".txt")).write_text(_yolo_line(BOX), encoding="utf-8")
    for f in TEST_NEG:
        (labels / "test" / f.replace(".jpg", ".txt")).write_text("", encoding="utf-8")
    for f in DEVVAL:
        (labels / "devval" / f.replace(".jpg", ".txt")).write_text(_yolo_line(BOX), encoding="utf-8")
    manifest = labels / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "approved_by_owner": False, "test_frames": TEST_POS + TEST_NEG,
                                    "devval_frames": DEVVAL, "hard_frames": [TEST_POS[1]]}), encoding="utf-8")
    return {"split": split, "labels": labels, "manifest": manifest, "dir": tmp_path}


def _csv(path: Path, rows: list[tuple]) -> Path:
    lines = ["frame,x1,y1,x2,y2,conf,ms"] + [",".join(str(v) for v in r) + ",10.0" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(toy: dict[str, Path], rows: list[tuple], **kw) -> dict:
    pred = _csv(toy["dir"] / "pred.csv", rows)
    kw.setdefault("boot_n", 50)
    return det.evaluate(pred, "test", toy["labels"], toy["manifest"], toy["split"], **kw)


def _v(r: dict, key: str) -> float:
    return r["metrics"][key]["value"]


def test_perfect_predictions_give_map_one(toy):
    rows = [(f, *BOX, 0.9) for f in TEST_POS + DEVVAL]
    r = _run(toy, rows)
    assert _v(r, "mAP50") == pytest.approx(1.0)
    assert _v(r, "mAP50_95") == pytest.approx(1.0)
    assert _v(r, "AP_small") == pytest.approx(1.0)
    assert _v(r, "AR_small") == pytest.approx(1.0)
    assert r["n"]["frames"] == 4 and r["n"]["frames_with_person"] == 2
    assert r["official"] is False and r["note"]
    assert r["ms_per_frame"]["mean"] == pytest.approx(10.0)


def test_no_predictions_give_zero(toy):
    r = _run(toy, [], f1_from_devval=False)
    assert _v(r, "mAP50") == 0.0
    assert _v(r, "AR_small") == 0.0
    assert _v(r, "fp_per_neg_frame@0.25") == 0.0


def test_half_correct_matches_hand_computation(toy):
    # khung 1: hộp đúng (0.9); khung 2: hộp lệch hẳn (0.8). Sắp theo điểm: TP rồi FP → recall 0,5, precision [1, 0,5].
    # Precision nội suy = 1 ở 51 mức recall 0,00..0,50, = 0 ở 50 mức còn lại → AP50 = 51/101.
    rows = [(TEST_POS[0], *BOX, 0.9), (TEST_POS[1], 400.0, 300.0, 420.0, 320.0, 0.8)]
    r = _run(toy, rows, f1_from_devval=False)
    assert _v(r, "mAP50") == pytest.approx(51 / 101)
    assert _v(r, "precision@0.25") == pytest.approx(0.5)
    assert _v(r, "recall@0.25") == pytest.approx(0.5)
    assert _v(r, "f1@0.25") == pytest.approx(0.5)


def test_fp_per_negative_frame(toy):
    # khung âm = 2 khung test không người + 2 khung neg = 4; hộp ≥ 0,25: 3 (2 trên neg[0], 1 trên test_neg[0]);
    # hộp 0,1 dưới ngưỡng không tính.
    rows = [(f, *BOX, 0.9) for f in TEST_POS]
    rows += [(NEG[0], 10, 10, 30, 30, 0.6), (NEG[0], 50, 50, 70, 70, 0.3), (TEST_NEG[0], 10, 10, 30, 30, 0.5),
             (NEG[1], 10, 10, 30, 30, 0.1)]
    r = _run(toy, rows, f1_from_devval=False)
    assert r["n"]["negative_frames"] == 4
    assert _v(r, "fp_per_neg_frame@0.25") == pytest.approx(3 / 4)
    # FP trên khung test không người cũng hạ precision của tập test
    assert _v(r, "precision@0.25") == pytest.approx(2 / 3)


def test_bootstrap_same_seed_same_ci(toy):
    rows = [(TEST_POS[0], *BOX, 0.9), (TEST_POS[1], 400.0, 300.0, 420.0, 320.0, 0.8), (NEG[0], 1, 1, 9, 9, 0.7)]
    a = _run(toy, rows, boot_n=200, seed=7, f1_from_devval=False)
    b = _run(toy, rows, boot_n=200, seed=7, f1_from_devval=False)
    assert a["metrics"] == b["metrics"]
    lo, hi = a["metrics"]["mAP50"]["ci95"]
    assert lo <= _v(a, "mAP50") <= hi
    assert a["bootstrap"]["n_blocks"] == {"flight": 2, "neg": 2}


def test_unknown_frame_in_csv_exits_1(toy, capsys):
    pred = _csv(toy["dir"] / "bad.csv", [("khong_co_trong_split.jpg", *BOX, 0.9)])
    code = det.main(["--pred", str(pred), "--labels-dir", str(toy["labels"]), "--split-json", str(toy["split"]),
                     "--boot", "0", "--json"])
    assert code == 1
    assert "không có trong split.json" in capsys.readouterr().err


def test_cli_ok_exit_0(toy, capsys):
    pred = _csv(toy["dir"] / "ok.csv", [(f, *BOX, 0.9) for f in TEST_POS + DEVVAL])
    code = det.main(["--pred", str(pred), "--labels-dir", str(toy["labels"]), "--split-json", str(toy["split"]),
                     "--boot", "20", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["mAP50"]["value"] == pytest.approx(1.0)
    assert out["f1_threshold_from_devval"]["thr"] == pytest.approx(0.9)


def test_threshold_selection_refuses_test_frames(toy):
    rows = [(f, *BOX, 0.9) for f in TEST_POS]
    pred = _csv(toy["dir"] / "p.csv", rows)
    split_of = {r["frame"]: r["split"] for r in _frames()}
    gt = {f: det.read_yolo_label(toy["labels"] / "test" / f.replace(".jpg", ".txt")) for f in TEST_POS}
    preds, _ = det.read_pred_csv(pred, set(split_of))
    evals, _ = det._coco_eval(TEST_POS, gt, preds)
    with pytest.raises(det.TestLeakError):
        det.select_f1_threshold(evals, split_of)


def test_threshold_from_devval_is_applied_to_test(toy):
    # dev-val: hai hộp đúng 0,6 và 0,4 → F1 = 1 khi cắt ở 0,4; ngưỡng đó áp lên test (hộp test 0,9 → recall 1/2)
    rows = [(DEVVAL[0], *BOX, 0.6), (DEVVAL[1], *BOX, 0.4), (TEST_POS[0], *BOX, 0.9)]
    r = _run(toy, rows)
    f1 = r["f1_threshold_from_devval"]
    assert f1["thr"] == pytest.approx(0.4) and f1["f1_devval"] == pytest.approx(1.0)
    assert 0.4 in r["thresholds"]
    assert _v(r, "recall@0.4") == pytest.approx(0.5)


def test_split_by_alt_bin_and_hard(toy):
    rows = [(TEST_POS[0], *BOX, 0.9)]
    r = _run(toy, rows, f1_from_devval=False)
    assert set(r["by_alt_bin"]) == {"3-8", "8-14"}
    assert r["by_alt_bin"]["3-8"]["metrics"]["mAP50"]["value"] == pytest.approx(1.0)
    assert r["by_alt_bin"]["8-14"]["metrics"]["mAP50"]["value"] == 0.0
    assert r["by_hard"]["hard"]["frames"] == 1 and r["by_hard"]["hard"]["metrics"]["recall@0.25"]["value"] == 0.0


def test_accumulate_matches_pycocotools_on_random_data(toy):
    rng = np.random.default_rng(0)
    frames = [f"r_001_{i:05d}.jpg" for i in range(30)]
    gt = {f: np.array([[x, y, x + w, y + h] for x, y, w, h in rng.uniform([0, 0, 5, 5], [500, 400, 80, 60], (k, 4))])
          .reshape(-1, 4) for f, k in zip(frames, rng.integers(0, 4, 30), strict=True)}
    preds = {}
    for f in frames:
        g = gt[f]
        jit = g + rng.normal(0, 4, g.shape) if len(g) else np.empty((0, 4))
        extra = np.array([[x, y, x + 20, y + 20] for x, y in rng.uniform(0, 400, (3, 2))])
        boxes = np.vstack([jit, extra])
        preds[f] = (boxes, rng.uniform(0.001, 1, len(boxes)))
    evals, stats = det._coco_eval(frames, gt, preds)
    p_all, r_all = det.accumulate(evals, "all")
    p_s, r_s = det.accumulate(evals, "small")
    assert p_all[0] == pytest.approx(stats[1], abs=1e-12)
    assert p_all.mean() == pytest.approx(stats[0], abs=1e-12)
    assert p_s.mean() == pytest.approx(stats[3], abs=1e-12)
    assert r_s.mean() == pytest.approx(stats[9], abs=1e-12)
    assert r_all.mean() == pytest.approx(stats[8], abs=1e-12)


def test_missing_label_is_error_unless_allowed(toy):
    (toy["labels"] / "test" / "r_001_00003.txt").unlink()
    with pytest.raises(det.EvalInputError):
        _run(toy, [], f1_from_devval=False)
    r = _run(toy, [], f1_from_devval=False, allow_missing_labels=True)
    assert r["n"]["labels_missing"] == 1 and r["official"] is False


# ---------------------------------------------------------------- tách khung âm + so sánh ghép cặp


def test_fp_split_flight_vs_indoor(toy):
    # khung bay không người: TEST_NEG (2) — 1 hộp ≥ 0,25 → 0,5; khung neg trong nhà (2) — 2 hộp → 1,0; gộp 3/4
    rows = [(f, *BOX, 0.9) for f in TEST_POS]
    rows += [(NEG[0], 10, 10, 30, 30, 0.6), (NEG[0], 50, 50, 70, 70, 0.3), (TEST_NEG[0], 10, 10, 30, 30, 0.5)]
    r = _run(toy, rows, f1_from_devval=False)
    assert _v(r, "fp_per_neg_flight@0.25") == pytest.approx(0.5)
    assert _v(r, "fp_per_neg_indoor@0.25") == pytest.approx(1.0)
    assert _v(r, "fp_per_neg_frame@0.25") == pytest.approx(0.75)
    assert r["n"]["negative_frames_flight"] == 2 and r["n"]["negative_frames_indoor"] == 2


def _cmp(toy, rows_a, rows_b, **kw):
    a = _csv(toy["dir"] / "a.csv", rows_a)
    b = _csv(toy["dir"] / "b.csv", rows_b)
    kw.setdefault("boot_n", 200)
    return det.compare(a, b, "test", toy["labels"], toy["manifest"], toy["split"], **kw)


WEAK = [(TEST_POS[0], 400.0, 300.0, 420.0, 320.0, 0.8), (TEST_POS[1], 400.0, 300.0, 420.0, 320.0, 0.8),
        (NEG[0], 10, 10, 30, 30, 0.9), (NEG[1], 10, 10, 30, 30, 0.9), (TEST_NEG[0], 10, 10, 30, 30, 0.9)]
STRONG = [(f, *BOX, 0.9) for f in TEST_POS]
DV = [(f, *BOX, 0.5) for f in DEVVAL]


def test_compare_identical_gives_zero_diff(toy):
    r = _cmp(toy, STRONG + DV, STRONG + DV)
    for k in ("mAP50", "mAP50_95", "AP_small", "recall@op", "fp_per_neg_frame@op", "fp_per_neg_flight@op"):
        assert r["metrics"][k]["diff"] == 0.0
        assert r["metrics"][k]["ci95"] == [0.0, 0.0]
        assert r["metrics"][k]["p_diff_gt_0"] == 0.0
    assert r["bootstrap"]["paired"] is True


def test_compare_clearly_better_b(toy):
    r = _cmp(toy, WEAK + DV, STRONG + DV)
    m = r["metrics"]["mAP50"]
    assert m["diff"] == pytest.approx(1.0)
    assert m["ci95"][0] > 0 and m["p_b_better"] == pytest.approx(1.0) and m["p_diff_gt_0"] == pytest.approx(1.0)
    fp = r["metrics"]["fp_per_neg_frame@op"]
    assert fp["diff"] < 0 and fp["ci95"][1] < 0 and fp["p_b_better"] == pytest.approx(1.0)
    assert fp["p_diff_gt_0"] == 0.0 and fp["higher_is_better"] is False


def test_compare_same_seed_same_result(toy):
    a = _cmp(toy, WEAK + DV, STRONG + DV, seed=3)
    b = _cmp(toy, WEAK + DV, STRONG + DV, seed=3)
    assert a["metrics"] == b["metrics"]


def test_compare_explicit_thresholds(toy):
    r = _cmp(toy, STRONG, STRONG, thr_a=0.5, thr_b=0.95)
    assert r["op_threshold"] == {"a": {"thr": 0.5, "source": "tham số"}, "b": {"thr": 0.95, "source": "tham số"}}
    assert r["metrics"]["recall@op"]["diff"] == pytest.approx(-1.0)


def test_compare_unknown_frame_exits_1(toy, capsys):
    a = _csv(toy["dir"] / "a.csv", STRONG + DV)
    b = _csv(toy["dir"] / "b.csv", STRONG + DV + [("la.jpg", *BOX, 0.9)])
    code = det.main(["compare", "--pred-a", str(a), "--pred-b", str(b), "--labels-dir", str(toy["labels"]),
                     "--split-json", str(toy["split"]), "--boot", "0", "--json"])
    assert code == 1
    assert "không có trong split.json" in capsys.readouterr().err


def test_compare_bad_usage_exits_2(toy):
    with pytest.raises(SystemExit) as e:
        det.main(["compare", "--pred-a", "x.csv"])
    assert e.value.code == 2


def test_compare_cli_ok(toy, capsys):
    a = _csv(toy["dir"] / "a.csv", WEAK + DV)
    b = _csv(toy["dir"] / "b.csv", STRONG + DV)
    code = det.main(["compare", "--pred-a", str(a), "--pred-b", str(b), "--labels-dir", str(toy["labels"]),
                     "--split-json", str(toy["split"]), "--boot", "50", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["mAP50"]["diff"] == pytest.approx(1.0)
