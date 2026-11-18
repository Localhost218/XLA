"""Phase 10 (E1): kết quả fine-tune YOLO11s trên D0/D1/D2/D2+RealSR — ``results/e1/e1.json`` + CSV dự đoán.

Ca không marker đọc file đã commit trong repo (e1.json, smoke.json, CSV) và kiểm luật chấm bằng dữ liệu tổng hợp. Ca
``data`` đọc lại manifest các tập sinh ở ổ D để kiểm sha256 đã ghi và không có khung thật nào trong tập huấn luyện.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from xla.camera.frames import CROP_LEFT_PX
from xla.detect import train as T
from xla.eval.detection import PRED_FIELDS
from xla.paths import data_root, repo_root

ROOT = repo_root()
E1 = ROOT / T.RESULTS_REL / "e1.json"
SETS = ("d0", "d1", "d2", "d2sr")
BEST_KEYS = {"set", "weights", "devval_map50", "conf_threshold", "threshold_source"}  # thuc-thi-phase-10-14.md §5.5


@pytest.fixture(scope="module")
def e1() -> dict:
    assert E1.is_file(), f"thiếu {E1}: chạy python -m xla.detect.train --score"
    return json.loads(E1.read_text(encoding="utf-8"))


def _split_of() -> dict[str, str]:
    rows = json.loads((ROOT / T.SPLIT_REL).read_text(encoding="utf-8"))["frames"]
    return {r["frame"]: r["split"] for r in rows}


def _all_ci(obj: object, path: str = ""):
    """Mọi cặp (đường dẫn, value, ci95) trong cây JSON."""
    if isinstance(obj, dict):
        if "value" in obj and "ci95" in obj:
            yield path, obj["value"], obj["ci95"]
        for k, v in obj.items():
            yield from _all_ci(v, f"{path}.{k}")


# ---------------------------------------------------------------- e1.json


def test_e1_has_four_sets_and_baseline(e1):
    assert set(e1["sets"]) == set(SETS)
    assert e1["baseline"]["variant"] == "edge16"
    assert e1["baseline"]["test"]["metrics"]["mAP50"]["value"] == pytest.approx(0.1265, abs=5e-4)


def test_each_set_selected_on_devval_with_timings(e1):
    for s in SETS:
        r = e1["sets"][s]
        assert r["selected_by"] == "devval", s
        assert r["smoke_hours_per_epoch"] > 0 and r["gpu_hours"] > 0 and r["hours_per_epoch"] > 0, s
        assert r["test"]["threshold_source"] == "devval", s
        hist = r["devval_history"]
        assert [h["epoch"] for h in hist] == list(range(1, r["epochs"] + 1)), s
        assert r["devval_map50"] == max(h["devval_mAP50"] for h in hist), s
        assert next(h for h in hist if h["epoch"] == r["selected_epoch"])["devval_mAP50"] == r["devval_map50"], s


def test_same_hyperparameters_for_all_sets(e1):
    epochs = {e1["sets"][s]["epochs"] for s in SETS}
    batch = {e1["sets"][s]["batch_effective"] for s in SETS}
    assert len(epochs) == 1 and len(batch) == 1, (epochs, batch)
    assert batch == {e1["config"]["train"]["batch"]}, "ultralytics đã tự hạ batch vì tràn VRAM"
    assert epochs == {e1["config"]["train"]["epochs"]}


def test_train_manifests_hold_no_real_frame(e1):
    real = set(_split_of())
    for s in SETS:
        td = e1["sets"][s]["train_data"]
        assert td["set"] == s and td["real_frames_in_train"] == 0
        assert len(td["manifest_sha256"]) == 64
    assert all(T.REAL_FRAME_RE.match(f) for f in real), "mẫu tên khung thật đã đổi: luật chặn rò rỉ không còn đúng"


def test_ci_brackets_point_estimate(e1):
    items = list(_all_ci(e1["sets"])) + list(_all_ci(e1["baseline"]))
    assert items
    for path, v, (lo, hi) in items:
        if v is None or lo is None or hi is None:
            continue
        assert lo <= v + 1e-9 and v <= hi + 1e-9, f"{path}: {v} ngoài [{lo}, {hi}]"
    for name, c in e1["compare"].items():
        lo, hi = c["metrics"]["mAP50"]["ci95"]
        assert lo <= hi, name


def test_best_matches_contract_and_is_chosen_on_devval(e1):
    b = e1["best"]
    assert set(b) == BEST_KEYS
    assert b["set"] in SETS and b["threshold_source"] == "devval"
    assert isinstance(b["devval_map50"], float) and isinstance(b["conf_threshold"], float)
    assert b["devval_map50"] == max(e1["sets"][s]["devval_map50"] for s in SETS)
    assert b["conf_threshold"] == e1["sets"][b["set"]]["test"]["conf_threshold"]
    w = Path(b["weights"])
    assert w.is_absolute() and w.suffix == ".pt"
    assert ROOT.resolve() not in w.resolve().parents, "trọng số không được nằm trong repo"


def test_paired_compares_present(e1):
    need = {f"edge16_vs_{s}" for s in SETS} | {"d1_vs_d2", "d1_vs_d2sr"}
    assert need <= set(e1["compare"])
    for s in SETS:
        c = e1["sets"][s]["criteria"]
        assert set(c) == {"B1", "B2", "B3"}
        assert c["B2"]["verdict"] in ("đạt", "chưa đủ dữ liệu", "không đạt")


def test_pred_csvs_follow_d11_and_drop_left_edge(e1):
    split_of = _split_of()
    devval = set(T.devval_frames(ROOT))
    for s in SETS:
        for sp in ("test", "neg", "devval"):
            p = ROOT / T.RESULTS_REL / "preds" / f"{s}_{sp}.csv"
            assert p.is_file() and p.stat().st_size < T.CSV_MAX_BYTES, p
            with p.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert rows and set(PRED_FIELDS) <= set(rows[0]), p
            frames = {r["frame"] for r in rows}
            ok = devval if sp == "devval" else {f for f, v in split_of.items() if v == sp}
            assert frames <= ok, f"{p}: khung lạ tập"
            assert min(float(r["x1"]) for r in rows) >= CROP_LEFT_PX, p
            assert min(float(r["conf"]) for r in rows) >= 0.001 - 1e-9, p


# ---------------------------------------------------------------- luật (dữ liệu tổng hợp)


def _summ(fp: float, hi: float, recall: float = 0.5) -> dict:
    return {"metrics": {"fp_per_neg_flight@op": {"value": fp, "ci95": [0.0, hi]},
                        "recall@op": {"value": recall, "ci95": [0.4, 0.6]}}}


def _cmp(b: float, lo: float) -> dict:
    return {"metrics": {"mAP50": {"b": b, "diff": b - 0.1265, "ci95": [lo, lo + 0.2]}}}


def test_criteria_rules():
    c = T.criteria(_cmp(0.30, 0.01), _summ(0.02, 0.08), 25.0)
    assert c["B1"]["pass"] and c["B2"]["verdict"] == "đạt" and c["B3"]["pass_pytorch"]
    assert not T.criteria(_cmp(0.30, -0.01), _summ(0.02, 0.08), 25.0)["B1"]["pass"]
    assert not T.criteria(_cmp(0.20, 0.05), _summ(0.02, 0.08), 25.0)["B1"]["pass"]
    assert T.criteria(_cmp(0.3, 0.1), _summ(0.04, 0.2), 25.0)["B2"]["verdict"] == "chưa đủ dữ liệu"
    assert T.criteria(_cmp(0.3, 0.1), _summ(0.09, 0.2), 31.0)["B2"]["verdict"] == "không đạt"
    assert not T.criteria(_cmp(0.3, 0.1), _summ(0.0, 0.0), 31.0)["B3"]["pass_pytorch"]


def test_read_history_keeps_latest_row_per_epoch(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (
        {"epoch": 1, "devval_mAP50": 0.1}, {"epoch": 2, "devval_mAP50": 0.2}, {"epoch": 2, "devval_mAP50": 0.3})) + "\n")
    assert [(h["epoch"], h["devval_mAP50"]) for h in T.read_history(p)] == [(1, 0.1), (2, 0.3)]


def test_dataset_info_rejects_real_frame(tmp_path, monkeypatch):
    d = tmp_path / "dx"
    for sp in ("train", "val"):
        (d / sp / "images").mkdir(parents=True)
    real = next(iter(_split_of()))
    (d / "manifest.jsonl").write_text(json.dumps({"name": "visdrone__a.jpg"}) + "\n" + json.dumps({"name": real}) + "\n")
    monkeypatch.setattr(T, "data_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="khung thật"):
        T.dataset_info("dx", ROOT)


# ---------------------------------------------------------------- dữ liệu ở ổ D


@pytest.mark.data
def test_train_manifests_on_disk_match_and_are_synthetic(e1, require_data):
    real = set(_split_of())
    for s in SETS:
        m = require_data(data_root() / s / "manifest.jsonl", "phase 07")
        assert hashlib.sha256(m.read_bytes()).hexdigest() == e1["sets"][s]["train_data"]["manifest_sha256"], s
        names = {json.loads(x)["name"] for x in m.read_text(encoding="utf-8").splitlines() if x.strip()}
        assert not names & real, s
        assert not any(T.REAL_FRAME_RE.match(n) for n in names), s
