"""Phase 11 (E2): luật nhãn giả của ``xla.detect.selftrain`` + kết quả ``results/e2/`` đã commit.

Ca không marker: luật track (giữ / cứu / bỏ lẻ / bỏ mép / khung mơ hồ), chọn τ_cao theo precision dev-val, chuỗi khung
chỉ lấy split dev. Ca ``data``: ``results/e2/pseudo_r*.json`` và ``e2.json`` (phase 11 tạo) không dùng khung test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xla.camera.frames import CROP_LEFT_PX
from xla.detect import selftrain as S
from xla.paths import repo_root

PCFG = {"min_track_len": 3, "tau_low": 0.05, "match_iou": 0.5, "drop_frames_with_unkept_high": True,
        "tau_high_grid": [0.1, 0.3, 0.5, 0.7], "precision_min": 0.9}


def _raw(n: int, tracks: list[list[tuple[float, ...]]], dets: list[list[tuple[float, ...]]] | None = None,
         seq: int = 0) -> dict[str, dict]:
    """``tracks[i]`` = các (x1,y1,x2,y2,tid,conf) ở khung i; ``dets`` mặc định = hộp của tracks."""
    out = {}
    for i in range(n):
        tr = [list(t) for t in tracks[i]] if i < len(tracks) else []
        dt = [list(d) for d in dets[i]] if dets is not None else [t[:4] + [t[5]] for t in tr]
        out[f"s{seq}_{i:03d}.jpg"] = {"seq": seq, "dets": dt, "tracks": tr}
    return out


BOX = (100.0, 100.0, 140.0, 180.0)


def test_long_high_track_kept_and_low_box_rescued() -> None:
    confs = [0.6, 0.08, 0.7, 0.2]
    raw = _raw(4, [[(*BOX, 1, c)] for c in confs])
    lab = S.apply_rules(raw, 0.5, PCFG)
    assert [len(lab[f]["boxes"]) for f in sorted(lab)] == [1, 1, 1, 1]  # 0,08 ≥ τ_thấp được cứu
    assert all(v["use"] for v in lab.values())


def test_short_track_and_low_only_track_dropped() -> None:
    raw = _raw(4, [[(*BOX, 1, 0.9), (300, 100, 340, 180, 2, 0.2)], [(*BOX, 1, 0.9), (300, 100, 340, 180, 2, 0.2)],
                   [(300, 100, 340, 180, 2, 0.2)], [(300, 100, 340, 180, 2, 0.2)]])
    lab = S.apply_rules(raw, 0.5, PCFG)
    # track 1 dài 2 < L: bỏ, và khung 0–1 có hộp ≥ τ_cao bị bỏ → khung mơ hồ không dùng
    assert not lab["s0_000.jpg"]["use"] and not lab["s0_001.jpg"]["use"]
    # track 2 dài 4 nhưng không có hộp ≥ τ_cao: không nhãn, khung 2–3 thành khung rỗng dùng được
    assert lab["s0_002.jpg"]["boxes"] == [] and lab["s0_002.jpg"]["use"]


def test_isolated_high_detection_without_track_marks_frame_ambiguous() -> None:
    raw = _raw(3, [[], [], []], dets=[[(*BOX, 0.95)], [], []])
    lab = S.apply_rules(raw, 0.5, PCFG)
    assert not lab["s0_000.jpg"]["use"] and lab["s0_001.jpg"]["use"]


def test_edge_boxes_dropped() -> None:
    edge = (CROP_LEFT_PX - 5.0, 100.0, 60.0, 180.0)
    raw = _raw(3, [[(*edge, 1, 0.9)]] * 3)
    lab = S.apply_rules(raw, 0.5, PCFG)
    assert all(v["boxes"] == [] for v in lab.values())


def test_short_sequence_not_used() -> None:
    raw = _raw(2, [[(*BOX, 1, 0.9)]] * 2)
    assert not any(v["use"] for v in S.apply_rules(raw, 0.5, PCFG).values())


def test_select_tau_high_reaches_precision() -> None:
    # khung 0..4: một người thật ở BOX (track 1, điểm 0,8); một hộp sai (track 2, điểm 0,4)
    tracks = [[(*BOX, 1, 0.8), (400, 300, 440, 380, 2, 0.4)] for _ in range(5)]
    raw = _raw(5, tracks)
    gt = {f: np.asarray([BOX]) for f in raw}
    tau, grid, _why = S.select_tau_high(raw, gt, PCFG)
    assert tau == 0.5, grid
    q = next(g for g in grid if g["tau_high"] == tau)
    assert q["precision"] == 1.0 and q["recall"] == 1.0
    assert next(g for g in grid if g["tau_high"] == 0.3)["precision"] == pytest.approx(0.5)


def test_dev_sequences_only_dev_split() -> None:
    rows = [{"frame": f"d{i}", "t_unix": i * 0.1, "split": "dev"} for i in range(5)]
    rows += [{"frame": f"t{i}", "t_unix": 0.05 + i * 0.1, "split": "test"} for i in range(5)]
    seqs = S.dev_sequences(rows, 1.0)
    assert seqs == [[f"d{i}" for i in range(5)]]


def test_bundle_metrics_keys() -> None:
    from xla import contract as C

    m = {k: {"value": 0.1, "ci95": [0, 1]} for k in ("mAP50", "mAP50_95", "precision@op", "recall@op",
                                                     "fp_per_neg_frame@op")}
    assert tuple(S.bundle_metrics({"metrics": m})) == C.METRICS_KEYS["detector"] == S.BUNDLE_KEYS


# ---------------------------------------------------------------- kết quả đã commit (phase 11)

E2 = repo_root() / S.RESULTS_REL


def _split_of() -> dict[str, str]:
    rows = json.loads((repo_root() / S.SPLIT_REL).read_text(encoding="utf-8"))["frames"]
    return {r["frame"]: r["split"] for r in rows}


@pytest.mark.data
def test_pseudo_datasets_have_no_test_or_devval(require_data) -> None:
    reps = sorted(E2.glob("pseudo_r*.json"))
    if not reps:
        require_data(E2 / "pseudo_r1.json", "phase 11: python -m xla.detect.selftrain --round 1")
    split = _split_of()
    dv = set(json.loads((repo_root() / S.LABELS_REL / "manifest.json").read_text(encoding="utf-8"))["devval_frames"])
    for p in reps:
        rep = json.loads(p.read_text(encoding="utf-8"))
        assert rep["dataset"]["test_frames_in_train"] == 0 and rep["dataset"]["devval_frames_in_train"] == 0
        txt = Path(rep["dataset"]["yaml"]).with_name("train.txt")
        if txt.is_file():  # tập ở ổ D (ngoài repo)
            names = {Path(line).name for line in txt.read_text(encoding="utf-8").split()}
            real = {n for n in names if n in split}
            assert not {n for n in real if split[n] == "test"}, p.name
            assert not real & dv, p.name
            assert {split[n] for n in real} <= {"dev", "neg"}, p.name


@pytest.mark.data
def test_e2_ship_selected_on_devval(require_data) -> None:
    e2 = json.loads(require_data(E2 / "e2.json", "phase 11: --score").read_text(encoding="utf-8"))
    cands = e2["ship"]["candidates"]
    assert "teacher" not in cands and "e1_d2" in cands
    assert e2["ship"]["name"] == max(cands, key=lambda n: cands[n]["devval_map50"])
    assert e2["bundle"]["is_bundle_metrics"] and e2["bundle"]["model"] == e2["ship"]["name"]
