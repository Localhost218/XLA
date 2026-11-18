"""Phase 08: nhãn người tập test thật + dev-val.

Ca không marker chạy trên dữ liệu tổng hợp (luật chọn dev-val, hợp hộp, định dạng YOLO, đếm sửa). Ca ``data`` đọc
``data/real/labels/`` và **chỉ xanh sau khi chủ dự án đã duyệt** trên Label Studio (``approved_by_owner: true``).
"""

from __future__ import annotations

import json
import math

import pytest

from xla.detect import devval as D
from xla.detect import labelstudio as L
from xla.detect import prelabel as P
from xla.detect import review as R
from xla.paths import repo_root

EDGE = P.EDGE_PX


# ---------------------------------------------------------------- dev-val


def _frames(n_dev: int, n_test: int, step: float = 0.4) -> list[dict]:
    out = []
    for i in range(n_dev):
        out.append({"frame": f"d{i:04d}.jpg", "t_unix": 1000.0 + i * step, "alt_m": 5.0 + (i % 3) * 5, "split": "dev"})
    for i in range(n_test):
        out.append({"frame": f"t{i:04d}.jpg", "t_unix": 9000.0 + i * 1.5, "alt_m": 5.0 + (i % 3) * 5, "split": "test"})
    return out


def test_allocate_is_exact_and_proportional():
    q = D.allocate({("a", True): 1, ("b", False): 3}, 100)
    assert sum(q.values()) == 100
    assert q == {("a", True): 25, ("b", False): 75}
    assert sum(D.allocate({("a", True): 1, ("b", False): 1, ("c", False): 1}, 100).values()) == 100


def test_select_devval_spacing_subset_and_determinism():
    frames = _frames(2000, 60)
    person = {f["frame"] for f in frames[::7]}
    a, rep = D.select_devval(frames, person, n=100, seed=1, min_gap_s=1.0)
    b, _ = D.select_devval(frames, person, n=100, seed=1, min_gap_s=1.0)
    assert a == b and len(a) == 100 and rep["shortfall"] == 0
    by = {f["frame"]: f for f in frames}
    assert all(by[x]["split"] == "dev" for x in a)
    assert D.min_gap(by, a) >= 1.0
    c, _ = D.select_devval(frames, person, n=100, seed=2, min_gap_s=1.0)
    assert c != a


def test_select_devval_reports_shortfall_when_infeasible():
    frames = _frames(20, 30, step=0.1)  # 20 khung dev trong 2 s → cách 1 s chỉ được ~2
    names, rep = D.select_devval(frames, set(), n=10, seed=1, min_gap_s=1.0)
    assert rep["shortfall"] == 10 - len(names) > 0


# ---------------------------------------------------------------- gán trước


def test_left_strip_rule_drops_only_boxes_fully_inside():
    assert P.in_left_strip([0, 10, EDGE, 40])
    assert not P.in_left_strip([0, 10, EDGE + 0.5, 40])  # chạm dải nhưng lấn ra ngoài: giữ (khác bộ lọc sơ bộ x1 < 16)


def test_fuse_marks_source_and_keeps_rule():
    a = {"x1": 100, "y1": 100, "x2": 130, "y2": 140, "conf": 0.8}
    per = {"rtdetr": [a, {"x1": 300, "y1": 300, "x2": 320, "y2": 330, "conf": 0.5}],
           "yolo": [{"x1": 101, "y1": 99, "x2": 131, "y2": 141, "conf": 0.7},
                    {"x1": 500, "y1": 50, "x2": 510, "y2": 60, "conf": 0.25}]}
    fused = P.fuse(per)
    src = sorted(b["source"] for b in fused)
    assert src == ["both", "rtdetr", "yolo"]
    kept = [b for b in fused if P.keep(b)]
    assert sorted(b["source"] for b in kept) == ["both", "rtdetr"]  # yolo-only 0,25 < 0,30 bị bỏ


def test_ls_result_roundtrip():
    boxes = [{"x1": 64.0, "y1": 48.0, "x2": 128.0, "y2": 96.0}]
    res = P.to_ls_result(boxes)
    back, hard = L.ann_boxes({"result": res + [{"type": "choices", "value": {"choices": ["hard"]}}]})
    assert hard and all(math.isclose(u, v, abs_tol=1e-6) for u, v in zip(back[0], [64, 48, 128, 96], strict=True))


# ---------------------------------------------------------------- soát + xuất


def test_apply_frame_actions():
    boxes = [{"id": 1, "x1": 10, "y1": 10, "x2": 30, "y2": 30, "source": "both"},
             {"id": 2, "x1": 100, "y1": 100, "x2": 140, "y2": 150, "source": "rtdetr"}]
    e = [{"frame": "f", "action": "drop", "box_id": "1", "x1": "", "y1": "", "x2": "", "y2": "", "reason": "đốm"},
         {"frame": "f", "action": "tighten", "box_id": "2", "x1": "102", "y1": "101", "x2": "138", "y2": "148",
          "reason": "siết"},
         {"frame": "f", "action": "add", "box_id": "", "x1": "200", "y1": "200", "x2": "215", "y2": "220",
          "reason": "sót"},
         {"frame": "f", "action": "add", "box_id": "", "x1": "0", "y1": "0", "x2": "12", "y2": "20", "reason": "mép"},
         {"frame": "f", "action": "hard", "box_id": "", "x1": "", "y1": "", "x2": "", "y2": "", "reason": "nhóm"}]
    final, hard = R.apply_frame(boxes, e)
    assert hard
    assert final == [[102.0, 101.0, 138.0, 148.0], [200.0, 200.0, 215.0, 220.0]]  # hộp trong dải mép bị bỏ
    with pytest.raises(ValueError):
        R.apply_frame(boxes, [e[0], e[0]])


def test_yolo_lines_format():
    s = R.yolo_lines([[0, 0, 640, 480], [320, 240, 352, 288]])
    assert s.splitlines() == ["0 0.500000 0.500000 1.000000 1.000000", "0 0.525000 0.550000 0.050000 0.100000"]
    assert R.yolo_lines([]) == ""


def test_diff_boxes_counts_owner_edits():
    agent = [[0, 0, 10, 10], [100, 100, 120, 130], [300, 300, 310, 320]]
    owner = [[0, 0, 10, 10], [101, 100, 125, 133], [400, 400, 410, 410]]
    assert L.diff_boxes(agent, owner) == {"deleted": 1, "added": 1, "modified": 1}


def test_submitted_ignores_cancelled():
    t = {"annotations": [{"id": 1, "was_cancelled": True, "result": []}]}
    assert L.submitted(t) is None
    t["annotations"].append({"id": 2, "was_cancelled": False, "updated_at": "2026-11-18T10:00:00", "result": []})
    assert L.submitted(t)["id"] == 2


# ---------------------------------------------------------------- dữ liệu thật (sau khi chủ dự án duyệt)


def _yolo(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        assert len(parts) == 5 and parts[0] == "0", f"{path.name}: dòng sai {line!r}"
        assert all(len(p.split(".")[1]) == 6 for p in parts[1:]), f"{path.name}: cần 6 chữ số thập phân {line!r}"
        rows.append([float(p) for p in parts[1:]])
    return rows


@pytest.fixture
def real(require_data):
    root = repo_root()
    m = json.loads(require_data(root / D.MANIFEST_REL, "phase 08").read_text(encoding="utf-8"))
    split = {f["frame"]: f for f in json.loads(require_data(root / D.SPLIT_REL).read_text(encoding="utf-8"))["frames"]}
    return root, m, split


@pytest.mark.data
def test_manifest_shape_and_owner_approval(real):
    _, m, _ = real
    keys = {"version", "approved_by_owner", "approved_date", "test_frames", "devval_frames", "hard_frames", "n_boxes",
            "prelabel_models", "owner_edits", "devval_seed", "devval_min_gap_s"}
    assert keys <= set(m)
    assert m["version"] == D.MANIFEST_VERSION
    assert {p["name"] for p in m["prelabel_models"]} == {"rtdetr-x", "yolo11x"}
    assert all(len(p["sha256"]) == 64 for p in m["prelabel_models"])
    assert m["approved_by_owner"] is True, "chủ dự án chưa duyệt nhãn test trên Label Studio (bước 5)"
    assert m["approved_date"] and isinstance(m["owner_edits"], int)


@pytest.mark.data
def test_devval_rules(real):
    _, m, split = real
    test = {f for f, v in split.items() if v["split"] == "test"}
    dv = m["devval_frames"]
    assert len(dv) == D.DEVVAL_N and len(set(dv)) == len(dv)
    assert set(dv).isdisjoint(test)
    assert all(split[f]["split"] == "dev" for f in dv)
    assert D.min_gap(split, dv) >= m["devval_min_gap_s"] >= 1.0
    assert sorted(m["test_frames"]) == sorted(test)
    assert set(m["hard_frames"]) <= set(dv) | test


@pytest.mark.data
@pytest.mark.parametrize("sub", ["test", "devval"])
def test_every_frame_has_valid_label_file(real, sub):
    root, m, _ = real
    names = m["test_frames"] if sub == "test" else m["devval_frames"]
    d = root / D.LABELS_REL / sub
    n_boxes = 0
    for f in names:
        p = d / (f[:-4] + ".txt")
        assert p.exists(), f"thiếu nhãn {p} (khung không người = file rỗng)"
        for cx, cy, w, h in _yolo(p):
            x1, x2 = (cx - w / 2) * P.W, (cx + w / 2) * P.W
            y1, y2 = (cy - h / 2) * P.H, (cy + h / 2) * P.H
            assert 0 < cx <= 1 and 0 < cy <= 1 and 0 < w <= 1 and 0 < h <= 1, f"{p.name}: toạ độ ngoài (0,1]"
            assert x1 >= -1e-3 and y1 >= -1e-3 and x2 <= P.W + 1e-3 and y2 <= P.H + 1e-3, f"{p.name}: hộp tràn khung"
            assert not P.in_left_strip([x1, y1, x2, y2]), f"{p.name}: hộp nằm trọn trong dải mép trái"
            n_boxes += 1
    extra = {x.stem for x in d.glob("*.txt")} - {f[:-4] for f in names}
    assert not extra, f"file nhãn lạ trong {d}: {sorted(extra)[:5]}"
    assert m["n_boxes"][sub] == n_boxes


@pytest.mark.data
def test_review_log_covers_every_frame(real):
    root, m, _ = real
    import csv

    with (root / R.LOG_REL).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen = {r["frame"] for r in rows}
    assert set(m["test_frames"]) | set(m["devval_frames"]) <= seen
    assert {r["action"] for r in rows} <= set(R.ACTIONS)
