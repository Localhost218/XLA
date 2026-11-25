"""AI v2 phase 04: tập nhãn ``devpool`` + ``folds.json`` theo phiên.

Ca không marker chạy trên dữ liệu giả (``tests/labels/fixtures/folds_case.json`` + khung tổng hợp trong test). Ca
``data`` đọc ``data/real/`` và **fail** khi thiếu, không skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xla.camera import sessions as S
from xla.detect import review as R
from xla.labels import devpool as D
from xla.paths import repo_root

FIX = Path(__file__).parent / "fixtures" / "folds_case.json"


@pytest.fixture(scope="module")
def fcase():
    return json.loads(FIX.read_text(encoding="utf-8"))["frames"]


def _pool(n=120, step=0.4, sessions=("s#0", "s#1")):
    out = []
    for i in range(n):
        s = sessions[i * len(sessions) // n]
        out.append({"frame": f"r_001_{i:05d}.jpg", "session": s, "t": 1000.0 + i * step, "alt_m": 5.0 + (i % 3) * 5,
                    "block": i // 75})
    return out


# ---------------------------------------------------------------- chọn khung


def test_select_forces_devval_spacing_and_candidates_first():
    pool = _pool()
    cand = {f["frame"] for f in pool[::4]}
    devval = [pool[0]["frame"], pool[10]["frame"], "ngoai_pool.jpg"]
    names, rep = D.select(pool, devval, cand)
    t = {f["frame"]: f["t"] for f in pool}
    assert set(devval[:2]) <= set(names) and "ngoai_pool.jpg" not in names
    assert D.min_gap([t[f] for f in names]) >= D.MIN_GAP_S - 1e-9
    assert len(names) == D.capacity([f["t"] for f in pool], 1.0) or len(names) <= D.TARGET_MAX
    assert rep["n_devval"] == 2 and rep["met"] is False  # sức chứa < 500: báo thiếu, không nới khoảng cách
    assert rep["min_gap_s"] == D.MIN_GAP_S and [r["min_gap_s"] for r in rep["capacity_table"]] == list(D.CAPACITY_GAPS)
    # khung có ứng viên được ưu tiên: khung ứng viên nào không được lấy đều vướng (< 1 s) một khung đã chọn
    for f in cand - set(names):
        assert any(abs(t[f] - t[g]) < D.MIN_GAP_S for g in names), f
    assert rep["n_candidate_person"] == len(cand & set(names))
    assert D.select(pool, devval, cand)[0] == names  # tất định


def test_select_rejects_devval_closer_than_gap():
    pool = _pool(10, step=0.2)
    with pytest.raises(ValueError, match="dev-val"):
        D.select(pool, [pool[0]["frame"], pool[1]["frame"]], set())


def test_pool_excludes_test_and_reserved_cal_blocks():
    rows, v1 = [], {}
    for i in range(240):  # khối 0 test, khối 1–3 dev; khối 2 là cal
        f = f"r_001_{i:05d}.jpg"
        b = i // 60
        rows.append({"frame": f, "rec": "r", "session": "r#0", "t_unix": 100.0 + i * 0.5, "alt_m": "10.0"})
        v1[f] = {"frame": f, "split": "test" if b == 0 else "dev", "block": b}
    cal = [f for f, v in v1.items() if v["block"] == 2][:40]  # khung có người của khối cal không nằm trong neg
    neg = {"seed": S.SEED, "cal_mode": "block", "sessions": {"r#0": "mine"},
           "frames": [{"frame": f, "session": "r#0", "split": "cal", "approved": False} for f in cal]}
    pool = {f["frame"] for f in D.pool_frames(rows, v1, neg)}
    assert pool == {f for f, v in v1.items() if v["block"] in (1, 3)}


# ---------------------------------------------------------------- fold


def test_session_folds_balance_and_min_person(fcase):
    person = {f["frame"] for f in fcase if f["person"]}
    folds, info = D.build_folds(fcase, person)
    assert set(folds) == {"groups", "folds"} and folds["groups"] == "session" and info["k"] == 3
    assert info["tried"][0]["k"] == 4 and min(info["tried"][0]["person_per_fold"]) < D.FOLD_PERSON_MIN
    assert min(info["person_frames_per_fold"]) >= D.FOLD_PERSON_MIN
    flat = [x for fold in folds["folds"] for x in fold]
    assert sorted(flat) == sorted(f["frame"] for f in fcase) and len(flat) == len(set(flat))
    sess = {f["frame"]: f["session"] for f in fcase}
    for fold in folds["folds"]:  # một phiên không bị chẻ qua hai fold
        for other in folds["folds"]:
            if other is not fold:
                assert not {sess[x] for x in fold} & {sess[x] for x in other}
    assert D.build_folds(fcase, person)[0] == folds  # chạy lại ra đúng folds.json


def test_block5_fallback_when_few_sessions(fcase):
    two = [f for f in fcase if f["session"] in ("20990101-100000#0", "20990101-100000#1")]
    person = {f["frame"] for f in two if f["person"]}
    folds, _ = D.build_folds(two, person)
    assert folds["groups"] == "block5" and len(folds["folds"]) == D.BLOCK5_FOLDS
    blocks = [{next(f["block"] for f in two if f["frame"] == x) for x in fold} for fold in folds["folds"]]
    assert all(len(b) == 1 for b in blocks)  # 5 khối → 5 fold một khối, liền nhau theo giờ
    assert [min(b) for b in blocks] == sorted(min(b) for b in blocks)


def test_contiguous_folds_maximizes_min():
    items = [("a", 5), ("b", 30), ("c", 5), ("d", 5), ("e", 25), ("f", 20)]
    parts = D.contiguous_folds(items, 3)
    loads = [sum(dict(items)[k] for k in p) for p in parts]
    assert [k for p in parts for k in p] == [k for k, _ in items] and min(loads) == 20  # [a b] [c d e] [f]
    with pytest.raises(ValueError):
        D.contiguous_folds(items[:2], 3)


# ---------------------------------------------------------------- nhãn


def test_yolo_roundtrip_and_validation():
    boxes = [[16.0, 10.0, 48.0, 60.0], [600.0, 400.0, 640.0, 480.0]]
    text = R.yolo_lines(boxes)
    assert D.valid_yolo(text) and D.valid_yolo("")
    assert all(abs(a - b) < 1e-3 for x, y in zip(D.yolo_boxes(text), boxes, strict=True) for a, b in zip(x, y,
                                                                                                         strict=True))
    assert not D.valid_yolo("1 0.5 0.5 0.1 0.1\n") and not D.valid_yolo("0 0.99 0.5 0.1 0.1\n")
    assert not D.valid_yolo("0 0.5 0.5 0 0.1\n")


def test_final_labels_apply_edits_and_report_missing():
    pre = {"a.jpg": {"boxes": [{"id": 1, "x1": 100, "y1": 100, "x2": 120, "y2": 130, "source": "both"},
                               {"id": 2, "x1": 300, "y1": 50, "x2": 310, "y2": 60, "source": "rtdetr"}]},
           "b.jpg": {"boxes": []}}
    edits = {"a.jpg": [{"frame": "a.jpg", "action": "drop", "box_id": "2", "x1": "", "y1": "", "x2": "", "y2": "",
                        "reason": "đốm"},
                       {"frame": "a.jpg", "action": "add", "box_id": "", "x1": "200", "y1": "200", "x2": "230",
                        "y2": "240", "reason": "người sót"}]}
    final, hard, missing = D.final_labels(pre, edits, ["a.jpg", "b.jpg"])
    assert final == {"a.jpg": [[100, 100, 120, 130], [200.0, 200.0, 230.0, 240.0]]}
    assert hard == [] and missing == ["b.jpg"]


# ---------------------------------------------------------------- dữ liệu thật


@pytest.fixture(scope="module")
def real():
    root = repo_root()
    for rel in (D.DEVPOOL_JSON_REL, D.FOLDS_REL, S.NEG_SPLIT_REL, D.FOLD_INFO_REL):
        if not (root / rel).exists():
            pytest.fail(f"thiếu dữ liệu local {rel}", pytrace=False)
    rows = S.flight_rows(root)
    return {"root": root, "doc": json.loads((root / D.DEVPOOL_JSON_REL).read_text(encoding="utf-8")),
            "folds": json.loads((root / D.FOLDS_REL).read_text(encoding="utf-8")), "rows": rows,
            "v1": S.load_split_v1(root), "neg": S.load_neg_split(root),
            "devval": json.loads((root / S.DEVVAL_MANIFEST_REL).read_text(encoding="utf-8"))["devval_frames"]}


@pytest.mark.data
def test_real_devpool_disjoint_from_test_cal_holdout(real):
    names = set(real["doc"]["frames"])
    assert not {f for f in names if real["v1"].get(f, {}).get("split") == "test"}
    assert not names & {x["frame"] for x in real["neg"]["frames"] if x["split"] in ("cal", "holdout")}
    assert not names & S.reserved_frames(real["neg"], real["rows"], real["v1"])
    assert not names & {x["frame"] for x in real["neg"]["frames"] if x["split"] == "mine"}


@pytest.mark.data
def test_real_devpool_labels_valid_and_devval_unchanged(real):
    root, names = real["root"], real["doc"]["frames"]
    on_disk = {p.name for p in (root / D.DEVPOOL_REL).glob("*.txt")}
    assert on_disk == {f"{Path(f).stem}.txt" for f in names}
    for f in names:
        assert D.valid_yolo((root / D.DEVPOOL_REL / f"{Path(f).stem}.txt").read_text(encoding="utf-8")), f
    kept = [f for f in real["devval"] if f in set(names)]
    for f in kept:  # dev-val cũ: nhãn không đổi (byte)
        a = (root / D.DEVPOOL_REL / f"{Path(f).stem}.txt").read_bytes()
        assert a == (root / D.DEVVAL_DIR_REL / f"{Path(f).stem}.txt").read_bytes(), f
    # dev-val nằm ngoài devpool đúng bằng dev-val trong khối cal (Lệch so với plan, README)
    cal_blocks = {real["v1"][x["frame"]]["block"] for x in real["neg"]["frames"] if x["split"] == "cal"}
    out = sorted(set(real["devval"]) - set(names))
    assert out == sorted(f for f in real["devval"] if real["v1"][f]["block"] in cal_blocks)
    assert real["doc"]["report"]["devval_not_in_devpool"] == out


@pytest.mark.data
def test_real_folds_contract_and_reproducible(real):
    folds = real["folds"]
    assert set(folds) == {"groups", "folds"} and folds["groups"] in ("session", "block5")
    flat = [x for fold in folds["folds"] for x in fold]
    assert sorted(flat) == sorted(real["doc"]["frames"]) and len(flat) == len(set(flat))
    frames, person = D.devpool_frames_meta(real["root"])
    assert min(sum(x in person for x in fold) for fold in folds["folds"]) >= D.FOLD_PERSON_MIN
    again, _ = D.build_folds(frames, person)
    assert again == folds
