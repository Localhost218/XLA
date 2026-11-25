"""AI v2 phase 03: phiên bay, chia ``neg_flight`` mine / cal / holdout theo phiên, trang duyệt holdout.

Ca không marker chạy trên dữ liệu giả ``tests/camera/fixtures/neg_split_case.json`` (CI CPU). Ca ``data`` đọc
``data/real/`` + manifest huấn luyện local (``XLA_DATA_ROOT``) và **fail** khi thiếu, không skip.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

from xla.camera import frames as F
from xla.camera import sessions as S
from xla.paths import data_root, repo_root

FIX = Path(__file__).parent / "fixtures" / "neg_split_case.json"
REC_A = "20990101-100000"
REC_B = "20990102-090000"


@pytest.fixture(scope="module")
def case():
    return json.loads(FIX.read_text(encoding="utf-8"))


def _plan(case, only_locked=False, devpool=frozenset()):
    rows = [r for r in case["rows"] if not only_locked or r["rec"] == REC_A]
    return S.plan_neg(rows, case["split_v1"], set(case["devval"]), set(devpool)), rows


def _split(case, only_locked=False, devpool=frozenset()):
    plan, rows = _plan(case, only_locked, devpool)
    return S.build_neg_split(plan, set(case["person"]), set(case["devval"]), case["split_v1"]), rows


# ---------------------------------------------------------------- phiên bay


def test_flight_sessions_id_and_split_across_recordings():
    fl = [F.FlightWindow("20990101-100100", 100.0, 200.0), F.FlightWindow("20990101-100500", 300.0, 400.0)]
    ranges = {"recA": (0.0, 150.0), "recB": (150.0, 500.0), "recC": (600.0, 700.0)}
    got = F.flight_sessions(fl, ranges)
    assert [(sid, rec, w.t0, w.t1) for sid, rec, w in got] == [
        ("recA#0", "recA", 100.0, 150.0), ("recB#0", "recB", 150.0, 200.0), ("recB#1", "recB", 300.0, 400.0)]


def test_sessions_doc_shape_matches_contract():
    wins = [("r#0", "r", F.FlightWindow("x", 1.0, 2.0)), ("r#1", "r", F.FlightWindow("y", 3.0, 4.0))]
    doc = S.build_sessions_doc(wins, [{"session": "r#0"}, {"session": "r#0"}])
    assert doc == {"sessions": [
        {"id": "r#0", "rec": "r", "t_start": 1.0, "t_end": 2.0, "n_frames": 2, "source": "flights-events"},
        {"id": "r#1", "rec": "r", "t_start": 3.0, "t_end": 4.0, "n_frames": 0, "source": "flights-events"}]}


def test_iotcv_child_env_drops_xla_venv_vars():
    env = S.iotcv_child_env({"UV_PROJECT_ENVIRONMENT": "D:/v", "VIRTUAL_ENV": "D:/v", "PATH": "p"})
    assert "UV_PROJECT_ENVIRONMENT" not in env and "VIRTUAL_ENV" not in env
    assert env["PATH"] == "p" and env["PYTHONUTF8"] == "1"


# ---------------------------------------------------------------- chia theo phiên


def test_session_mode_holdout_and_cal_only_from_free_sessions(case):
    (doc, _), rows = _split(case)
    assert doc["cal_mode"] == "session" and doc["seed"] == S.SEED
    assert S.check_split(doc, rows, case["split_v1"]) == []
    locked = {f"{REC_A}#0", f"{REC_A}#1"}
    assert {s for s, r in doc["sessions"].items() if r == "mine"} >= locked
    hold = {s for s, r in doc["sessions"].items() if r == "holdout"}
    cal = {s for s, r in doc["sessions"].items() if r == "cal"}
    assert len(hold) >= S.HOLDOUT_MIN_SESSIONS and len(cal) == S.CAL_MIN_SESSIONS and not (hold | cal) & locked
    v1 = case["split_v1"]
    for x in doc["frames"]:
        if x["split"] in ("cal", "holdout"):
            assert x["frame"] not in v1, x  # (i) phiên cal/holdout không có khung dev/test v1
        assert x["approved"] is False
        assert v1.get(x["frame"], {}).get("split") != "test"


def test_block_mode_when_no_free_session(case):
    (doc, rep), rows = _split(case, only_locked=True)
    v1 = case["split_v1"]
    assert doc["cal_mode"] == "block" and set(doc["sessions"].values()) == {"mine"}
    assert S.check_split(doc, rows, v1) == []
    cal_blocks = {v1[x["frame"]]["block"] for x in doc["frames"] if x["split"] == "cal"}
    assert cal_blocks == set(rep["cal_blocks"]) and len(cal_blocks) == 2
    # khối nguyên: không khung mine nào trong khối cal; khung cal đều là dev v1, không có test trong khối
    for x in doc["frames"]:
        b = v1[x["frame"]]["block"]
        assert (x["split"] == "cal") == (b in cal_blocks)
        assert v1[x["frame"]]["split"] == "dev"
    for b in cal_blocks:
        assert {v["split"] for v in v1.values() if v["block"] == b} == {"dev"}
    # mỗi phiên còn ít nhất một khối mine
    mine_sessions = {x["session"] for x in doc["frames"] if x["split"] == "mine"}
    assert mine_sessions == {f"{REC_A}#0", f"{REC_A}#1"}
    # reserved_frames giữ cả khung có người của khối cal khỏi devpool
    res = S.reserved_frames(doc, rows, v1)
    assert {r["frame"] for r in rows if v1[r["frame"]]["block"] in cal_blocks} <= res


def test_person_and_devpool_frames_never_in_neg(case):
    devpool = {x["frame"] for x in _split(case)[0][0]["frames"] if x["split"] == "mine"}
    pick = set(sorted(devpool)[:15])
    (doc, rep), _ = _split(case, devpool=pick)
    names = {x["frame"] for x in doc["frames"]}
    assert not names & set(case["person"])
    assert not {x["frame"] for x in doc["frames"] if x["split"] == "mine"} & pick
    assert rep["skipped"]["devpool"] == len(pick)


def test_split_is_deterministic(case):
    a, b = _split(case)[0][0], _split(case)[0][0]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_check_split_flags_leaks(case):
    (doc, _), rows = _split(case)
    v1 = case["split_v1"]
    bad = copy.deepcopy(doc)
    test_frame = next(f for f, v in v1.items() if v["split"] == "test")
    bad["frames"].append({"frame": test_frame, "session": f"{REC_A}#0", "split": "holdout", "approved": False})
    bad["sessions"][f"{REC_A}#0"] = "holdout"
    errs = S.check_split(bad, rows, v1)
    assert any("khung test v1" in e for e in errs) and any("phiên holdout" in e for e in errs)
    bad2 = copy.deepcopy(doc)
    bad2["frames"][0] = bad2["frames"][0] | {"split": "train"}
    assert any("sai dạng" in e for e in S.check_split(bad2, rows, v1))


def test_assign_sessions_prefers_two_holdout_sessions_then_cal():
    rng = np.random.default_rng(1)
    roles, mode = S.assign_sessions({"a": 10, "b": 400, "c": 50, "d": 5}, {"a"}, rng)
    assert roles["a"] == "mine" and mode == "session"
    assert sum(r == "holdout" for r in roles.values()) >= 2 and sum(r == "cal" for r in roles.values()) == 1
    roles2, mode2 = S.assign_sessions({"a": 10, "b": 400}, {"a"}, np.random.default_rng(1))
    assert roles2 == {"a": "mine", "b": "holdout"} and mode2 == "block"


# ---------------------------------------------------------------- soát từng khung


def test_plan_does_not_depend_on_review_and_person_only_removes(case):
    plan, _ = _plan(case)
    planned = {x["frame"] for x in plan["planned"]}
    doc, rep = S.build_neg_split(plan, set(case["person"]), set(case["devval"]), case["split_v1"])
    assert {x["frame"] for x in doc["frames"]} == planned - set(case["person"])
    assert sum(rep["removed_person"].values()) == len(planned & set(case["person"]))
    doc2, _ = S.build_neg_split(plan, set(), set(case["devval"]), case["split_v1"])
    assert {x["frame"] for x in doc2["frames"]} == planned


def test_mine_spacing_in_locked_sessions():
    rows, v1 = [], {}
    for i in range(20):  # 5 khung/s, một khối dev
        f = f"r_001_{i:05d}.jpg"
        rows.append({"frame": f, "rec": "r", "session": "r#0", "t_unix": 100.0 + i * 0.2})
        v1[f] = {"frame": f, "split": "dev", "block": 0}
    plan = S.plan_neg(rows, v1, set(), set())
    ts = [100.0 + int(x["frame"][6:11]) * 0.2 for x in plan["planned"]]
    assert len(ts) == 7 and min(b - a for a, b in itertools.pairwise(ts)) >= S.MINE_MIN_GAP_S - 1e-9


def test_read_agent_review_rejects_bad_rows(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("frame,verdict,reason\na.jpg,person,người\nb.jpg,no_person,\n", encoding="utf-8")
    assert {k: v["verdict"] for k, v in S.read_agent_review(p).items()} == {"a.jpg": "person", "b.jpg": "no_person"}
    p.write_text("frame,verdict,reason\na.jpg,maybe,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verdict lạ"):
        S.read_agent_review(p)
    p.write_text("frame,verdict,reason\na.jpg,person,\na.jpg,no_person,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hai dòng"):
        S.read_agent_review(p)


# ---------------------------------------------------------------- trang duyệt + apply-review


def _with_holdout(case):
    return _split(case)[0][0]


def test_review_page_is_standalone_html(case, tmp_path):
    doc = _with_holdout(case)
    out = tmp_path / "results" / "v2" / "neg-holdout" / "review" / "index.html"
    frames_dir = tmp_path / "data" / "real" / "frames"
    info = S.build_review_page(doc, "ab" * 32, out, frames_dir)
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "<head>" in html and "<body>" in html
    assert "Xuất JSON" in html and "localStorage" in html and "repeat(4, 1fr)" in html and "có người" in html
    data = json.loads(html.split('<script id="data" type="application/json">')[1].split("</script>")[0])
    n_hold = sum(x["split"] == "holdout" for x in doc["frames"])
    assert len(data["frames"]) == n_hold == info["frames"] and info["grids"] == -(-n_hold // 16)
    assert data["img_prefix"] == "../../../../data/real/frames/" and data["storage_key"] == S.REVIEW_STORAGE_KEY


def _export(doc, sha, person=(), seen=None):
    hold = [x["frame"] for x in doc["frames"] if x["split"] == "holdout"]
    grids = -(-len(hold) // 16)
    return {"version": S.REVIEW_EXPORT_VERSION, "split_sha256": sha, "exported_at": "2026-11-25T00:00:00Z",
            "grids": grids, "grids_seen": list(range(grids)) if seen is None else seen,
            "frames": [{"frame": f, "person": f in set(person)} for f in hold]}


def test_apply_review_sets_approved_and_drops_marked(case):
    doc = _with_holdout(case)
    hold = [x["frame"] for x in doc["frames"] if x["split"] == "holdout"]
    marked = hold[:3]
    new, rep = S.apply_review(doc, _export(doc, "s", marked), "s")
    new_hold = [x for x in new["frames"] if x["split"] == "holdout"]
    assert len(new_hold) == len(hold) - 3 and all(x["approved"] for x in new_hold)
    assert not {x["frame"] for x in new["frames"]} & set(marked)
    assert [x for x in new["frames"] if x["split"] != "holdout"] == [x for x in doc["frames"] if x["split"] != "holdout"]
    assert rep["removed_person"] == sorted(marked) and rep["approved"] == len(hold) - 3
    assert rep["kill_a"]["verdict"] == ("đủ" if len(hold) - 3 >= 150 else "không đủ chứng minh")
    with pytest.raises(ValueError, match="đã áp"):
        S.apply_review(new, _export(new, "s"), "s")


@pytest.mark.parametrize("mutate, msg", [
    (lambda e: e.update(version="x"), "version"),
    (lambda e: e.update(split_sha256="other"), "dựng từ"),
    (lambda e: e["frames"].pop(), "thiếu 1"),
    (lambda e: e["frames"].append({"frame": "zzz.jpg", "person": False}), "thừa 1"),
    (lambda e: e.update(grids_seen=[]), "chưa đánh dấu"),
    (lambda e: e["frames"][0].update(person="yes"), "sai dạng"),
])
def test_apply_review_rejects_bad_export(case, mutate, msg):
    doc = _with_holdout(case)
    e = _export(doc, "s")
    mutate(e)
    with pytest.raises(ValueError, match=msg):
        S.apply_review(doc, e, "s")


def test_apply_review_allow_unseen(case):
    doc = _with_holdout(case)
    _, rep = S.apply_review(doc, _export(doc, "s", seen=[]), "s", allow_unseen=True)
    assert rep["unseen_grids"]


def test_apply_review_cli_exit_codes(tmp_path):
    assert S.main(["apply-review", str(tmp_path / "none.json"), "--json"]) == 2
    with pytest.raises(SystemExit) as e:
        S.main(["apply-review"])
    assert e.value.code == 2


# ---------------------------------------------------------------- dữ liệu thật


@pytest.fixture(scope="module")
def real(request):
    root = repo_root()
    for rel in (S.SESSIONS_REL, S.NEG_SPLIT_REL, S.SPLIT_V1_REL):
        if not (root / rel).exists():
            pytest.fail(f"thiếu dữ liệu local {rel}", pytrace=False)
    return {"rows": S.flight_rows(root), "v1": S.load_split_v1(root), "neg": S.load_neg_split(root),
            "sessions": S.load_sessions(root)}


@pytest.mark.data
def test_real_split_v1_unchanged():
    # (iii) sha256 của split.json đúng hợp đồng §14.2 (load_split_v1 ném lỗi nếu lệch)
    assert S.sha256_file(repo_root() / S.SPLIT_V1_REL) == S.SPLIT_V1_SHA256


@pytest.mark.data
def test_real_sessions_and_neg_split_contract(real):
    assert all(set(s) == {"id", "rec", "t_start", "t_end", "n_frames", "source"} for s in real["sessions"])
    assert all(s["id"].startswith(s["rec"] + "#") for s in real["sessions"])
    assert sum(s["n_frames"] for s in real["sessions"]) == len(real["rows"])
    assert S.check_split(real["neg"], real["rows"], real["v1"]) == []  # (i)
    assert set(real["neg"]["sessions"]) == {s["id"] for s in real["sessions"]}


V1_TRAIN_MANIFEST_EXEMPT = {
    # Dataset self-training v1 (E2 r1/r2, 18/11/2026) đã dùng 578/573 khung dev v1 của MỌI khối dev; detector-v1.0.0 =
    # E2 r1. Với cal_mode "block", khối cal là khối dev v1 nên các mục này chứa khung cal. Không xoá được lịch sử v1;
    # phase 05+ không được đọc chúng làm dữ liệu v2. Companion trong test bắt cả hai chiều.
    "e2_r1/train.txt", "e2_r2/train.txt", "e2_r1/train/images", "e2_r1/train/labels", "e2_r2/train/images",
    "e2_r2/train/labels",
    # Hiệu chỉnh độ thật của ảnh làm xấu v1 (phase realism, 18/11/2026) đo thống kê trên khung dev v1 của mọi khối, kể cả
    # khối cal: ảnh huấn luyện tổng hợp v1 được chỉnh theo đó. Cùng lý do với E2 ở trên.
    "results/realism/calibration.json", "realism-smoke/dev_b",
}
MANIFEST_SUFFIXES = (".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml")
NOT_TRAINING_INDEX = (
    # Chỉ mục / báo cáo / dự đoán / nhãn chấm, không phải đầu vào huấn luyện hay khai thác.
    "data/real/manifest.csv", "data/real/split.json", "data/real/sessions.json", "data/real/neg_flight_split.json",
    "data/real/labels/manifest.json", "data/real/README.md", "data/real/frames/",
    "data/real/labels/devval/", "data/real/labels/test/",  # nhãn dev-val/test v1 (chọn model, chấm)
    "results/v2/neg-holdout/", "results/data/", "results/labels/", "results/camera/", "results/baseline/pred/",
    "results/e1/preds/", "results/e2/preds/", "results/e4/preds/", "results/ab/", "results/ab-zoom/",
    "results/restore/export.json",  # khung vàng kiểm parity ONNX của enhancer v1 (so số, không huấn luyện)
)
FRAME_TOKEN = re.compile(r"\d{8}-\d{6}_\d{3}_\d{5}")


def _scan_manifests(names: set[str]) -> dict[str, set[str]]:
    """``{mục: khung tìm thấy}``: tên ảnh/nhãn (mục = thư mục) và nội dung manifest text (mục = file) trong ``data/``,
    ``results/`` và ``XLA_DATA_ROOT``, trừ ``NOT_TRAINING_INDEX``."""
    stems = {Path(n).stem for n in names}
    hits: dict[str, set[str]] = {}
    root = repo_root()
    for base, prefix in ((root / "data", "data/"), (root / "results", "results/"), (data_root(), "")):
        for p in base.rglob("*"):
            rel = prefix + p.relative_to(base).as_posix()
            if p.is_dir() or any(rel.startswith(x) for x in NOT_TRAINING_INDEX):
                continue
            if p.stem in stems and p.suffix.lower() in (".jpg", ".png", ".txt"):
                hits.setdefault(rel.rsplit("/", 1)[0], set()).add(p.stem)
            if (p.suffix.lower() in MANIFEST_SUFFIXES and "/labels/" not in f"/{rel}"
                    and p.stat().st_size < 50_000_000):
                found = set(FRAME_TOKEN.findall(p.read_text(encoding="utf-8", errors="replace"))) & stems
                if found:
                    hits.setdefault(rel, set()).update(found)
    return hits


@pytest.mark.data
def test_real_no_cal_or_holdout_in_training_manifests(real, require_data):
    require_data(data_root(), "XLA_DATA_ROOT")
    protected = {x["frame"] for x in real["neg"]["frames"] if x["split"] in ("cal", "holdout")}
    protected |= S.reserved_frames(real["neg"], real["rows"], real["v1"]) - {
        x["frame"] for x in real["neg"]["frames"] if x["split"] == "mine"}
    hits = _scan_manifests(protected)
    exempt = V1_TRAIN_MANIFEST_EXEMPT
    unexpected = {k: len(v) for k, v in hits.items() if k not in exempt}
    assert unexpected == {}, f"khung cal/holdout trong manifest huấn luyện/khai thác: {unexpected}"
    # companion: mục miễn trừ v1 nào không còn chứa khung cal là mục cũ, phải xoá khỏi danh sách (không nới thêm)
    if real["neg"]["cal_mode"] == "block":
        stale = sorted(k for k in exempt if k not in hits)
        assert stale == [], f"miễn trừ v1 không còn đúng (xoá khỏi danh sách): {stale}"
    else:
        assert not hits, "cal theo phiên: không được còn miễn trừ v1"
