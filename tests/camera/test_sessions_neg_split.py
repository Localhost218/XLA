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
    # parity ONNX của enhancer-v2.0.0 classic (phase 10, e6b5035): 6 khung cal chỉ nằm ở frames/parity (đã đo 26/11),
    # cùng loại với restore/export.json; classic không học từ dữ liệu.
    "results/v2/classic/export.json",
    # phase 14 v2: guard_fp đo báo nhầm trên đúng 150 khung cal (vai trò của cal theo thiết kế); trang A/B lấy khung dev v1
    # có nhãn người nên dính 4 + 3 khung thuộc khối cal 13/21 (đã chấm 26/11, không dựng lại). Cả hai chỉ đánh giá.
    "results/ab-v2/", "results/ab-v2-zoom/",
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


# ---------------------------------------------------------------- holdout yếu "cùng buổi" (26/11/2026)


def _weak_case():
    """Một phiên 0–200 s, khung mỗi 0,2 s; khối dev 0 ở 0–20 s, khối test 1 ở 100–110 s (chỉ vài khung vào v1)."""
    rows, v1 = [], {}
    for i in range(1001):
        t = round(i * 0.2, 3)
        f = f"r_001_{i:05d}.jpg"
        rows.append({"frame": f, "rec": "r", "session": "r#0", "t_unix": t})
        if t <= 20.0 and i % 5 == 0:
            v1[f] = {"frame": f, "split": "dev", "block": 0, "t_unix": t}
        if 100.0 <= t <= 110.0 and i % 10 == 0:
            v1[f] = {"frame": f, "split": "test", "block": 1, "t_unix": t}
    rows.append({"frame": "r_001_99999.jpg", "rec": "r", "session": "", "t_unix": 60.0})  # ngoài phiên
    mine = next(f for f, v in v1.items() if v["split"] == "dev")
    neg = {"seed": S.SEED, "cal_mode": "block", "sessions": {"r#0": "mine"},
           "frames": [{"frame": mine, "session": "r#0", "split": "mine", "approved": False}]}
    sessions = [{"id": "r#0", "rec": "r", "t_start": 0.0, "t_end": 200.0}]
    devpool = {"r_001_00300.jpg"}  # t = 60 s, ngoài v1 nhưng là devpool → không bao giờ vào holdout yếu
    return rows, v1, neg, sessions, devpool


def test_weak_plan_distance_spacing_and_exclusions():
    rows, v1, neg, sessions, devpool = _weak_case()
    plan = S.plan_weak_holdout(rows, v1, neg, devpool, sessions)
    c = plan["candidates"]
    assert c and plan["rule"] == {"min_dist_s": 30.0, "min_gap_s": 0.5, "block_s": 30.0}
    iv = S.v1_block_intervals(v1)
    assert iv == [(0.0, 20.0), (100.0, 110.0)]
    for x in c:
        assert x["frame"] not in v1 and x["frame"] not in devpool and x["session"] == "r#0"
        assert S.dist_to_blocks(x["t_unix"], iv) >= 30.0 and x["dist_s"] >= 30.0
        assert 50.0 <= x["t_unix"] <= 70.0 or x["t_unix"] >= 140.0
        assert x["block"] == f"r#0/{int(x['t_unix'] // 30)}"
    ts = [x["t_unix"] for x in c]
    assert min(b - a for a, b in itertools.pairwise(ts)) >= S.WEAK_MIN_GAP_S - 1e-9
    h = [plan["dist_hist"][f"{d:g}"] for d in S.WEAK_DIST_STEPS]
    assert h == sorted(h, reverse=True) and plan["dist_hist"]["30"] == len(c)
    assert plan["skipped"]["outside_session"] == 1 and plan["skipped"]["neg_or_reserved"] == 1
    # luật chặt hơn → ít khung hơn; quá xa → 0 khung, không lỗi
    assert len(S.plan_weak_holdout(rows, v1, neg, devpool, sessions, min_dist_s=45.0)["candidates"]) < len(c)
    assert S.plan_weak_holdout(rows, v1, neg, devpool, sessions, min_dist_s=500.0)["candidates"] == []


def test_weak_section_keeps_only_agent_no_person_and_is_labelled():
    rows, v1, neg, sessions, devpool = _weak_case()
    plan = S.plan_weak_holdout(rows, v1, neg, devpool, sessions)
    names = [x["frame"] for x in plan["candidates"]]
    review = {f: {"frame": f, "verdict": "person" if i < 3 else "no_person", "reason": "người" if i < 3 else ""}
              for i, f in enumerate(names)}
    sec, rep = S.build_weak_section(plan, review)
    assert sec["kind"] == S.WEAK_KIND == "weak_same_day" and sec["label"] == S.WEAK_LABEL
    assert sec["approved_by"] == "agent" and sec["owner_approved"] is False
    assert {x["frame"] for x in sec["frames"]} == set(names[3:])
    assert all(x["approved"] is True and x["approved_by"] == "agent" for x in sec["frames"])
    assert [d["frame"] for d in rep["dropped_person_or_doubt"]] == names[:3] and rep["kept"] == len(names) - 3
    assert S.check_weak_section(sec, neg, rows, v1, devpool) == []
    with pytest.raises(ValueError, match="chưa soát"):
        S.build_weak_section(plan, {})


def test_check_weak_section_flags_bad_rows():
    rows, v1, neg, sessions, devpool = _weak_case()
    plan = S.plan_weak_holdout(rows, v1, neg, devpool, sessions)
    sec, _ = S.build_weak_section(plan, {x["frame"]: {"verdict": "no_person"} for x in plan["candidates"]})
    test_f = next(f for f, v in v1.items() if v["split"] == "test")
    near = next(r for r in rows if r["frame"] not in v1 and 30.0 < float(r["t_unix"]) < 40.0)
    cases = [
        (lambda s: s.update(owner_approved=True), "owner_approved"),
        (lambda s: s.update(label="x"), "nhãn"),
        (lambda s: s["frames"].append(s["frames"][0] | {"frame": test_f}), "split.json v1"),
        (lambda s: s["frames"].append(s["frames"][0] | {"frame": near["frame"], "t_unix": near["t_unix"]}),
         "gần khối"),
        (lambda s: s["frames"].append(s["frames"][0] | {"frame": "r_001_00300.jpg", "t_unix": 60.0}), "devpool"),
        (lambda s: s["frames"][1].update(t_unix=s["frames"][0]["t_unix"] + 0.1), "< 0.5 s"),
        (lambda s: s["frames"][0].update(approved_by="owner"), "sai dạng"),
    ]
    for mutate, msg in cases:
        bad = copy.deepcopy(sec)
        mutate(bad)
        assert any(msg in e for e in S.check_weak_section(bad, neg, rows, v1, devpool)), msg


def test_weak_section_invisible_to_strict_split(case):
    (doc, _), rows = _split(case)
    v1 = case["split_v1"]
    sec = {"kind": S.WEAK_KIND, "label": S.WEAK_LABEL, "approved_by": "agent", "owner_approved": False,
           "rule": {"min_dist_s": 30.0, "min_gap_s": 0.5, "block_s": 30.0}, "frames": []}
    with_weak = doc | {S.WEAK_KEY: sec}
    assert S.check_split(with_weak, rows, v1) == []  # đường chặt vẫn đạt, frames không đổi
    assert with_weak["frames"] == doc["frames"] and S.weak_frames(doc) == []
    assert any("khoá/seed" in e for e in S.check_split(doc | {"holdout_extra": sec}, rows, v1))
    # apply-review (đường chặt) giữ nguyên khối holdout yếu
    new, _ = S.apply_review(with_weak, _export(with_weak, "s"), "s")
    assert new[S.WEAK_KEY] == sec


def _weak_frames(n_sessions=3, per=4):
    return [{"frame": f"r_00{s}_{i:05d}.jpg", "session": f"r#{s}", "t_unix": float(i), "dist_s": 40.0,
             "block": f"r#{s}/{i // 2}", "approved": True, "approved_by": "agent"}
            for s in range(n_sessions) for i in range(per)]


def _fp_csv(path, counts):
    lines = ["frame,x1,y1,x2,y2,conf"]
    for f, n in counts.items():
        lines += [f"{f},1,1,5,5,0.9"] * n
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_weak_report_per_stage_session_block_ci(tmp_path):
    fr = _weak_frames()
    counts = {"detector": {fr[0]["frame"]: 2, fr[5]["frame"]: 1}, "verifier": {fr[0]["frame"]: 1}, "track": {}}
    csvs = {s: _fp_csv(tmp_path / f"{s}.csv", c) for s, c in counts.items()}
    rep = S.weak_report(csvs, fr, boot_n=200)
    assert rep["kind"] == S.WEAK_KIND and rep["approved_by"] == "agent" and rep["owner_approved"] is False
    assert rep["n_frames"] == 12 and rep["n_sessions"] == 3 and rep["n_blocks"] == 6
    det = rep["by_stage"]["detector"]
    assert det["fp"] == 3 and det["fp_per_frame"] == pytest.approx(0.25)
    assert rep["by_stage"]["verifier"]["by_session"]["r#0"] == {"n": 4, "fp": 1, "fp_per_frame": 0.25}
    for st in ("detector", "verifier"):
        b = rep["by_stage"][st]
        for k in ("ci95_session", "ci95_block"):
            assert b[k][0] <= b["fp_per_frame"] <= b[k][1]
    assert rep["by_stage"]["track"]["fp_per_frame"] == 0.0
    assert rep["verdict"].startswith("đạt (không đủ chứng minh") and S.WEAK_LABEL in rep["verdict"]
    empty = S.weak_report({}, [])
    assert empty["verdict"] == "không đủ chứng minh (0 khung)" and empty["by_stage"]["track"] is None


def test_score_weak_reuses_holdout_path_without_touching_score_json(tmp_path, monkeypatch):
    from xla.eval import system as Y

    root = repo_root()
    real = S.load_neg_split(root)
    rec = "20261102-100941"
    fake = [{"frame": f"{rec}_008_{90000 + i:05d}.jpg", "session": f"{rec}#{i % 2}", "t_unix": float(i),
             "dist_s": 40.0, "block": f"{rec}#{i % 2}/0", "approved": True, "approved_by": "agent"} for i in range(6)]
    monkeypatch.setattr(S, "load_neg_split", lambda _root=None: real | {S.WEAK_KEY: {"rule": {}, "frames": fake}})
    monkeypatch.setattr(Y, "iotcv_venv", lambda _iotcv: Path("D:/fake-iotcv/.venv"))
    sets = Y.load_frame_sets(root)
    calls = []

    def runner(call):
        calls.append(call)
        cmd = call.cmd
        if "scripts/vision/lay_model.py" in cmd:
            return {}
        out = Path(cmd[cmd.index("--out") + 1])
        stage = cmd[cmd.index("--stage") + 1]
        frames = Y.read_list(Path(cmd[cmd.index("--frames") + 1]))
        out.parent.mkdir(parents=True, exist_ok=True)
        _fp_csv(out, {f: 1 for f in frames[:1]} if stage == "detector" else {})
        return {"frames_written": len(frames), "stage": stage}

    known = sorted({x["frame"] for x in fake} | sets.test | sets.cal | sets.mine)
    score_before = (root / Y.SYSTEM_REL / "score.json").read_bytes()
    doc = S.score_weak(Path("D:/fake-iotcv"), tmp_path / "runs", runner, root, boot_n=50,
                       rec_frames=lambda _i, r: [f for f in known if Y.rec_of(f) == r], out=tmp_path / "hw.json")
    assert (root / Y.SYSTEM_REL / "score.json").read_bytes() == score_before
    replays = [c for c in calls if "scripts/vision/replay_pred.py" in c.cmd]
    assert sorted(c.cmd[c.cmd.index("--stage") + 1] for c in replays) == sorted(Y.STAGES)
    for c in replays:
        cmd = c.cmd
        frames = set(Y.read_list(Path(cmd[cmd.index("--frames") + 1])))
        blank = set(Y.read_list(Path(cmd[cmd.index("--blank") + 1])))
        assert frames == {x["frame"] for x in fake} and not frames & blank
        assert sets.test | sets.cal | sets.mine <= blank
        if cmd[cmd.index("--stage") + 1] == "track":
            assert blank == set(Y.score_holdout_blank(sets))
        assert "--track-k" not in cmd and "--verifier-conf" not in cmd
        assert "VIRTUAL_ENV" not in c.env and c.env["UV_PROJECT_ENVIRONMENT"] == str(Path("D:/fake-iotcv/.venv"))
    assert doc["n_frames"] == 6 and doc["by_stage"]["detector"]["fp"] == 1 and doc["label"] == S.WEAK_LABEL
    assert json.loads((tmp_path / "hw.json").read_text(encoding="utf-8"))["kind"] == S.WEAK_KIND


@pytest.mark.data
def test_real_weak_holdout_section_and_score(real, require_data):
    sec = real["neg"].get(S.WEAK_KEY)
    assert sec is not None, "thiếu khối holdout_weak (chạy `sessions weak-holdout`)"
    root = repo_root()
    devpool_p = root / S.DEVPOOL_JSON_REL
    devpool = set(json.loads(devpool_p.read_text(encoding="utf-8"))["frames"]) if devpool_p.exists() else set()
    assert S.check_weak_section(sec, real["neg"], real["rows"], real["v1"], devpool) == []
    plan = json.loads((root / S.WEAK_PLAN_REL).read_text(encoding="utf-8"))
    assert plan["rule"] == sec["rule"]
    # plan đã ghi khớp dữ liệu hiện tại (tất định), và mọi khung giữ là ứng viên của plan
    again = S.plan_weak_holdout(real["rows"], real["v1"], real["neg"], devpool, real["sessions"],
                                sec["rule"]["min_dist_s"], sec["rule"]["min_gap_s"])
    assert again["candidates"] == plan["candidates"] and again["dist_hist"] == plan["dist_hist"]
    assert {x["frame"] for x in sec["frames"]} <= {c["frame"] for c in plan["candidates"]}
    review = S.read_agent_review(root / S.WEAK_REVIEW_REL)
    assert {c["frame"] for c in plan["candidates"]} <= set(review)  # mọi ứng viên đã được agent soát
    score = json.loads((root / S.WEAK_SCORE_REL).read_text(encoding="utf-8"))
    assert score["kind"] == S.WEAK_KIND and score["label"] == S.WEAK_LABEL and score["owner_approved"] is False
    assert score["n_frames"] == len(S.weak_frames(real["neg"])) and score["rule"] == sec["rule"]
    if sec["frames"]:  # khung holdout yếu không được nằm trong manifest huấn luyện/khai thác nào
        require_data(data_root(), "XLA_DATA_ROOT")
        hits = {k: len(v) for k, v in _scan_manifests({x["frame"] for x in sec["frames"]}).items()}
        assert hits == {}, f"khung holdout yếu trong manifest huấn luyện: {hits}"
    else:
        assert score["verdict"] == "không đủ chứng minh (0 khung)"
