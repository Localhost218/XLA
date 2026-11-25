"""Phase 05 AI v2: cấu hình thầy, danh sách cho phép khung thật của ``train.py``, manifest nhãn giả, smoke thầy.

Ca không marker dùng dữ liệu thật đã commit (``split.json``, ``neg_flight_split.json``, ``folds.json``) và manifest giả;
ca kết quả đọc ``results/v2/teacher/`` (phase 05 tạo, đã commit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xla.detect import selftrain as S
from xla.detect import train as T
from xla.paths import repo_root

ROOT = repo_root()
CFG_P = ROOT / "configs" / "teacher_v2.yaml"
RES = ROOT / "results" / "v2" / "teacher"


def _cfg() -> dict:
    return yaml.safe_load(CFG_P.read_text(encoding="utf-8"))


def _split() -> dict[str, str]:
    return {f["frame"]: f["split"] for f in json.loads((ROOT / "data/real/split.json").read_text("utf-8"))["frames"]}


def _neg() -> dict[str, dict]:
    doc = json.loads((ROOT / "data/real/neg_flight_split.json").read_text("utf-8"))
    return {f["frame"]: f for f in doc["frames"]}


def _of(kind: str) -> str:
    return next(f for f, r in _neg().items() if r["split"] == kind)


def _dev_free() -> str:
    """Khung dev v1 không là mine/cal (chưa gán gì)."""
    neg = _neg()
    return next(f for f, sp in sorted(_split().items()) if sp == "dev" and f not in neg)


# ---------------------------------------------------------------- cấu hình


def test_config_valid() -> None:
    c = _cfg()
    assert c["source_set"] == "d02x" and c["version"] == 2
    assert float(c["gpu_cap_h"]) <= 1.0, "trần GPU phase 05 = 1 giờ (chủ dự án 25/11)"
    assert c["teacher"]["size"] in ("auto", *c["teacher"]["candidates"])
    assert set(c["teacher"]["candidates"]) <= {"l", "m"}
    assert int(c["train"]["workers"]) <= 4 and int(c["train"]["imgsz"]) == 1280
    assert float(c["data"]["mine_max_frac"]) <= 0.10
    assert float(c["lock_timeout_s"]) >= 21600
    assert c["pseudo"]["tracker"]["tracker_type"] == "bytetrack"


def test_cli_rejects_unknown_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        S.main(["teacherx", "--config", str(CFG_P)])
    assert e.value.code == 2


# ---------------------------------------------------------------- danh sách cho phép (train.py)


@pytest.mark.parametrize("kind", ["test", "cal", "unassigned", "indoor_neg"])
def test_allowlist_rejects(kind: str) -> None:
    sp = _split()
    frame = {"test": next(f for f, s in sp.items() if s == "test"), "cal": _of("cal"), "unassigned": _dev_free(),
             "indoor_neg": next(f for f, s in sp.items() if s == "neg")}[kind]
    with pytest.raises(ValueError, match="danh sách cho phép"):
        T.check_real_frames(["visdrone__a.jpg", frame], ROOT, n_total=1000)


def test_allowlist_rejects_holdout(monkeypatch, tmp_path) -> None:
    """Holdout hiện 0 khung (Kill (a)): dựng neg_flight_split giả có một khung holdout."""
    doc = json.loads((ROOT / "data/real/neg_flight_split.json").read_text("utf-8"))
    f = _dev_free()
    doc["frames"].append({"frame": f, "session": "x#1", "split": "holdout", "approved": True})
    fake = tmp_path / "root"
    (fake / "data/real").mkdir(parents=True)
    (fake / "data/real/neg_flight_split.json").write_text(json.dumps(doc), "utf-8")
    (fake / "data/real/split.json").write_bytes((ROOT / "data/real/split.json").read_bytes())
    with pytest.raises(ValueError, match="holdout"):
        T.check_real_frames([f], fake, pseudo={f: "x#1"}, n_total=1000)


def test_allowlist_accepts_mine_and_pseudo_and_excludes_fold() -> None:
    mine, dev = _of("mine"), _dev_free()
    sess = _neg()[mine]["session"]
    got = T.check_real_frames(["visdrone__a.jpg", mine, dev], ROOT, pseudo={dev: "s#9"}, n_total=1000)
    assert got == {"mine": 1, "pseudo": 1, "real_other": 0}
    with pytest.raises(ValueError, match="fold đang loại"):
        T.check_real_frames([dev], ROOT, pseudo={dev: "s#9"}, excluded_sessions={"s#9"}, n_total=1000)
    with pytest.raises(ValueError, match="mine của fold"):
        T.check_real_frames([mine], ROOT, excluded_sessions={sess}, n_total=1000)


def test_allowlist_pseudo_on_test_or_cal_rejected() -> None:
    t = next(f for f, s in _split().items() if s == "test")
    for f in (t, _of("cal")):
        with pytest.raises(ValueError):
            T.check_real_frames([f], ROOT, pseudo={f: "s#1"}, n_total=1000)


def test_allowlist_mine_cap() -> None:
    mines = [f for f, r in _neg().items() if r["split"] == "mine"][:20]
    with pytest.raises(ValueError, match="10 %|10%"):
        T.check_real_frames(mines, ROOT, n_total=100)


def test_dataset_info_rejects_mine_in_synthetic_set(tmp_path, monkeypatch) -> None:
    d = tmp_path / "dx"
    for sp in ("train", "val"):
        (d / sp / "images").mkdir(parents=True)
    (d / "manifest.jsonl").write_text(json.dumps({"name": _of("mine")}) + "\n", "utf-8")
    monkeypatch.setattr(T, "data_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="khung thật"):
        T.dataset_info("dx", ROOT)


def test_sample_by_source_stratified() -> None:
    rows = [{"name": f"a{i}.jpg", "split": "train", "source": "a"} for i in range(80)] + \
           [{"name": f"b{i}.jpg", "split": "train", "source": "b"} for i in range(20)]
    got = S.sample_by_source(rows, "train", 10, 1)
    assert len(got) == 10 and sum(n.startswith("a") for n in got) == 8
    assert got == S.sample_by_source(rows, "train", 10, 1)


def test_fold_sessions_disjoint() -> None:
    fs = S.fold_sessions(ROOT, S.frame_sessions(ROOT))
    assert len(fs) == 2 and not (fs[0] & fs[1]) and all(fs)


# ---------------------------------------------------------------- kết quả phase 05 (đã commit)


def test_smoke_json_has_vram_and_hours() -> None:
    sm = json.loads((RES / "smoke.json").read_text("utf-8"))
    assert sm["chosen"] in ("l", "m") and sm["machine"]["gpu"]
    ok = [r for r in sm["sizes"].values() if r.get("fits_8gb")]
    assert ok, "ít nhất một cỡ thầy vừa 8 GB"
    for r in ok:
        assert r["vram_peak_gb"] > 0 and r["hours_per_epoch_est"] > 0


def test_pseudo_manifest_no_test_cal_holdout_and_has_session_fold() -> None:
    man = json.loads((RES / "pseudo_manifest.json").read_text("utf-8"))
    sp, neg = _split(), _neg()
    folds = json.loads((ROOT / "data/real/labels/folds.json").read_text("utf-8"))["folds"]
    assert man["frames"]
    for r in man["frames"]:
        assert sp.get(r["frame"]) == "dev", r["frame"]
        assert neg.get(r["frame"], {}).get("split") not in ("cal", "holdout", "mine"), r["frame"]
        assert r["session"] and r["fold"] in range(len(folds))
    fs = S.fold_sessions(ROOT, S.frame_sessions(ROOT))
    for r in man["frames"]:
        assert r["session"] in fs[r["fold"]]


def test_teacher_json_reports_devpool_and_gpu() -> None:
    doc = json.loads((RES / "teacher.json").read_text("utf-8"))
    assert doc["devpool"]["teacher"]["metrics"]["mAP50"]["value"] is not None
    assert doc["gpu_hours"]["total"] <= doc["gpu_hours"]["cap"] + 1e-6 or doc["gpu_hours"].get("over_cap_note")
    assert Path(doc["pseudo_dir"]).name.startswith("pseudo_v2")
