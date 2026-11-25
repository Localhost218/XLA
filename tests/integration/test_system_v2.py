"""Phase 09 AI v2 — cả hệ detector → verifier → track (``xla.eval.system``).

Khẳng định theo "Tiêu chí xong" của phase: ``calibrate`` luôn ``--blank`` đủ test + holdout (+ mine khi ``cal_mode: block``)
ở MỌI lượt gọi sang IOT-CV (bắt lệnh subprocess bằng runner giả, kể cả khi holdout không rỗng); ``calibrate`` không đọc
nhãn test/holdout; ``score`` không ghi meta; CSV cả hệ đúng cột §14.1; env con bỏ biến uv của XLA; README đủ năm dòng
chỉ số nhánh B; kết quả đã commit khớp meta.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from xla.eval import detection as D
from xla.eval import system as S
from xla.paths import repo_root

ROOT = repo_root()
HEADER = ["frame", "x1", "y1", "x2", "y2", "conf"]
REC = "20261102-100941"
FAKE_VENV = Path("D:/fake-iotcv/.venv")
FAKE_HOLDOUT = frozenset(f"{REC}_008_{i:05d}.jpg" for i in range(90000, 90006))  # khung giả, không thuộc tập nào


def _sets_with_holdout() -> S.FrameSets:
    s = S.load_frame_sets(ROOT)
    assert not (FAKE_HOLDOUT & (s.test | s.cal | s.mine | s.devpool | s.neg_indoor))
    session = dict(s.session_of) | {f: f"{REC}#9" for f in FAKE_HOLDOUT}
    day = dict(s.day_of) | {f: REC[:8] for f in FAKE_HOLDOUT}
    return dataclasses.replace(s, holdout=FAKE_HOLDOUT, holdout_approved=FAKE_HOLDOUT, session_of=session, day_of=day)


def _arg(cmd: list[str], flag: str) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class FakeRunner:
    """Thay subprocess sang IOT-CV: ghi lại mọi lệnh + env, ghi CSV đúng chỗ (dự đoán lấy từ CSV v1 của XLA)."""

    def __init__(self) -> None:
        self.calls: list[S.IotcvCall] = []
        self.v1 = {}
        with (ROOT / S.V1_TEST_CSV_REL).open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                self.v1.setdefault(r["frame"], []).append([r[k] for k in HEADER])

    def _write(self, out: Path, frames: list[str]) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(HEADER)
            for f in sorted(frames):
                w.writerows(self.v1.get(f, []))

    def __call__(self, call: S.IotcvCall) -> dict:
        self.calls.append(call)
        cmd = call.cmd
        if "scripts/vision/lay_model.py" in cmd:
            return {}
        if "_iotcv" in cmd:
            spec = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
            if spec["mode"] == "cache":
                return {"frames_written": 1, "frames_replayed": 1, "frames_blank": 0, "track_resets": 0, "bundles": {}}
            if spec["mode"] == "grid":
                frames = S.read_list(Path(spec["frames"]))
                for p in spec["points"]:
                    self._write(Path(p["out"]), frames)
                return {"points": [{"out": p["out"], "frames_written": len(frames), "frames_replayed": 0,
                                    "frames_blank": 0, "track_resets": 0, "track": p["track"]} for p in spec["points"]]}
            raise AssertionError(spec["mode"])
        assert "scripts/vision/replay_pred.py" in cmd, cmd
        self._write(Path(_arg(cmd, "--out")), S.read_list(Path(_arg(cmd, "--frames"))))
        return {"frames_written": 0, "frames_replayed": 0, "frames_blank": 0, "track_resets": 0, "track": {},
                "bundles": {}, "stage": _arg(cmd, "--stage")}

    def replay_lists(self) -> list[tuple[list[str], set[str], set[str], str]]:
        """(lệnh, khung ghi ra, khung blank, stage) của mọi lượt phát lại — replay_pred thật lẫn spec mô phỏng."""
        out = []
        for c in self.calls:
            if "scripts/vision/replay_pred.py" in c.cmd:
                out.append((c.cmd, set(S.read_list(Path(_arg(c.cmd, "--frames")))),
                            set(S.read_list(Path(_arg(c.cmd, "--blank")))), _arg(c.cmd, "--stage")))
            elif "_iotcv" in c.cmd:
                spec = json.loads(Path(c.cmd[-1]).read_text(encoding="utf-8"))
                src = spec["replay"] if spec["mode"] == "cache" else spec
                out.append((c.cmd, set(S.read_list(Path(src["frames"]))), set(S.read_list(Path(src["blank"]))),
                            src.get("stage", "track")))
        return out


def _meta_bytes() -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted((ROOT / S.RELEASE_REL).glob("*/meta.json"))}


# ---------------------------------------------------------------- luật thuần


def test_env_con_bo_bien_uv_cua_xla():
    env = S.iotcv_child_env({"UV_PROJECT_ENVIRONMENT": "D:/v", "VIRTUAL_ENV": "D:/v", "PATH": "p"},
                            extra={"DETECTOR_ONNX": "x"})
    assert "UV_PROJECT_ENVIRONMENT" not in env and "VIRTUAL_ENV" not in env
    assert env["PATH"] == "p" and env["DETECTOR_ONNX"] == "x" and env["PYTHONUTF8"] == "1"


def test_env_con_tro_venv_cua_iotcv_khong_phai_xla(tmp_path, monkeypatch):
    own = tmp_path / "wt"
    for rel in ("Scripts/python.exe", "bin/python"):  # Windows / POSIX
        (own / ".venv" / rel).parent.mkdir(parents=True, exist_ok=True)
        (own / ".venv" / rel).write_bytes(b"")
    env = S.iotcv_child_env({"UV_PROJECT_ENVIRONMENT": "D:/DevCache/venvs/xla", "VIRTUAL_ENV": "D:/x"}, iotcv=own)
    assert env["UV_PROJECT_ENVIRONMENT"] == str(own / ".venv") and "VIRTUAL_ENV" not in env
    bare = tmp_path / "bare"  # không .venv, pyproject khác clone chính -> lỗi, không tự tạo venv rỗng
    bare.mkdir()
    (bare / "pyproject.toml").write_text("a", encoding="utf-8")
    (bare / "uv.lock").write_text("", encoding="utf-8")
    main = tmp_path / "main"
    main.mkdir()
    (main / "pyproject.toml").write_text("b", encoding="utf-8")
    (main / "uv.lock").write_text("", encoding="utf-8")
    monkeypatch.setenv("XLA_IOTCV_ROOT", str(main))
    with pytest.raises(RuntimeError):
        S.iotcv_venv(bare)


def test_lenh_replay_luon_co_blank_va_dung_cach_goi():
    cmd = S.replay_cmd(REC, Path("f.txt"), "track", Path("o.csv"), Path("b.txt"), Path("d.onnx"), Path("v.onnx"),
                       {"k": 2, "n": 4, "iou": 0.3, "max_gap": 1}, 0.35)
    assert cmd[:5] == ["uv", "run", "--no-sync", "python", "scripts/vision/replay_pred.py"]
    assert _arg(cmd, "--blank") == "b.txt" and _arg(cmd, "--track-k") == "2" and _arg(cmd, "--verifier-conf") == "0.35"
    assert cmd[-1] == "--json"
    with pytest.raises(ValueError):
        S.replay_cmd(REC, Path("f"), "bogus", Path("o"), Path("b"), Path("d"), None)


def test_blank_calibrate_gom_test_holdout_va_mine_khi_block():
    s = _sets_with_holdout()
    assert s.cal_mode == "block"
    b = S.calibrate_blank(s)
    assert s.test <= b and s.holdout <= b and s.mine <= b
    assert not (b & S.calibrate_frames(s))
    b2 = S.calibrate_blank(dataclasses.replace(s, cal_mode="session"))
    assert s.test <= b2 and s.holdout <= b2 and not (s.mine & b2)
    assert S.score_test_blank(s) == s.holdout | s.cal
    assert S.score_holdout_blank(s) == s.test | s.cal | s.mine
    with pytest.raises(ValueError):
        S.check_disjoint(["a"], ["a"], "x")


def test_select_d7_uu_tien_diem_don_gian_trong_vung_hoa():
    rows = [
        {"thr": 0.1, "k": 1, "n": 1, "max_gap": 1, "cal_fp": 9, "cal_fp_per_frame": 0.06, "tp": 70},  # vượt cal
        {"thr": 0.3, "k": 1, "n": 1, "max_gap": 1, "cal_fp": 4, "cal_fp_per_frame": 0.027, "tp": 59},
        {"thr": 0.2, "k": 2, "n": 4, "max_gap": 1, "cal_fp": 3, "cal_fp_per_frame": 0.02, "tp": 60},
        {"thr": 0.1, "k": 3, "n": 5, "max_gap": 2, "cal_fp": 1, "cal_fp_per_frame": 0.007, "tp": 66},
    ]
    sel = S.select_d7(rows)
    assert sel["chosen"]["k"] == 3  # 66 hơn 60 quá 1 người: không hoà
    rows[3]["tp"] = 61
    assert S.select_d7(rows)["chosen"]["k"] == 2  # 61 vs 60: hoà, (2,4,1) đơn giản hơn (3,5,2); 59 ngoài vùng hoà
    assert S.select_d7([rows[0]])["chosen"] is None


# ---------------------------------------------------------------- calibrate: bắt lệnh, không đọc nhãn test/holdout


def test_calibrate_luon_blank_test_holdout_va_khong_doc_nhan_test(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "iotcv_venv", lambda _iotcv: FAKE_VENV)
    s = _sets_with_holdout()
    seen: list[tuple[str, list[str]]] = []
    real_load = D.load_labels

    def spy(labels_dir, sub, frames, allow_missing, *a, **kw):
        seen.append((sub, list(frames)))
        return real_load(labels_dir, sub, frames, allow_missing, *a, **kw)

    monkeypatch.setattr(D, "load_labels", spy)
    before = _meta_bytes()
    runner = FakeRunner()
    folds = [{"fold": k, "onnx": str(tmp_path / f"verifier-fold{k}" / "verifier.onnx")} for k in range(len(s.folds))]
    pts = [S.Point(0.35, 1, 1, 1), S.Point(0.2, 2, 4, 2)]
    doc = S.calibrate(Path("D:/fake-iotcv"), tmp_path / "runs", runner, pts, ROOT, s, workers=1, parity_n=2,
                      fold_verifiers=folds, out_dir=tmp_path / "out")

    replays = runner.replay_lists()
    assert len(replays) == 2 * (1 + len(s.folds)) + 2  # cache + grid (1 khúc/lượt) mỗi verifier + parity
    must = s.test | s.holdout | s.mine
    for cmd, frames, blank, _ in replays:
        assert must <= blank, cmd
        assert not (frames & blank), cmd
        assert frames <= s.cal | s.devpool, cmd
    for c in runner.calls:
        assert "VIRTUAL_ENV" not in c.env and c.env["UV_PROJECT_ENVIRONMENT"] == str(FAKE_VENV)
        assert c.cmd[:3] == ["uv", "run", "--no-sync"]
    for c in runner.calls:  # GPU (khoá D6) đúng ở lượt suy luận thật; lưới mô phỏng và lay_model không giữ khoá
        cpu_only = "scripts/vision/lay_model.py" in c.cmd or ("_iotcv" in c.cmd and "grid_" in Path(c.cmd[-1]).name)
        assert c.gpu is not cpu_only, c.cmd
    assert seen and all(sub == "devpool" for sub, _ in seen)
    assert all(set(fr) <= s.devpool for _, fr in seen)
    assert _meta_bytes() == before
    assert doc["blank"]["test"] == len(s.test) and doc["blank"]["holdout"] == len(FAKE_HOLDOUT)
    assert all(p["identical"] for p in doc["parity"]) and len(doc["parity"]) == 2
    assert (tmp_path / "out" / "calibrate.json").is_file() and (tmp_path / "out" / "calibrate_curve.png").is_file()


# ---------------------------------------------------------------- score: không ghi meta, CSV §14.1, blank đúng


@pytest.mark.skipif(not (ROOT / S.RELEASE_REL / S.DETECTOR_SYSTEM / "meta.json").is_file(),
                    reason="chưa chạy apply (detector-v1.0.1)")
def test_score_khong_ghi_meta_csv_dung_cot_va_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "iotcv_venv", lambda _iotcv: FAKE_VENV)
    s = _sets_with_holdout()
    known = s.test | s.neg_indoor | s.cal | s.mine | s.devpool | s.holdout

    def rec_frames(_iotcv: Path, rec: str) -> list[str]:
        return sorted(f for f in known if S.rec_of(f) == rec)

    before = _meta_bytes()
    runner = FakeRunner()
    doc = S.score(Path("D:/fake-iotcv"), tmp_path / "runs", runner, ROOT, s, boot_n=20, rec_frames=rec_frames,
                  out_dir=tmp_path / "out")
    assert _meta_bytes() == before
    for cmd, frames, blank, stage in runner.replay_lists():
        assert "--track-k" not in cmd and "--verifier-conf" not in cmd  # điểm vận hành chỉ từ meta
        assert not (frames & blank)
        if frames <= FAKE_HOLDOUT:
            assert s.test | s.cal | s.mine <= blank
        else:
            assert s.holdout | s.cal <= blank
            if stage == "track":
                assert blank == s.holdout | s.cal
    for p in (tmp_path / "out" / "preds").glob("*.csv"):
        with p.open(encoding="utf-8", newline="") as fh:
            assert next(csv.reader(fh)) == HEADER, p
    assert {p.name for p in (tmp_path / "out" / "preds").glob("*.csv")} == {
        f"system_{st}_{n}.csv" for st in S.STAGES for n in ("test", "holdout")}
    assert doc["holdout"]["n_frames"] == len(FAKE_HOLDOUT)
    assert set(doc["test"]["stages"]) == set(S.STAGES)


def test_holdout_rong_bao_khong_du_chung_minh():
    s = dataclasses.replace(S.load_frame_sets(ROOT), holdout_approved=frozenset())
    r = S.holdout_report({}, s)
    assert r["n_frames"] == 0 and r["verdict"] == "không đủ chứng minh (0 khung)"


# ---------------------------------------------------------------- kết quả đã commit


def _load(rel: str) -> dict:
    p = ROOT / S.SYSTEM_REL / rel
    if not p.is_file():
        pytest.fail(f"thiếu {p} (chạy python -m xla.eval.system ...)", pytrace=False)
    return json.loads(p.read_text(encoding="utf-8"))


def test_ket_qua_calibrate_khop_meta():
    cal = _load("calibrate.json")
    ch = cal["selection"]["chosen"]
    det = json.loads((ROOT / S.RELEASE_REL / S.DETECTOR_SYSTEM / "meta.json").read_text(encoding="utf-8"))
    ver = json.loads((ROOT / S.RELEASE_REL / S.VERIFIER_SYSTEM / "meta.json").read_text(encoding="utf-8"))
    assert det["recommended_track"] == {"k": ch["k"], "n": ch["n"], "iou": ch["iou"], "max_gap": ch["max_gap"]}
    assert ver["conf_threshold"] == ch["thr"]
    assert ch["cal_fp_per_frame"] <= S.D7_CAL_MAX
    assert cal["parity"] and all(p["identical"] for p in cal["parity"])
    s = S.load_frame_sets(ROOT)
    assert cal["blank"]["test"] == len(s.test) and cal["blank"]["holdout"] == len(s.holdout)
    assert cal["blank"]["mine"] == (len(s.mine) if s.cal_mode == "block" else 0)
    assert len(cal["rows"]) == len(S.grid_points())


def test_detector_v101_cung_onnx_v100():
    from xla import contract as C

    a = json.loads((ROOT / S.RELEASE_REL / S.DETECTOR_BASE / "meta.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / S.RELEASE_REL / S.DETECTOR_SYSTEM / "meta.json").read_text(encoding="utf-8"))
    assert b["onnx_sha256"] == a["onnx_sha256"] and b["contract_version"] == "1.1" and b["version"] == "1.0.1"
    assert {k: v for k, v in b.items() if k not in ("recommended_track", "contract_version", "version")} == {
        k: v for k, v in a.items() if k not in ("contract_version", "version")}
    for g in ("000.jpg", "001.jpg", "002.jpg", "expected.json"):
        ga, gb = (ROOT / S.RELEASE_REL / x / "golden" / g for x in (S.DETECTOR_BASE, S.DETECTOR_SYSTEM))
        assert hashlib.sha256(ga.read_bytes()).hexdigest() == hashlib.sha256(gb.read_bytes()).hexdigest()
    if (ROOT / S.RELEASE_REL / S.DETECTOR_SYSTEM / "detector.onnx").is_file():
        assert C.validate_bundle(ROOT / S.RELEASE_REL / S.DETECTOR_SYSTEM) == []


def test_csv_ca_he_da_commit_dung_cot():
    files = sorted((ROOT / S.SYSTEM_REL / "preds").glob("system_*_test.csv"))
    assert {p.name for p in files} == {f"system_{st}_test.csv" for st in S.STAGES}
    s = S.load_frame_sets(ROOT)
    for p in files:
        with p.open(encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            assert next(rd) == HEADER
            frames = {r[0] for r in rd}
        assert frames <= s.test | s.neg_indoor, p


def test_readme_du_nam_dong_chi_so_nhanh_b():
    text = (ROOT / S.SYSTEM_REL / "README.md").read_text(encoding="utf-8")
    for label in ("B1: test mAP50", "B1 tối thiểu", "B2: báo nhầm", "Recall ở ngưỡng vận hành", "ms trực tiếp"):
        assert label in text, label
    assert "detector-v1.0.1" in text and "Kill (a)" in text
