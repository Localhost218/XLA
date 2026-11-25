"""Cả hệ detector → verifier → lọc track: hiệu chỉnh trên ``cal`` + devpool, chấm một lần trên test + holdout (AI v2
phase 09; hợp đồng ``docs/hop-dong-model.md`` §10, §14; D4, D5, D7, D11).

Mọi dự đoán cả hệ ra từ **code runtime của IOT-CV**: ``scripts/vision/replay_pred.py`` (§14.1) chạy trong venv của IOT-CV
(``uv run --no-sync``, env con bỏ ``UV_PROJECT_ENVIRONMENT``/``VIRTUAL_ENV`` của XLA). Chạy GPU nào cũng bọc trong
``xla.gpu_lock.gpu_lock`` và nhả ngay sau mỗi lần.

AX-01::

    python -m xla.eval.system calibrate --iotcv D:/IOTCV-wt/iotcv --json   # lưới k/n/max_gap/ngưỡng verifier -> calibrate.json
    python -m xla.eval.system apply --json                                  # recommended_track + conf_threshold vào meta
    python -m xla.eval.system score --iotcv D:/IOTCV-wt/iotcv --json       # test + holdout, một lần -> score.json
    python -m xla.eval.system bench --iotcv D:/IOTCV-wt/iotcv --json       # ms từng bước v1 vs cả hệ -> bench.json
    # mã thoát 0 xong; 1 dữ liệu / bundle / replay lỗi; 2 sai cách gọi

**calibrate** (bước 2, D7): ``--blank`` luôn = mọi khung test + holdout (+ mine khi ``cal_mode: block``), nên chỉ khung
không bị cấm tạo ngữ cảnh track; ``--frames`` = cal ∪ devpool. Lưới đầy đủ (720 điểm × 3 lượt) quá chậm nếu mỗi điểm là
một lượt phát lại GPU (~5 phút), nên detector + verifier chạy **thật** một lượt cho mỗi verifier (bản cuối cho cal; bản
fold i cho khung devpool fold i = OOF của verifier), ghi lại đầu ra từng khung; mỗi điểm lưới chạy lại đúng hàm
``replay_pred.run`` + ``_Replay.rows_for`` + ``TrackFilter`` của IOT-CV, chỉ thay phép suy luận ONNX bằng đầu ra đã ghi.
Vài điểm (gồm điểm chọn) được phát lại thật bằng ``replay_pred.py --stage track`` và phải trùng **từng byte** CSV mô phỏng.

**score** (bước 4): chỉ đọc meta (không ghi), chép bundle sang IOT-CV bằng ``lay_model.py``, phát lại ba tầng trên test v1
(+ 300 khung ``neg`` trong nhà) với ``--blank`` = holdout + cal (tầng track), rồi chấm bằng ``xla.eval.detection``; holdout
(khung ``approved``) với ``--blank`` = test + cal + mine.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = ("detector", "verifier", "track")
SYSTEM_REL = Path("results/v2/system")
RELEASE_REL = Path("models/release")
DETECTOR_BASE = "detector-v1.0.0"
DETECTOR_SYSTEM = "detector-v1.0.1"  # = v1.0.0 + recommended_track (Kill (b) phase 06: không có detector-v2.0.0)
VERIFIER_SYSTEM = "verifier-v1.0.0"
VERIFIER_CAND = "effb0_96"  # ứng viên phase 07 đã ship (results/v2/verifier/select.json)
V1_TEST_CSV_REL = Path("results/e2/preds/onnx_detector-v1.0.0_all.csv")  # = xla.detect.kfold.V1_TEST_CSV_REL
EDGE16_CSV_REL = Path("results/baseline/pred/edge16.csv")
GRID_K = (1, 2, 3, 4)
GRID_N_MAX = 6
GRID_GAP = (1, 2)
GRID_THR = tuple(round(0.05 * i, 2) for i in range(20))  # 0,00 … 0,95 bước 0,05
TRACK_IOU = 0.3  # = FALLBACK_IOU của backend/vision/track_filter.py; plan không đưa iou vào lưới
D7_CAL_MAX = 0.03  # báo nhầm/khung cal tối đa (D7)
B2_MAX = 0.05
TIE_PERSONS = 1  # vùng hoà: recall kém điểm tốt nhất ≤ 1 người (độ phân giải của 218 người devpool)
PARITY_POINTS = 3
IOTCV_TIMEOUT_S = 4 * 3600
GPU_LOCK_TIMEOUT_S = 21600
WORKERS = 4  # luật máy: tối đa 4 worker
BOOT_N = 2000


LF = "\n"  # file ghi ra luôn LF (.gitattributes eol=lf), kể cả trên Windows


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8", newline=LF)
    os.replace(tmp, path)
    return path


def write_list(path: Path, frames: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f + LF for f in sorted(frames)), encoding="utf-8", newline=LF)
    return path


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def rec_of(frame: str) -> str:
    return frame.split("_", 1)[0]


# ---------------------------------------------------------------- tập khung (D5, §14.2) + luật --blank


@dataclass(frozen=True)
class FrameSets:
    test: frozenset[str]
    neg_indoor: frozenset[str]
    cal: frozenset[str]
    holdout: frozenset[str]
    holdout_approved: frozenset[str]
    mine: frozenset[str]
    devpool: frozenset[str]
    folds: tuple[frozenset[str], ...]
    cal_mode: str
    session_of: dict[str, str]
    day_of: dict[str, str]


def load_frame_sets(root: Path | None = None) -> FrameSets:
    """Đọc ``split.json`` (chỉ đọc), ``neg_flight_split.json``, ``folds.json``. Không đọc nhãn."""
    from xla.paths import repo_root

    root = root or repo_root()
    split = json.loads((root / "data/real/split.json").read_text(encoding="utf-8"))
    neg = json.loads((root / "data/real/neg_flight_split.json").read_text(encoding="utf-8"))
    folds = json.loads((root / "data/real/labels/folds.json").read_text(encoding="utf-8"))
    by_split: dict[str, set[str]] = {}
    for r in split["frames"]:
        by_split.setdefault(r["split"], set()).add(r["frame"])
    nf: dict[str, set[str]] = {}
    for r in neg["frames"]:
        nf.setdefault(r["split"], set()).add(r["frame"])
    approved = {r["frame"] for r in neg["frames"] if r["split"] == "holdout" and r.get("approved")}
    session_of = {r["frame"]: r["session"] for r in neg["frames"]}
    day_of = {f: rec_of(f)[:8] for f in session_of}
    fold_sets = tuple(frozenset(f) for f in folds["folds"])
    return FrameSets(test=frozenset(by_split.get("test", ())), neg_indoor=frozenset(by_split.get("neg", ())),
                     cal=frozenset(nf.get("cal", ())), holdout=frozenset(nf.get("holdout", ())),
                     holdout_approved=frozenset(approved), mine=frozenset(nf.get("mine", ())),
                     devpool=frozenset().union(*fold_sets) if fold_sets else frozenset(), folds=fold_sets,
                     cal_mode=str(neg.get("cal_mode", "session")), session_of=session_of, day_of=day_of)


def calibrate_blank(s: FrameSets) -> frozenset[str]:
    """``--blank`` của mọi lượt phát lại khi hiệu chỉnh: test + holdout (+ mine khi ``cal_mode: block``)."""
    out = set(s.test) | set(s.holdout)
    if s.cal_mode == "block":
        out |= set(s.mine)
    return frozenset(out)


def calibrate_frames(s: FrameSets) -> frozenset[str]:
    return frozenset(s.cal | s.devpool)


def score_test_blank(s: FrameSets) -> frozenset[str]:
    """Chấm test: tầng track phát lại với ``--blank`` = holdout + cal (phase 09 bước 4)."""
    return frozenset(s.holdout | s.cal)


def score_holdout_blank(s: FrameSets) -> frozenset[str]:
    return frozenset(s.test | s.cal | s.mine)


def check_disjoint(frames: Iterable[str], blank: Iterable[str], what: str) -> None:
    both = sorted(set(frames) & set(blank))
    if both:
        raise ValueError(f"{what}: --frames giao --blank ở {len(both)} khung (vd. {both[:3]})")


# ---------------------------------------------------------------- gọi IOT-CV


def iotcv_venv(iotcv: Path) -> Path:
    """venv của IOT-CV cho ``uv run --no-sync``: ``<iotcv>/.venv`` nếu có; worktree lane không có venv riêng thì dùng venv
    của clone chính (``xla.paths.iotcv_root()``), chỉ khi ``pyproject.toml`` + ``uv.lock`` hai bên trùng từng byte (không
    thì không biết venv đó có đúng gói không → lỗi). Không bao giờ là venv của XLA."""
    from xla.paths import iotcv_root

    py = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    own = iotcv / ".venv"
    if (own / py).is_file():
        return own
    main = Path(iotcv_root())
    for f in ("pyproject.toml", "uv.lock"):
        if (iotcv / f).read_bytes() != (main / f).read_bytes():
            raise RuntimeError(f"{iotcv} không có .venv và {f} khác clone chính {main}: không dùng chung venv được")
    if not (main / ".venv" / py).is_file():
        raise RuntimeError(f"không thấy venv IOT-CV ở {own} hay {main / '.venv'}")
    return main / ".venv"


def iotcv_child_env(environ: dict[str, str] | None = None, extra: dict[str, str] | None = None,
                    iotcv: Path | None = None) -> dict[str, str]:
    """Env con sang IOT-CV: bỏ ``UV_PROJECT_ENVIRONMENT``/``VIRTUAL_ENV`` của XLA (không thì uv xoá venv XLA); có
    ``iotcv`` thì trỏ ``UV_PROJECT_ENVIRONMENT`` vào venv CỦA IOT-CV (``iotcv_venv``; không có thì ``uv run`` tự tạo một
    ``.venv`` rỗng trong worktree). Thêm ``extra`` (vd. ``DETECTOR_ONNX``). PATH ``nvidia/*/bin`` cho ORT GPU do
    ``backend/vision/model_bundle.py`` tự lo."""
    from xla.camera.sessions import iotcv_child_env as base

    env = base(environ)
    if iotcv is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(iotcv_venv(iotcv))
    env.update(extra or {})
    return env


@dataclass
class IotcvCall:
    """Một lượt gọi sang IOT-CV: lệnh, cwd, env con, có dùng GPU (bọc khoá D6) không, nhãn khoá."""

    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    gpu: bool
    label: str
    log: Path | None = None


Runner = Callable[[IotcvCall], dict[str, Any]]


def last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"không thấy dòng JSON trong stdout: {stdout[-400:]!r}")


_GPU_HELD: list[str] = []  # nhãn khoá D6 tiến trình này đang giữ (held_gpu); khoá lần hai cùng tiến trình sẽ treo


@contextlib.contextmanager
def held_gpu(label: str) -> Iterator[None]:
    """Giữ khoá D6 cho MỘT việc gồm nhiều lượt GPU nối tiếp (chấm ba tầng, bench hai lượt), nhả ngay khi việc xong: mỗi
    lần nhả khoá giữa các lượt là một lần lane khác chen vào giữ 20+ phút. Lượt ``run_iotcv`` bên trong không khoá lại."""
    from xla.gpu_lock import gpu_lock

    if _GPU_HELD:
        yield
        return
    with gpu_lock(label, timeout_s=GPU_LOCK_TIMEOUT_S):
        _GPU_HELD.append(label)
        try:
            yield
        finally:
            _GPU_HELD.pop()


def run_iotcv(call: IotcvCall) -> dict[str, Any]:
    """Chạy thật: GPU thì giữ khoá D6 suốt lượt (trừ khi đã giữ qua ``held_gpu``) và nhả ngay sau; mã ≠ 0 →
    RuntimeError (không lùi im lặng)."""
    ctx = held_gpu(call.label) if call.gpu else contextlib.nullcontext()
    t0 = time.time()
    with ctx:
        p = subprocess.run(call.cmd, cwd=call.cwd, env=call.env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=IOTCV_TIMEOUT_S, check=False)
    wall = time.time() - t0
    if call.log is not None:
        call.log.parent.mkdir(parents=True, exist_ok=True)
        with call.log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n# {utc_now()} {call.label} exit {p.returncode} {wall:.1f}s\n$ {' '.join(call.cmd)}\n"
                     f"{p.stdout[-4000:]}\n{p.stderr[-4000:]}\n")
    if p.returncode != 0:
        raise RuntimeError(f"{call.label}: exit {p.returncode}: {(p.stderr or p.stdout).strip()[-1200:]}")
    out = last_json(p.stdout)
    out["_wall_s"] = round(wall, 1)
    return out


def replay_cmd(rec: str, frames_file: Path, stage: str, out: Path, blank_file: Path, detector_onnx: Path,
               verifier_onnx: Path | None, track: dict[str, Any] | None = None,
               verifier_conf: float | None = None) -> list[str]:
    """Lệnh ``replay_pred.py`` đúng §14.1. ``--blank`` bắt buộc ở mọi lượt (không có đường gọi nào thiếu nó)."""
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r}")
    cmd = ["uv", "run", "--no-sync", "python", "scripts/vision/replay_pred.py", "--recording",
           f"logs/recordings/{rec}", "--frames", str(frames_file), "--stage", stage, "--out", str(out),
           "--blank", str(blank_file), "--detector-onnx", str(detector_onnx)]
    if verifier_onnx is not None:
        cmd += ["--verifier-onnx", str(verifier_onnx)]
    if track is not None:
        cmd += ["--track-k", str(track["k"]), "--track-n", str(track["n"]), "--track-iou", str(track["iou"]),
                "--track-max-gap", str(track["max_gap"])]
    if verifier_conf is not None:
        cmd += ["--verifier-conf", f"{verifier_conf:g}"]
    return cmd + ["--json"]


def sim_cmd(spec: Path) -> list[str]:
    """Chạy chính file này trong venv IOT-CV (chế độ ``_iotcv``, chỉ thư viện chuẩn + code IOT-CV)."""
    return ["uv", "run", "--no-sync", "python", str(Path(__file__).resolve()), "_iotcv", str(spec)]


def recording_frames(iotcv: Path, rec: str) -> list[str]:
    """Tên khung (D11) của mọi dòng ``frames.csv`` hợp lệ, đúng thứ tự phát lại (như ``replay_bench.read_rows``)."""
    out = []
    with (iotcv / "logs" / "recordings" / rec / "frames.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                seg, vf = int(row["segment"]), int(row["video_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(f"{rec}_{seg:03d}_{vf:05d}.jpg")
    return out


# ---------------------------------------------------------------- phần chạy TRONG venv IOT-CV (chế độ ``_iotcv``)
# Chỉ thư viện chuẩn + numpy + code IOT-CV; không import ``xla``. Spec JSON: {"mode", "iotcv", ...}.


def _iotcv_import(iotcv: Path) -> Any:
    for p in (iotcv, iotcv / "scripts" / "vision"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import replay_pred  # type: ignore[import-not-found]

    return replay_pred


def _replay_args(rp: Any, a: dict[str, Any], iotcv: Path) -> argparse.Namespace:
    argv = ["--recording", str(iotcv / "logs" / "recordings" / a["rec"]), "--frames", a["frames"], "--stage",
            a["stage"], "--out", a["out"], "--blank", a["blank"], "--detector-onnx", a["detector_onnx"]]
    if a.get("verifier_onnx"):
        argv += ["--verifier-onnx", a["verifier_onnx"]]
    if a.get("track"):
        t = a["track"]
        argv += ["--track-k", str(t["k"]), "--track-n", str(t["n"]), "--track-iou", str(t["iou"]), "--track-max-gap",
                 str(t["max_gap"])]
    if a.get("verifier_conf") is not None:
        argv += ["--verifier-conf", repr(float(a["verifier_conf"]))]
    return rp._parser().parse_args(argv)


def _iotcv_cache(spec: dict[str, Any]) -> dict[str, Any]:
    """Một lượt phát lại THẬT (``replay_pred.run``, ``--stage verifier``) và ghi lại, theo đúng thứ tự gọi, đầu ra
    detector (``raw``) + verifier (``verify``) của từng khung, cùng dãy khung của ``replay_frames``."""
    import pickle

    iotcv = Path(spec["iotcv"])
    rp = _iotcv_import(iotcv)
    state: dict[str, Any] = {"idx": -1, "items": [], "det": {}, "ver": {}, "det_conf": None}
    real_frames = rp.replay_frames

    def frames_rec(rec_dir: Path, *a: Any, **kw: Any) -> Any:
        for i, item in enumerate(real_frames(rec_dir, *a, **kw)):
            state["idx"] = i
            fr = item.frame
            state["items"].append((item.n, item.segment, item.video_frame, None if fr is None else fr.timestamp_ms))
            yield item

    base = rp._Replay

    class Recording(base):  # type: ignore[misc, valid-type]
        def __init__(self, args: Any, rec_dir: Path) -> None:
            super().__init__(args, rec_dir)
            det, ver = self.detector, self.verifier
            real_raw, real_verify = det.raw, ver.verify
            state["det_conf"] = float(det.confidence)

            def raw(img: Any) -> Any:
                out = real_raw(img)
                state["det"][state["idx"]] = None if out is None else (out[0].copy(), out[1].copy())
                return out

            def verify(img: Any, boxes: Any) -> Any:
                v = real_verify(img, boxes)
                state["ver"][state["idx"]] = v.copy()
                return v

            det.raw, ver.verify = raw, verify
            state["bundles"] = {"detector": rp._describe(det.bundle), "verifier": rp._describe(ver.bundle)}

    rp.replay_frames, rp._Replay = frames_rec, Recording
    try:
        res = rp.run(_replay_args(rp, spec["replay"], iotcv))
    finally:
        rp.replay_frames, rp._Replay = real_frames, base
    with open(spec["cache"], "wb") as fh:
        pickle.dump({k: state[k] for k in ("items", "det", "ver", "det_conf", "bundles")}, fh, protocol=4)
    return res | {"cache": spec["cache"], "frames_ok": len(state["det"])}


def _iotcv_grid(spec: dict[str, Any]) -> dict[str, Any]:
    """Mỗi điểm lưới: ``replay_pred.run`` + ``_Replay.rows_for`` + ``TrackFilter`` của IOT-CV, chỉ thay ONNX và đọc JPEG
    bằng đầu ra đã ghi ở ``_iotcv_cache`` (cùng dãy khung, cùng luật blank / khung đọc không được)."""
    import pickle
    import types

    import cv2  # type: ignore[import-not-found]
    import numpy as np

    iotcv = Path(spec["iotcv"])
    rp = _iotcv_import(iotcv)
    from backend import config  # type: ignore[import-not-found]
    from backend.vision.track_filter import TrackFilter, resolve_track  # type: ignore[import-not-found]

    with open(spec["cache"], "rb") as fh:
        cache = pickle.load(fh)
    state = {"idx": -1}
    dummy = np.zeros((480, 640, 3), np.uint8)
    ok = set(cache["det"])
    good, bad = b"\x01", b"\x00"

    def frames_fake(rec_dir: Path, *a: Any, **kw: Any) -> Any:
        for i, (n, seg, vf, ts) in enumerate(cache["items"]):
            state["idx"] = i
            fr = None if ts is None else types.SimpleNamespace(jpeg=good if i in ok else bad, timestamp_ms=ts)
            yield types.SimpleNamespace(n=n, segment=seg, video_frame=vf, frame=fr)

    def imdecode_fake(buf: Any, flag: Any) -> Any:
        return dummy if len(buf) and int(buf[0]) == 1 else None

    class Det:
        bundle = None
        confidence = cache["det_conf"]

        def raw(self, img: Any) -> Any:
            return cache["det"][state["idx"]]

    class Ver:
        bundle = None

        def __init__(self, conf: float) -> None:
            self.confidence = conf

        def verify(self, img: Any, boxes: Any) -> Any:
            v = cache["ver"][state["idx"]]
            if len(v) != len(boxes):
                raise RuntimeError(f"khung {state['idx']}: verifier ghi {len(v)} điểm, hộp vào {len(boxes)}")
            return v

    det_meta = rp.read_meta(Path(spec["detector_onnx"]))
    real = (rp.replay_frames, rp._Replay, cv2.imdecode)
    results = []
    try:
        for pt in spec["points"]:
            a = pt | {"rec": spec["rec"], "stage": "track", "detector_onnx": spec["detector_onnx"],
                      "verifier_onnx": spec["verifier_onnx"], "frames": spec["frames"], "blank": spec["blank"]}
            args = _replay_args(rp, a, iotcv)

            def factory(ar: Any, rec_dir: Path, _conf: float = float(pt["verifier_conf"])) -> Any:
                r = object.__new__(real[1])
                r.args, r.rec_dir, r.rec = ar, rec_dir, rec_dir.name
                r.detector, r.verifier = Det(), Ver(_conf)
                params, per_track = resolve_track(det_meta, k=ar.track_k, n=ar.track_n, iou=ar.track_iou,
                                                  max_gap=ar.track_max_gap)
                r.track = TrackFilter(params, cooldown_s=config.DETECTION_COOLDOWN_S, per_track_cooldown=per_track)
                return r

            rp.replay_frames, rp._Replay, cv2.imdecode = frames_fake, factory, imdecode_fake
            res = rp.run(args)
            results.append({"out": pt["out"], "frames_written": res["frames_written"],
                            "frames_replayed": res["frames_replayed"], "frames_blank": res["frames_blank"],
                            "track_resets": res["track_resets"], "track": res["track"]})
    finally:
        rp.replay_frames, rp._Replay, cv2.imdecode = real
    return {"points": results, "frames_ok": len(ok)}


def _iotcv_bench(spec: dict[str, Any]) -> dict[str, Any]:
    """``replay_bench.main`` của IOT-CV; có ``verifier_onnx`` thì pipeline có verifier (bước ``verifier`` §13)."""
    import contextlib
    import functools
    import io

    iotcv = Path(spec["iotcv"])
    _iotcv_import(iotcv)
    import replay_bench  # type: ignore[import-not-found]

    real = replay_bench.VisionPipeline
    if spec.get("verifier_onnx"):
        from backend.vision.model_bundle import load_bundle, read_meta  # type: ignore[import-not-found]
        from backend.vision.verifier import BoxVerifier  # type: ignore[import-not-found]

        vp = Path(spec["verifier_onnx"])
        vb = load_bundle(vp, str(read_meta(vp)["onnx_sha256"]), "verifier")
        replay_bench.VisionPipeline = functools.partial(real, verifier=BoxVerifier(vb))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = replay_bench.main(["--recording", str(iotcv / "logs" / "recordings" / spec["rec"]), "--frames",
                                      str(spec.get("frames", 600)), "--json"])
    finally:
        replay_bench.VisionPipeline = real
    res = json.loads(buf.getvalue().strip().splitlines()[-1])
    return res | {"exit": code, "verifier_onnx": spec.get("verifier_onnx")}


def iotcv_main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    fn = {"cache": _iotcv_cache, "grid": _iotcv_grid, "bench": _iotcv_bench}[spec["mode"]]
    print(json.dumps(fn(spec), ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- bundle: chép sang IOT-CV, verifier từng fold (OOF)

RUNS_DEFAULT = Path("D:/DevCache/runs/xla/v2/system")


def install_bundle(iotcv: Path, bundle: Path, runner: Runner, log: Path | None = None) -> Path:
    """``scripts/vision/lay_model.py --from <bundle>`` (kiểm cấu trúc + sha, chép vào ``models/vision/``, gitignored)."""
    runner(IotcvCall(["uv", "run", "--no-sync", "python", "scripts/vision/lay_model.py", "--from", str(bundle), "--json"],
                     iotcv, iotcv_child_env(iotcv=iotcv), False, f"lay_model {bundle.name}", log))
    kind = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))["model"]
    return iotcv / "models" / "vision" / bundle.name / f"{kind}.onnx"


def export_fold_verifiers(runs: Path, root: Path) -> list[dict[str, Any]]:
    """``verifier.onnx`` §9 của từng model fold phase 07 (``<cand>_f<k>`` = huấn luyện KHÔNG có fold k) vào
    ``<runs>/verifier-fold<k>/`` kèm ``meta.json`` (meta bản ship, đổi sha + ``train_run``). Chỉ để đo recall OOF khi
    hiệu chỉnh; không phải bundle phát hành. Parity ORT CPU ↔ PyTorch trên golden của bundle ship."""
    import numpy as np

    from xla import contract as C
    from xla.verify import common as V
    from xla.verify import export_onnx as E
    from xla.verify import train as T

    cfg = V.load_cfg()
    c = cfg["train"]["candidates"][VERIFIER_CAND]
    ship = root / RELEASE_REL / VERIFIER_SYSTEM
    meta0 = json.loads((ship / "meta.json").read_text(encoding="utf-8"))
    if meta0["train_run"] != f"v2/verifier/{VERIFIER_CAND}_final":
        raise RuntimeError(f"verifier ship train_run {meta0['train_run']!r} không phải {VERIFIER_CAND}_final")
    gboxes = json.loads((ship / "golden" / C.BOXES_FILE).read_text(encoding="utf-8"))
    feeds = {k: (C.decode_frame(ship / "golden" / k), np.asarray(b, np.float32).reshape(-1, 4)) for k, b in gboxes.items()}
    out = []
    n_folds = len(json.loads((root / "data/real/labels/folds.json").read_text(encoding="utf-8"))["folds"])
    for k in range(n_folds):
        d = runs / f"verifier-fold{k}"
        onnx_path = d / "verifier.onnx"
        tag = f"{VERIFIER_CAND}_f{k}"
        if not (onnx_path.is_file() and (d / "meta.json").is_file()):
            d.mkdir(parents=True, exist_ok=True)
            module = E.graph_module(T.load_model(cfg, VERIFIER_CAND, tag), int(c["crop_px"]),
                                    float(cfg["crop"]["context_scale"]), float(cfg["crop"]["min_side_px"]))
            E.export(module, onnx_path)
            par = E.parity(module, onnx_path, feeds, cuda=False)
            if not par["cpu"]["pass"]:
                raise RuntimeError(f"{tag}: parity ORT CPU không đạt {par}")
            meta = dict(meta0) | {"onnx_sha256": C.sha256_file(onnx_path), "train_run": f"v2/verifier/{tag}",
                                  "oof_fold": k, "note": "chỉ hiệu chỉnh phase 09 (recall OOF), không phát hành",
                                  "export": dict(meta0.get("export", {})) | {"parity": par}}
            write_json(d / "meta.json", meta)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        out.append({"fold": k, "onnx": str(onnx_path), "onnx_sha256": meta["onnx_sha256"], "train_run": meta["train_run"],
                    "parity": meta["export"]["parity"]})
    return out


# ---------------------------------------------------------------- calibrate (bước 2)


@dataclass(frozen=True)
class Point:
    thr: float
    k: int
    n: int
    max_gap: int

    @property
    def key(self) -> str:
        return f"t{self.thr:.2f}_k{self.k}_n{self.n}_g{self.max_gap}"

    def track(self) -> dict[str, Any]:
        return {"k": self.k, "n": self.n, "iou": TRACK_IOU, "max_gap": self.max_gap}


def grid_points(thr: Iterable[float] = GRID_THR, ks: Iterable[int] = GRID_K, n_max: int = GRID_N_MAX,
                gaps: Iterable[int] = GRID_GAP) -> list[Point]:
    """k ∈ {1..4}, n ∈ {k..6}, max_gap ∈ {1, 2}, ngưỡng verifier bước 0,05 (phase 09 bước 2)."""
    return [Point(float(t), k, n, g) for t in thr for k in ks for n in range(k, n_max + 1) for g in gaps]


def _chunks(xs: list[Any], n: int) -> list[list[Any]]:
    n = max(1, min(n, len(xs)))
    return [xs[i::n] for i in range(n)]


def read_rows(path: Path) -> dict[str, int]:
    """Số hộp theo khung của một CSV D11."""
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            counts[r["frame"]] = counts.get(r["frame"], 0) + 1
    return counts


def merge_csv(parts: Iterable[Path], out: Path, keep: set[str] | None = None) -> Path:
    """Gộp CSV D11 (giữ thứ tự khung đã sắp, bỏ khung ngoài ``keep``)."""
    rows: list[list[str]] = []
    header = None
    for p in parts:
        with p.open(encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            h = next(rd)
            if header is None:
                header = h
            elif h != header:
                raise ValueError(f"{p}: header {h} khác {header}")
            rows += [r for r in rd if keep is None or r[0] in keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header or ["frame", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(sorted(rows, key=lambda r: r[0]))
    return out


def eval_point(args: tuple[str, str, list[str], list[str]]) -> dict[str, Any]:
    """(khoá điểm, CSV lượt verifier bản cuối trên cal, CSV từng fold, khung cal) → báo nhầm cal + recall devpool OOF.

    Chỉ đọc nhãn devpool (``prepare(split="devpool")``); cal không có người (agent soát từng khung, phase 03) nên mọi hộp
    trên khung cal là báo nhầm."""
    from xla.eval import detection as D

    key, final_csv, fold_csvs, cal = args
    counts = read_rows(Path(final_csv))
    cal_fp = int(sum(counts.get(f, 0) for f in cal))
    merged = Path(fold_csvs[0]).parent.parent / "merged" / f"{key}.csv"
    merge_csv([Path(p) for p in fold_csvs], merged)
    pr = D.prepare(merged, "devpool", thresholds=[0.0], f1_from_devval=False)
    evals = [pr.s.evals[f] for f in pr.frames]
    prf = D.prf_at(evals, 0.0)
    empty = [f for f in pr.frames if len(pr.gt[f]) == 0]
    return {"key": key, "cal_fp": cal_fp, "cal_fp_per_frame": cal_fp / len(cal), "tp": prf["tp"],
            "n_gt": prf["tp"] + prf["fn"], "recall": prf["recall"], "devpool_fp": prf["fp"],
            "devpool_fp_empty_frames": int(sum(pr.s.fp_count(0.0, f) for f in empty)), "n_cal": len(cal),
            "n_devpool": len(pr.frames), "n_devpool_empty": len(empty)}


def select_d7(rows: list[dict[str, Any]], cal_max: float = D7_CAL_MAX, tie_persons: int = TIE_PERSONS) -> dict[str, Any]:
    """D7: báo nhầm/khung cal ≤ ``cal_max``, trong đó recall OOF lớn nhất; vùng hoà = kém điểm tốt nhất ≤ ``tie_persons``
    người, chọn điểm đơn giản nhất (k, n, max_gap nhỏ nhất), rồi recall cao, báo nhầm cal thấp, ngưỡng cao."""
    feasible = [r for r in rows if r["cal_fp_per_frame"] <= cal_max + 1e-12]
    if not feasible:
        return {"chosen": None, "reason": f"không điểm nào có báo nhầm cal ≤ {cal_max}", "n_feasible": 0}
    best = max(r["tp"] for r in feasible)
    tie = [r for r in feasible if r["tp"] >= best - tie_persons]
    pick = min(tie, key=lambda r: (r["k"], r["n"], r["max_gap"], -r["tp"], r["cal_fp"], -r["thr"]))
    best_row = min((r for r in feasible if r["tp"] == best), key=lambda r: (r["cal_fp"], r["k"], r["n"], r["max_gap"],
                                                                             -r["thr"]))
    return {"chosen": pick, "best_recall_point": best_row, "n_feasible": len(feasible), "n_tie": len(tie),
            "rule": f"cal ≤ {cal_max} hộp/khung; recall OOF lớn nhất; vùng hoà ≤ {tie_persons} người; đơn giản nhất = "
                    "(k, n, max_gap) nhỏ nhất, rồi recall cao, báo nhầm cal thấp, ngưỡng cao"}


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Biên báo nhầm cal ↔ recall: điểm không bị điểm nào khác hơn ở cả hai chiều."""
    out, best_tp = [], -1
    for r in sorted(rows, key=lambda r: (r["cal_fp"], -r["tp"])):
        if r["tp"] > best_tp:
            out.append(r)
            best_tp = r["tp"]
    return out


def plot_curve(rows: list[dict[str, Any]], chosen: dict[str, Any] | None, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=120)
    ax.scatter([r["cal_fp_per_frame"] for r in rows], [r["recall"] for r in rows], s=8, c="#b8b8b8",
               label="mọi điểm lưới")
    for k, color in zip(GRID_K, ("#1f5fa8", "#d1792b", "#3a8f4b", "#8a3fa0"), strict=True):
        sub = pareto([r for r in rows if r["k"] == k])
        ax.plot([r["cal_fp_per_frame"] for r in sub], [r["recall"] for r in sub], "-o", ms=3, lw=1.4, color=color,
                label=f"biên k = {k}")
    ax.axvline(D7_CAL_MAX, color="#a33", ls="--", lw=1, label=f"D7: cal ≤ {D7_CAL_MAX}")
    if chosen:
        ax.scatter([chosen["cal_fp_per_frame"]], [chosen["recall"]], s=90, facecolors="none", edgecolors="black", lw=1.6,
                   label=f"chọn: ngưỡng {chosen['thr']:.2f}, k {chosen['k']}/n {chosen['n']}, max_gap {chosen['max_gap']}")
    ax.set_xlabel("báo nhầm / khung cal (150 khung, verifier bản cuối)")
    ax.set_ylabel("recall devpool (verifier OOF; detector v1 không OOF)")
    ax.set_title("Hiệu chỉnh cả hệ phase 09: detector v1 → verifier → track")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _map(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=8))


def _run_parallel(calls: list[IotcvCall], runner: Runner, workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(calls) <= 1:
        return [runner(c) for c in calls]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(runner, calls))


def _sha_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_pass(name: str, iotcv: Path, runs: Path, rec: str, frames_file: Path, blank_file: Path, det_onnx: Path,
                ver_onnx: Path, runner: Runner, log: Path) -> dict[str, Any]:
    """Lượt GPU thật ``--stage verifier`` + ghi đầu ra từng khung (chạy lại được: bỏ qua khi spec không đổi)."""
    spec = {"mode": "cache", "iotcv": str(iotcv), "cache": str(runs / "cache" / f"{name}.pkl"),
            "replay": {"rec": rec, "frames": str(frames_file), "stage": "verifier", "blank": str(blank_file),
                       "out": str(runs / "cache" / f"{name}_verifier.csv"), "detector_onnx": str(det_onnx),
                       "verifier_onnx": str(ver_onnx)}}
    key = _sha_text(json.dumps(spec, sort_keys=True) + frames_file.read_text("utf-8") + blank_file.read_text("utf-8"))
    done = runs / "cache" / f"{name}.done.json"
    if done.is_file() and Path(spec["cache"]).is_file():
        prev = json.loads(done.read_text(encoding="utf-8"))
        if prev.get("key") == key:
            return prev["result"] | {"resumed": True}
    spec_path = write_json(runs / "specs" / f"cache_{name}.json", spec)
    Path(spec["cache"]).parent.mkdir(parents=True, exist_ok=True)
    res = runner(IotcvCall(sim_cmd(spec_path), iotcv, iotcv_child_env(iotcv=iotcv), True, f"v2 phase09 cache {name}", log))
    write_json(done, {"key": key, "result": res})
    return res


def calibrate(iotcv: Path, runs: Path = RUNS_DEFAULT, runner: Runner = run_iotcv, points: list[Point] | None = None,
              root: Path | None = None, sets: FrameSets | None = None, workers: int = WORKERS,
              parity_n: int = PARITY_POINTS, fold_verifiers: list[dict[str, Any]] | None = None,
              out_dir: Path | None = None) -> dict[str, Any]:
    """Bước 2 phase 09. Không đọc nhãn test/holdout, không ghi meta. Kết quả: ``results/v2/system/calibrate.json``.

    Cả lượt hiệu chỉnh là MỘT việc GPU (3 lượt ghi + parity, lưới CPU ~2 phút ở giữa): giữ khoá D6 một lần, nhả ngay
    khi xong, để lane khác không chen 20+ phút vào giữa từng lượt."""
    with held_gpu("v2 phase09 calibrate") if runner is run_iotcv else contextlib.nullcontext():
        return _calibrate(iotcv, runs, runner, points, root, sets, workers, parity_n, fold_verifiers, out_dir)


def _calibrate(iotcv: Path, runs: Path, runner: Runner, points: list[Point] | None, root: Path | None,
               sets: FrameSets | None, workers: int, parity_n: int, fold_verifiers: list[dict[str, Any]] | None,
               out_dir: Path | None) -> dict[str, Any]:
    from xla.paths import repo_root

    root = root or repo_root()
    sets = sets or load_frame_sets(root)
    points = points or grid_points()
    log = runs / "calibrate.log"
    blank = calibrate_blank(sets)
    wanted = calibrate_frames(sets)
    check_disjoint(wanted, blank, "calibrate")
    recs = sorted({rec_of(f) for f in wanted})
    if len(recs) != 1:
        raise ValueError(f"calibrate: cal + devpool nằm trên {recs}; code này giả định một bản ghi")
    rec = recs[0]
    lists = runs / "lists"
    blank_file = write_list(lists / "calibrate_blank.txt", blank)
    frames_file = write_list(lists / "calibrate_frames.txt", wanted)
    # Lượt bản cuối ghi cả devpool (không dùng để chấm recall, chỉ để parity có hộp người thật, không xanh rỗng).
    pass_frames = {"final": write_list(lists / "pass_final.txt", sets.cal | sets.devpool)}
    for k, fold in enumerate(sets.folds):
        pass_frames[f"f{k}"] = write_list(lists / f"pass_f{k}.txt", fold)
    for f in pass_frames.values():
        check_disjoint(read_list(f), blank, f"calibrate {f.name}")

    det_onnx = install_bundle(iotcv, root / RELEASE_REL / DETECTOR_BASE, runner, log)
    ver_ship = install_bundle(iotcv, root / RELEASE_REL / VERIFIER_SYSTEM, runner, log)
    folds = fold_verifiers if fold_verifiers is not None else export_fold_verifiers(runs, root)
    verifiers = {"final": ver_ship} | {f"f{f['fold']}": Path(f["onnx"]) for f in folds}
    if set(verifiers) != set(pass_frames):
        raise RuntimeError(f"verifier từng fold {sorted(verifiers)} không khớp fold devpool {sorted(pass_frames)}")

    t0 = time.time()
    cache_res = {name: _cache_pass(name, iotcv, runs, rec, frames_file, blank_file, det_onnx, v, runner, log)
                 for name, v in verifiers.items()}
    gpu_s = sum(float(r.get("_wall_s", 0.0)) for r in cache_res.values() if not r.get("resumed"))

    calls, outs = [], {}
    for name in verifiers:
        gdir = runs / "grid" / name
        gdir.mkdir(parents=True, exist_ok=True)
        pts = [{"track": p.track(), "verifier_conf": p.thr, "out": str(gdir / f"{p.key}.csv")} for p in points]
        outs[name] = {p.key: gdir / f"{p.key}.csv" for p in points}
        for i, chunk in enumerate(_chunks(pts, workers)):
            spec = {"mode": "grid", "iotcv": str(iotcv), "cache": str(runs / "cache" / f"{name}.pkl"), "rec": rec,
                    "frames": str(pass_frames[name]), "blank": str(blank_file), "detector_onnx": str(det_onnx),
                    "verifier_onnx": str(verifiers[name]), "points": chunk}
            sp = write_json(runs / "specs" / f"grid_{name}_{i}.json", spec)
            calls.append(IotcvCall(sim_cmd(sp), iotcv, iotcv_child_env(iotcv=iotcv), False, f"grid {name} {i}", log))
    t1 = time.time()
    grid_res = _run_parallel(calls, runner, workers)
    grid_s = time.time() - t1
    replay_info = {}
    for r in grid_res:
        for p in r["points"]:
            replay_info[p["out"]] = p

    cal = sorted(sets.cal)
    fold_names = [f"f{k}" for k in range(len(sets.folds))]
    jobs = [(p.key, str(outs["final"][p.key]), [str(outs[n][p.key]) for n in fold_names], cal) for p in points]
    t2 = time.time()
    evals = {e["key"]: e for e in _map(eval_point, jobs, workers)}
    eval_s = time.time() - t2
    rows = [{"thr": p.thr, "k": p.k, "n": p.n, "max_gap": p.max_gap, "iou": TRACK_IOU} | evals[p.key] for p in points]
    sel = select_d7(rows)

    parity = check_parity(sel.get("chosen"), points, iotcv, runs, rec, pass_frames, blank_file, det_onnx, verifiers,
                          outs, runner, log, parity_n)
    n_gt = rows[0]["n_gt"] if rows else 0
    ref = {name: next((r for r in rows if (r["k"], r["n"], r["max_gap"], r["thr"]) == spec), None)
           for name, spec in (("detector_only", (1, 1, 1, 0.0)), ("verifier_phase07_0.35", (1, 1, 1, 0.35)),
                              ("verifier_phase07_0.40", (1, 1, 1, 0.4)))}
    doc = {
        "created": utc_now(), "phase": "09 bước 2 (calibrate)", "rule": "D7", "recording": rec,
        "detector": {"bundle": DETECTOR_BASE, "onnx": str(det_onnx),
                     "conf_threshold": cache_res["final"].get("bundles", {}).get("detector")},
        "verifiers": {"final": str(ver_ship)} | {f"f{f['fold']}": f for f in folds},
        "blank": {"n": len(blank), "test": len(sets.test & blank), "holdout": len(sets.holdout & blank),
                  "mine": len(sets.mine & blank), "cal_mode": sets.cal_mode, "sha256": _sha_text(blank_file.read_text("utf-8"))},
        "frames": {"cal": len(sets.cal), "devpool": len(sets.devpool), "folds": [len(f) for f in sets.folds],
                   "n_gt_devpool": n_gt},
        "oof": {"verifier": "OOF: khung devpool fold k chấm bằng verifier fold k (huấn luyện không có fold k); cal chấm "
                            "bằng verifier bản cuối (cái ship, cal chưa từng vào huấn luyện)",
                "detector": "KHÔNG OOF: detector-v1.0.0 (E2 r1) học nhãn giả của khung dev v1 (gồm phần lớn devpool) và "
                            "chọn checkpoint + ngưỡng trên dev-val v1 (86 khung trong devpool) — results/v2/detect/README.md "
                            "§4. Recall devpool vì vậy lạc quan; khung cal cũng là khung dev v1 detector v1 đã thấy"},
        "grid": {"thr": sorted({p.thr for p in points}), "k": sorted({p.k for p in points}),
                 "n": sorted({p.n for p in points}), "max_gap": sorted({p.max_gap for p in points}), "iou": TRACK_IOU,
                 "n_points": len(points), "passes": list(verifiers)},
        "method": "detector + verifier chạy thật một lượt GPU mỗi verifier (replay_pred.run, --stage verifier); mỗi điểm "
                  "lưới chạy lại replay_pred.run + _Replay.rows_for + TrackFilter của IOT-CV với đầu ra ONNX đã ghi; "
                  f"{len(parity)} điểm phát lại thật --stage track, so CSV từng byte (parity)",
        "cache": cache_res, "parity": parity,
        "timing_s": {"gpu_cache": round(gpu_s, 1), "grid_cpu": round(grid_s, 1), "eval_cpu": round(eval_s, 1),
                     "total": round(time.time() - t0, 1)},
        "selection": sel, "reference_points": ref, "pareto": pareto(rows), "rows": rows,
    }
    out_dir = out_dir or root / SYSTEM_REL
    write_json(out_dir / "calibrate.json", doc)
    plot_curve(rows, sel.get("chosen"), out_dir / "calibrate_curve.png")
    if not all(p["identical"] for p in parity):
        raise RuntimeError(f"parity mô phỏng ↔ replay_pred thật không đạt: {parity}")
    if parity and not any(p["rows_real"] for p in parity):
        raise RuntimeError("parity rỗng: mọi điểm so đều 0 hộp, không chứng minh được gì")
    return doc


def check_parity(chosen: dict[str, Any] | None, points: list[Point], iotcv: Path, runs: Path, rec: str,
                 pass_frames: dict[str, Path], blank_file: Path, det_onnx: Path, verifiers: dict[str, Path],
                 outs: dict[str, dict[str, Path]], runner: Runner, log: Path, n: int) -> list[dict[str, Any]]:
    """Phát lại thật ``replay_pred.py --stage track`` ở ``n`` điểm (điểm chọn trên cal + điểm phức tạp trên từng fold),
    CSV phải trùng từng byte với CSV mô phỏng cùng điểm."""
    by = {(p.thr, p.k, p.n, p.max_gap): p for p in points}
    want: list[tuple[str, Point]] = []
    if chosen is not None:
        want.append(("final", by[(chosen["thr"], chosen["k"], chosen["n"], chosen["max_gap"])]))
    extra = [p for p in points if p.k >= 2 and p.max_gap == max(q.max_gap for q in points)] or points
    for i, name in enumerate(x for x in verifiers if x != "final"):
        want.append((name, extra[(len(extra) * (i + 1)) // (len(verifiers) + 1)]))
    out = []
    for name, p in want[:n]:
        real = runs / "parity" / f"{name}_{p.key}.csv"
        cmd = replay_cmd(rec, pass_frames[name], "track", real, blank_file, det_onnx, verifiers[name], p.track(), p.thr)
        res = runner(IotcvCall(cmd, iotcv, iotcv_child_env(iotcv=iotcv), True, f"v2 phase09 parity {name} {p.key}", log))
        sim = outs[name][p.key]
        same = real.is_file() and sim.is_file() and real.read_bytes() == sim.read_bytes()
        out.append({"pass": name, "point": p.key, "identical": same, "real": str(real), "sim": str(sim),
                    "rows_real": max(0, len(real.read_text("utf-8").splitlines()) - 1) if real.is_file() else None,
                    "replay": {k: res.get(k) for k in ("frames_written", "frames_replayed", "frames_blank",
                                                        "track_resets", "track", "_wall_s")}})
    return out


# ---------------------------------------------------------------- apply (bước 3): meta


def apply(root: Path | None = None, calib: dict[str, Any] | None = None) -> dict[str, Any]:
    """``detector-v1.0.1`` = ONNX + golden của ``detector-v1.0.0`` (sha không đổi) + meta v1.0.0 thêm
    ``recommended_track``, ``contract_version`` "1.1", ``version`` "1.0.1" (bundle v1.0.0 đã phát hành không sửa);
    ``verifier-v1.0.0/meta.json`` chỉ đổi ``conf_threshold``. ``xla.contract validate`` cả hai."""
    from xla import contract as C
    from xla.paths import repo_root

    root = root or repo_root()
    calib = calib or json.loads((root / SYSTEM_REL / "calibrate.json").read_text(encoding="utf-8"))
    ch = calib["selection"]["chosen"]
    if ch is None:
        raise RuntimeError("calibrate không chọn được điểm vận hành (D7): không ghi meta")
    rel = root / RELEASE_REL
    src, dst = rel / DETECTOR_BASE, rel / DETECTOR_SYSTEM
    meta0 = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if C.sha256_file(src / "detector.onnx") != meta0["onnx_sha256"]:
        raise RuntimeError(f"{src}: sha256 detector.onnx khác meta")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "golden").mkdir(exist_ok=True)
    shutil.copyfile(src / "detector.onnx", dst / "detector.onnx")
    for g in sorted((src / "golden").iterdir()):
        shutil.copyfile(g, dst / "golden" / g.name)
    track = {"k": int(ch["k"]), "n": int(ch["n"]), "iou": float(ch["iou"]), "max_gap": int(ch["max_gap"])}
    meta = dict(meta0) | {"contract_version": "1.1", "version": "1.0.1", "recommended_track": track}
    write_json(dst / "meta.json", meta)
    vpath = rel / VERIFIER_SYSTEM / "meta.json"
    vmeta = json.loads(vpath.read_text(encoding="utf-8"))
    old = vmeta["conf_threshold"]
    vmeta["conf_threshold"] = float(ch["thr"])
    write_json(vpath, vmeta)
    errs = {DETECTOR_SYSTEM: C.validate_bundle(dst), VERIFIER_SYSTEM: C.validate_bundle(rel / VERIFIER_SYSTEM)}
    res = {"detector": {"bundle": str(dst), "recommended_track": track, "onnx_sha256": meta["onnx_sha256"]},
           "verifier": {"conf_threshold": {"before": old, "after": vmeta["conf_threshold"]}}, "validate": errs}
    if any(errs.values()):
        raise RuntimeError(f"xla.contract validate không đạt: {errs}")
    return res


# ---------------------------------------------------------------- score (bước 4): test + holdout, một lần


def _meta(root: Path, bundle: str) -> dict[str, Any]:
    return json.loads((root / RELEASE_REL / bundle / "meta.json").read_text(encoding="utf-8"))


def _replay_set(name: str, wanted: frozenset[str], blank_track: frozenset[str], iotcv: Path, runs: Path,
                det_onnx: Path, ver_onnx: Path, runner: Runner, log: Path, out_dir: Path,
                rec_frames: Callable[[Path, str], list[str]]) -> dict[str, Any]:
    """Ba tầng trên ``wanted``. Tầng track: ``--blank`` = ``blank_track`` (ngữ cảnh = mọi khung còn lại). Tầng detector /
    verifier không có trạng thái giữa khung, nên blank thêm mọi khung ngoài ``wanted`` (kết quả như nhau, nhanh hơn)."""
    info: dict[str, Any] = {}
    parts: dict[str, list[Path]] = {s: [] for s in STAGES}
    for rec in sorted({rec_of(f) for f in wanted}):
        w = frozenset(f for f in wanted if rec_of(f) == rec)
        check_disjoint(w, blank_track, f"{name} {rec}")
        frames_file = write_list(runs / "lists" / f"{name}_{rec}_frames.txt", w)
        all_frames = rec_frames(iotcv, rec)
        missing = sorted(w - set(all_frames))
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} khung không có trong {rec}/frames.csv (vd. {missing[:3]})")
        blanks = {"track": write_list(runs / "lists" / f"{name}_{rec}_blank_track.txt", blank_track),
                  "frame": write_list(runs / "lists" / f"{name}_{rec}_blank_frame.txt",
                                      (set(all_frames) - w) | set(blank_track))}
        for stage in STAGES:
            out = runs / name / f"{rec}_{stage}.csv"
            bf = blanks["track" if stage == "track" else "frame"]
            cmd = replay_cmd(rec, frames_file, stage, out, bf, det_onnx, ver_onnx)
            res = runner(IotcvCall(cmd, iotcv, iotcv_child_env(iotcv=iotcv), True, f"v2 phase09 score {name} {rec} {stage}", log))
            info[f"{rec}/{stage}"] = {k: res.get(k) for k in ("frames_written", "frames_replayed", "frames_blank",
                                                               "track_resets", "track", "bundles", "_wall_s")}
            parts[stage].append(out)
    csvs = {s: merge_csv(parts[s], out_dir / f"system_{s}_{name}.csv", set(wanted)) for s in STAGES} if wanted else {}
    return {"replays": info, "csv": {s: str(p) for s, p in csvs.items()}}


def holdout_report(csvs: dict[str, str], s: FrameSets, boot_n: int = BOOT_N, seed: int = 20261125) -> dict[str, Any]:
    """B2: báo nhầm/khung trên khung holdout đã duyệt (không người), theo tầng, phiên, ngày; CI bootstrap theo phiên."""
    import numpy as np

    frames = sorted(s.holdout_approved)
    if not frames:
        return {"n_frames": 0, "n_sessions": 0, "verdict": "không đủ chứng minh (0 khung)",
                "note": "Kill (a): holdout rỗng (mọi phiên dính khung test v1, phase 03); không lấy khung khác làm holdout, "
                        "không dùng cal để báo B2", "by_stage": {s_: None for s_ in STAGES}}
    by_stage: dict[str, Any] = {}
    sess = sorted({s.session_of[f] for f in frames})
    rng = np.random.default_rng(seed)
    for stage in STAGES:
        counts = read_rows(Path(csvs[stage]))
        per = np.asarray([counts.get(f, 0) for f in frames], float)
        idx = {x: [i for i, f in enumerate(frames) if s.session_of[f] == x] for x in sess}
        boots = []
        for _ in range(boot_n):
            pick = rng.choice(sess, size=len(sess), replace=True)
            ii = [i for x in pick for i in idx[x]]
            boots.append(float(per[ii].mean()) if ii else float("nan"))
        ci = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))] if len(sess) > 1 else [None, None]
        days = sorted({s.day_of[f] for f in frames})
        by_stage[stage] = {"fp": int(per.sum()), "fp_per_frame": float(per.mean()), "ci95": ci,
                           "by_session": {x: float(per[idx[x]].mean()) for x in sess},
                           "by_day": {d: float(per[[i for i, f in enumerate(frames) if s.day_of[f] == d]].mean())
                                      for d in days}}
    fp23 = by_stage["track"]["fp_per_frame"]
    enough = len(frames) >= 150
    return {"n_frames": len(frames), "n_sessions": len(sess), "by_stage": by_stage,
            "verdict": ("đạt" if fp23 <= B2_MAX else "không đạt") + ("" if enough else " (không đủ chứng minh: < 150 khung)"),
            "ci_unit": "phiên bay (bootstrap theo cụm)"}


def _pick(r: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    m = r["metrics"]
    return {k: m[k] for k in keys if k in m}


def score(iotcv: Path, runs: Path = RUNS_DEFAULT, runner: Runner = run_iotcv, root: Path | None = None,
          sets: FrameSets | None = None, boot_n: int = BOOT_N,
          rec_frames: Callable[[Path, str], list[str]] = recording_frames, out_dir: Path | None = None) -> dict[str, Any]:
    """Bước 4 phase 09, một lần. Chỉ ĐỌC meta (ngưỡng, k/n lấy từ meta qua IOT-CV: không truyền ``--track-*``)."""
    from xla.eval import detection as D
    from xla.paths import repo_root

    root = root or repo_root()
    sets = sets or load_frame_sets(root)
    log = runs / "score.log"
    det_meta, ver_meta = _meta(root, DETECTOR_SYSTEM), _meta(root, VERIFIER_SYSTEM)
    if "recommended_track" not in det_meta:
        raise RuntimeError(f"{DETECTOR_SYSTEM}: thiếu recommended_track (chạy apply trước)")
    det_onnx = install_bundle(iotcv, root / RELEASE_REL / DETECTOR_SYSTEM, runner, log)
    ver_onnx = install_bundle(iotcv, root / RELEASE_REL / VERIFIER_SYSTEM, runner, log)
    res_dir = out_dir or root / SYSTEM_REL
    out_dir = res_dir / "preds"
    test_wanted = frozenset(sets.test | sets.neg_indoor)
    # Một việc GPU (mọi lượt phát lại của test + holdout) = một lần giữ khoá D6, nhả ngay sau.
    with held_gpu("v2 phase09 score") if runner is run_iotcv else contextlib.nullcontext():
        test = _replay_set("test", test_wanted, score_test_blank(sets), iotcv, runs, det_onnx, ver_onnx, runner, log,
                           out_dir, rec_frames)
        hold = (_replay_set("holdout", frozenset(sets.holdout_approved), score_holdout_blank(sets), iotcv, runs,
                            det_onnx, ver_onnx, runner, log, out_dir, rec_frames)
                if sets.holdout_approved else {"replays": {}, "csv": {}})

    ops = {"detector": float(det_meta["conf_threshold"]), "verifier": float(ver_meta["conf_threshold"]),
           "track": float(ver_meta["conf_threshold"])}
    base = {"edge16": (root / EDGE16_CSV_REL, None), "v1": (root / V1_TEST_CSV_REL, float(_meta(root, DETECTOR_BASE)[
        "conf_threshold"]))}
    keys = ("mAP50", "mAP50_95", "AP_small", "AR_small", "precision@{op}", "recall@{op}",
            "fp_per_neg_flight@{op}", "fp_per_neg_indoor@{op}")
    stages: dict[str, Any] = {}
    for st in STAGES:
        p = Path(test["csv"][st])
        ev = D.evaluate(p, "test", thresholds=[ops[st]], boot_n=boot_n, f1_from_devval=False)
        op = f"{ops[st]:g}"
        stages[st] = {"csv": (str(p.relative_to(root)) if p.is_relative_to(root) else str(p)).replace("\\", "/"), "op_threshold": ops[st],
                      "metrics": _pick(ev, [k.replace("{op}", op) for k in keys]), "n": ev["n"],
                      "official": ev["official"]}
        cmp = {}
        for name, (bp, thr) in base.items():
            c = D.compare(bp, p, "test", thr_a=thr, thr_b=ops[st], boot_n=boot_n)
            cmp[name] = {"op_threshold": c["op_threshold"], "metrics": c["metrics"]}
        stages[st]["compare"] = cmp
    b1 = stages["verifier"]
    doc = {
        "created": utc_now(), "phase": "09 bước 4 (score, một lần)",
        "bundles": {"detector": {"bundle": DETECTOR_SYSTEM, "onnx_sha256": det_meta["onnx_sha256"],
                                 "recommended_track": det_meta["recommended_track"],
                                 "conf_threshold": det_meta["conf_threshold"]},
                    "verifier": {"bundle": VERIFIER_SYSTEM, "onnx_sha256": ver_meta["onnx_sha256"],
                                 "conf_threshold": ver_meta["conf_threshold"]}},
        "test": {"blank_track": {"holdout": len(sets.holdout), "cal": len(sets.cal)}, "replays": test["replays"],
                 "stages": stages},
        "holdout": holdout_report(hold["csv"], sets, boot_n) | {"replays": hold["replays"],
                                                                "blank": "test + cal + mine"},
        "b1": {"map50_system_verifier": b1["metrics"]["mAP50"], "target": 0.30, "min": 0.227,
               "vs_edge16_map50": b1["compare"]["edge16"]["metrics"]["mAP50"]},
    }
    write_json(res_dir / "score.json", doc)
    return doc


# ---------------------------------------------------------------- bench (bước 5): ms từng bước v1 vs cả hệ


BENCH_REC = "20261102-100941"
MS_CEILING = 5.0  # cả hệ được chậm hơn v1 tối đa +5 ms p95 (brainstorm §5)
ENHANCER_BASE = "enhancer-v1.0.0"


def bench(iotcv: Path, runs: Path = RUNS_DEFAULT, runner: Runner = run_iotcv, root: Path | None = None,
          frames: int = 600) -> dict[str, Any]:
    """``replay_bench.py`` của IOT-CV trên ``logs/recordings/20261102-100941``: v1 (detector-v1.0.0 + enhancer) rồi cả hệ
    (detector-v1.0.1 + verifier + enhancer), cùng đoạn khung, nối tiếp, mỗi lượt giữ khoá GPU. Ghi khối ``machine``."""
    from xla.detect.selftrain import machine_info
    from xla.paths import repo_root

    root = root or repo_root()
    log = runs / "bench.log"
    enh = install_bundle(iotcv, root / RELEASE_REL / ENHANCER_BASE, runner, log)
    det_v1 = install_bundle(iotcv, root / RELEASE_REL / DETECTOR_BASE, runner, log)
    det_sys = install_bundle(iotcv, root / RELEASE_REL / DETECTOR_SYSTEM, runner, log)
    ver = install_bundle(iotcv, root / RELEASE_REL / VERIFIER_SYSTEM, runner, log)
    out: dict[str, Any] = {"created": utc_now(), "recording": BENCH_REC, "frames": frames}
    with held_gpu("v2 phase09 bench") if runner is run_iotcv else contextlib.nullcontext():
        for name, det, v in (("v1", det_v1, None), ("system", det_sys, ver)):
            spec = {"mode": "bench", "iotcv": str(iotcv), "rec": BENCH_REC, "frames": frames,
                    "verifier_onnx": None if v is None else str(v)}
            sp = write_json(runs / "specs" / f"bench_{name}.json", spec)
            env = iotcv_child_env(iotcv=iotcv, extra={"DETECTOR_ONNX": str(det), "DETECTOR_SHA256": "",
                                                      "ENHANCER_ONNX": str(enh), "ENHANCER_SHA256": ""})
            out[f"machine_{name}"] = machine_info()
            out[name] = runner(IotcvCall(sim_cmd(sp), iotcv, env, True, f"v2 phase09 bench {name}", log))
    rows = {}
    for mode in ("realtime", "realtime_detector_only", "maxrate"):
        a, b = out["v1"][mode]["ms"], out["system"][mode]["ms"]
        rows[mode] = {step: {"v1_p95": a.get(step, {}).get("p95"), "system_p95": b.get(step, {}).get("p95"),
                             "v1_p50": a.get(step, {}).get("p50"), "system_p50": b.get(step, {}).get("p50")}
                      for step in sorted(set(a) | set(b))}
    tot = {mode: (rows[mode].get("total", {}).get("system_p95"), rows[mode].get("total", {}).get("v1_p95"))
           for mode in ("realtime", "realtime_detector_only")}
    out["steps"] = rows
    out["ceiling"] = {mode: {"v1_p95": v1, "system_p95": sy, "delta_ms": None if None in (v1, sy) else round(sy - v1, 2),
                             "max_delta_ms": MS_CEILING,
                             "pass": None if None in (v1, sy) else (sy - v1) <= MS_CEILING}
                      for mode, (sy, v1) in tot.items()}
    write_json(root / SYSTEM_REL / "bench.json", out)
    return out


# ---------------------------------------------------------------- CLI


def _emit(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, default=str))
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=1, default=str)[:4000])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_iotcv"]:
        return iotcv_main(argv[1])
    ap = argparse.ArgumentParser(prog="python -m xla.eval.system")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "apply", "score", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--iotcv", type=Path, default=None, help="gốc worktree IOT-CV (mặc định xla.paths.iotcv_root)")
        p.add_argument("--runs", type=Path, default=RUNS_DEFAULT)
        p.add_argument("--workers", type=int, default=WORKERS)
        p.add_argument("--boot", type=int, default=BOOT_N)
        p.add_argument("--json", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if a.workers > WORKERS:
        print(f"--workers tối đa {WORKERS} (luật máy)", file=sys.stderr)
        return 2
    from xla.paths import iotcv_root

    iotcv = (a.iotcv or iotcv_root()).resolve()
    try:
        if a.cmd == "calibrate":
            d = calibrate(iotcv, a.runs, workers=a.workers)
            res = {"chosen": d["selection"]["chosen"], "parity": d["parity"], "timing_s": d["timing_s"]}
        elif a.cmd == "apply":
            res = apply()
        elif a.cmd == "score":
            d = score(iotcv, a.runs, boot_n=a.boot)
            res = {"b1": d["b1"], "holdout": {k: d["holdout"][k] for k in ("n_frames", "verdict")},
                   "track": d["test"]["stages"]["track"]["metrics"]}
        else:
            d = bench(iotcv, a.runs)
            res = {"ceiling": d["ceiling"]}
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        _emit({"error": f"{type(exc).__name__}: {exc}"}, a.json)
        return 1
    _emit(res, a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
