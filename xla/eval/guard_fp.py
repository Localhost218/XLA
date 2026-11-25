"""Hàng rào báo nhầm của ảnh rõ offline (AI v2 phase 14 bước 1, Kill (d)).

AX::

    python -m xla.eval.guard_fp --bundle models/release/enhancer-offline-v1.0.0 --json [--write-meta]
    # exit 0 đo xong (kể cả khi bị loại: đọc "decision"); 1 dữ liệu / bundle sai; 2 sai cách gọi

Detector (mặc định ``detector-v1.0.1``, cùng ``conf_threshold`` của meta) chạy trên khung ``cal`` gốc và trên ``cal`` sau
enhancer offline (ảnh ra 1280×960 hạ ×½ ``INTER_AREA`` về 640×480). Báo nhầm/khung = số hộp ``score ≥ conf_threshold``
chia số khung (``cal`` không có người). CI 95 % của **hiệu** (offline − gốc) bằng bootstrap ghép cặp theo khối: trong mỗi
phiên, khung xếp theo tên rồi chia khối ``--block-frames`` khung liền nhau, lấy lại khối có hoàn lại trong từng phiên.

Quyết định chỉ dựa trên ``cal``: cận dưới CI hiệu > 0 → ``reject`` (không vào A/B, không ship). Holdout chỉ **ghi**
(``fp_holdout_*``); hiện holdout 0 khung → ghi "0 khung / không đủ chứng minh", không đọc khung holdout nào.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla.paths import repo_root

NEG_SPLIT_REL = Path("data/real/neg_flight_split.json")
FRAMES_REL = Path("data/real/frames")
DETECTOR_REL = Path("models/release/detector-v1.0.1")
OUT_REL = Path("results/ab-v2/guard_fp.json")
SEED = 20261125
N_BOOT = 10000
BLOCK_FRAMES = 10
LOCK_TIMEOUT_S = 6 * 3600.0
ORT_THREADS = 4
HOLDOUT_EMPTY = "0 khung / không đủ chứng minh"
RULE = ("Chỉ dựa trên cal: cận dưới CI 95 % (bootstrap ghép cặp theo khối trong phiên) của hiệu báo nhầm/khung "
        "(offline − gốc) > 0 → reject (Kill (d): không vào A/B, không ship); còn lại → pass. Holdout chỉ ghi.")


class GuardError(Exception):
    """Dữ liệu / bundle sai → exit 1."""


# ---------------------------------------------------------------- phần thuần (test không cần GPU)


def split_frames(neg_doc: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Khung của một tập trong ``neg_flight_split.json``, xếp theo tên."""
    return sorted((f for f in neg_doc["frames"] if f["split"] == split), key=lambda f: f["frame"])


def block_ids(frames: Sequence[dict[str, Any]], block_frames: int) -> list[str]:
    """Mã khối ``<phiên>/<k>``: trong mỗi phiên, khung xếp theo tên, cứ ``block_frames`` khung liền nhau một khối."""
    if block_frames < 1:
        raise ValueError("block_frames phải ≥ 1")
    seen: dict[str, int] = {}
    out = []
    for f in sorted(frames, key=lambda f: f["frame"]):
        i = seen.get(f["session"], 0)
        seen[f["session"]] = i + 1
        out.append(f"{f['session']}/{i // block_frames}")
    return out


def paired_block_bootstrap(raw: Sequence[int], off: Sequence[int], blocks: Sequence[str], *, n_boot: int = N_BOOT,
                           seed: int = SEED) -> dict[str, Any]:
    """CI 95 % (phân vị) cho hiệu trung bình ``off − raw`` báo nhầm/khung; lấy lại khối trong từng phiên."""
    raw_a, off_a = np.asarray(raw, float), np.asarray(off, float)
    if not (len(raw_a) == len(off_a) == len(blocks)) or len(raw_a) == 0:
        raise ValueError("raw, off, blocks phải cùng độ dài > 0")
    by_sess: dict[str, dict[str, list[int]]] = {}
    for i, b in enumerate(blocks):
        by_sess.setdefault(b.split("/")[0], {}).setdefault(b, []).append(i)
    groups = [[np.asarray(ix) for ix in d.values()] for d in by_sess.values()]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        idx = np.concatenate([g[j] for g in groups for j in rng.integers(0, len(g), len(g))])
        diffs[t] = off_a[idx].mean() - raw_a[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(off_a.mean() - raw_a.mean()), "ci95": [float(lo), float(hi)], "n_boot": n_boot,
            "n_blocks": len(set(blocks)), "n_sessions": len(by_sess), "seed": seed}


def decide(ci95: Sequence[float]) -> str:
    """``reject`` khi cận dưới CI hiệu > 0 (offline tăng báo nhầm có ý nghĩa), ngược lại ``pass``."""
    return "reject" if ci95[0] > 0 else "pass"


def fp_summary(counts: Sequence[int]) -> dict[str, Any]:
    n = len(counts)
    return {"n_frames": n, "n_boxes": int(sum(counts)), "fp_per_frame": float(sum(counts)) / n if n else None,
            "frames_with_fp": int(sum(c > 0 for c in counts))}


def evaluate(cal: Sequence[dict[str, Any]], count_raw: Callable[[str], int], count_off: Callable[[str], int], *,
             block_frames: int = BLOCK_FRAMES, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Đo + quyết định **chỉ** trên ``cal`` (danh sách khung ``{"frame","session"}``). Hai hàm đếm trả số hộp
    ``score ≥ conf`` của một khung (gốc / sau offline). Không nhận holdout: quyết định không thể phụ thuộc holdout."""
    if not cal:
        raise GuardError("cal không có khung nào")
    names = [f["frame"] for f in sorted(cal, key=lambda f: f["frame"])]
    raw = [int(count_raw(n)) for n in names]
    off = [int(count_off(n)) for n in names]
    boot = paired_block_bootstrap(raw, off, block_ids(cal, block_frames), n_boot=n_boot, seed=seed)
    return {"cal_raw": fp_summary(raw), "cal_offline": fp_summary(off), "diff": boot, "block_frames": block_frames,
            "decision": decide(boot["ci95"]), "rule": RULE,
            "per_frame": [{"frame": n, "raw": r, "offline": o} for n, r, o in zip(names, raw, off, strict=True)]}


def holdout_report(holdout: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Phần holdout (chỉ ghi). Hiện chưa có khung holdout nào: trả trạng thái "không đủ chứng minh"."""
    if not holdout:
        return {"n_frames": 0, "status": HOLDOUT_EMPTY, "fp_holdout_raw": None, "fp_holdout_offline": None}
    raise GuardError(f"holdout có {len(holdout)} khung: phép đo holdout chưa được bật trong phase này (chỉ báo cáo); "
                     "chạy lại khi holdout đã duyệt")


def meta_metrics(res: dict[str, Any]) -> dict[str, Any]:
    """Khối ``metrics`` của meta.json (hợp đồng §11): bốn khoá đã biết + chi tiết CI / trạng thái holdout."""
    return {"fp_cal_raw": res["cal_raw"]["fp_per_frame"], "fp_cal_offline": res["cal_offline"]["fp_per_frame"],
            "fp_holdout_raw": res["holdout"]["fp_holdout_raw"],
            "fp_holdout_offline": res["holdout"]["fp_holdout_offline"],
            "fp_cal_diff": res["diff"]["diff"], "fp_cal_diff_ci95": res["diff"]["ci95"],
            "fp_cal_n_frames": res["cal_raw"]["n_frames"], "fp_holdout_n_frames": res["holdout"]["n_frames"],
            "fp_holdout_status": res["holdout"].get("status"), "guard_decision": res["decision"],
            "guard_detector": res["detector"]["version"], "guard_conf_threshold": res["detector"]["conf_threshold"],
            "guard_measured": res["created"]}


# ---------------------------------------------------------------- ORT GPU


def add_nvidia_path() -> list[str]:
    """Đưa ``site-packages/nvidia/*/bin`` vào PATH (ORT GPU trên Windows cần, ``preload_dlls`` chưa đủ)."""
    import importlib.util
    import os

    spec = importlib.util.find_spec("nvidia")
    dirs: list[str] = []
    for base in (spec.submodule_search_locations or []) if spec else []:
        dirs += [str(p) for p in sorted(Path(base).glob("*/bin")) if p.is_dir()]
    cur = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([d for d in dirs if d not in cur] + cur)
    return dirs


def ort_cuda(onnx_path: Path) -> Any:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = ORT_THREADS
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def require_cuda(sess: Any, what: str) -> list[str]:
    """Đọc provider SAU lần chạy đầu (thiếu cuDNN thì ORT vẫn báo CUDA lúc tạo phiên nhưng chạy CPU)."""
    prov = list(sess.get_providers())
    if prov[0] != "CUDAExecutionProvider":
        raise GuardError(f"{what} không chạy CUDA EP: {prov}")
    return prov


def open_bundle(path: Path, kind: str) -> tuple[dict[str, Any], Path]:
    import hashlib

    meta_p, onnx_p = path / "meta.json", path / f"{kind}.onnx"
    if not meta_p.is_file() or not onnx_p.is_file():
        raise GuardError(f"bundle {path} thiếu meta.json hoặc {kind}.onnx")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("model") != kind:
        raise GuardError(f"{path}: meta.model = {meta.get('model')!r}, cần {kind!r}")
    h = hashlib.sha256(onnx_p.read_bytes()).hexdigest()
    if h != meta.get("onnx_sha256"):
        raise GuardError(f"{onnx_p}: sha256 {h[:12]}… khác meta.onnx_sha256")
    return meta, onnx_p


def machine_info() -> dict[str, Any]:
    import platform
    import subprocess

    out: dict[str, Any] = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "host": platform.node()}
    q = "name,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,pstate"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10, check=True)
        out["gpu"] = dict(zip(q.split(","), [s.strip() for s in r.stdout.strip().split(",")], strict=False))
    except (OSError, subprocess.SubprocessError) as e:
        out["gpu_error"] = str(e)
    return out


def run(bundle: Path, detector: Path, *, block_frames: int, n_boot: int, seed: int,
        lock_timeout: float) -> dict[str, Any]:
    import cv2

    from xla.contract import decode_frame
    from xla.gpu_lock import gpu_lock

    cv2.setNumThreads(ORT_THREADS)
    root = repo_root()
    neg = json.loads((root / NEG_SPLIT_REL).read_text(encoding="utf-8"))
    cal = split_frames(neg, "cal")
    holdout = split_frames(neg, "holdout")
    off_meta, off_onnx = open_bundle(bundle, "enhancer-offline")
    det_meta, det_onnx = open_bundle(detector, "detector")
    conf = float(det_meta["conf_threshold"])
    w, h = off_meta["frame_size"]
    fdir = root / FRAMES_REL
    frames = {f["frame"]: decode_frame(fdir / f["frame"]) for f in cal}
    nv = add_nvidia_path()
    t_wait = time.perf_counter()
    with gpu_lock("v2 phase14 guard_fp", timeout_s=lock_timeout):
        t0 = time.perf_counter()
        det, off = ort_cuda(det_onnx), ort_cuda(off_onnx)

        def count(img: np.ndarray) -> int:
            _b, s = det.run(["boxes", "scores"], {"frame": img})
            return int((s >= conf).sum())

        def enhance(img: np.ndarray) -> np.ndarray:
            out = off.run(["frame_out"], {"frame": img})[0]
            if out.shape != (2 * h, 2 * w, 3):
                raise GuardError(f"enhancer-offline ra shape {out.shape}, cần {(2 * h, 2 * w, 3)}")
            return np.ascontiguousarray(cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA))

        down = {n: enhance(f) for n, f in frames.items()}
        prov_off = require_cuda(off, "enhancer-offline")
        res = evaluate(cal, lambda n: count(frames[n]), lambda n: count(down[n]),
                       block_frames=block_frames, n_boot=n_boot, seed=seed)
        prov_det = require_cuda(det, "detector")
        gpu_s = time.perf_counter() - t0

    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)

    return res | {
        "created": datetime.now(UTC).isoformat(timespec="seconds"), "seed": seed,
        "bundle": {"path": rel(bundle), "version": off_meta["version"], "onnx_sha256": off_meta["onnx_sha256"],
                   "providers": prov_off},
        "detector": {"path": rel(detector), "version": det_meta["version"], "onnx_sha256": det_meta["onnx_sha256"],
                     "conf_threshold": conf, "providers": prov_det},
        "measured_on": "cal (neg_flight_split.json), ảnh offline 1280×960 hạ ×½ INTER_AREA về 640×480",
        "holdout": holdout_report(holdout), "gpu_seconds": round(gpu_s, 1),
        "wait_lock_seconds": round(t0 - t_wait, 1), "nvidia_path_dirs": len(nv), "machine": machine_info()}


def write_meta(bundle: Path, res: dict[str, Any]) -> None:
    """Ghi **chỉ** khoá ``metrics`` của meta.json bundle (giữ nguyên mọi khoá khác)."""
    p = bundle / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["metrics"] = meta_metrics(res)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.eval.guard_fp", description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, required=True, help="thư mục enhancer-offline-v*")
    ap.add_argument("--detector", type=Path, default=None, help=f"mặc định {DETECTOR_REL}")
    ap.add_argument("--block-frames", type=int, default=BLOCK_FRAMES)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=None, help=f"mặc định {OUT_REL}")
    ap.add_argument("--write-meta", action="store_true", help="ghi metrics vào meta.json của bundle")
    ap.add_argument("--lock-timeout", type=float, default=LOCK_TIMEOUT_S)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    root = repo_root()
    bundle = a.bundle if a.bundle.is_absolute() else root / a.bundle
    det = a.detector or root / DETECTOR_REL
    det = det if det.is_absolute() else root / det
    if a.block_frames < 1 or a.boot < 100:
        print("LỖI cách gọi: --block-frames ≥ 1, --boot ≥ 100", file=sys.stderr)
        return 2
    try:
        res = run(bundle, det, block_frames=a.block_frames, n_boot=a.boot, seed=a.seed, lock_timeout=a.lock_timeout)
    except (GuardError, TimeoutError, ValueError, KeyError, OSError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    out = a.out or root / OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    if a.write_meta:
        write_meta(bundle, res)
    brief = {k: v for k, v in res.items() if k != "per_frame"} | {"out": str(out), "meta_written": a.write_meta}
    print(json.dumps(brief, ensure_ascii=False) if a.json else json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
