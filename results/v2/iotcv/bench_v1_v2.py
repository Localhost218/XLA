"""Bench ms trực tiếp v1 vs v2 của IOT-CV, XEN KẼ trong MỘT tiến trình GPU (phase 15 plan 261125-ai-v2).

Chạy bằng venv của IOT-CV (không phải venv XLA), từ gốc worktree IOT-CV:

    UV_PROJECT_ENVIRONMENT=D:/Coding/IOT-CV/.venv uv run --no-sync python \
        D:/XLA/results/v2/iotcv/bench_v1_v2.py --iotcv D:/IOTCV-wt/iotcv \
        --recording D:/Coding/IOT-CV/logs/recordings/20261102-100941 \
        --out D:/XLA/results/v2/iotcv/bench.json

v1 = detector-v1.0.0 + enhancer-v1.0.0 (không verifier, track k = n = DETECTION_MIN_FRAMES như trước 25/11);
v2 = detector-v1.0.1 + verifier-v1.0.0 + enhancer-v2.0.0 (track theo `recommended_track` k1/n1). Nạp cả năm
bundle một lần, rồi chạy các lượt theo thứ tự ABBA (v1, v2, v2, v1) để trôi nhiệt/xung chia đều cho hai bên.
Mỗi lượt: warmup 30 khung, rồi `replay_bench._realtime` (nhịp thật của bản ghi) có người xem "Ảnh rõ" và chỉ
detector, cùng đoạn khung 284 → 884 như phase 09. Một thread đọc `nvidia-smi` mỗi 2 s suốt lượt (xung, nhiệt,
công suất, pstate) để số ms đọc kèm điều kiện máy. Không giữ khoá GPU của XLA; từ chối chạy khi khoá đang bị giữ.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Self

LOCK = Path("D:/DevCache/locks/gpu.lock")
RELEASE = Path("D:/XLA/models/release")
START, FRAMES, WARMUP = 284, 600, 30
ORDER = ("v1", "v2", "v2", "v1")
PHASE09_SYSTEM_P95 = 173.3  # results/v2/system/README.md §6, trước khi sửa verifier theo bậc
CEILING_MS = 5.0
SMI_Q = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,pstate,utilization.gpu"


def smi() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_Q}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        clk, clk_max, temp, power, pstate, util = (x.strip() for x in out.split(","))
        return {
            "clock_mhz": float(clk),
            "clock_max_mhz": float(clk_max),
            "temp_c": float(temp),
            "power_w": float(power),
            "pstate": pstate,
            "util_pct": float(util),
        }
    except (OSError, ValueError):
        return None


class Sampler:
    """Đọc nvidia-smi mỗi `period` giây tới khi `stop()`; trả min/median/max của xung, nhiệt, công suất."""

    def __init__(self, period: float = 2.0) -> None:
        self.period, self.rows, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            r = smi()
            if r:
                self.rows.append(r)
            self._stop.wait(self.period)

    def __enter__(self) -> Self:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=15)

    def summary(self) -> dict:
        if not self.rows:
            return {"samples": 0}
        out: dict = {
            "samples": len(self.rows),
            "clock_max_mhz": self.rows[0]["clock_max_mhz"],
            "pstates": sorted({r["pstate"] for r in self.rows}),
        }
        for k in ("clock_mhz", "temp_c", "power_w", "util_pct"):
            v = [r[k] for r in self.rows]
            out[k] = {"min": min(v), "median": statistics.median(v), "max": max(v)}
        return out


def pct(vals: list[float], q: float) -> float | None:
    """Cùng cách lấy phân vị với `replay_bench._summary` (chỉ số int(len·q) của dãy đã xếp)."""
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(len(s) * q))], 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iotcv", type=Path, required=True)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if LOCK.is_file() and not LOCK.read_text(encoding="utf-8").strip().startswith("free"):
        print(f"khoá GPU của XLA đang bị giữ: {LOCK.read_text(encoding='utf-8').strip()}", file=sys.stderr)
        return 3
    sys.path[:0] = [str(a.iotcv), str(a.iotcv / "scripts" / "vision")]
    import replay_bench as rb
    from backend.vision.detector import PersonDetector
    from backend.vision.enhancer import Enhancer
    from backend.vision.model_bundle import load_bundle, read_meta
    from backend.vision.pipeline import VisionPipeline
    from backend.vision.verifier import BoxVerifier
    from kiem_bundle import check as check_bundle

    vis = a.iotcv / "models" / "vision"

    def bundle(name: str, kind: str):
        p = vis / name / f"{kind}.onnx"
        return load_bundle(p, str(read_meta(p)["onnx_sha256"]), kind)

    names = {
        "v1": ("detector-v1.0.0", None, "enhancer-v1.0.0"),
        "v2": ("detector-v1.0.1", "verifier-v1.0.0", "enhancer-v2.0.0"),
    }
    pipes, info = {}, {}
    for cfg, (det, ver, enh) in names.items():
        db, eb = bundle(det, "detector"), bundle(enh, "enhancer")
        vb = bundle(ver, "verifier") if ver else None
        pipes[cfg] = VisionPipeline(
            rb._NullReader(),
            PersonDetector(db),
            Enhancer(eb),
            verifier=BoxVerifier(vb) if vb else None,
            timing_window=100_000,
        )
        info[cfg] = {
            k: {"bundle": n, "version": b.version, "onnx_sha256": b.meta["onnx_sha256"], "providers": b.providers}
            for k, n, b in (("detector", det, db), ("verifier", ver, vb), ("enhancer", enh, eb))
            if b
        }
    frames = [
        r.frame for r in rb.replay_frames(a.recording, range(START, START + WARMUP + FRAMES)) if r.frame is not None
    ]
    warm, body = frames[:WARMUP], frames[WARMUP:]
    runs, pooled = [], {cfg: {"realtime": {}, "realtime_detector_only": {}} for cfg in names}
    errors = {cfg: 0 for cfg in names}
    machine_start = smi()
    for i, cfg in enumerate(ORDER):
        pipe = pipes[cfg]
        row: dict = {"round": i, "config": cfg, "machine_before": smi()}
        with Sampler() as s:
            pipe.acquire_enhanced()  # có người xem "Ảnh rõ": trường hợp nặng nhất
            for f in warm:
                pipe.process(f)
            for mode in ("realtime", "realtime_detector_only"):
                if mode == "realtime_detector_only":
                    pipe.release_enhanced()  # chỉ detector (+ verifier ở v2)
                res = rb._realtime(pipe, body)
                errors[cfg] += res["errors"]
                for step, vals in pipe.timings().items():
                    pooled[cfg][mode].setdefault(step, []).extend(vals)
                row[mode] = res
        row["machine_during"] = s.summary()
        runs.append(row)
        print(
            f"lượt {i} {cfg}: tổng p95 {row['realtime']['ms'].get('total', {}).get('p95')} ms, "
            f"xung median {row['machine_during'].get('clock_mhz', {}).get('median')} MHz",
            file=sys.stderr,
        )
    steps = {}
    for mode in ("realtime", "realtime_detector_only"):
        keys = sorted(set(pooled["v1"][mode]) | set(pooled["v2"][mode]))
        steps[mode] = {
            k: {
                f"{cfg}_{q}": pct(pooled[cfg][mode].get(k, []), v)
                for cfg in names
                for q, v in (("p50", 0.5), ("p95", 0.95))
            }
            | {"n_v1": len(pooled["v1"][mode].get(k, [])), "n_v2": len(pooled["v2"][mode].get(k, []))}
            for k in keys
        }
    ceiling = {}
    for mode in ("realtime", "realtime_detector_only"):
        v1, v2 = steps[mode]["total"]["v1_p95"], steps[mode]["total"]["v2_p95"]
        ceiling[mode] = {
            "v1_p95": v1,
            "v2_p95": v2,
            "delta_ms": round(v2 - v1, 2),
            "max_delta_ms": CEILING_MS,
            "pass": (v2 - v1) <= CEILING_MS,
        }
    golden = {
        f"{cfg}/{k}": check_bundle(vis / d["bundle"], d["onnx_sha256"], None)["pass"]
        for cfg in names
        for k, d in info[cfg].items()
    }
    out = {
        "format": "iotcv-bench-v1-v2",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "recording": a.recording.name,
        "start": START,
        "frames": FRAMES,
        "warmup": WARMUP,
        "order": list(ORDER),
        "pooled_over": "hai lượt mỗi cấu hình (ABBA), gộp mọi khung rồi lấy phân vị",
        "bundles": info,
        "machine_start": machine_start,
        "machine_end": smi(),
        "steps": steps,
        "ceiling": ceiling,
        "errors": errors,
        "golden_cuda": golden,
        "phase09": {
            "system_p95_realtime": PHASE09_SYSTEM_P95,
            "note": "phase 09 (trước sửa verifier theo bậc 78a16af), GPU 780 MHz",
        },
        "runs": runs,
    }
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ceiling": ceiling, "errors": errors, "golden_cuda": golden}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
