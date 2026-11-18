"""Xuất ``enhancer.onnx`` đúng hợp đồng §3 và dựng bundle §4 (phase 13 bước 4–5).

Lệnh (AX)::

    python -m xla.enhance.export_onnx --ckpt <pt> --out models/release/enhancer-v1.0.0 --json

Các bước: nạp checkpoint ``train_net`` → bọc ``xla.enhance.net.EnhancerGraph`` (uint8 BGR [480,640,3] → uint8 BGR, cột
``[0, crop_left)`` lặp cột ``crop_left``) ở fp16 → ``torch.onnx.export`` opset 17 → so ORT CPU với PyTorch fp32 trên
``--parity-n`` khung dev (MAE ≤ 1,0; ghi cả lệch lớn nhất và so ORT CUDA với ORT CPU) → đo ms ORT CUDA EP p50/p95
(warmup 30, đo 300, trong ``xla.gpu_lock``) → golden 3 khung dev (JPEG gốc camera) + ``expected.json`` + ``*.out.png``
từ chính file ``.onnx`` chạy ORT CPU → ``meta.json`` → ``xla.contract.validate_bundle``.

``meta.metrics`` (hợp đồng: số trên tập TEST thật) để ``null``: lane này không đụng khung test; phase 14 đo. Số chọn
checkpoint trên dev ghi riêng ở khoá thêm ``metrics_dev``.

Mã thoát: 0 = bundle hợp lệ + parity đạt; 1 = lệch hợp đồng / parity; 2 = sai cách gọi.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from xla import contract
from xla.enhance.net import EnhancerGraph, export_onnx_cuda, load_checkpoint, measure_ort_ms, ort_session
from xla.paths import repo_root

PARITY_MAE_MAX = 1.0
GPU_LOCK_TIMEOUT_S = 6 * 3600
DEVICE_NAME = "RTX 4060 Laptop"


def dev_frames(n: int, offset: int = 0) -> list[str]:
    """``n`` khung dev cách đều theo thời gian (tất định), bắt đầu lệch ``offset`` để tách golden khỏi parity."""
    s = json.loads((repo_root() / "data" / "real" / "split.json").read_text(encoding="utf-8"))
    dev = sorted((f for f in s["frames"] if f["split"] == "dev"), key=lambda f: float(f["t_unix"]))
    step = max(1, len(dev) // n)
    return [dev[(offset + i * step) % len(dev)]["frame"] for i in range(n)]


def test_frame_names() -> set[str]:
    s = json.loads((repo_root() / "data" / "real" / "split.json").read_text(encoding="utf-8"))
    return {f["frame"] for f in s["frames"] if f["split"] == "test"}


def export_graph(graph: EnhancerGraph, onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(contract.FRAME_SHAPE, dtype=torch.uint8, device="cuda")
    export_onnx_cuda(graph, (dummy,), onnx_path, [contract.INPUT_NAME], ["frame_out"])
    import onnx

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)


def _mae_max(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), int(d.max())


def export_bundle(
    ckpt: str | Path,
    out_dir: str | Path,
    *,
    version: str = "1.0.0",
    half: bool = True,
    parity_n: int = 20,
    skip_cuda: bool = False,
    metrics_dev: dict[str, Any] | None = None,
    train_run: str | None = None,
) -> dict[str, Any]:
    from xla.gpu_lock import gpu_lock

    out = Path(out_dir)
    out = out if out.is_absolute() else repo_root() / out
    gdir = out / "golden"
    gdir.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(ckpt)
    crop_left = int(ck["crop_left"])
    onnx_path = out / "enhancer.onnx"
    fdir = repo_root() / "data" / "real" / "frames"
    parity_names = dev_frames(parity_n)
    golden_names = dev_frames(3, offset=len(parity_names) // 2 + 7)
    tests = test_frame_names()
    if set(golden_names) & tests or set(parity_names) & tests:
        raise RuntimeError("khung golden/parity rơi vào tập test")
    frames = {n: contract.decode_frame(fdir / n) for n in parity_names}

    report: dict[str, Any] = {"ckpt": str(ckpt), "bundle": str(out), "half": half, "crop_left": crop_left}
    t_wait = time.time()
    with gpu_lock(f"phase13 restore export {out.name}", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g16 = EnhancerGraph(copy.deepcopy(model), crop_left, half=half)
        g16 = (g16.half() if half else g16).cuda().eval()
        export_graph(g16, onnx_path)
        # PyTorch fp32 tham chiếu
        g32 = EnhancerGraph(model, crop_left, half=False).cuda().eval()
        with torch.no_grad():
            ref = {n: g32(torch.from_numpy(f).cuda()).cpu().numpy() for n, f in frames.items()}
        del g16, g32
        torch.cuda.empty_cache()
        cpu = ort_session(onnx_path, ["CPUExecutionProvider"])
        got = {n: cpu.run(None, {contract.INPUT_NAME: f})[0] for n, f in frames.items()}
        maes = [_mae_max(got[n], ref[n]) for n in parity_names]
        report["parity"] = {
            "n_frames": len(parity_names),
            "frames": parity_names,
            "reference": "PyTorch fp32 (CUDA, TF32 tắt)",
            "ort_cpu_vs_torch_mae_mean": float(np.mean([m for m, _ in maes])),
            "ort_cpu_vs_torch_mae_worst": float(max(m for m, _ in maes)),
            "ort_cpu_vs_torch_abs_max": int(max(x for _, x in maes)),
            "mae_max_allowed": PARITY_MAE_MAX,
        }
        if not skip_cuda:
            import onnxruntime as ort

            cuda = ort_session(onnx_path)
            gc = {n: cuda.run(None, {contract.INPUT_NAME: f})[0] for n, f in frames.items()}
            cm = [_mae_max(gc[n], got[n]) for n in parity_names]
            report["parity"]["ort_cuda_vs_cpu_mae_worst"] = float(max(m for m, _ in cm))
            report["parity"]["ort_cuda_vs_cpu_abs_max"] = int(max(x for _, x in cm))
            ms = measure_ort_ms(cuda, {contract.INPUT_NAME: frames[parity_names[0]]})
            report["ms_per_frame"] = {"device": DEVICE_NAME, "runtime": f"onnxruntime-gpu {ort.__version__} CUDA EP",
                                      **ms}
            del cuda
        report["gpu_seconds"] = round(time.time() - t0, 1)
        report["wait_lock_seconds"] = round(t0 - t_wait, 1)
    parity_ok = report["parity"]["ort_cpu_vs_torch_mae_worst"] <= PARITY_MAE_MAX

    # golden: JPEG gốc camera của 3 khung dev, kết quả của chính file .onnx trên ORT CPU
    expected: dict[str, Any] = {}
    for gname, src in zip(contract.GOLDEN_FRAMES, golden_names, strict=True):
        shutil.copyfile(fdir / src, gdir / gname)
        f = cpu.run(None, {contract.INPUT_NAME: contract.decode_frame(gdir / gname)})[0]
        ok, buf = cv2.imencode(".png", f)
        if not ok:
            raise RuntimeError(f"không mã hoá được PNG {gname}")
        (gdir / gname.replace(".jpg", ".out.png")).write_bytes(buf.tobytes())
        expected[gname] = {"sha256_out": hashlib.sha256(f.tobytes()).hexdigest(),
                           "mean": f.reshape(-1, 3).mean(0).tolist()}
    (gdir / contract.EXPECTED_FILE).write_text(json.dumps(expected, indent=2), encoding="utf-8")

    msf = report.get("ms_per_frame", {"device": "không đo (--skip-cuda)", "runtime": "không đo", "p50": -1.0,
                                       "p95": -1.0})
    meta: dict[str, Any] = {
        "contract_version": contract.CONTRACT_VERSION,
        "model": "enhancer",
        "version": version,
        "onnx_sha256": contract.sha256_file(onnx_path),
        "frame_size": [contract.FRAME_W, contract.FRAME_H],
        "crop_left": crop_left,
        "camera_profile_version": ck["camera_profile_version"],
        "train_run": train_run or ck.get("run_id", "unknown"),
        "metrics": {k: None for k in contract.METRICS_KEYS["enhancer"]},
        "ms_per_frame": {k: msf[k] for k in ("device", "runtime", "p50", "p95")},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        # khoá thêm (ngoài hợp đồng, tuỳ chọn):
        "metrics_note": "metrics (tập test thật) = null: lane restore không đụng khung test; phase 14 đo",
        "metrics_dev": metrics_dev,
        "network": {"candidate": ck["candidate"], "arch_kwargs": ck.get("arch_kwargs") or {}, "precision":
                    "fp16" if half else "fp32", "step": ck.get("step")},
        "golden_sources": dict(zip(contract.GOLDEN_FRAMES, golden_names, strict=True)),
    }
    (out / contract.META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    errors = contract.validate_bundle(out)
    report.update(errors=errors, parity_ok=parity_ok, onnx_sha256=meta["onnx_sha256"], golden_sources=golden_names)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.export_onnx", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--fp32", action="store_true", help="xuất fp32 thay vì fp16")
    ap.add_argument("--parity-n", type=int, default=20)
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--selection", default="results/restore/train.json",
                    help="file có khoá select (metrics_dev lấy từ checkpoint đã chọn)")
    ap.add_argument("--report", default="results/restore/export.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.ckpt).is_file():
        print(f"không có checkpoint: {args.ckpt}", file=sys.stderr)
        return 2
    if not contract.BUNDLE_DIR_RE.match(Path(args.out).name) or Path(args.out).name != f"enhancer-v{args.version}":
        print(f"--out phải tên enhancer-v{args.version}", file=sys.stderr)
        return 2
    metrics_dev = None
    train_run = None
    sel_path = repo_root() / args.selection
    if sel_path.is_file():
        sel = json.loads(sel_path.read_text(encoding="utf-8")).get("select")
        if sel:
            train_run = sel.get("run_id")
            row = next((r for r in sel["checkpoints"] if r["path"] and Path(r["path"]) == Path(args.ckpt)), None)
            if row:
                metrics_dev = {"split": "dev", "pink_frac_drop": row["pink_frac_drop"],
                               "dark_channel_drop": row["dark_channel_drop"], "flicker_ratio": row["flicker_ratio"],
                               "blocking_excess_drop": row["blocking_excess_drop"], "checkpoint": row["checkpoint"]}
    report = export_bundle(args.ckpt, args.out, version=args.version, half=not args.fp32, parity_n=args.parity_n,
                           skip_cuda=args.skip_cuda, metrics_dev=metrics_dev, train_run=train_run)
    rp = repo_root() / args.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False) if args.json else json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["errors"] and report["parity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
