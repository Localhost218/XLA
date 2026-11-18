"""Mạng khôi phục phase 13: dựng/nạp ba ứng viên (NAFNet, FBCNN, SCUNet) và đo tốc độ ORT CUDA EP.

Lệnh (AX)::

    python -m xla.enhance.net speed --candidates nafnet,fbcnn,scunet --json

Đo: mỗi ứng viên nạp trọng số công bố, xuất ONNX fp16 (opset 17) phần mạng với ảnh vào cố định ``[1, 3, 480, 624]``
(khung 640 cột bỏ ``crop_left`` = 16), chạy ORT CUDA EP mặc định, warmup 30 rồi đo 300 lần ``session.run`` (gồm chép
host↔GPU), trong ``xla.gpu_lock``. Ghi ``results/restore/speed.json`` kèm giấy phép đọc từ file LICENSE của repo gốc.
Chọn: ứng viên nhanh nhất có p95 ≤ ``P95_BUDGET_MS``; không ứng viên nào đạt → ``chosen = null`` (Forecast Kill (c)).

Mã thoát ``speed``: 0 = có ứng viên đạt, 1 = không ứng viên nào đạt, 2 = sai cách gọi / thiếu trọng số.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from xla.paths import repo_root

WEIGHTS_DIR = Path("D:/DevCache/weights/restore")
DOWNLOADS_DIR = Path("D:/IOT_Tools/downloads/restore")
RUNS_DIR = Path("D:/DevCache/runs/xla/restore")
SPEED_JSON_REL = Path("results/restore/speed.json")

NET_H, NET_W = 480, 624  # khung 480×640 bỏ crop_left = 16 cột
P95_BUDGET_MS = 20.0
WARMUP = 30
ITERS = 300
GPU_LOCK_TIMEOUT_S = 6 * 3600

CANDIDATES: dict[str, dict[str, Any]] = {
    "nafnet": {
        "arch": "nafnet",
        "kwargs": {"width": 32, "middle_blk_num": 12, "enc_blk_nums": (2, 2, 4, 8), "dec_blk_nums": (2, 2, 2, 2)},
        "weights": "NAFNet-SIDD-width32.pth",
        "weights_key": "params",
        "published": "NAFNet-SIDD-width32 (khử nhiễu SIDD, bản nhỏ nhất được công bố)",
        "code_url": "https://github.com/megvii-research/NAFNet",
        "weights_url": "https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ",
        "repo_dir": "NAFNet",
    },
    "fbcnn": {
        "arch": "fbcnn",
        "kwargs": {"nc": (64, 128, 256, 512), "nb": 4},
        "weights": "fbcnn_color.pth",
        "weights_key": None,
        "published": "fbcnn_color (khử artifact JPEG màu, mù QF)",
        "code_url": "https://github.com/jiaxi-jiang/FBCNN",
        "weights_url": "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
        "repo_dir": "FBCNN",
    },
    "scunet": {
        "arch": "scunet",
        "kwargs": {"config": (4, 4, 4, 4, 4, 4, 4), "dim": 64},
        "weights": "scunet_color_real_psnr.pth",
        "weights_key": None,
        "published": "scunet_color_real_psnr (khử nhiễu ảnh thật, tối ưu PSNR)",
        "code_url": "https://github.com/cszn/SCUNet",
        "weights_url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "repo_dir": "SCUNet",
    },
}


def _arch_class(arch: str) -> type[nn.Module]:
    if arch == "nafnet":
        from xla.enhance.arch.nafnet import NAFNet

        return NAFNet
    if arch == "fbcnn":
        from xla.enhance.arch.fbcnn import FBCNN

        return FBCNN
    if arch == "scunet":
        from xla.enhance.arch.scunet import SCUNet

        return SCUNet
    raise ValueError(f"arch không biết: {arch}")


class RestoreNet(nn.Module):
    """Bọc mạng: vào RGB float NCHW 0–1, ra RGB float NCHW (chưa kẹp). FBCNN trả tuple → lấy ảnh."""

    def __init__(self, arch: str, net: nn.Module) -> None:
        super().__init__()
        self.arch = arch
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y[0] if isinstance(y, tuple) else y


class EnhancerGraph(nn.Module):
    """Graph hợp đồng §3: ``frame`` uint8 [480,640,3] BGR → ``frame_out`` uint8 [480,640,3] BGR.

    Cắt cột ``[crop_left, 640)`` → BGR→RGB → /255 → (fp16 nếu ``half``) → mạng → kẹp [0,1] → ×255 → làm tròn (nửa về
    chẵn, như ONNX ``Round``) → uint8 → RGB→BGR → đệm cột ``[0, crop_left)`` bằng lặp cột ``crop_left`` của ảnh ra.
    Mạng nhận 480×624 (bội 16 với NAFNet; FBCNN/SCUNet tự đệm và bỏ đệm bên trong).
    """

    def __init__(self, net: RestoreNet, crop_left: int, half: bool = False) -> None:
        super().__init__()
        self.net = net
        self.crop_left = int(crop_left)
        self.half = half
        self.register_buffer("perm", torch.tensor([2, 1, 0], dtype=torch.long), persistent=False)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = frame[:, self.crop_left :, :].index_select(2, self.perm).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.half:
            x = x.half()
        y = self.net(x).float().clamp(0.0, 1.0) * 255.0
        y = torch.round(y)[0].permute(1, 2, 0).index_select(2, self.perm).to(torch.uint8)
        strip = y[:, :1, :].expand(-1, self.crop_left, -1)
        return torch.cat([strip, y], dim=1)


@torch.no_grad()
def enhance_frames(graph: EnhancerGraph, frames: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    """Chạy ``graph`` (PyTorch) trên từng khung uint8 BGR, trả uint8 BGR cùng cỡ."""
    out = []
    for f in frames:
        t = torch.from_numpy(np.ascontiguousarray(f)).to(device)
        out.append(graph(t).cpu().numpy())
    return out


def build(name: str, kwargs: dict[str, Any] | None = None) -> RestoreNet:
    """Dựng ứng viên ``name`` (khoá ``CANDIDATES``); ``kwargs`` ghi đè tham số kiến trúc (biến thể thu nhỏ)."""
    if name not in CANDIDATES:
        raise ValueError(f"ứng viên không biết: {name}; có {sorted(CANDIDATES)}")
    spec = CANDIDATES[name]
    kw = dict(spec["kwargs"])
    kw.update(kwargs or {})
    return RestoreNet(spec["arch"], _arch_class(spec["arch"])(**kw))


def load_pretrained(model: RestoreNet, name: str, weights_dir: Path = WEIGHTS_DIR, strict: bool = True) -> RestoreNet:
    """Nạp trọng số công bố của ``name`` vào ``model.net``."""
    spec = CANDIDATES[name]
    path = weights_dir / spec["weights"]
    if not path.is_file():
        raise FileNotFoundError(f"thiếu trọng số {path} (tải từ {spec['weights_url']})")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if spec["weights_key"]:
        sd = sd[spec["weights_key"]]
    model.net.load_state_dict(sd, strict=strict)
    return model


def load_checkpoint(path: str | Path) -> tuple[RestoreNet, dict[str, Any]]:
    """Nạp checkpoint do ``train_net`` ghi: ``{"candidate", "arch_kwargs", "model", ...}``."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = build(ck["candidate"], ck.get("arch_kwargs"))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def read_license(name: str, downloads_dir: Path = DOWNLOADS_DIR) -> dict[str, str]:
    """Đọc dòng tiêu đề file LICENSE của repo gốc (đã clone ở ``downloads_dir``); không đọc được → "chưa xác minh"."""
    spec = CANDIDATES[name]
    lic = downloads_dir / spec["repo_dir"] / "LICENSE"
    out = {"code_url": spec["code_url"], "weights_url": spec["weights_url"], "license_file": str(lic)}
    try:
        text = lic.read_text(encoding="utf-8", errors="replace")
    except OSError:
        out.update(code="chưa xác minh", weights="chưa xác minh")
        return out
    head = " ".join(text.split()[:12])
    if "MIT License" in head:
        code = "MIT"
    elif "Apache License" in head and "Version 2.0" in head:
        code = "Apache-2.0"
    else:
        code = f"chưa xác minh (đầu file: {head[:60]})"
    out["code"] = code
    out["weights"] = (
        f"chưa xác minh riêng: trọng số phát hành cùng repo, repo không ghi giấy phép riêng cho trọng số (repo: {code})"
    )
    return out


# ---- đo tốc độ ORT -------------------------------------------------------------------------------------------
def ort_session(onnx_path: str | Path, providers: list[str] | None = None) -> Any:
    import onnxruntime as ort  # torch đã import ở đầu module: DLL CUDA/cuDNN của torch có sẵn cho CUDA EP

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers or ["CUDAExecutionProvider"])
    if providers is None and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"CUDA EP không nạp được, chỉ có {sess.get_providers()}")
    return sess


def measure_ort_ms(sess: Any, feed: dict[str, np.ndarray], warmup: int = WARMUP, iters: int = ITERS) -> dict[str, float]:
    """Đo ms mỗi lần ``sess.run`` (gồm chép vào/ra): warmup rồi ``iters`` lần; trả p50/p95/mean/min."""
    for _ in range(warmup):
        sess.run(None, feed)
    t = np.empty(iters, np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "p50": float(np.percentile(t, 50)),
        "p95": float(np.percentile(t, 95)),
        "mean": float(t.mean()),
        "min": float(t.min()),
        "n": int(iters),
    }


def export_net_fp16(model: RestoreNet, out: Path, h: int = NET_H, w: int = NET_W) -> Path:
    """Xuất riêng phần mạng ở fp16 (để đo tốc độ chọn ứng viên), input ``x`` [1,3,h,w] fp16."""
    out.parent.mkdir(parents=True, exist_ok=True)
    m = model.eval().half().cuda()
    x = torch.rand(1, 3, h, w, device="cuda").half()
    export_onnx_cuda(m, (x,), out, ["x"], ["y"])
    return out


def export_onnx_cuda(m: nn.Module, args: tuple[Any, ...], out: Path, inputs: list[str], outputs: list[str]) -> bool:
    """``torch.onnx.export`` opset 17 cho module trên CUDA; trả ``True`` nếu đã gấp hằng số.

    torch 2.6 gấp hằng số SCUNet (mask/relative position) lẫn tensor CPU với CUDA → ``RuntimeError`` "same device". Khi
    đó xuất lại với ``do_constant_folding=False`` (ORT tự gấp hằng số lúc tạo session) và báo ra stderr, không im lặng.
    """
    kw = {"opset_version": 17, "input_names": inputs, "output_names": outputs, "dynamo": False}
    with torch.no_grad():
        try:
            torch.onnx.export(m, args, str(out), do_constant_folding=True, **kw)
            return True
        except RuntimeError as e:
            if "same device" not in str(e):
                raise
            print(f"[export] {out.name}: gấp hằng số lỗi thiết bị, xuất lại không gấp ({e})", file=sys.stderr)
            torch.onnx.export(m, args, str(out), do_constant_folding=False, **kw)
            return False


def speed_one(name: str, warmup: int, iters: int, out_dir: Path) -> dict[str, Any]:
    import onnxruntime as ort

    model = load_pretrained(build(name), name)
    params = n_params(model)
    onnx_path = export_net_fp16(model, out_dir / f"{name}_fp16_{NET_H}x{NET_W}.onnx")
    del model
    torch.cuda.empty_cache()
    sess = ort_session(onnx_path)
    feed = {"x": np.random.default_rng(0).random((1, 3, NET_H, NET_W)).astype(np.float16)}
    ms = measure_ort_ms(sess, feed, warmup, iters)
    del sess
    return {
        "candidate": name,
        "published_weights": CANDIDATES[name]["published"],
        "arch_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in CANDIDATES[name]["kwargs"].items()},
        "params": params,
        "input": [1, 3, NET_H, NET_W],
        "precision": "fp16",
        "runtime": f"onnxruntime-gpu {ort.__version__} CUDA EP (mặc định)",
        "ms": ms,
        "p95_le_budget": ms["p95"] <= P95_BUDGET_MS,
        "license": read_license(name),
    }


def run_speed(names: list[str], warmup: int = WARMUP, iters: int = ITERS) -> dict[str, Any]:
    from xla.gpu_lock import gpu_lock

    out_dir = RUNS_DIR / "speed"
    rows = []
    t0 = time.time()
    with gpu_lock("phase13 restore speed", timeout_s=GPU_LOCK_TIMEOUT_S):
        t_lock = time.time()
        for n in names:
            rows.append(speed_one(n, warmup, iters, out_dir))
        gpu_s = time.time() - t_lock
    ok = [r for r in rows if r["p95_le_budget"]]
    chosen = min(ok, key=lambda r: r["ms"]["p95"])["candidate"] if ok else None
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "budget_p95_ms": P95_BUDGET_MS,
        "warmup": warmup,
        "iters": iters,
        "candidates": rows,
        "chosen": chosen,
        "rule": f"nhanh nhất (p95) trong các ứng viên có p95 ≤ {P95_BUDGET_MS} ms; không ai đạt → null (Kill (c))",
        "gpu_seconds": round(gpu_s, 1),
        "wait_lock_seconds": round(t_lock - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.net", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speed", help="đo ms/khung ORT CUDA EP fp16 của các ứng viên")
    s.add_argument("--candidates", default="nafnet,fbcnn,scunet")
    s.add_argument("--warmup", type=int, default=WARMUP)
    s.add_argument("--iters", type=int, default=ITERS)
    s.add_argument("--out", default=str(SPEED_JSON_REL), help="tương đối gốc repo")
    s.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    bad = [n for n in names if n not in CANDIDATES]
    if bad:
        print(f"ứng viên không biết: {bad}", file=sys.stderr)
        return 2
    missing = [n for n in names if not (WEIGHTS_DIR / CANDIDATES[n]["weights"]).is_file()]
    if missing:
        print(f"thiếu trọng số: {missing} trong {WEIGHTS_DIR}", file=sys.stderr)
        return 2
    report = run_speed(names, args.warmup, args.iters)
    out = Path(args.out)
    out = out if out.is_absolute() else repo_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        for r in report["candidates"]:
            print(f"{r['candidate']:8s} p50 {r['ms']['p50']:7.2f} ms  p95 {r['ms']['p95']:7.2f} ms  {r['params']/1e6:.1f}M")
        print(f"chọn: {report['chosen']}")
    return 0 if report["chosen"] else 1


if __name__ == "__main__":
    sys.exit(main())
