"""Ảnh rõ offline ×2 (phase 11 AI v2): đo, fine-tune, chọn, xuất ``enhancer-offline-v1.0.0`` (hợp đồng §11).

Lệnh (AX-01)::

    python -m xla.enhance.sr bench  --config configs/sr_x2.yaml --json
    python -m xla.enhance.sr train  --config configs/sr_x2.yaml --variant l1_percep [--time-budget-min M] --json
    python -m xla.enhance.sr select --config configs/sr_x2.yaml --json
    python -m xla.enhance.sr export --config configs/sr_x2.yaml [--version 1.0.0] --json

- ``bench``: mỗi ứng viên zero-shot (trọng số công bố, kiểm sha256) chạy graph §11 trên ``bench.n_frames`` khung dev
  640×480, PyTorch fp16 và ORT CUDA fp16: s/khung, VRAM đỉnh, ước giờ cho ``flight_minutes`` phút video ở fps thật của
  bản ghi. Chậm hơn ``max_s_per_frame`` → ứng viên bị loại (Kill (d) nếu không còn ai).
- ``train``: fine-tune Real-ESRGAN x2plus trên cặp ``xla.enhance.sr_pairs`` (L1 + perceptual, GAN tuỳ biến thể), bf16,
  ``last.pt`` mỗi ``last_every`` bước + resume, dừng theo trần phút GPU của biến thể.
- ``select``: trên khung dev thật (chuỗi liền kề): nhấp nháy, thống kê mức thấp; trên khung ``cal`` (không người): báo
  nhầm detector v1 sau SR hạ ×½ về 640×480. Ứng viên tăng báo nhầm bị loại; chọn một (``choose``).
- ``export``: graph §11 fp16 opset 17, parity ORT CPU vs PyTorch fp32, golden + meta §11, ``xla.contract``.

Mọi lệnh GPU giữ ``xla.gpu_lock`` chỉ lúc chạy, ghi giờ GPU vào ``results/v2/sr/gpu_ledger.jsonl``; lệnh từ chối chạy
khi trần ``gpu_budget_hours`` đã hết. Mã thoát: 0 = xong, 1 = lỗi / không đạt / hết trần, 2 = sai cách gọi.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from xla import contract
from xla.paths import iotcv_root, repo_root

GPU_LOCK_TIMEOUT_S = 6 * 3600
ORT_CPU_THREADS = 4  # máy từng sập dưới tải: job CPU tối đa 4 luồng
DEVICE_NAME = "RTX 4060 Laptop"
SCALE = contract.OFFLINE_SCALE


# ---- cấu hình / sổ giờ GPU ---------------------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else repo_root() / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def results_dir(cfg: dict[str, Any]) -> Path:
    d = repo_root() / cfg["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(cfg: dict[str, Any]) -> Path:
    d = Path(cfg["runs_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ledger_path(cfg: dict[str, Any]) -> Path:
    return results_dir(cfg) / "gpu_ledger.jsonl"


def gpu_hours_used(cfg: dict[str, Any]) -> float:
    p = ledger_path(cfg)
    if not p.is_file():
        return 0.0
    return sum(float(json.loads(x)["gpu_seconds"]) for x in p.read_text(encoding="utf-8").splitlines() if x) / 3600


def record_gpu(cfg: dict[str, Any], step: str, gpu_seconds: float, note: str = "") -> None:
    with open(ledger_path(cfg), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"step": step, "gpu_seconds": round(gpu_seconds, 1), "at": now(), "note": note},
                           ensure_ascii=False) + "\n")


def budget_left_s(cfg: dict[str, Any]) -> float:
    return float(cfg["gpu_budget_hours"]) * 3600 - gpu_hours_used(cfg) * 3600


def camera_crop_left() -> tuple[int, str]:
    from xla.enhance.train_net import camera_crop_left as ccl

    return ccl()


# ---- mạng / graph §11 --------------------------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arch_module(arch: str, kwargs: dict[str, Any]) -> nn.Module:
    if arch == "rrdbnet":
        from xla.enhance.arch.rrdbnet import RRDBNet

        return RRDBNet(**kwargs)
    if arch == "swinir":
        from xla.enhance.arch.swinir import SwinIR

        return SwinIR(**kwargs)
    raise ValueError(f"arch không biết: {arch}")


def published_state(spec: dict[str, Any], weights_dir: Path, check_sha: bool = True) -> dict[str, torch.Tensor]:
    """state_dict trọng số công bố của ``spec``; sha256 file khác config → ValueError (không nạp file lạ)."""
    path = weights_dir / spec["weights"]
    if not path.is_file():
        raise FileNotFoundError(f"thiếu trọng số {path} (tải {spec['url']})")
    if check_sha and sha256_of(path) != spec["sha256"]:
        raise ValueError(f"sha256 của {path.name} khác config ({spec['sha256'][:12]}…)")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if spec.get("weights_key"):
        sd = sd[spec["weights_key"]]
    return {k: v for k, v in sd.items() if not k.endswith("attn_mask")}  # SwinIR: mặt nạ theo img_size 64, tính lại


def build_candidate(cfg: dict[str, Any], name: str) -> nn.Module:
    """Ứng viên zero-shot (khoá ``candidates``) nạp trọng số công bố, strict."""
    spec = cfg["candidates"][name]
    net = arch_module(spec["arch"], spec["kwargs"])
    net.load_state_dict(published_state(spec, Path(cfg["weights_dir"])), strict=True)
    return net.eval()


def load_finetuned(cfg: dict[str, Any], ckpt: Path) -> tuple[nn.Module, dict[str, Any]]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = cfg["candidates"][ck["base"]]
    net = arch_module(spec["arch"], spec["kwargs"])
    net.load_state_dict(ck["model"], strict=True)
    return net.eval(), ck


class SRGraph(nn.Module):
    """Graph §11 ảnh đơn: ``frame`` uint8 [480,640,3] BGR → ``frame_out`` uint8 [960,1280,3] BGR.

    Cắt cột ``[crop_left, 640)`` → BGR→RGB → /255 → (fp16) → mạng ×2 → kẹp [0,1] → ×255 → làm tròn → uint8 → RGB→BGR →
    cột ``[0, 2·crop_left)`` lặp cột ``2·crop_left`` của ảnh ra (dải mép §3 nhân 2).
    """

    def __init__(self, net: nn.Module, crop_left: int, half: bool = False) -> None:
        super().__init__()
        self.net = net
        self.crop_left = int(crop_left)
        self.use_half = half
        self.register_buffer("perm", torch.tensor([2, 1, 0], dtype=torch.long), persistent=False)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = frame[:, self.crop_left :, :].index_select(2, self.perm).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.use_half:
            x = x.half()
        y = self.net(x).float().clamp(0.0, 1.0) * 255.0
        y = torch.round(y)[0].permute(1, 2, 0).index_select(2, self.perm).to(torch.uint8)
        strip = y[:, :1, :].expand(-1, SCALE * self.crop_left, -1)
        return torch.cat([strip, y], dim=1)


def make_graph(net: nn.Module, crop_left: int, half: bool, device: str = "cuda") -> SRGraph:
    g = SRGraph(copy.deepcopy(net), crop_left, half=half)
    return (g.half() if half else g).to(device).eval()


@torch.no_grad()
def run_graph(graph: SRGraph, frames: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    return [graph(torch.from_numpy(np.ascontiguousarray(f)).to(device)).cpu().numpy() for f in frames]


def down2(img: np.ndarray) -> np.ndarray:
    """Ảnh ra ×2 → 640×480 (``INTER_AREA``) để đo cùng thang với khung gốc / chạy detector v1."""
    return cv2.resize(img, (contract.FRAME_W, contract.FRAME_H), interpolation=cv2.INTER_AREA)


# ---- đo VRAM / thời gian -----------------------------------------------------------------------------------------
class DeviceVramPeak:
    """Đỉnh VRAM cả thiết bị (``cudaMemGetInfo``) trừ mức nền lúc bắt đầu, lấy mẫu mỗi ``every_s`` giây.

    Dùng cho ORT (bộ cấp phát riêng, ``torch.cuda.max_memory_allocated`` không thấy). Tiến trình khác dùng GPU cùng lúc
    làm số lệch: mọi job GPU XLA chạy qua ``gpu_lock`` nên chỉ còn phần nền cố định (desktop), đã trừ ở mức nền.
    """

    def __init__(self, every_s: float = 0.02) -> None:
        self.every_s = every_s
        self.base_mb = self.peak_mb = 0.0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    @staticmethod
    def used_mb() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**20

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self.used_mb())
            time.sleep(self.every_s)

    def __enter__(self) -> Self:
        self.base_mb = self.peak_mb = self.used_mb()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t:
            self._t.join()
        self.peak_mb = max(self.peak_mb, self.used_mb())

    @property
    def delta_mb(self) -> float:
        return round(self.peak_mb - self.base_mb, 1)


def ort_session(onnx_path: Path, cuda: bool) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = ORT_CPU_THREADS
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
    return sess


def ort_run(sess: Any, frame: np.ndarray) -> np.ndarray:
    return sess.run(None, {contract.INPUT_NAME: frame})[0]


def time_ort(sess: Any, frames: list[np.ndarray], warmup: int) -> dict[str, Any]:
    """s/khung ORT (gồm chép host↔GPU); đọc ``get_providers`` SAU lần chạy thử (thiếu cuDNN thì lặng lẽ chạy CPU)."""
    for f in frames[:warmup]:
        ort_run(sess, f)
    prov = sess.get_providers()
    t = []
    for f in frames:
        t0 = time.perf_counter()
        ort_run(sess, f)
        t.append(time.perf_counter() - t0)
    return {"mean": float(np.mean(t)), "p50": float(np.median(t)), "p95": float(np.percentile(t, 95)), "n": len(t),
            "providers": prov}


@torch.no_grad()
def time_torch(graph: SRGraph, frames: list[np.ndarray], warmup: int) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    for f in frames[:warmup]:
        graph(torch.from_numpy(f).cuda())
    torch.cuda.synchronize()
    t = []
    for f in frames:
        t0 = time.perf_counter()
        graph(torch.from_numpy(f).cuda()).cpu()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    return {"mean": float(np.mean(t)), "p50": float(np.median(t)), "p95": float(np.percentile(t, 95)), "n": len(t),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def recording_fps(rec: str) -> dict[str, Any]:
    """fps thật của bản ghi IOT-CV: số khung / tổng thời lượng các đoạn (``segment``) trong ``frames.csv``."""
    import csv

    p = iotcv_root() / "logs" / "recordings" / rec / "frames.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError(f"{p}: quá ít khung ({len(rows)})")
    seg: dict[str, list[float]] = {}
    for r in rows:
        seg.setdefault(r["segment"], []).append(float(r["unix_time"]))
    dur = sum(max(v) - min(v) for v in seg.values())
    span = max(float(r["unix_time"]) for r in rows) - min(float(r["unix_time"]) for r in rows)
    return {"recording": rec, "frames": len(rows), "segments": len(seg), "seconds_in_segments": round(dur, 1),
            "fps": len(rows) / dur, "fps_wall_span": len(rows) / span}


def hours_for_flight(s_per_frame: float, fps: float, minutes: float) -> float:
    return s_per_frame * fps * minutes * 60 / 3600


def export_graph(graph: SRGraph, onnx_path: Path) -> None:
    from xla.enhance.net import export_onnx_cuda

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(contract.FRAME_SHAPE, dtype=torch.uint8, device="cuda")
    with torch.no_grad():
        graph(dummy)  # SwinIR: dựng sẵn mặt nạ cửa sổ dịch cho cỡ thật → hằng số trong graph
    export_onnx_cuda(graph, (dummy,), onnx_path, [contract.INPUT_NAME], ["frame_out"])
    import onnx

    onnx.checker.check_model(onnx.load(str(onnx_path)))


# ---- bench ---------------------------------------------------------------------------------------------------------
def dev_frames(n: int, offset: int = 0) -> list[str]:
    from xla.enhance.export_onnx import dev_frames as df

    return df(n, offset)


def frames_dir() -> Path:
    return repo_root() / "data" / "real" / "frames"


def bench(cfg: dict[str, Any], names: list[str] | None = None) -> dict[str, Any]:
    from xla.detect.selftrain import machine_info
    from xla.gpu_lock import gpu_lock

    cb = cfg["bench"]
    names = names or list(cfg["candidates"])
    crop_left, _ = camera_crop_left()
    fr = [contract.decode_frame(frames_dir() / n) for n in dev_frames(int(cb["n_frames"]))]
    fps = recording_fps(cb["recording"])
    rows: list[dict[str, Any]] = []
    t_wait = time.time()
    with gpu_lock("v2 phase11 sr bench", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        mach_before = machine_info()
        for name in names:
            spec = cfg["candidates"][name]
            net = build_candidate(cfg, name)
            n_params = int(sum(p.numel() for p in net.parameters()))
            row: dict[str, Any] = {"candidate": name, "arch": spec["arch"], "params": n_params,
                                   "weights": spec["weights"], "sha256_ok": True, "license": spec["license"]}
            try:
                g = make_graph(net, crop_left, half=True)
                row["torch_fp16"] = time_torch(g, fr, int(cb["warmup"]))
                out = run_graph(g, fr[:1])[0]
                row["out_shape"] = list(out.shape)
                onnx_path = runs_dir(cfg) / "bench" / f"{name}_fp16.onnx"
                t_exp = time.time()
                export_graph(g, onnx_path)
                row["export_s"] = round(time.time() - t_exp, 1)
                del g
                torch.cuda.empty_cache()
                with DeviceVramPeak() as vp:
                    sess = ort_session(onnx_path, cuda=True)
                    row["ort_cuda_fp16"] = time_ort(sess, fr, int(cb["warmup"]))
                row["ort_cuda_fp16"]["peak_vram_mb"] = vp.delta_mb
                if row["ort_cuda_fp16"]["providers"][0] != "CUDAExecutionProvider":
                    raise RuntimeError(f"ORT không chạy CUDA EP: {row['ort_cuda_fp16']['providers']}")
                del sess
            except torch.OutOfMemoryError as e:
                row["error"] = f"tràn VRAM: {str(e)[:200]}"
            del net
            torch.cuda.empty_cache()
            s = row.get("ort_cuda_fp16", {}).get("mean")
            if s is not None:
                row["hours_per_flight"] = round(hours_for_flight(s, fps["fps"], cb["flight_minutes"]), 3)
                row["pass_speed"] = s <= float(cb["max_s_per_frame"])
            else:
                row["pass_speed"] = False
            rows.append(row)
        mach_after = machine_info()
        gpu_s = time.time() - t0
    record_gpu(cfg, "bench", gpu_s, ",".join(names))
    return {"created": now(), "n_frames": len(fr), "frame_size": [contract.FRAME_W, contract.FRAME_H],
            "output_size": list(contract.OFFLINE_OUTPUT_SIZE), "precision": "fp16", "recording_fps": fps,
            "flight_minutes": cb["flight_minutes"], "max_s_per_frame": cb["max_s_per_frame"], "candidates": rows,
            "any_pass_speed": any(r["pass_speed"] for r in rows),
            "note": "s/khung = mean của session.run gồm chép host↔GPU; VRAM ORT = đỉnh cudaMemGetInfo trừ nền; "
                    "VRAM PyTorch = max_memory_allocated",
            "machine": {"before": mach_before, "after": mach_after}, "gpu_seconds": round(gpu_s, 1),
            "wait_lock_seconds": round(t0 - t_wait, 1)}


# ---- train (fine-tune) ---------------------------------------------------------------------------------------------
def _read(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"không giải mã được {path}")
    return img


@torch.no_grad()
def synthetic_val_sr(net: nn.Module, cfg: dict[str, Any], crop_left: int) -> dict[str, Any]:
    """PSNR trên cặp val tổng hợp (khung đủ): mạng vs nội suy bicubic ×2 — SỐ TỔNG HỢP, không dùng để kết luận."""
    from xla.enhance.train_net import pair_names, psnr, to_tensor_rgb
    from xla.paths import data_root

    root = data_root()
    names = pair_names("val", cfg)
    net.eval()
    ps, pb = [], []
    for nm in names:
        a = to_tensor_rgb(_read(root / cfg["data"]["input_set"] / "val" / "images" / nm)[:, crop_left:])[None].cuda()
        b = to_tensor_rgb(_read(root / cfg["data"]["target_set"] / "val" / "images" / nm))[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(a)
        y = torch.round(y.float().clamp(0, 1) * 255) / 255
        bic = F.interpolate(a, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)
        ps.append(psnr(y, b))
        pb.append(psnr(bic, b))
    return {"kind": "số tổng hợp (cặp sr_pairs val, ảnh tự sinh)", "n": len(names), "psnr": float(np.mean(ps)),
            "psnr_bicubic": float(np.mean(pb))}


def train(cfg: dict[str, Any], variant: str, time_budget_min: float | None = None) -> dict[str, Any]:
    """Fine-tune ``train.base`` theo ``train.variants[variant]`` (gan_weight 0 = không GAN). Resume từ ``last.pt``."""
    from xla.enhance.arch.rrdbnet import UNetDiscriminatorSN
    from xla.enhance.train_net import PairPatches, PerceptualLoss, _lr_at, _save, gan_loss, pair_names
    from xla.gpu_lock import gpu_lock

    ct = cfg["train"]
    vc = ct["variants"][variant]
    base = ct["base"]
    gan_w = float(vc["gan_weight"])
    budget_s = float(time_budget_min if time_budget_min is not None else vc["time_budget_min"]) * 60
    left = budget_left_s(cfg)
    if left < 300:
        raise RuntimeError(f"trần GPU phase 11 còn {left:.0f} s (< 300 s): không huấn luyện")
    budget_s = min(budget_s, left - 60)
    total = int(ct["max_steps"])
    crop_left, profile_version = camera_crop_left()
    run_dir = runs_dir(cfg) / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    torch.manual_seed(ct["seed"])
    random.seed(ct["seed"])
    np.random.seed(ct["seed"] % (2**32))

    net = build_candidate(cfg, base).train()
    disc = None
    if gan_w > 0:
        disc = UNetDiscriminatorSN(3, 64, True)
        disc.load_state_dict(published_state(cfg["discriminator"], Path(cfg["weights_dir"])), strict=True)
    step, used_s = 0, 0.0
    last = run_dir / "last.pt"
    ck: dict[str, Any] | None = None
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        if disc is not None:
            disc.load_state_dict(ck["disc"])
        step, used_s = int(ck["step"]), float(ck.get("train_seconds", 0.0))
        torch.manual_seed(ct["seed"] + step)
    batch = int(ct["batch"])
    ds = PairPatches(pair_names("train", cfg), cfg, crop_left, length=(total - step) * batch, scale=SCALE)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=min(4, int(ct["workers"])), pin_memory=True,
                                     drop_last=True)
    meta = {"base": base, "variant": variant, "crop_left": crop_left, "camera_profile_version": profile_version,
            "config_version": cfg["version"], "run_id": f"v2-sr-{variant}", "gan_weight": gan_w,
            "perceptual_weight": ct["perceptual_weight"], "pixel_weight": ct["pixel_weight"]}
    t_wait = time.time()
    with gpu_lock(f"v2 phase11 sr train {variant}", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        net.cuda()
        percep = PerceptualLoss().cuda()
        opt = torch.optim.Adam(net.parameters(), lr=ct["lr"], betas=tuple(ct["betas"]), weight_decay=ct["weight_decay"])
        opt_d = None
        if disc is not None:
            disc.cuda().train()
            opt_d = torch.optim.Adam(disc.parameters(), lr=float(vc.get("d_lr", ct["lr"])), betas=tuple(ct["betas"]))
        if ck is not None:
            opt.load_state_dict(ck["opt"])
            if opt_d is not None:
                opt_d.load_state_dict(ck["opt_d"])
            del ck
        torch.cuda.reset_peak_memory_stats()
        val0 = synthetic_val_sr(net, cfg, crop_left) if step == 0 else None
        net.train()
        t_steps = time.time()
        steps_done = 0
        stop = "max_steps"
        it = iter(dl)

        def state() -> dict[str, Any]:
            out = {**meta, "model": net.state_dict(), "opt": opt.state_dict(), "step": step,
                   "train_seconds": used_s + time.time() - t_steps}
            if disc is not None and opt_d is not None:
                out.update(disc=disc.state_dict(), opt_d=opt_d.state_dict())
            return out

        while step < total:
            if used_s + time.time() - t_steps >= budget_s:
                stop = "time_budget"
                break
            a, b = next(it)
            a, b = a.cuda(non_blocking=True), b.cuda(non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, ct, total)
            if disc is not None:
                disc.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = net(a)
                l_per = percep(y.float(), b) * float(ct["perceptual_weight"])
                l_g = gan_loss(disc(y), True) * gan_w if disc is not None else y.new_zeros(())
            y = y.float()
            l_pix = F.l1_loss(y, b) * float(ct["pixel_weight"])
            loss = l_pix + l_per + l_g
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ct["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(net.parameters(), ct["grad_clip"])
            opt.step()
            l_d = 0.0
            if disc is not None and opt_d is not None:
                disc.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ld = gan_loss(disc(b), True) + gan_loss(disc(y.detach()), False)
                ld.backward()
                opt_d.step()
                l_d = float(ld)
            step += 1
            steps_done += 1
            if not math.isfinite(float(loss)):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}")
            if step % 50 == 0:
                rec = {"step": step, "loss": float(loss), "l1": float(l_pix), "percep": float(l_per), "g": float(l_g),
                       "d": l_d, "lr": opt.param_groups[0]["lr"], "s_per_step": (time.time() - t_steps) / steps_done,
                       "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30}
                with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec) + "\n")
            if step % int(ct["last_every"]) == 0:
                _save(last, state())
        final = state()
        _save(last, final)
        _save(run_dir / "final.pt", {k: v for k, v in final.items() if k not in ("opt", "opt_d", "disc")})
        s_per_step = (time.time() - t_steps) / max(1, steps_done)
        val1 = synthetic_val_sr(net, cfg, crop_left)
        vram = torch.cuda.max_memory_allocated() / 2**30
        gpu_s = time.time() - t0
    record_gpu(cfg, f"train {variant}", gpu_s, f"steps {step}, stop {stop}")
    out = {"variant": variant, "base": base, "gan_weight": gan_w, "steps": step, "steps_this_run": steps_done,
           "stop": stop, "time_budget_s": round(budget_s, 1), "batch": batch, "patch_lr": cfg["data"]["patch"],
           "s_per_step": s_per_step, "vram_peak_gb": round(vram, 2), "val_synthetic_start": val0,
           "val_synthetic_end": val1, "final": str(run_dir / "final.pt"), "gpu_seconds": round(gpu_s, 1),
           "wait_lock_seconds": round(t0 - t_wait, 1), "finished": now()}
    write_json(run_dir / "train_summary.json", out)
    return out


# ---- select --------------------------------------------------------------------------------------------------------
SELECT_RULE = (
    "loại: s/khung ORT CUDA > max_s_per_frame (bench; fine-tune cùng kiến trúc với gốc của nó) hoặc báo nhầm detector "
    "v1 trên cal sau SR (hạ ×½) > trên khung gốc. Trong số còn lại: có ứng viên qua flicker_ratio ≤ flicker_max → điểm "
    "(pink_frac_drop + dark_channel_drop + blocking_excess_drop) cao nhất trong số đó; không ai qua → flicker_ratio "
    "thấp nhất (hoà thì điểm cao hơn), như luật chọn checkpoint phase 13 (xla.enhance.train_net.SELECT_RULE): nhấp nháy "
    "tăng là dấu hiệu mạng bịa chi tiết khác nhau giữa các khung. Không còn ai → null (Kill (d))"
)


def choose(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Áp ``SELECT_RULE``: mỗi dòng cần ``pass_speed``, ``fp_cal_raw``, ``fp_cal_offline``, ``pass_flicker``, ``score``."""
    ok = [r for r in rows if r.get("kind") != "reference" and r["pass_speed"] and r["fp_cal_offline"] <= r["fp_cal_raw"]]
    if not ok:
        return None, "không ứng viên nào qua tốc độ + không tăng báo nhầm cal: Kill (d)"
    calm = [r for r in ok if r["pass_flicker"]]
    if calm:
        return max(calm, key=lambda r: (r["score"], r["candidate"])), "qua flicker_max, điểm cao nhất"
    best = min(ok, key=lambda r: (r["flicker_ratio"], -r["score"]))
    return best, "không ai qua flicker_max: flicker_ratio thấp nhất trong số qua tốc độ + báo nhầm"


def rescore(cfg: dict[str, Any]) -> dict[str, Any]:
    """Áp lại ``SELECT_RULE`` lên ``select.json`` đã lưu (không chạy GPU); ghi lựa chọn cũ vào ``rescored_from``."""
    rep = json.loads((results_dir(cfg) / "select.json").read_text(encoding="utf-8"))
    rows = [{**r, "fp_cal_offline": math.inf if r.get("fp_cal_offline") is None else r["fp_cal_offline"],
             "score": -math.inf if r.get("score") is None else r["score"]} for r in rep["candidates"]]
    best, how = choose(rows)
    prev = {"chosen": rep.get("chosen"), "chosen_how": rep.get("chosen_how"), "rule": rep.get("rule")}
    rep.update(rule=SELECT_RULE, chosen=best["candidate"] if best else None, chosen_how=how, rescored_at=now(),
               rescored_from=prev)
    return rep


def select_frames(cfg: dict[str, Any]) -> tuple[list[list[str]], list[str]]:
    """(chuỗi dev liền kề ≥ ``min_run`` khung, cắt cách đều tới ``max_frames``; khung cal của neg_flight_split)."""
    from xla.enhance.train_net import dev_runs

    cs = cfg["select"]
    runs = [r for r in dev_runs(cs["split"]) if len(r) >= int(cs["min_run"])]
    cap = int(cs["max_frames"])
    picked: list[list[str]] = []
    total = sum(len(r) for r in runs)
    step = max(1, math.ceil(total / cap)) if total > cap else 1
    for i, r in enumerate(runs):  # mỗi ``step`` chuỗi lấy một chuỗi (giữ chuỗi nguyên vẹn để đo nhấp nháy)
        if i % step == 0 and sum(len(p) for p in picked) + len(r) <= cap:
            picked.append(r)
    neg = json.loads((repo_root() / "data" / "real" / "neg_flight_split.json").read_text(encoding="utf-8"))
    cal = sorted(f["frame"] for f in neg["frames"] if f["split"] == "cal")
    from xla.enhance.export_onnx import test_frame_names

    tests = test_frame_names()
    if any(n in tests for r in picked for n in r) or any(n in tests for n in cal):
        raise RuntimeError("khung chọn ứng viên rơi vào tập test")
    return picked, cal


def detector_fp(sess: Any, frames: list[np.ndarray], conf: float) -> dict[str, Any]:
    n_boxes = 0
    hit = 0
    for f in frames:
        _b, s = sess.run(["boxes", "scores"], {contract.INPUT_NAME: f})
        k = int((s >= conf).sum())
        n_boxes += k
        hit += k > 0
    return {"n_frames": len(frames), "n_boxes": n_boxes, "fp_per_frame": n_boxes / max(1, len(frames)),
            "frames_with_fp": hit}


def candidate_items(cfg: dict[str, Any], bench_rep: dict[str, Any]) -> list[dict[str, Any]]:
    """Ứng viên cho select: zero-shot qua bench + các ``final.pt`` fine-tune (kế thừa tốc độ của kiến trúc gốc)."""
    speed = {r["candidate"]: r for r in bench_rep["candidates"]}
    items = [{"candidate": n, "kind": "zero-shot", "base": n, "ckpt": None} for n in cfg["candidates"]]
    for v in cfg["train"]["variants"]:
        p = runs_dir(cfg) / v / "final.pt"
        if p.is_file():
            items.append({"candidate": f"ft_{v}", "kind": "fine-tune", "base": cfg["train"]["base"], "ckpt": str(p)})
    for it in items:
        b = speed.get(it["base"], {})
        it["s_per_frame"] = b.get("ort_cuda_fp16", {}).get("mean")
        it["peak_vram_mb"] = b.get("ort_cuda_fp16", {}).get("peak_vram_mb")
        it["hours_per_flight"] = b.get("hours_per_flight")
        it["pass_speed"] = bool(b.get("pass_speed"))
    return items


def _sample(raw: np.ndarray, out: np.ndarray, path: Path, box: tuple[int, int, int, int]) -> None:
    """Ảnh mẫu: gốc phóng ×2 (nearest) | ảnh ra, cùng vùng ``box`` (x, y, w, h theo px khung ra)."""
    x, y, w, h = box
    up = cv2.resize(raw, (raw.shape[1] * SCALE, raw.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
    pair = np.concatenate([up[y : y + h, x : x + w], np.full((h, 8, 3), 255, np.uint8), out[y : y + h, x : x + w]], 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])


def select(cfg: dict[str, Any]) -> dict[str, Any]:
    from xla.enhance.train_net import frame_stats, score_of
    from xla.eval.temporal import flicker_ratio
    from xla.gpu_lock import gpu_lock

    cs = cfg["select"]
    bench_rep = json.loads((results_dir(cfg) / "bench.json").read_text(encoding="utf-8"))
    items = candidate_items(cfg, bench_rep)
    crop_left, _ = camera_crop_left()
    runs, cal = select_frames(cfg)
    fdir = frames_dir()
    raw_runs = [[contract.decode_frame(fdir / n) for n in r] for r in runs]
    raw_cal = [contract.decode_frame(fdir / n) for n in cal]
    every = int(cs["stats_every"])
    raw_flat = [f for r in raw_runs for f in r]
    raw_stats = frame_stats(raw_flat[::every], crop_left)
    dmeta = json.loads((repo_root() / cs["detector_bundle"] / "meta.json").read_text(encoding="utf-8"))
    conf = float(dmeta["conf_threshold"])
    det_onnx = Path(cs["detector_onnx"])
    if not det_onnx.is_file() or sha256_of(det_onnx) != dmeta["onnx_sha256"]:
        raise ValueError(f"{det_onnx} thiếu hoặc sha256 khác meta.onnx_sha256 của {cs['detector_bundle']}")
    sample_dir = results_dir(cfg) / "samples"
    sample_box = (560, 300, 480, 360)
    kept: list[tuple[dict[str, Any], list[list[np.ndarray]], dict[str, Any]]] = []
    t_wait = time.time()
    with gpu_lock("v2 phase11 sr select", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        det = ort_session(det_onnx, cuda=True)
        fp_raw = detector_fp(det, raw_cal, conf)
        if det.get_providers()[0] != "CUDAExecutionProvider":  # đọc SAU lần chạy đầu (thiếu cuDNN → lặng lẽ CPU)
            raise RuntimeError(f"detector không chạy CUDA EP: {det.get_providers()}")
        # mốc tham chiếu (không phải ứng viên): bicubic ×2 rồi hạ ×½ — tách phần "đổi" do chính phép đổi cỡ gây ra
        bic = [[down2(cv2.resize(f, (2 * f.shape[1], 2 * f.shape[0]), interpolation=cv2.INTER_CUBIC)) for f in r]
               for r in raw_runs]
        bic_cal = [down2(cv2.resize(f, (2 * f.shape[1], 2 * f.shape[0]), interpolation=cv2.INTER_CUBIC))
                   for f in raw_cal]
        kept.append(({"candidate": "ref_bicubic", "kind": "reference", "base": None, "ckpt": None,
                      "pass_speed": True}, bic, detector_fp(det, bic_cal, conf)))
        for it in items:
            if not it["pass_speed"]:
                continue  # quá chậm: không ship được, không tốn giờ GPU đo tiếp
            net = build_candidate(cfg, it["base"]) if it["ckpt"] is None else load_finetuned(cfg, Path(it["ckpt"]))[0]
            g = make_graph(net, crop_left, half=True)
            out_runs = []
            for r_i, r in enumerate(raw_runs):
                outs = run_graph(g, r)
                if r_i == 0:
                    _sample(r[0], outs[0], sample_dir / f"{it['candidate']}.jpg", sample_box)
                out_runs.append([down2(o) for o in outs])
            out_cal = [down2(o) for o in run_graph(g, raw_cal)]
            fp_off = detector_fp(det, out_cal, conf)
            del g, net
            torch.cuda.empty_cache()
            kept.append((it, out_runs, fp_off))
        del det
        gpu_s = time.time() - t0
    record_gpu(cfg, "select", gpu_s, f"{len(kept)} ứng viên")
    rows: list[dict[str, Any]] = []
    for it, out_runs, fp_off in kept:
        fl = flicker_ratio(raw_runs, out_runs, crop_left)
        st = frame_stats([f for r in out_runs for f in r][::every], crop_left)
        rows.append({**it, "flicker_ratio": fl, "pass_flicker": fl <= float(cs["flicker_max"]), "stats": st,
                     **score_of(raw_stats, st), "fp_cal_raw": fp_raw["fp_per_frame"],
                     "fp_cal_offline": fp_off["fp_per_frame"], "fp_cal_detail": fp_off,
                     "pass_fp": fp_off["fp_per_frame"] <= fp_raw["fp_per_frame"],
                     "sample": (sample_dir / f"{it['candidate']}.jpg").relative_to(repo_root()).as_posix()
                     if it.get("kind") != "reference" else None})
    for it in items:
        if not it["pass_speed"]:
            rows.append({**it, "skipped": "quá chậm (bench), không đo select", "fp_cal_raw": fp_raw["fp_per_frame"],
                         "fp_cal_offline": math.inf, "pass_flicker": False, "score": -math.inf, "pass_fp": None})
    best, how = choose(rows)
    for r in rows:  # JSON không có inf
        for k in ("fp_cal_offline", "score"):
            if isinstance(r.get(k), float) and not math.isfinite(r[k]):
                r[k] = None
    return {"created": now(), "split": cs["split"], "n_runs": len(runs), "n_frames_dev": len(raw_flat),
            "n_frames_stats": len(raw_flat[::every]), "n_frames_cal": len(cal), "detector": cs["detector_bundle"],
            "conf_threshold": conf, "fp_cal_raw": fp_raw, "measured_on": "ảnh ra hạ ×½ INTER_AREA về 640×480",
            "raw_stats": raw_stats, "flicker_max": cs["flicker_max"], "rule": SELECT_RULE, "candidates": rows,
            "chosen": best["candidate"] if best else None, "chosen_how": how, "gpu_seconds": round(gpu_s, 1),
            "wait_lock_seconds": round(t0 - t_wait, 1)}


# ---- export --------------------------------------------------------------------------------------------------------
PARITY_MAE_MAX = 1.0


def _mae_max(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), int(d.max())


def offline_meta(*, version: str, onnx_sha: str, crop_left: int, profile_version: str, train_run: str,
                 network: dict[str, Any], s_per_frame: float, peak_vram_mb: float, machine: Any,
                 extra: dict[str, Any] | None = None, fp16: bool = True) -> dict[str, Any]:
    """meta.json §11 (ảnh đơn, ``temporal_window`` 1). ``metrics`` (fp cal/holdout) để phase 14 ghi."""
    return {
        "contract_version": contract.CONTRACT_VERSION,
        "model": "enhancer-offline",
        "version": version,
        "onnx_sha256": onnx_sha,
        "frame_size": [contract.FRAME_W, contract.FRAME_H],
        "crop_left": int(crop_left),
        "camera_profile_version": profile_version,
        "train_run": train_run,
        "created": now(),
        "scale": SCALE,
        "output_size": list(contract.OFFLINE_OUTPUT_SIZE),
        "temporal_window": 1,
        "network": network,
        "fp16": bool(fp16),
        "peak_vram_mb": float(peak_vram_mb),
        "s_per_frame": float(s_per_frame),
        "machine": machine,
        **(extra or {}),
    }


def export(cfg: dict[str, Any], version: str = "1.0.0", parity_extra: int = 2, half: bool = True) -> dict[str, Any]:
    from xla.detect.selftrain import machine_info
    from xla.enhance.export_onnx import test_frame_names
    from xla.gpu_lock import gpu_lock

    sel = json.loads((results_dir(cfg) / "select.json").read_text(encoding="utf-8"))
    if not sel.get("chosen"):
        raise RuntimeError("select không chọn ứng viên nào (Kill (d)): không xuất bundle")
    row = next(r for r in sel["candidates"] if r["candidate"] == sel["chosen"])
    crop_left, profile_version = camera_crop_left()
    if row["ckpt"]:
        net, ck = load_finetuned(cfg, Path(row["ckpt"]))
        train_run, step = ck["run_id"], ck["step"]
    else:
        net, train_run, step = build_candidate(cfg, row["base"]), f"zero-shot {row['base']}", None
    spec = cfg["candidates"][row["base"]]
    out = repo_root() / "models" / "release" / f"enhancer-offline-v{version}"
    gdir = out / "golden"
    gdir.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "enhancer-offline.onnx"
    golden_names = dev_frames(3, offset=11)
    parity_names = [*golden_names, *dev_frames(parity_extra, offset=29)]
    bench_names = dev_frames(int(cfg["bench"]["n_frames"]))
    tests = test_frame_names()
    if set(golden_names + parity_names + bench_names) & tests:
        raise RuntimeError("khung golden/parity/đo rơi vào tập test")
    fdir = frames_dir()
    pf = {n: contract.decode_frame(fdir / n) for n in parity_names}
    bf = [contract.decode_frame(fdir / n) for n in bench_names]
    rep: dict[str, Any] = {"chosen": sel["chosen"], "bundle": out.relative_to(repo_root()).as_posix()}
    t_wait = time.time()
    with gpu_lock(f"v2 phase11 sr export {out.name}", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g16 = make_graph(net, crop_left, half=half)
        export_graph(g16, onnx_path)
        del g16
        g32 = make_graph(net, crop_left, half=False)
        ref = dict(zip(parity_names, run_graph(g32, list(pf.values())), strict=True))
        del g32
        torch.cuda.empty_cache()
        mach = machine_info()
        with DeviceVramPeak() as vp:
            cuda = ort_session(onnx_path, cuda=True)
            timing = time_ort(cuda, bf, int(cfg["bench"]["warmup"]))
            got_cuda = {n: ort_run(cuda, f) for n, f in pf.items()}
        if timing["providers"][0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ORT không chạy CUDA EP: {timing['providers']}")
        del cuda
        gpu_s = time.time() - t0
    record_gpu(cfg, "export", gpu_s, out.name)
    cpu = ort_session(onnx_path, cuda=False)  # ORT CPU (hợp đồng: golden từ chính file .onnx trên CPU), ngoài khoá GPU
    got = {n: ort_run(cpu, f) for n, f in pf.items()}
    m_ref = [_mae_max(got[n], ref[n]) for n in parity_names]
    m_cuda = [_mae_max(got_cuda[n], got[n]) for n in parity_names]
    par = {"n_frames": len(parity_names), "frames": parity_names, "reference": "PyTorch fp32 (CUDA, TF32 tắt)",
           "ort_cpu_vs_torch_mae_worst": max(m for m, _ in m_ref), "ort_cpu_vs_torch_abs_max": max(x for _, x in m_ref),
           "ort_cuda_vs_cpu_mae_worst": max(m for m, _ in m_cuda), "ort_cuda_vs_cpu_abs_max": max(x for _, x in m_cuda),
           "mae_max_allowed": PARITY_MAE_MAX}
    par["ok"] = (par["ort_cpu_vs_torch_mae_worst"] <= PARITY_MAE_MAX
                 and par["ort_cuda_vs_cpu_mae_worst"] <= contract.ENH_MAE_MAX
                 and par["ort_cuda_vs_cpu_abs_max"] <= contract.ENH_ABS_MAX)
    expected: dict[str, Any] = {}
    for gname, src in zip(contract.GOLDEN_FRAMES, golden_names, strict=True):
        shutil.copyfile(fdir / src, gdir / gname)
        f = got[src]
        ok, buf = cv2.imencode(".png", f)
        if not ok:
            raise RuntimeError(f"không mã hoá được PNG {gname}")
        (gdir / gname.replace(".jpg", ".out.png")).write_bytes(buf.tobytes())
        expected[gname] = {"sha256_out": hashlib.sha256(f.tobytes()).hexdigest(), "mean": f.reshape(-1, 3).mean(0).tolist()}
    write_json(gdir / contract.EXPECTED_FILE, expected)
    network = {"candidate": sel["chosen"], "base": row["base"], "arch": spec["arch"], "arch_kwargs": spec["kwargs"],
               "init_weights": spec["weights"], "init_weights_sha256": spec["sha256"], "license": spec["license"],
               "step": step, "precision": "fp16" if half else "fp32", "opset": contract.OPSET}
    meta = offline_meta(version=version, onnx_sha=contract.sha256_file(onnx_path), crop_left=crop_left,
                        profile_version=profile_version, train_run=train_run, network=network,
                        s_per_frame=round(timing["mean"], 4), peak_vram_mb=vp.delta_mb,
                        machine={"device": DEVICE_NAME, **mach},
                        extra={"s_per_frame_detail": timing,
                               "metrics_note": "metrics (fp cal/holdout trên ảnh offline) phase 14 ghi",
                               "metrics_dev": {k: row.get(k) for k in ("flicker_ratio", "pink_frac_drop",
                                                                       "dark_channel_drop", "blocking_excess_drop",
                                                                       "fp_cal_raw", "fp_cal_offline")},
                               "golden_sources": dict(zip(contract.GOLDEN_FRAMES, golden_names, strict=True))},
                        fp16=half)
    write_json(out / contract.META_FILE, meta)
    errors = contract.validate_bundle(out)
    rep.update(parity=par, s_per_frame=timing, peak_vram_mb=vp.delta_mb, errors=errors,
               ok=not errors and par["ok"], onnx_sha256=meta["onnx_sha256"], gpu_seconds=round(gpu_s, 1),
               wait_lock_seconds=round(t0 - t_wait, 1), created=now())
    return rep


# ---- CLI -----------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.sr", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "train", "select", "export"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/sr_x2.yaml")
        p.add_argument("--json", action="store_true")
        if name == "bench":
            p.add_argument("--candidates", default=None, help="danh sách, mặc định mọi khoá candidates")
        if name == "train":
            p.add_argument("--variant", required=True)
            p.add_argument("--time-budget-min", type=float, default=None)
        if name == "export":
            p.add_argument("--version", default="1.0.0")
            p.add_argument("--fp32", action="store_true", help="xuất fp32 thay vì fp16")
        if name == "select":
            p.add_argument("--rescore", action="store_true", help="áp lại choose() lên select.json đã lưu, không GPU")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd != "export" and budget_left_s(cfg) <= 0:
        print(f"trần GPU {cfg['gpu_budget_hours']} h của phase 11 đã hết ({gpu_hours_used(cfg):.2f} h)", file=sys.stderr)
        return 1
    try:
        if args.cmd == "bench":
            names = [n.strip() for n in args.candidates.split(",")] if args.candidates else None
            bad = [n for n in names or [] if n not in cfg["candidates"]]
            if bad:
                print(f"ứng viên không biết: {bad}", file=sys.stderr)
                return 2
            rep = bench(cfg, names)
            write_json(results_dir(cfg) / "bench.json", rep)
            ok = rep["any_pass_speed"]
        elif args.cmd == "train":
            if args.variant not in cfg["train"]["variants"]:
                print(f"biến thể không biết: {args.variant}", file=sys.stderr)
                return 2
            rep = train(cfg, args.variant, args.time_budget_min)
            write_json(results_dir(cfg) / f"train_{args.variant}.json", rep)
            ok = True
        elif args.cmd == "select":
            rep = rescore(cfg) if args.rescore else select(cfg)
            write_json(results_dir(cfg) / "select.json", rep)
            ok = rep["chosen"] is not None
        else:
            rep = export(cfg, args.version, half=not args.fp32)
            write_json(results_dir(cfg) / "export.json", rep)
            ok = rep["ok"]
    except (OSError, ValueError, RuntimeError) as e:
        print(f"sr {args.cmd} lỗi: {e}", file=sys.stderr)
        return 1
    rep["gpu_hours_used_total"] = round(gpu_hours_used(cfg), 3)
    print(json.dumps(rep, ensure_ascii=False) if args.json else json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
