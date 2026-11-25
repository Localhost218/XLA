"""Xuất ``xla.enhance.classic`` thành ``enhancer.onnx`` đúng hợp đồng §3 + bundle §4/§12 (AI v2 phase 10).

Lệnh (AX-01)::

    python -m xla.enhance.export_classic --params configs/enhance_classic.yaml --out models/release/enhancer-v2.0.0 --json
    python -m xla.enhance.export_classic --params configs/enhance_classic.yaml --out <dir> --skip-cuda --json
    python -m xla.enhance.export_classic hybrid --config configs/restore_hybrid.yaml --json   # bước 4 (lai, có trần)

numpy ``xla/enhance/classic.py`` là SSOT; file này dựng lại **cùng phép tính** bằng op ONNX opset 17 từ cùng tham số
(``classic.load_params``), không sửa ``classic.py``:

- lọc Gauss = tổng các lát dịch có trọng số cố định trên ảnh đệm phản chiếu (``BORDER_REFLECT_101`` như cv2, đệm bằng
  ``Gather`` chỉ số hằng), không ``Conv``/``MatMul`` — ORT CUDA EP có thể chạy chúng bằng TF32 (~1e-3 tương đối), đủ lệch
  làm tròn uint8; trung bình box dùng ``AveragePool`` (không bị TF32);
- erode / dilate = ``MaxPool`` (biên bỏ qua như cv2); phân vị = ``TopK`` + nội suy tuyến tính đúng công thức của
  ``np.percentile`` (``kthvalue``/``quantile`` không xuất được ở opset 17);
- mọi phép uint8 / int16 của cv2 = phép float rồi ``Round`` (nửa về chẵn như ``cvRound``) + kẹp bão hoà ở đúng chỗ;
- nhánh phụ thuộc dữ liệu: ``if lo_out < floor`` của khử mờ = ``Where``; hai lệnh ``break`` theo ``countNonZero`` của khử
  hồng bỏ hẳn vì khi cổng toàn 0 một lượt là phép đồng nhất (α = 0 ⇒ Δ = 0, bơm chi tiết = 0), nên không cần nhánh.

Mã thoát: 0 = bundle hợp lệ + parity đạt; 1 = lệch hợp đồng / parity; 2 = sai cách gọi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from xla import contract
from xla.enhance import classic as C
from xla.enhance.net import ort_session
from xla.paths import repo_root

STAGE_NAMES = tuple(name for name, _ in C.STAGES)
"""Năm tầng theo đúng thứ tự chạy của ``classic.STAGES``."""
_LUMA = tuple(np.float32(v) for v in C._LUMA_BGR)  # B, G, R (BT.601) — cùng hằng của classic
_INV255 = float(np.float32(1 / 255))
_IMMERKAER = ((1.0, -2.0, 1.0), (-2.0, 4.0, -2.0), (1.0, -2.0, 1.0))  # = xla.camera.measure._IMMERKAER


# ----------------------------------------------------------------------------- phép cơ bản (tensor [C, H, W] float)

def _sat8(x: torch.Tensor) -> torch.Tensor:
    """``saturate_cast<uchar>``: làm tròn nửa về chẵn rồi kẹp 0–255 (giá trị vẫn là float)."""
    return torch.clamp(torch.round(x), 0.0, 255.0)


def _luma(b: torch.Tensor, g: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Y float như ``classic._luma``: ``(B·0,114 + G·0,587) + R·0,299`` (hai lần ``addWeighted``)."""
    return (b * _LUMA[0] + g * _LUMA[1]) + r * _LUMA[2]


def _reflect(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """Đệm ``BORDER_REFLECT_101`` ``r`` điểm mỗi phía của trục ``dim`` bằng một ``Gather`` chỉ số hằng. Không dùng
    ``F.pad`` reflect: exporter sinh ``If`` (luồng điều khiển theo shape) + ``pads`` tính ra, graph mất shape tĩnh; lát +
    lật + nối thì tốn 5 nút mỗi trục (đo ORT CUDA: ``Concat`` + ``Slice`` ~4,5 ms/khung)."""
    if r == 0:
        return x
    n = int(x.shape[dim])  # cỡ tĩnh (hợp đồng §1): hằng trong graph
    idx = list(range(r, 0, -1)) + list(range(n)) + list(range(n - 2, n - 2 - r, -1))
    return x.index_select(dim, torch.tensor(idx, dtype=torch.long))


def _pad_reflect(x: torch.Tensor, ry: int, rx: int) -> torch.Tensor:
    """Đệm ``BORDER_REFLECT_101`` ``ry`` hàng / ``rx`` cột mỗi phía; x [..., H, W]."""
    return _reflect(_reflect(x, rx, -1), ry, -2)


def _sep_filter(x: torch.Tensor, kern: list[float]) -> torch.Tensor:
    """Lọc tách được (hàng rồi cột, như ``sepFilter2D``) bằng tổng lát dịch có trọng số; x [..., H, W]. Không ``Conv``:
    ORT CUDA EP có thể chạy ``Conv`` fp32 bằng TF32 (lệch ~1e-3 tương đối, đủ đổi làm tròn uint8); bản ``Conv`` tách đúng
    TF32 đã thử, chậm hơn 3 ms/khung (p50 30,1 so với 26,9 ms, cùng lần giữ khoá GPU)."""
    r = len(kern) // 2
    h, w = int(x.shape[-2]), int(x.shape[-1])
    xp = _pad_reflect(x, r, r)
    row = sum(xp[..., i : i + w] * kern[i] for i in range(len(kern)))
    return sum(row[..., i : i + h, :] * kern[i] for i in range(len(kern)))


def _box_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """``cv2.blur`` / ``boxFilter`` chuẩn hoá, cạnh ``k`` lẻ, biên phản chiếu (``AveragePool``, không TF32); x [C, H, W]."""
    return F.avg_pool2d(_pad_reflect(x, k // 2, k // 2).unsqueeze(0), k, stride=1).squeeze(0)


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    """``cv2.dilate`` nhân vuông cạnh ``k`` lẻ (biên không góp, như giá trị biên mặc định của cv2); x [C, H, W]."""
    return F.max_pool2d(x.unsqueeze(0), k, stride=1, padding=k // 2).squeeze(0)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -_dilate(-x, k)


def gauss_kernel(sigma: float, depth_8u: bool) -> list[float]:
    """Nhân Gauss 1-D mà ``cv2.GaussianBlur(ksize=(0, 0), sigma)`` dùng: cạnh ``cvRound(σ·(3 nếu 8U, 4 nếu float)·2 + 1)|1``
    (``createGaussianKernels``), hệ số ``getGaussianKernel`` float32. Bản 8U của cv2 tính dấu phẩy tĩnh: lệch ≤ 1 mức."""
    ks = round(sigma * (3 if depth_8u else 4) * 2 + 1) | 1
    return [float(v) for v in cv2.getGaussianKernel(ks, sigma, ktype=cv2.CV_32F).ravel()]


def _percentile(flat: torch.Tensor, q: float, n: int) -> torch.Tensor:
    """``np.percentile(a, q)`` (nội suy ``linear``) của vector cỡ tĩnh bằng ``TopK``: chỉ lấy hai phần tử kề vị trí
    ``q/100·(n−1)`` từ phía gần hơn, rồi nội suy như ``numpy._lerp`` (``t ≥ 0,5`` tính từ phần tử trên); ``n`` = cỡ tĩnh."""
    idx = q / 100.0 * (n - 1)
    lo = math.floor(idx)
    hi = min(lo + 1, n - 1)
    t = idx - lo
    if q >= 50:
        vals = torch.topk(flat, n - lo, largest=True, sorted=True).values  # giảm dần: a[i] = vals[n − 1 − i]
        a, b = vals[n - 1 - lo], vals[n - 1 - hi]
    else:
        vals = torch.topk(flat, hi + 1, largest=False, sorted=True).values  # tăng dần: a[i] = vals[i]
        a, b = vals[lo], vals[hi]
    return b - (b - a) * (1.0 - t) if t >= 0.5 else a + (b - a) * t


# ----------------------------------------------------------------------------- năm tầng (vào/ra [3, H, W] float nguyên)

class Deblock(nn.Module):
    """``classic.deblock``: san bậc thang Y trên biên lưới khối của khung gốc, biên dọc rồi biên ngang."""

    def __init__(self, params: dict, h: int, w: int) -> None:
        super().__init__()
        d = params["deblock"]
        self.blk = int(d["block_px"])
        self.inv_alpha = 1.0 / float(d["alpha"])
        self.inv_beta = 1.0 / float(d["beta"])
        self.strength = float(d["strength"])
        self.inner = float(d["inner_weight"])
        self.phase = (-int(params["crop_left"])) % self.blk
        self.h, self.w = h, w

    def _delta(self, y: torch.Tensor, c0: int, h: int, w: int) -> torch.Tensor:
        """Lượng cộng [h, w] cho biên dọc của ``y`` [h, w] (cột lệch ``c0``); 0 ở mọi điểm không sửa."""
        blk = self.blk
        nb = (w - c0) // blk
        yb = y[:, c0 : c0 + nb * blk].reshape(h, nb, blk)
        f1, f0 = yb[:, :-1, blk - 2], yb[:, :-1, blk - 1]
        g0, g1 = yb[:, 1:, 0], yb[:, 1:, 1]
        step = g0 - f0
        w_edge = torch.clamp(1.0 - torch.abs(step) * self.inv_alpha, 0.0, 1.0)
        w_tex = torch.clamp(1.0 - torch.maximum(torch.abs(f1 - f0), torch.abs(g1 - g0)) * self.inv_beta, 0.0, 1.0)
        corr = self.strength * w_edge * w_tex * step
        inner = self.inner * corr
        z1 = torch.zeros(h, 1)
        zn = torch.zeros(h, nb)
        right = {blk - 1: corr, blk - 2: inner}  # điểm bên trái biên (khối i): cộng
        left = {0: -corr, 1: -inner}  # điểm bên phải biên (khối i + 1): trừ
        cols = [torch.cat([right[j], z1], 1) if j in right else torch.cat([z1, left[j]], 1) if j in left else zn
                for j in range(blk)]
        delta = torch.stack(cols, -1).reshape(h, nb * blk)
        parts = [torch.zeros(h, c0)] if c0 else []
        parts += [delta] + ([torch.zeros(h, w - c0 - nb * blk)] if w - c0 - nb * blk else [])
        return torch.cat(parts, 1) if len(parts) > 1 else delta

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        y = _sat8(_luma(p[0], p[1], p[2]))
        dv = self._delta(y, self.phase, self.h, self.w)
        p = _sat8(p + dv)
        y = _sat8(y + dv)
        dh = self._delta(y.transpose(0, 1), 0, self.w, self.h).transpose(0, 1)
        return _sat8(p + dh)


def _ramp8(v: torch.Tensor, start: float, ramp: float) -> torch.Tensor:
    """``classic._ramp_u8``: ``addWeighted(v, k, v, 0, −start·k, CV_8U)`` với ``k = 255/ramp``."""
    k = 255.0 / ramp
    return _sat8(v * k + (-start * k))


class Demagenta(nn.Module):
    """``classic.demagenta``; hai ``break`` theo ``countNonZero`` bỏ được vì lượt có cổng toàn 0 là phép đồng nhất."""

    def __init__(self, params: dict, h: int, w: int) -> None:
        super().__init__()
        self.h, self.w = h, w
        m = params["magenta"]
        self.m = {k: float(v) for k, v in m.items()}
        self.passes = int(m["passes"])
        self.dil = int(m["dilate_px"])
        self.blur = int(m["blur_px"])
        self.gk = gauss_kernel(float(m["texture_sigma"]), depth_8u=True)
        self.g_comp = float(np.float32(C._LUMA_G_COMP))

    def gate_active(self, p: torch.Tensor) -> torch.Tensor:
        """Lượt đầu có điểm qua cổng bão hoà × hồng không (= ``countNonZero(gate) > 0`` của classic)."""
        m = self.m
        b, g, r = p[0], p[1], p[2]
        sat = _ramp8(torch.maximum(r, b), m["sat_level"], m["sat_ramp"])
        pink = _ramp8(torch.clamp(torch.minimum(r, b) - g, min=0.0), m["excess_min"], m["excess_ramp"])
        return (_sat8(sat * pink * _INV255) > 0).any()

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        m = self.m
        b, g, r = p[0], p[1], p[2]
        sat_full = m["sat_level"] + m["sat_ramp"]
        for _ in range(self.passes):
            mx = torch.maximum(r, b)
            sat = _ramp8(mx, m["sat_level"], m["sat_ramp"])
            pink = _ramp8(torch.clamp(torch.minimum(r, b) - g, min=0.0), m["excess_min"], m["excess_ramp"])
            gate = _sat8(sat * pink * _INV255)
            alpha = _sat8(_box_mean(_dilate(gate.unsqueeze(0), self.dil), self.blur))[0]
            excess = torch.clamp(_sat8(r * 0.5 + b * 0.5) - g, min=0.0)
            delta = _sat8(alpha * excess * (m["strength"] / 255.0))
            tex_w = _sat8(alpha * _ramp8(mx, sat_full, m["texture_sat_ramp"]) * _INV255)
            hp = g - _sat8(_sep_filter(g, self.gk))
            inj = torch.round(hp * tex_w * (m["texture_gain"] / 255.0))
            g_up = _sat8(delta * self.g_comp)
            b = _sat8(b - delta + inj)
            r = _sat8(r - delta + inj)
            g = _sat8(g + g_up + inj)
        return torch.stack([b, g, r])


class FlatField(nn.Module):
    """``classic.flat_field``: nhân bản đồ bù tối góc của chính ``classic._vignette_gain`` (hằng trong graph)."""

    def __init__(self, params: dict, h: int, w: int) -> None:
        super().__init__()
        gain = C._vignette_gain(h, w, float(params["vignette_focal_px"]), float(params["vignette"]["max_gain"]))
        self.register_buffer("gain", torch.from_numpy(np.array(gain, np.float32)), persistent=False)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return _sat8(p * self.gain)


class Dehaze(nn.Module):
    """``classic.dehaze``: dark channel → A từng kênh (phân vị bằng ``TopK``) → ``t`` qua fast guided filter → phóng
    song tuyến → cận dưới từng điểm → chặn mức đen bằng ``Where`` (không ``if``) → ``J = (I − A)·k/t + k·A + off``."""

    def __init__(self, params: dict, h: int, w: int) -> None:
        super().__init__()
        d = params["dehaze"]
        self.d = d
        self.s = int(d["stat_stride"])
        self.patch = int(d["patch_px"])
        scale = float(d["guided_scale"])
        size = (max(1, round(w * scale)), max(1, round(h * scale)))  # đúng biểu thức của classic.dehaze
        if w % size[0] or h % size[1] or w // size[0] != h // size[1]:
            raise ValueError(f"guided_scale {scale}: INTER_AREA {w}×{h} → {size[0]}×{size[1]} không phải hệ số nguyên "
                             "đều hai chiều; graph chỉ dựng phép thu nhỏ nguyên (AveragePool)")
        self.factor = w // size[0]
        self.radius = max(1, round(int(d["guided_radius"]) * scale))
        self.omega = float(d["omega"])
        self.lo_a, self.hi_a = (float(v) for v in d["airlight_range"])
        self.h, self.w = h, w
        self.n_stat = (-(-h // self.s)) * (-(-w // self.s))  # số điểm lưới lấy mẫu [::s, ::s]
        self.force_branch: bool | None = None
        """Chỉ cho test: ghim nhánh mức đen thành hằng (mô phỏng graph bị trace cứng) để chứng minh test parity bắt được."""

    def _guided(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        k = 2 * self.radius + 1
        mean_g = _box_mean(guide, k)
        mean_s = _box_mean(src, k)
        cov = _box_mean(guide * src, k) - mean_g * mean_s
        var = _box_mean(guide * guide, k) - mean_g * mean_g
        a = cov / (var + float(self.d["guided_eps"]))
        b = mean_s - a * mean_g
        return _box_mean(a, k) * guide + _box_mean(b, k)

    def black_branch(self, p: torch.Tensor) -> torch.Tensor:
        """``lo_out < floor`` (nhánh kéo giãn mức đen) dạng tensor bool — để test chọn khung kích hoạt từng nhánh."""
        return self._core(p)[3]

    def _core(self, p: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s, f = self.s, self.factor
        mn = torch.minimum(torch.minimum(p[0], p[1]), p[2]).unsqueeze(0)
        dc = _erode(mn, self.patch)
        dc_s = dc[:, ::s, ::s]
        x_s = p[:, ::s, ::s]
        n_s = self.n_stat
        sel = (dc_s >= _percentile(dc_s.reshape(-1), float(self.d["airlight_percentile"]), n_s)).float()
        air = torch.clamp((x_s * sel).sum((1, 2)) / sel.sum(), self.lo_a, self.hi_a)  # [3]
        coef = -self.omega / air.mean()
        dc_small = _sat8(F.avg_pool2d(dc.unsqueeze(0), f).squeeze(0))
        t0 = dc_small * coef + 1.0
        small = _sat8(F.avg_pool2d(p.unsqueeze(0), f).squeeze(0))
        guide = _luma(small[0:1], small[1:2], small[2:3]) * _INV255
        t_small = torch.clamp(self._guided(guide, t0), float(self.d["t_min"]), 1.0)
        t = F.interpolate(t_small.unsqueeze(0), size=(self.h, self.w), mode="bilinear", align_corners=False).squeeze(0)
        t = torch.maximum(t, mn * coef + 1.0)
        bp = float(self.d["black_percentile"])
        inv_s = 1.0 / t[:, ::s, ::s]
        j = (x_s - air[:, None, None]) * inv_s + air[:, None, None]
        j_min = torch.minimum(torch.minimum(j[0], j[1]), j[2])
        lo_in = _percentile(mn[:, ::s, ::s].reshape(-1), bp, n_s)
        lo_out = _percentile(j_min.reshape(-1), bp, n_s)
        floor = torch.clamp(lo_in, max=float(self.d["black_target"]))
        cond = lo_out < floor if self.force_branch is None else torch.tensor(self.force_branch)
        k = torch.where(cond, (255.0 - floor) / torch.clamp(255.0 - lo_out, min=1e-6), torch.ones_like(lo_out))
        off = torch.where(cond, floor - lo_out * k, torch.zeros_like(lo_out))
        return t, air, k, cond, off

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        t, air, k, _cond, off = self._core(p)
        gain = k / t
        return _sat8((p - air[:, None, None]) * gain + (k * air + off)[:, None, None])


class Sharpen(nn.Module):
    """``classic.sharpen``: unsharp trên Y có ngưỡng theo σ nhiễu Immerkær, chặn vọt lố trong [min, max] 3×3."""

    def __init__(self, params: dict, h: int, w: int) -> None:
        super().__init__()
        s = params["sharpen"]
        self.s = {k: float(v) for k, v in s.items()}
        self.gk = gauss_kernel(float(s["sigma"]), depth_8u=False)
        self.h, self.w = h, w
        rh, rw = h - 2 * (h // 4), w - 2 * (w // 4)  # vùng giữa của classic.sharpen
        self.imm_scale = math.sqrt(math.pi / 2) / (6 * (rw - 2) * (rh - 2))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        h, w = self.h, self.w
        y = _luma(p[0:1], p[1:2], p[2:3])  # [1, H, W]
        reg = y[:, h // 4 : h - h // 4, w // 4 : w - w // 4]
        rh, rw = h - 2 * (h // 4), w - 2 * (w // 4)
        conv = sum(reg[:, i : i + rh - 2, j : j + rw - 2] * _IMMERKAER[i][j]
                   for i in range(3) for j in range(3) if _IMMERKAER[i][j])
        sigma_n = torch.clamp(torch.abs(conv).sum() * self.imm_scale, min=self.s["noise_floor"])
        det = y - _sep_filter(y, self.gk)
        thr = self.s["coring_k"] * sigma_n
        det = (det - torch.clamp(det, min=-thr, max=thr)) * self.s["amount"]
        y8 = _sat8(torch.abs(y))
        det = det + y
        det = torch.maximum(torch.minimum(det, _dilate(y8, 3)), _erode(y8, 3))
        det = det - y
        return _sat8(p + det)


def build_stages(params: dict, h: int, w: int) -> nn.ModuleDict:
    """Năm tầng theo ``STAGE_NAMES`` cho phần đã cắt mép cỡ ``h × w``."""
    mods = {"deblock": Deblock(params, h, w), "magenta": Demagenta(params, h, w), "vignette": FlatField(params, h, w),
            "dehaze": Dehaze(params, h, w), "sharpen": Sharpen(params, h, w)}
    if tuple(mods) != STAGE_NAMES:
        raise RuntimeError(f"thứ tự tầng {tuple(mods)} khác classic.STAGES {STAGE_NAMES}")
    return nn.ModuleDict(mods)


class ClassicGraph(nn.Module):
    """Graph hợp đồng §3: ``frame`` uint8 [480, 640, 3] BGR → cắt ``[crop_left, 640)`` → năm tầng → uint8 → đệm cột
    ``[0, crop_left)`` bằng lặp cột ``crop_left`` → ``frame_out`` uint8 [480, 640, 3] BGR."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.crop_left = int(params["crop_left"])
        self.stages = build_stages(params, contract.FRAME_H, contract.FRAME_W - self.crop_left)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        p = frame[:, self.crop_left :, :].permute(2, 0, 1).float()
        for st in self.stages.values():
            p = st(p)
        y = p.permute(1, 2, 0).to(torch.uint8)
        return torch.cat([y[:, :1, :].expand(-1, self.crop_left, -1), y], dim=1)


class StageGraph(nn.Module):
    """Một tầng riêng để đo parity từng tầng: ``planes`` uint8 [3, H, W] (B, G, R đã cắt mép) → ``planes_out``."""

    def __init__(self, stage: nn.Module) -> None:
        super().__init__()
        self.stage = stage

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        return self.stage(planes.float()).to(torch.uint8)


# ----------------------------------------------------------------------------- xuất + đo trong ORT

PARITY_MAE_MAX = 1.0
PARITY_ABS_MAX = 8
GPU_LOCK_TIMEOUT_S = 6 * 3600
RUNS_DIR = Path("D:/DevCache/runs/xla/v2/classic")
REPORT_REL = Path("results/v2/classic/export.json")
DEFAULT_PARAMS_REL = C.CONFIG_REL


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_onnx(module: nn.Module, dummy: torch.Tensor, path: Path, inp: str, out: str) -> None:
    """``torch.onnx.export`` (TorchScript, opset 17) trên CPU + ``onnx.checker``; output phải có shape tĩnh bằng shape
    PyTorch trả trên ``dummy`` (hợp đồng §3 ghi shape cố định), không thì lỗi."""
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        want = list(module.eval()(dummy).shape)
        torch.onnx.export(module, (dummy,), str(path), opset_version=contract.OPSET, input_names=[inp],
                          output_names=[out], do_constant_folding=True, dynamo=False)
    import onnxruntime as ort

    model = onnx.load(str(path))
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(dims) != len(want) or any(d.HasField("dim_value") and d.dim_value != v for d, v in zip(dims, want, strict=False)):
        raise RuntimeError(f"{path.name}: output shape {dims} khác {want}")
    for d, v in zip(dims, want, strict=True):  # Pad với pads tính ra làm mất shape tĩnh: ghi lại shape đã biết
        d.ClearField("dim_param")
        d.dim_value = v
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = list(sess.run(None, {inp: dummy.numpy()})[0].shape)
    if got != want:
        raise RuntimeError(f"{path.name}: ORT trả shape {got}, khác {want}")


def export_graph(params: dict, path: Path, trace_frame: np.ndarray) -> None:
    """Xuất graph đủ §3, trace trên ``trace_frame`` (khung dev riêng, không nằm trong tập parity/golden)."""
    export_onnx(ClassicGraph(params), torch.from_numpy(np.ascontiguousarray(trace_frame)), path, contract.INPUT_NAME,
                "frame_out")


def ort_run(sess: Any, x: np.ndarray) -> np.ndarray:
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def cuda_session(path: Path) -> Any:
    """ORT CUDA EP; đọc ``get_providers()`` SAU một lần chạy thử (thiếu cuDNN thì ORT vẫn báo CUDA rồi chạy CPU)."""
    sess = ort_session(path)
    inp = sess.get_inputs()[0]
    shape = [int(d) for d in inp.shape]
    ort_run(sess, np.zeros(shape, np.uint8))
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA EP không chạy, providers {sess.get_providers()}")
    return sess


def _diff(a: np.ndarray, b: np.ndarray) -> tuple[float, int, float]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), int(d.max()), float((d > 0).mean())


def _summ(rows: list[tuple[float, int, float]]) -> dict[str, Any]:
    a = np.asarray(rows, float)
    return {"mae_mean": round(float(a[:, 0].mean()), 6), "mae_worst": round(float(a[:, 0].max()), 6),
            "abs_max": int(a[:, 1].max()), "frac_px_diff_worst": round(float(a[:, 2].max()), 6),
            "pass": bool(a[:, 0].max() <= PARITY_MAE_MAX and a[:, 1].max() <= PARITY_ABS_MAX)}


def stage_inputs(frames: list[np.ndarray], params: dict) -> dict[str, list[np.ndarray]]:
    """Đầu vào của từng tầng theo chuỗi numpy (mỗi tầng đo riêng, sai số không cộng dồn) + ra numpy của tầng."""
    fn = dict(C.STAGES)
    c = int(params["crop_left"])
    ins: dict[str, list[np.ndarray]] = {n: [] for n in STAGE_NAMES}
    refs: dict[str, list[np.ndarray]] = {n: [] for n in STAGE_NAMES}
    for f in frames:
        p = C.to_planes(f[:, c:])
        for n in STAGE_NAMES:
            ins[n].append(np.ascontiguousarray(np.stack(p)))
            p = fn[n](p, params)
            refs[n].append(np.stack(p))
    return {"in": ins, "ref": refs}  # type: ignore[dict-item]


def export_stages(params: dict, out_dir: Path) -> dict[str, Path]:
    h, w = contract.FRAME_H, contract.FRAME_W - int(params["crop_left"])
    stages = build_stages(params, h, w)
    paths = {}
    for n, mod in stages.items():
        paths[n] = out_dir / f"stage_{n}.onnx"
        export_onnx(StageGraph(mod), torch.zeros(3, h, w, dtype=torch.uint8), paths[n], "planes", "planes_out")
    return paths


def stage_parity(paths: dict[str, Path], si: dict[str, Any], provider: str) -> dict[str, Any]:
    out = {}
    for n, p in paths.items():
        sess = cuda_session(p) if provider == "cuda" else ort_session(p, ["CPUExecutionProvider"])
        out[n] = _summ([_diff(ort_run(sess, x), r) for x, r in zip(si["in"][n], si["ref"][n], strict=True)])
    return out


def branch_coverage(frames: list[np.ndarray], params: dict) -> dict[str, int]:
    """Số khung kích hoạt từng nhánh dữ liệu (đo bằng các tầng PyTorch trên đầu vào numpy của tầng đó)."""
    h, w = contract.FRAME_H, contract.FRAME_W - int(params["crop_left"])
    st = build_stages(params, h, w)
    si = stage_inputs(frames, params)
    with torch.no_grad():
        black = sum(bool(st["dehaze"].black_branch(torch.from_numpy(x).float())) for x in si["in"]["dehaze"])
        pink = sum(bool(st["magenta"].gate_active(torch.from_numpy(x).float())) for x in si["in"]["magenta"])
    n = len(frames)
    return {"n": n, "dehaze_black_stretch": black, "dehaze_no_stretch": n - black, "magenta_gate_active": pink,
            "magenta_gate_empty": n - pink}


CPU_THREADS = 4
"""Luồng ORT CPU / cv2 khi đo (luật máy: job CPU tối đa 4 worker)."""


def cpu_session(path: Path) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = CPU_THREADS
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def pick_frames(parity_n: int) -> dict[str, Any]:
    """Khung dev cho parity (cách đều), golden (3 khung) và trace (1 khung): ba tập rời nhau, không khung test."""
    from xla.enhance.export_onnx import dev_frames, test_frame_names

    parity = dev_frames(parity_n)
    golden: list[str] = []
    for off in range(parity_n // 2 + 7, parity_n // 2 + 7 + 200):
        golden = dev_frames(3, offset=off)
        if not set(golden) & set(parity):
            break
    used = set(parity) | set(golden)
    trace = next(n for off in range(1, 2000) for n in dev_frames(1, offset=off) if n not in used)
    if (used | {trace}) & test_frame_names() or set(golden) & set(parity) or len(set(golden)) != 3:
        raise RuntimeError("chọn khung lỗi: chồng nhau hoặc rơi vào test")
    return {"parity": parity, "golden": golden, "trace": trace}


def dev_quality(raw: list[np.ndarray], out: list[np.ndarray], crop_left: int) -> dict[str, float]:
    """Chỉ số so sánh lai ↔ classic trên dev: dark channel trung vị (trung vị theo khung của
    ``classic.dark_channel_median``), % điểm hồng (luật hồ sơ, ``classic.pink_count``), và mức giảm so với gốc."""
    npx = contract.FRAME_H * (contract.FRAME_W - crop_left)

    def stats(fs: list[np.ndarray]) -> tuple[float, float]:
        return (float(np.median([C.dark_channel_median(f, crop_left) for f in fs])),
                float(np.mean([C.pink_count(f, crop_left) / npx for f in fs])))

    dr, pr = stats(raw)
    do, po = stats(out)
    return {"dark_channel_median_raw": dr, "dark_channel_median": do, "dark_channel_drop": (dr - do) / max(dr, 1e-9),
            "pink_frac_raw": pr, "pink_frac": po, "pink_frac_drop": (pr - po) / max(pr, 1e-12)}


def flicker_dev(fns: dict[str, Any], crop_left: int, max_frames: int = 0) -> dict[str, Any]:
    """``flicker_ratio`` (``xla.eval.temporal``) trên mọi chuỗi dev liền kề cho từng hàm ``khung → khung``; gộp đúng như
    ``flicker_ratio`` nhiều chuỗi (Σ|Δ| theo cặp), xử lý từng chuỗi để không giữ cả nghìn khung trong RAM."""
    from xla.enhance.train_net import dev_runs
    from xla.eval.temporal import mean_abs_diff

    fdir = repo_root() / "data" / "real" / "frames"
    runs = dev_runs("dev")
    den = 0.0
    num = dict.fromkeys(fns, 0.0)
    n_frames = 0
    for run in runs:
        if max_frames and n_frames >= max_frames:
            break
        raw = [contract.decode_frame(fdir / n) for n in run]
        pairs = len(raw) - 1
        den += mean_abs_diff(raw, crop_left) * pairs
        for k, fn in fns.items():
            num[k] += mean_abs_diff([fn(f) for f in raw], crop_left) * pairs
        n_frames += len(raw)
    return {"n_runs": len(runs), "n_frames": n_frames, **{k: v / den for k, v in num.items()}}


def numpy_ms(frame: np.ndarray, params: dict, warmup: int = 30, iters: int = 300) -> dict[str, float]:
    for _ in range(warmup):
        C.enhance(frame, params)
    t = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        C.enhance(frame, params)
        t[i] = (time.perf_counter() - t0) * 1000.0
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)), "mean": float(t.mean()), "n": iters}


def machine() -> dict[str, Any]:
    import platform

    import onnxruntime as ort

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"cpu": C._cpu_name(), "logical_cpus": __import__("os").cpu_count(), "gpu": gpu,
            "onnxruntime": ort.__version__, "torch": torch.__version__, "opencv": cv2.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "ort_cpu_threads": CPU_THREADS,
            "cv2_threads": CPU_THREADS}


def _round_ms(ms: dict[str, float]) -> dict[str, Any]:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ms.items()}


def camera_profile_path(params_path: Path) -> Path:
    """``camera_profile`` ghi trong config classic (tương đối gốc repo, như ``classic.load_params``)."""
    import yaml

    rel = Path(str(yaml.safe_load(params_path.read_text(encoding="utf-8"))["camera_profile"]))
    return rel if rel.is_absolute() else repo_root() / rel


def _golden(cpu: Any, params: dict, sources: list[str], gdir: Path) -> list[tuple[float, int, float]]:
    """3 khung golden: JPEG gốc camera + kết quả của chính file .onnx trên ORT CPU (``*.out.png``, ``expected.json``)."""
    fdir = repo_root() / "data" / "real" / "frames"
    expected: dict[str, Any] = {}
    par = []
    for gname, src in zip(contract.GOLDEN_FRAMES, sources, strict=True):
        shutil.copyfile(fdir / src, gdir / gname)
        f = contract.decode_frame(gdir / gname)
        o = ort_run(cpu, f)
        par.append(_diff(o, C.enhance(f, params)))
        ok, buf = cv2.imencode(".png", o)
        if not ok:
            raise RuntimeError(f"không mã hoá được PNG {gname}")
        (gdir / gname.replace(".jpg", ".out.png")).write_bytes(buf.tobytes())
        expected[gname] = {"sha256_out": hashlib.sha256(o.tobytes()).hexdigest(),
                           "mean": o.reshape(-1, 3).mean(0).tolist()}
    (gdir / contract.EXPECTED_FILE).write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return par


def _cuda_part(rep: dict[str, Any], stage_paths: dict[str, Path], si: dict[str, Any], onnx_path: Path,
               frames: list[np.ndarray], ref: list[np.ndarray], got_cpu: list[np.ndarray], label: str) -> None:
    """Phần GPU (giữ ``xla.gpu_lock`` chỉ trong lúc này): parity từng tầng + đủ graph trên ORT CUDA EP, ms p50/p95."""
    import onnxruntime as ort

    from xla.enhance.net import gpu_state, measure_ort_ms
    from xla.gpu_lock import gpu_lock

    t_wait = time.time()
    with gpu_lock(label, timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        rep["parity"]["stages"]["ort_cuda"] = stage_parity(stage_paths, si, "cuda")
        cuda = cuda_session(onnx_path)
        got = [ort_run(cuda, f) for f in frames]
        rep["gpu_state_before"] = gpu_state()
        ms = measure_ort_ms(cuda, {contract.INPUT_NAME: frames[0]})
        rep["gpu_state_after"] = gpu_state()
        rep["cuda_providers"] = cuda.get_providers()
        del cuda
        rep["gpu_seconds"] = round(time.time() - t0, 1)
        rep["wait_lock_seconds"] = round(t0 - t_wait, 1)
    full = rep["parity"]["full"]
    full["ort_cuda_vs_numpy"] = _summ([_diff(a, b) for a, b in zip(got, ref, strict=True)])
    full["ort_cuda_vs_ort_cpu"] = _summ([_diff(a, b) for a, b in zip(got, got_cpu, strict=True)])
    rep["ms_per_frame"] = {"device": rep["machine"]["gpu"] or "?",
                           "runtime": f"onnxruntime-gpu {ort.__version__} CUDA EP", **_round_ms(ms)}


def export_bundle(params_path: str | Path, out_dir: str | Path, *, parity_n: int = 50, skip_cuda: bool = False,
                  flicker_max_frames: int = 0) -> dict[str, Any]:
    """Xuất graph + 5 graph từng tầng, đo parity (ORT CPU, CUDA) trên khung dev ≠ khung trace, golden, ms, nhấp nháy,
    ghi ``meta.json`` rồi kiểm bằng ``contract.validate_bundle``. Trả báo cáo (dict JSON được)."""
    from xla.enhance.net import cpu_perf_check, measure_ort_ms

    params_path = Path(params_path)
    params_path = params_path if params_path.is_absolute() else repo_root() / params_path
    params = C.load_params(params_path)
    out = Path(out_dir)
    out = out if out.is_absolute() else repo_root() / out
    m = contract.BUNDLE_DIR_RE.match(out.name)
    if not m or m.group(1) != "enhancer":
        raise ValueError(f"--out phải là thư mục enhancer-vX.Y.Z, nhận {out.name}")
    c = int(params["crop_left"])
    cv2.setNumThreads(CPU_THREADS)
    sel = pick_frames(parity_n)
    fdir = repo_root() / "data" / "real" / "frames"
    frames = [contract.decode_frame(fdir / n) for n in sel["parity"]]
    onnx_path = out / "enhancer.onnx"
    gdir = out / "golden"
    gdir.mkdir(parents=True, exist_ok=True)
    t_cpu = time.time()
    export_graph(params, onnx_path, contract.decode_frame(fdir / sel["trace"]))
    stage_paths = export_stages(params, RUNS_DIR / out.name / "stages")
    si = stage_inputs(frames, params)
    ref = [C.enhance(f, params) for f in frames]
    cpu = cpu_session(onnx_path)
    got_cpu = [ort_run(cpu, f) for f in frames]
    rep: dict[str, Any] = {
        "bundle": str(out), "params": params_path.name, "frames": sel, "precision": "fp32", "opset": contract.OPSET,
        "tolerance": {"mae_max": PARITY_MAE_MAX, "abs_max": PARITY_ABS_MAX},
        "reference": "numpy xla.enhance.classic (SSOT) cùng khung; từng tầng: vào = ra numpy của tầng trước",
        "branches": branch_coverage(frames, params),
        "parity": {"stages": {"ort_cpu": stage_parity(stage_paths, si, "cpu")},
                   "full": {"ort_cpu_vs_numpy": _summ([_diff(a, b) for a, b in zip(got_cpu, ref, strict=True)])}},
        "machine": machine(),
    }
    rep["parity"]["golden_ort_cpu_vs_numpy"] = _summ(_golden(cpu, params, sel["golden"], gdir))
    rep["quality_dev"] = {"numpy": dev_quality(frames, ref, c), "onnx_cpu": dev_quality(frames, got_cpu, c)}
    rep["flicker_dev"] = flicker_dev({"numpy": lambda f: C.enhance(f, params), "onnx_cpu": lambda f: ort_run(cpu, f)},
                                     c, flicker_max_frames)
    rep["cpu_perf_before"] = cpu_perf_check(10)
    rep["ms_cpu"] = {"onnx_ort_cpu": _round_ms(measure_ort_ms(cpu, {contract.INPUT_NAME: frames[0]})),
                     "numpy_classic": _round_ms(numpy_ms(frames[0], params))}
    rep["cpu_perf_after"] = cpu_perf_check(10)
    rep["cpu_seconds"] = round(time.time() - t_cpu, 1)
    if not skip_cuda:
        _cuda_part(rep, stage_paths, si, onnx_path, frames, ref, got_cpu, f"v2 phase10 classic export {out.name}")
    meta = bundle_meta(rep, params, params_path, onnx_path, m.group(2), sel)
    (out / contract.META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    errors = contract.validate_bundle(out)
    par = rep["parity"]
    passes = [v["pass"] for v in par["full"].values()] + [par["golden_ort_cpu_vs_numpy"]["pass"]]
    passes += [v["pass"] for prov in par["stages"].values() for v in prov.values()]
    rep.update(errors=errors, parity_ok=all(passes), onnx_sha256=meta["onnx_sha256"],
               classic_params_sha256=meta["classic_params_sha256"])
    return rep


def bundle_meta(rep: dict[str, Any], params: dict, params_path: Path, onnx_path: Path, version: str,
                sel: dict[str, Any]) -> dict[str, Any]:
    """``meta.json`` §4 + §12 (``network: "classic"``, ``classic_params_sha256``) + khoá thêm tuỳ chọn."""
    prof_path = camera_profile_path(params_path)
    msf = rep.get("ms_per_frame", {"device": "không đo (--skip-cuda)", "runtime": "không đo", "p50": -1.0, "p95": -1.0})
    return {
        "contract_version": contract.CONTRACT_VERSION,
        "model": "enhancer",
        "version": version,
        "onnx_sha256": contract.sha256_file(onnx_path),
        "frame_size": [contract.FRAME_W, contract.FRAME_H],
        "crop_left": int(params["crop_left"]),
        "camera_profile_version": str(contract.load_profile(prof_path)["profile_version"]),
        "train_run": f"classic {params_path.name}@{params['version']} (không huấn luyện)",
        "metrics": {k: None for k in contract.METRICS_KEYS["enhancer"]},
        "ms_per_frame": {k: msf[k] for k in ("device", "runtime", "p50", "p95")},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "network": "classic",
        "classic_params_sha256": file_sha256(params_path),
        # khoá thêm (ngoài hợp đồng, tuỳ chọn):
        "camera_profile_sha256": file_sha256(prof_path),
        "classic_params_version": params["version"],
        "precision": "fp32",
        "metrics_note": "metrics (tập test thật) = null: lane classic không đụng khung test; phase 14 đo",
        "metrics_dev": {**rep["quality_dev"]["onnx_cpu"], "flicker_ratio": rep["flicker_dev"]["onnx_cpu"],
                        "split": "dev"},
        "ms_per_frame_cpu": {"runtime": f"onnxruntime CPU EP, {CPU_THREADS} luồng", **rep["ms_cpu"]["onnx_ort_cpu"]},
        "bench_machine": rep["machine"],  # "machine" là khoá riêng của enhancer-offline (§11)
        "parity": {"vs": "numpy xla.enhance.classic", "n_frames": len(sel["parity"]), **rep["parity"]["full"]},
        "trace_frame": sel["trace"],
        "golden_sources": dict(zip(contract.GOLDEN_FRAMES, sel["golden"], strict=True)),
    }


def stale_reasons(meta: dict[str, Any], params_path: str | Path | None = None) -> list[str]:
    """Bundle classic còn khớp tham số hiện tại không: ``classic_params_sha256`` = sha256 của yaml (§12) và
    ``camera_profile_sha256`` = sha256 của hồ sơ camera mà yaml trỏ tới. Rỗng = khớp; đổi tham số mà chưa xuất lại → lỗi."""
    pp = Path(params_path) if params_path is not None else repo_root() / DEFAULT_PARAMS_REL
    pp = pp if pp.is_absolute() else repo_root() / pp
    out = []
    if meta.get("network") != "classic":
        out.append(f"meta.network = {meta.get('network')!r}, cần 'classic'")
    if meta.get("classic_params_sha256") != file_sha256(pp):
        out.append(f"classic_params_sha256 khác sha256 của {pp.name}: đổi tham số mà chưa xuất lại bundle")
    if "camera_profile_sha256" in meta and meta["camera_profile_sha256"] != file_sha256(camera_profile_path(pp)):
        out.append("camera_profile_sha256 khác hồ sơ camera hiện tại: xuất lại bundle")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["hybrid"]:
        return hybrid_main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.export_classic", description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=str(DEFAULT_PARAMS_REL), help="config classic (mặc định configs/enhance_classic.yaml)")
    ap.add_argument("--out", required=True, help="thư mục bundle enhancer-vX.Y.Z")
    ap.add_argument("--parity-n", type=int, default=50, help="số khung dev đo parity (khác khung trace/golden)")
    ap.add_argument("--flicker-max-frames", type=int, default=0, help="0 = mọi chuỗi dev liền kề")
    ap.add_argument("--skip-cuda", action="store_true", help="không đo ORT CUDA (không giữ khoá GPU)")
    ap.add_argument("--report", default=str(REPORT_REL))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.parity_n < 3:
        ap.error("--parity-n ≥ 3")
    pp = Path(args.params)
    if not (pp if pp.is_absolute() else repo_root() / pp).is_file():
        print(f"lỗi: không có config {args.params}", file=sys.stderr)
        return 2
    try:
        rep = export_bundle(pp, args.out, parity_n=args.parity_n, skip_cuda=args.skip_cuda,
                            flicker_max_frames=args.flicker_max_frames)
    except ValueError as e:
        print(f"lỗi: {e}", file=sys.stderr)
        return 2
    rp = Path(args.report)
    rp = rp if rp.is_absolute() else repo_root() / rp
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False) if args.json else json.dumps(rep, indent=2, ensure_ascii=False))
    return 0 if not rep["errors"] and rep["parity_ok"] else 1


# ----------------------------------------------------------------------------- bước 4: lai classic → NAFNet (có trần)

HYBRID_REPORT_REL = Path("results/v2/classic/hybrid.json")
HYBRID_RULE = ("hơn classic = không kém ở cả ba chỉ số dev (dark channel trung vị, % điểm hồng, flicker_ratio; thấp hơn "
               "là tốt hơn) và tốt hơn ít nhất một; thêm tổng ms ORT CUDA ≤ ms_max mới được A/B")
_POOL_PARAMS: dict | None = None


def _pool_init(params_path: str) -> None:
    global _POOL_PARAMS
    cv2.setNumThreads(1)
    _POOL_PARAMS = C.load_params(params_path)


def _classic_file(job: tuple[str, str, int]) -> str:
    src, dst, q = job
    out = C.enhance(contract.decode_frame(Path(src)), _POOL_PARAMS)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if not ok:
        raise RuntimeError(f"không mã hoá được {dst}")
    Path(dst).write_bytes(buf.tobytes())
    return dst


def build_classic_input_set(cfg: dict[str, Any], params_path: Path) -> dict[str, Any]:
    """``classic(D2)`` cho mọi ảnh ``train``/``val`` của ``data.target_set`` ∩ D2, ghi ``<data_root>/<input_set>``
    (4 tiến trình, bỏ ảnh đã có cùng ``classic_params_sha256``)."""
    from multiprocessing import Pool

    from xla.paths import data_root

    root = data_root()
    dst_root = root / cfg["data"]["input_set"]
    key = {"classic_params_sha256": file_sha256(params_path), "source": "d2",
           "jpeg_quality": int(cfg["hybrid"]["input_jpeg_quality"]), "sampling": "4:4:4"}
    key_path = dst_root / "build_key.json"
    if key_path.is_file() and json.loads(key_path.read_text(encoding="utf-8")) != key:
        raise RuntimeError(f"{dst_root} dựng từ tham số khác ({key_path}); xoá thư mục rồi chạy lại")
    jobs = []
    for split in ("train", "val"):
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        for src in sorted((root / "d2" / split / "images").glob("*.jpg")):
            dst = dst_root / split / "images" / src.name
            if not dst.is_file():
                jobs.append((str(src), str(dst), key["jpeg_quality"]))
    t0 = time.time()
    if jobs:
        with Pool(4, initializer=_pool_init, initargs=(str(params_path),)) as pool:
            for _ in pool.imap_unordered(_classic_file, jobs, chunksize=16):
                pass
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    counts = {s: len(list((dst_root / s / "images").glob("*.jpg"))) for s in ("train", "val")}
    return {"dir": str(dst_root), "built": len(jobs), "counts": counts, "seconds": round(time.time() - t0, 1), **key}


def _hybrid_eval(ckpts: list[Path], params: dict, cap_left_s: float, parity_n: int) -> dict[str, Any]:
    """Mỗi checkpoint: ``mạng(classic(khung))`` trên khung dev (``dev_quality``) + mọi chuỗi dev liền kề (flicker), so
    với classic; giữ ``xla.gpu_lock`` trong lúc đo, dừng khi hết ``cap_left_s`` giây GPU."""
    from xla.enhance.net import EnhancerGraph, enhance_frames, load_checkpoint
    from xla.gpu_lock import gpu_lock

    c = int(params["crop_left"])
    fdir = repo_root() / "data" / "real" / "frames"
    frames = [contract.decode_frame(fdir / n) for n in pick_frames(parity_n)["parity"]]
    base = [C.enhance(f, params) for f in frames]
    rows: list[dict[str, Any]] = []
    t_wait = time.time()
    with gpu_lock("v2 phase10 classic hybrid eval", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        for ck in ckpts:
            if time.time() - t0 > cap_left_s:
                rows.append({"checkpoint": ck.name, "skipped": "hết trần GPU"})
                continue
            model, meta = load_checkpoint(ck)
            graph = EnhancerGraph(model, c).cuda().eval()

            def hyb(f: np.ndarray, g: EnhancerGraph = graph) -> np.ndarray:
                return enhance_frames(g, [C.enhance(f, params)])[0]

            outs = [enhance_frames(graph, [b])[0] for b in base]
            fl = flicker_dev({"classic": lambda f: C.enhance(f, params), "hybrid": hyb}, c)
            rows.append({"checkpoint": ck.name, "step": meta.get("step"), "quality": dev_quality(frames, outs, c),
                         "flicker_ratio": fl["hybrid"], "flicker_classic": fl["classic"], "n_flicker_frames": fl["n_frames"]})
            del graph, model
            torch.cuda.empty_cache()
        gpu_s = time.time() - t0
    return {"rows": rows, "classic": dev_quality(frames, base, c), "gpu_seconds": round(gpu_s, 1),
            "wait_lock_seconds": round(t0 - t_wait, 1)}


def _beats_classic(row: dict[str, Any], classic: dict[str, Any]) -> bool:
    if "quality" not in row:
        return False
    a = (row["quality"]["dark_channel_median"], row["quality"]["pink_frac"], row["flicker_ratio"])
    b = (classic["dark_channel_median"], classic["pink_frac"], row["flicker_classic"])
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def run_hybrid(cfg_path: str | Path) -> dict[str, Any]:
    """Bước 4: dựng classic(D2) → smoke ``train_net.train`` đo s/bước → chạy tiếp số bước vừa trần → đo dev từng
    checkpoint so classic. Tổng giờ GPU (smoke + huấn luyện + đo) ≤ ``hybrid.gpu_cap_s``; không ứng viên nào hơn
    classic → không xuất ``enhancer-v2.1.0`` (ghi "không đạt trong trần")."""
    import copy
    import os

    from xla.enhance import train_net

    cfg = train_net.load_config(cfg_path)
    hy = cfg["hybrid"]
    cap = float(hy["gpu_cap_s"])
    params_path = repo_root() / hy["classic_params"]
    params = C.load_params(params_path)
    os.environ["XLA_ALLOW_THROTTLED"] = "1"  # không hoãn vì hạ xung: đo trong điều kiện hiện tại, ghi kèm (luật máy)
    rep: dict[str, Any] = {"config": str(cfg_path), "rule": HYBRID_RULE, "gpu_cap_s": cap,
                           "input_set": build_classic_input_set(cfg, params_path)}
    run_id = str(hy["run_id"])
    run_dir = train_net.RUNS_DIR / run_id
    smoke_json = run_dir / "smoke.json"
    smoke = (json.loads(smoke_json.read_text(encoding="utf-8")) if smoke_json.is_file()
             else train_net.train(copy.deepcopy(cfg), run_id, smoke=True, max_steps=None))
    rep["smoke"] = {k: smoke[k] for k in ("steps", "s_per_step", "gpu_seconds", "vram_peak_gb", "val_synthetic")}
    used = float(smoke["gpu_seconds"])
    room = cap - used - float(hy["final_val_reserve_s"]) - float(hy["eval_reserve_s"])
    more = int(room // float(smoke["s_per_step"])) // 100 * 100
    rep["plan"] = {"gpu_used_after_smoke_s": round(used, 1), "room_s": round(room, 1), "more_steps": more}
    if more < 100:
        rep.update(verdict="không đạt trong trần", reason="trần GPU không đủ cho ≥ 100 bước sau smoke", gpu_seconds=used)
        return rep
    cfg_t = copy.deepcopy(cfg)
    total = int(smoke["steps"]) + more
    cfg_t["train"]["max_steps"] = total
    cfg_t["train"]["ckpt_every"] = max(100, more // 3 // 100 * 100)
    tr = train_net.train(cfg_t, run_id, smoke=False, max_steps=total)
    rep["train"] = {k: tr[k] for k in ("steps", "s_per_step", "gpu_seconds", "vram_peak_gb", "wait_lock_seconds",
                                       "params", "run_dir")}
    rep["train"]["val_synthetic_final"] = tr["val_synthetic"][-1]
    used = float(tr["gpu_seconds"])
    ckpts = sorted(run_dir.glob("step_*.pt"))
    ev = _hybrid_eval(ckpts, params, cap - used, int(hy["parity_n"]))
    rep["eval"] = ev
    used += ev["gpu_seconds"]
    winners = [r for r in ev["rows"] if _beats_classic(r, ev["classic"])]
    rep["gpu_seconds"] = round(used, 1)
    rep["gpu_minutes"] = round(used / 60, 2)
    rep["within_cap"] = used <= cap
    if winners:
        rep.update(verdict="hơn classic trên dev: cần đo tổng ms rồi xuất enhancer-v2.1.0",
                   winners=[r["checkpoint"] for r in winners])
    else:
        rep.update(verdict="không đạt trong trần", reason="không checkpoint nào hơn classic trên dev theo luật",
                   winners=[])
    return rep


def hybrid_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.export_classic hybrid",
                                 description="bước 4: lai classic → NAFNet-w16 có trần GPU")
    ap.add_argument("--config", default="configs/restore_hybrid.yaml")
    ap.add_argument("--report", default=str(HYBRID_REPORT_REL))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not (repo_root() / args.config).is_file():
        print(f"lỗi: không có {args.config}", file=sys.stderr)
        return 2
    rep = run_hybrid(args.config)
    rp = repo_root() / args.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False) if args.json else json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
