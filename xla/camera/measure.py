"""Phép đo camera trên một khung (hàm thuần, không đọc file) — SSOT cho hồ sơ camera (phase 04), bộ làm xấu D2
(phase 05/07) và độ đo giống thật (phase 06). Phase sau gọi các hàm này, không viết lại phép đo.

Quy ước chung:

- Ảnh vào: ``numpy.ndarray`` uint8 ``[H, W, 3]`` thứ tự **BGR** (như ``cv2.imdecode``), thang 0–255.
- Độ sáng Y = ``cv2.cvtColor(BGR2GRAY)`` (Rec.601: 0,299 R + 0,587 G + 0,114 B), float32, thang 0–255.
- Mọi phép đo bỏ cột ``[0, crop_left)`` (dải tối mép trái của camera), **trừ** phép đo dải mép.
- Không xác định được (ví dụ không có điểm hồng, không đủ cạnh) → ``NaN``, không đoán.
"""

from __future__ import annotations

import math
import struct
from itertools import pairwise
from typing import Any

import cv2
import numpy as np

CROP_LEFT_DEFAULT = 16
"""Số cột bỏ ở mép trái (dải tối đo được ~13 px, lấy dư)."""

PINK_DIFF = 25
"""Luật điểm hồng/tím: R − G > 25 **và** B − G > 25 (thang 0–255)."""
DARK_CHANNEL_PATCH = 15
VIGNETTE_PATCH_FRAC = 1 / 6
"""Cạnh ô góc / ô giữa khi đo tối góc, theo tỉ lệ cạnh ảnh (đã cắt mép)."""
STRIP_SCAN_COLS = 64
STRIP_EDGE_COLS = 4
"""Mức tối của dải = trung bình Y của ``STRIP_EDGE_COLS`` cột đầu."""
STRIP_REF = (24, 64)
"""Mức tham chiếu = trung vị Y theo cột trong khoảng cột này (ngoài dải)."""
STRIP_MIN_CONTRAST = 5.0
STRIP_MEAN_REF = (4, 24)
"""Y tham chiếu của dải mép: trung bình cột ``[crop_left + 4, crop_left + 24)`` (giống ``features.py``: cột 20–40)."""

NLF_GRAD_PERCENTILE = 30
"""Vùng phẳng cho NLF: điểm có độ lớn gradient (ảnh đã làm mịn) dưới phân vị này."""
NLF_MIN_PIXELS = 500
PSD_HIGHPASS_SIGMA = 2.0
MTF_MAX_EDGES = 200
MTF_MIN_EDGES = 20
MTF_HALF_LEN = 8.0
MTF_STEP = 0.25
MTF_MIN_CONTRAST = 20.0
MTF_FFT_N = 256

MEASURE_KEYS = (
    "dark_channel_15px", "y_p1", "y_p99", "saturation_mean", "pink_frac", "pink_highlight_y", "blocking_ratio_8x8",
    "laplacian_var", "sigma_immerkaer", "sobel_x2_over_y2", "vignette_corner_center_ratio", "r_over_g", "b_over_g",
    "mtf50_cyc_per_px", "left_strip_width_px", "left_strip_mean_y",
)

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
_ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47,
    55, 62, 63,
)


def luma(img_bgr: np.ndarray) -> np.ndarray:
    """Y float32 0–255 (Rec.601)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _check(img_bgr: np.ndarray, crop_left: int) -> None:
    if img_bgr.dtype != np.uint8 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"cần uint8 [H,W,3] BGR, nhận {img_bgr.dtype} {img_bgr.shape}")
    if not 0 <= crop_left < img_bgr.shape[1] - 32:
        raise ValueError(f"crop_left={crop_left} ngoài khoảng hợp lệ cho ảnh rộng {img_bgr.shape[1]}")


def immerkaer_sigma(y: np.ndarray) -> float:
    """σ nhiễu (thang 0–255) theo Immerkær (1996): ``sqrt(π/2)/(6(W−2)(H−2)) Σ|Y * M|``."""
    h, w = y.shape
    conv = cv2.filter2D(y, -1, _IMMERKAER)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(conv).sum())


def blocking_ratio(y: np.ndarray, col_phase: int = 0) -> float:
    """Chênh lệch tuyệt đối trung bình ở ranh khối 8×8 chia cho bên trong (1 = không lộ khối).

    ``col_phase`` = số cột đã cắt trước ``y`` để lưới 8 px khớp lưới JPEG của khung gốc.
    """
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    bcols = np.zeros(dx.shape[1], bool)
    bcols[[c for c in range(dx.shape[1]) if (c + col_phase) % 8 == 7]] = True
    brows = np.zeros(dy.shape[0], bool)
    brows[7::8] = True
    bx = dx[:, bcols].mean() / max(float(dx[:, ~bcols].mean()), 1e-3)
    by = dy[brows].mean() / max(float(dy[~brows].mean()), 1e-3)
    return float((bx + by) / 2)


def left_strip(img_bgr: np.ndarray, crop_left: int = CROP_LEFT_DEFAULT) -> tuple[float, float, float]:
    """Dải tối mép trái, đo trên khung **nguyên** (không cắt).

    Trả ``(width_px, mean_y_strip, mean_y_ref)``: ``width_px`` = số cột đầu liên tiếp có Y trung bình cột dưới điểm giữa
    của mức dải (``STRIP_EDGE_COLS`` cột đầu) và mức tham chiếu (trung vị cột ``STRIP_REF``); 0 khi chênh lệch dưới
    ``STRIP_MIN_CONTRAST``. ``mean_y_strip`` = Y trung bình của các cột dải (NaN khi rộng 0); ``mean_y_ref`` = Y trung bình
    cột ``[crop_left + 4, crop_left + 24)``.
    """
    y = luma(img_bgr)
    col = y[:, :STRIP_SCAN_COLS].mean(axis=0)
    level = float(col[:STRIP_EDGE_COLS].mean())
    ref = float(np.median(col[STRIP_REF[0]:STRIP_REF[1]]))
    a, b = crop_left + STRIP_MEAN_REF[0], crop_left + STRIP_MEAN_REF[1]
    mean_ref = float(y[:, a:b].mean())
    if ref - level < STRIP_MIN_CONTRAST:
        return 0.0, float("nan"), mean_ref
    mid = (level + ref) / 2
    above = np.flatnonzero(col >= mid)
    width = int(above[0]) if above.size else STRIP_SCAN_COLS
    return float(width), float(y[:, :width].mean()), mean_ref


def vignette_ratio(y: np.ndarray) -> float:
    """Y trung bình bốn ô góc / Y trung bình ô giữa (ô vuông cạnh ``VIGNETTE_PATCH_FRAC`` × cạnh ngắn)."""
    h, w = y.shape
    s = max(4, int(min(h, w) * VIGNETTE_PATCH_FRAC))
    corners = [y[:s, :s], y[:s, -s:], y[-s:, :s], y[-s:, -s:]]
    cy, cx = h // 2, w // 2
    center = y[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]
    return float(np.mean([c.mean() for c in corners]) / max(float(center.mean()), 1e-3))


def mtf50(y: np.ndarray) -> float:
    """MTF50 (chu kỳ/px) **ước lượng từ cạnh mạnh của cảnh tự nhiên**, không phải bảng thử chuẩn.

    Lấy tối đa ``MTF_MAX_EDGES`` điểm cực đại gradient mạnh nhất, cắt profile dọc pháp tuyến (±8 px, bước 0,25 px, nội suy
    song tuyến), chuẩn hoá thành ESF 0–1, lấy trung bình, đạo hàm thành LSF (cửa sổ Hann), FFT → MTF; trả tần số đầu tiên
    MTF < 0,5 (nội suy tuyến tính), kẹp ≤ 0,5. NaN khi dưới ``MTF_MIN_EDGES`` cạnh đủ tương phản.
    """
    ys = cv2.GaussianBlur(y, (0, 0), 0.7)
    gx = cv2.Sobel(ys, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(ys, cv2.CV_32F, 0, 1, ksize=3) / 8
    mag = np.hypot(gx, gy)
    b = int(MTF_HALF_LEN) + 2
    mag_in = np.zeros_like(mag)
    mag_in[b:-b, b:-b] = mag[b:-b, b:-b]
    peak = (mag_in == cv2.dilate(mag_in, np.ones((7, 7), np.uint8))) & (mag_in > 0)
    thr = np.percentile(mag_in[b:-b, b:-b], 99)
    rr, cc = np.nonzero(peak & (mag_in >= thr))
    if rr.size == 0:
        return float("nan")
    order = np.argsort(-mag_in[rr, cc])[:MTF_MAX_EDGES]
    rr, cc = rr[order], cc[order]
    s = np.arange(-MTF_HALF_LEN, MTF_HALF_LEN + 1e-9, MTF_STEP, dtype=np.float32)
    nx = gx[rr, cc] / mag[rr, cc]
    ny = gy[rr, cc] / mag[rr, cc]
    mapx = (cc[:, None] + s[None, :] * nx[:, None]).astype(np.float32)
    mapy = (rr[:, None] + s[None, :] * ny[:, None]).astype(np.float32)
    prof = cv2.remap(y, mapx, mapy, cv2.INTER_LINEAR)
    k = int(2 / MTF_STEP)
    lo, hi = prof[:, :k].mean(axis=1), prof[:, -k:].mean(axis=1)
    good = (hi - lo) >= MTF_MIN_CONTRAST
    if good.sum() < MTF_MIN_EDGES:
        return float("nan")
    esf = ((prof[good] - lo[good, None]) / (hi - lo)[good, None]).mean(axis=0)
    lsf = np.diff(esf) * np.hanning(len(esf) - 1)
    spec = np.abs(np.fft.rfft(lsf, n=MTF_FFT_N))
    if spec[0] <= 0:
        return float("nan")
    mtf = spec / spec[0]
    f = np.fft.rfftfreq(MTF_FFT_N, d=MTF_STEP)
    below = np.flatnonzero(mtf < 0.5)
    if below.size == 0:
        return 0.5
    i = int(below[0])
    f50 = f[i - 1] + (mtf[i - 1] - 0.5) / (mtf[i - 1] - mtf[i]) * (f[i] - f[i - 1])
    return float(min(f50, 0.5))


def measure_image(img_bgr: np.ndarray, crop_left: int = CROP_LEFT_DEFAULT) -> dict[str, float]:
    """Mọi đại lượng một-số-mỗi-ảnh của camera (khoá ``MEASURE_KEYS``).

    Đơn vị: ``dark_channel_15px``, ``y_p1``, ``y_p99``, ``saturation_mean`` (S của HSV), ``pink_highlight_y``,
    ``sigma_immerkaer``, ``left_strip_mean_y`` thang 0–255; ``pink_frac`` tỉ lệ điểm 0–1; ``blocking_ratio_8x8``,
    ``sobel_x2_over_y2`` (E[Gx²]/E[Gy²], Sobel 3×3), ``vignette_corner_center_ratio``, ``r_over_g``, ``b_over_g`` không thứ
    nguyên; ``laplacian_var`` (thang 0–255)²; ``mtf50_cyc_per_px`` chu kỳ/px; ``left_strip_width_px`` px (đo trên khung
    nguyên).
    """
    _check(img_bgr, crop_left)
    width, strip_y, _ref = left_strip(img_bgr, crop_left)
    im = np.ascontiguousarray(img_bgr[:, crop_left:])
    y = luma(im)
    b, g, r = (im[..., k].astype(np.float32) for k in range(3))
    pink = ((r - g) > PINK_DIFF) & ((b - g) > PINK_DIFF)
    dark = cv2.erode(im.min(axis=2), np.ones((DARK_CHANNEL_PATCH, DARK_CHANNEL_PATCH), np.uint8))
    sx = cv2.Sobel(y, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(y, cv2.CV_32F, 0, 1)
    gm = float(g.mean())
    return {
        "dark_channel_15px": float(dark.mean()),
        "y_p1": float(np.percentile(y, 1)),
        "y_p99": float(np.percentile(y, 99)),
        "saturation_mean": float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean()),
        "pink_frac": float(pink.mean()),
        "pink_highlight_y": float(y[pink].mean()) if pink.any() else float("nan"),
        "blocking_ratio_8x8": blocking_ratio(y, crop_left),
        "laplacian_var": float(cv2.Laplacian(y, cv2.CV_32F).var()),
        "sigma_immerkaer": immerkaer_sigma(y),
        "sobel_x2_over_y2": float((sx ** 2).mean() / max(float((sy ** 2).mean()), 1e-6)),
        "vignette_corner_center_ratio": vignette_ratio(y),
        "r_over_g": float(r.mean() / max(gm, 1e-3)),
        "b_over_g": float(b.mean() / max(gm, 1e-3)),
        "mtf50_cyc_per_px": mtf50(y),
        "left_strip_width_px": width,
        "left_strip_mean_y": strip_y,
    }


def nlf(img_bgr: np.ndarray, crop_left: int = CROP_LEFT_DEFAULT, n_bins: int = 8) -> list[tuple[float, float]]:
    """Hàm mức nhiễu (sau ISP): ``n_bins`` bin đều trên Y 0–255, mỗi bin ``(Y trung bình, σ)`` trên vùng phẳng.

    Vùng phẳng = điểm có gradient (ảnh làm mịn σ=1,5) dưới phân vị ``NLF_GRAD_PERCENTILE``; σ của bin =
    ``sqrt(π/2)/6 · E|Y * M|`` (toán tử Immerkær) trên các điểm phẳng trong bin. Bin dưới ``NLF_MIN_PIXELS`` điểm: σ = NaN
    và Y = tâm bin.
    """
    _check(img_bgr, crop_left)
    y = luma(np.ascontiguousarray(img_bgr[:, crop_left:]))
    ys = cv2.GaussianBlur(y, (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(ys, cv2.CV_32F, 1, 0), cv2.Sobel(ys, cv2.CV_32F, 0, 1))
    resp = np.abs(cv2.filter2D(y, -1, _IMMERKAER))
    inner = np.zeros(y.shape, bool)
    inner[2:-2, 2:-2] = True
    flat = inner & (grad <= np.percentile(grad[inner], NLF_GRAD_PERCENTILE))
    level = cv2.blur(y, (5, 5))
    edges = np.linspace(0, 256, n_bins + 1)
    out = []
    for lo, hi in pairwise(edges):
        m = flat & (level >= lo) & (level < hi)
        if m.sum() < NLF_MIN_PIXELS:
            out.append((float((lo + hi) / 2), float("nan")))
        else:
            out.append((float(level[m].mean()), float(math.sqrt(math.pi / 2) / 6 * resp[m].mean())))
    return out


def psd_radial(img_bgr: np.ndarray, crop_left: int = CROP_LEFT_DEFAULT, n_bins: int = 16) -> list[tuple[float, float]]:
    """Phổ công suất trung bình theo bán kính của phần dư cao tần ``Y − Gauss(Y, σ=2)`` (cửa sổ Hann 2D).

    ``n_bins`` bin đều trên tần số 0–0,5 chu kỳ/px; trả ``(tâm bin, công suất)``, công suất = ``|FFT|²/(H·W)`` trung bình
    trong bin (thang Y 0–255).
    """
    _check(img_bgr, crop_left)
    y = luma(np.ascontiguousarray(img_bgr[:, crop_left:]))
    res = y - cv2.GaussianBlur(y, (0, 0), PSD_HIGHPASS_SIGMA)
    h, w = res.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    p = np.abs(np.fft.fft2(res * win)) ** 2 / (h * w) / float((win ** 2).mean())
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    edges = np.linspace(0, 0.5, n_bins + 1)
    out = []
    for lo, hi in pairwise(edges):
        m = (fr >= lo) & (fr < hi)
        out.append((float((lo + hi) / 2), float(p[m].mean()) if m.any() else float("nan")))
    return out


def jpeg_segments(data: bytes) -> dict[str, Any]:
    """Phân tích header JPEG tới SOS: bảng DQT (thứ tự tự nhiên, theo id bảng), SOF (marker, w, h, thành phần)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("không phải JPEG (thiếu SOI)")
    dqt: dict[int, list[int]] = {}
    sof = None
    i, n = 2, len(data)
    while i < n - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
            i += 1 if marker == 0xFF else 2
            continue
        if marker in (0xD9, 0xDA):
            break
        seglen = struct.unpack_from(">H", data, i + 2)[0]
        seg = data[i + 4:i + 2 + seglen]
        if marker == 0xDB:
            k = 0
            while k < len(seg):
                pq, tq = seg[k] >> 4, seg[k] & 15
                k += 1
                if pq == 0:
                    vals = list(seg[k:k + 64])
                    k += 64
                else:
                    vals = list(struct.unpack_from(">64H", seg, k))
                    k += 128
                nat = [0] * 64
                for z, v in enumerate(vals):
                    nat[_ZIGZAG[z]] = v
                dqt[tq] = nat
        elif marker in (0xC0, 0xC1, 0xC2, 0xC3):
            prec, h, w, nc = struct.unpack_from(">BHHB", seg, 0)
            comps = []
            for c in range(nc):
                cid, hv, tq = struct.unpack_from(">BBB", seg, 6 + 3 * c)
                comps.append((cid, hv >> 4, hv & 15, tq))
            sof = {"marker": marker, "precision": prec, "width": w, "height": h, "components": comps}
        i += 2 + seglen
    if sof is None:
        raise ValueError("không thấy SOF trước SOS")
    return {"dqt": dqt, "sof": sof}


def _subsampling(comps: list[tuple[int, int, int, int]]) -> str:
    if len(comps) != 3:
        return f"{len(comps)}-component"
    (_, h0, v0, _), (_, h1, v1, _), (_, h2, v2, _) = comps
    if (h1, v1) != (h2, v2) or h0 % h1 or v0 % v1:
        return f"{h0}x{v0},{h1}x{v1},{h2}x{v2}"
    return {(1, 1): "4:4:4", (2, 1): "4:2:2", (2, 2): "4:2:0", (1, 2): "4:4:0", (4, 1): "4:1:1"}.get(
        (h0 // h1, v0 // v1), f"{h0}x{v0},{h1}x{v1},{h2}x{v2}")


def read_jpeg_header(data: bytes) -> dict:
    """``{"qtable_luma": [64], "qtable_chroma": [64], "subsampling": "4:2:2", "baseline": bool}``.

    Bảng theo **thứ tự tự nhiên** (hàng-cột), lấy theo id bảng mà thành phần Y (thứ nhất) và Cb (thứ hai) dùng trong SOF.
    ``baseline`` = SOF0 và độ chính xác 8 bit.
    """
    seg = jpeg_segments(data)
    comps = seg["sof"]["components"]
    luma_t = seg["dqt"][comps[0][3]]
    chroma_t = seg["dqt"][comps[1][3]] if len(comps) > 1 else []
    return {
        "qtable_luma": list(luma_t),
        "qtable_chroma": list(chroma_t),
        "subsampling": _subsampling(comps),
        "baseline": seg["sof"]["marker"] == 0xC0 and seg["sof"]["precision"] == 8,
    }
