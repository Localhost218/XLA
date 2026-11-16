"""
Core Hybrid Image Processing & Open-Source Restoration Pipeline for ESP32-CAM (OV2640) UAV Imagery (v2).

Phiên bản v2 tập trung vào CHẤT LƯỢNG ẢNH ĐẦU RA, đặc biệt với ảnh tối / thiếu sáng / ngược sáng:

- Stage 1 (Ch. 2 & 3 + Khử nhiễu thích nghi):
  1. Khử ám hồng bảo toàn texture: triệt tiêu thành phần "excess = (R+B)/2 - G" nhưng GIỮ NGUYÊN luminance Y
     (chỉ dịch chroma), cộng thêm high-pass injection từ kênh G ở vùng R/B bão hòa, mặt nạ alpha được dilate để
     xử lý sạch vùng biên chuyển tiếp.
  2. AWB có kiểm soát (Shades-of-Gray p=6) loại trừ highlight bão hòa, vùng cây xanh, pixel quá tối và vật thể
     bão hòa màu mạnh; biên độ gain tới ±25 %, giữ nguyên độ sáng.
  3. Khử nhiễu thích nghi: ước lượng sigma bằng MAD trên sóng con HH1, cường độ NLM tự điều chỉnh theo sigma,
     mặt nạ cấu trúc (gradient chuẩn hoá theo sigma) bảo toàn biên/texture; chroma lọc bằng Guided Filter.
- Stage 2 (Ch. 3): Phân tích phân bố sáng (p5, p50, p95, dark ratio, dark channel) và tự động pha trộn:
  (a) Ảnh tối: Retinex/LIME — bản đồ chiếu sáng + Guided Filter + Adaptive Gamma, khử nhiễu vùng tối;
  (b) Ảnh ngược sáng: cùng cơ chế nén dải động cục bộ (gain chỉ tác động vùng có illumination thấp);
  (c) Ảnh sương mờ: khử sương dark-channel thích nghi + CLAHE nhẹ trên L* (CIE LAB).
- Stage 3 (Ch. 3 & 4): ESPCN ONNX trên kênh Y (BT.601 YCrCb), batch toàn bộ tile trong một lần inference,
  Richardson-Lucy + làm nét có cổng biên, chống halo bằng local min-max overshoot clamping.
"""

import os
import time
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort
import pywt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy

# Load Pretrained Open-Source ESPCN Super-Resolution ONNX Model once at module import
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "espcn_super_res.onnx")
_ORT_SESS: Optional[ort.InferenceSession] = None
if os.path.exists(_MODEL_PATH):
    _so = ort.SessionOptions()
    _so.log_severity_level = 3
    _ORT_SESS = ort.InferenceSession(_MODEL_PATH, sess_options=_so)


# =============================================================================
# HELPERS
# =============================================================================

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Chuẩn hoá đầu vào về BGR uint8 3 kênh: hỗ trợ Grayscale 2D/(H,W,1), BGR, BGRA; float [0,1] hoặc [0,255]."""
    if img is None:
        raise ValueError("Ảnh đầu vào là None")
    img = np.asarray(img)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and img.size and float(np.nanmax(img)) <= 1.0 + 1e-6:
            img = img * 255.0
        img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if c == 3:
            return np.ascontiguousarray(img)
        if c == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Định dạng ảnh không hỗ trợ: shape={img.shape}")


def _smoothstep(x, e0: float, e1: float):
    t = np.clip((np.asarray(x, dtype=np.float32) - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided Filter (He et al.) bằng box filter; guide & src là float32 1 kênh."""
    ksz = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, ksz)
    mean_p = cv2.boxFilter(src, -1, ksz)
    cov_ip = cv2.boxFilter(guide * src, -1, ksz) - mean_i * mean_p
    var_i = cv2.boxFilter(guide * guide, -1, ksz) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, ksz) * guide + cv2.boxFilter(b, -1, ksz)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Ước lượng độ lệch chuẩn nhiễu (thang 0-255) bằng MAD trên sóng con HH1 (Donoho): sigma = median(|HH1|)/0.6745."""
    img = _ensure_bgr(img)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if min(y.shape) < 8:
        return 0.0
    _, (_, _, hh) = pywt.dwt2(y, "db1")
    return float(np.median(np.abs(hh)) / 0.6745)


# =============================================================================
# STAGE 1: PINK RECOVERY, CONTROLLED AWB, NOISE-ADAPTIVE DENOISING
# =============================================================================

def recover_pink_highlights(
    img_bgr: np.ndarray,
    t_high: float = 110.0,
    delta_t: float = 6.0,
    strength: float = 0.56,
    passes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Khử ám hồng bảo toàn texture (Ch. 2 & 3).
    Thành phần hồng: excess = max(0, (R+B)/2 - G). Hiệu chỉnh chỉ dịch CHROMA, KHÔNG thay thế luminance:
        R' = R - k*a*excess,  B' = B - k*a*excess,  G' = G + 0.704*k*a*excess
    (hệ số 0.704 = (0.299+0.114)/0.587 giữ nguyên Y BT.601) nên vi cấu trúc của vỉa hè/bầu trời không bị phẳng hoá.
    Ở vùng R/B bão hoà (>=240) tiêm thêm high-pass của kênh G (kênh còn thông tin). Mặt nạ alpha được dilate rồi
    làm mờ để vùng biên chuyển tiếp cũng được khử sạch (Stage 2 không khuếch đại lại ám hồng).
    """
    img = _ensure_bgr(img_bgr)
    f = img.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha_first = None
    for _ in range(max(1, passes)):
        b, g, r = cv2.split(f)
        rb_mean = 0.5 * (r + b)
        excess = np.maximum(0.0, rb_mean - g)
        bright_gate = np.clip((rb_mean - t_high) / 50.0, 0.0, 1.0)
        pink_gate = np.clip((excess - delta_t) / 14.0, 0.0, 1.0)
        alpha = cv2.GaussianBlur(cv2.dilate(bright_gate * pink_gate, kernel), (0, 0), 2.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        if alpha_first is None:
            alpha_first = alpha
        delta = strength * alpha * excess
        sat_w = alpha * np.clip((np.maximum(r, b) - 240.0) / 15.0, 0.0, 1.0)
        hp_g = g - cv2.GaussianBlur(g, (0, 0), 2.0)
        inj = 0.5 * sat_w * hp_g
        f = cv2.merge([
            np.clip(b - delta + inj, 0.0, 255.0),
            np.clip(g + 0.704 * delta + inj, 0.0, 255.0),
            np.clip(r - delta + inj, 0.0, 255.0),
        ])
    return np.clip(f + 0.5, 0, 255).astype(np.uint8), alpha_first


def masked_gray_world_awb(
    img_bgr: np.ndarray,
    alpha_mask: Optional[np.ndarray] = None,
    max_gain_shift: float = 0.25,
    p_norm: float = 6.0,
    strength: float = 0.85,
    deadzone: float = 0.05,
) -> np.ndarray:
    """
    AWB có kiểm soát (Shades-of-Gray, Minkowski p=6) trên các pixel hợp lệ: loại trừ highlight bão hoà, pixel quá tối,
    vùng cây xanh, vật thể bão hoà màu mạnh (đèn, xe màu) và vùng vừa khử hồng.
    Gain ∈ [1-max_gain_shift, 1+max_gain_shift], pha trộn theo `strength` và chuẩn hoá để giữ nguyên độ sáng.
    """
    img = _ensure_bgr(img_bgr)
    f = img.astype(np.float32)
    b, g, r = cv2.split(f)
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    mx = f.max(axis=2)
    mn = f.min(axis=2)

    lo = max(6.0, float(np.percentile(lum, 10)))
    valid = (lum > lo) & (mx < 245.0) & ((mx - mn) < 90.0) & (((g - np.maximum(r, b)) < 6.0) | (lum < 35.0))
    if alpha_mask is not None:
        valid &= (alpha_mask < 0.15)
    if np.count_nonzero(valid) < 0.01 * lum.size:
        return img

    est = []
    for ch in (b, g, r):
        v = ch[valid] / 255.0
        est.append(float(np.mean(v ** p_norm) ** (1.0 / p_norm)) * 255.0 + 1e-6)
    gray = sum(est) / 3.0
    gains = np.array([gray / e for e in est], dtype=np.float32)
    gains = np.clip(gains, 1.0 - max_gain_shift, 1.0 + max_gain_shift)
    dev = gains - 1.0                       # vùng chết: chỉ sửa khi lệch màu thật sự (> deadzone)
    gains = 1.0 + strength * np.sign(dev) * np.maximum(np.abs(dev) - deadzone, 0.0)
    gains /= (0.114 * gains[0] + 0.587 * gains[1] + 0.299 * gains[2])
    out = np.clip(f * gains.reshape(1, 1, 3), 0, 255)
    return (out + 0.5).astype(np.uint8)


def bilateral_denoise(
    img_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 14.0,
    sigma_space: float = 7.0,
    sigma_hat: Optional[float] = None,
    texture_keep: float = 0.12,
) -> np.ndarray:
    """
    Khử nhiễu thích nghi, không "tranh sơn dầu":
    1. Ước lượng sigma thật của ảnh (MAD / sóng con HH1) -> h_NLM = 0.85*sigma (kẹp [1, 7]); ảnh sạch gần như không bị lọc.
    2. Mặt nạ cấu trúc từ gradient (ảnh làm mịn) chuẩn hoá theo sigma: biên/chữ/xe/vạch đường dùng nhánh nhẹ
       (bilateral), vùng phẳng dùng nhánh mạnh (NLM); `texture_keep` giữ lại một phần hạt tự nhiên ở vùng phẳng.
    3. Chroma (Cr, Cb) được lọc bằng Guided Filter với luminance làm guide -> không loang màu qua biên vật thể.
    (`sigma_color`, `sigma_space` giữ để tương thích API; cường độ nay tự điều chỉnh theo sigma ước lượng.)
    """
    img = _ensure_bgr(img_bgr)
    sigma = estimate_noise_sigma(img) if sigma_hat is None else float(sigma_hat)
    if sigma < 0.8:
        return img

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_u8, cr, cb = cv2.split(ycc)
    y = y_u8.astype(np.float32)

    h_nlm = float(np.clip(0.85 * sigma, 1.0, 7.0))
    strong = cv2.fastNlMeansDenoising(y_u8, None, h=h_nlm, templateWindowSize=5, searchWindowSize=15).astype(np.float32)
    weak = cv2.bilateralFilter(y, d=d, sigmaColor=float(1.5 * sigma + 4.0), sigmaSpace=2.0)

    y_s = cv2.GaussianBlur(y, (0, 0), 1.5)
    gx = cv2.Sobel(y_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_s, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)
    struct_w = _smoothstep(grad, 1.2 * sigma + 2.0, 3.2 * sigma + 8.0)

    y_dn = struct_w * weak + (1.0 - struct_w) * strong
    y_dn = y_dn + texture_keep * (1.0 - struct_w) * (y - strong)

    guide = y_dn / 255.0
    eps = (0.012 + 0.0035 * sigma) ** 2
    cr_f = _guided_filter(guide, cv2.medianBlur(cr, 3).astype(np.float32) / 255.0, 5, eps) * 255.0
    cb_f = _guided_filter(guide, cv2.medianBlur(cb, 3).astype(np.float32) / 255.0, 5, eps) * 255.0

    out = cv2.merge([
        np.clip(y_dn + 0.5, 0, 255).astype(np.uint8),
        np.clip(cr_f + 0.5, 0, 255).astype(np.uint8),
        np.clip(cb_f + 0.5, 0, 255).astype(np.uint8),
    ])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def _smooth_chroma_in_mask(img_bgr: np.ndarray, mask: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Nhiễu ở vùng cháy hồng chủ yếu là nhiễu chroma: làm mịn Cr/Cb có trọng số theo mặt nạ (luminance giữ nguyên)."""
    if mask is None or float(mask.max()) < 0.05:
        return img_bgr
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    w = np.clip(1.5 * mask, 0.0, 1.0)
    for ch in (1, 2):
        ycc[:, :, ch] = w * cv2.GaussianBlur(ycc[:, :, ch], (0, 0), sigma) + (1.0 - w) * ycc[:, :, ch]
    return cv2.cvtColor(np.clip(ycc + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def run_stage1_preprocess(
    img_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 14.0,
    sigma_space: float = 7.0,
    t_high: float = 110.0,
    delta_t: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 1: Khử ám hồng bảo toàn texture -> AWB có kiểm soát -> Khử nhiễu thích nghi theo sigma ước lượng."""
    img = _ensure_bgr(img_bgr)
    sigma = estimate_noise_sigma(img)
    img_pink_fixed, alpha_mask = recover_pink_highlights(img, t_high=t_high, delta_t=delta_t)
    img_pink_fixed = _smooth_chroma_in_mask(img_pink_fixed, alpha_mask)
    img_awb = masked_gray_world_awb(img_pink_fixed, alpha_mask=alpha_mask)
    img_s1 = bilateral_denoise(img_awb, d=d, sigma_color=sigma_color, sigma_space=sigma_space, sigma_hat=sigma)
    # Vùng cháy hồng có nhiễu cục bộ cao hơn mức nhiễu toàn ảnh (pixel R/B bão hoà): lọc mạnh hơn có trọng số theo alpha
    if float(alpha_mask.mean()) > 0.01:
        img_strong = bilateral_denoise(img_awb, d=d, sigma_hat=max(sigma, 3.5), texture_keep=0.05)
        w = np.clip(1.5 * alpha_mask, 0.0, 1.0)[..., None]
        img_s1 = np.clip(w * img_strong.astype(np.float32) + (1.0 - w) * img_s1.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
    return img_s1, alpha_mask


# =============================================================================
# STAGE 2: ADAPTIVE ILLUMINATION (RETINEX/LIME) + DEHAZE + CLAHE IN CIE LAB
# =============================================================================

STAGE2_CFG: Dict[str, float] = {
    "lift_target": 0.48,   # illumination mục tiêu (phân vị 35) sau nâng sáng
    "gate_lo": 0.26,       # T_35 <= gate_lo  -> nâng sáng toàn phần
    "gate_hi": 0.40,       # T_35 >= gate_hi  -> không can thiệp (ảnh đủ sáng)
    "g_min": 0.45,
    "gain_max": 5.0,
    "haze_omega": 0.60,    # cường độ khử sương tối đa
    "a_max": 0.85,         # trần ước lượng airlight
    "awb2": 0.0,           # cường độ AWB sau nâng sáng (0 = tắt)
    "airlight": 0.64,      # airlight A (chuẩn hoá) dùng để suy ra hệ số truyền c từ veil V đo được
    "c_min": 0.55,
    "veil_chroma_keep": 0.0,
    "veil_dark_relax": 0.6,
    "veil_chroma_keep_dark": 1.0,
    "black_anchor": 0.4,   # neo lại điểm đen sau nâng sáng (ảnh tối phẳng): tránh nền đen bị "sương xám"
    "black_target": 0.06,
    "bp_max": 0.35,
    "span_flat": 55.0,     # ảnh tối có dải động (p95-p5) < span_flat được coi là thiếu sáng phẳng -> nâng toàn phần
    "span_ok": 90.0,
    "back_lo": 140.0,      # p95 >= back_hi: có highlight đi cùng vùng tối -> coi là ngược sáng
    "back_hi": 190.0,
    "chroma_target": 15.0,
    "c_gain_max": 1.25,
}


def analyze_scene(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Phân tích phân bố sáng để chọn chế độ Stage 2: p5, p50, p95 (Y BT.601), dark_ratio (Y<40),
    bright_ratio (Y>=245), dark-channel trung bình, cùng các điểm số mode ∈ [0,1]:
    s_dark (thiếu sáng), s_backlit (ngược sáng / tương phản gắt), s_haze (sương mờ / sáng ám).
    """
    img = _ensure_bgr(img_bgr)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p5, p50, p95 = [float(v) for v in np.percentile(y, [5, 50, 95])]
    dark_ratio = float(np.mean(y < 40))
    bright_ratio = float(np.mean(y >= 245))
    dc = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    dc_mean = float(dc.mean()) / 255.0
    s_dark = float(1.0 - _smoothstep(p50, 55.0, 110.0))
    s_backlit = float(_smoothstep(dark_ratio, 0.10, 0.35) * _smoothstep(p95, 165.0, 220.0))
    # Sương mờ = nền đen bị nâng (p5 cao) + dải động hẹp + dark channel cao
    s_haze = float(_smoothstep(p5, 55.0, 90.0) * (1.0 - _smoothstep(p95 - p5, 100.0, 150.0))
                   * _smoothstep(dc_mean, 0.20, 0.34))
    return {
        "p5": p5, "p50": p50, "p95": p95, "dark_ratio": dark_ratio, "bright_ratio": bright_ratio,
        "dc_mean": dc_mean, "s_dark": s_dark, "s_backlit": s_backlit, "s_haze": s_haze,
    }


def _illumination_gain(f: np.ndarray, target: float = 0.42, g_min: float = 0.45, gain_max: float = 5.0,
                       gate: Tuple[float, float] = (0.20, 0.36)):
    """
    Bản đồ chiếu sáng kiểu LIME: T = GuidedFilter(MaxFilter(max_c I)). Adaptive Gamma chọn từ phân vị 35 của T:
        g = ln(target)/ln(T_35) kẹp [g_min, 1]   (ảnh sáng bình thường -> g = 1, không can thiệp)
    Gain = T^(g-1) là bản đồ MƯỢT nhân lên cả 3 kênh: vùng chiếu sáng thấp (tối/bóng) được nâng, vùng T≈1 (đèn,
    bầu trời) gain≈1 nên không cháy; vi cấu trúc (reflectance) giữ nguyên.
    """
    h, w = f.shape[:2]
    m = f.max(axis=2)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    t0 = cv2.dilate(m, np.ones((7, 7), np.uint8))
    radius = int(max(8, min(h, w) // 30))
    t = np.clip(_guided_filter(gray, t0, radius, 0.01), 0.03, 1.0)
    t35 = float(np.percentile(t, 35))
    g = float(np.clip(np.log(target) / np.log(max(t35, 0.03)), g_min, 1.0))
    w_gate = 1.0 - float(_smoothstep(t35, gate[0], gate[1]))      # ảnh đủ sáng -> w=0 -> g_eff=1
    g = 1.0 - w_gate * (1.0 - g)
    gain = np.minimum(np.power(t, g - 1.0), gain_max)
    return gain.astype(np.float32), g, t35


def _dehaze_dark_channel(f: np.ndarray, omega: float, t_min: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    """Khử sương dark-channel (He et al.) với transmission tinh chỉnh bằng Guided Filter, cường độ omega thích nghi."""
    dc = cv2.erode(f.min(axis=2), np.ones((15, 15), np.uint8))
    n_top = max(1, int(0.001 * dc.size))
    idx = np.argpartition(dc.ravel(), -n_top)[-n_top:]
    flat = f.reshape(-1, 3)[idx]
    a_light = np.clip(flat[np.argmax(flat.sum(axis=1))], 0.6, STAGE2_CFG["a_max"])
    a_scalar = float(a_light.mean())
    t = 1.0 - omega * dc / max(a_scalar, 1e-3)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    t = np.clip(_guided_filter(gray, t.astype(np.float32), 30, 1e-3), t_min, 1.0)
    j = np.clip((f - a_light.reshape(1, 1, 3)) / t[..., None] + a_light.reshape(1, 1, 3), 0.0, 1.0)
    return j, t


def run_stage2_adaptive(
    img_bgr: np.ndarray,
    clip_limit: float = 1.6,
    tile_grid_size: Tuple[int, int] = (8, 8),
    clahe_blend: float = 0.20,
    alpha_mask: Optional[np.ndarray] = None,
    chroma_target: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Stage 2 thích nghi (không còn haze_pivot cứng): 1) phân tích phân bố sáng; 2) nâng sáng Retinex/LIME + adaptive
    gamma (ảnh tối & ngược sáng) với vai highlight mềm và khử nhiễu vùng đã khuếch đại; 3) khử sương dark-channel
    với cường độ theo thống kê ảnh; 4) CLAHE nhẹ trên L* (CIE LAB); 5) bù bão hoà theo từng pixel (giảm ở vùng tối
    và vùng vừa khử hồng). Trả về (ảnh, L trước, L sau, info).
    """
    img = _ensure_bgr(img_bgr)
    info = analyze_scene(img)
    f = img.astype(np.float32) / 255.0

    c = STAGE2_CFG
    sigma_in = estimate_noise_sigma(img)

    # Mô hình veil khí quyển / black-level pedestal của OV2640:  I = c*J + (1-c)*A.
    # V = (1-c)*A được ĐO từ ảnh (percentile 1 % từng kênh = sàn đen bị nâng), c = 1 - V/A  => J = (I - V)/c.
    # Không có veil (V≈0) thì c≈1 và bước này gần như không làm gì.
    v_ch = np.percentile(f.reshape(-1, 3), 1.0, axis=0)
    # Ảnh tối PHẲNG (dải động hẹp, p50 thấp): phần "veil" chủ yếu là black-level/nhiễu nền thật -> chỉ trừ một phần,
    # phần còn lại do bước nâng sáng Retinex xử lý (tránh kéo ảnh tối tụt xuống thấp hơn cả ảnh gốc)
    span0 = info["p95"] - info["p5"]
    w_dark_flat = float((1.0 - _smoothstep(span0, 45.0, 70.0)) * (1.0 - _smoothstep(info["p50"], 70.0, 110.0)))
    info["w_dark_flat"] = w_dark_flat
    # veil gần như trung tính (tránh tự tạo ám màu do sàn đen từng kênh khác nhau, vd. cảnh nhiều cây xanh); riêng ảnh tối
    # phẳng giữ chênh lệch kênh để sửa black-level lệch màu (ám xanh/ô liu) trước khi khuếch đại.
    keep = c["veil_chroma_keep"] + (c["veil_chroma_keep_dark"] - c["veil_chroma_keep"]) * w_dark_flat
    v_c = np.clip(v_ch.mean() + keep * (v_ch - v_ch.mean()), 0.0, c["bp_max"]).astype(np.float32)
    v_c = v_c * (1.0 - c["veil_dark_relax"] * w_dark_flat)
    c_est = float(np.clip(1.0 - float(v_c.mean()) / c["airlight"], c["c_min"], 1.0))
    f = np.clip((f - v_c.reshape(1, 1, 3)) / c_est, 0.0, 1.0)
    info["veil"] = float(v_c.mean())
    info["veil_c"] = c_est
    info.update({k + "_post": v for k, v in analyze_scene(np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)).items()})
    info["s_haze"] = info["s_haze_post"]

    # (a)+(b) Illumination lift
    gain, gamma, t35 = _illumination_gain(f, c["lift_target"], c["g_min"], c["gain_max"], (c["gate_lo"], c["gate_hi"]))
    # Chính sách nâng sáng: tối & phẳng (thiếu sáng thật) hoặc có highlight đi cùng vùng tối (ngược sáng) -> nâng toàn phần;
    # ảnh tối nhưng còn dải động tốt (cảnh vốn tối) -> nâng một phần để không làm sai lệch cảnh.
    y_chk = cv2.cvtColor(np.clip(f * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    q5, q95 = np.percentile(y_chk, [5, 95])
    w_flat = 1.0 - float(_smoothstep(q95 - q5, c["span_flat"], c["span_ok"]))
    w_back = float(_smoothstep(q95, c["back_lo"], c["back_hi"]))
    w_policy = max(w_flat, w_back)
    gain = np.power(gain, w_policy).astype(np.float32)
    info["lift_policy"] = w_policy
    out = f * gain[..., None]
    shoulder = 0.80
    over = np.maximum(out - shoulder, 0.0)
    out = np.where(out > shoulder, shoulder + (1.0 - shoulder) * np.tanh(over / (1.0 - shoulder)), out)
    out = np.clip(out, 0.0, 1.0)
    info.update({"gamma": gamma, "illum_p35": t35, "gain_mean": float(gain.mean())})

    # Khử nhiễu vùng vừa được khuếch đại
    amp_map = gain.copy()
    w_n = _smoothstep(gain, 1.3, 3.0)
    if float(w_n.mean()) > 0.01:
        out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ycc = cv2.cvtColor(out_u8, cv2.COLOR_BGR2YCrCb)
        y_u8, cr, cb = cv2.split(ycc)
        sigma_l = estimate_noise_sigma(out_u8)
        h_n = float(np.clip(0.9 * sigma_l, 1.5, 8.0))
        y_dn = cv2.fastNlMeansDenoising(y_u8, None, h=h_n, templateWindowSize=5, searchWindowSize=15).astype(np.float32)
        # Vùng tối được nâng sáng: khử nhiễu hạt + khối JPEG 8x8 bằng Guided Filter tự dẫn (edge-preserving)
        amp_loc = float((gain * w_n).sum() / max(float(w_n.sum()), 1e-3))
        y01 = y_dn / 255.0
        y_gf = _guided_filter(y01, y01, 6, (0.010 * max(amp_loc, 1.0)) ** 2) * 255.0
        y_new = w_n * y_gf + (1.0 - w_n) * y_u8.astype(np.float32)
        # Chroma: Guided Filter dẫn bởi luminance (bán kính lớn, eps lớn) -> khử khối màu nhưng không loang quanh đèn
        cr_b = _guided_filter(y01, cr.astype(np.float32) / 255.0, 8, 0.01) * 255.0
        cb_b = _guided_filter(y01, cb.astype(np.float32) / 255.0, 8, 0.01) * 255.0
        cr_new = w_n * cr_b + (1.0 - w_n) * cr.astype(np.float32)
        cb_new = w_n * cb_b + (1.0 - w_n) * cb.astype(np.float32)
        merged = cv2.merge([np.clip(y_new + 0.5, 0, 255).astype(np.uint8),
                            np.clip(cr_new + 0.5, 0, 255).astype(np.uint8),
                            np.clip(cb_new + 0.5, 0, 255).astype(np.uint8)])
        out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR).astype(np.float32) / 255.0
        info["post_lift_sigma"] = sigma_l

    # Neo lại điểm đen sau nâng sáng: vùng nền tối thật sự phải tối hơn vùng có chi tiết (chỉ khi nâng sáng mạnh)
    if w_policy > 0.05 and float(gain.mean()) > 1.15:
        y_o = out.mean(axis=2)
        p2 = float(np.percentile(y_o, 2.0))
        sh = float(np.clip(c["black_anchor"] * (p2 - c["black_target"]), 0.0, 0.30)) * float(_smoothstep(float(gain.mean()), 1.15, 1.6))
        out = np.clip((out - sh) / (1.0 - sh), 0.0, 1.0)
        info["black_anchor_shift"] = sh

    # AWB lần 2 trên ảnh đã nâng sáng (ước lượng màu ổn định hơn nhiều so với ảnh tối): chỉ khi thực sự nâng sáng
    if c["awb2"] > 0 and float(gain.mean()) > 1.25:
        out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out = masked_gray_world_awb(out_u8, strength=c["awb2"]).astype(np.float32) / 255.0

    # (c) Dehaze thích nghi (chỉ khi thống kê thực sự giống sương mờ)
    s_haze = info["s_haze"]
    if s_haze > 0.02:
        omega = c["haze_omega"] * s_haze
        out, t_map = _dehaze_dark_channel(out, omega)
        amp_map = amp_map / t_map

    # Khử nhiễu cuối theo hệ số khuếch đại nhiễu thực tế (nâng sáng + khử sương): sigma_eff = sigma_in * mean(amp)
    out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    amp = float(amp_map.mean())
    if amp > 1.15:
        sigma_eff = max(sigma_in, 1.0) * amp
        out_u8 = bilateral_denoise(out_u8, sigma_hat=sigma_eff, texture_keep=0.10)
        info["sigma_eff"] = sigma_eff
    info["noise_amp"] = amp

    # CLAHE nhẹ trên L* + bù bão hoà theo pixel
    lab = cv2.cvtColor(out_u8, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan_in = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))[0]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_chan)
    blend = clahe_blend * (1.0 - 0.5 * info["s_dark"])
    l_out = np.clip(blend * l_clahe.astype(np.float32) + (1.0 - blend) * l_chan.astype(np.float32), 0, 255)

    a_f = a_chan.astype(np.float32) - 128.0
    b_f = b_chan.astype(np.float32) - 128.0
    chroma_rms = float(np.sqrt(np.mean(a_f ** 2 + b_f ** 2)))
    c_gain = float(np.clip((chroma_target or c["chroma_target"]) / max(chroma_rms, 4.0), 1.0, c["c_gain_max"]))
    w_lum = _smoothstep(l_out, 25.0, 90.0)
    w_pink = 1.0 - np.clip(2.0 * alpha_mask, 0.0, 1.0) if alpha_mask is not None and alpha_mask.shape == l_out.shape else 1.0
    pix_gain = 1.0 + (c_gain - 1.0) * w_lum * w_pink
    a_out = np.clip(128.0 + a_f * pix_gain + 0.5, 0, 255).astype(np.uint8)
    b_out = np.clip(128.0 + b_f * pix_gain + 0.5, 0, 255).astype(np.uint8)
    info["c_gain"] = c_gain

    l_out_u8 = (l_out + 0.5).astype(np.uint8)
    img_s2 = cv2.cvtColor(cv2.merge([l_out_u8, a_out, b_out]), cv2.COLOR_LAB2BGR)
    return img_s2, l_chan_in, l_out_u8, info


def run_stage2_contrast(
    img_bgr: np.ndarray,
    clip_limit: float = 1.6,
    tile_grid_size: Tuple[int, int] = (8, 8),
    haze_pivot: Optional[float] = None,
    contrast_gain: Optional[float] = None,
    clahe_blend: float = 0.20,
    alpha_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper tương thích API cũ: trả về (img_s2, L_before, L_after).
    `haze_pivot` / `contrast_gain` được GIỮ trong chữ ký chỉ để tương thích và KHÔNG còn được sử dụng (đã bỏ pivot cứng).
    """
    img_s2, l_in, l_out, _ = run_stage2_adaptive(
        img_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
        clahe_blend=clahe_blend, alpha_mask=alpha_mask,
    )
    return img_s2, l_in, l_out


# =============================================================================
# STAGE 3: ESPCN (Y CHANNEL, BATCHED) + RICHARDSON-LUCY + HALO-SAFE SHARPENING
# =============================================================================

def _run_espcn_sr_luminance(y_chan: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Chạy ESPCN ONNX (`models/espcn_super_res.onnx`, x3) trên kênh Y BT.601 (từ YCrCb, thang [0,1]) — KHÔNG phải L* của LAB.
    Toàn bộ tile 224x224 (stride 200, lề 12 px) được gom thành MỘT batch và suy luận trong một lần gọi `session.run`.
    Kết quả được resize về (target_w, target_h), kẹp ±12 quanh Lanczos-4 để chống overshoot rồi pha trộn 65/35.
    """
    y32 = y_chan.astype(np.float32)
    lanczos_res = np.clip(cv2.resize(y32, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4), 0.0, 255.0)
    if _ORT_SESS is None or min(y32.shape) < 16:
        return lanczos_res.astype(np.uint8)

    h, w = y32.shape
    stride, margin, tile = 200, 12, 224
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    y_pad = cv2.copyMakeBorder(y32 / 255.0, margin, pad_h + margin, margin, pad_w + margin, cv2.BORDER_REFLECT_101)
    hp, wp = y_pad.shape
    coords = [(yy, xx) for yy in range(0, hp - 2 * margin, stride) for xx in range(0, wp - 2 * margin, stride)]
    batch = np.stack([y_pad[yy:yy + tile, xx:xx + tile] for yy, xx in coords])[:, None, :, :].astype(np.float32)
    sr = _ORT_SESS.run(None, {"input": batch})[0][:, 0] * 255.0  # (N, 672, 672)

    out_3x = np.zeros(((hp - 2 * margin) * 3, (wp - 2 * margin) * 3), dtype=np.float32)
    m3 = margin * 3
    for k, (yy, xx) in enumerate(coords):
        out_3x[yy * 3:(yy + stride) * 3, xx * 3:(xx + stride) * 3] = sr[k, m3:m3 + stride * 3, m3:m3 + stride * 3]

    out_crop = np.clip(out_3x[:h * 3, :w * 3], 0.0, 255.0)
    sr_res = np.clip(cv2.resize(out_crop, (target_w, target_h), interpolation=cv2.INTER_AREA), 0.0, 255.0)
    sr_clean = np.clip(sr_res, lanczos_res - 12.0, lanczos_res + 12.0)
    return np.clip(0.65 * sr_clean + 0.35 * lanczos_res, 0.0, 255.0).astype(np.uint8)


def run_stage3_sharpen(
    img_bgr: np.ndarray,
    psf_sigma: float = 0.88,
    rl_iter: int = 2,
    use_espcn: bool = True,
    strength: float = 0.6,
    detail_gain: float = 0.0,
    clamp_margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Làm nét có cổng biên, không halo (Ch. 3 & 4) trên kênh Y BT.601:
    1. (tuỳ chọn) ESPCN trên Y để lấy chi tiết thần kinh; 2. Richardson-Lucy với PSF Gaussian của OV2640;
    3. Cổng biên bằng gradient chuẩn hoá theo độ phân giải (nhân h/480) và theo nhiễu (ngưỡng >= 2.5*sigma_hat)
       nên chạy đúng ở cả 480p/720p/1080p và không khuếch đại nhiễu vùng tối;
    4. Chống halo: kết quả bị kẹp trong [local_min - m, local_max + m] của ảnh gốc -> không còn viền cứng
       (overshoot) quanh mảng phẳng.
    """
    img = _ensure_bgr(img_bgr)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_u8, cr, cb = cv2.split(ycc)
    y_in = y_u8.astype(np.float32)
    h, w = y_in.shape
    scale = max(h / 480.0, 1.0)
    sigma_n = estimate_noise_sigma(img)

    if use_espcn:
        y_espcn = _run_espcn_sr_luminance(y_u8, w, h)
        y_seed = 0.55 * y_espcn.astype(np.float32) + 0.45 * y_in
    else:
        y_seed = y_in

    k_1d = cv2.getGaussianKernel(5, psf_sigma)
    psf = (k_1d @ k_1d.T).astype(np.float32)
    y_rl = richardson_lucy(np.clip(y_seed / 255.0, 1e-4, 1.0), psf, num_iter=rl_iter, clip=True) * 255.0

    b_c, g_c, r_c = cv2.split(img.astype(np.float32))
    fol_w = cv2.GaussianBlur(np.clip((g_c - np.maximum(r_c, b_c) - 6.0) / 14.0, 0.0, 1.0), (0, 0), 1.2 * scale)

    y_pre = cv2.GaussianBlur(y_in, (0, 0), 1.0 * scale)
    gx = cv2.Sobel(y_pre, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_pre, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2 * scale) * scale
    t_lo = max(10.0, 2.5 * sigma_n)
    edge_w = cv2.GaussianBlur(_smoothstep(grad, t_lo, t_lo + 24.0), (0, 0), 1.0 * scale)
    active_w = np.maximum(0.6 * fol_w, edge_w)

    g1 = cv2.GaussianBlur(y_rl, (0, 0), 0.85 * scale)
    d_fine = y_rl - g1
    boost = strength * active_w * ((y_rl - y_in) + detail_gain * d_fine)
    boost = 18.0 * np.tanh(boost / 18.0)

    y_sharp = y_in + boost
    ksz = 2 * int(round(scale)) + 3
    kern = np.ones((ksz, ksz), np.uint8)
    margin = clamp_margin
    y_sharp = np.clip(y_sharp, cv2.erode(y_in, kern) - margin, cv2.dilate(y_in, kern) + margin)
    y_sharp = np.clip(y_sharp, 0.0, 255.0)

    img_sharp = cv2.cvtColor(cv2.merge([(y_sharp + 0.5).astype(np.uint8), cr, cb]), cv2.COLOR_YCrCb2BGR)
    detail_vis = np.clip(128.0 + 4.0 * (y_sharp - y_in), 0, 255).astype(np.uint8)
    return img_sharp, detail_vis


def fit_size(src_w: int, src_h: int, box_w: int, box_h: int) -> Tuple[int, int]:
    """Kích thước lớn nhất giữ NGUYÊN tỷ lệ khung hình của (src_w, src_h) nằm trong hộp (box_w, box_h), làm tròn chẵn."""
    s = min(box_w / float(src_w), box_h / float(src_h))
    return max(2, int(round(src_w * s / 2.0)) * 2), max(2, int(round(src_h * s / 2.0)) * 2)


def letterbox(img: np.ndarray, out_w: int, out_h: int, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Đặt ảnh vào canvas out_w x out_h, giữ tỷ lệ khung hình, đệm viền đen (không méo hình)."""
    h, w = img.shape[:2]
    nw, nh = fit_size(w, h, out_w, out_h)
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation) if (nw, nh) != (w, h) else img
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0, y0 = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def upscale_to_hd(
    img_stage2_bgr: np.ndarray,
    target_height: int = 720,
    target_width: int = 0,
) -> np.ndarray:
    """
    Siêu phân giải HD/Full HD GIỮ TỶ LỆ KHUNG HÌNH: ảnh được phóng to vừa trong hộp (target_width x target_height)
    (nếu target_width <= 0 chỉ ràng buộc theo chiều cao). Ví dụ 640x480 -> 960x720 / 1440x1080; ảnh 16:9 -> 1280x720.
    Y: ESPCN (batched) ; Cr/Cb: bicubic ; sau đó Richardson-Lucy + làm nét có cổng biên (không halo).
    """
    img = _ensure_bgr(img_stage2_bgr)
    h, w = img.shape[:2]
    if target_width <= 0:
        out_h = int(round(target_height))
        out_w = int(round(w * out_h / float(h)))
    else:
        out_w, out_h = fit_size(w, h, int(target_width), int(target_height))

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_in, cr_in, cb_in = cv2.split(ycc)
    y_hd = _run_espcn_sr_luminance(y_in, out_w, out_h)
    cr_hd = cv2.resize(cr_in, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    cb_hd = cv2.resize(cb_in, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    up_bgr = cv2.cvtColor(cv2.merge([y_hd, cr_hd, cb_hd]), cv2.COLOR_YCrCb2BGR)
    psf_s = 0.92 if out_h <= 720 else 1.05
    up_sharp, _ = run_stage3_sharpen(up_bgr, psf_sigma=psf_s, use_espcn=False)
    return up_sharp


def upscale_to_720p(img_stage2_bgr: np.ndarray, target_height: int = 720) -> np.ndarray:
    """Alias for upscale_to_hd(img_stage2_bgr, target_height=target_height)."""
    return upscale_to_hd(img_stage2_bgr, target_height=target_height)


# =============================================================================
# FULL PIPELINE & VIDEO
# =============================================================================

def run_full_pipeline(
    img_bgr: np.ndarray,
    return_intermediates: bool = False,
    upscale_hd: bool = False,
    hd_height: int = 720,
    hd_heights: Optional[Tuple[int, ...]] = None,
) -> Dict[str, np.ndarray]:
    """
    Chạy pipeline 3 giai đoạn. Chỉ tính những đầu ra được yêu cầu:
    - `upscale_hd=False`: output = Stage 3 ở độ phân giải gốc.
    - `upscale_hd=True`: output = HD `hd_height` (bỏ qua Stage 3 native trừ khi return_intermediates=True).
    - `hd_heights=(720, 1080)`: thêm output_720p / output_1080p (mỗi bản chỉ tính khi được yêu cầu).
    `runtime_ms` đo đúng thời gian tạo ra `output` (Stage 1 + 2 + [Stage 3 | upscale]).
    """
    img = _ensure_bgr(img_bgr)
    t0 = time.perf_counter()
    img_s1, alpha_mask = run_stage1_preprocess(img)
    t1 = time.perf_counter()
    img_s2, l_before, l_after, scene_info = run_stage2_adaptive(img_s1, alpha_mask=alpha_mask)
    t2 = time.perf_counter()

    need_native = (not upscale_hd) or return_intermediates
    img_s3, detail_vis, stage3_ms = None, None, 0.0
    if need_native:
        img_s3, detail_vis = run_stage3_sharpen(img_s2)
        stage3_ms = (time.perf_counter() - t2) * 1000.0

    heights = list(hd_heights) if hd_heights is not None else []
    if upscale_hd and hd_height not in heights:
        heights.append(hd_height)
    hd_out: Dict[int, np.ndarray] = {}
    hd_ms: Dict[int, float] = {}
    for hh in heights:
        ts = time.perf_counter()
        hd_out[hh] = upscale_to_hd(img_s2, target_height=hh)
        hd_ms[hh] = (time.perf_counter() - ts) * 1000.0

    stage1_ms, stage2_ms = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
    if upscale_hd:
        out_main = hd_out[hd_height]
        runtime = stage1_ms + stage2_ms + hd_ms[hd_height]
    else:
        out_main = img_s3
        runtime = stage1_ms + stage2_ms + stage3_ms

    result = {
        "output": out_main,
        "output_native": img_s3,
        "runtime_ms": runtime,
        "stage1_ms": stage1_ms,
        "stage2_ms": stage2_ms,
        "stage3_ms": stage3_ms,
        "hd_ms": hd_ms,
        "scene_info": scene_info,
    }
    for hh, im in hd_out.items():
        result[f"output_{hh}p"] = im
    if return_intermediates:
        result.update({
            "stage1": img_s1,
            "alpha_mask": alpha_mask,
            "stage2": img_s2,
            "l_before": l_before,
            "l_after": l_after,
            "detail_vis": detail_vis,
        })
    return result


def process_uav_video(
    input_video_path: str,
    output_video_path: str,
    mode: str = "side_by_side_720p",
    max_frames: int = 180,
) -> Dict[str, float]:
    """
    Xử lý video khung-theo-khung và mã hoá H.264. GIỮ TỶ LỆ KHUNG HÌNH (letterbox, không méo).
    `max_frames <= 0` nghĩa là xử lý hết video. mode: 'side_by_side_720p' | 'side_by_side_1080p' | 'hd_720p' | 'full_hd_1080p'.
    """
    import subprocess
    cap = cv2.VideoCapture(input_video_path)
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = (1920, 1080) if mode in ("side_by_side_1080p", "full_hd_1080p") else (1280, 720)
    sbs = mode.startswith("side_by_side")

    tmp_mp4v = output_video_path + ".tmp_mp4v.mp4"
    writer = cv2.VideoWriter(tmp_mp4v, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (out_w, out_h))

    frames_done = 0
    total_ms = 0.0
    while max_frames <= 0 or frames_done < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _ensure_bgr(frame)
        fh, fw = frame.shape[:2]
        # Xử lý ở độ phân giải làm việc (cao tối đa 360) nhưng giữ nguyên tỷ lệ
        if fh > 360:
            pw, ph = fit_size(fw, fh, 10 ** 6, 360)
            proc_in = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            proc_in = frame
        t0 = time.perf_counter()
        s1, alpha = run_stage1_preprocess(proc_in)
        s2, _, _, _ = run_stage2_adaptive(s1, alpha_mask=alpha)
        restored = upscale_to_hd(s2, target_height=out_h, target_width=out_w)
        restored_c = letterbox(restored, out_w, out_h)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if sbs:
            in_c = letterbox(proc_in, out_w, out_h, interpolation=cv2.INTER_CUBIC)
            half = out_w // 2
            out_frame = np.hstack([in_c[:, :half], restored_c[:, half:]])
            cv2.line(out_frame, (half, 0), (half, out_h), (0, 255, 180), max(2, out_h // 360))
            fs = out_h / 1080.0
            cv2.putText(out_frame, "INPUT: ESP32-CAM (OV2640)", (int(24 * fs) + 4, int(46 * fs) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85 * fs + 0.1, (70, 70, 255), 2, cv2.LINE_AA)
            cv2.putText(out_frame, f"OUTPUT: Restored {out_h}p ({1000.0 / max(dt_ms, 1e-3):.1f} FPS CPU)",
                        (half + int(24 * fs) + 4, int(46 * fs) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85 * fs + 0.1, (70, 255, 100), 2, cv2.LINE_AA)
        else:
            out_frame = restored_c

        total_ms += dt_ms
        writer.write(out_frame)
        frames_done += 1

    cap.release()
    writer.release()

    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", tmp_mp4v,
                "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                output_video_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(tmp_mp4v)
    except Exception:
        os.replace(tmp_mp4v, output_video_path)

    avg_ms = total_ms / max(frames_done, 1)
    return {"frames": frames_done, "avg_ms": avg_ms, "fps": 1000.0 / max(avg_ms, 1e-3)}


def baseline_global_he_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Baseline 1: Global Histogram Equalization độc lập trên R, G, B."""
    b, g, r = cv2.split(_ensure_bgr(img_bgr))
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


def baseline_global_he_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Baseline 2: Global Histogram Equalization trên kênh L của CIE LAB."""
    lab = cv2.cvtColor(_ensure_bgr(img_bgr), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([cv2.equalizeHist(l), a, b]), cv2.COLOR_LAB2BGR)


# =============================================================================
# QUANTITATIVE EVALUATION METRICS (SECTION 4.2)
# =============================================================================

def compute_psnr(img_gt: np.ndarray, img_pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(img_gt, img_pred, data_range=255))


def compute_ssim(img_gt: np.ndarray, img_pred: np.ndarray) -> float:
    return float(structural_similarity(img_gt, img_pred, channel_axis=2, data_range=255))


def compute_delta_e(img_gt: np.ndarray, img_pred: np.ndarray) -> float:
    lab1 = cv2.cvtColor(img_gt, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img_pred, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    return float(np.mean(diff))


def compute_entropy(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(_ensure_bgr(img_bgr), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / (np.sum(hist) + 1e-12)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_laplacian_var(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(_ensure_bgr(img_bgr), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_pink_cast_index(img_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(_ensure_bgr(img_bgr).astype(np.float32))
    rb_mean = 0.5 * (r + b)
    pink_pixels = (rb_mean > 140.0) & ((rb_mean - g) > 10.0)
    return float(np.mean(pink_pixels) * 100.0)


def compute_l_mean(img_bgr: np.ndarray) -> float:
    """L* trung bình (thang CIE 0-100)."""
    l = cv2.cvtColor(_ensure_bgr(img_bgr), cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    return float(l.mean() * 100.0 / 255.0)


def compute_crush_ratio(img_bgr: np.ndarray, thr: int = 3) -> float:
    """Tỷ lệ (%) pixel bị 'nghiền' về đen (Y BT.601 <= thr)."""
    y = cv2.cvtColor(_ensure_bgr(img_bgr), cv2.COLOR_BGR2GRAY)
    return float(np.mean(y <= thr) * 100.0)


def compute_clip_ratio(img_bgr: np.ndarray, thr: int = 253) -> float:
    """Tỷ lệ (%) pixel cháy sáng (Y >= thr)."""
    y = cv2.cvtColor(_ensure_bgr(img_bgr), cv2.COLOR_BGR2GRAY)
    return float(np.mean(y >= thr) * 100.0)
