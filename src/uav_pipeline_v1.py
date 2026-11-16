"""
Core Hybrid Image Processing & Open-Source Restoration Pipeline for ESP32-CAM (OV2640) UAV Imagery.
Combines Course Chapters 2, 3, and 4 with Open-Source Libraries & Pretrained Models:
- Stage 1 (Ch. 2 & 3 + Open-Source Denoising):
  1. Soft Highlight Pink/Magenta Cast Recovery & Masked Gray World AWB.
  2. Dual-Domain Open-Source Denoising: BayesShrink Wavelet Denoising (PyWavelets + scikit-image
     `denoise_wavelet` in YCbCr space) fused with OpenCV Non-Local Means (`fastNlMeansDenoising`)
     and Chroma Median-Bilateral Filtering — eliminates CMOS sensor speckle noise and JPEG 8x8 block
     artifacts on roads/sky/water while preserving 100% of fine tree foliage and object edges.
- Stage 2 (Ch. 3):
  CIE LAB Color Space Transformation + Haze-Pivot Contrast Restoration & Local CLAHE on L-channel.
- Stage 3 (Ch. 3 & 4 + Open-Source Deconvolution & Super-Resolution):
  1. Iterative Richardson-Lucy PSF Deconvolution (`skimage.restoration.richardson_lucy`) to invert
     the OV2640 optical blur kernel without noise amplification.
  2. Pretrained Open-Source Sub-Pixel CNN Super-Resolution (`ESPCN` ONNX model via `onnxruntime`)
     combined with Lanczos-4 interpolation for crisp 720p HD output.
"""

import os
import time
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_wavelet, richardson_lucy

# Load Pretrained Open-Source ESPCN Super-Resolution ONNX Model once at module import
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "espcn_super_res.onnx")
_ORT_SESS: Optional[ort.InferenceSession] = None
if os.path.exists(_MODEL_PATH):
    _so = ort.SessionOptions()
    _so.log_severity_level = 3
    _ORT_SESS = ort.InferenceSession(_MODEL_PATH, sess_options=_so)


# =============================================================================
# STAGE 1: PREPROCESSING — PINK HIGHLIGHT RECOVERY & WAVELET + NLM DENOISING
# =============================================================================

def recover_pink_highlights(
    img_bgr: np.ndarray,
    t_high: float = 135.0,
    delta_t: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Highlight Color Recovery (Ch. 2 & 3):
    Detects OV2640 pink/magenta highlight cast where R and B saturate while G drops,
    and interpolates (R, G, B) back to the true neutral pre-saturation luminance.
    """
    img_f = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_f)

    rb_mean = 0.5 * (r + b)
    excess = np.maximum(0.0, rb_mean - g)

    bright_gate = np.clip((rb_mean - t_high) / 40.0, 0.0, 1.0)
    pink_gate = np.clip((excess - delta_t) / 16.0, 0.0, 1.0)

    alpha = np.clip(bright_gate * pink_gate, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 1.5)

    neutral_est = g + 0.45 * excess
    g_rec = np.clip((1.0 - alpha) * g + alpha * neutral_est, 0.0, 255.0)
    r_rec = np.clip((1.0 - alpha) * r + alpha * (neutral_est + 0.08 * (r - neutral_est)), 0.0, 255.0)
    b_rec = np.clip((1.0 - alpha) * b + alpha * (neutral_est + 0.08 * (b - neutral_est)), 0.0, 255.0)

    img_rec = cv2.merge([b_rec, g_rec, r_rec]).astype(np.uint8)
    return img_rec, alpha


def masked_gray_world_awb(
    img_bgr: np.ndarray,
    alpha_mask: Optional[np.ndarray] = None,
    max_gain_shift: float = 0.06,
) -> np.ndarray:
    """
    Apply subtle Masked Gray World Automatic White Balance on non-saturated mid-tone pixels.
    """
    img_f = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_f)
    lum = 0.114 * b + 0.587 * g + 0.299 * r

    valid = (lum > 30.0) & (lum < 215.0)
    if alpha_mask is not None:
        valid = valid & (alpha_mask < 0.20)

    if np.count_nonzero(valid) < 100:
        valid = np.ones_like(lum, dtype=bool)

    mu_b = float(np.mean(b[valid]))
    mu_g = float(np.mean(g[valid]))
    mu_r = float(np.mean(r[valid]))
    mu_gray = (mu_b + mu_g + mu_r) / 3.0

    kb = np.clip(mu_gray / (mu_b + 1e-6), 1.0 - max_gain_shift, 1.0 + max_gain_shift)
    kg = np.clip(mu_gray / (mu_g + 1e-6), 1.0 - max_gain_shift, 1.0 + max_gain_shift)
    kr = np.clip(mu_gray / (mu_r + 1e-6), 1.0 - max_gain_shift, 1.0 + max_gain_shift)

    kb = 0.70 + 0.30 * kb
    kg = 0.70 + 0.30 * kg
    kr = 0.70 + 0.30 * kr

    out = cv2.merge([
        np.clip(b * kb, 0, 255),
        np.clip(g * kg, 0, 255),
        np.clip(r * kr, 0, 255),
    ]).astype(np.uint8)
    return out


def bilateral_denoise(
    img_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 14.0,
    sigma_space: float = 7.0,
) -> np.ndarray:
    """
    Hybrid Open-Source Wavelet (BayesShrink) + Non-Local Means + Chroma Bilateral Denoising:
    1. Applies BayesShrink Wavelet Denoising (`skimage.restoration.denoise_wavelet`, `sym4`, 3 levels)
       in YCbCr space to suppress multi-scale CMOS sensor noise while preserving fine foliage veins.
    2. Applies OpenCV Fast Non-Local Means (`cv2.fastNlMeansDenoising`) on Luminance L for flat
       homogeneous surfaces (roads, concrete walkways, sky, water) so grain is 100% eliminated.
    3. Applies Median + tight Bilateral filtering on Chroma (a, b) to remove color speckles
       without bleeding colors across small objects (cars, cyclists).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    wav_dn = denoise_wavelet(
        img_rgb,
        method="BayesShrink",
        mode="soft",
        wavelet="sym4",
        wavelet_levels=3,
        convert2ycbcr=True,
        channel_axis=-1,
        rescale_sigma=True,
    )
    img_wav_bgr = np.clip(cv2.cvtColor((np.clip(wav_dn, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR), 0, 255)

    lab_w = cv2.cvtColor(img_wav_bgr, cv2.COLOR_BGR2LAB)
    lw, aw, bw = cv2.split(lab_w)

    aw_clean = cv2.bilateralFilter(cv2.medianBlur(aw, 3), d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    bw_clean = cv2.bilateralFilter(cv2.medianBlur(bw, 3), d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    l_nlm = cv2.fastNlMeansDenoising(lw, None, h=4.5, templateWindowSize=7, searchWindowSize=15).astype(np.float32)

    b_c, g_c, r_c = cv2.split(img_bgr.astype(np.float32))
    foliage_w = cv2.GaussianBlur(np.clip((g_c - np.maximum(r_c, b_c) - 6.0) / 14.0, 0.0, 1.0), (5, 5), 1.2)
    l_clean = np.clip(foliage_w * lw.astype(np.float32) + (1.0 - foliage_w) * l_nlm, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_clean, aw_clean, bw_clean]), cv2.COLOR_LAB2BGR)


def run_stage1_preprocess(
    img_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 14.0,
    sigma_space: float = 7.0,
    t_high: float = 135.0,
    delta_t: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute full Stage 1:
    Soft Pink Highlight Recovery -> Masked Gray World AWB -> Hybrid BayesShrink Wavelet + NLM Denoising.
    """
    img_pink_fixed, alpha_mask = recover_pink_highlights(img_bgr, t_high=t_high, delta_t=delta_t)
    img_awb = masked_gray_world_awb(img_pink_fixed, alpha_mask=alpha_mask)
    img_s1 = bilateral_denoise(img_awb, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
    return img_s1, alpha_mask


# =============================================================================
# STAGE 2: MAIN PROCESSING — CIE LAB HAZE-PIVOT RESTORATION & LOCAL CLAHE
# =============================================================================

def run_stage2_contrast(
    img_bgr: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
    haze_pivot: float = 172.0,
    contrast_gain: float = 1.26,
    clahe_blend: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Haze-Pivot Contrast Restoration & Local CLAHE in CIE LAB space (Ch. 3):
    1. Converts BGR -> CIE LAB to separate Luminance (L) from Chroma (a, b).
    2. Expands L around the atmospheric haze veil pivot (L_pivot = 172.0):
         L_dehaze = L_pivot + contrast_gain * (L - L_pivot)
       Inverts the physical haze veil I_deg = c*I + (1-c)*A without crushing shadows.
    3. Applies local CLAHE (clipLimit=1.5, tileGridSize=(8,8)) blended at 22%.
    4. Restores chroma saturation (a, b) attenuated by haze.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    l_f = l_chan.astype(np.float32)
    p2, p50, p98 = np.percentile(l_f, [2, 50, 98])
    span = max(float(p98 - p2), 30.0)
    if span < 92.0:
        eff_gain = min(158.0, span * 2.05) / span
        eff_pivot = 0.50 * float(p50) + 0.50 * haze_pivot
    else:
        eff_gain = contrast_gain
        eff_pivot = haze_pivot

    l_dehaze = np.clip(eff_pivot + (l_f - eff_pivot) * eff_gain, 0.0, 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_dehaze)

    l_out = np.clip(
        clahe_blend * l_clahe.astype(np.float32) + (1.0 - clahe_blend) * l_dehaze.astype(np.float32),
        0.0,
        255.0,
    ).astype(np.uint8)

    a_f = a_chan.astype(np.float32) - 128.0
    b_f = b_chan.astype(np.float32) - 128.0
    chroma_rms = float(np.sqrt(np.mean(a_f ** 2 + b_f ** 2)))
    c_gain = float(np.clip(16.0 / max(chroma_rms, 6.0), 1.24, 1.58))
    a_out = np.clip(128.0 + a_f * c_gain, 0, 255).astype(np.uint8)
    b_out = np.clip(128.0 + b_f * c_gain, 0, 255).astype(np.uint8)

    lab_out = cv2.merge([l_out, a_out, b_out])
    img_s2 = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return img_s2, l_chan, l_out


# =============================================================================
# STAGE 3: RICHARDSON-LUCY PSF DECONVOLUTION & PRETRAINED ESPCN 720P SUPER-RES
# =============================================================================

def _run_espcn_sr_luminance(l_chan: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Runs Pretrained Open-Source Sub-Pixel CNN (`models/espcn_super_res.onnx`) via `onnxruntime`
    on Luminance channel L in safe BT.601 Y range [16, 235], clamped against Lanczos-4 to prevent
    any overshoot artifacts while restoring crisp high-frequency neural edges.
    """
    lanczos_res = np.clip(
        cv2.resize(l_chan.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LANCZOS4),
        0.0,
        255.0,
    )
    if _ORT_SESS is None:
        return lanczos_res.astype(np.uint8)

    h, w = l_chan.shape
    stride = 200
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    l_safe = 16.0 + l_chan.astype(np.float32) * (219.0 / 255.0)
    l_pad = cv2.copyMakeBorder(l_safe, 12, pad_h + 12, 12, pad_w + 12, cv2.BORDER_REFLECT_101)
    hp, wp = l_pad.shape
    out_3x = np.zeros(((hp - 24) * 3, (wp - 24) * 3), dtype=np.float32)

    for y in range(0, hp - 24, stride):
        for x in range(0, wp - 24, stride):
            tile = l_pad[y:y + 224, x:x + 224] / 255.0
            sr_tile = _ORT_SESS.run(None, {"input": tile[None, None, :, :]})[0][0, 0] * 255.0
            sr_full = (sr_tile[36:636, 36:636] - 16.0) * (255.0 / 219.0)
            out_3x[y * 3:(y + stride) * 3, x * 3:(x + stride) * 3] = np.clip(sr_full, 0.0, 255.0)

    out_crop = out_3x[:h * 3, :w * 3]
    sr_res = np.clip(cv2.resize(out_crop, (target_w, target_h), interpolation=cv2.INTER_AREA), 0.0, 255.0)
    sr_clean = np.clip(sr_res, lanczos_res - 18.0, lanczos_res + 18.0)
    return np.clip(0.65 * sr_clean + 0.35 * lanczos_res, 0.0, 255.0).astype(np.uint8)


def run_stage3_sharpen(
    img_bgr: np.ndarray,
    psf_sigma: float = 0.88,
    rl_iter: int = 4,
    use_espcn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open-Source Richardson-Lucy PSF Deconvolution + Pretrained ESPCN Edge Restoration (Ch. 3 & 4):
    1. Refines Luminance L via Pretrained ESPCN Sub-Pixel CNN (`onnxruntime`).
    2. Applies 4 iterations of Richardson-Lucy Deconvolution (`skimage.restoration.richardson_lucy`)
       using the OV2640 optical Gaussian PSF (sigma=0.88) to reverse optical defocus blur.
    3. Structure- & Foliage-Gated Unsharp Detail injection: keeps flat surfaces (roads, sky, water)
       100% smooth and noise-free while sharpening trees, cars, road markings, and roofs.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_in, a_in, b_in = cv2.split(lab)
    h, w = l_in.shape

    if use_espcn:
        l_espcn = _run_espcn_sr_luminance(l_in, w, h)
        l_seed = 0.55 * l_espcn.astype(np.float32) + 0.45 * l_in.astype(np.float32)
    else:
        l_seed = l_in.astype(np.float32)

    k_1d = cv2.getGaussianKernel(5, psf_sigma)
    psf = (k_1d @ k_1d.T).astype(np.float32)
    l_rl = richardson_lucy(np.clip(l_seed / 255.0, 1e-4, 1.0), psf, num_iter=rl_iter, clip=True) * 255.0

    b_c, g_c, r_c = cv2.split(img_bgr.astype(np.float32))
    fol_w = cv2.GaussianBlur(np.clip((g_c - np.maximum(r_c, b_c) - 6.0) / 14.0, 0.0, 1.0), (5, 5), 1.2)

    l_pre = cv2.GaussianBlur(l_in.astype(np.float32), (5, 5), 1.0)
    gx = cv2.Sobel(l_pre, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_pre, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (5, 5), 1.0)
    edge_w = np.clip((grad - 12.0) / 22.0, 0.0, 1.0)
    active_w = np.maximum(fol_w, edge_w)

    g1 = cv2.GaussianBlur(l_rl, (5, 5), sigmaX=0.85, sigmaY=0.85)
    d_fine = l_rl - g1
    boost = active_w * ((l_rl - l_in.astype(np.float32)) + 0.65 * d_fine)
    boost = 24.0 * np.tanh(boost / 24.0)

    l_sharp = np.clip(l_in.astype(np.float32) + boost, 0.0, 255.0).astype(np.uint8)
    img_sharp = cv2.cvtColor(cv2.merge([l_sharp, a_in, b_in]), cv2.COLOR_LAB2BGR)
    detail_vis = np.clip(128.0 + 4.0 * boost, 0, 255).astype(np.uint8)
    return img_sharp, detail_vis


def upscale_to_hd(
    img_stage2_bgr: np.ndarray,
    target_height: int = 720,
    target_width: int = 0,
) -> np.ndarray:
    """
    Super-Resolves Stage 2 image to HD 720p (960x720 or 1280x720) or Full HD 1080p (1440x1080 or 1920x1080)
    using Pretrained 3x ESPCN ONNX Sub-Pixel CNN on Luminance L + Lanczos-4 on Chroma (a, b),
    followed by Richardson-Lucy PSF Deconvolution and Edge/Foliage Structure-Gated Sharpening.
    """
    h, w = img_stage2_bgr.shape[:2]
    if target_width <= 0:
        scale = target_height / float(h)
        target_width = int(round(w * scale))

    lab = cv2.cvtColor(img_stage2_bgr, cv2.COLOR_BGR2LAB)
    l_in, a_in, b_in = cv2.split(lab)

    l_hd = _run_espcn_sr_luminance(l_in, target_width, target_height)
    a_hd = cv2.resize(a_in, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    b_hd = cv2.resize(b_in, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    up_bgr = cv2.cvtColor(cv2.merge([l_hd, a_hd, b_hd]), cv2.COLOR_LAB2BGR)
    psf_s = 0.92 if target_height <= 720 else 1.05
    up_sharp, _ = run_stage3_sharpen(up_bgr, psf_sigma=psf_s, rl_iter=4, use_espcn=False)
    return up_sharp


def upscale_to_720p(
    img_stage2_bgr: np.ndarray,
    target_height: int = 720,
) -> np.ndarray:
    """
    Alias for upscale_to_hd(img_stage2_bgr, target_height=target_height).
    """
    return upscale_to_hd(img_stage2_bgr, target_height=target_height)


# =============================================================================
# FULL PIPELINE & BASELINES
# =============================================================================

def run_full_pipeline(
    img_bgr: np.ndarray,
    return_intermediates: bool = False,
    upscale_hd: bool = False,
    hd_height: int = 720,
) -> Dict[str, np.ndarray]:
    """
    Run the complete 3-stage ESP32-CAM UAV image restoration pipeline.
    Also produces `output_hd` (720p HD) and `output_fhd` (1080p Full HD) when requested or in intermediates.
    """
    t0 = time.perf_counter()
    img_s1, alpha_mask = run_stage1_preprocess(img_bgr)
    t1 = time.perf_counter()
    img_s2, l_before, l_after = run_stage2_contrast(img_s1)
    t2 = time.perf_counter()
    img_s3, detail_vis = run_stage3_sharpen(img_s2)
    t3 = time.perf_counter()

    out_main = upscale_to_hd(img_s2, target_height=hd_height) if upscale_hd else img_s3

    result = {
        "output": out_main,
        "output_native": img_s3,
        "runtime_ms": (t3 - t0) * 1000.0,
        "stage1_ms": (t1 - t0) * 1000.0,
        "stage2_ms": (t2 - t1) * 1000.0,
        "stage3_ms": (t3 - t2) * 1000.0,
    }
    if return_intermediates:
        img_720p = upscale_to_hd(img_s2, target_height=720)
        img_1080p = upscale_to_hd(img_s2, target_height=1080)
        result.update({
            "stage1": img_s1,
            "alpha_mask": alpha_mask,
            "stage2": img_s2,
            "l_before": l_before,
            "l_after": l_after,
            "detail_vis": detail_vis,
            "output_720p": img_720p,
            "output_1080p": img_1080p,
        })
    return result


def process_uav_video(
    input_video_path: str,
    output_video_path: str,
    mode: str = "side_by_side_720p",
    max_frames: int = 180,
) -> Dict[str, float]:
    """
    Processes an input ESP32-CAM UAV video frame-by-frame through the 3-stage open-source
    restoration + ESPCN Super-Resolution pipeline and encodes H.264 MP4 (`libx264`, `yuv420p`)
    so it plays natively in VS Code, browsers, and media players.

    Supported `mode` values:
    - 'side_by_side_720p': 1280x720 split-screen (Left: Input, Right: Restored + ESPCN)
    - 'side_by_side_1080p': 1920x1080 split-screen (Left: Input, Right: Restored + ESPCN Full HD)
    - 'hd_720p': Full-frame 1280x720 HD restored output
    - 'full_hd_1080p': Full-frame 1920x1080 Full HD restored output
    """
    import subprocess
    cap = cv2.VideoCapture(input_video_path)
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if mode in ("side_by_side_1080p", "full_hd_1080p"):
        out_w, out_h = 1920, 1080
    else:
        out_w, out_h = 1280, 720

    tmp_mp4v = output_video_path + ".tmp_mp4v.mp4"
    writer = cv2.VideoWriter(tmp_mp4v, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (out_w, out_h))

    frames_done = 0
    total_ms = 0.0
    while frames_done < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Process at native VGA width for fast Wavelet+NLM+RL+ESPCN restoration
        proc_in = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        s1, _ = run_stage1_preprocess(proc_in)
        s2, _, _ = run_stage2_contrast(s1)
        if mode == "full_hd_1080p":
            out_frame = upscale_to_hd(s2, target_height=1080, target_width=1920)
        elif mode == "hd_720p":
            out_frame = upscale_to_hd(s2, target_height=720, target_width=1280)
        elif mode == "side_by_side_1080p":
            restored_fhd = upscale_to_hd(s2, target_height=1080, target_width=1920)
            in_fhd = cv2.resize(proc_in, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            out_frame = np.hstack([in_fhd[:, :960], restored_fhd[:, 960:]])
            cv2.line(out_frame, (960, 0), (960, 1080), (0, 255, 180), 4)
            cv2.putText(out_frame, "INPUT: ESP32-CAM (Noise + Blur + Pink)", (24, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (70, 70, 255), 2, cv2.LINE_AA)
            cv2.putText(out_frame, "OUTPUT: Wavelet+NLM + RL + ESPCN Full HD 1080p", (984, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, (70, 255, 100), 2, cv2.LINE_AA)
        else:
            restored_hd = upscale_to_hd(s2, target_height=720, target_width=1280)
            in_hd = cv2.resize(proc_in, (1280, 720), interpolation=cv2.INTER_NEAREST)
            out_frame = np.hstack([in_hd[:, :640], restored_hd[:, 640:]])
            cv2.line(out_frame, (640, 0), (640, 720), (0, 255, 180), 3)
            cv2.putText(out_frame, "Input: ESP32-CAM (OV2640)", (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (70, 70, 255), 2, cv2.LINE_AA)
            cv2.putText(out_frame, "Output: Wavelet+NLM + RL + ESPCN HD 720p", (658, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (70, 255, 100), 2, cv2.LINE_AA)

        dt_ms = (time.perf_counter() - t0) * 1000.0
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
    """
    Baseline 1: Standard Global Histogram Equalization applied independently to R, G, B channels.
    """
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


def baseline_global_he_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Baseline 2: Global Histogram Equalization applied to the L channel in CIE LAB space.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / (np.sum(hist) + 1e-12)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_laplacian_var(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_pink_cast_index(img_bgr: np.ndarray) -> float:
    img_f = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_f)
    rb_mean = 0.5 * (r + b)
    pink_pixels = (rb_mean > 140.0) & ((rb_mean - g) > 10.0)
    return float(np.mean(pink_pixels) * 100.0)
