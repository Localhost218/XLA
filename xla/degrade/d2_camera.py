"""D2: "bản sao số" của camera ESP32-S3 + OV3660 — làm xấu theo đường vật lý, mỗi tham số khớp một số đo hồ sơ.

Thứ tự (phase 05 bước 5): (a) màn mờ ``I = J·t + A·(1−t)`` → (b) PSF Gauss + tối góc cos⁴ → (c) độ sáng, cắt từng kênh
ở 1,0 **rồi mới** nhân hệ số cân bằng trắng → vùng sáng bão hoà thành hồng → (d) ISP: làm mềm + nhiễu theo NLF, lọc
cho đúng PSD → (e) dải mép trái → (f) JPEG **một lần, cuối cùng**, Pillow ``qtables`` của hồ sơ + 4:2:2, không ``quality``.

(a)–(c) tính trong không gian tuyến tính (sRGB → tuyến tính ở đầu), (d)–(f) trên sRGB.
Hồ sơ: ``configs/camera_profile.yaml`` (contract §6). Tham số chuỗi: ``configs/degrade_d2.yaml``; mỗi khối ghi
``matches:`` trường hồ sơ nó khớp. Thiếu trường hồ sơ cần dùng → ``KeyError`` nêu tên trường, không đoán thay.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from xla.degrade.geometry import CROP_LEFT, FRAME_H, FRAME_W, WIN_H, WIN_W, sample_dist

# ITU-T T.81 Annex K (K.1 luma, K.2 chroma), thứ tự tự nhiên hàng-cột.
ANNEX_K_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
ANNEX_K_CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
                  47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32
REQUIRED_PROFILE_FIELDS = (
    "jpeg.qtable_luma", "jpeg.qtable_chroma", "jpeg.esp32_quality", "jpeg.subsampling",
    "left_strip.width_px", "vignette.corner_center_ratio", "haze.dark_channel_15px_0_255",
    "sharpness.psf_sigma_px_fit", "noise.nlf", "noise.psd_radial", "color.r_over_g", "color.b_over_g",
    "magenta.highlight_y_0_255", "person_box_px.short_side",
)
LUMA_W = np.array([0.114, 0.587, 0.299], np.float32)  # BGR


# ----------------------------------------------------------------------------- hồ sơ

def get_field(profile: dict, dotted: str):
    cur = profile
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] is None:
            raise KeyError(f"camera_profile thiếu trường '{dotted}' (contract §6)")
        cur = cur[part]
    return cur


def load_profile(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"không thấy hồ sơ camera {p} (phase 04 tạo; truyền --profile)")
    prof = yaml.safe_load(p.read_text(encoding="utf-8"))
    from xla.contract import validate_camera_profile  # SSOT schema §6 (phase 02)

    errors = validate_camera_profile(prof)
    if errors:
        raise ValueError(f"{p} lệch schema hồ sơ camera (contract §6): " + "; ".join(errors[:10]))
    for f in REQUIRED_PROFILE_FIELDS:
        get_field(prof, f)
    if str(get_field(prof, "jpeg.subsampling")) != "4:2:2":
        raise ValueError(f"D2 chỉ dựng cho 4:2:2, hồ sơ ghi {get_field(prof, 'jpeg.subsampling')}")
    for k in ("jpeg.qtable_luma", "jpeg.qtable_chroma"):
        t = get_field(prof, k)
        if len(t) != 64 or not all(1 <= int(v) <= 255 for v in t):
            raise ValueError(f"{k} phải là 64 số nguyên 1–255 (thứ tự tự nhiên)")
    return prof


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- tiện ích màu

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def dark_channel(img01: np.ndarray, patch: int = 15) -> np.ndarray:
    return cv2.erode(img01.min(axis=2), np.ones((patch, patch), np.uint8))


def qtables_for(q: int, profile: dict) -> tuple[list[int], list[int], bool]:
    """Bảng DQT (tự nhiên) cho mức quality ``q`` của esp32-camera. q của hồ sơ = bảng đo thật (verified);
    q khác = giả thuyết ``annex_k * q / 16`` (Deferred 1, ``verified: false``)."""
    if int(q) == int(get_field(profile, "jpeg.esp32_quality")):
        return ([int(v) for v in get_field(profile, "jpeg.qtable_luma")],
                [int(v) for v in get_field(profile, "jpeg.qtable_chroma")], True)
    s = 100.0 * q / 16.0

    def scale(tbl):
        return [int(min(255, max(1, math.floor((v * s + 50) / 100)))) for v in tbl]

    return scale(ANNEX_K_LUMA), scale(ANNEX_K_CHROMA), False


# ----------------------------------------------------------------------------- các khâu

def stage_haze(lin: np.ndarray, rng, cfg: dict, profile: dict, p: dict) -> np.ndarray:
    """(a) màn mờ. t chọn theo từng ảnh để dark channel đích (lấy mẫu từ hồ sơ) đạt được: d* = d_J·t + A·(1−t)."""
    target_srgb = sample_dist(get_field(profile, "haze.dark_channel_15px_0_255"), rng) / 255.0 * cfg["dark_scale"]
    target = float(srgb_to_linear(np.array(target_srgb)))
    a_srgb = float(rng.uniform(*cfg["airlight_0_1"]))
    a_lin = float(srgb_to_linear(np.array(a_srgb)))
    tint = np.array(cfg["airlight_tint_bgr"], np.float32)
    dj = float(np.median(dark_channel(lin)))
    t = (a_lin - target) / max(a_lin - dj, 1e-6)
    t = float(np.clip(t, *cfg["t_range"]))
    p.update(dark_target_0_255=round(target_srgb * 255, 2), airlight_0_1=round(a_srgb, 4), t=round(t, 4),
             dark_src_lin=round(dj, 4))
    return lin * t + (a_lin * tint) * (1.0 - t)


def stage_lens(lin: np.ndarray, rng, cfg: dict, profile: dict, p: dict) -> np.ndarray:
    """(b) PSF Gauss (σ theo ``sharpness.psf_sigma_px_fit``) + tối góc cos⁴ (tâm = tâm khung 640×480)."""
    sigma = float(get_field(profile, "sharpness.psf_sigma_px_fit")) * float(rng.uniform(*cfg["psf_sigma_scale"]))
    out = cv2.GaussianBlur(lin, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101) if sigma > 0.05 else lin
    vig = profile.get("vignette", {})
    f = vig.get("focal_px_fit")
    if f is None:  # suy từ tỉ số góc/giữa: cos⁴(θ_góc) = ratio
        ratio = float(get_field(profile, "vignette.corner_center_ratio")["p50"])
        theta = math.acos(min(1.0, ratio) ** 0.25)
        f = math.hypot(FRAME_W / 2, FRAME_H / 2) / max(math.tan(theta), 1e-6)
    f = float(f) * float(rng.uniform(*cfg["vignette_focal_scale"]))
    yy, xx = np.mgrid[0:WIN_H, 0:WIN_W].astype(np.float32)
    r = np.hypot(xx + CROP_LEFT - FRAME_W / 2, yy - FRAME_H / 2)
    fall = np.cos(np.arctan(r / f)) ** 4
    p.update(psf_sigma_px=round(sigma, 4), vignette_focal_px=round(f, 2))
    return out * fall[..., None]


def highlight_gain(profile: dict) -> float:
    """Hệ số ra k (tuyến tính) sao cho điểm bão hoà (R=B=255 sau WB, G = k) có Y = ``magenta.highlight_y`` p50."""
    y = float(get_field(profile, "magenta.highlight_y_0_255")["p50"]) / 255.0
    g_srgb = (y - 0.299 - 0.114) / 0.587
    if not 0.0 < g_srgb < 1.0:
        raise ValueError(f"magenta.highlight_y p50 = {y * 255:.1f} ngoài khoảng mô hình hồng (106–255)")
    return float(srgb_to_linear(np.array(g_srgb)))


def stage_tone(lin: np.ndarray, rng, cfg: dict, profile: dict, p: dict) -> np.ndarray:
    """(c) về miền cảm biến (bỏ hệ số WB của ISP), nhân phơi sáng, cắt từng kênh ở 1,0, nhân lại hệ số WB, hệ số ra k
    và ám màu còn lại.

    Hệ số WB của ISP (``isp_wb_gain_r/b`` trong cấu hình, > 1) KHÔNG đo được từ khung JPEG sau ISP nên không lấy từ hồ
    sơ: ``color.wb_gain_*_fit`` của hồ sơ v1.1.0 chính là p50 của ``r_over_g``/``b_over_g`` (< 1), dùng làm hệ số ISP
    thì vùng bão hoà ra xanh lá thay vì hồng. G bão hoà trước (lin ≥ k/brightness) và dừng ở k, R/B còn tăng →
    R−G, B−G > 0: hồng/tím. Tông chưa bão hoà = ``lin·brightness·cast``; cast = p50 ``r_over_g``/``b_over_g`` của hồ sơ ×
    ``cast_r``/``cast_b``.
    """
    gr = float(cfg["isp_wb_gain_r"])
    gb = float(cfg["isp_wb_gain_b"])
    if min(gr, gb) <= 1.0:
        raise ValueError(f"isp_wb_gain_r/b phải > 1 (hệ số ISP so với G), cấu hình ghi {gr}, {gb}")
    cr = float(get_field(profile, "color.r_over_g")["p50"]) * cfg["cast_r"]
    cb = float(get_field(profile, "color.b_over_g")["p50"]) * cfg["cast_b"]
    k = highlight_gain(profile) * cfg["highlight_gain_scale"]
    bright = float(rng.uniform(*cfg["brightness"]))
    e = bright / k
    raw = lin * np.array([1 / gb, 1.0, 1 / gr], np.float32) * e
    sat = float((raw.max(axis=2) >= 1.0).mean())
    raw = np.minimum(raw, 1.0)
    out = raw * np.array([gb * cb, 1.0, gr * cr], np.float32) * k
    p.update(isp_wb_gain_r=round(gr, 4), isp_wb_gain_b=round(gb, 4), cast_r=round(cr, 4), cast_b=round(cb, 4),
             k_out=round(k, 4), exposure=round(e, 4), brightness=round(bright, 4), sat_frac=round(sat, 5))
    return np.clip(out, 0.0, 1.0)


def _psd_filter(shape: tuple[int, int], psd: list) -> np.ndarray:
    """Biên độ |H(f)| theo PSD xuyên tâm của hồ sơ, chuẩn hoá sao cho nhiễu trắng phương sai 1 ra phương sai 1."""
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    fr = np.hypot(fx, fy)
    f_pts = np.array([float(a) for a, _ in psd])
    p_pts = np.maximum(np.array([float(b) for _, b in psd]), 1e-12)
    order = np.argsort(f_pts)
    logp = np.interp(fr, f_pts[order], np.log(p_pts[order]))
    amp = np.sqrt(np.exp(logp))
    amp[0, 0] = 0.0  # bỏ thành phần DC
    # Phương sai đầu ra của nhiễu trắng phương sai 1 qua H: trung bình |H|² trên toàn phổ (rfft: nhân đôi trừ cột 0/Nyquist).
    wgt = np.full(amp.shape, 2.0)
    wgt[:, 0] = 1.0
    if w % 2 == 0:
        wgt[:, -1] = 1.0
    var = float((wgt * amp ** 2).sum() / (h * w))
    return (amp / math.sqrt(max(var, 1e-12))).astype(np.float32)


def shaped_noise(shape: tuple[int, int], psd: list, rng) -> np.ndarray:
    white = rng.standard_normal(shape).astype(np.float32)
    spec = np.fft.rfft2(white) * _psd_filter(shape, psd)
    return np.fft.irfft2(spec, s=shape).astype(np.float32)


def stage_isp(srgb: np.ndarray, rng, cfg: dict, profile: dict, p: dict) -> np.ndarray:
    """(d) ISP: làm mềm (khử nhiễu của ISP trên ảnh nguồn sạch tương đương làm mềm), rồi nhiễu sau ISP:
    σ(Y) nội suy theo ``noise.nlf``, phổ lọc theo ``noise.psd_radial``; nhiễu chủ yếu ở độ sáng, thêm phần màu nhỏ."""
    soft = float(rng.uniform(*cfg["soften_sigma"]))
    out = cv2.GaussianBlur(srgb, (0, 0), sigmaX=soft) if soft > 0.05 else srgb
    nlf = sorted((float(a), float(b)) for a, b in get_field(profile, "noise.nlf"))
    psd = get_field(profile, "noise.psd_radial")
    scale = float(rng.uniform(*cfg["noise_scale"]))
    y255 = (out @ LUMA_W) * 255.0
    sigma = np.interp(y255, [a for a, _ in nlf], [b for _, b in nlf]).astype(np.float32) * scale / 255.0
    n_y = shaped_noise(out.shape[:2], psd, rng) * sigma
    chroma = float(cfg["chroma_noise_frac"])
    n_c = np.stack([shaped_noise(out.shape[:2], psd, rng) for _ in range(3)], axis=2) * (sigma * chroma)[..., None]
    p.update(soften_sigma=round(soft, 4), noise_scale=round(scale, 4))
    return np.clip(out + n_y[..., None] + n_c, 0.0, 1.0)


def stage_strip(srgb: np.ndarray, rng, cfg: dict, profile: dict, p: dict) -> np.ndarray:
    """(e) đệm 16 cột mép trái → 640×480. Cột [0, width) từ ``start_row`` trở xuống: khối tối xen vạch đỏ đều
    (chu kỳ, pha, màu đo trên khung thật 20261102-100941); phần còn lại của 16 cột lặp phản chiếu nội dung ảnh."""
    width = round(float(get_field(profile, "left_strip.width_px")) + rng.uniform(*cfg["width_jitter_px"]))
    width = int(np.clip(width, 0, CROP_LEFT))
    frame = np.empty((FRAME_H, FRAME_W, 3), np.float32)
    frame[:, CROP_LEFT:] = srgb
    frame[:, :CROP_LEFT] = srgb[:, CROP_LEFT - 1::-1]
    start = int(rng.integers(cfg["start_row"][0], cfg["start_row"][1] + 1))
    period = float(cfg["red_period_px"])
    phase = float(rng.uniform(*cfg["red_phase_px"]))
    band = float(cfg["red_band_px"])
    dark = np.array([cfg["dark_y_0_255"]] * 3, np.float32) / 255.0
    red = np.array(cfg["red_bgr_0_255"], np.float32) / 255.0
    rows = np.arange(FRAME_H)
    is_red = ((rows - phase) % period) < band
    strip = np.where(is_red[:, None, None], red, dark) * np.ones((FRAME_H, max(width, 1), 1), np.float32)
    if width > 0:
        frame[start:, :width] = strip[start:, :width]
        frame[:, : min(width + 3, CROP_LEFT)] = cv2.GaussianBlur(
            frame[:, : min(width + 3, CROP_LEFT)], (0, 0), sigmaX=float(cfg["edge_blur_sigma"]))
    p.update(strip_width=width, strip_start_row=start, red_phase=round(phase, 2))
    return frame


def choose_q(rng, cfg: dict) -> int:
    real_q = int(cfg["real_q"])
    if rng.random() < float(cfg["real_q_weight"]):
        return real_q
    lo, hi = cfg["hypothesis_q_range"]
    return int(rng.integers(lo, hi + 1))


def degrade(win_bgr: np.ndarray, rng, cfg: dict, profile: dict) -> tuple[np.ndarray, int, dict]:
    """Cửa sổ 624×480 uint8 BGR → (khung 640×480 uint8 BGR trước JPEG, q, tham số). JPEG do ``encode_final``."""
    if win_bgr.shape[:2] != (WIN_H, WIN_W):
        raise ValueError(f"D2 nhận cửa sổ {WIN_W}x{WIN_H}, gặp {win_bgr.shape}")
    params: dict = {}
    lin = srgb_to_linear(win_bgr.astype(np.float32) / 255.0)
    lin = stage_haze(lin, rng, cfg["haze"], profile, params.setdefault("haze", {}))
    lin = stage_lens(lin, rng, cfg["lens"], profile, params.setdefault("lens", {}))
    lin = stage_tone(lin, rng, cfg["tone"], profile, params.setdefault("tone", {}))
    srgb = linear_to_srgb(lin)
    srgb = stage_isp(srgb, rng, cfg["isp"], profile, params.setdefault("isp", {}))
    frame = stage_strip(srgb, rng, cfg["strip"], profile, params.setdefault("strip", {}))
    q = choose_q(rng, cfg["jpeg"])
    params["jpeg"] = {"q": q}
    return np.round(frame * 255.0).clip(0, 255).astype(np.uint8), q, params


def encode_final(frame_bgr: np.ndarray, q: int, profile: dict) -> tuple[bytes, bool]:
    """JPEG duy nhất của D2: Pillow, ``qtables`` (thứ tự tự nhiên) + ``subsampling="4:2:2"``, không truyền ``quality``."""
    luma, chroma, verified = qtables_for(q, profile)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_bgr[..., ::-1])).save(
        buf, format="JPEG", qtables=[luma, chroma], subsampling="4:2:2")
    return buf.getvalue(), verified
