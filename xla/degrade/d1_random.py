"""D1: làm xấu ngẫu nhiên bậc hai kiểu Real-ESRGAN (Wang et al., ICCVW 2021) — đối chứng "không đo".

Hai vòng [nhòe Gauss đẳng hướng/bất đẳng hướng → resize ngẫu nhiên → nhiễu Gauss/Poisson → JPEG cv2 Q ngẫu nhiên],
kết thúc ở 624×480, dán 16 cột đen ngẫu nhiên → 640×480, JPEG vòng hai chính là lần ghi file.
Module này **không** đọc ``camera_profile.yaml`` (phase 05 bước 4): mọi khoảng lấy từ ``configs/degrade_d1.yaml``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from xla.degrade.geometry import CROP_LEFT, FRAME_H, WIN_H, WIN_W

INTERPS = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def _u(rng: np.random.Generator, lohi) -> float:
    return float(rng.uniform(float(lohi[0]), float(lohi[1])))


def gaussian_kernel(size: int, sx: float, sy: float, theta: float) -> np.ndarray:
    """Kernel Gauss bất đẳng hướng (sx, sy, góc theta rad), chuẩn hoá tổng = 1."""
    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    c, s = math.cos(theta), math.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (k / k.sum()).astype(np.float32)


def _blur(img: np.ndarray, rng: np.random.Generator, st: dict, p: dict) -> np.ndarray:
    size = int(rng.choice(np.arange(st["kernel_size"][0], st["kernel_size"][1] + 1, 2)))
    if rng.random() < st["aniso_prob"]:
        sx, sy, th = _u(rng, st["sigma"]), _u(rng, st["sigma"]), float(rng.uniform(0, math.pi))
    else:
        sx = sy = _u(rng, st["sigma"])
        th = 0.0
    p.update(kernel=size, sigma_x=round(sx, 3), sigma_y=round(sy, 3), theta=round(th, 3))
    return cv2.filter2D(img, -1, gaussian_kernel(size, sx, sy, th), borderType=cv2.BORDER_REFLECT_101)


def _resize(img: np.ndarray, rng: np.random.Generator, st: dict, p: dict) -> np.ndarray:
    mode = rng.choice(["up", "down", "keep"], p=st["resize_prob"])
    if mode == "up":
        f = float(rng.uniform(1.0, st["resize_range"][1]))
    elif mode == "down":
        f = float(rng.uniform(st["resize_range"][0], 1.0))
    else:
        f = 1.0
    interp = str(rng.choice(list(INTERPS)))
    p.update(resize=round(f, 4), interp=interp)
    h, w = img.shape[:2]
    return cv2.resize(img, (max(8, round(w * f)), max(8, round(h * f))), interpolation=INTERPS[interp])


def _noise(img: np.ndarray, rng: np.random.Generator, st: dict, p: dict) -> np.ndarray:
    gray = rng.random() < st["gray_noise_prob"]
    shape = img.shape[:2] + ((1,) if gray else (3,))
    if rng.random() < st["gaussian_noise_prob"]:
        sigma = _u(rng, st["noise_range"]) / 255.0
        n = rng.standard_normal(shape).astype(np.float32) * sigma
        p.update(noise="gauss", noise_sigma_0_255=round(sigma * 255, 3), gray=bool(gray))
    else:
        scale = _u(rng, st["poisson_scale_range"])
        # Real-ESRGAN: vals = 10**(2*rand+4) mức lượng tử; nhiễu = (poisson(img*vals)/vals - img) * scale
        vals = 10 ** (2 * rng.random() + 4)
        base = img.mean(axis=2, keepdims=True) if gray else img
        n = (rng.poisson(np.clip(base, 0, 1) * vals) / vals - base).astype(np.float32) * scale
        p.update(noise="poisson", poisson_scale=round(scale, 4), gray=bool(gray))
    return np.clip(img + n, 0.0, 1.0)


def _jpeg_cv2(img01: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", np.round(img01 * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG hỏng")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0


def random_strip(rng: np.random.Generator, st: dict, p: dict) -> np.ndarray:
    """16 cột đen ngẫu nhiên (không theo đo đạc): độ rộng phần tối, mức tối, nhiễu ngẫu nhiên."""
    width = int(rng.integers(st["dark_width_px"][0], st["dark_width_px"][1] + 1))
    level = _u(rng, st["dark_level_0_255"])
    strip = np.empty((FRAME_H, CROP_LEFT, 3), np.float32)
    strip[:] = level
    strip += rng.normal(0, st["noise_0_255"], strip.shape).astype(np.float32)
    p.update(strip_width=width, strip_level_0_255=round(level, 2))
    return np.clip(strip / 255.0, 0, 1), width


def degrade(win_bgr: np.ndarray, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, int, dict]:
    """Ảnh cửa sổ 624×480 uint8 BGR → (ảnh 640×480 float 0–1 trước JPEG cuối, Q JPEG cuối, tham số).

    JPEG vòng hai được ``encode_final`` thực hiện đúng một lần khi ghi file.
    """
    if win_bgr.shape[:2] != (WIN_H, WIN_W):
        raise ValueError(f"D1 nhận cửa sổ {WIN_W}x{WIN_H}, gặp {win_bgr.shape}")
    params: dict = {"stage1": {}, "stage2": {}}
    img = win_bgr.astype(np.float32) / 255.0
    s1, s2 = cfg["stage1"], cfg["stage2"]

    p = params["stage1"]
    if rng.random() < s1["blur_prob"]:
        img = _blur(img, rng, s1, p)
    img = _resize(img, rng, s1, p)
    img = _noise(img, rng, s1, p)
    q1 = int(rng.integers(s1["jpeg_range"][0], s1["jpeg_range"][1] + 1))
    p["jpeg_q"] = q1
    img = _jpeg_cv2(img, q1)

    p = params["stage2"]
    if rng.random() < s2["blur_prob"]:
        img = _blur(img, rng, s2, p)
    img = _resize(img, rng, s2, p)
    img = _noise(img, rng, s2, p)
    back = str(rng.choice(list(INTERPS)))
    img = cv2.resize(img, (WIN_W, WIN_H), interpolation=INTERPS[back])
    p["resize_back_interp"] = back
    q2 = int(rng.integers(s2["jpeg_range"][0], s2["jpeg_range"][1] + 1))
    p["jpeg_q"] = q2

    strip, width = random_strip(rng, cfg["strip"], params)
    frame = np.concatenate([strip, np.clip(img, 0, 1)], axis=1)
    if width < CROP_LEFT:  # phần cột sau dải tối lấy nội dung ảnh (lặp mép), như khung thật
        frame[:, width:CROP_LEFT] = frame[:, CROP_LEFT:CROP_LEFT + 1]
    return frame, q2, params


def encode_final(frame01: np.ndarray, q: int) -> bytes:
    """JPEG cuối của D1: cv2, chất lượng IJG ``q`` (4:2:0 mặc định của libjpeg) — đúng một lần."""
    ok, buf = cv2.imencode(".jpg", np.round(frame01 * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG hỏng")
    return buf.tobytes()
