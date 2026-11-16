"""
Data Augmentation & mô phỏng suy giảm vật lý camera UAV ESP32-CAM (OV2640) — tái lập được (có seed).

Sinh tập đánh giá đa kịch bản có Ground Truth từ `data/synthetic_gt/`:
  1. haze_pink       : sương mờ  I = c*J + (1-c)*A  + bão hoà lệch hồng (R/B tăng, G giảm) ở vùng sáng.
  2. low_light       : thiếu sáng ban đêm/chiều tối — giảm phơi sáng phi tuyến (miền tuyến tính + tone curve ISP),
                       nhiễu Poisson-Gaussian CMOS, color cast, nâng mức đen (black level) + nén JPEG.
  3. backlit_uneven  : chiếu sáng không đều / bóng râm cục bộ / ngược sáng (trường chiếu sáng mượt ngẫu nhiên).
  4. motion_jpeg     : motion blur theo góc bay ngẫu nhiên + defocus (đĩa) + nén JPEG khối 8x8 Q=25..60.
  + geometric        : xoay roll, crop/scale, flip (áp dụng đồng thời cho GT và ảnh suy giảm) — kiểm tra độ ổn định
                       tham số (tránh overfit).

Mỗi mẫu dùng một `numpy.random.Generator` riêng suy ra từ (seed, scenario, index) nên kết quả hoàn toàn tái lập.
Chạy:  python -m src.data_augmentation --seed 2026 --out data/augmented
"""

import argparse
import glob
import json
import os
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ("haze_pink", "low_light", "backlit_uneven", "motion_jpeg")


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _smooth_field(h: int, w: int, rng: np.random.Generator, n_blobs: int = 4) -> np.ndarray:
    """Trường mượt ngẫu nhiên [0,1] gồm tổng các blob Gauss + gradient tuyến tính."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), np.float32)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(0.15, 0.45) * w, rng.uniform(0.15, 0.45) * h
        field += rng.uniform(0.5, 1.0) * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    field = field + rng.uniform(0.2, 0.7) * grad
    field -= field.min()
    return field / max(float(field.max()), 1e-6)


# ----------------------------------------------------------------------------- scenarios

def degrade_haze_pink(gt: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Kịch bản 1: sương mờ + bão hoà lệch hồng R/B, giảm G ở vùng sáng."""
    c = rng.uniform(0.55, 0.80)                       # contrast_keep
    airlight = rng.uniform(170, 215)
    a_bgr = np.array([airlight + rng.uniform(0, 14), airlight - rng.uniform(0, 8), airlight + rng.uniform(0, 14)])
    hl = rng.uniform(150, 185)
    r_boost, b_boost, g_drop = rng.uniform(35, 70), rng.uniform(25, 60), rng.uniform(15, 45)
    f = gt.astype(np.float32)
    out = c * f + (1.0 - c) * a_bgr.reshape(1, 1, 3)
    y = 0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]
    gate = np.clip((y - hl) / 40.0, 0.0, 1.0)
    gate = cv2.GaussianBlur(gate, (0, 0), 3.0)
    out[..., 2] += r_boost * gate
    out[..., 0] += b_boost * gate
    out[..., 1] -= g_drop * gate
    sigma = rng.uniform(1.5, 4.0)
    out += rng.normal(0.0, sigma, out.shape).astype(np.float32)
    q = int(rng.integers(55, 85))
    res = _jpeg(np.clip(out, 0, 255).astype(np.uint8), q)
    return res, dict(contrast_keep=round(c, 3), airlight=round(float(airlight), 1), hl_thresh=round(hl, 1),
                     r_boost=round(r_boost, 1), b_boost=round(b_boost, 1), g_drop=round(g_drop, 1),
                     noise_sigma=round(sigma, 2), jpeg_quality=q)


def degrade_low_light(gt: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Kịch bản 2: thiếu sáng — giảm phơi sáng trong miền tuyến tính, nhiễu Poisson-Gaussian, color cast, black level."""
    exposure = float(np.exp(rng.uniform(np.log(0.06), np.log(0.30))))
    k_full = rng.uniform(250.0, 900.0)                # electron ở full-scale (ảnh hưởng độ mạnh nhiễu Poisson)
    sigma_read = rng.uniform(0.004, 0.015)
    cast = np.array([rng.uniform(0.95, 1.20), rng.uniform(0.95, 1.10), rng.uniform(0.80, 1.05)])  # B, G, R
    black = rng.uniform(0.01, 0.08)                   # black level nâng (ISP OV2640 thiếu bù đen)
    lin = np.power(gt.astype(np.float32) / 255.0, 2.2) * exposure * cast.reshape(1, 1, 3)
    photons = rng.poisson(np.clip(lin, 0, None) * k_full).astype(np.float32) / k_full
    noisy = photons + rng.normal(0.0, sigma_read, lin.shape).astype(np.float32)
    enc = np.power(np.clip(noisy, 0.0, 1.0), 1.0 / 2.2)
    enc = enc * (1.0 - black) + black                 # nâng đen
    q = int(rng.integers(70, 92))
    res = _jpeg(np.clip(enc * 255.0, 0, 255).astype(np.uint8), q)
    return res, dict(exposure=round(exposure, 3), photon_full_scale=round(k_full, 0), read_sigma=round(sigma_read, 4),
                     cast_b=round(float(cast[0]), 3), cast_g=round(float(cast[1]), 3), cast_r=round(float(cast[2]), 3),
                     black_level=round(black, 3), jpeg_quality=q)


def degrade_backlit_uneven(gt: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Kịch bản 3: chiếu sáng không đều / ngược sáng / bóng râm cục bộ."""
    h, w = gt.shape[:2]
    field = _smooth_field(h, w, rng, n_blobs=int(rng.integers(2, 5)))
    shadow_floor = rng.uniform(0.18, 0.40)
    hi_gain = rng.uniform(1.05, 1.45)
    illum = shadow_floor + (hi_gain - shadow_floor) * field
    lin = np.power(gt.astype(np.float32) / 255.0, 2.2) * (illum[..., None] ** 2.2)
    sigma_read = rng.uniform(0.003, 0.010)
    lin += rng.normal(0.0, sigma_read, lin.shape).astype(np.float32) * np.sqrt(np.clip(lin, 0, 1) + 0.02)
    enc = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2)
    q = int(rng.integers(65, 90))
    res = _jpeg(np.clip(enc * 255.0, 0, 255).astype(np.uint8), q)
    return res, dict(shadow_floor=round(shadow_floor, 3), highlight_gain=round(hi_gain, 3),
                     read_sigma=round(sigma_read, 4), jpeg_quality=q)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = length // 2
    ang = np.deg2rad(angle_deg)
    for t in np.linspace(-c, c, 4 * length):
        x, y = int(round(c + t * np.cos(ang))), int(round(c - t * np.sin(ang)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    return k / max(float(k.sum()), 1e-6)


def degrade_motion_jpeg(gt: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Kịch bản 4: motion blur theo góc bay ngẫu nhiên + defocus + nén JPEG khối 8x8 (Q=25..60)."""
    length = int(rng.choice([5, 7, 9, 11, 13, 15]))
    angle = float(rng.uniform(0, 180))
    radius = float(rng.uniform(0.8, 2.5))
    out = cv2.filter2D(gt, -1, _motion_kernel(length, angle), borderType=cv2.BORDER_REFLECT)
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx ** 2 + yy ** 2 <= radius ** 2).astype(np.float32)
    out = cv2.filter2D(out, -1, disc / disc.sum(), borderType=cv2.BORDER_REFLECT)
    sigma = rng.uniform(1.0, 3.0)
    out = np.clip(out.astype(np.float32) + rng.normal(0, sigma, out.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(25, 61))
    return _jpeg(out, q), dict(motion_len=length, motion_angle=round(angle, 1), defocus_radius=round(radius, 2),
                               noise_sigma=round(sigma, 2), jpeg_quality=q)


DEGRADATIONS: Dict[str, Callable] = {
    "haze_pink": degrade_haze_pink,
    "low_light": degrade_low_light,
    "backlit_uneven": degrade_backlit_uneven,
    "motion_jpeg": degrade_motion_jpeg,
}


def geometric_augment(img: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Geometric augmentation: xoay roll ±7°, crop/scale 0.85-1.0 (giữ kích thước), flip ngang ngẫu nhiên."""
    h, w = img.shape[:2]
    rot = float(rng.uniform(-7, 7))
    scale = float(rng.uniform(1.0, 1.18))
    flip = bool(rng.random() < 0.5)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, scale)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    if flip:
        out = cv2.flip(out, 1)
    return out, dict(rot_deg=round(rot, 2), scale=round(scale, 3), flip=flip)


def _rng_for(seed: int, tag: str, idx: int) -> np.random.Generator:
    return np.random.default_rng([seed, abs(hash_str(tag)) % (2 ** 31), idx])


def hash_str(s: str) -> int:
    """Hash chuỗi ổn định (không phụ thuộc PYTHONHASHSEED) để tái lập."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def generate_augmented_benchmark(
    gt_dir: str = os.path.join(ROOT, "data", "synthetic_gt"),
    out_dir: str = os.path.join(ROOT, "data", "augmented"),
    seed: int = 2026,
    geo_per_scenario: int = 8,
) -> pd.DataFrame:
    """
    Sinh tập augmented benchmark: mỗi ảnh GT được suy giảm theo CẢ 4 kịch bản (ảnh tham chiếu = GT gốc, không lưu lại),
    cộng thêm nhóm `geometric` (GT bị biến đổi hình học và lưu kèm, sau đó suy giảm theo kịch bản xoay vòng).
    Ghi manifest `manifest.csv` (đường dẫn, kịch bản, seed, tham số JSON).
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.jpg")))
    records = []
    for sc in SCENARIOS:
        os.makedirs(os.path.join(out_dir, sc), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "geometric", "gt"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "geometric", "degraded"), exist_ok=True)

    for i, gf in enumerate(gt_files):
        gt = cv2.imread(gf)
        for sc in SCENARIOS:
            rng = _rng_for(seed, sc, i)
            deg, params = DEGRADATIONS[sc](gt, rng)
            name = f"{sc}_{i + 1:03d}.jpg"
            cv2.imwrite(os.path.join(out_dir, sc, name), deg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append(dict(scenario=sc, filename=f"{sc}/{name}", gt=os.path.relpath(gf, ROOT).replace("\\", "/"),
                                seed=seed, index=i, params=json.dumps(params)))

    n_geo = geo_per_scenario * len(SCENARIOS)
    for j in range(n_geo):
        i = j % len(gt_files)
        sc = SCENARIOS[j % len(SCENARIOS)]
        gt0 = cv2.imread(gt_files[i])
        rng_g = _rng_for(seed, "geometric", j)
        gt_geo, gparams = geometric_augment(gt0, rng_g)
        deg, params = DEGRADATIONS[sc](gt_geo, _rng_for(seed, "geo_" + sc, j))
        name = f"geo_{j + 1:03d}_{sc}.jpg"
        cv2.imwrite(os.path.join(out_dir, "geometric", "gt", name), gt_geo, [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(os.path.join(out_dir, "geometric", "degraded", name), deg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append(dict(scenario="geometric", filename=f"geometric/degraded/{name}",
                            gt=f"data/augmented/geometric/gt/{name}", seed=seed, index=j,
                            params=json.dumps({"base_scenario": sc, **gparams, **params})))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sinh tập augmented benchmark có seed tái lập")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "augmented"))
    ap.add_argument("--geo-per-scenario", type=int, default=8)
    a = ap.parse_args()
    d = generate_augmented_benchmark(out_dir=a.out, seed=a.seed, geo_per_scenario=a.geo_per_scenario)
    print(d.groupby("scenario").size())
    print("Manifest:", os.path.join(a.out, "manifest.csv"))
