"""Tầng thống kê mức thấp của phép đo độ giống thật (phase 06 bước 2).

Mọi phép đo lấy từ `xla.camera.measure` (phase 04), không viết lại. Mỗi đại lượng: Wasserstein-1 giữa phân bố trên tập
thật và tập tổng hợp, chia cho IQR của tập thật; NLF và PSD tính theo từng bin rồi lấy trung bình. Tổng = trung bình có
trọng số (mặc định bằng nhau). DQT + lấy mẫu màu so khớp chính xác và báo riêng, không trộn vào tổng.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xla.eval.patches import read_image


def default_measure() -> SimpleNamespace:
    """Các hàm đo của phase 04; import lười để module này nạp được trước khi lane data gộp."""
    from xla.camera import measure as m

    return SimpleNamespace(
        measure_image=m.measure_image, nlf=m.nlf, psd_radial=m.psd_radial, read_jpeg_header=m.read_jpeg_header
    )


def measure_set(paths: list[Path], crop_left: int, measure: SimpleNamespace | None = None, nlf_bins: int = 8) -> dict:
    """Đo từng ảnh (giải mã qua `read_image` chung). Trả giá trị thô theo khoá để tính khoảng cách."""
    measure = measure or default_measure()
    scalars: dict[str, list[float]] = {}
    nlf_vals: dict[int, list[float]] = {}
    psd_vals: dict[float, list[float]] = {}
    headers: list[dict | None] = []
    bin_w = 256.0 / nlf_bins
    for p in paths:
        img = read_image(p)
        for k, v in measure.measure_image(img, crop_left=crop_left).items():
            scalars.setdefault(k, []).append(float(v))
        for level, sigma in measure.nlf(img, crop_left=crop_left, n_bins=nlf_bins):
            if math.isfinite(level) and math.isfinite(sigma):
                nlf_vals.setdefault(min(int(level // bin_w), nlf_bins - 1), []).append(float(sigma))
        for f, power in measure.psd_radial(img, crop_left=crop_left):
            if math.isfinite(f) and math.isfinite(power) and power > 0:
                psd_vals.setdefault(round(float(f), 4), []).append(math.log10(power))
        headers.append(
            measure.read_jpeg_header(Path(p).read_bytes()) if Path(p).suffix.lower() in (".jpg", ".jpeg") else None
        )
    return {"n_images": len(paths), "scalars": scalars, "nlf": nlf_vals, "psd": psd_vals, "headers": headers}


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """W1 giữa hai phân bố thực nghiệm 1 chiều = ∫|F_a − F_b|."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.sort(np.concatenate([a, b]))
    deltas = np.diff(allv)
    fa = np.searchsorted(a, allv[:-1], side="right") / len(a)
    fb = np.searchsorted(b, allv[:-1], side="right") / len(b)
    return float(np.sum(np.abs(fa - fb) * deltas))


def scaled_w1(real: list[float], synth: list[float]) -> dict:
    """W1 / IQR(thật). IQR = 0 → chia cho độ lệch tuyệt đối trung bình; vẫn 0 → không chia, ghi rõ `scale_kind`."""
    r = np.asarray([v for v in real if math.isfinite(v)])
    s = np.asarray([v for v in synth if math.isfinite(v)])
    if len(r) == 0 or len(s) == 0:
        return {
            "value": float("nan"),
            "w1": float("nan"),
            "scale": float("nan"),
            "scale_kind": "empty",
            "n_real": len(r),
            "n_synth": len(s),
        }
    w = wasserstein1(r, s)
    q75, q25 = np.percentile(r, [75, 25])
    scale, kind = float(q75 - q25), "iqr_real"
    if scale <= 1e-12:
        scale, kind = float(np.mean(np.abs(r - np.median(r)))), "mad_real"
    if scale <= 1e-12:
        scale, kind = 1.0, "none"
    return {"value": w / scale, "w1": w, "scale": scale, "scale_kind": kind, "n_real": len(r), "n_synth": len(s)}


def _curve_distance(real: dict, synth: dict) -> dict:
    bins = sorted(set(real) & set(synth))
    per = {str(b): scaled_w1(real[b], synth[b]) for b in bins}
    vals = [v["value"] for v in per.values() if math.isfinite(v["value"])]
    return {
        "value": float(np.mean(vals)) if vals else float("nan"),
        "bins_compared": len(vals),
        "bins_only_real": len(set(real) - set(synth)),
        "bins_only_synth": len(set(synth) - set(real)),
        "per_bin": per,
    }


def _header_signatures(headers: list[dict | None]) -> dict:
    sigs: dict[tuple, int] = {}
    for h in headers:
        key = (
            ("non_jpeg",)
            if h is None
            else (h.get("subsampling"), tuple(h.get("qtable_luma") or ()), tuple(h.get("qtable_chroma") or ()))
        )
        sigs[key] = sigs.get(key, 0) + 1
    return sigs


def dqt_match(real_headers: list, synth_headers: list) -> dict:
    """So khớp chính xác bảng lượng tử + lấy mẫu màu giữa hai tập (báo riêng, không vào tổng)."""
    r, s = _header_signatures(real_headers), _header_signatures(synth_headers)

    def show(sigs):
        out = []
        for k, n in sorted(sigs.items(), key=lambda kv: -kv[1]):
            out.append(
                {"non_jpeg": True, "n": n}
                if k == ("non_jpeg",)
                else {"subsampling": k[0], "qtable_luma": list(k[1]), "qtable_chroma": list(k[2]), "n": n}
            )
        return out

    return {"match": set(r) == set(s), "real": show(r), "synth": show(s)}


def stat_distance(real_meas: dict, synth_meas: dict, weights: dict[str, float] | None = None) -> dict:
    """Khoảng cách thống kê theo từng đại lượng + tổng có trọng số (trung bình có trọng số, bỏ mục NaN và ghi tên)."""
    weights = weights or {}
    items: dict[str, dict] = {}
    for key in sorted(set(real_meas["scalars"]) | set(synth_meas["scalars"])):
        items[key] = scaled_w1(real_meas["scalars"].get(key, []), synth_meas["scalars"].get(key, []))
    items["nlf"] = _curve_distance(real_meas["nlf"], synth_meas["nlf"])
    items["psd_radial_log10"] = _curve_distance(real_meas["psd"], synth_meas["psd"])
    unknown = sorted(set(weights) - set(items))
    if unknown:
        raise ValueError(f"trọng số cho đại lượng không có: {unknown}")
    used = {k: v["value"] for k, v in items.items() if math.isfinite(v["value"]) and weights.get(k, 1.0) > 0}
    wsum = sum(weights.get(k, 1.0) for k in used)
    total = sum(weights.get(k, 1.0) * v for k, v in used.items()) / wsum if wsum else float("nan")
    return {
        "total": float(total),
        "total_kind": "trung bình có trọng số của W1/IQR(thật)",
        "weights": {k: weights.get(k, 1.0) for k in items},
        "excluded_nan": sorted(k for k, v in items.items() if not math.isfinite(v["value"])),
        "items": items,
        "dqt": dqt_match(real_meas["headers"], synth_meas["headers"]),
        "n_real": real_meas["n_images"],
        "n_synth": synth_meas["n_images"],
    }
