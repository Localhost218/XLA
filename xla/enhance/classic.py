"""Đối chứng cổ điển (phase 12): làm rõ khung camera thật bằng numpy/cv2, không mạng nơ-ron, ≤ 30 ms/khung CPU.

Viết lại phần cổ điển của nhóm (``src/uav_pipeline.py``, bản v2 đóng băng, chỉ đọc) sau khi sửa các lỗi đã đo
(``plans/reports/261118-bang-chung/kiem-xla/kiem-xla.md`` §6–8). Đủ bảy việc (a)–(g) của phase 12 bước 1, chạy theo
thứ tự **đảo chuỗi suy giảm của camera** (cảnh mờ → ống kính tối góc → cảm biến cắt kênh + WB của ISP → JPEG):

(a) cắt ``[crop_left, 640)`` → (e) khử khối trên biên lưới 8×8 của khung gốc → (d) khử hồng chỉ khi gần bão hoà **và**
cả R lẫn B vượt G (bỏ cổng sáng 110, không bạc vật đỏ) → (b) chia cho bản đồ tối góc cos⁴ của hồ sơ (flat-field, có
trần hệ số) → (c) khử màn mờ dark channel (A từ phân vị 99,9, ``t`` có sàn + cận dưới từng điểm, guided filter), chặn
mức đen → (f) unsharp trên Y có ngưỡng theo σ nhiễu, chặn vọt lố → (g) đệm cột ``[0, crop_left)`` bằng lặp cột
``crop_left`` (hợp đồng §3).

Vì sao không chạy đúng thứ tự liệt kê a→g (đo trên 200 khung dev, bench phase 12): khử màn mờ nhân chênh lệch màu với
1/t ≈ 2, nên viền hồng nhạt chưa tới luật hồ sơ thành hồng sau khi khử mờ — khử hồng sau khử mờ chỉ giảm 46–68 % điểm
hồng, trước khử mờ giảm 93 %; khử khối sau khử mờ + làm nét thì tỉ số khối tăng 2,73 → 2,81, đặt ngay sau giải mã thì
giảm còn 2,15. Cắt kênh xảy ra ở cảm biến (sau tối góc) nên cổng bão hoà xét trên giá trị chưa bù tối góc.

Giao diện (hợp đồng giữa lane, ``thuc-thi-phase-10-14.md`` §5.1): ``load_params`` và ``enhance``. Tham số ở
``configs/enhance_classic.yaml``; số đo camera (``crop_left``, tiêu cự cos⁴) đọc từ ``configs/camera_profile.yaml``.

CLI::

    python -m xla.enhance.classic --in <jpg|dir> --out <dir> [--bench] --json
    python -m xla.enhance.classic --split dev --limit 200 --bench --bench-out results/enhance/classic_bench.json --json

Mã thoát: 0 = xong (``--bench``: p50 ≤ ``BUDGET_MS``, hoặc vượt nhưng máy hạ xung đo được — mẫu ``% Processor
Performance`` < ``PERF_OK_PCT`` — và có ``--note`` nêu lý do); 1 = bench vượt ngân sách; 2 = sai cách gọi / thiếu dữ liệu.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from xla import contract
from xla.camera import measure as M
from xla.paths import repo_root

CONFIG_REL = Path("configs/enhance_classic.yaml")
BENCH_REL = Path("results/enhance/classic_bench.json")
FRAMES_REL = Path("data/real/frames")
BUDGET_MS = 30.0
"""Ngân sách ms/khung (p50, CPU, 640×480) của phase 12."""
BLACK_CHECK_LEVEL = 3
"""Điểm "bị đẩy về đen" = kênh nhỏ nhất ≤ mức này (thang 0–255); test và bench dùng chung."""
STEPS = ("crop", "deblock", "magenta", "vignette", "dehaze", "sharpen", "pad")
BENCH_NOTES = (
    ("Thứ tự chạy: (a) cắt → (e) khử khối → (d) khử hồng → (b) tối góc → (c) khử mờ → (f) làm nét → (g) đệm; lý do ở "
    "docstring module (khử mờ nhân sắc độ 1/t: khử hồng sau khử mờ chỉ giảm 46–68 % điểm hồng, trước khử mờ 93 %)."),
    "Cổng khử hồng: min(R, B) − G thay (R+B)/2 − G (vật đỏ bão hoà có (R+B)/2 − G lớn); bão hoà từ 230, đủ ở 240.",
    ("A của khử mờ theo từng kênh (A xám làm ám xanh lá của khung bị khuếch đại 1/t); t có cận dưới từng điểm "
    "1 − omega·min_c(I)/mean(A) để J không âm ở vùng tối sát vùng mờ sáng."),
    "Mức đen: phân vị 0,5 của kênh min ra ≥ min(phân vị 0,5 vào, black_target); làm nét chặn vọt lố trong [min, max] 3×3.",
    "Dạng mặt phẳng uint8 giữa các bước: mảng mới > 0,5 MB trên Windows tốn ~0,2 ms/MB cấp trang.",
)
"""Quyết định lane classic tự chốt (chủ dự án vắng, ``thuc-thi-phase-10-14.md``), ghi kèm ``classic_bench.json``."""

_SECTIONS: dict[str, tuple[str, ...]] = {
    "vignette": ("max_gain",),
    "dehaze": ("patch_px", "airlight_percentile", "airlight_range", "stat_stride", "omega", "t_min", "guided_radius",
               "guided_eps", "guided_scale", "black_percentile", "black_target"),
    "magenta": ("sat_level", "sat_ramp", "excess_min", "excess_ramp", "dilate_px", "blur_px", "strength", "passes",
                "texture_sat_ramp", "texture_sigma", "texture_gain"),
    "deblock": ("block_px", "alpha", "beta", "strength", "inner_weight"),
    "sharpen": ("sigma", "amount", "coring_k", "noise_floor"),
}
_LUMA_G_COMP = (0.299 + 0.114) / 0.587
"""Bù G khi trừ R, B cùng một lượng: giữ nguyên Y BT.601 (công thức gốc ``recover_pink_highlights``)."""


# ----------------------------------------------------------------------------- tham số

def load_params(path: str | Path | None = None) -> dict:
    """Đọc ``configs/enhance_classic.yaml`` (hoặc ``path``) + số đo camera của hồ sơ; sai/thiếu → ``ValueError``.

    Trả dict các khối của config cộng ``crop_left`` (``left_strip.crop_left_px``), ``vignette_focal_px``
    (``vignette.focal_px_fit``) và ``profile_id`` lấy từ hồ sơ (qua ``xla.contract.load_profile``). Đường dẫn
    ``camera_profile`` tương đối tính từ gốc repo.
    """
    cfg_path = Path(path) if path is not None else repo_root() / CONFIG_REL
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"không đọc được config đối chứng cổ điển {cfg_path}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: cần ánh xạ YAML ở gốc")  # noqa: TRY004 — giao diện §5.1: lỗi config = ValueError
    prof_rel = raw.get("camera_profile")
    if not isinstance(prof_rel, str) or not prof_rel:
        raise ValueError(f"{cfg_path}: thiếu khoá 'camera_profile'")
    prof_path = Path(prof_rel) if Path(prof_rel).is_absolute() else repo_root() / prof_rel
    try:
        prof = contract.load_profile(prof_path)
    except OSError as e:
        raise ValueError(f"không đọc được hồ sơ camera {prof_path}: {e}") from e
    try:
        crop_left = int(prof["left_strip"]["crop_left_px"])
        focal = float(prof["vignette"]["focal_px_fit"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"{prof_path}: thiếu left_strip.crop_left_px hoặc vignette.focal_px_fit ({e})") from e
    params: dict[str, Any] = {"version": str(raw.get("version", "")), "profile_id": str(prof.get("profile_id", "")),
                              "crop_left": crop_left, "vignette_focal_px": focal}
    for section, keys in _SECTIONS.items():
        block = raw.get(section)
        if not isinstance(block, dict):
            raise ValueError(f"{cfg_path}: thiếu khối '{section}'")  # noqa: TRY004 — như trên
        missing = [k for k in keys if k not in block]
        if missing:
            raise ValueError(f"{cfg_path}: khối '{section}' thiếu {missing}")
        params[section] = {k: block[k] for k in keys}
    _validate(params, cfg_path)
    return params


def _validate(p: dict, src: Path) -> None:
    def need(cond: bool, what: str) -> None:
        if not cond:
            raise ValueError(f"{src}: {what}")

    def num(v: Any) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)

    for section, keys in _SECTIONS.items():
        for k in keys:
            v = p[section][k]
            if k == "airlight_range":
                need(isinstance(v, list) and len(v) == 2 and all(num(x) for x in v) and 0 < v[0] < v[1] <= 255,
                     f"dehaze.airlight_range phải là [lo, hi] với 0 < lo < hi ≤ 255, nhận {v!r}")
            else:
                need(num(v), f"{section}.{k} phải là số hữu hạn, nhận {v!r}")
    need(0 <= p["crop_left"] < contract.FRAME_W - 32, f"crop_left={p['crop_left']} ngoài khoảng hợp lệ")
    need(p["vignette_focal_px"] > 0, "vignette.focal_px_fit của hồ sơ phải > 0")
    need(p["vignette"]["max_gain"] >= 1, "vignette.max_gain phải ≥ 1")
    d = p["dehaze"]
    need(isinstance(d["patch_px"], int) and d["patch_px"] >= 1 and d["patch_px"] % 2 == 1, "dehaze.patch_px lẻ ≥ 1")
    need(0 < d["airlight_percentile"] < 100 and 0 < d["black_percentile"] < 100, "dehaze: phân vị trong (0, 100)")
    need(isinstance(d["stat_stride"], int) and d["stat_stride"] >= 1, "dehaze.stat_stride nguyên ≥ 1")
    need(0 <= d["omega"] <= 1 and 0 < d["t_min"] <= 1, "dehaze: 0 ≤ omega ≤ 1, 0 < t_min ≤ 1")
    need(isinstance(d["guided_radius"], int) and d["guided_radius"] >= 1 and d["guided_eps"] > 0,
         "dehaze.guided_radius nguyên ≥ 1, guided_eps > 0")
    need(0 < d["guided_scale"] <= 1 and 0 <= d["black_target"] < 255, "dehaze: 0 < guided_scale ≤ 1, black_target")
    m = p["magenta"]
    need(isinstance(m["passes"], int) and m["passes"] >= 1, "magenta.passes nguyên ≥ 1")
    need(isinstance(m["dilate_px"], int) and m["dilate_px"] >= 1, "magenta.dilate_px nguyên ≥ 1")
    need(m["sat_ramp"] > 0 and m["excess_ramp"] > 0 and m["texture_sat_ramp"] > 0, "magenta: *_ramp > 0")
    need(isinstance(m["blur_px"], int) and m["blur_px"] >= 1, "magenta.blur_px nguyên ≥ 1")
    need(m["texture_sigma"] > 0 and 0 <= m["strength"] <= 1, "magenta: texture_sigma > 0, 0 ≤ strength ≤ 1")
    b = p["deblock"]
    need(isinstance(b["block_px"], int) and b["block_px"] >= 4, "deblock.block_px nguyên ≥ 4")
    need(b["alpha"] > 0 and b["beta"] > 0 and 0 <= b["strength"] <= 0.5 and 0 <= b["inner_weight"] <= 1,
         "deblock: alpha, beta > 0; 0 ≤ strength ≤ 0,5; 0 ≤ inner_weight ≤ 1")
    s = p["sharpen"]
    need(s["sigma"] > 0 and s["amount"] >= 0 and s["coring_k"] >= 0 and s["noise_floor"] >= 0, "sharpen: giá trị âm")


@lru_cache(maxsize=1)
def _default_params() -> dict:
    return load_params()


# ----------------------------------------------------------------------------- các bước (mặt phẳng B, G, R uint8)
# Mỗi bước nhận và trả ``Planes`` = ba mặt phẳng uint8 [h, w] liền bộ nhớ (B, G, R) của phần đã cắt mép; không đổi mảng
# vào. Dạng mặt phẳng uint8 vì bench phase 12 đo được: trên Windows mỗi mảng mới > ~0,5 MB (ảnh 3 kênh, float32 đầy khung)
# là một lần cấp phát trang mới, ~0,2 ms/MB; mặt phẳng uint8 300 KB thì được heap dùng lại. Phép float chỉ làm trên ảnh
# nhỏ, trên các cột/hàng biên khối, hoặc một mặt phẳng.

Planes = tuple[np.ndarray, np.ndarray, np.ndarray]
_LUMA_BGR = (0.114, 0.587, 0.299)
"""Hệ số Y BT.601 theo thứ tự B, G, R (như ``cv2.COLOR_BGR2GRAY``)."""


def _u8(v: np.ndarray) -> np.ndarray:
    """float → uint8, làm tròn, bão hoà 0–255."""
    return cv2.convertScaleAbs(np.maximum(v, 0.0, out=v))


def _luma(p: Planes) -> np.ndarray:
    """Y float32 0–255 (BT.601) của ba mặt phẳng."""
    y = cv2.addWeighted(p[0], _LUMA_BGR[0], p[1], _LUMA_BGR[1], 0.0, dtype=cv2.CV_32F)
    return cv2.addWeighted(y, 1.0, p[2], _LUMA_BGR[2], 0.0, dst=y, dtype=cv2.CV_32F)


def _deblock_axis(planes: list[np.ndarray], y: np.ndarray, d: dict, axis: int, phase: int) -> None:
    """San bậc thang theo Y trên biên lưới của một trục (1 = biên dọc, 0 = biên ngang); sửa tại chỗ ``planes`` và
    ``y`` (cộng cùng lượng vào cả ba kênh nên Y theo kịp, khỏi tính lại)."""
    blk = int(d["block_px"])
    c0 = phase if axis == 1 else 0
    nb = (y.shape[axis] - c0) // blk

    def blocks(a: np.ndarray) -> np.ndarray:
        return a[:, c0:c0 + nb * blk].reshape(a.shape[0], nb, blk) if axis == 1 else a[:nb * blk].reshape(nb, blk, -1)

    def at(v: np.ndarray, j: int, left: bool) -> np.ndarray:
        if axis == 1:
            return v[:, :-1, j] if left else v[:, 1:, j]
        return v[:-1, j] if left else v[1:, j]

    yv = blocks(y)
    f1, f0, g0, g1 = (at(yv, j, left).astype(np.float32)
                      for j, left in ((blk - 2, True), (blk - 1, True), (0, False), (1, False)))
    step = g0 - f0
    w_edge = np.clip(1.0 - np.abs(step) * (1.0 / float(d["alpha"])), 0.0, 1.0)
    w_tex = np.clip(1.0 - np.maximum(np.abs(f1 - f0), np.abs(g1 - g0)) * (1.0 / float(d["beta"])), 0.0, 1.0)
    corr = float(d["strength"]) * w_edge * w_tex * step
    inner = float(d["inner_weight"]) * corr
    for v in [blocks(pl) for pl in planes] + [yv]:
        for j, left, c in ((blk - 1, True, corr), (0, False, -corr), (blk - 2, True, inner), (1, False, -inner)):
            tgt = at(v, j, left)
            tgt[...] = _u8(tgt.astype(np.float32) + c)


def deblock(p: Planes, params: dict) -> Planes:
    """(e) Khử khối JPEG trên biên lưới ``block_px`` của **khung gốc** (cột lệch ``crop_left``): theo Y, sửa 2 điểm mỗi
    bên khi bậc thang nhỏ (< ``alpha``) và hai bên phẳng (< ``beta``); bậc lớn hoặc có vân là cạnh thật, giữ nguyên.
    Lượng sửa cộng như nhau vào ba kênh (giữ sắc độ). Biên dọc trước, rồi biên ngang trên ảnh đã sửa; chỉ xét biên giữa
    hai khối đủ cỡ (khối lẻ ở mép bỏ qua)."""
    d = params["deblock"]
    phase = (-int(params["crop_left"])) % int(d["block_px"])
    planes = [pl.copy() for pl in p]
    y = _u8(_luma(p))
    _deblock_axis(planes, y, d, 1, phase)
    _deblock_axis(planes, y, d, 0, phase)
    return planes[0], planes[1], planes[2]


def _ramp_u8(v: np.ndarray, start: float, ramp: float) -> np.ndarray:
    """``clip((v − start)/ramp, 0, 1)`` theo thang uint8 0–255."""
    k = 255.0 / ramp
    return cv2.addWeighted(v, k, v, 0.0, -start * k, dtype=cv2.CV_8U)


def demagenta(p: Planes, params: dict) -> Planes:
    """(d) Khử hồng do ISP cắt kênh: chỉ dịch sắc độ, giữ Y (công thức ``recover_pink_highlights``, đã sửa cổng).

    Cổng = bão hoà (``max(R, B)`` từ ``sat_level``, đủ 1 ở ``sat_level + sat_ramp``) × hồng (``min(R, B) − G`` từ
    ``excess_min``): cả R lẫn B phải vượt G như luật hồng của hồ sơ, nên vật đỏ bão hoà (B thấp) không bị đụng; bỏ cổng
    sáng 110 của bản cũ. Mặt nạ nở ``dilate_px`` (vuông) rồi làm mềm bằng box ``blur_px`` → α. Mỗi lượt:
    ``Δ = strength·α·((R+B)/2 − G)``; ``R −= Δ``, ``B −= Δ``, ``G += 0,704·Δ``; nơi ``max(R, B)`` từ
    ``sat_level + sat_ramp`` trở lên được bơm thêm high-pass của G (kênh còn thông tin). Tính bằng số nguyên của cv2
    (uint8/int16, làm tròn từng phép, sai ≤ 1 mức so với float). Không điểm nào qua cổng → trả bản sao nguyên ảnh.
    """
    m = params["magenta"]
    b, g, r = (pl.copy() for pl in p)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(m["dilate_px"]), int(m["dilate_px"])))
    box = (int(m["blur_px"]), int(m["blur_px"]))
    sat_lo = float(m["sat_level"])
    sat_full = sat_lo + float(m["sat_ramp"])
    for _ in range(int(m["passes"])):
        mx = cv2.max(r, b)
        sat = _ramp_u8(mx, sat_lo, float(m["sat_ramp"]))
        if not cv2.countNonZero(sat):
            break
        pink = _ramp_u8(cv2.subtract(cv2.min(r, b), g), float(m["excess_min"]), float(m["excess_ramp"]))
        gate = cv2.multiply(sat, pink, scale=1.0 / 255.0)
        if not cv2.countNonZero(gate):
            break
        alpha = cv2.blur(cv2.dilate(gate, kern), box)
        excess = cv2.subtract(cv2.addWeighted(r, 0.5, b, 0.5, 0.0), g)
        delta = cv2.multiply(alpha, excess, scale=float(m["strength"]) / 255.0)
        tex_w = cv2.multiply(alpha, _ramp_u8(mx, sat_full, float(m["texture_sat_ramp"])), scale=1.0 / 255.0)
        hp = cv2.subtract(g, cv2.GaussianBlur(g, (0, 0), float(m["texture_sigma"])), dtype=cv2.CV_16S)
        inj = cv2.multiply(hp, tex_w, scale=float(m["texture_gain"]) / 255.0, dtype=cv2.CV_16S)
        g_up = cv2.addWeighted(delta, _LUMA_G_COMP, delta, 0.0, 0.0)
        b = cv2.add(cv2.subtract(b, delta, dtype=cv2.CV_16S), inj, dtype=cv2.CV_8U)
        r = cv2.add(cv2.subtract(r, delta, dtype=cv2.CV_16S), inj, dtype=cv2.CV_8U)
        g = cv2.add(cv2.add(g, g_up, dtype=cv2.CV_16S), inj, dtype=cv2.CV_8U)
    return b, g, r


@lru_cache(maxsize=8)
def _vignette_gain(h: int, w: int, focal_px: float, max_gain: float) -> np.ndarray:
    """``1/cos⁴(atan(r/f)) = (1 + (r/f)²)²``, kẹp ``max_gain``; tâm = tâm phần đã cắt (như phép đo hồ sơ)."""
    yy = (np.arange(h, dtype=np.float32) + 0.5 - h / 2) / focal_px
    xx = (np.arange(w, dtype=np.float32) + 0.5 - w / 2) / focal_px
    u2 = yy[:, None] ** 2 + xx[None, :] ** 2
    gain = np.minimum((1.0 + u2) ** 2, np.float32(max_gain)).astype(np.float32)
    gain.setflags(write=False)
    return gain


def flat_field(p: Planes, params: dict) -> Planes:
    """(b) Chia cho tối góc cos⁴ của hồ sơ (= nhân hệ số bù ``≤ max_gain``), bão hoà ở 255."""
    h, w = p[0].shape
    gain = _vignette_gain(h, w, float(params["vignette_focal_px"]), float(params["vignette"]["max_gain"]))
    b, g, r = (cv2.multiply(pl, gain, dtype=cv2.CV_8U) for pl in p)
    return b, g, r


def _guided(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter (He 2013) bằng box filter; ảnh dẫn và ảnh vào float32 một kênh cùng cỡ."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_g = cv2.boxFilter(guide, -1, k)
    mean_s = cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(guide * src, -1, k) - mean_g * mean_s
    var = cv2.boxFilter(guide * guide, -1, k) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def dehaze(p: Planes, params: dict) -> Planes:
    """(c) Khử màn mờ ``J_c = (I_c − A_c)/t + A_c`` (He 2009) rồi chặn mức đen.

    ``dc`` = dark channel (min kênh, erode ``patch_px``); ``A_c`` = trung bình từng kênh của các điểm có ``dc`` ≥ phân vị
    ``airlight_percentile`` (giữ màu của màn mờ), kẹp ``airlight_range``; ``t = 1 − omega·dc/mean(A)`` tinh chỉnh bằng
    guided filter (ảnh dẫn Y) ở ``guided_scale`` độ phân giải rồi phóng song tuyến lên đầy khung, sàn ``t_min`` — camera
    nhìn thẳng xuống nên bề dày màn mờ gần như đều, ``t`` không cần biên sắc từng điểm. Mức đen (phân vị
    ``black_percentile`` của kênh min) của ảnh ra không thấp hơn ``min(mức đen vào, black_target)``: thấp hơn thì kéo giãn
    tuyến tính, giữ điểm trắng 255. Phân vị tính trên lưới lấy mẫu ``stat_stride``.
    """
    d = params["dehaze"]
    s = int(d["stat_stride"])
    mn = cv2.min(cv2.min(p[0], p[1]), p[2])
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(d["patch_px"]), int(d["patch_px"])))
    dc = cv2.erode(mn, kern)
    dc_s = dc[::s, ::s]
    sel = dc_s >= np.percentile(dc_s, float(d["airlight_percentile"]))
    lo_a, hi_a = (float(v) for v in d["airlight_range"])
    x_s = [np.asarray(pl[::s, ::s], np.float32) for pl in p]
    air = [float(np.clip(c[sel].mean(), lo_a, hi_a)) for c in x_s]
    h, w = mn.shape
    scale = float(d["guided_scale"])
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    dc_small = cv2.resize(dc, size, interpolation=cv2.INTER_AREA)
    t0 = cv2.addWeighted(dc_small, -float(d["omega"]) / float(np.mean(air)), dc_small, 0.0, 1.0, dtype=cv2.CV_32F)
    small = [cv2.resize(pl, size, interpolation=cv2.INTER_AREA) for pl in p]
    guide = _luma((small[0], small[1], small[2]))
    guide *= np.float32(1 / 255)
    t_small = _guided(guide, t0, max(1, round(int(d["guided_radius"]) * scale)), float(d["guided_eps"]))
    np.clip(t_small, float(d["t_min"]), 1.0, out=t_small)
    t = cv2.resize(t_small, (w, h), interpolation=cv2.INTER_LINEAR)
    # Cận dưới từng điểm: t ≥ 1 − omega·min_c(I)/mean(A) là t của chính điểm đó khi chưa làm mịn, nên J không âm —
    # làm mịn (guided, phóng) kéo t của vùng tối sát vùng mờ sáng xuống dưới mức này thì thành viền đen.
    cv2.max(t, cv2.addWeighted(mn, -float(d["omega"]) / float(np.mean(air)), mn, 0.0, 1.0, dtype=cv2.CV_32F), dst=t)
    # Chặn mức đen tính trên lưới lấy mẫu trước khi dựng J đầy khung.
    bp = float(d["black_percentile"])
    inv_s = 1.0 / t[::s, ::s]
    j_min = np.minimum(np.minimum((x_s[0] - air[0]) * inv_s + air[0], (x_s[1] - air[1]) * inv_s + air[1]),
                       (x_s[2] - air[2]) * inv_s + air[2])
    lo_in = float(np.percentile(mn[::s, ::s], bp))
    lo_out = float(np.percentile(j_min, bp))
    floor = min(lo_in, float(d["black_target"]))
    k, off = 1.0, 0.0
    if lo_out < floor:
        k = (255.0 - floor) / (255.0 - lo_out)
        off = floor - lo_out * k
    gain = cv2.divide(k, t)  # k·J + off = (I − A)·k/t + (k·A + off)
    buf = t  # t không dùng nữa: dùng lại bộ nhớ
    out = []
    for pl, a_c in zip(p, air, strict=True):
        cv2.subtract(pl, a_c, dst=buf, dtype=cv2.CV_32F)
        cv2.multiply(buf, gain, dst=buf)
        out.append(cv2.add(buf, k * a_c + off, dtype=cv2.CV_8U))
    return out[0], out[1], out[2]


def sharpen(p: Planes, params: dict) -> Planes:
    """(f) Unsharp trên Y: chi tiết ``Y − Gauss(Y, σ)``, bỏ phần ``|d| ≤ coring_k·σ_nhiễu`` (σ Immerkær đo trên nửa
    giữa ảnh vào bước này — ¼ số điểm, cho nhanh — sàn ``noise_floor``), nhân ``amount``, kẹp để Y mới không vượt
    [min, max] lân cận 3×3 (không vọt lố thành viền), rồi cộng như nhau vào ba kênh."""
    s = params["sharpen"]
    y = _luma(p)
    h, w = y.shape
    sigma_n = max(M.immerkaer_sigma(y[h // 4:h - h // 4, w // 4:w - w // 4]), float(s["noise_floor"]))
    det = cv2.GaussianBlur(y, (0, 0), float(s["sigma"]))
    np.subtract(y, det, out=det)
    thr = float(s["coring_k"]) * sigma_n
    coring = np.clip(det, -thr, thr)
    det -= coring
    det *= float(s["amount"])
    # Chặn vọt lố: Y sau làm nét nằm trong [min, max] lân cận 3×3 của Y trước → không viền đen/trắng ở cạnh mạnh.
    # min/max lấy trên Y uint8 (rẻ hơn float, lệch ≤ 0,5 mức).
    y8 = cv2.convertScaleAbs(y)
    box = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    det += y
    np.minimum(det, cv2.dilate(y8, box), out=det)
    np.maximum(det, cv2.erode(y8, box), out=det)
    det -= y
    b, g, r = (cv2.add(pl, det, dtype=cv2.CV_8U) for pl in p)
    return b, g, r


# ----------------------------------------------------------------------------- toàn chuỗi

STAGES: tuple[tuple[str, Callable[[Planes, dict], Planes]], ...] = (
    ("deblock", deblock), ("magenta", demagenta), ("vignette", flat_field), ("dehaze", dehaze), ("sharpen", sharpen))
"""Thứ tự = đảo chuỗi suy giảm của camera (JPEG ← cắt kênh/WB của ISP ← tối góc ← màn mờ), xem docstring module."""


def to_planes(img_bgr: np.ndarray) -> Planes:
    """uint8 [h, w, 3] BGR → ba mặt phẳng (B, G, R) liền bộ nhớ (bản sao)."""
    b, g, r = cv2.split(img_bgr)
    return b, g, r


def from_planes(p: Planes) -> np.ndarray:
    """Ba mặt phẳng (B, G, R) → uint8 [h, w, 3] BGR."""
    return cv2.merge(list(p))


def _check_frame(frame_bgr: Any) -> None:
    if not isinstance(frame_bgr, np.ndarray):
        # Hợp đồng §5.1 chốt "sai kiểu/cỡ → ValueError" (abeval bắt đúng lớp này), nên không đổi sang TypeError.
        raise ValueError(f"cần numpy.ndarray uint8 {contract.FRAME_SHAPE} BGR, nhận {type(frame_bgr).__name__}")  # noqa: TRY004
    if frame_bgr.dtype != np.uint8 or frame_bgr.shape != contract.FRAME_SHAPE:
        raise ValueError(f"cần uint8 {contract.FRAME_SHAPE} BGR, nhận {frame_bgr.dtype} {frame_bgr.shape}")


def _run(frame_bgr: np.ndarray, params: dict, timings: dict[str, float] | None) -> np.ndarray:
    _check_frame(frame_bgr)
    c = int(params["crop_left"])
    clock = time.perf_counter
    marks: list[tuple[str, float]] = [("", clock())]

    def mark(name: str) -> None:
        if timings is not None:
            marks.append((name, clock()))

    p = to_planes(frame_bgr[:, c:])
    mark("crop")
    for name, fn in STAGES:
        p = fn(p, params)
        mark(name)
    out = np.empty(contract.FRAME_SHAPE, np.uint8)
    out[:, c:] = from_planes(p)
    if c:
        out[:, :c] = out[:, c:c + 1]
    mark("pad")
    if timings is not None:
        for (_, t_prev), (name, t_now) in pairwise(marks):
            timings[name] = timings.get(name, 0.0) + (t_now - t_prev) * 1000.0
    return out


def enhance(frame_bgr: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Làm rõ một khung: uint8 ``[480, 640, 3]`` BGR nguyên khung camera → uint8 ``[480, 640, 3]`` BGR.

    Cột ``[0, crop_left)`` của ảnh ra = lặp cột ``crop_left`` (hợp đồng §3). Hàm thuần, tất định, không đổi mảng vào.
    Sai kiểu/cỡ → ``ValueError``. ``params=None`` → ``load_params()`` (đọc một lần, giữ trong tiến trình).
    """
    return _run(frame_bgr, _default_params() if params is None else params, None)


# ----------------------------------------------------------------------------- bench + CLI

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _dist(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, float)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "mean": round(float(a.mean()), 3)}


def black_frac(img_bgr: np.ndarray, crop_left: int, level: int = BLACK_CHECK_LEVEL) -> float:
    """Tỉ lệ điểm (bỏ cột ``[0, crop_left)``) có kênh nhỏ nhất ≤ ``level``."""
    return float((img_bgr[:, crop_left:].min(axis=2) <= level).mean())


def dark_channel_median(img_bgr: np.ndarray, crop_left: int) -> float:
    """Trung vị dark channel ``M.DARK_CHANNEL_PATCH`` px (bỏ cột ``[0, crop_left)``), cùng cách đo với hồ sơ."""
    im = np.ascontiguousarray(img_bgr[:, crop_left:])
    k = np.ones((M.DARK_CHANNEL_PATCH, M.DARK_CHANNEL_PATCH), np.uint8)
    return float(np.median(cv2.erode(im.min(axis=2), k)))


def pink_count(img_bgr: np.ndarray, crop_left: int) -> int:
    """Số điểm hồng theo luật hồ sơ (``R − G > PINK_DIFF`` và ``B − G > PINK_DIFF``, ``xla.camera.measure``)."""
    im = img_bgr[:, crop_left:].astype(np.int16)
    b, g, r = im[..., 0], im[..., 1], im[..., 2]
    return int((((r - g) > M.PINK_DIFF) & ((b - g) > M.PINK_DIFF)).sum())


def bench(frames: list[np.ndarray], params: dict, warmup: int) -> dict[str, Any]:
    """Đo ms/khung trong một tiến trình (số luồng cv2 do người gọi đặt): chạy ``warmup`` khung đầu không tính, rồi đo
    mọi khung (tổng + từng bước, ``STEPS``); sau đó so ảnh vào/ra bằng các phép đo của hồ sơ (không tính giờ)."""
    for f in frames[:warmup]:
        _run(f, params, None)
    totals: list[float] = []
    per_step: dict[str, list[float]] = {s: [] for s in STEPS}
    outs: list[np.ndarray] = []
    for f in frames:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        outs.append(_run(f, params, t))
        totals.append((time.perf_counter() - t0) * 1000.0)
        for s in STEPS:
            per_step[s].append(t[s])
    c = int(params["crop_left"])
    quality: dict[str, Any] = {}
    for key, fn in (("pink_px", pink_count), ("dark_channel_median", dark_channel_median),
                    ("black_frac_le3", black_frac), ("blocking_ratio_8x8", None)):
        before, after = [], []
        for f, o in zip(frames, outs, strict=True):
            if fn is None:
                before.append(M.blocking_ratio(M.luma(np.ascontiguousarray(f[:, c:])), c))
                after.append(M.blocking_ratio(M.luma(np.ascontiguousarray(o[:, c:])), c))
            else:
                before.append(fn(f, c))
                after.append(fn(o, c))
        quality[key] = {"before_mean": round(float(np.mean(before)), 6), "after_mean": round(float(np.mean(after)), 6)}
    pink_b = sum(pink_count(f, c) for f in frames)
    pink_a = sum(pink_count(o, c) for o in outs)
    quality["pink_px"]["reduction_total"] = round(1.0 - pink_a / pink_b, 6) if pink_b else None
    quality["black_frac_le3"]["max_increase"] = round(
        max(black_frac(o, c) - black_frac(f, c) for f, o in zip(frames, outs, strict=True)), 6)
    total = _dist(totals)
    return {"ms": {"total": total, "steps": {s: _dist(v) for s, v in per_step.items()}},
            "budget_ms": BUDGET_MS, "pass": total["p50"] <= BUDGET_MS, "quality": quality, "outputs": outs}


def _cpu_load_probe() -> Callable[[], float | None]:
    """Đo tải CPU toàn máy (%) trong khoảng bench — các lane khác chạy song song làm số ms dao động. Không có psutil
    (gói kéo theo bởi ultralytics) thì ghi ``None``."""
    try:
        import psutil
    except ImportError:
        print("cảnh báo: không có psutil, không ghi tải CPU lúc bench", file=sys.stderr)
        return lambda: None
    psutil.cpu_percent(interval=None)
    return lambda: float(psutil.cpu_percent(interval=None))


PERF_OK_PCT = 80.0
"""``% Processor Performance`` tối thiểu (mọi mẫu trong lúc bench) để coi máy không hạ xung."""
PERF_SAMPLE_S = 2


class _PerfSampler:
    """Lấy mẫu ``% Processor Performance`` của Windows (100 = xung danh định) mỗi ``PERF_SAMPLE_S`` giây bằng một tiến
    trình PowerShell chạy song song với bench. Máy hạ xung (quá nhiệt, sạc yếu) thì ms tăng vài lần, nên số này đi kèm
    kết quả. Không phải Windows / PowerShell lỗi → ``summary()`` trả ``None`` kèm cảnh báo, không đoán."""

    _CMD = (r"Get-Counter '\Processor Information(_Total)\% Processor Performance' -SampleInterval {s} -Continuous | "
            "ForEach-Object {{ [Console]::Out.WriteLine($_.CounterSamples[0].CookedValue); [Console]::Out.Flush() }}")

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        if sys.platform != "win32":
            print("cảnh báo: không phải Windows, không ghi % Processor Performance", file=sys.stderr)
            return
        try:
            self._proc = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", self._CMD.format(s=PERF_SAMPLE_S)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            print(f"cảnh báo: không chạy được PowerShell để đo % Processor Performance ({e})", file=sys.stderr)
        time.sleep(PERF_SAMPLE_S + 1)  # có ít nhất một mẫu trước khi bench bắt đầu

    def summary(self) -> dict[str, Any] | None:
        if self._proc is None:
            return None
        self._proc.terminate()
        try:
            out, _err = self._proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            out, _err = self._proc.communicate()
        vals = []
        for line in out.splitlines():
            try:
                vals.append(float(line.strip().replace(",", ".")))
            except ValueError:
                continue
        if not vals:
            print("cảnh báo: không đọc được mẫu % Processor Performance nào", file=sys.stderr)
            return None
        return {"min": round(min(vals), 1), "mean": round(float(np.mean(vals)), 1), "max": round(max(vals), 1),
                "n_samples": len(vals), "interval_s": PERF_SAMPLE_S}


def _env() -> dict[str, Any]:
    return {"cpu": _cpu_name(), "logical_cpus": os.cpu_count(), "cv2_threads": cv2.getNumThreads(),
            "numpy": np.__version__, "opencv": cv2.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "processes": 1}


def _list_inputs(args: argparse.Namespace) -> tuple[list[Path], dict[str, Any]]:
    if args.split:
        split_doc = json.loads((repo_root() / "data/real/split.json").read_text(encoding="utf-8"))
        names = sorted(d["frame"] for d in split_doc["frames"] if d["split"] == args.split)
        n_total = len(names)
        if args.limit and args.limit < len(names):
            idx = np.unique(np.round(np.linspace(0, len(names) - 1, args.limit)).astype(int))
            names = [names[i] for i in idx]
        root = repo_root() / FRAMES_REL
        return [root / n for n in names], {"split": args.split, "n_split": n_total, "frames_dir": FRAMES_REL.as_posix(),
                                           "selection": "cách đều theo tên đã sắp (linspace)"}
    src = Path(args.inp)
    paths = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) if src.is_dir() else [src]
    if args.limit:
        paths = paths[:args.limit]
    return paths, {"input": str(src)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.classic", description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="inp", help="một ảnh .jpg/.png hoặc thư mục")
    g.add_argument("--split", choices=["dev", "test", "neg"], help="khung thật của data/real/split.json")
    ap.add_argument("--out", help="thư mục ghi ảnh ra (.png, không nén mất)")
    ap.add_argument("--config", help="config thay configs/enhance_classic.yaml")
    ap.add_argument("--limit", type=int, default=0, help="số khung tối đa (split: lấy cách đều)")
    ap.add_argument("--bench", action="store_true", help="đo ms/khung p50/p95 + từng bước")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--bench-out", help=f"ghi JSON bench (vd. {BENCH_REL.as_posix()})")
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads (0 = mặc định của cv2)")
    ap.add_argument("--note", action="append", default=[], help="ghi chú của lần đo (vd. lý do máy hạ xung), lặp được")
    ap.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    args = ap.parse_args(argv)
    if args.limit < 0 or args.warmup < 0 or args.threads < 0:
        ap.error("--limit, --warmup, --threads phải ≥ 0")
    if args.threads:
        cv2.setNumThreads(args.threads)
    try:
        params = load_params(args.config)
        paths, source = _list_inputs(args)
    except (ValueError, OSError, KeyError) as e:
        print(f"lỗi: {e}", file=sys.stderr)
        return 2
    if not paths:
        print("lỗi: không có ảnh vào", file=sys.stderr)
        return 2
    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None or img.shape != contract.FRAME_SHAPE:
            print(f"lỗi: {p} không đọc được hoặc không phải {contract.FRAME_SHAPE}", file=sys.stderr)
            return 2
        frames.append(img)
    report: dict[str, Any] = {"n": len(frames), "config_version": params["version"], "profile_id": params["profile_id"],
                              "source": source}
    if args.bench:
        sampler = _PerfSampler()
        load = _cpu_load_probe()
        res = bench(frames, params, args.warmup)
        res["cpu_load_pct_during"] = load()
        perf = sampler.summary()
        res["cpu_perf_pct"] = perf
        res["throttled"] = None if perf is None else perf["min"] < PERF_OK_PCT
        res["run_notes"] = list(args.note)
        # Cổng pass phase 12: p50 ≤ ngân sách, hoặc "lý do + số đo" — máy hạ xung đo được và có ghi chú lý do.
        res["gate"] = ("pass" if res["pass"] else
                       "over_budget_throttled" if res["throttled"] and args.note else "fail")
        outs = res.pop("outputs")
        report.update({"schema": "classic_bench/1", "generated": datetime.now(UTC).isoformat(timespec="seconds"),
                       "command": "python -m xla.enhance.classic " + " ".join(argv if argv is not None else sys.argv[1:]),
                       "warmup": min(args.warmup, len(frames)), "env": _env(), "step_order": list(STEPS),
                       "notes": list(BENCH_NOTES), **res})
    else:
        outs = [enhance(f, params) for f in frames]
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p, o in zip(paths, outs, strict=True):
            if not cv2.imwrite(str(out_dir / f"{p.stem}.png"), o):
                print(f"lỗi: không ghi được {out_dir / p.stem}.png", file=sys.stderr)
                return 2
        report["out_dir"] = str(out_dir)
    if args.bench_out:
        bp = Path(args.bench_out)
        bp = bp if bp.is_absolute() else repo_root() / bp
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.bench:
        t = report["ms"]["total"]
        print(f"{report['n']} khung: p50 {t['p50']} ms, p95 {t['p95']} ms (ngân sách {BUDGET_MS} ms)")
    if args.bench and report["gate"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
