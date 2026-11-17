"""Hồ sơ camera ``configs/camera_profile.yaml`` (hợp đồng §6), đo **chỉ trên khung dev** của ``data/real/split.json``.

Lệnh::

    python -m xla.camera.profile build --split dev --json    # đo, kiểm DQT đồng nhất, kiểm tái tạo JPEG, ghi hồ sơ
    python -m xla.camera.profile check configs/camera_profile.yaml   # exit 1 khi lệch schema

Hồ sơ mô tả camera, không chứa tham số làm xấu. Chỉ phase 04 ghi file này (D10); đổi số đo = tăng ``profile_version``.
Phép đo nằm ở ``xla.camera.measure`` (SSOT).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xla.camera import measure as M
from xla.camera.frames import FRAMES_REL, PRELIM_REL, REC_FLIGHT
from xla.camera.split import SPLIT_REL
from xla.paths import repo_root

SCHEMA_VERSION = "1.0"
PROFILE_VERSION = "1.1.0"
"""1.0.0: dev 470 khung (split test 300-400); 1.1.0: dev 1 115 khung sau khi Owner chia lại test 200-400 (18/11/2026)."""
PROFILE_ID = "ov3660-esp32s3-q12-20261102"
ESP32_QUALITY = 12
CROP_LEFT_PX = M.CROP_LEFT_DEFAULT
PERSON_CONF_MIN = 0.5
PERSON_SOURCE = "yolo11s-so-bo"
PINK_FRAME_FRAC = 0.005
MOTION_ANISO = 1.5
NLF_BINS = 8
PSD_BINS = 16
NLF_MIN_FRAME_FRAC = 0.10
"""Bin NLF chỉ vào hồ sơ khi ít nhất 10 % khung dev đo được σ ở bin đó."""
REENCODE_N = 20
ANNEX_K_LUMA = (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17,
                22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78,
                87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99)
ANNEX_K_CHROMA = tuple([17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
                        47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32)
SENSOR = {"model": "OV3660", "board": "ESP32-S3 N16R8", "xclk_mhz": 10, "mount": "nadir"}
FRAME = {"width_px": 640, "height_px": 480, "format": "MJPEG"}

PROFILE_REL = Path("configs/camera_profile.yaml")
MEAS_REL = Path("results/camera/profile_measurements.csv")
REPORT_REL = Path("results/camera/profile_report.md")
EVIDENCE_FEATURES_REL = Path("plans/reports/261118-bang-chung/video/features.csv")


def annex_k_scaled(table: tuple[int, ...], scale_pct: int) -> list[int]:
    """Bảng Annex K nhân ``scale_pct`` % theo công thức IJG ``(v·s + 50) // 100`` kẹp 1–255."""
    return [min(255, max(1, (v * scale_pct + 50) // 100)) for v in table]


def find_annex_k_scale(luma: list[int], chroma: list[int]) -> float | None:
    """Hệ số Annex K (0–2) tái tạo đúng cả hai bảng; None nếu không có."""
    for s in range(1, 201):
        if annex_k_scaled(ANNEX_K_LUMA, s) == luma and annex_k_scaled(ANNEX_K_CHROMA, s) == chroma:
            return s / 100
    return None


def dist(values: Any) -> dict[str, float]:
    """``{p5, p50, p95}`` trên giá trị hữu hạn (bỏ NaN)."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("không có giá trị hữu hạn để gom phân bố")
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    return {"p5": _r(p5), "p50": _r(p50), "p95": _r(p95)}


def _r(x: float, sig: int = 5) -> float:
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, max(0, sig - 1 - math.floor(math.log10(abs(x)))))


def load_split(path: Path | None = None) -> dict:
    return json.loads((path or repo_root() / SPLIT_REL).read_text(encoding="utf-8"))


def frames_of(split: str, split_doc: dict) -> list[str]:
    return sorted(d["frame"] for d in split_doc["frames"] if d["split"] == split)


def header_groups(frames_dir: Path, names: list[str]) -> dict[tuple, list[str]]:
    """Nhóm khung theo (bảng luma, bảng chroma, lấy mẫu màu, baseline)."""
    groups: dict[tuple, list[str]] = {}
    for n in names:
        h = M.read_jpeg_header((frames_dir / n).read_bytes())
        key = (tuple(h["qtable_luma"]), tuple(h["qtable_chroma"]), h["subsampling"], h["baseline"])
        groups.setdefault(key, []).append(n)
    return groups


def _sof_key(s: dict) -> tuple:
    return s["marker"], s["precision"], s["width"], s["height"], [(c[0], c[1], c[2]) for c in s["components"]]


def reencode_check(frames_dir: Path, names: list[str], luma: list[int], chroma: list[int]) -> list[dict]:
    """Giải mã rồi nén lại bằng Pillow ``qtables=[luma, chroma]``, ``subsampling="4:2:2"``, không truyền ``quality``.

    Trả mỗi khung ``{"frame", "dqt_match", "sof_match"}``: so bảng DQT theo thành phần và SOF (marker, độ chính xác, cỡ,
    id thành phần, hệ số lấy mẫu) của file ra với file gốc.
    """
    from PIL import Image

    out = []
    for n in names:
        src = (frames_dir / n).read_bytes()
        buf = io.BytesIO()
        with Image.open(io.BytesIO(src)) as im:
            im.convert("RGB").save(buf, format="JPEG", qtables=[list(luma), list(chroma)], subsampling="4:2:2")
        a, b = M.jpeg_segments(src), M.jpeg_segments(buf.getvalue())
        ca, cb = a["sof"]["components"], b["sof"]["components"]
        dqt_ok = len(ca) == len(cb) and all(a["dqt"][x[3]] == b["dqt"][y[3]] for x, y in zip(ca, cb, strict=True))
        out.append({"frame": n, "dqt_match": bool(dqt_ok), "sof_match": _sof_key(a["sof"]) == _sof_key(b["sof"])})
    return out


def pick_even(names: list[str], k: int) -> list[str]:
    idx = np.unique(np.round(np.linspace(0, len(names) - 1, min(k, len(names)))).astype(int))
    return [names[i] for i in idx]


def measure_frames(frames_dir: Path, names: list[str]) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Đo mọi khung: (dòng đo từng khung, mảng NLF [n, bins, 2], mảng PSD [n, bins, 2], Y tham chiếu dải mép)."""
    rows, nlfs, psds, refs = [], [], [], []
    for n in names:
        im = cv2.imdecode(np.frombuffer((frames_dir / n).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"không giải mã được {n}")
        m = M.measure_image(im, CROP_LEFT_PX)
        refs.append(M.left_strip(im, CROP_LEFT_PX)[2])
        rows.append({"frame": n, **m})
        nlfs.append(M.nlf(im, CROP_LEFT_PX, NLF_BINS))
        psds.append(M.psd_radial(im, CROP_LEFT_PX, PSD_BINS))
    return rows, np.array(nlfs, float), np.array(psds, float), np.array(refs, float)


def person_boxes(prelim_csv: Path, names: set[str]) -> dict[str, float]:
    short = []
    with prelim_csv.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["frame"] in names and float(r["conf"]) >= PERSON_CONF_MIN and float(r["x1"]) >= CROP_LEFT_PX:
                short.append(min(float(r["x2"]) - float(r["x1"]), float(r["y2"]) - float(r["y1"])))
    if not short:
        raise RuntimeError("không có hộp người sơ bộ nào trên khung dev (conf ≥ 0,5)")
    d = dist(short)
    return {**d, "min": _r(min(short)), "max": _r(max(short)), "n_boxes": len(short)}


def assemble(rows: list[dict], nlfs: np.ndarray, psds: np.ndarray, refs: np.ndarray, header: dict,
             person: dict, alts: list[float], measured: str) -> dict:
    col = {k: np.array([r[k] for r in rows], float) for k in M.MEASURE_KEYS}
    luma, chroma = header["qtable_luma"], header["qtable_chroma"]
    scale = find_annex_k_scale(luma, chroma)
    if scale is None:
        raise RuntimeError("bảng DQT không phải Annex K nhân hệ số nguyên %")
    vig = dist(col["vignette_corner_center_ratio"])
    # cos⁴: tỉ số = cos⁴θ ở tâm ô góc, tanθ = r/f
    w, h = FRAME["width_px"] - CROP_LEFT_PX, FRAME["height_px"]
    s = max(4, int(min(h, w) * M.VIGNETTE_PATCH_FRAC))
    r_corner = math.hypot(w / 2 - s / 2, h / 2 - s / 2)
    theta = math.acos(min(vig["p50"], 0.999999) ** 0.25)
    mtf = dist(col["mtf50_cyc_per_px"])
    nlf_ok = np.isfinite(nlfs[:, :, 1]).mean(axis=0) >= NLF_MIN_FRAME_FRAC
    nlf_curve = [[_r(np.nanmedian(nlfs[:, i, 0][np.isfinite(nlfs[:, i, 1])])), _r(np.nanmedian(nlfs[:, i, 1]))]
                 for i in range(nlfs.shape[1]) if nlf_ok[i]]
    psd_curve = [[_r(psds[0, i, 0]), _r(np.nanmedian(psds[:, i, 1]))] for i in range(psds.shape[1])]
    sob = col["sobel_x2_over_y2"]
    aniso = np.maximum(sob, 1 / np.maximum(sob, 1e-9))
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_version": PROFILE_VERSION,
        "profile_id": PROFILE_ID,
        "measured_on": {"recording": REC_FLIGHT, "split_file": SPLIT_REL.as_posix(), "n_frames": len(rows),
                        "measured": measured},
        "sensor": dict(SENSOR),
        "frame": dict(FRAME),
        "jpeg": {
            "esp32_quality": ESP32_QUALITY,
            "baseline": header["baseline"],
            "subsampling": header["subsampling"],
            "qtable_luma": list(luma),
            "qtable_chroma": list(chroma),
            "annex_k_scale": scale,
            "other_qualities": {"rule": "annex_k * q / 16", "verified": False},
        },
        "left_strip": {"width_px": dist(col["left_strip_width_px"])["p50"], "crop_left_px": CROP_LEFT_PX,
                       "mean_y_strip": dist(col["left_strip_mean_y"])["p50"], "mean_y_ref": dist(refs)["p50"]},
        "vignette": {"corner_center_ratio": vig, "model": "cos4", "focal_px_fit": _r(r_corner / math.tan(theta))},
        "haze": {"dark_channel_15px_0_255": dist(col["dark_channel_15px"]), "y_p1_0_255": dist(col["y_p1"]),
                 "saturation_0_255": dist(col["saturation_mean"])},
        "sharpness": {"laplacian_var": dist(col["laplacian_var"]), "mtf50_cyc_per_px": mtf,
                      "psf_sigma_px_fit": _r(math.sqrt(math.log(2) / (2 * math.pi ** 2)) / mtf["p50"])},
        "noise": {"sigma_immerkaer_0_255": dist(col["sigma_immerkaer"]), "nlf": nlf_curve, "psd_radial": psd_curve},
        "color": {"r_over_g": dist(col["r_over_g"]), "b_over_g": dist(col["b_over_g"]),
                  "wb_gain_r_fit": dist(col["r_over_g"])["p50"], "wb_gain_b_fit": dist(col["b_over_g"])["p50"]},
        "magenta": {"rule": "R-G>25 and B-G>25", "frac_pixels": dist(col["pink_frac"]),
                    "frac_frames_gt_0p5pct": _r(float((col["pink_frac"] > PINK_FRAME_FRAC).mean())),
                    "highlight_y_0_255": dist(col["pink_highlight_y"])},
        "blocking": {"ratio_8x8": dist(col["blocking_ratio_8x8"])},
        "motion_blur": {"sobel_x2_over_y2": dist(sob), "frac_frames_gt_1p5": _r(float((aniso > MOTION_ANISO).mean()))},
        "person_box_px": {"short_side": {k: person[k] for k in ("p5", "p50", "p95", "min", "max")},
                          "source": PERSON_SOURCE},
        "altitude_m": {"flight_frames": dist(alts), "filter": "rel_alt_m > 3, bỏ điểm vọt"},
    }


class _Dumper:
    """YAML: danh sách số in một dòng (flow), còn lại dạng khối; giữ thứ tự khoá."""

    @staticmethod
    def dump(obj: dict) -> str:
        import yaml

        class D(yaml.SafeDumper):
            pass

        def rep_list(dumper, data):
            flow = all(isinstance(x, (int, float)) or (isinstance(x, list) and all(isinstance(y, (int, float))
                                                                                  for y in x)) for x in data)
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

        def rep_dict(dumper, data):
            flow = len(data) <= 5 and all(isinstance(v, (int, float)) for v in data.values())
            return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=flow)

        D.add_representer(list, rep_list)
        D.add_representer(dict, rep_dict)
        return yaml.dump(obj, Dumper=D, allow_unicode=True, sort_keys=False, width=120)


def validate(profile: dict) -> list[str]:
    """Lỗi hồ sơ (rỗng = đạt): schema ``configs/camera_profile.schema.json`` + thứ tự p5 ≤ p50 ≤ p95, qua
    ``xla.contract.validate_camera_profile`` (SSOT của phase 02)."""
    from xla.contract import validate_camera_profile

    return validate_camera_profile(profile)


def load_profile(path: Path) -> dict:
    from xla.contract import load_profile as _load

    return _load(path)


def _evidence_rows() -> list[dict[str, str]]:
    p = repo_root() / EVIDENCE_FEATURES_REL
    with p.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["flying"] == "True"]


# (khoá hồ sơ để in, khoá measure_image hoặc hàm(dòng đo) -> số, cột features.csv hoặc None, hệ số đổi đơn vị, ghi chú)
_REPORT_ROWS = (
    ("haze.dark_channel_15px_0_255", "dark_channel_15px", "dark_ch", 1.0, ""),
    ("haze.y_p1_0_255", "y_p1", "p1", 1.0, "mẫu cũ tính cả dải tối mép trái → p1 thấp hơn"),
    ("y_p99 (không vào hồ sơ riêng)", "y_p99", "p99", 1.0, ""),
    ("haze.saturation_0_255", "saturation_mean", "sat", 1.0, ""),
    ("magenta.frac_pixels", "pink_frac", "mag_px", 0.01,
     ("mẫu cũ ghi %, đã đổi ra tỉ lệ. Δ do tập đo khác: dev là vài khối 30 s liền (độ cao: `altitude_m` của hồ sơ), "
     "mẫu rải đều 6 chuyến; % hồng tăng theo độ cao (phan-tich-video.md); giá trị nhỏ, đuôi nặng nên p50 dao động mạnh")),
    ("blocking.ratio_8x8", "blocking_ratio_8x8", "block", 1.0, "mẫu cũ có cả dải mép, lưới lệch pha"),
    ("sharpness.laplacian_var", "laplacian_var", "lap", 1.0, ""),
    ("noise.sigma_immerkaer_0_255", "sigma_immerkaer", "noise", 1.0, ""),
    ("motion_blur.sobel_x2_over_y2", "sobel_x2_over_y2", "grad_aniso", 1.0,
     "khác định nghĩa: mẫu cũ đo max/min (≥ 1), hồ sơ đo đúng tên Gx²/Gy² (< 1 khi nhoè dọc); so cùng định nghĩa ở dòng dưới"),
    ("max(Gx²/Gy², Gy²/Gx²) (dùng cho frac_frames_gt_1p5)",
     lambda r: max(r["sobel_x2_over_y2"], 1 / max(r["sobel_x2_over_y2"], 1e-9)), "grad_aniso", 1.0, ""),
    ("color.r_over_g", "r_over_g", "R_G", 1.0, ""),
    ("color.b_over_g", "b_over_g", "B_G", 1.0, ""),
    ("left_strip.mean_y_strip", "left_strip_mean_y", "left_strip", 1.0, "mẫu cũ lấy cố định cột 0–11"),
    ("vignette.corner_center_ratio", "vignette_corner_center_ratio", None, 1.0, "mẫu cũ chỉ có số 0,77 trong văn bản"),
    ("sharpness.mtf50_cyc_per_px", "mtf50_cyc_per_px", None, 1.0, "mới, ước lượng từ cạnh cảnh tự nhiên"),
    ("left_strip.width_px", "left_strip_width_px", None, 1.0, "mới; kỳ vọng ~13 px"),
)


_COMPARE_FIELDS = (
    "measured_on.n_frames", "left_strip.width_px", "left_strip.mean_y_strip", "vignette.corner_center_ratio.p50",
    "vignette.focal_px_fit", "haze.dark_channel_15px_0_255.p50", "haze.y_p1_0_255.p50", "haze.saturation_0_255.p50",
    "sharpness.laplacian_var.p50", "sharpness.mtf50_cyc_per_px.p50", "sharpness.psf_sigma_px_fit",
    "noise.sigma_immerkaer_0_255.p50", "color.r_over_g.p50", "color.b_over_g.p50", "magenta.frac_pixels.p50",
    "magenta.frac_frames_gt_0p5pct", "magenta.highlight_y_0_255.p50", "blocking.ratio_8x8.p50",
    "motion_blur.sobel_x2_over_y2.p50", "motion_blur.frac_frames_gt_1p5", "person_box_px.short_side.p50",
    "altitude_m.flight_frames.p50",
)


def _get(d: dict, dotted: str) -> Any:
    for k in dotted.split("."):
        d = d[k]
    return d


def compare_lines(prev: dict, cur: dict) -> list[str]:
    """Bảng so các đại lượng chính giữa hồ sơ trước và hồ sơ mới (Δ theo %)."""
    a, b = prev["profile_version"], cur["profile_version"]
    out = [f"## So v{a} → v{b}", "", f"| Trường | v{a} | v{b} | Δ |", "|---|---|---|---|"]
    for f in _COMPARE_FIELDS:
        x, y = float(_get(prev, f)), float(_get(cur, f))
        delta = f"{(y - x) / x * 100:+.0f} %" if x else "–"
        out.append(f"| `{f}` | {_r(x)} | {_r(y)} | {delta} |")
    return [*out, ""]


def write_report(path: Path, rows: list[dict], profile: dict, reenc: list[dict], groups: int, n_test_overlap: int,
                 person: dict, prev: dict | None = None, prev_path: str = "") -> None:
    ev = _evidence_rows()
    lines = [
        "# Hồ sơ camera v1 — báo cáo đo (phase 04)",
        "",
        (f"Hồ sơ: `{PROFILE_REL.as_posix()}` `profile_version` {profile['profile_version']}, `profile_id` "
        f"`{profile['profile_id']}`. Đo trên **{len(rows)} khung dev** của `{SPLIT_REL.as_posix()}` (D10; 0 khung test "
        f"— đếm giao với test: {n_test_overlap}). Số đo từng khung: `{MEAS_REL.as_posix()}`. Lệnh: "
        "`python -m xla.camera.profile build --split dev --json`."),
        "",
        ("Cột \"mẫu 18/11\" tính lại từ `plans/reports/261118-bang-chung/video/features.csv` (300 khung bay rải đều, "
        "gồm cả khung nay thuộc test, đo trên khung **nguyên** không cắt dải mép 16 px). Chênh lệch p50 > 20 % có giải "
        "thích ở cột ghi chú."),
        "",
        "| Đại lượng (trường hồ sơ) | dev p5 | dev p50 | dev p95 | mẫu 18/11 p5 | p50 | p95 | Δp50 | Ghi chú |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, key, evcol, k, note in _REPORT_ROWS:
        d = dist([key(r) if callable(key) else r[key] for r in rows])
        if evcol:
            e = dist([float(r[evcol]) * k for r in ev if r[evcol] not in ("", "nan")])
            delta = (d["p50"] - e["p50"]) / e["p50"] * 100 if e["p50"] else float("nan")
            ecells = f"{e['p5']} | {e['p50']} | {e['p95']} | {delta:+.0f} %"
            flag = " **>20 %**" if abs(delta) > 20 else ""
        elif key == "vignette_corner_center_ratio":
            ecells, flag = "– | 0.77 | – | " + f"{(d['p50'] - 0.77) / 0.77 * 100:+.0f} %", ""
            flag = " **>20 %**" if abs(d["p50"] - 0.77) / 0.77 > 0.2 else ""
        else:
            ecells, flag = "– | – | – | –", ""
        lines.append(f"| `{label}` | {d['p5']} | {d['p50']} | {d['p95']} | {ecells}{flag} | {note} |")
    ok = sum(r["dqt_match"] and r["sof_match"] for r in reenc)
    j = profile["jpeg"]
    lines += [
        "",
        *([f"Hồ sơ trước để so: `{prev_path}`.", "", *compare_lines(prev, profile)] if prev else []),
        "## JPEG",
        "",
        (f"- Mọi khung dev cùng **{groups}** bộ (DQT luma, DQT chroma, lấy mẫu, baseline): `{j['subsampling']}`, baseline "
        f"{j['baseline']}, Annex K × {j['annex_k_scale']} (ESP32 quality {j['esp32_quality']})."),
        (f"- Tái tạo bằng Pillow `qtables=[luma, chroma]`, `subsampling=\"4:2:2\"`, không truyền `quality`: "
        f"**{ok}/{len(reenc)}** khung dev khớp cả DQT lẫn SOF."),
        "- Bảng cho auto-quality 19–40: giả thuyết `annex_k * q / 16`, `verified: false` (Deferred 1).",
        "",
        "## Số suy ra",
        "",
        (f"- Dải mép: rộng p50 {profile['left_strip']['width_px']} px (định nghĩa: số cột đầu có Y dưới điểm giữa mức dải "
        f"và mức nền; dải là dốc mềm ~8 px chứ không phải cạnh sắc), giữ `crop_left_px` {CROP_LEFT_PX}."),
        f"- Tối góc cos⁴: `focal_px_fit` {profile['vignette']['focal_px_fit']} px suy từ p50 tỉ số góc/giữa (ô cạnh 1/6).",
        (f"- PSF Gauss từ MTF50 p50: σ = {profile['sharpness']['psf_sigma_px_fit']} px (MTF50 là **ước lượng từ cảnh "
        "tự nhiên**, nhiễu; phase 07 hiệu chỉnh cùng Laplacian + PSD)."),
        "- `motion_blur.frac_frames_gt_1p5` dùng max(Gx²/Gy², Gy²/Gx²) > 1,5 (bất đẳng hướng, như mẫu 18/11).",
        (f"- `person_box_px`: {person['n_boxes']} hộp YOLO11s @1280 conf ≥ {PERSON_CONF_MIN} trên khung dev, ngoài dải "
        f"mép; nguồn `{PERSON_SOURCE}` (chưa phải nhãn, còn hộp sai do đốm hồng)."),
        (f"- NLF: {len(profile['noise']['nlf'])}/{NLF_BINS} bin Y đủ điểm phẳng (≥ {NLF_MIN_FRAME_FRAC:.0%} khung) — ảnh "
        "sáng, mức đen nâng, nên bin tối không có dữ liệu."),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def cmd_build(args: argparse.Namespace) -> int:
    root = repo_root()
    split_doc = load_split()
    names = frames_of(args.split, split_doc)
    if args.split != "dev":
        print("CẢNH BÁO: hồ sơ v1 chỉ đo trên dev (D10)", file=sys.stderr)
    test_names = set(frames_of("test", split_doc))
    frames_dir = root / FRAMES_REL
    groups = header_groups(frames_dir, names)
    if len(groups) != 1:
        biggest = max(groups.values(), key=len)
        odd = sorted(n for g in groups.values() if g is not biggest for n in g)
        print(json.dumps({"error": "dqt_not_uniform", "groups": len(groups), "odd_frames": odd[:50]},
                         ensure_ascii=False))
        return 1
    (luma, chroma, sub, base), = groups.keys()
    header = {"qtable_luma": list(luma), "qtable_chroma": list(chroma), "subsampling": sub, "baseline": base}
    reenc = reencode_check(frames_dir, pick_even(names, REENCODE_N), list(luma), list(chroma))
    rows, nlfs, psds, refs = measure_frames(frames_dir, names)
    person = person_boxes(root / PRELIM_REL, set(names))
    alts = [d["alt_m"] for d in split_doc["frames"] if d["frame"] in set(names)]
    profile = assemble(rows, nlfs, psds, refs, header, person, alts, args.measured or date.today().isoformat())
    errors = validate(profile)
    meas = root / MEAS_REL
    meas.parent.mkdir(parents=True, exist_ok=True)
    with meas.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["frame", *M.MEASURE_KEYS, "left_strip_ref_y"], lineterminator="\n")
        w.writeheader()
        for r, ref in zip(rows, refs, strict=True):
            w.writerow({k: (_r(v) if isinstance(v, float) else v) for k, v in {**r, "left_strip_ref_y": ref}.items()})
    (root / PROFILE_REL).write_text(
        "# Hồ sơ camera — sinh bởi `python -m xla.camera.profile build --split dev`; KHÔNG sửa tay.\n"
        "# Đổi số đo = tăng profile_version (hợp đồng docs/hop-dong-model.md §6). Báo cáo: "
        f"{REPORT_REL.as_posix()}\n" + _Dumper.dump(profile), encoding="utf-8", newline="\n")
    overlap = len(test_names & set(names))
    prev = load_profile(Path(args.previous)) if args.previous else None
    write_report(root / REPORT_REL, rows, profile, reenc, len(groups), overlap, person, prev,
                 Path(args.previous).as_posix() if args.previous else "")
    reenc_ok = all(r["dqt_match"] and r["sof_match"] for r in reenc)
    summary = {"n_frames": len(rows), "dqt_groups": len(groups), "reencode_match": f"{sum(r['dqt_match'] and r['sof_match'] for r in reenc)}/{len(reenc)}",
               "schema_errors": errors, "test_overlap": overlap,
               "left_strip_width_px": profile["left_strip"]["width_px"],
               "subsampling": sub, "annex_k_scale": profile["jpeg"]["annex_k_scale"],
               "person_boxes": person["n_boxes"]}
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    return 0 if not errors and reenc_ok and overlap == 0 else 1


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.profile)
    errors = validate(load_profile(path))
    if args.json:
        print(json.dumps({"profile": str(path), "errors": errors}, ensure_ascii=False))
    else:
        print("\n".join(errors) or "đạt schema")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.camera.profile", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="đo khung dev, ghi configs/camera_profile.yaml + báo cáo")
    b.add_argument("--split", default="dev", choices=["dev"])
    b.add_argument("--previous", help="hồ sơ phiên bản trước (yaml) để báo cáo in bảng so sánh")
    b.add_argument("--measured", help="ngày đo ghi vào hồ sơ (YYYY-MM-DD), mặc định hôm nay")
    b.add_argument("--json", action="store_true")
    c = sub.add_parser("check", help="kiểm hồ sơ theo configs/camera_profile.schema.json")
    c.add_argument("profile")
    c.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
