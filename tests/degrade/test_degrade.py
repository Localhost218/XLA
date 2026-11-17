"""Phase 05 Accept: bộ máy làm xấu D0/D1/D2 (CPU, ảnh tổng hợp, hồ sơ camera tổng hợp đúng contract §6).

Không phụ thuộc dữ liệu thật: ảnh nguồn và hồ sơ đều sinh trong test.
"""

from __future__ import annotations

import copy
import inspect
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, JpegImagePlugin

from xla import contract
from xla.degrade import build, d1_random, d2_camera, geometry, sources

REPO = Path(__file__).resolve().parents[2]


def _ijg(tbl: list[int], scale_pct: float) -> list[int]:
    return [int(min(255, max(1, math.floor((v * scale_pct + 50) / 100)))) for v in tbl]


def _d(p5: float, p50: float, p95: float) -> dict:
    return {"p5": p5, "p50": p50, "p95": p95}


SYN_PROFILE: dict = {
    "schema_version": "1.0",
    "profile_version": "0.0.1",
    "profile_id": "synthetic-test",
    "measured_on": {"recording": "synthetic", "split_file": "data/real/split.json", "n_frames": 1,
                    "measured": "2026-11-18"},
    "sensor": {"model": "OV3660", "board": "ESP32-S3 N16R8", "xclk_mhz": 10, "mount": "nadir"},
    "frame": {"width_px": 640, "height_px": 480, "format": "MJPEG"},
    "jpeg": {"esp32_quality": 12, "baseline": True, "subsampling": "4:2:2",
             "qtable_luma": _ijg(d2_camera.ANNEX_K_LUMA, 75), "qtable_chroma": _ijg(d2_camera.ANNEX_K_CHROMA, 75),
             "annex_k_scale": 0.75, "other_qualities": {"rule": "annex_k * q / 16", "verified": False}},
    "left_strip": {"width_px": 13, "crop_left_px": 16, "mean_y_strip": 64.0, "mean_y_ref": 120.0},
    "vignette": {"corner_center_ratio": _d(0.72, 0.77, 0.82), "model": "cos4", "focal_px_fit": 520.0},
    "haze": {"dark_channel_15px_0_255": _d(120, 142, 160), "y_p1_0_255": _d(60, 80, 100),
             "saturation_0_255": _d(10, 16, 25)},
    "sharpness": {"laplacian_var": _d(15, 33, 80), "mtf50_cyc_per_px": _d(0.12, 0.17, 0.22), "psf_sigma_px_fit": 1.2},
    "noise": {"sigma_immerkaer_0_255": _d(0.3, 0.36, 0.6), "nlf": [[32, 0.8], [128, 1.2], [224, 0.9]],
              "psd_radial": [[0.02, 1.0], [0.1, 0.6], [0.3, 0.2], [0.5, 0.05]]},
    "color": {"r_over_g": _d(0.9, 0.97, 1.05), "b_over_g": _d(0.9, 1.0, 1.1), "wb_gain_r_fit": 1.6,
              "wb_gain_b_fit": 1.5},
    "magenta": {"rule": "R-G>25 and B-G>25", "frac_pixels": _d(0.0, 0.002, 0.03), "frac_frames_gt_0p5pct": 0.31,
                "highlight_y_0_255": _d(205, 222, 240)},
    "blocking": {"ratio_8x8": _d(1.8, 2.6, 3.4)},
    "motion_blur": {"sobel_x2_over_y2": _d(0.8, 1.1, 1.8), "frac_frames_gt_1p5": 0.19},
    "person_box_px": {"short_side": {"p5": 21, "p50": 50, "p95": 139, "min": 15, "max": 160}, "source": "tổng hợp"},
    "altitude_m": {"flight_frames": _d(8, 14, 19), "filter": "rel_alt_m > 3, bỏ điểm vọt"},
}


@pytest.fixture(scope="module")
def cfgs() -> dict:
    return {
        "d1": yaml.safe_load((REPO / "configs" / "degrade_d1.yaml").read_text(encoding="utf-8")),
        "d2": yaml.safe_load((REPO / "configs" / "degrade_d2.yaml").read_text(encoding="utf-8")),
        "src": yaml.safe_load((REPO / "configs" / "sources.yaml").read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def profile_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("prof") / "camera_profile.yaml"
    p.write_text(yaml.safe_dump(SYN_PROFILE, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def source_item(tmp_path_factory) -> dict:
    """Ảnh nguồn tổng hợp 1300×900: nền có vân + một "người" 40×90 px + một vùng sáng trắng."""
    rng = np.random.default_rng(7)
    h, w = 900, 1300
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 40 * np.sin(xx / 37.0), 110 + 30 * np.cos(yy / 23.0), 70 + 20 * np.sin((xx + yy) / 51.0)], 2)
    img = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    img[400:490, 600:640] = (40, 40, 160)
    img[100:200, 100:300] = 255
    p = tmp_path_factory.mktemp("src") / "syn.png"
    cv2.imwrite(str(p), img)
    return {"key": "syn/one", "source": "syn", "image": str(p), "split": "train", "wh": [w, h],
            "boxes": [[600.0, 400.0, 640.0, 490.0]], "ignore": [], "alt_m": None}


@pytest.fixture(scope="module")
def ctx(cfgs, profile_path) -> dict:
    return build.make_context(["d0", "d1", "d2"], profile_path, cfgs["src"],
                              REPO / "configs" / "degrade_d1.yaml", REPO / "configs" / "degrade_d2.yaml")


def test_synthetic_profile_matches_contract_schema() -> None:
    assert contract.validate_camera_profile(copy.deepcopy(SYN_PROFILE)) == []


@pytest.mark.parametrize("set_name", ["d0", "d1", "d2"])
def test_same_seed_same_bytes_other_seed_differs(set_name, source_item, ctx) -> None:
    a = build.render_unit(set_name, source_item, "syn/one#0", 20261118, ctx)
    b = build.render_unit(set_name, source_item, "syn/one#0", 20261118, ctx)
    c = build.render_unit(set_name, source_item, "syn/one#0", 20261119, ctx)
    assert "skipped" not in a
    assert a["data"] == b["data"] and a["label"] == b["label"]
    assert a["data"] != c["data"]
    assert a["record"]["sha256"] == b["record"]["sha256"]


@pytest.mark.parametrize("set_name", ["d0", "d1", "d2"])
def test_output_is_640x480_uint8(set_name, source_item, ctx) -> None:
    r = build.render_unit(set_name, source_item, "syn/one#1", 20261118, ctx)
    img = cv2.imdecode(np.frombuffer(r["data"], np.uint8), cv2.IMREAD_COLOR)
    assert img.dtype == np.uint8 and img.shape == (480, 640, 3)


def test_same_window_across_sets(source_item, ctx) -> None:
    recs = [build.render_unit(s, source_item, "syn/one#0", 5, ctx)["record"] for s in ("d0", "d1", "d2")]
    assert recs[0]["window"] == recs[1]["window"] == recs[2]["window"]
    assert recs[0]["boxes_xyxy_640"] == recs[1]["boxes_xyxy_640"] == recs[2]["boxes_xyxy_640"]


def test_box_transform_hand_computed() -> None:
    """Ảnh 1248×960, hộp người 100×200 (cạnh ngắn 100), cỡ đích cố định 50 px → s = 0,5 → ảnh 624×480 = đúng cửa sổ,
    x0 = y0 = 0; hộp → [200,150,250,250] trong cửa sổ → [216,150,266,250] trong khung 640 (dải mép 16 cột)."""
    fixed = {"p5": 50, "p50": 50, "p95": 50, "min": 50, "max": 50}
    cfg = geometry.GeomConfig(neg_window_frac=0.0)
    win = geometry.plan_window((1248, 960), np.array([[400, 300, 500, 500]], np.float32), np.zeros((0, 4)),
                               fixed, np.random.default_rng(0), cfg)
    assert win is not None and win.kind == "person"
    assert (win.scale_x, win.scale_y, win.x0, win.y0) == (0.5, 0.5, 0, 0)
    np.testing.assert_allclose(win.boxes, [[200, 150, 250, 250]])
    fb = geometry.to_frame_boxes(win.boxes)
    np.testing.assert_allclose(fb, [[216, 150, 266, 250]])
    assert geometry.yolo_lines(fb) == f"0 {241 / 640:.6f} {200 / 480:.6f} {50 / 640:.6f} {100 / 480:.6f}\n"


def test_clip_keeps_only_boxes_with_60pct_area() -> None:
    cfg = geometry.GeomConfig()
    boxes = np.array([[-30, 10, 70, 110],      # còn 70 % → giữ, cắt về [0,10,70,110]
                      [-60, 10, 40, 110],      # còn 40 % → bỏ
                      [600, 400, 620, 440]], np.float32)  # trọn trong cửa sổ
    kept, dropped = geometry.clip_boxes(boxes, 0, 0, cfg)
    np.testing.assert_allclose(kept, [[0, 10, 70, 110], [600, 400, 620, 440]])
    assert dropped == 1


def test_window_avoids_ignored_regions() -> None:
    fixed = {"p5": 40, "p50": 40, "p95": 40, "min": 40, "max": 40}
    boxes = np.array([[100, 100, 140, 180]], np.float32)
    ignore = np.array([[1200, 0, 2000, 1500]], np.float32)
    for s in range(20):
        win = geometry.plan_window((2000, 1500), boxes, ignore, fixed, np.random.default_rng(s), geometry.GeomConfig())
        assert win is not None
        assert win.x0 + geometry.WIN_W <= 1200 * win.scale_x + 1e-6


def test_d2_jpeg_dqt_and_sof_match_profile(source_item, ctx) -> None:
    frame = np.full((480, 640, 3), 128, np.uint8)
    data, verified = d2_camera.encode_final(frame, 12, SYN_PROFILE)
    assert verified
    im = Image.open(io.BytesIO(data))
    assert list(im.quantization[0]) == SYN_PROFILE["jpeg"]["qtable_luma"]
    assert list(im.quantization[1]) == SYN_PROFILE["jpeg"]["qtable_chroma"]
    assert JpegImagePlugin.get_sampling(im) == 1          # Pillow: 1 = 4:2:2
    sof = data.index(b"\xff\xc0")                         # baseline SOF0
    ncomp = data[sof + 9]
    comps = [data[sof + 10 + 3 * i: sof + 13 + 3 * i] for i in range(ncomp)]
    assert [(c[0], c[1]) for c in comps] == [(1, 0x21), (2, 0x11), (3, 0x11)]   # Y 2×1, Cb/Cr 1×1
    # Ảnh D2 thật từ render_unit cũng mang đúng bảng của mức q đã chọn
    r = build.render_unit("d2", source_item, "syn/one#0", 20261118, ctx)
    q = r["record"]["params"]["jpeg"]["q"]
    luma, chroma, _ = d2_camera.qtables_for(q, SYN_PROFILE)
    im2 = Image.open(io.BytesIO(r["data"]))
    assert list(im2.quantization[0]) == luma and list(im2.quantization[1]) == chroma
    assert JpegImagePlugin.get_sampling(im2) == 1


def test_hypothesis_q_tables_are_annex_k_times_q_over_16() -> None:
    luma, chroma, verified = d2_camera.qtables_for(32, SYN_PROFILE)
    assert not verified
    assert luma == _ijg(d2_camera.ANNEX_K_LUMA, 200) and chroma == _ijg(d2_camera.ANNEX_K_CHROMA, 200)
    # q12 theo luật cũng ra đúng bảng đo (Annex K × 75 %)
    s = 100 * 12 / 16
    assert _ijg(d2_camera.ANNEX_K_LUMA, s) == SYN_PROFILE["jpeg"]["qtable_luma"]


def test_tone_stage_turns_white_pink_not_gray(cfgs) -> None:
    img = np.zeros((40, 80, 3), np.float32)
    img[:, :40] = 1.0                    # trắng
    img[:, 40:] = 128 / 255.0            # xám trung bình
    lin = d2_camera.srgb_to_linear(img)
    cfg = dict(cfgs["d2"]["tone"], brightness=[1.0, 1.0])
    out = d2_camera.stage_tone(lin, np.random.default_rng(0), cfg, SYN_PROFILE, {})
    bgr = np.round(d2_camera.linear_to_srgb(out) * 255).astype(int)
    b, g, r = bgr[:, :40, 0].mean(), bgr[:, :40, 1].mean(), bgr[:, :40, 2].mean()
    assert r - g > 25 and b - g > 25, (b, g, r)
    b2, g2, r2 = bgr[:, 40:, 0].mean(), bgr[:, 40:, 1].mean(), bgr[:, 40:, 2].mean()
    assert not (r2 - g2 > 25 and b2 - g2 > 25)
    assert max(abs(r2 - g2), abs(b2 - g2)) <= 2, (b2, g2, r2)


def test_d1_does_not_read_camera_profile(cfgs, source_item, ctx, monkeypatch) -> None:
    import ast

    tree = ast.parse(inspect.getsource(d1_random))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            used |= {a.name for a in node.names} | {getattr(node, "module", "") or ""}
    for token in ("d2_camera", "xla.degrade.d2_camera", "load_profile", "get_field", "profile"):
        assert token not in used, token
    d1_ctx = {k: v for k, v in ctx.items() if k not in ("d2", "profile")}   # make_context(["d1"]) cũng không có

    def boom(*_a, **_k):
        raise AssertionError("D1 đã đọc hồ sơ camera")

    monkeypatch.setattr(d2_camera, "load_profile", boom)
    monkeypatch.setattr(d2_camera, "get_field", boom)
    r = build.render_unit("d1", source_item, "syn/one#0", 20261118, d1_ctx)
    assert "skipped" not in r


def test_d1_signature_has_no_profile() -> None:
    assert list(inspect.signature(d1_random.degrade).parameters) == ["win_bgr", "rng", "cfg"]


def test_visdrone_annotation_parsing() -> None:
    text = "10,20,5,15,1,1,0,0\n30,40,8,9,1,2,0,1\n50,60,100,80,0,0,0,0\n1,1,4,4,1,4,0,0\n"
    boxes, ignore = sources.parse_visdrone_annotation(text)
    assert boxes == [[10, 20, 15, 35], [30, 40, 38, 49]]
    assert ignore == [[50, 60, 150, 140]]


def test_sample_dist_respects_bounds() -> None:
    dist = {"p5": 21, "p50": 50, "p95": 139, "min": 15, "max": 160}
    rng = np.random.default_rng(0)
    v = np.array([geometry.sample_dist(dist, rng) for _ in range(4000)])
    assert v.min() >= 15 and v.max() <= 160
    assert abs(np.median(v) - 50) < 4
    assert abs(np.quantile(v, 0.05) - 21) < 3 and abs(np.quantile(v, 0.95) - 139) < 8


def test_build_set_writes_manifest_and_dataset_yaml(source_item, ctx, tmp_path) -> None:
    units = [(source_item, f"syn/one#{i}") for i in range(3)]
    summary = build.build_set("d2", units, 20261118, ctx, tmp_path, workers=1)
    lines = (tmp_path / "d2" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["written"] == len(lines) == 3
    rec = json.loads(lines[0])
    assert {"source_image", "seed", "params", "sha256", "window"} <= set(rec)
    img = tmp_path / "d2" / rec["split"] / "images" / rec["name"]
    import hashlib

    assert hashlib.sha256(img.read_bytes()).hexdigest() == rec["sha256"]
    ds = yaml.safe_load((tmp_path / "d2" / "dataset.yaml").read_text(encoding="utf-8"))
    assert ds["names"] == {0: "person"} and ds["train"] == "train/images"
