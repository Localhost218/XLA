"""Phase 04: phép đo camera (ca tổng hợp biết trước đáp số) và hồ sơ ``configs/camera_profile.yaml``."""

from __future__ import annotations

import csv
import io
import json
import math

import cv2
import numpy as np
import pytest
from PIL import Image

from xla.camera import measure as M
from xla.camera import profile as P
from xla.camera.frames import FRAMES_REL
from xla.paths import repo_root


def _gray(v: int = 128, h: int = 480, w: int = 640) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


# ---------------------------------------------------------------- ca tổng hợp


def test_measure_keys_exact():
    m = M.measure_image(_gray())
    assert tuple(m) == M.MEASURE_KEYS


def test_uniform_gray():
    m = M.measure_image(_gray())
    assert m["pink_frac"] == 0 and math.isnan(m["pink_highlight_y"])
    assert m["dark_channel_15px"] == pytest.approx(128, abs=0.5)
    assert m["saturation_mean"] == 0
    assert m["vignette_corner_center_ratio"] == pytest.approx(1.0)
    assert m["r_over_g"] == pytest.approx(1.0) and m["b_over_g"] == pytest.approx(1.0)
    assert m["sigma_immerkaer"] == pytest.approx(0, abs=1e-6)
    assert m["left_strip_width_px"] == 0 and math.isnan(m["left_strip_mean_y"])
    assert math.isnan(m["mtf50_cyc_per_px"])


def test_left_strip_13px():
    im = _gray(140)
    im[:, :13] = 20
    w, strip_y, ref = M.left_strip(im)
    assert abs(w - 13) <= 1 and strip_y == pytest.approx(20, abs=1) and ref == pytest.approx(140, abs=1)
    assert M.measure_image(im)["left_strip_width_px"] == pytest.approx(13, abs=1)
    # dải nằm ngoài vùng đo: các phép đo khác như ảnh không có dải
    assert M.measure_image(im)["y_p1"] == pytest.approx(140, abs=1)


def test_pink_fraction_and_highlight():
    im = _gray(100)
    im[100:200, 100:200] = (230, 180, 240)  # B, G, R: R−G=60, B−G=50
    m = M.measure_image(im)
    assert m["pink_frac"] == pytest.approx(100 * 100 / (480 * 624), rel=1e-6)
    y = 0.299 * 240 + 0.587 * 180 + 0.114 * 230
    assert m["pink_highlight_y"] == pytest.approx(y, abs=1)


def test_immerkaer_on_gaussian_noise():
    rng = np.random.default_rng(0)
    im = np.clip(112 + rng.normal(0, 5, (480, 640)), 0, 255).astype(np.uint8)  # giữa bin NLF 96–128
    m = M.measure_image(np.dstack([im] * 3))
    assert m["sigma_immerkaer"] == pytest.approx(5, rel=0.1)
    assert m["blocking_ratio_8x8"] == pytest.approx(1.0, abs=0.05)
    nlf = M.nlf(np.dstack([im] * 3))
    sig = [s for lvl, s in nlf if not math.isnan(s)]
    assert len(sig) == 1 and sig[0] == pytest.approx(5, rel=0.15)


def test_blocking_detects_8x8_grid():
    rng = np.random.default_rng(1)
    blocks = rng.integers(60, 200, (60, 80)).astype(np.uint8)
    im = cv2.resize(blocks, (640, 480), interpolation=cv2.INTER_NEAREST)
    assert M.measure_image(np.dstack([im] * 3))["blocking_ratio_8x8"] > 5


def test_mtf50_gaussian_edge():
    sigma = 1.2
    # sọc dọc 80 px (nhiều cạnh), làm mờ PSF Gauss σ → MTF50 = sqrt(ln2 / 2π²) / σ
    rows = np.tile(np.where((np.arange(640) // 80) % 2 == 0, 60.0, 200.0), (480, 1)).astype(np.float32)
    im = cv2.GaussianBlur(rows, (0, 0), sigma)
    im8 = np.dstack([np.clip(im, 0, 255).astype(np.uint8)] * 3)
    expect = math.sqrt(math.log(2) / (2 * math.pi ** 2)) / sigma
    assert M.measure_image(im8)["mtf50_cyc_per_px"] == pytest.approx(expect, rel=0.15)


def test_sobel_ratio_direction():
    rng = np.random.default_rng(2)
    noise = rng.normal(128, 30, (480, 640)).astype(np.float32)
    blur_y = cv2.blur(noise, (1, 15))  # nhoè dọc → gradient dọc yếu → Gx²/Gy² > 1
    im = np.dstack([np.clip(blur_y, 0, 255).astype(np.uint8)] * 3)
    assert M.measure_image(im)["sobel_x2_over_y2"] > 2


def test_psd_radial_shape():
    rng = np.random.default_rng(3)
    im = np.clip(128 + rng.normal(0, 10, (480, 640)), 0, 255).astype(np.uint8)
    psd = M.psd_radial(np.dstack([im] * 3), n_bins=16)
    assert len(psd) == 16 and psd[0][0] == pytest.approx(1 / 64) and psd[-1][0] == pytest.approx(0.5 - 1 / 64)
    p = [v for _, v in psd]
    assert p[-1] > p[0]  # phần dư cao tần: năng lượng dồn về tần số cao


def test_read_jpeg_header_roundtrip_pillow():
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 255, (48, 64, 3), np.uint8)
    for sub, name in ((0, "4:4:4"), (1, "4:2:2"), (2, "4:2:0")):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=62, subsampling=sub)
        h = M.read_jpeg_header(buf.getvalue())
        q = Image.open(io.BytesIO(buf.getvalue())).quantization
        assert h["subsampling"] == name and h["baseline"] is True
        assert h["qtable_luma"] == list(q[0]) and h["qtable_chroma"] == list(q[1])


def test_annex_k_scale_75():
    assert P.find_annex_k_scale(P.annex_k_scaled(P.ANNEX_K_LUMA, 75), P.annex_k_scaled(P.ANNEX_K_CHROMA, 75)) == 0.75


# ---------------------------------------------------------------- hồ sơ thật


@pytest.fixture(scope="module")
def profile():
    return P.load_profile(repo_root() / P.PROFILE_REL)


@pytest.fixture(scope="module")
def split_doc():
    return P.load_split()


def test_profile_passes_schema(profile):
    assert P.validate(profile) == []
    assert P.main(["check", str(repo_root() / P.PROFILE_REL)]) == 0


def test_profile_rejects_missing_qtable(profile):
    bad = json.loads(json.dumps(profile))
    del bad["jpeg"]["qtable_luma"]
    assert P.validate(bad)


def test_profile_counts_match_split(profile, split_doc):
    dev = P.frames_of("dev", split_doc)
    assert profile["measured_on"]["n_frames"] == len(dev)
    assert profile["profile_version"] == P.PROFILE_VERSION and profile["profile_id"] == P.PROFILE_ID
    assert profile["left_strip"]["crop_left_px"] == 16
    assert 11 <= profile["left_strip"]["width_px"] <= 16


def test_measured_frames_are_dev_only(split_doc):
    with (repo_root() / P.MEAS_REL).open(encoding="utf-8", newline="") as f:
        measured = {r["frame"] for r in csv.DictReader(f)}
    assert measured == set(P.frames_of("dev", split_doc))
    assert not measured & set(P.frames_of("test", split_doc))


@pytest.mark.data
def test_profile_dqt_equals_dev_frame(profile, split_doc):
    dev = P.frames_of("dev", split_doc)
    h = M.read_jpeg_header((repo_root() / FRAMES_REL / dev[len(dev) // 2]).read_bytes())
    assert h["qtable_luma"] == profile["jpeg"]["qtable_luma"]
    assert h["qtable_chroma"] == profile["jpeg"]["qtable_chroma"]
    assert h["subsampling"] == profile["jpeg"]["subsampling"] == "4:2:2"


@pytest.mark.data
def test_reencode_matches_all(profile, split_doc):
    names = P.pick_even(P.frames_of("dev", split_doc), P.REENCODE_N)
    res = P.reencode_check(repo_root() / FRAMES_REL, names, profile["jpeg"]["qtable_luma"],
                           profile["jpeg"]["qtable_chroma"])
    assert len(res) == P.REENCODE_N
    assert all(r["dqt_match"] and r["sof_match"] for r in res), [r for r in res if not r["dqt_match"]]
