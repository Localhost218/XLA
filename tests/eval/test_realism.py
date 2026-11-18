"""Phase 06 bước 6: kiểm phép đo độ giống thật trên CPU với dữ liệu và đặc trưng giả (không cần DINOv2, không cần GPU)."""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from xla.eval import c2st as c2st_mod
from xla.eval.kid import extract_features, kid, mmd2_unbiased
from xla.eval.lowlevel import measure_set, stat_distance, wasserstein1
from xla.eval.patches import ReadProtocolError, read_image
from xla.eval.realism import compute_realism, load_patches, main

W, H = 160, 120
CROP = 16


def make_set(directory: Path, n: int, seed: int, noise: float = 0.0, size=(W, H), ext=".png") -> list[Path]:
    """Ảnh giả: nền mịn ngẫu nhiên + hạt nhiễu nhẹ, cộng thêm nhiễu Gauss σ=`noise` (thang 0–255)."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        base = cv2.resize(rng.normal(0, 1, (12, 16, 3)).astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
        img = 128 + 35 * base + rng.normal(0, 3, (size[1], size[0], 3))
        if noise:
            img += rng.normal(0, noise, img.shape)
        p = directory / f"{i:04d}{ext}"
        cv2.imwrite(str(p), np.clip(img, 0, 255).astype(np.uint8))
        paths.append(p)
    return paths


class FakeFeatures:
    """Đặc trưng giả thay DINOv2: trung bình, độ lệch, năng lượng Laplacian theo kênh + ảnh xám thu 4×4."""

    name = "fake_stats_v1"

    def __call__(self, pixels_bgr: np.ndarray) -> np.ndarray:
        out = []
        for p in pixels_bgr.astype(np.float32) / 255.0:
            lap = np.abs(cv2.Laplacian(p, cv2.CV_32F)).mean(axis=(0, 1))
            small = cv2.resize(p.mean(axis=2), (4, 4), interpolation=cv2.INTER_AREA).ravel()
            out.append(np.concatenate([p.mean(axis=(0, 1)), p.std(axis=(0, 1)), 10 * lap, small]))
        return np.asarray(out, dtype=np.float32)


def fake_measure() -> SimpleNamespace:
    """Hàm đo giả cùng chữ ký `xla.camera.measure` (phase 04) — tất định theo ảnh."""

    def measure_image(img, crop_left=16):
        g = cv2.cvtColor(img[:, crop_left:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return {
            "y_mean": float(g.mean()),
            "y_std": float(g.std()),
            "laplacian_var": float(cv2.Laplacian(g, cv2.CV_32F).var()),
            "always_nan": float("nan"),
        }

    def nlf(img, crop_left=16, n_bins=8):
        g = cv2.cvtColor(img[:, crop_left:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return [(float(g.mean()), float((g - cv2.GaussianBlur(g, (0, 0), 2)).std()))]

    def psd_radial(img, crop_left=16, n_bins=16):
        g = cv2.cvtColor(img[:, crop_left:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = g - cv2.GaussianBlur(g, (0, 0), 2)
        return [(0.1, float((r**2).mean()) + 1e-6), (0.3, float(np.abs(r).mean()) + 1e-6)]

    def read_jpeg_header(data):
        return {"qtable_luma": [1] * 64, "qtable_chroma": [2] * 64, "subsampling": "4:2:2", "baseline": True}

    return SimpleNamespace(
        measure_image=measure_image, nlf=nlf, psd_radial=psd_radial, read_jpeg_header=read_jpeg_header
    )


# ---------------------------------------------------------------- nạp patch


def test_load_patches_same_seed_same_patches(tmp_path):
    paths = make_set(tmp_path / "a", 6, seed=1)
    a = load_patches(paths, size=32, n_patches=20, seed=7, crop_left=CROP)
    b = load_patches(paths, size=32, n_patches=20, seed=7, crop_left=CROP)
    c = load_patches(paths, size=32, n_patches=20, seed=8, crop_left=CROP)
    assert np.array_equal(a.pixels, b.pixels) and np.array_equal(a.boxes, b.boxes)
    assert not np.array_equal(a.boxes, c.boxes)


def test_load_patches_crops_from_original_after_left_strip(tmp_path):
    paths = make_set(tmp_path / "a", 3, seed=2)
    pt = load_patches(paths, size=40, n_patches=9, seed=0, crop_left=CROP)
    assert pt.pixels.shape == (9, 40, 40, 3) and pt.pixels.dtype == np.uint8
    assert (pt.boxes[:, 0] >= CROP).all()
    for i in range(9):
        img = read_image(paths[pt.source[i]])
        x, y = pt.boxes[i]
        assert np.array_equal(pt.pixels[i], img[y : y + 40, x : x + 40])


def test_protocol_mismatch_exits_2(tmp_path):
    make_set(tmp_path / "real", 3, seed=1)
    make_set(tmp_path / "synth", 3, seed=2, size=(W + 8, H))
    code = main(
        [
            "--real",
            str(tmp_path / "real"),
            "--synth",
            str(tmp_path / "synth"),
            "--tiers",
            "stats",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert code == 2
    with pytest.raises(ReadProtocolError):
        compute_realism(
            sorted((tmp_path / "real").glob("*.png")),
            sorted((tmp_path / "synth").glob("*.png")),
            tiers=("stats",),
            measure=fake_measure(),
        )


# ---------------------------------------------------------------- thống kê mức thấp


def test_wasserstein1_known_values():
    assert wasserstein1([0.0], [1.0]) == pytest.approx(1.0)
    assert wasserstein1([0, 1, 2], [0, 1, 2]) == 0.0
    assert wasserstein1([0, 1, 2], [5, 6, 7]) == pytest.approx(5.0)


def test_stat_distance_zero_for_identical_sets(tmp_path):
    paths = make_set(tmp_path / "a", 8, seed=3)
    m = measure_set(paths, CROP, fake_measure())
    res = stat_distance(m, measure_set(paths, CROP, fake_measure()))
    assert res["total"] == 0.0
    assert all(v["value"] == 0.0 for k, v in res["items"].items() if k != "always_nan")
    assert res["excluded_nan"] == ["always_nan"]
    assert res["dqt"]["match"] is True


def test_stat_distance_grows_with_noise_and_weights_apply(tmp_path):
    real = measure_set(make_set(tmp_path / "r", 10, seed=4), CROP, fake_measure())
    near = stat_distance(real, measure_set(make_set(tmp_path / "s1", 10, seed=5), CROP, fake_measure()))
    far = stat_distance(real, measure_set(make_set(tmp_path / "s2", 10, seed=5, noise=20), CROP, fake_measure()))
    assert far["total"] > near["total"] > 0
    only_mean = stat_distance(
        real,
        measure_set(make_set(tmp_path / "s2", 10, seed=5, noise=20), CROP, fake_measure()),
        weights={k: 0.0 for k in far["items"]} | {"y_mean": 1.0},
    )
    assert only_mean["total"] == pytest.approx(far["items"]["y_mean"]["value"])
    with pytest.raises(ValueError):
        stat_distance(real, real, weights={"khong_co": 1.0})


def test_dqt_reported_separately(tmp_path):
    jpg = make_set(tmp_path / "j", 3, seed=6, ext=".jpg")
    png = make_set(tmp_path / "p", 3, seed=6)
    res = stat_distance(measure_set(jpg, CROP, fake_measure()), measure_set(png, CROP, fake_measure()))
    assert res["dqt"]["match"] is False
    assert res["dqt"]["synth"] == [{"non_jpeg": True, "n": 3}]
    assert "dqt" not in res["items"]


def test_stats_with_real_measure_module(tmp_path):
    paths = make_set(tmp_path / "a", 4, seed=9, size=(640, 480), ext=".jpg")
    res = compute_realism(paths, paths, tiers=("stats",), crop_left=CROP)
    assert res["stats"]["total"] == 0.0
    assert res["stats"]["dqt"]["match"] is True


# ---------------------------------------------------------------- KID


def test_kid_unbiased_near_zero_same_distribution():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(600, 16)), rng.normal(size=(600, 16))
    r = kid(x, y, n_subsets=10, subset_size=300, seed=1)
    assert abs(r["mean"]) < 3 * r["se"]
    assert r["ci95"][0] <= r["mean"] <= r["ci95"][1]
    assert mmd2_unbiased(x, x + 0.5) > 10 * abs(r["mean"])


def test_kid_same_distribution_images_with_fake_features(tmp_path):
    a = make_set(tmp_path / "a", 30, seed=10)
    b = make_set(tmp_path / "b", 30, seed=11)
    res = compute_realism(
        a, b, tiers=("kid",), feature_fn=FakeFeatures(), n_patches=400, kid_patch=48, kid_subset_size=200
    )
    k = res["kid"]
    assert abs(k["mean"]) < 3 * k["se"], k
    assert k["encoder"] == "fake_stats_v1"
    assert "kid" in res["timing_s"]


def test_kid_increases_monotonically_with_noise(tmp_path):
    real = make_set(tmp_path / "r", 20, seed=12)
    means = []
    for sigma in (0, 4, 12, 30):
        synth = make_set(tmp_path / f"s{sigma}", 20, seed=13, noise=sigma)
        res = compute_realism(
            real, synth, tiers=("kid",), feature_fn=FakeFeatures(), n_patches=300, kid_patch=48, kid_subset_size=200
        )
        means.append(res["kid"]["mean"])
    assert all(b > a for a, b in itertools.pairwise(means)), means


def test_feature_cache_by_sha256(tmp_path):
    paths = make_set(tmp_path / "a", 4, seed=14)
    pt = load_patches(paths, size=32, n_patches=12, seed=0, crop_left=CROP)
    calls = []

    class Counting(FakeFeatures):
        def __call__(self, pixels):
            calls.append(len(pixels))
            return super().__call__(pixels)

    f1, i1 = extract_features(pt, Counting(), cache_dir=tmp_path / "cache")
    f2, i2 = extract_features(pt, Counting(), cache_dir=tmp_path / "cache")
    assert calls == [12] and i1["n_computed"] == 12 and i2["n_cached"] == 12
    assert np.allclose(f1, f2)


# ---------------------------------------------------------------- C2ST


def test_c2st_net_is_small():
    net = c2st_mod.build_net()
    assert sum(p.numel() for p in net.parameters()) <= 100_000


def test_c2st_split_by_source_has_no_leak():
    sources = np.repeat(np.arange(20), 5)
    parts = c2st_mod.split_by_source(sources, np.random.default_rng(0))
    sets = [set(sources[p]) for p in parts]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(p) for p in parts) == len(sources)


def test_c2st_chance_on_same_distribution(tmp_path):
    a = load_patches(make_set(tmp_path / "a", 30, seed=20), 64, 240, seed=0, crop_left=CROP)
    b = load_patches(make_set(tmp_path / "b", 30, seed=21), 64, 240, seed=1, crop_left=CROP)
    r = c2st_mod.c2st(a, b, seed=0, epochs=6, n_boot=200)
    assert abs(r["accuracy"] - 0.5) < 0.15, r


def test_c2st_separates_clearly_different_sets(tmp_path):
    a = load_patches(make_set(tmp_path / "a", 30, seed=22), 64, 240, seed=0, crop_left=CROP)
    b = load_patches(make_set(tmp_path / "b", 30, seed=23, noise=25), 64, 240, seed=1, crop_left=CROP)
    r = c2st_mod.c2st(a, b, seed=0, epochs=6, n_boot=200)
    assert r["accuracy"] > 0.9, r
    assert r["n_params"] <= 100_000
