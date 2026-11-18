"""Phase 12: đối chứng cổ điển ``xla.enhance.classic`` — ảnh tổng hợp biết trước đáp số (CPU), và bench (marker data).

Điểm hồng theo luật hồ sơ (``configs/camera_profile.yaml`` ``magenta.rule`` = ``xla.camera.measure.PINK_DIFF``), dark
channel theo cách đo của hồ sơ (``measure.DARK_CHANNEL_PATCH``), "bị đẩy về đen" = kênh nhỏ nhất ≤ 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from skimage.color import deltaE_ciede2000, rgb2lab

from xla import contract
from xla.camera import measure as M
from xla.enhance import classic as C
from xla.paths import repo_root

H, W = contract.FRAME_H, contract.FRAME_W


@pytest.fixture(scope="module")
def params() -> dict:
    return C.load_params()


def _field(rng: np.random.Generator, lo: float, hi: float, cells: tuple[int, int] = (12, 16), detail: float = 6.0,
           h: int = H, w: int = W) -> np.ndarray:
    """Trường màu mịn (ô ``cells`` nội suy bicubic) + vân nhỏ, float32 [h, w, 3] trong [lo, hi]."""
    base = cv2.resize(rng.random((*cells, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    base = lo + (hi - lo) * np.clip(base, 0.0, 1.0)
    fine = cv2.GaussianBlur(rng.normal(0.0, detail, (h, w, 3)).astype(np.float32), (0, 0), 1.0)
    return np.clip(base + fine, lo, hi)


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _hazy(seed: int = 1, t: float = 0.5, airlight: float = 220.0) -> np.ndarray:
    """Ảnh màn mờ tổng hợp ``I = J·t + A·(1 − t)``, J có dark channel thấp (màu từng kênh độc lập 10–230)."""
    j = _field(np.random.default_rng(seed), 10.0, 230.0)
    return _u8(j * t + airlight * (1.0 - t))


# ---------------------------------------------------------------- hợp đồng §5.1


def test_output_contract_and_left_columns(params):
    frame = _hazy(seed=2)
    out = C.enhance(frame, params)
    c = params["crop_left"]
    assert out.dtype == np.uint8 and out.shape == contract.FRAME_SHAPE
    assert c == 16  # hồ sơ v1.2.0; đổi hồ sơ thì test này nói rõ
    assert np.array_equal(out[:, :c], np.repeat(out[:, c:c + 1], c, axis=1))


def test_pure_deterministic_and_input_untouched(params):
    frame = _hazy(seed=3)
    before = frame.copy()
    a = C.enhance(frame, params)
    b = C.enhance(frame, params)
    assert np.array_equal(frame, before)
    assert np.array_equal(a, b)


def test_default_params_match_explicit(params):
    frame = _hazy(seed=4)
    assert np.array_equal(C.enhance(frame), C.enhance(frame, params))


@pytest.mark.parametrize("bad", [
    np.zeros((H, W, 3), np.float32),
    np.zeros((H, W - 1, 3), np.uint8),
    np.zeros((H, W), np.uint8),
    np.zeros((H, W, 4), np.uint8),
    np.zeros((W, H, 3), np.uint8),
    [[0, 0, 0]],
])
def test_rejects_wrong_type_or_shape(params, bad):
    with pytest.raises(ValueError):
        C.enhance(bad, params)


def test_load_params_reads_profile():
    p = C.load_params()
    prof = contract.load_profile(repo_root() / "configs/camera_profile.yaml")
    assert p["crop_left"] == prof["left_strip"]["crop_left_px"]
    assert p["vignette_focal_px"] == pytest.approx(prof["vignette"]["focal_px_fit"])
    assert p["profile_id"] == prof["profile_id"]


def _write_cfg(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load((repo_root() / C.CONFIG_REL).read_text(encoding="utf-8"))
    raw["camera_profile"] = str(repo_root() / raw["camera_profile"])
    mutate(raw)
    path = tmp_path / "enhance_classic.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("dehaze"),
    lambda r: r["magenta"].pop("strength"),
    lambda r: r["dehaze"].update(t_min=0.0),
    lambda r: r["dehaze"].update(airlight_range=[250.0, 180.0]),
    lambda r: r["vignette"].update(max_gain=0.5),
    lambda r: r["deblock"].update(block_px="8"),
    lambda r: r.update(camera_profile="khong-co.yaml"),
])
def test_load_params_rejects_bad_config(tmp_path, mutate):
    with pytest.raises(ValueError):
        C.load_params(_write_cfg(tmp_path, mutate))


def test_load_params_accepts_copied_config(tmp_path):
    assert C.load_params(_write_cfg(tmp_path, lambda r: None))["crop_left"] == 16


# ---------------------------------------------------------------- tiêu chí phase 12 bước 3


def test_saturated_red_roof_not_bleached(params):
    """Mái ngói đỏ bão hoà (R tới 255, B ≤ G — không hồng): bước khử hồng giữ nguyên màu, ΔE2000 < 3 mọi điểm."""
    rng = np.random.default_rng(5)
    h, w = H, W - params["crop_left"]
    tile = np.empty((h, w, 3), np.float32)
    tile[..., 2] = _field(rng, 190.0, 255.0, h=h, w=w)[..., 0] + 20.0  # R, nhiều điểm cắt ở 255
    tile[..., 1] = _field(rng, 50.0, 95.0, h=h, w=w)[..., 0]           # G
    tile[..., 0] = tile[..., 1] - rng.uniform(5.0, 25.0, (h, w))        # B < G (ngói đỏ/cam, không tím)
    img = _u8(tile)
    assert (img[..., 2] == 255).mean() > 0.1 and (img[..., 2] >= 240).mean() > 0.3  # đúng là gần/đã bão hoà
    assert C.pink_count(img, 0) == 0
    out = C.from_planes(C.demagenta(C.to_planes(img), params))
    de = deltaE_ciede2000(rgb2lab(img[..., ::-1]), rgb2lab(out[..., ::-1]))
    assert float(de.max()) < 3.0


def test_demagenta_noop_below_saturation_gate(params):
    """Không điểm nào qua cổng (max(R, B) < sat_level) → khử hồng trả đúng ảnh vào, kể cả vật hồng nhạt."""
    rng = np.random.default_rng(6)
    img = _u8(_field(rng, 60.0, params["magenta"]["sat_level"] - 1.0, h=H, w=W - params["crop_left"]))
    out = C.from_planes(C.demagenta(C.to_planes(img), params))
    assert np.array_equal(out, img)


def test_profile_pink_rule_is_measure_rule():
    prof = contract.load_profile(repo_root() / "configs/camera_profile.yaml")
    assert prof["magenta"]["rule"] == f"R-G>{M.PINK_DIFF} and B-G>{M.PINK_DIFF}"


def test_near_saturated_pink_reduced_90pct(params):
    """Mảng hồng gần bão hoà (R, B 242–255, G 175–215) trên nền xám có vân: còn ≤ 10 % điểm hồng theo luật hồ sơ."""
    rng = np.random.default_rng(7)
    gray = _field(rng, 120.0, 200.0, detail=4.0).mean(axis=2, keepdims=True)
    img = np.repeat(gray, 3, axis=2)
    yy, xx = np.mgrid[0:H, 0:W]
    blobs = np.zeros((H, W), bool)
    for _ in range(14):
        cy, cx, r = rng.integers(40, H - 40), rng.integers(60, W - 40), rng.integers(10, 28)
        blobs |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    n = int(blobs.sum())
    img[blobs, 2] = rng.uniform(242, 255, n)
    img[blobs, 0] = rng.uniform(242, 255, n)
    img[blobs, 1] = rng.uniform(175, 215, n)
    frame = _u8(img)
    c = params["crop_left"]
    before = C.pink_count(frame, c)
    assert before > 0.03 * frame[:, c:].shape[0] * frame[:, c:].shape[1]
    after = C.pink_count(C.enhance(frame, params), c)
    assert after <= 0.10 * before, f"còn {after}/{before} điểm hồng"


def test_synthetic_haze_dark_channel_halved(params):
    frame = _hazy(seed=8)
    c = params["crop_left"]
    before = C.dark_channel_median(frame, c)
    after = C.dark_channel_median(C.enhance(frame, params), c)
    assert before > 100  # màn mờ đặc thật (A·(1 − t) = 110)
    assert after <= 0.5 * before, f"dark channel trung vị {before:.1f} → {after:.1f}"


@pytest.mark.parametrize("make", [
    lambda: _hazy(seed=9),
    lambda: _u8(_field(np.random.default_rng(10), 40.0, 235.0)),
    # cảnh có bóng tối thật (5–60) cạnh vùng sáng: chỗ dễ bị đẩy về 0 nhất
    lambda: _u8(np.concatenate([_field(np.random.default_rng(11), 5.0, 60.0, h=H, w=W // 2),
                                _field(np.random.default_rng(12), 150.0, 240.0, h=H, w=W // 2)], axis=1)),
])
def test_black_level_not_crushed(params, make):
    """Tỉ lệ điểm có kênh nhỏ nhất ≤ 3 tăng không quá 0,1 điểm %."""
    frame = make()
    c = params["crop_left"]
    inc = C.black_frac(C.enhance(frame, params), c) - C.black_frac(frame, c)
    assert inc <= 0.001, f"tăng {inc * 100:.3f} điểm %"


# ---------------------------------------------------------------- từng bước


def test_flat_field_flattens_profile_vignette(params):
    h, w = H, W - params["crop_left"]
    gain = C._vignette_gain(h, w, float(params["vignette_focal_px"]), float(params["vignette"]["max_gain"]))
    shaded = _u8(np.repeat((200.0 / gain)[..., None], 3, axis=2))
    before = M.vignette_ratio(M.luma(shaded))
    after = M.vignette_ratio(M.luma(C.from_planes(C.flat_field(C.to_planes(shaded), params))))
    assert before < 0.9
    assert after == pytest.approx(1.0, abs=0.02)


def test_flat_field_gain_capped(params):
    gain = C._vignette_gain(H, W - params["crop_left"], 100.0, float(params["vignette"]["max_gain"]))
    assert float(gain.max()) == pytest.approx(params["vignette"]["max_gain"])
    assert float(gain.min()) >= 1.0


def test_deblock_reduces_jpeg_blocking(params):
    rng = np.random.default_rng(13)
    smooth = _u8(_field(rng, 60.0, 200.0, cells=(4, 5), detail=0.0))
    ok, buf = cv2.imencode(".jpg", smooth, [cv2.IMWRITE_JPEG_QUALITY, 15])
    assert ok
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    c = params["crop_left"]
    crop = np.ascontiguousarray(frame[:, c:])
    before = M.blocking_ratio(M.luma(crop), c)
    after = M.blocking_ratio(M.luma(C.from_planes(C.deblock(C.to_planes(crop), params))), c)
    assert before > 1.5
    assert after < before - 0.3 * (before - 1.0), f"blocking {before:.2f} → {after:.2f}"


def test_deblock_keeps_real_edge_on_grid(params):
    """Bậc thang lớn đúng trên biên lưới (cạnh thật) không bị san."""
    c = params["crop_left"]
    img = np.full((H, W - c, 3), 60, np.uint8)
    x_edge = (8 - c % 8) % 8 + 8 * 20  # biên khối của khung gốc
    img[:, x_edge:] = 180
    out = C.from_planes(C.deblock(C.to_planes(img), params))
    assert np.array_equal(out, img)


def test_cli_writes_png(tmp_path):
    src = tmp_path / "khung.jpg"
    assert cv2.imwrite(str(src), _hazy(seed=14))
    out_dir = tmp_path / "out"
    assert C.main(["--in", str(src), "--out", str(out_dir), "--json"]) == 0
    res = cv2.imread(str(out_dir / "khung.png"))
    assert res is not None and res.shape == contract.FRAME_SHAPE


def test_cli_bad_input_exit_2(tmp_path):
    bad = tmp_path / "nho.png"
    assert cv2.imwrite(str(bad), np.zeros((10, 10, 3), np.uint8))
    assert C.main(["--in", str(bad)]) == 2


# ---------------------------------------------------------------- bench thật (marker data)


@pytest.mark.data
def test_bench_json_within_budget(require_data):
    path = require_data(repo_root() / C.BENCH_REL, "python -m xla.enhance.classic --split dev --limit 200 --bench "
                                                   f"--bench-out {C.BENCH_REL.as_posix()}")
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert rep["n"] == 200 and rep["warmup"] == 20 and rep["source"]["split"] == "dev"
    assert {"cpu", "numpy", "opencv"} <= set(rep["env"])
    assert set(rep["ms"]["steps"]) == set(C.STEPS)
    # Cổng pass phase 12: p50 ≤ 30 ms, hoặc "lý do + số đo": máy hạ xung đo được (mẫu % Processor Performance < 80)
    # và có ghi chú lý do. Máy không hạ xung mà vẫn vượt → đỏ.
    p50 = rep["ms"]["total"]["p50"]
    if p50 > C.BUDGET_MS:
        assert rep["throttled"] is True and rep["cpu_perf_pct"]["min"] < C.PERF_OK_PCT, f"p50 {p50} ms, máy không hạ xung"
        assert rep["run_notes"], "vượt ngân sách vì hạ xung phải có ghi chú lý do (--note)"
        assert rep["gate"] == "over_budget_throttled"
    else:
        assert rep["gate"] == "pass"
    q = rep["quality"]
    assert q["black_frac_le3"]["max_increase"] <= 0.001
    assert q["dark_channel_median"]["after_mean"] < q["dark_channel_median"]["before_mean"]
    assert q["blocking_ratio_8x8"]["after_mean"] < q["blocking_ratio_8x8"]["before_mean"]
