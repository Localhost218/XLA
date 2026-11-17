"""Phase 02 Accept: hợp đồng model (docs/hop-dong-model.md) — bundle giả đạt, bundle hỏng báo đúng mục, schema hồ sơ."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
import pytest

from xla import contract as C


@pytest.fixture(scope="module")
def dummy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dummy")
    for model in C.MODELS:
        C.make_dummy_bundle(root, model)
    return root


def _copy(dummy_root: Path, tmp_path: Path, model: str = "detector") -> Path:
    name = f"{model}-v{C.DUMMY_VERSION}"
    dst = tmp_path / name
    shutil.copytree(dummy_root / name, dst)
    return dst


def _meta(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / C.META_FILE).read_text(encoding="utf-8"))


def _write_meta(bundle: Path, meta: dict[str, Any]) -> None:
    (bundle / C.META_FILE).write_text(json.dumps(meta), encoding="utf-8")


def _save_model(bundle: Path, m: onnx.ModelProto, model: str = "detector") -> None:
    path = bundle / f"{model}.onnx"
    onnx.save(m, str(path))
    meta = _meta(bundle)
    meta["onnx_sha256"] = C.sha256_file(path)  # chỉ hỏng đúng một chỗ: sha256 vẫn khớp
    _write_meta(bundle, meta)


def test_constants_match_contract() -> None:
    assert C.CONTRACT_VERSION == "1.0"
    assert (C.FRAME_W, C.FRAME_H) == (640, 480)
    assert C.FRAME_SHAPE == (480, 640, 3)
    assert C.OPSET == 17
    assert C.INPUT_NAME == "frame"
    assert set(C.DETECTOR_OUTPUTS) == {"boxes", "scores"}
    assert set(C.ENHANCER_OUTPUTS) == {"frame_out"}
    assert (C.DET_IOU_MIN, C.DET_SCORE_ABS_MAX, C.ENH_MAE_MAX, C.ENH_ABS_MAX) == (0.95, 0.02, 1.0, 8)


@pytest.mark.parametrize("model", C.MODELS)
def test_dummy_bundle_passes(dummy_root: Path, model: str) -> None:
    bundle = dummy_root / f"{model}-v{C.DUMMY_VERSION}"
    assert C.validate_bundle(bundle) == []
    names = sorted(p.name for p in (bundle / "golden").iterdir())
    assert {"000.jpg", "001.jpg", "002.jpg", "expected.json"} <= set(names)


def test_dummy_detector_returns_fixed_box(dummy_root: Path) -> None:
    exp = json.loads((dummy_root / f"detector-v{C.DUMMY_VERSION}" / "golden" / "expected.json").read_text())
    for v in exp.values():
        assert v["boxes"] == [list(C.DUMMY_BOX)]
        assert v["scores"] == pytest.approx([C.DUMMY_SCORE])


def test_dummy_enhancer_repeats_left_strip(dummy_root: Path) -> None:
    bundle = dummy_root / f"enhancer-v{C.DUMMY_VERSION}"
    out = C.decode_png(bundle / "golden" / "000.out.png")
    src = C.decode_frame(bundle / "golden" / "000.jpg")
    assert out.shape == C.FRAME_SHAPE
    assert (out[:, :16] == out[:, 16:17]).all()
    assert (out[:, 16:] == src[:, 16:]).all()


def _break_sha(b: Path) -> None:
    meta = _meta(b)
    meta["onnx_sha256"] = "0" * 64
    _write_meta(b, meta)


def _break_input_name(b: Path) -> None:
    m = onnx.load(str(b / "detector.onnx"))
    m.graph.input[0].name = "image"
    for node in m.graph.node:
        node.input[:] = ["image" if x == C.INPUT_NAME else x for x in node.input]
    _save_model(b, m)


def _break_output_dtype(b: Path) -> None:
    m = onnx.load(str(b / "detector.onnx"))
    for o in m.graph.output:
        if o.name == "scores":
            o.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    _save_model(b, m)


def _break_missing_meta_key(b: Path) -> None:
    meta = _meta(b)
    del meta["train_run"]
    _write_meta(b, meta)


def _break_major(b: Path) -> None:
    meta = _meta(b)
    meta["contract_version"] = "2.0"
    _write_meta(b, meta)


@pytest.mark.parametrize(
    ("breaker", "section", "needle"),
    [
        (_break_sha, "§4 sha256", "meta.onnx_sha256"),
        (_break_input_name, "§2 chữ ký", "'frame'"),
        (_break_output_dtype, "§2 chữ ký", "'scores' dtype float64"),
        (_break_missing_meta_key, "§4 meta.json", "'train_run'"),
        (_break_major, "§4 contract_version", "major 2"),
    ],
    ids=["sai-sha256", "sai-ten-input", "sai-dtype-output", "thieu-khoa-meta", "major-2"],
)
def test_broken_bundle_reports_exactly_one_error(
    dummy_root: Path, tmp_path: Path, breaker: Any, section: str, needle: str
) -> None:
    b = _copy(dummy_root, tmp_path)
    breaker(b)
    errors = C.validate_bundle(b)
    assert len(errors) == 1, errors
    assert errors[0].startswith(section + ":"), errors
    assert needle in errors[0], errors


def test_cli_validate_exit_codes(dummy_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert C.main(["validate", str(dummy_root / f"detector-v{C.DUMMY_VERSION}")]) == 0
    assert capsys.readouterr().out.strip() == "[]"

    b = _copy(dummy_root, tmp_path)
    _break_major(b)
    assert C.main(["validate", str(b), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["errors"][0]["section"] == "§4 contract_version"

    assert C.main(["validate", str(tmp_path / "khong-co")]) == 2


def test_cli_make_dummy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert C.main(["make-dummy", str(tmp_path), "--json"]) == 0
    made = json.loads(capsys.readouterr().out)["bundles"]
    assert [Path(p).name for p in made] == ["detector-v0.0.0", "enhancer-v0.0.0"]


# ---- §6 schema hồ sơ camera ----------------------------------------------------------------------------------------
def _d(p5: float, p50: float, p95: float) -> dict[str, float]:
    return {"p5": p5, "p50": p50, "p95": p95}


MIN_PROFILE: dict[str, Any] = {
    "schema_version": "1.0",
    "profile_version": "1.0.0",
    "profile_id": "ov3660-esp32s3-q12-20261102",
    "measured_on": {
        "recording": "20261102-100941",
        "split_file": "data/real/split.json",
        "n_frames": 10,
        "measured": "2026-11-18",
    },
    "sensor": {"model": "OV3660", "board": "ESP32-S3 N16R8", "xclk_mhz": 10, "mount": "nadir"},
    "frame": {"width_px": 640, "height_px": 480, "format": "MJPEG"},
    "jpeg": {
        "esp32_quality": 12,
        "baseline": True,
        "subsampling": "4:2:2",
        "qtable_luma": [8] * 64,
        "qtable_chroma": [9] * 64,
        "annex_k_scale": 0.75,
        "other_qualities": {"rule": "annex_k * q / 16", "verified": False},
    },
    "left_strip": {"width_px": 13, "crop_left_px": 16, "mean_y_strip": 20.0, "mean_y_ref": 110.0},
    "vignette": {"corner_center_ratio": _d(0.6, 0.7, 0.8), "model": "cos4", "focal_px_fit": 500.0},
    "haze": {
        "dark_channel_15px_0_255": _d(10, 30, 60),
        "y_p1_0_255": _d(5, 15, 30),
        "saturation_0_255": _d(20, 40, 70),
    },
    "sharpness": {"laplacian_var": _d(50, 120, 300), "mtf50_cyc_per_px": _d(0.1, 0.15, 0.2), "psf_sigma_px_fit": 1.1},
    "noise": {
        "sigma_immerkaer_0_255": _d(1, 2, 4),
        "nlf": [[64, 2.0], [128, 2.5]],
        "psd_radial": [[0.1, 1.0], [0.4, 0.2]],
    },
    "color": {"r_over_g": _d(0.8, 0.9, 1.0), "b_over_g": _d(0.7, 0.8, 0.9), "wb_gain_r_fit": 1.0, "wb_gain_b_fit": 1.0},
    "magenta": {
        "rule": "R-G>25 and B-G>25",
        "frac_pixels": _d(0.0, 0.001, 0.01),
        "frac_frames_gt_0p5pct": 0.1,
        "highlight_y_0_255": _d(200, 230, 250),
    },
    "blocking": {"ratio_8x8": _d(1.0, 1.1, 1.3)},
    "motion_blur": {"sobel_x2_over_y2": _d(0.8, 1.0, 1.6), "frac_frames_gt_1p5": 0.05},
    "person_box_px": {
        "short_side": {"p5": 21, "p50": 50, "p95": 139, "min": 15, "max": 160},
        "source": "yolo11s-so-bo",
    },
    "altitude_m": {"flight_frames": _d(5, 12, 25), "filter": "rel_alt_m > 3, bỏ điểm vọt"},
}


def test_schema_is_valid_draft_2020_12() -> None:
    import jsonschema

    schema = C.camera_profile_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)


def test_minimal_profile_passes() -> None:
    assert C.validate_camera_profile(copy.deepcopy(MIN_PROFILE)) == []


def test_profile_without_qtable_luma_rejected() -> None:
    p = copy.deepcopy(MIN_PROFILE)
    del p["jpeg"]["qtable_luma"]
    errors = C.validate_camera_profile(p)
    assert len(errors) == 1 and "qtable_luma" in errors[0], errors


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (("jpeg", "qtable_chroma"), [9] * 63, "qtable_chroma"),
        (("jpeg", "qtable_luma"), [0] + [8] * 63, "qtable_luma"),
        (("jpeg", "subsampling"), "4:1:1", "subsampling"),
        (("haze", "y_p1_0_255"), _d(30, 15, 5), "p5 ≤ p50 ≤ p95"),
        (("person_box_px", "short_side"), {"p5": 21, "p50": 50, "p95": 139, "min": 30, "max": 160}, "min ≤ p5"),
    ],
    ids=["qtable-63", "qtable-0", "subsampling", "dist-nguoc", "short-side-min"],
)
def test_profile_constraints(path: tuple[str, ...], value: Any, needle: str) -> None:
    p = copy.deepcopy(MIN_PROFILE)
    node = p
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    errors = C.validate_camera_profile(p)
    assert len(errors) == 1 and needle in errors[0], errors
