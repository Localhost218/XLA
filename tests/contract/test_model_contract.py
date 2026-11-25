"""Accept hợp đồng model (docs/hop-dong-model.md): phase 02 v1 (bundle giả đạt, bundle hỏng báo đúng mục, schema hồ sơ)
và phase 01 plan 261125-ai-v2 (v1.1: verifier, enhancer-offline, recommended_track, classic, docs §9–§14, parity
RoiAlign)."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
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
    assert C.CONTRACT_VERSION == "1.1"
    assert (C.FRAME_W, C.FRAME_H) == (640, 480)
    assert C.FRAME_SHAPE == (480, 640, 3)
    assert C.OPSET == 17
    assert C.INPUT_NAME == "frame"
    assert set(C.DETECTOR_OUTPUTS) == {"boxes", "scores"}
    assert set(C.ENHANCER_OUTPUTS) == {"frame_out"}
    assert (C.DET_IOU_MIN, C.DET_SCORE_ABS_MAX, C.ENH_MAE_MAX, C.ENH_ABS_MAX) == (0.95, 0.02, 1.0, 8)
    # v1.1 (§9–§11)
    assert C.MODELS == ("detector", "enhancer", "verifier", "enhancer-offline")
    assert list(C.VERIFIER_INPUTS) == ["frame", "boxes"] and set(C.VERIFIER_OUTPUTS) == {"scores"}
    assert (C.VERIFIER_ROI_MODE, C.VERIFIER_SAMPLING_RATIO, C.VERIFIER_CROP_PX) == ("output_half_pixel", 2, (64, 96))
    assert C.VER_SCORE_ABS_MAX == 0.02
    assert C.OFFLINE_SHAPE == (960, 1280, 3) and C.OFFLINE_OUTPUT_SIZE == [1280, 960]


# ---- docs = SSOT: phải khai 1.1 và có đủ §9–§14 (chép từ plans/261125-ai-v2/contract.md) ----------------------------
DOCS = Path(__file__).resolve().parents[2] / "docs" / "hop-dong-model.md"
DOC_SECTIONS = {
    9: ("verifier", "output_half_pixel", "aligned=False", "golden/boxes.json", "VERIFIER_CONFIDENCE"),
    10: ("recommended_track", "track_filter.py", "DETECTION_TRACK_MAX_GAP", "verifier_score", "track_id"),
    11: ("enhancer-offline", "temporal_window", "output_size", "ENHANCER_OFFLINE_SHA256", "2·crop_left"),
    12: ("classic", "classic_params_sha256", "enhancer-v2.1.0", "hybrid"),
    13: ("/api/flights/enhance-offline", "link_recent", "disk_full", "variant=offline", "enhancer_default_on"),
    14: ("replay_pred.py", "--blank", "neg_flight_split.json", "folds.json", "c1b0172840972b3951ffcd2060b482e8"),
}


def test_docs_declare_v1_1_and_sections_9_to_14() -> None:
    text = DOCS.read_text(encoding="utf-8")
    assert text.splitlines()[0].endswith("`contract_version` 1.1"), text.splitlines()[0]
    heads = [ln for ln in text.splitlines() if ln.startswith("## ")]
    for n in range(1, 15):
        assert sum(h.startswith(f"## {n}. ") for h in heads) == 1, (n, heads)
    for n, needles in DOC_SECTIONS.items():
        body = text.split(f"\n## {n}. ", 1)[1].split("\n## ", 1)[0]
        missing = [s for s in needles if s not in body]
        assert not missing, (n, missing)


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


@pytest.mark.parametrize("kind", ["verifier", "enhancer-offline"])
def test_cli_make_dummy_kind_then_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str) -> None:
    """AX-01: make-dummy --kind <k> --json rồi validate --json ra 0."""
    assert C.main(["make-dummy", str(tmp_path), "--kind", kind, "--json"]) == 0
    made = json.loads(capsys.readouterr().out)["bundles"]
    assert [Path(p).name for p in made] == [f"{kind}-v0.0.0"]
    assert C.main(["validate", made[0], "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"bundle": made[0], "ok": True, "errors": []}


def test_cli_make_dummy_unknown_kind_exit_2(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        C.main(["make-dummy", str(tmp_path), "--kind", "enhancer-hybrid"])
    assert e.value.code == 2


# ---- v1.1: kind mới, bundle v1 vẫn hợp lệ --------------------------------------------------------------------------
RELEASE = Path(__file__).resolve().parents[2] / "models" / "release"


@pytest.mark.parametrize("name", ["detector-v1.0.0", "enhancer-v1.0.0"])
def test_v1_release_bundle_still_valid_under_1_1(name: str) -> None:
    bundle = RELEASE / name
    onnx_file = bundle / f"{name.split('-v')[0]}.onnx"
    if not onnx_file.is_file():  # .onnx bị .gitignore chặn; lấy từ GitHub Release models-v1.0.0
        pytest.skip(f"{onnx_file} chưa có (tải Release models-v1.0.0)")
    assert _meta(bundle)["contract_version"] == "1.0"
    assert C.validate_bundle(bundle) == []


def test_dummy_verifier_golden_has_n0_and_constant_scores(dummy_root: Path) -> None:
    gdir = dummy_root / f"verifier-v{C.DUMMY_VERSION}" / "golden"
    boxes = json.loads((gdir / C.BOXES_FILE).read_text(encoding="utf-8"))
    exp = json.loads((gdir / C.EXPECTED_FILE).read_text(encoding="utf-8"))
    assert boxes["002.jpg"] == [] and exp["002.jpg"]["scores"] == []
    for name, lst in boxes.items():
        assert exp[name]["scores"] == pytest.approx([C.DUMMY_VERIFIER_SCORE] * len(lst))


def test_dummy_verifier_runs_probe_and_empty(dummy_root: Path) -> None:
    """Chạy thử lúc IOT-CV nạp (§9): khung zeros + một hộp → [1] điểm; N = 0 → shape [0]."""
    onnx_path = dummy_root / f"verifier-v{C.DUMMY_VERSION}" / "verifier.onnx"
    zeros = np.zeros(C.FRAME_SHAPE, np.uint8)
    outs = C.run_feeds(
        onnx_path,
        {
            "probe": {"frame": zeros, "boxes": np.asarray([C.VERIFIER_PROBE_BOX], np.float32)},
            "empty": {"frame": zeros, "boxes": np.zeros((0, 4), np.float32)},
        },
    )
    assert outs["probe"]["scores"].shape == (1,) and outs["probe"]["scores"].dtype == np.float32
    assert outs["empty"]["scores"].shape == (0,)


def test_dummy_enhancer_offline_is_2x_with_doubled_strip(dummy_root: Path) -> None:
    bundle = dummy_root / f"enhancer-offline-v{C.DUMMY_VERSION}"
    out = C.decode_png(bundle / "golden" / "000.out.png")
    src = C.decode_frame(bundle / "golden" / "000.jpg")
    assert out.shape == C.OFFLINE_SHAPE
    cl2 = 2 * C.CROP_LEFT_V1
    assert (out[:, :cl2] == out[:, cl2 : cl2 + 1]).all()
    assert (out[::2, ::2][:, C.CROP_LEFT_V1 :] == src[:, C.CROP_LEFT_V1 :]).all()


def _set(key: str, value: Any) -> Any:
    def f(meta: dict[str, Any]) -> None:
        meta[key] = value

    return f


def _drop(key: str) -> Any:
    def f(meta: dict[str, Any]) -> None:
        del meta[key]

    return f


@pytest.mark.parametrize(
    ("kind", "mutate", "section", "needle"),
    [
        ("detector", _set("recommended_track", {"k": 4, "n": 3, "iou": 0.3, "max_gap": 2}), "§10", "k 4 > n 3"),
        ("detector", _set("recommended_track", {"k": 0, "n": 3, "iou": 0.3, "max_gap": 2}), "§10", "k 0"),
        ("detector", _set("recommended_track", {"k": 2, "n": 3, "iou": 0.3}), "§10", "'max_gap'"),
        ("detector", _set("recommended_track", {"k": 2, "n": 3, "iou": 1.5, "max_gap": 2}), "§10", "iou 1.5"),
        ("enhancer-offline", _set("output_size", [640, 480]), "§11 meta.json", "output_size"),
        ("enhancer-offline", _set("temporal_window", 2), "§11 meta.json", "temporal_window 2"),
        ("enhancer-offline", _drop("s_per_frame"), "§11 meta.json", "'s_per_frame'"),
        ("verifier", _drop("crop_px"), "§9 meta.json", "'crop_px'"),
        ("verifier", _set("crop_px", 80), "§9 meta.json", "crop_px 80"),
        ("verifier", _set("input_variant", "enhanced"), "§9 meta.json", "input_variant"),
        ("verifier", _set("contract_version", "1.0"), "§4 contract_version", "'verifier'"),
        ("verifier", _set("imgsz", 640), "§9 meta.json", "'imgsz'"),
        ("enhancer", _set("network", "classic"), "§12 meta.json", "classic_params_sha256"),
    ],
    ids=[
        "track-k-lon-hon-n",
        "track-k-0",
        "track-thieu-max-gap",
        "track-iou-ngoai",
        "offline-output-size",
        "offline-T-chan",
        "offline-thieu-s-per-frame",
        "verifier-thieu-crop-px",
        "verifier-crop-px-80",
        "verifier-input-enhanced",
        "verifier-cv-1.0",
        "verifier-khoa-detector",
        "classic-thieu-sha",
    ],
)
def test_bad_v1_1_meta_rejected_with_reason(
    dummy_root: Path, tmp_path: Path, kind: str, mutate: Any, section: str, needle: str
) -> None:
    b = _copy(dummy_root, tmp_path, kind)
    meta = _meta(b)
    mutate(meta)
    _write_meta(b, meta)
    errors = C.validate_bundle(b)
    assert len(errors) == 1, errors
    assert errors[0].startswith(section), errors
    assert needle in errors[0], errors


def test_good_recommended_track_and_classic_pass(dummy_root: Path, tmp_path: Path) -> None:
    det = _copy(dummy_root, tmp_path, "detector")
    meta = _meta(det)
    meta["recommended_track"] = {"k": 3, "n": 3, "iou": 0.3, "max_gap": 0}
    _write_meta(det, meta)
    assert C.validate_bundle(det) == []
    enh = _copy(dummy_root, tmp_path, "enhancer")
    meta = _meta(enh)
    meta.update(network="classic", classic_params_sha256="a" * 64)
    _write_meta(enh, meta)
    assert C.validate_bundle(enh) == []


@pytest.mark.parametrize("bad_name", ["enhancer-v2.1.0-hybrid", "verifier-1.0.0", "enhancer_offline-v1.0.0"])
def test_bad_dir_name_rejected(dummy_root: Path, tmp_path: Path, bad_name: str) -> None:
    src = {"enhancer-v": "enhancer", "verifier-": "verifier", "enhancer_offline-": "enhancer-offline"}
    kind = next(v for k, v in src.items() if bad_name.startswith(k))
    dst = tmp_path / bad_name
    shutil.copytree(dummy_root / f"{kind}-v{C.DUMMY_VERSION}", dst)
    errors = C.validate_bundle(dst)
    assert len(errors) == 1, errors
    assert errors[0].startswith("§4 bundle:") and bad_name in errors[0], errors


def test_verifier_signature_needs_boxes_input(dummy_root: Path, tmp_path: Path) -> None:
    """Verifier mà graph chỉ có input ``frame`` (vd. chép nhầm graph enhancer) → lỗi chữ ký §9."""
    b = _copy(dummy_root, tmp_path, "verifier")
    _save_model(b, C._dummy_graph("enhancer", C.CROP_LEFT_V1), "verifier")
    errors = C.validate_bundle(b)
    assert any(e.startswith("§9 chữ ký:") and "'boxes'" in e for e in errors), errors


def test_offline_temporal_window_bundle_validates(tmp_path: Path) -> None:
    """Bundle video (T = 3): input ``frames`` [3, 480, 640, 3], golden chạy cửa sổ lặp khung biên."""
    from onnx import TensorProto, helper

    b = C.make_dummy_bundle(tmp_path, "enhancer-offline")
    frames_in = helper.make_tensor_value_info("frames", TensorProto.UINT8, [3, *C.FRAME_SHAPE])
    nodes = [
        C._const("mid", TensorProto.INT64, [], [1]),
        helper.make_node("Gather", ["frames", "mid"], ["frame"], axis=0),  # khung giữa → đúng graph T = 1
        *C._dummy_graph("enhancer-offline", C.CROP_LEFT_V1).graph.node,
    ]
    out = helper.make_tensor_value_info("frame_out", TensorProto.UINT8, list(C.OFFLINE_SHAPE))
    _save_model(b, C._finish_model(nodes, "t3", [frames_in], [out]), "enhancer-offline")
    meta = _meta(b)
    meta["temporal_window"] = 3
    _write_meta(b, meta)
    assert C.validate_bundle(b) == []
    seq = [np.full(C.FRAME_SHAPE, i, np.uint8) for i in range(3)]
    assert [int(x[0, 0, 0]) for x in C.temporal_stack(seq, 0, 3)] == [0, 0, 1]
    assert [int(x[0, 0, 0]) for x in C.temporal_stack(seq, 2, 5)] == [0, 1, 2, 2, 2]


# ---- §9 parity RoiAlign ORT ↔ torchvision.ops.roi_align(aligned=False) ----------------------------------------------
@pytest.mark.parametrize("crop_px", C.VERIFIER_CROP_PX)
def test_roi_align_parity_with_torchvision(tmp_path: Path, crop_px: int) -> None:
    torch = pytest.importorskip("torch")
    tv_ops = pytest.importorskip("torchvision.ops")

    rng = np.random.default_rng(20261125)
    frame = rng.integers(0, 256, C.FRAME_SHAPE, dtype=np.uint8)
    wh = rng.uniform(4, 220, (20, 2))
    xy = rng.uniform(0, 1, (20, 2)) * (np.array([C.FRAME_W, C.FRAME_H]) - wh)
    boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    boxes[0] = [10.3, 20.7, 11.1, 21.2]  # hộp < 1 px: hai bên cùng nâng cạnh lên 1
    path = tmp_path / "roi.onnx"
    onnx.save(C.roi_align_probe_model(crop_px), str(path))
    got = C.run_feeds(
        path,
        {"20": {"frame": frame, "boxes": boxes}, "0": {"frame": frame, "boxes": np.zeros((0, 4), np.float32)}},
    )
    x = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)[None]
    for key, bx in (("20", boxes), ("0", np.zeros((0, 4), np.float32))):
        ref = tv_ops.roi_align(
            x, [torch.from_numpy(bx)], output_size=crop_px, spatial_scale=1.0, sampling_ratio=2, aligned=False
        ).numpy()
        assert got[key]["crops"].shape == ref.shape == (len(bx), 3, crop_px, crop_px)
        if len(bx):
            assert float(np.abs(got[key]["crops"] - ref).max()) <= 1e-3


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
