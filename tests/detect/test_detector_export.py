"""Phase 11 bước 5–7: ``xla.detect.export_onnx`` — graph bọc đúng §2, bundle phát hành đúng §4.

Phần đơn vị (không marker, CPU, không dữ liệu thật): letterbox khớp ``ultralytics.data.augment.LetterBox``, toạ độ quy về
khung gốc, cắt/kẹp mép, lọc lớp người kiểu ultralytics, ``score_floor``, NMS, ``max_det``, onnxsim, cách gọi CLI.
Phần ``data``: bundle thật ``models/release/detector-v1.0.0`` + ``results/e2/*.json`` (phase 11 tạo); ``gpu``: đo lại ms.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from xla import contract as C
from xla.detect import export_onnx as X
from xla.paths import repo_root

CROP = 16


# ---- graph lõi giả: output0 hằng [1, 4+nc, A] nhưng vẫn phụ thuộc input --------------------------------------
def fake_core(pred: np.ndarray, imgsz: int) -> onnx.ModelProto:
    """``pred`` [4+nc, A]: hàng 0–3 = cx, cy, w, h (px letterbox), hàng 4.. = điểm lớp."""
    pred = np.asarray(pred, np.float32)[None]
    nodes = [
        helper.make_node("ReduceMax", ["images"], ["m"], keepdims=0),
        helper.make_node("Constant", [], ["zero"], value=numpy_helper.from_array(np.float32(0.0), "z")),
        helper.make_node("Mul", ["m", "zero"], ["dep"]),
        helper.make_node("Constant", [], ["p"], value=numpy_helper.from_array(pred, "pv")),
        helper.make_node("Add", ["p", "dep"], ["output0"]),
    ]
    g = helper.make_graph(nodes, "fake_core",
                          [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(pred.shape))])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", C.OPSET)])
    m.ir_version = 8
    return m


def anchors(rows: list[tuple[float, ...]]) -> np.ndarray:
    """Mỗi phần tử ``(cx, cy, w, h, s0, s1, ...)`` → mảng [4+nc, A]."""
    return np.asarray(rows, np.float32).T


def run(pred: np.ndarray, imgsz: int = 640, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    m = X.build_detector_model(fake_core(pred, imgsz), imgsz=imgsz, crop_left=CROP, **kw)
    frame = np.full(C.FRAME_SHAPE, 50, np.uint8)
    return X.run_detector(X.ort_session(m), frame)


def to_orig(xyxy_lb: list[float], imgsz: int = 640) -> np.ndarray:
    """Công thức §2, tính độc lập với module."""
    r = min(imgsz / 480, imgsz / (640 - CROP))
    new_w, new_h = round((640 - CROP) * r), round(480 * r)
    pad_x, pad_y = round((imgsz - new_w) / 2 - 0.1), round((imgsz - new_h) / 2 - 0.1)
    x1, y1, x2, y2 = xyxy_lb
    return np.asarray([(x1 - pad_x) / r + CROP, (y1 - pad_y) / r, (x2 - pad_x) / r + CROP, (y2 - pad_y) / r])


# ---- letterbox -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("imgsz", [320, 640, 960, 1280])
@pytest.mark.parametrize("crop", [0, CROP])
def test_letterbox_params_match_ultralytics(imgsz: int, crop: int) -> None:
    from ultralytics.data.augment import LetterBox

    p = LetterBox(new_shape=(imgsz, imgsz), auto=False).get_params({"img": np.zeros((480, 640 - crop, 3), np.uint8)})
    lb = X.Letterbox(imgsz, crop)
    assert (lb.new_w, lb.new_h) == p["new_unpad"]
    assert lb.pads == (p["top"], p["bottom"], p["left"], p["right"])
    assert lb.new_h + p["top"] + p["bottom"] == imgsz and lb.new_w + p["left"] + p["right"] == imgsz


@pytest.mark.parametrize("imgsz", [320, 640, 1280])  # thu nhỏ, gần 1:1, phóng to
@pytest.mark.parametrize("noise", [0.0, 25.0, 80.0])
def test_preprocess_bit_exact_with_ultralytics_letterbox(imgsz: int, noise: float) -> None:
    """Tiền xử lý trong graph = ``LetterBox`` (cv2 INTER_LINEAR uint8) → RGB → /255, **không lệch mức nào**.

    Float ``Resize`` + ``Round`` lệch +1 mức ở ~12 % điểm ảnh (phép dịch cắt cụt của OpenCV); test này bắt lại lỗi đó."""
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(20261118)
    yy, xx = np.mgrid[0:480, 0:640]
    smooth = np.stack([(xx * 0.4 + yy * 0.1), (yy * 0.5), (xx * 0.2 + 60)], axis=2)
    frame = np.clip(smooth + rng.normal(0, noise, smooth.shape), 0, 255).astype(np.uint8)
    frame[:, :CROP] = 255  # dải mép phải bị cắt, không lọt vào ảnh
    got = X.ort_session(X.build_preprocess_model(imgsz, CROP)).run(None, {C.INPUT_NAME: frame})[0]
    ref = LetterBox(new_shape=(imgsz, imgsz), auto=False)(image=np.ascontiguousarray(frame[:, CROP:]))
    ref = ref[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    assert got.shape == ref.shape == (1, 3, imgsz, imgsz)
    assert np.array_equal(np.rint(got * 255.0), np.rint(ref * 255.0))


@pytest.mark.parametrize(("src", "dst"), [(624, 1280), (480, 985), (624, 320), (480, 246)])
def test_resize_taps_match_cv2(src: int, dst: int) -> None:
    """Hệ số nguyên của một trục tái tạo đúng ``cv2.resize`` trên ảnh một hàng/một cột."""
    import cv2

    line = np.random.default_rng(src + dst).integers(0, 256, src).astype(np.uint8)
    s0, s1, a0, a1 = X.resize_taps(src, dst, clamp=True)
    h = line[s0].astype(np.int64) * a0 + line[s1].astype(np.int64) * a1
    ref = cv2.resize(line[None, :, None].repeat(3, 2), (dst, 1), interpolation=cv2.INTER_LINEAR)[0, :, 0]
    # một hàng: dọc b0 = 2048 → ((2048·(h>>4))>>16 + 2) >> 2
    assert np.array_equal(((((h >> 4) * 2048) >> 16) + 2) >> 2, ref)


# ---- hậu xử lý ---------------------------------------------------------------------------------------------------
def test_signature_matches_contract() -> None:
    m = X.build_detector_model(fake_core(anchors([(320, 320, 50, 50, 0.9)]), 640), imgsz=640, crop_left=CROP)
    assert {o.domain: o.version for o in m.opset_import}[""] == C.OPSET
    (inp,) = m.graph.input
    assert inp.name == C.INPUT_NAME and inp.type.tensor_type.elem_type == TensorProto.UINT8
    assert C._dims(inp) == C.FRAME_SHAPE
    outs = {o.name: (o.type.tensor_type.elem_type, C._dims(o)) for o in m.graph.output}
    assert outs == {"boxes": (TensorProto.FLOAT, ("N", 4)), "scores": (TensorProto.FLOAT, ("N",))}
    assert any(n.op_type == "NonMaxSuppression" for n in m.graph.node)


def test_coordinates_map_back_to_original_frame() -> None:
    boxes, scores = run(anchors([(320, 320, 100, 50, 0.9), (100, 200, 40, 80, 0.6)]))
    assert boxes.dtype == np.float32 and scores.dtype == np.float32
    assert np.allclose(scores, [0.9, 0.6], atol=1e-6)
    assert np.allclose(boxes[0], to_orig([270, 295, 370, 345]), atol=1e-3)
    assert np.allclose(boxes[1], to_orig([80, 160, 120, 240]), atol=1e-3)


def test_coordinates_agree_with_ultralytics_scale_boxes() -> None:
    """ultralytics mới chia theo từng trục (``new_h/h``), §2 chia một ``r``: lệch < 0,5 px."""
    import torch
    from ultralytics.utils.ops import scale_boxes

    boxes, _ = run(anchors([(640, 700, 300, 200, 0.9)]), imgsz=1280)
    ref = scale_boxes((1280, 1280), torch.tensor([[490.0, 600.0, 790.0, 800.0]]), (480, 640 - CROP)).numpy()[0]
    ref[[0, 2]] += CROP
    assert np.abs(boxes[0] - ref).max() < 0.5


def test_boxes_clipped_to_crop_and_frame() -> None:
    lb = X.Letterbox(640, CROP)
    # tràn trái (vào dải mép) và tràn dưới (vào lề letterbox)
    boxes, _ = run(anchors([(10, 300, 60, 40, 0.9), (400, 640 - lb.pads[1] + 10, 50, 60, 0.8)]))
    assert len(boxes) == 2
    assert boxes[0, 0] == pytest.approx(CROP) and boxes[0, 0] < boxes[0, 2]
    assert boxes[1, 3] == pytest.approx(480.0) and boxes[1, 1] < boxes[1, 3]
    assert boxes.min() >= 0 and boxes[:, [0, 2]].max() <= 640 and boxes[:, [1, 3]].max() <= 480


def test_box_entirely_in_padding_is_dropped() -> None:
    top = X.Letterbox(640, CROP).pads[0]
    boxes, scores = run(anchors([(300, top / 2, 40, 20, 0.95), (300, 300, 40, 40, 0.5)]))
    assert np.allclose(scores, [0.5]) and boxes.shape == (1, 4)


def test_score_floor_inclusive() -> None:
    _, scores = run(anchors([(100, 300, 30, 30, 0.049), (300, 300, 30, 30, 0.05), (500, 300, 30, 30, 0.051)]))
    assert np.allclose(sorted(scores), [0.05, 0.051], atol=1e-6)


def test_empty_output_has_contract_shapes() -> None:
    boxes, scores = run(anchors([(100, 300, 30, 30, 0.01), (300, 300, 30, 30, 0.02)]))
    assert boxes.shape == (0, 4) and scores.shape == (0,)
    assert boxes.dtype == np.float32 and scores.dtype == np.float32


def test_nms_suppresses_overlap_keeps_distinct_sorted() -> None:
    # A và B IoU = 0,6 > 0,5 → B bị bỏ; C IoU với A ≈ 0,33 → giữ; D xa
    rows = [(300, 300, 100, 100, 0.7), (325, 300, 100, 100, 0.6), (350, 300, 100, 100, 0.65),
            (100, 400, 40, 40, 0.9)]
    _, scores = run(anchors(rows), nms_iou=0.5)
    assert np.allclose(scores, [0.9, 0.7, 0.65], atol=1e-6)
    assert np.all(np.diff(scores) <= 0)


def test_max_det_keeps_top_scores() -> None:
    rows = [(60 + 100 * i, 300, 40, 40, 0.1 + 0.1 * i) for i in range(6)]
    _, scores = run(anchors(rows), max_det=3)
    assert np.allclose(scores, [0.6, 0.5, 0.4], atol=1e-6)


def test_multiclass_keeps_person_only_when_argmax() -> None:
    """Như ``non_max_suppression(classes=[0])``: lớp tốt nhất trước, rồi mới lọc lớp người."""
    rows = [(100, 300, 40, 40, 0.6, 0.1, 0.2),  # người thắng → giữ
            (300, 300, 40, 40, 0.5, 0.8, 0.1),  # lớp 1 thắng → bỏ dù điểm người ≥ floor
            (500, 300, 40, 40, 0.02, 0.01, 0.0)]  # người thắng nhưng < floor → bỏ
    _, scores = run(anchors(rows))
    assert np.allclose(scores, [0.6], atol=1e-6)


def test_core_must_match_imgsz() -> None:
    with pytest.raises(ValueError, match="input"):
        X.build_detector_model(fake_core(anchors([(1, 1, 1, 1, 0.9)]), 320), imgsz=640, crop_left=CROP)


def test_simplify_checked_preserves_outputs() -> None:
    rows = [(300, 300, 100, 100, 0.7), (100, 400, 40, 40, 0.9), (325, 300, 100, 100, 0.6)]
    m = X.build_detector_model(fake_core(anchors(rows), 640), imgsz=640, crop_left=CROP)
    frames = [np.full(C.FRAME_SHAPE, v, np.uint8) for v in (0, 200)]
    simp, applied, note = X.simplify_checked(m, frames)
    assert applied, note
    for fr in frames:
        (b0, s0), (b1, s1) = X.run_detector(X.ort_session(m), fr), X.run_detector(X.ort_session(simp), fr)
        assert np.allclose(b0, b1) and np.allclose(s0, s1)


@pytest.mark.gpu
def test_cuda_ep_empty_and_sorted_outputs() -> None:
    """ORT CUDA EP: N = 0 không văng (Compress của ORT CUDA 1.24 từng hỏng khi rỗng) và scores giảm dần như CPU."""
    import torch  # noqa: F401 - nạp DLL CUDA trước session ORT CUDA

    from xla.gpu_lock import gpu_lock

    frame = np.full(C.FRAME_SHAPE, 50, np.uint8)
    cases = [anchors([(100, 300, 30, 30, 0.01)]),
             anchors([(300, 300, 100, 100, 0.6), (100, 400, 40, 40, 0.9), (500, 200, 40, 40, 0.7),
                      (325, 300, 100, 100, 0.65)])]
    with gpu_lock("test_detector_export cuda ep", timeout_s=X.GPU_LOCK_TIMEOUT_S):
        for pred in cases:
            m = X.build_detector_model(fake_core(pred, 640), imgsz=640, crop_left=CROP)
            (bc, sc), (bg, sg) = X.run_detector(X.ort_session(m), frame), X.run_detector(X.ort_session(m, True), frame)
            assert bg.shape == bc.shape and sg.shape == sc.shape
            assert np.allclose(bg, bc, atol=1e-3) and np.allclose(sg, sc, atol=1e-6)
            assert np.all(np.diff(sg) <= 0)


def test_cpu_perf_warning_flags_throttle() -> None:
    ok = {"min": 95.0, "mean": 99.0, "window_s": 60, "throttled": False}
    slow = {"min": 21.8, "mean": 22.0, "window_s": 60, "throttled": True}
    assert X._cpu_perf_warning(ok) is None
    assert "hạ xung" in X._cpu_perf_warning(slow)
    assert "không đọc được" in X._cpu_perf_warning(None)


# ---- metrics + CLI ---------------------------------------------------------------------------------------------
METRICS = {"map50": 0.5, "map50_95": 0.3, "precision": 0.6, "recall": 0.4, "fp_per_neg_frame": 0.1}


def test_load_metrics_flat_and_nested(tmp_path: Path) -> None:
    flat, nested = tmp_path / "a.json", tmp_path / "b.json"
    flat.write_text(json.dumps(METRICS | {"extra": 1}), encoding="utf-8")
    nested.write_text(json.dumps({"metrics": METRICS, "other": {}}), encoding="utf-8")
    assert X.load_metrics(flat) == METRICS
    assert X.load_metrics(nested) == METRICS
    with pytest.raises(X.UsageError, match="thiếu"):
        X.check_metrics({k: v for k, v in METRICS.items() if k != "recall"})
    with pytest.raises(X.UsageError, match="không phải số"):
        X.check_metrics(METRICS | {"map50": "0.5"})


@pytest.mark.parametrize(("change", "msg"), [
    ({"weights": "missing.pt"}, "trọng số"),
    ({"out": "detector-standin"}, "detector-v1.0.0"),
    ({"version": "1.0"}, "semver"),
    ({"conf": "0.01"}, "score_floor"),
    ({"imgsz": "1000"}, "bội"),
])
def test_cli_usage_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, str],
                                 msg: str) -> None:
    (tmp_path / "w.pt").write_bytes(b"x")
    (tmp_path / "m.json").write_text(json.dumps(METRICS), encoding="utf-8")
    args = {"weights": "w.pt", "out": "detector-v1.0.0", "version": "1.0.0", "conf": "0.25", "imgsz": "1280"} | change
    argv = ["--weights", str(tmp_path / args["weights"]), "--out", str(tmp_path / args["out"]),
            "--version", args["version"], "--conf", args["conf"], "--metrics", str(tmp_path / "m.json"),
            "--train-run", "t", "--imgsz", args["imgsz"], "--json"]
    assert X.main(argv) == 2
    assert msg in capsys.readouterr().err
    assert not (tmp_path / args["out"]).exists()  # sai cách gọi thì không đụng tới thư mục ra


# ---- bundle phát hành (phase 11 tạo; đỏ tới khi có) ---------------------------------------------------------------
BUNDLE = Path("models/release/detector-v1.0.0")
E2 = Path("results/e2")
HINT = "phase 11 bước 6: python -m xla.detect.export_onnx ... --out models/release/detector-v1.0.0"


def _meta() -> dict[str, Any]:
    return json.loads((repo_root() / BUNDLE / C.META_FILE).read_text(encoding="utf-8"))


def _dicts(o: Any) -> Iterator[dict[str, Any]]:
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _dicts(v)
    elif isinstance(o, list):
        for v in o:
            yield from _dicts(v)


@pytest.mark.data
def test_release_bundle_validates(require_data: Any) -> None:
    require_data(repo_root() / BUNDLE / "detector.onnx", HINT)
    assert C.validate_bundle(repo_root() / BUNDLE) == []
    ms = _meta()["ms_per_frame"]
    assert "CUDA EP" in ms["runtime"], "bundle phát hành phải đo ms trên ORT CUDA EP (không --skip-cuda)"


@pytest.mark.data
def test_parity_within_tolerance(require_data: Any) -> None:
    p = json.loads(require_data(repo_root() / E2 / "parity.json", HINT + " --parity-out results/e2/parity.json")
                   .read_text(encoding="utf-8"))
    assert p["onnx_sha256"] == _meta()["onnx_sha256"], "parity.json không phải của detector.onnx trong bundle"
    for k in ("ort_cpu", "ort_cuda"):
        c = p["candidates"][k]
        assert c is not None, f"thiếu parity {k}"
        assert abs(c["mAP50"] - p["reference"]["mAP50"]) <= X.PARITY_TOL, (k, c["mAP50"], p["reference"]["mAP50"])
        assert c["pass"] is True
    assert p["split"] == "devval" and p["n_frames"] > 0


@pytest.mark.data
def test_meta_metrics_complete_and_match_e2(require_data: Any) -> None:
    e2 = json.loads(require_data(repo_root() / E2 / "e2.json", "phase 11 bước 4").read_text(encoding="utf-8"))
    m = _meta()["metrics"]
    keys = C.METRICS_KEYS["detector"]
    assert sorted(m) == sorted(keys)
    assert all(C._is_number(m[k]) for k in keys), f"metrics phát hành phải là số: {m}"
    found = any(all(C._is_number(d.get(k)) and abs(d[k] - m[k]) <= 1e-9 for k in keys) for d in _dicts(e2))
    assert found, "không có object nào trong results/e2/e2.json mang đúng 5 chỉ số của meta.metrics"


@pytest.mark.data
def test_golden_frames_are_dev_not_test(require_data: Any) -> None:
    root = repo_root()
    gdir = require_data(root / BUNDLE / "golden", HINT)
    split = {f["frame"]: f["split"] for f in json.loads((root / X.SPLIT_REL).read_text(encoding="utf-8"))["frames"]}
    frames_dir = root / X.FRAMES_REL
    test_sha = {hashlib.sha256((frames_dir / f).read_bytes()).hexdigest() for f, sp in split.items() if sp == "test"}
    golden_sha = {hashlib.sha256((gdir / n).read_bytes()).hexdigest() for n in C.GOLDEN_FRAMES}
    assert not golden_sha & test_sha, "golden chứa khung test"
    sources = _meta()["golden_sources"]
    assert [split.get(s) for s in sources] == ["dev"] * len(C.GOLDEN_FRAMES)
    assert golden_sha == {hashlib.sha256((frames_dir / s).read_bytes()).hexdigest() for s in sources}


@pytest.mark.gpu
@pytest.mark.data
def test_ms_per_frame_reproducible_on_cuda(require_data: Any) -> None:
    import torch  # noqa: F401 - nạp DLL CUDA trước session ORT CUDA

    from xla.gpu_lock import gpu_lock

    root = repo_root()
    onnx_path = require_data(root / BUNDLE / "detector.onnx", HINT)
    split = json.loads((root / X.SPLIT_REL).read_text(encoding="utf-8"))["frames"]
    dev = sorted(f["frame"] for f in split if f["split"] == "dev")
    frames = [C.decode_frame(root / X.FRAMES_REL / dev[int(i)]) for i in np.linspace(0, len(dev) - 1, 300)]
    with gpu_lock("test_detector_export ms", timeout_s=X.GPU_LOCK_TIMEOUT_S):
        ms = X.measure_ms(X.ort_session(onnx_path, cuda=True), frames)
    meta_ms = _meta()["ms_per_frame"]
    assert ms["p50"] <= 2 * meta_ms["p50"] + 5.0, (ms, meta_ms)
