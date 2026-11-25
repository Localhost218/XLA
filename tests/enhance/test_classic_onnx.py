"""AI v2 phase 10: ``enhancer-v2.0.0`` = ``xla.enhance.classic`` xuất ONNX, parity đo trong ORT.

Test không marker chạy ở CI (CPU, không dữ liệu): xuất graph từ ``configs/enhance_classic.yaml`` trên khung tổng hợp,
đúng chữ ký §3, parity ORT CPU với numpy trên khung KHÁC khung trace và phủ cả hai phía của mọi nhánh dữ liệu, graph
ghim cứng nhánh thì test parity đỏ, đổi tham số yaml mà không xuất lại thì đỏ.
Test ``data`` cần bundle đã dựng (``python -m xla.enhance.export_classic --out models/release/enhancer-v2.0.0``) và
khung thật; thiếu thì FAIL, không skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from xla import contract
from xla.enhance import classic as C
from xla.enhance import export_classic as E
from xla.paths import repo_root

BUNDLE = repo_root() / "models" / "release" / "enhancer-v2.0.0"
REPORT = repo_root() / E.REPORT_REL
H, W = contract.FRAME_H, contract.FRAME_W


def synth(kind: str, seed: int = 0) -> np.ndarray:
    """Khung tổng hợp 480×640 BGR: ``haze`` (mờ sáng + chấm tối → nhánh kéo giãn mức đen), ``dark`` (tối, cũng kéo
    giãn), ``pink`` (vùng hồng bão hoà → cổng khử hồng mở, không kéo giãn mức đen)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    if kind == "haze":
        base = 170 + 20 * np.sin(xx / 37.0) + 15 * np.cos(yy / 23.0)
        f = np.stack([base, base + 5, base - 5], -1) + rng.normal(0, 3, (H, W, 3))
        dots = rng.random((H, W)) < 0.02
        f[dots] = rng.uniform(25, 45, (int(dots.sum()), 1))
    elif kind == "dark":
        base = 40 + 25 * np.sin(xx / 41.0) * np.cos(yy / 29.0)
        f = np.stack([base, base + 3, base + 6], -1) + rng.normal(0, 3, (H, W, 3))
    elif kind == "pink":
        base = 120 + 30 * np.sin(xx / 31.0)
        f = np.stack([base, base, base], -1) + rng.normal(0, 3, (H, W, 3))
        f[100:180, 200:330] = (245, 200, 250)
    else:
        raise ValueError(kind)
    f[:, :8] = 10  # dải tối giả ở mép trái
    return np.clip(f, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def params() -> dict:
    return C.load_params()


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory: pytest.TempPathFactory, params: dict) -> Path:
    p = tmp_path_factory.mktemp("classic") / "enhancer.onnx"
    E.export_graph(params, p, synth("pink", seed=99))  # trace trên khung không dùng để đo
    return p


def _branches(frame: np.ndarray, params: dict) -> tuple[bool, bool]:
    st = E.build_stages(params, H, W - int(params["crop_left"]))
    si = E.stage_inputs([frame], params)
    with torch.no_grad():
        black = bool(st["dehaze"].black_branch(torch.from_numpy(si["in"]["dehaze"][0]).float()))
        gate = bool(st["magenta"].gate_active(torch.from_numpy(si["in"]["magenta"][0]).float()))
    return black, gate


def _within(got: np.ndarray, ref: np.ndarray) -> tuple[bool, float, int]:
    mae, mx, _ = E._diff(got, ref)
    return mae <= E.PARITY_MAE_MAX and mx <= E.PARITY_ABS_MAX, mae, mx


def test_stage_order_matches_classic() -> None:
    assert tuple(n for n, _ in C.STAGES) == E.STAGE_NAMES


@pytest.mark.parametrize("q", [0.5, 25.0, 50.0, 75.0, 99.9])
def test_percentile_topk_matches_numpy(q: float) -> None:
    rng = np.random.default_rng(int(q * 10))
    for dtype in (np.uint8, np.float32):
        a = rng.integers(0, 256, 18720).astype(dtype) if dtype is np.uint8 else rng.normal(50, 30, 18720).astype(dtype)
        got = float(E._percentile(torch.from_numpy(a.astype(np.float32)), q, a.size))
        assert got == pytest.approx(float(np.percentile(a, q)), abs=1e-3)


@pytest.mark.parametrize(("sigma", "depth_8u"), [(2.0, True), (1.2, False)])
def test_sep_filter_matches_cv2_gaussian(sigma: float, depth_8u: bool) -> None:
    """Lọc Gauss của graph = ``cv2.GaussianBlur(ksize=(0, 0), σ)`` (cạnh nhân theo depth, biên phản chiếu); bản 8U của
    cv2 tính dấu phẩy tĩnh nên so sau làm tròn, lệch ≤ 1 mức."""
    import cv2

    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (120, 156)).astype(np.uint8)
    x = img if depth_8u else (img.astype(np.float32) * np.float32(0.731))
    with torch.no_grad():
        got = E._sep_filter(torch.from_numpy(x.astype(np.float32))[None], E.gauss_kernel(sigma, depth_8u))[0].numpy()
    ref = cv2.GaussianBlur(x, (0, 0), sigma)
    if depth_8u:
        assert np.abs(np.round(got) - ref.astype(np.float32)).max() <= 1
    else:
        assert np.abs(got - ref).max() < 1e-3


def test_graph_io_contract(onnx_path: Path, params: dict) -> None:
    import onnx
    import onnxruntime as ort

    errs = contract._Errors()
    assert contract._check_signature(onnx_path, "enhancer", errs), errs.as_strings()
    m = onnx.load(str(onnx_path))
    assert [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")] == [contract.OPSET]
    ops = {n.op_type for n in m.graph.node}
    assert not ops & {"If", "Loop", "Scan"}, f"graph còn luồng điều khiển: {ops & {'If', 'Loop', 'Scan'}}"
    assert {"Where", "TopK"} <= ops  # nhánh mức đen = Where, phân vị = TopK
    assert not ops & {"Conv", "MatMul", "Gemm"}, "Conv/MatMul có thể chạy TF32 trên ORT CUDA EP: lệch làm tròn uint8"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"frame": synth("haze")})[0]
    c = int(params["crop_left"])
    assert out.dtype == np.uint8 and out.shape == contract.FRAME_SHAPE
    assert np.array_equal(out[:, :c], np.repeat(out[:, c : c + 1], c, axis=1))


def test_parity_both_branches_vs_numpy(onnx_path: Path, params: dict) -> None:
    """Parity ORT CPU ↔ ``classic.enhance`` trên khung khác khung trace, phủ cả hai phía của nhánh mức đen
    (``if lo_out < floor``) và của cổng khử hồng (``countNonZero``)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    seen: set[tuple[bool, bool]] = set()
    for kind in ("haze", "dark", "pink"):
        f = synth(kind, seed=1)
        seen.add(_branches(f, params))
        ok, mae, mx = _within(sess.run(None, {"frame": f})[0], C.enhance(f, params))
        assert ok, f"{kind}: MAE {mae:.4f} (≤ {E.PARITY_MAE_MAX}), lớn nhất {mx} (≤ {E.PARITY_ABS_MAX})"
    assert {b for b, _ in seen} == {True, False}, f"khung thử không phủ hai phía nhánh mức đen: {seen}"
    assert {g for _, g in seen} == {True, False}, f"khung thử không phủ hai phía cổng khử hồng: {seen}"


def test_hard_traced_branch_is_caught(tmp_path: Path, params: dict) -> None:
    """Graph có nhánh mức đen bị ghim thành hằng (như Python ``if`` lúc trace) → parity trên khung nhánh kia ĐỎ.
    Khung ``dark``: mức đen ra thấp hẳn dưới sàn nên hai nhánh lệch rõ (khung ``haze`` lệch ít, MAE ~0,8)."""
    import onnxruntime as ort

    f = synth("dark", seed=1)
    black, _ = _branches(f, params)
    assert black
    g = E.ClassicGraph(params)
    g.stages["dehaze"].force_branch = not black
    p = tmp_path / "hard.onnx"
    E.export_onnx(g, torch.from_numpy(synth("pink", seed=99)), p, "frame", "frame_out")
    out = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"frame": f})[0]
    ok, mae, mx = _within(out, C.enhance(f, params))
    assert not ok, f"graph ghim nhánh vẫn qua parity (MAE {mae:.4f}, max {mx}): test không bắt được trace cứng"


def test_param_change_without_reexport_fails(tmp_path: Path, params: dict) -> None:
    src = repo_root() / E.DEFAULT_PARAMS_REL
    meta = {"network": "classic", "classic_params_sha256": E.file_sha256(src),
            "camera_profile_sha256": E.file_sha256(E.camera_profile_path(src))}
    assert E.stale_reasons(meta, src) == []
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["dehaze"]["omega"] = 0.6
    changed = tmp_path / "enhance_classic.yaml"
    changed.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert any("classic_params_sha256" in r for r in E.stale_reasons(meta, changed))
    # graph dựng từ tham số cũ lệch numpy chạy tham số mới quá dung sai: golden/parity cũng đỏ
    with torch.no_grad():
        old = E.ClassicGraph(params)(torch.from_numpy(synth("haze", seed=2))).numpy()
    assert not _within(old, C.enhance(synth("haze", seed=2), C.load_params(changed)))[0]


# ----------------------------------------------------------------------------- bundle thật (data)

def _meta(require_data) -> dict:
    require_data(BUNDLE / "enhancer.onnx", "python -m xla.enhance.export_classic --out models/release/enhancer-v2.0.0")
    return json.loads((BUNDLE / contract.META_FILE).read_text(encoding="utf-8"))


@pytest.mark.data
def test_bundle_valid_and_current(require_data) -> None:
    meta = _meta(require_data)
    assert contract.validate_bundle(BUNDLE) == []
    assert meta["network"] == "classic" and meta["version"] == "2.0.0"
    assert E.stale_reasons(meta) == [], "đổi configs/enhance_classic.yaml hoặc hồ sơ camera mà chưa xuất lại bundle"


@pytest.mark.data
def test_bundle_golden_parity_vs_numpy(require_data, params: dict) -> None:
    import onnxruntime as ort

    meta = _meta(require_data)
    assert meta["trace_frame"] not in meta["golden_sources"].values()
    sess = ort.InferenceSession(str(BUNDLE / "enhancer.onnx"), providers=["CPUExecutionProvider"])
    for name in contract.GOLDEN_FRAMES:
        f = contract.decode_frame(BUNDLE / "golden" / name)
        ok, mae, mx = _within(sess.run(None, {"frame": f})[0], C.enhance(f, params))
        assert ok, f"{name}: MAE {mae:.4f}, lớn nhất {mx}"


@pytest.mark.data
def test_report_parity_per_stage(require_data) -> None:
    _meta(require_data)
    rep = json.loads(require_data(REPORT).read_text(encoding="utf-8"))
    fr = rep["frames"]
    assert len(fr["parity"]) == 50 and fr["trace"] not in fr["parity"] and not set(fr["golden"]) & set(fr["parity"])
    for prov in ("ort_cpu", "ort_cuda"):
        stages = rep["parity"]["stages"][prov]
        assert tuple(stages) == E.STAGE_NAMES
        bad = {k: v for k, v in stages.items() if not v["pass"]}
        assert not bad, f"{prov}: tầng lệch quá dung sai {bad}"
    assert all(v["pass"] for v in rep["parity"]["full"].values())
    assert rep["errors"] == [] and rep["parity_ok"]
