"""Phase 12 AI v2 (tuỳ chọn): BasicVSR++ vendor, graph video §11, luật dừng probe, ``status.json`` không im lặng."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from xla import contract
from xla.enhance import sr, vsr
from xla.enhance.arch.basicvsrpp import BasicVSRPlusPlus, SecondOrderDeformableAlignment
from xla.paths import repo_root

CFG = sr.load_config("configs/vsr_x2.yaml")
STATUS = repo_root() / CFG["results_dir"] / "status.json"
BUNDLE = repo_root() / "models" / "release" / "enhancer-offline-v1.1.0"


@pytest.mark.data
def test_vendor_loads_reds4_weights_strict_with_checksum(require_data) -> None:
    require_data(Path(CFG["weights_dir"]) / CFG["model"]["weights"], CFG["model"]["url"])
    net = vsr.build_model(CFG)  # sha256 khác → ValueError; khoá lệch → strict lỗi
    with torch.no_grad():
        y = net(torch.rand(1, 3, 3, 64, 64))
    assert y.shape == (1, 3, 3, 256, 256)


def test_deform_conv_zero_offset_equals_conv() -> None:
    """Cách gọi ``deform_conv2d`` (thay mmcv): offset 0, mask 1 phải bằng conv thường."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8, 12, 12, generator=g)
    w = torch.randn(4, 8, 3, 3, generator=g)
    b = torch.randn(4, generator=g)
    off = torch.zeros(1, 2 * 9 * 2, 12, 12)
    mask = torch.ones(1, 9 * 2, 12, 12)
    assert torch.allclose(deform_conv2d(x, off, w, b, padding=(1, 1), mask=mask), F.conv2d(x, w, b, padding=1),
                          atol=1e-4)


def test_second_order_alignment_shape() -> None:
    m = SecondOrderDeformableAlignment(2 * 16, 16, 3, padding=1, deform_groups=4)
    y = m(torch.randn(1, 32, 10, 10), torch.randn(1, 48, 10, 10), torch.zeros(1, 2, 10, 10), torch.zeros(1, 2, 10, 10))
    assert y.shape == (1, 16, 10, 10)


def test_vsr_graph_shape_strip_and_middle_frame() -> None:
    net = BasicVSRPlusPlus(mid_channels=8, num_blocks=1).eval()
    g = vsr.VSRGraph(net, crop_left=16).eval()
    frames = np.random.default_rng(0).integers(0, 256, (3, *contract.FRAME_SHAPE), dtype=np.uint8)
    with torch.no_grad():
        out = g(torch.from_numpy(frames)).numpy()
    assert out.shape == contract.OFFLINE_SHAPE and out.dtype == np.uint8
    assert (out[:, :32] == out[:, 32:33]).all()


def test_custom_symbolic_exports_deformconv(tmp_path: Path) -> None:
    import onnx

    vsr.register_deform_conv_onnx(19)
    g = vsr.VSRGraph(BasicVSRPlusPlus(mid_channels=8, num_blocks=1), crop_left=16).eval()
    x = torch.zeros(3, *contract.FRAME_SHAPE, dtype=torch.uint8)
    out = tmp_path / "t.onnx"
    with torch.no_grad():
        torch.onnx.export(g, (x,), str(out), opset_version=19, input_names=[contract.FRAMES_NAME],
                          output_names=["frame_out"], dynamo=False)
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    assert "DeformConv" in {n.op_type for n in m.graph.node}


@pytest.mark.parametrize(("m", "want"), [
    ({"onnx_export_ok": True, "ort_cpu_ok": False, "ort_cuda_ok": False}, "onnx"),
    ({"onnx_export_ok": False}, "onnx"),
    ({"onnx_export_ok": False, "torch_fp16_error": "OutOfMemoryError: CUDA out of memory"}, "vram"),
    ({"onnx_export_ok": True, "ort_cpu_ok": True, "ort_cuda_ok": True, "ort_cuda_peak_vram_gb": 7.5,
      "ort_cuda_s_per_frame": 0.2}, "vram"),
    ({"onnx_export_ok": True, "ort_cpu_ok": True, "ort_cuda_ok": True, "ort_cuda_peak_vram_gb": 3.0,
      "ort_cuda_s_per_frame": 1.4}, "speed"),
    ({"onnx_export_ok": True, "ort_cpu_ok": True, "ort_cuda_ok": True, "ort_cuda_peak_vram_gb": 3.0,
      "ort_cuda_s_per_frame": 0.6}, None),
])
def test_probe_verdict(m: dict, want: str | None) -> None:
    assert vsr.verdict(m, CFG["probe"]["kill"]) == want


@pytest.mark.data
def test_status_records_result_or_kill_with_measurements() -> None:
    assert STATUS.is_file(), "phase 12 phải ghi results/v2/vsr/status.json (kết quả hoặc killed kèm số đo)"
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert st.get("measurements"), "status.json thiếu số đo"
    assert st.get("gpu_hours_used_total", 0) <= float(CFG["gpu_budget_hours"])
    if st.get("killed"):
        assert st["killed"] in vsr.KILL_REASONS
        assert not BUNDLE.exists(), "đã dừng thì không được có bundle video"
        if st["killed"] == "onnx":
            m = st["measurements"]
            assert not (m["onnx_export_ok"] and m["ort_cpu_ok"] and m["ort_cuda_ok"])
            assert m.get("onnx_export_error") or m.get("ort_cpu_error") or m.get("ort_cuda_error")
    else:
        assert contract.validate_bundle(BUNDLE) == []
        meta = json.loads((BUNDLE / "meta.json").read_text(encoding="utf-8"))
        assert meta["temporal_window"] == 5
