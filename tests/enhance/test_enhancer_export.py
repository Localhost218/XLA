"""Phase 13 bước 6: bundle ``enhancer-v1.0.0`` đúng hợp đồng §3–§4, ``speed.json`` đủ ba ứng viên + giấy phép.

Test không marker chạy được ở CI (CPU, không dữ liệu): bọc ``EnhancerGraph`` xuất ONNX ra đúng chữ ký và dải mép.
Test ``data`` cần bundle đã dựng (``python -m xla.enhance.export_onnx ...``) và khung thật; thiếu thì FAIL, không skip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xla import contract
from xla.enhance.net import CANDIDATES, EnhancerGraph, RestoreNet
from xla.paths import repo_root

BUNDLE = repo_root() / "models" / "release" / "enhancer-v1.0.0"
SPEED_JSON = repo_root() / "results" / "restore" / "speed.json"
SPLIT = repo_root() / "data" / "real" / "split.json"
MS_P95_MAX = 30.0


class _Tiny(torch.nn.Module):
    """Mạng nhỏ thay mạng thật: một conv 3×3 khởi tạo gần đồng nhất + dịch sáng (output khác input)."""

    def __init__(self) -> None:
        super().__init__()
        self.c = torch.nn.Conv2d(3, 3, 3, padding=1)
        with torch.no_grad():
            self.c.weight.zero_()
            for i in range(3):
                self.c.weight[i, i, 1, 1] = 0.9
            self.c.bias.fill_(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 256, contract.FRAME_SHAPE, dtype=np.uint8)
    f[:, :8] = 10  # dải tối giả ở mép trái: không được lọt vào ảnh ra
    return f


def test_graph_wrapper_contract(tmp_path: Path) -> None:
    import onnxruntime as ort

    crop = 16
    graph = EnhancerGraph(RestoreNet("tiny", _Tiny()), crop, half=False).eval()
    onnx_path = tmp_path / "enhancer.onnx"
    dummy = torch.zeros(contract.FRAME_SHAPE, dtype=torch.uint8)
    torch.onnx.export(graph, (dummy,), str(onnx_path), opset_version=17, input_names=["frame"],
                      output_names=["frame_out"], dynamo=False)
    errs = contract._Errors()
    assert contract._check_signature(onnx_path, "enhancer", errs), errs.as_strings()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    f = _frame()
    out = sess.run(None, {"frame": f})[0]
    assert out.dtype == np.uint8 and out.shape == contract.FRAME_SHAPE
    assert np.array_equal(out[:, :crop], np.repeat(out[:, crop : crop + 1], crop, axis=1))
    with torch.no_grad():
        ref = graph(torch.from_numpy(f)).numpy()
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    # kênh BGR giữ đúng thứ tự: tăng riêng kênh B của vào thì kênh B của ra tăng
    f2 = f.copy()
    f2[..., 0] = np.clip(f2[..., 0].astype(int) + 40, 0, 255).astype(np.uint8)
    out2 = sess.run(None, {"frame": f2})[0]
    d = out2[:, crop:].astype(int) - out[:, crop:].astype(int)
    assert d[..., 0].mean() > 10 and abs(d[..., 1].mean()) < 1 and abs(d[..., 2].mean()) < 1


def test_candidates_registry_has_three() -> None:
    assert set(CANDIDATES) == {"nafnet", "fbcnn", "scunet"}


@pytest.mark.data
def test_speed_json_three_candidates_with_license(require_data) -> None:
    rep = json.loads(require_data(SPEED_JSON, "python -m xla.enhance.net speed").read_text(encoding="utf-8"))
    rows = {r["candidate"]: r for r in rep["candidates"]}
    assert set(rows) == {"nafnet", "fbcnn", "scunet"}
    for r in rows.values():
        lic = r["license"]
        assert lic.get("code") and lic.get("weights"), r["candidate"]
        assert r["ms"]["p95"] > 0
    if rep["chosen"] is not None:
        assert rows[rep["chosen"]]["ms"]["p95"] <= rep["budget_p95_ms"]


@pytest.mark.data
def test_bundle_validates(require_data) -> None:
    require_data(BUNDLE / "enhancer.onnx", "dựng bằng python -m xla.enhance.export_onnx")
    assert contract.validate_bundle(BUNDLE) == []


@pytest.mark.data
def test_bundle_ms_p95(require_data) -> None:
    meta = json.loads(require_data(BUNDLE / "meta.json").read_text(encoding="utf-8"))
    assert 0 < meta["ms_per_frame"]["p95"] <= MS_P95_MAX


@pytest.mark.data
def test_bundle_left_strip_repeats_crop_column(require_data) -> None:
    require_data(BUNDLE / "enhancer.onnx")
    meta = json.loads((BUNDLE / "meta.json").read_text(encoding="utf-8"))
    cl = meta["crop_left"]
    frame = contract.decode_frame(BUNDLE / "golden" / "000.jpg")
    out = contract.run_model(BUNDLE / "enhancer.onnx", {"f": frame})["f"]["frame_out"]
    assert out.shape == contract.FRAME_SHAPE and out.dtype == np.uint8
    assert np.array_equal(out[:, :cl], np.repeat(out[:, cl : cl + 1], cl, axis=1))


@pytest.mark.data
def test_golden_not_from_test_split(require_data) -> None:
    split = json.loads(require_data(SPLIT).read_text(encoding="utf-8"))
    meta = json.loads(require_data(BUNDLE / "meta.json").read_text(encoding="utf-8"))
    by_name = {f["frame"]: f["split"] for f in split["frames"]}
    sources = meta["golden_sources"]
    assert sorted(sources) == sorted(contract.GOLDEN_FRAMES)
    assert all(by_name.get(src) == "dev" for src in sources.values()), sources
    # đối chiếu theo nội dung: byte golden không trùng khung test nào
    fdir = repo_root() / "data" / "real" / "frames"
    test_hashes = {
        hashlib.sha256((fdir / n).read_bytes()).hexdigest() for n, s in by_name.items() if s == "test"
    }
    for g in contract.GOLDEN_FRAMES:
        assert hashlib.sha256((BUNDLE / "golden" / g).read_bytes()).hexdigest() not in test_hashes, g
