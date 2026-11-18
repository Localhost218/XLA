"""Phase 16 — hai model thật chạy trong IOT-CV trên bản ghi 02/11: ngân sách ms, golden GPU, khung/s.

Đọc file bench MỚI NHẤT của IOT-CV (`scripts/vision/replay_bench.py` ghi `logs/vision-bench/*.json`).
Thiếu file → FAIL. Ngân sách giữ đúng plan (không nới theo số đo): tổng p95 ≤ 50 ms, detector p95 ≤ 30,
enhancer p95 ≤ 30 (luồng có người xem "Ảnh rõ"); khung/s xử lý được (hết tốc lực) ≥ 6; golden GPU đạt;
sha256 trong bench = `meta.onnx_sha256` của bundle XLA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xla.paths import iotcv_root, repo_root

pytestmark = [pytest.mark.data, pytest.mark.gpu]

BUDGET = {"total": 50.0, "detector": 30.0, "enhancer": 30.0}
MIN_FPS = 6.0


@pytest.fixture
def bench(require_data) -> dict:
    d = require_data(Path(iotcv_root()) / "logs" / "vision-bench", "chạy replay_bench.py ở IOT-CV")
    files = sorted(d.glob("*.json"))
    if not files:
        pytest.fail(f"không có file bench trong {d}", pytrace=False)
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    data["_file"] = files[-1].name
    return data


def test_sha256_khop_bundle_xla(bench):
    for model in ("detector", "enhancer"):
        meta = json.loads((repo_root() / "models" / "release" / f"{model}-v1.0.0" / "meta.json").read_text("utf-8"))
        assert bench[model]["onnx_sha256"] == meta["onnx_sha256"], model


def test_golden_tren_gpu(bench):
    assert bench["gpu"] is True, bench["_file"]
    for model, g in bench["golden"].items():
        assert g["pass"] is True, model
        assert g["providers"][0] == "CUDAExecutionProvider", model


def test_ngan_sach_ms(bench):
    ms = bench["realtime"]["ms"]
    over = {k: ms[k]["p95"] for k, v in BUDGET.items() if ms[k]["p95"] > v}
    assert not over, (
        f"{bench['_file']}: p95 vượt ngân sách {over} (ngân sách {BUDGET}); điều kiện máy lúc đo "
        f"{bench.get('machine')}. Không nới ngân sách: đo lại khi GPU không hạ xung hoặc giảm tải model."
    )


def test_khung_moi_giay(bench):
    assert bench["maxrate"]["fps"] >= MIN_FPS, bench["maxrate"]
    assert bench["realtime"]["errors"] == 0
