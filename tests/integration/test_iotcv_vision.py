"""Phase 15 — IOT-CV nạp bundle theo hợp đồng: test của IOT-CV xanh + hai bản cài hợp đồng đồng ý.

Hai bản cài độc lập của `docs/hop-dong-model.md`: `xla.contract` (XLA dựng/kiểm bundle) và
`backend/vision/model_bundle.py` (IOT-CV nạp). Bundle giả do XLA dựng phải qua được bộ nạp của
IOT-CV (`scripts/vision/kiem_bundle.py`). Thiếu repo IOT-CV → FAIL (marker `data`, CI loại).
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from xla.contract import make_dummy_bundle
from xla.paths import iotcv_root

pytestmark = pytest.mark.data


def _iotcv(require_data):
    root = require_data(iotcv_root(), "repo IOT-CV (configs/paths.yaml: iotcv_root)")
    require_data(root / "backend" / "vision" / "model_bundle.py", "IOT-CV chưa có code phase 15")
    return root


def _uv(root, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """Chạy trong venv CỦA IOT-CV. Bỏ UV_PROJECT_ENVIRONMENT / VIRTUAL_ENV của XLA khỏi env: giữ
    lại thì `uv run --project IOT-CV` dựng lại venv XLA theo Python 3.13 của IOT-CV và xoá sạch nó
    (đã xảy ra 19/11/2026, phải `uv sync --locked` lại)."""
    env = {k: v for k, v in os.environ.items() if k not in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")}
    cmd = ["uv", "run", "--project", str(root), "--extra", "dev", "--extra", "vision", *args]
    return subprocess.run(cmd, check=False, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)


def test_test_vision_cua_iotcv_xanh(require_data):
    root = _iotcv(require_data)
    r = _uv(
        root, "pytest", "backend/tests/test_vision_ai.py", "backend/tests/test_fake_stream.py", "-p", "no:cacheprovider"
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


@pytest.mark.parametrize("model", ["detector", "enhancer"])
def test_bundle_gia_cua_xla_qua_bo_nap_iotcv(require_data, tmp_path, model):
    root = _iotcv(require_data)
    bundle = make_dummy_bundle(tmp_path, model)
    r = _uv(root, "python", "scripts/vision/kiem_bundle.py", str(bundle), "--cpu", "--json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["pass"] is True and out["model"] == model and out["golden"]
