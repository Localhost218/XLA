"""Cấu hình pytest dùng chung cho XLA.

Marker:

- ``gpu``  — cần GPU CUDA thật. CI loại bằng ``-m "not gpu"``.
- ``data`` — cần dữ liệu local (khung bay thật, dataset ở ổ D). CI loại bằng ``-m "not data"``. Chạy local mà thiếu
  dữ liệu thì test **fail** (không skip), để không có cái xanh giả: dùng fixture ``require_data``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: cần GPU CUDA thật; CI chạy -m 'not gpu'")
    config.addinivalue_line("markers", "data: cần dữ liệu local; thiếu thì FAIL, CI loại bằng -m 'not data'")


@pytest.fixture
def require_data() -> Callable[..., Path]:
    """Trả hàm ``require_data(path, hint="")``: đường dẫn không tồn tại thì ``pytest.fail`` (không skip)."""

    def _require(path: str | Path, hint: str = "") -> Path:
        p = Path(path)
        if not p.exists():
            pytest.fail(f"thiếu dữ liệu local {p}" + (f" ({hint})" if hint else ""), pytrace=False)
        return p

    return _require
