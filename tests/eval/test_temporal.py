"""xla.eval.temporal: nhóm chuỗi liền kề + tỉ lệ nhấp nháy (CPU, ảnh tổng hợp)."""

from __future__ import annotations

import numpy as np
import pytest

from xla.eval.temporal import consecutive_runs, flicker_ratio, mean_abs_diff


def _seq(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(n)]


def test_identity_is_one() -> None:
    s = _seq(5)
    assert flicker_ratio(s, s, crop_left=16) == pytest.approx(1.0)


def test_multi_runs_and_amplified() -> None:
    a, b = _seq(4, 1), _seq(3, 2)
    # nhân đôi độ lệch quanh 128 → |Δ| gấp đôi (trong khoảng không bão hoà)
    half = [(x // 4 + 96).astype(np.uint8) for x in a]
    dbl = [((x.astype(np.int16) - 128) * 2 + 128).astype(np.uint8) for x in half]
    assert flicker_ratio([half, b], [dbl, b], crop_left=0) > 1.0
    assert mean_abs_diff(dbl, 0) == pytest.approx(2 * mean_abs_diff(half, 0), rel=1e-6)


def test_crop_left_excluded() -> None:
    s = _seq(3)
    t = [x.copy() for x in s]
    for i, x in enumerate(t):
        x[:, :16] = 255 * (i % 2)  # chỉ cột mép nhấp nháy
    assert flicker_ratio(s, t, crop_left=16) == pytest.approx(1.0)


def test_errors() -> None:
    s = _seq(3)
    with pytest.raises(ValueError):
        flicker_ratio(s, s[:2])
    with pytest.raises(ValueError):
        flicker_ratio([s[0], s[0]], [s[0], s[1]])  # gốc đứng yên
    with pytest.raises(ValueError):
        mean_abs_diff(s[:1])


def test_consecutive_runs() -> None:
    fr = [
        {"frame": "a", "t_unix": 0.0, "split": "test"},
        {"frame": "b", "t_unix": 0.2, "split": "test"},
        {"frame": "c", "t_unix": 0.35, "split": "test"},
        {"frame": "d", "t_unix": 1.0, "split": "test"},  # khoảng hở ≥ 250 ms
        {"frame": "e", "t_unix": 1.1, "split": "test"},
        {"frame": "x", "t_unix": 0.1, "split": "dev"},
        {"frame": "z", "t_unix": 5.0, "split": "test"},  # lẻ → bỏ
    ]
    assert consecutive_runs(fr, split="test") == [["a", "b", "c"], ["d", "e"]]
    assert consecutive_runs(fr, split="dev") == []
