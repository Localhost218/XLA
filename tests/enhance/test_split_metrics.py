"""``xla.enhance.split_metrics``: chỉ số test của bundle enhancer — dựng khối test đúng như ``split.build``, hàm đồng nhất
cho mức giảm 0 và flicker 1 (định nghĩa ``train_net.score_of`` + ``temporal.flicker_ratio``)."""

from __future__ import annotations

import numpy as np
import pytest

from xla import contract
from xla.enhance import split_metrics as SM


def _frame(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, contract.FRAME_SHAPE, dtype=np.uint8)


def test_block_rows_takes_every_flight_frame_of_test_blocks_only() -> None:
    man = [{"frame": f"f{i}", "kind": "flight", "flight_id": "A", "t_unix": str(100.0 + i)} for i in range(90)]
    man.append({"frame": "n0", "kind": "neg", "flight_id": "", "t_unix": "130"})
    split = {"block_s": 30, "frames": [{"frame": "f35", "t_unix": 135.0, "split": "test"},
                                       {"frame": "f5", "t_unix": 105.0, "split": "dev"}]}
    rows = SM.block_rows(split, man, {"A": 100.0}, 30.0)
    assert [r["frame"] for r in rows] == [f"f{i}" for i in range(30, 60)]  # khối k = 1, không neg, không khối dev
    assert {r["split"] for r in rows} == {SM.TEST_BLOCK_SPLIT}
    with pytest.raises(SM.MetricsError):
        SM.block_rows({"frames": [{"frame": "zz", "t_unix": 1.0, "split": "test"}]}, man, {"A": 100.0}, 30.0)


def test_identity_enhancer_scores_zero_drop_and_unit_flicker() -> None:
    frames = {f"k{i}": _frame(i) for i in range(6)}
    res = SM.measure(lambda f: f, frames.__getitem__, ["k0", "k3"], [["k0", "k1", "k2"], ["k3", "k4", "k5"]], 16)
    m = res["metrics"]
    assert set(m) == set(contract.METRICS_KEYS["enhancer"])
    assert m["pink_frac_drop"] == pytest.approx(0.0) and m["dark_channel_drop"] == pytest.approx(0.0)
    assert m["flicker_ratio"] == pytest.approx(1.0)
    assert res["detail"]["n_frames_stats"] == 2 and res["detail"]["n_frames_flicker"] == 6
    dark = SM.measure(lambda f: (f // 2).astype(np.uint8), frames.__getitem__, ["k0"], [["k0", "k1"]], 16)["metrics"]
    assert dark["dark_channel_drop"] > 0.3 and dark["flicker_ratio"] == pytest.approx(0.5, abs=0.02)
    with pytest.raises(SM.MetricsError):
        SM.measure(lambda f: f, frames.__getitem__, ["k0"], [], 16)
