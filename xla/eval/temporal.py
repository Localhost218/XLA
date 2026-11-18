"""Đo theo thời gian trên chuỗi khung liền kề: nhấp nháy sau xử lý (phase 13 chọn checkpoint, phase 14 chấm).

SSOT cho cả hai lane, không chép lại chỗ khác.

- ``consecutive_runs``: nhóm khung của ``split.json`` thành các chuỗi liền kề (cùng split, cách nhau < ``max_gap_s``).
- ``flicker_ratio``: trung bình |Δ| giữa khung liền kề của ảnh đã xử lý chia cho của ảnh gốc, bỏ cột ``[0, crop_left)``.
  1,0 = nhấp nháy không đổi; > 1 = tăng.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from xla.camera.measure import CROP_LEFT_DEFAULT

MAX_GAP_S = 0.25  # plan phase 14 bước 3: khung liên tiếp cách nhau < 250 ms


def consecutive_runs(frames: Iterable[dict[str, Any]], split: str | None = None, max_gap_s: float = MAX_GAP_S,
                     min_len: int = 2) -> list[list[str]]:
    """Nhóm bản ghi ``{"frame", "t_unix", "split"}`` thành chuỗi tên khung liền kề, theo thứ tự thời gian.

    Hai khung cùng chuỗi khi cùng ``split`` và cách nhau ``0 < Δt < max_gap_s``. Chuỗi ngắn hơn ``min_len`` bị bỏ.
    """
    rows = [f for f in frames if split is None or f.get("split") == split]
    rows.sort(key=lambda f: (f.get("split"), float(f["t_unix"])))
    runs: list[list[str]] = []
    cur: list[str] = []
    prev: dict[str, Any] | None = None
    for f in rows:
        if prev is not None and prev.get("split") == f.get("split") and 0 < float(f["t_unix"]) - float(prev["t_unix"]) < max_gap_s:
            cur.append(f["frame"])
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = [f["frame"]]
        prev = f
    if len(cur) >= min_len:
        runs.append(cur)
    return runs


def mean_abs_diff(frames: Sequence[np.ndarray], crop_left: int = CROP_LEFT_DEFAULT) -> float:
    """Trung bình |khung[i+1] − khung[i]| (mọi kênh, float) trên cột ``[crop_left, W)``."""
    if len(frames) < 2:
        raise ValueError(f"cần ≥ 2 khung, có {len(frames)}")
    shape = frames[0].shape
    total, count = 0.0, 0
    for a, b in pairwise(frames):
        if a.shape != shape or b.shape != shape:
            raise ValueError(f"khung lệch cỡ: {a.shape} / {b.shape}, mong {shape}")
        d = np.abs(b[:, crop_left:].astype(np.float32) - a[:, crop_left:].astype(np.float32))
        total += float(d.sum())
        count += d.size
    return total / count


def flicker_ratio(raw: Sequence[Sequence[np.ndarray]] | Sequence[np.ndarray],
                  processed: Sequence[Sequence[np.ndarray]] | Sequence[np.ndarray],
                  crop_left: int = CROP_LEFT_DEFAULT) -> float:
    """Tỉ lệ nhấp nháy ``Σ|Δ processed| / Σ|Δ raw|`` gộp theo cặp khung trên một hoặc nhiều chuỗi.

    Nhận một chuỗi (list ảnh) hoặc nhiều chuỗi (list các list ảnh); ``raw`` và ``processed`` phải cùng cấu trúc.
    Lỗi (ValueError) khi lệch số khung, chuỗi < 2 khung, hoặc ảnh gốc đứng yên tuyệt đối (mẫu số 0).
    """
    raw_runs = [raw] if isinstance(raw[0], np.ndarray) else list(raw)
    out_runs = [processed] if isinstance(processed[0], np.ndarray) else list(processed)
    if len(raw_runs) != len(out_runs):
        raise ValueError(f"số chuỗi lệch: gốc {len(raw_runs)}, xử lý {len(out_runs)}")
    num = den = 0.0
    for r, p in zip(raw_runs, out_runs, strict=True):
        if len(r) != len(p):
            raise ValueError(f"số khung lệch trong chuỗi: gốc {len(r)}, xử lý {len(p)}")
        pairs = len(r) - 1
        den += mean_abs_diff(r, crop_left) * pairs
        num += mean_abs_diff(p, crop_left) * pairs
    if den == 0:
        raise ValueError("ảnh gốc không đổi giữa các khung: không chia được tỉ lệ nhấp nháy")
    return num / den
