"""Hình học chung D0/D1/D2: thu nhỏ theo cỡ người thật, cắt cửa sổ 624×480, quy đổi hộp nhãn.

Một ảnh nguồn + một seed → cùng một cửa sổ cho cả ba tập (so sánh công bằng). Hạ mẫu bằng ``INTER_AREA`` **trước**
mọi khâu làm xấu (``nghien-cuu-tai-lieu.md`` §1 "Về GSD"). Kết quả 624×480; dải mép trái 16 cột do từng tập dán sau.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

WIN_W, WIN_H = 624, 480          # khung 640×480 trừ 16 cột mép trái (contract §1, crop_left = 16)
FRAME_W, FRAME_H = 640, 480
CROP_LEFT = FRAME_W - WIN_W


@dataclass
class GeomConfig:
    neg_window_frac: float = 0.2      # tỉ lệ cửa sổ không người (phase 05 bước 3)
    min_keep_area_frac: float = 0.6   # hộp bị cắt còn < 60 % diện tích thì bỏ
    min_box_px: float = 2.0           # cạnh ngắn hộp sau thu nhỏ dưới ngưỡng này thì bỏ (không còn là người thấy được)
    max_tries: int = 40               # số lần thử đặt cửa sổ tránh vùng ignored
    allow_upscale: bool = False       # không phóng to ảnh nguồn (bịa chi tiết)


@dataclass
class Window:
    scale_x: float
    scale_y: float
    x0: int
    y0: int
    kind: str                         # "person" | "negative"
    target_short_px: float | None
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy trong cửa sổ 624×480
    dropped: int = 0


def sample_dist(dist: dict, rng: np.random.Generator) -> float:
    """Lấy mẫu một đại lượng mô tả bằng phân vị ``{p5, p50, p95[, min, max]}`` (nội suy tuyến tính hàm phân vị).

    Hai đuôi ngoài p5/p95 kéo dài tuyến tính tới ``min``/``max`` nếu có, không thì giữ bằng p5/p95.
    """
    for k in ("p5", "p50", "p95"):
        if k not in dist or dist[k] is None:
            raise KeyError(f"phân bố thiếu khoá {k}: {dist}")
    qs = [0.05, 0.5, 0.95]
    vs = [float(dist["p5"]), float(dist["p50"]), float(dist["p95"])]
    lo = float(dist["min"]) if dist.get("min") is not None else vs[0]
    hi = float(dist["max"]) if dist.get("max") is not None else vs[2]
    qs = [0.0, *qs, 1.0]
    vs = [min(lo, vs[0]), *vs, max(hi, vs[2])]
    return float(np.interp(rng.random(), qs, vs))


def _intersects(win: tuple[int, int, int, int], boxes: np.ndarray) -> bool:
    if len(boxes) == 0:
        return False
    x0, y0, x1, y1 = win
    return bool(np.any((boxes[:, 0] < x1) & (boxes[:, 2] > x0) & (boxes[:, 1] < y1) & (boxes[:, 3] > y0)))


def clip_boxes(boxes: np.ndarray, x0: int, y0: int, cfg: GeomConfig) -> tuple[np.ndarray, int]:
    """Dời hộp (đã thu nhỏ) vào hệ cửa sổ, cắt theo mép, giữ hộp còn ≥ ``min_keep_area_frac`` diện tích."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), 0
    b = boxes.astype(np.float64) - [x0, y0, x0, y0]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, WIN_W)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, WIN_H)
    carea = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
    visible = carea > 0
    keep = visible & (carea >= cfg.min_keep_area_frac * np.maximum(area, 1e-9)) & (
        np.minimum(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]) >= cfg.min_box_px)
    return c[keep].astype(np.float32), int((visible & ~keep).sum())


def plan_window(img_wh: tuple[int, int], boxes: np.ndarray, ignore: np.ndarray, person_short_px: dict,
                rng: np.random.Generator, cfg: GeomConfig) -> Window | None:
    """Chọn hệ số thu nhỏ + vị trí cửa sổ. ``None`` khi ảnh không đặt được cửa sổ hợp lệ (ghi vào thống kê bỏ)."""
    w, h = img_wh
    min_scale = max(WIN_W / w, WIN_H / h)
    if min_scale > 1.0 and not cfg.allow_upscale:
        return None
    negative = len(boxes) == 0 or rng.random() < cfg.neg_window_frac
    target = None
    if negative:
        s = float(rng.uniform(min_scale, max(min_scale, 1.0)))
    else:
        anchor = boxes[int(rng.integers(len(boxes)))]
        short = float(min(anchor[2] - anchor[0], anchor[3] - anchor[1]))
        target = sample_dist(person_short_px, rng)
        s = target / max(short, 1e-6)
        s = min(s, 1.0) if not cfg.allow_upscale else s
        s = max(s, min_scale)
    nw, nh = max(WIN_W, round(w * s)), max(WIN_H, round(h * s))
    sx, sy = nw / w, nh / h
    sb = boxes * [sx, sy, sx, sy] if len(boxes) else boxes
    si = ignore * [sx, sy, sx, sy] if len(ignore) else ignore
    for _ in range(cfg.max_tries):
        if negative:
            x0 = int(rng.integers(0, nw - WIN_W + 1))
            y0 = int(rng.integers(0, nh - WIN_H + 1))
            win = (x0, y0, x0 + WIN_W, y0 + WIN_H)
            if _intersects(win, sb) or _intersects(win, si):
                continue
        else:
            ab = anchor * [sx, sy, sx, sy]
            cx = float(rng.uniform(ab[0], ab[2]))
            cy = float(rng.uniform(ab[1], ab[3]))
            x0 = int(np.clip(round(cx - rng.uniform(0.1, 0.9) * WIN_W), 0, nw - WIN_W))
            y0 = int(np.clip(round(cy - rng.uniform(0.1, 0.9) * WIN_H), 0, nh - WIN_H))
            win = (x0, y0, x0 + WIN_W, y0 + WIN_H)
            if _intersects(win, si):
                continue
        kept, dropped = clip_boxes(sb, x0, y0, cfg)
        if not negative and len(kept) == 0:
            continue
        return Window(sx, sy, x0, y0, "negative" if negative else "person", target, kept, dropped)
    if negative and len(boxes):
        # Không chỗ trống cho cửa sổ không người: chuyển sang cửa sổ có người (giữ tất định theo rng).
        return plan_window(img_wh, boxes, ignore, person_short_px, rng,
                           GeomConfig(0.0, cfg.min_keep_area_frac, cfg.min_box_px, cfg.max_tries, cfg.allow_upscale))
    return None


def apply_window(img: np.ndarray, win: Window) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = round(w * win.scale_x), round(h * win.scale_y)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) if (nw, nh) != (w, h) else img
    out = small[win.y0:win.y0 + WIN_H, win.x0:win.x0 + WIN_W]
    if out.shape[:2] != (WIN_H, WIN_W):
        raise ValueError(f"cửa sổ lệch cỡ {out.shape} (ảnh {w}x{h}, scale {win.scale_x:.4f})")
    return np.ascontiguousarray(out)


def to_frame_boxes(win_boxes: np.ndarray) -> np.ndarray:
    """Hộp hệ cửa sổ 624×480 → hệ khung 640×480 (dải mép trái chiếm cột [0, 16))."""
    if len(win_boxes) == 0:
        return win_boxes
    return win_boxes + [CROP_LEFT, 0, CROP_LEFT, 0]


def yolo_lines(frame_boxes: np.ndarray) -> str:
    """Nhãn YOLO lớp 0, chuẩn hoá theo khung 640×480 (contract §7). Không người → chuỗi rỗng."""
    lines = []
    for x1, y1, x2, y2 in frame_boxes:
        cx, cy = (x1 + x2) / 2 / FRAME_W, (y1 + y2) / 2 / FRAME_H
        bw, bh = (x2 - x1) / FRAME_W, (y2 - y1) / FRAME_H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")
