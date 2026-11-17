"""Một đường nạp ảnh/patch chung cho cả tập thật lẫn tập tổng hợp (phase 06 bước 1).

Mọi tầng đo độ giống thật (thống kê, KID, C2ST) đi qua đây để hai tập được giải mã bằng cùng thư viện
(`cv2.imdecode(..., IMREAD_COLOR)`, đúng hợp đồng §1), bỏ cùng dải mép trái và cắt patch không resize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CROP_LEFT_DEFAULT = 16  # trùng xla.camera.measure.CROP_LEFT_DEFAULT (hợp đồng §1, meta.crop_left v1)


class ReadProtocolError(ValueError):
    """Hai tập (hoặc ảnh trong một tập) không đọc được theo cùng một quy trình → CLI exit 2."""


def read_image(path: str | Path) -> np.ndarray:
    """Giải mã một ảnh thành BGR uint8 [H, W, 3] bằng đúng quy trình của IOT-CV (hợp đồng §1)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ReadProtocolError(f"không giải mã được ảnh: {path}")
    return img


def list_images(directory: str | Path) -> list[Path]:
    """Mọi ảnh (jpg/jpeg/png) dưới thư mục, đệ quy, sắp theo đường dẫn để tất định."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"không có thư mục ảnh: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not paths:
        raise FileNotFoundError(f"thư mục không có ảnh {IMAGE_EXTS}: {root}")
    return paths


def split_images(split_file: str | Path, split: str, frames_dir: str | Path | None = None) -> list[Path]:
    """Khung của một phần (`dev`/`test`/`neg`) trong `data/real/split.json` (hợp đồng §7)."""
    split_file = Path(split_file)
    manifest = json.loads(split_file.read_text(encoding="utf-8"))
    frames_dir = Path(frames_dir) if frames_dir else split_file.parent / "frames"
    names = [f["frame"] for f in manifest["frames"] if f.get("split") == split]
    if not names:
        raise FileNotFoundError(f"split '{split}' rỗng trong {split_file}")
    paths = [frames_dir / n for n in names]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(paths)} khung thiếu trong {frames_dir}, ví dụ {missing[0].name}")
    return paths


def subsample(paths: list[Path], max_images: int | None, seed: int) -> list[Path]:
    """Lấy tối đa `max_images` ảnh, chọn có seed, giữ thứ tự gốc."""
    if max_images is None or len(paths) <= max_images:
        return list(paths)
    idx = np.sort(np.random.default_rng(seed).choice(len(paths), size=max_images, replace=False))
    return [paths[i] for i in idx]


@dataclass
class SetProtocol:
    """Quy trình đọc của một tập: cỡ ảnh, số kênh sau giải mã, đuôi file."""

    size_wh: tuple[int, int]
    n_images: int
    exts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"size_wh": list(self.size_wh), "n_images": self.n_images, "exts": self.exts}


def probe_protocol(paths: list[Path]) -> SetProtocol:
    """Đọc header (PIL, không giải mã điểm ảnh) để kiểm mọi ảnh cùng cỡ trước khi đo."""
    from PIL import Image

    sizes: dict[tuple[int, int], int] = {}
    for p in paths:
        with Image.open(p) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
    if len(sizes) != 1:
        raise ReadProtocolError(f"ảnh trong cùng một tập khác cỡ: {sizes}")
    exts = sorted({p.suffix.lower() for p in paths})
    return SetProtocol(size_wh=next(iter(sizes)), n_images=len(paths), exts=exts)


def check_same_protocol(real: SetProtocol, synth: SetProtocol) -> list[str]:
    """Lỗi cứng (exit 2) khi hai tập khác cỡ; trả cảnh báo mềm (vd. JPEG so với PNG)."""
    if real.size_wh != synth.size_wh:
        raise ReadProtocolError(
            f"hai tập khác cỡ ảnh: thật {real.size_wh}, tổng hợp {synth.size_wh}; "
            "không resize một phía (phase 06 bước 1)"
        )
    warnings = []
    if real.exts != synth.exts:
        warnings.append(f"đuôi file khác nhau: thật {real.exts}, tổng hợp {synth.exts} (DQT sẽ báo riêng)")
    return warnings


@dataclass
class Patches:
    """Patch cắt từ một tập ảnh. `source[i]` là chỉ số ảnh nguồn của patch i trong `paths`."""

    pixels: np.ndarray  # [N, size, size, 3] uint8 BGR
    source: np.ndarray  # [N] int
    boxes: np.ndarray  # [N, 2] (x, y) góc trên trái theo toạ độ ảnh GỐC (đã cộng crop_left)
    paths: list[Path]
    size: int


def _plan_boxes(n_images: int, w: int, h: int, size: int, n_patches: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Chia đều số patch cho các ảnh (ảnh chọn bằng hoán vị có seed), toạ độ ngẫu nhiên trong vùng hợp lệ."""
    if size > w or size > h:
        raise ReadProtocolError(f"patch {size} lớn hơn vùng ảnh sau khi bỏ mép trái {w}x{h}")
    order = rng.permutation(n_images)
    source = np.sort(np.resize(order, n_patches)) if n_patches >= n_images else np.sort(order[:n_patches])
    xs = rng.integers(0, w - size + 1, size=n_patches)
    ys = rng.integers(0, h - size + 1, size=n_patches)
    return source, np.stack([xs, ys], axis=1)


def load_patches(
    paths: list[Path],
    size: int,
    n_patches: int,
    seed: int,
    crop_left: int = CROP_LEFT_DEFAULT,
) -> Patches:
    """Cắt `n_patches` patch `size`×`size` từ `paths`: cùng seed → cùng patch.

    Bỏ cột `[0, crop_left)` trước khi chọn toạ độ; không resize. Mỗi ảnh giải mã đúng một lần.
    """
    if not paths:
        raise ValueError("danh sách ảnh rỗng")
    proto = probe_protocol(paths)
    w, h = proto.size_wh
    rng = np.random.default_rng(seed)
    source, boxes = _plan_boxes(len(paths), w - crop_left, h, size, n_patches, rng)
    pixels = np.empty((n_patches, size, size, 3), dtype=np.uint8)
    for img_idx in np.unique(source):
        img = read_image(paths[img_idx])
        if img.shape[:2] != (h, w):
            raise ReadProtocolError(f"cỡ sau giải mã {img.shape[:2]} khác header {(h, w)}: {paths[img_idx]}")
        img = img[:, crop_left:]
        for i in np.flatnonzero(source == img_idx):
            x, y = boxes[i]
            pixels[i] = img[y : y + size, x : x + size]
    boxes = boxes.copy()
    boxes[:, 0] += crop_left
    return Patches(pixels=pixels, source=source, boxes=boxes, paths=list(paths), size=size)
