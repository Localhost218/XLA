"""KID trên đặc trưng DINOv2 (phase 06 bước 3).

KID: MMD² không chệch với nhân đa thức bậc 3 `k(x, y) = (x·y / d + 1)^3` (Bińkowski et al., ICLR 2018), tính trên
nhiều tập con rời nhau có seed, báo trung bình, SE và CI 95 % bootstrap của trung bình.

Hàm trích đặc trưng truyền vào được (`FeatureFn`): test dùng hàm giả trên CPU, mặc định là DINOv2 ViT-L/14 với tiền xử lý
y hệt `dgm-eval` (PIL RGB → resize 224×224 bicubic → /255 → chuẩn hoá ImageNet → token CLS 1024 chiều). Có `dgm_eval`
thì dùng encoder của nó; không thì nạp cùng model qua `torch.hub` (phương án thay Owner đã duyệt, xem
`thuc-thi-phase-01-07.md` §2).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from xla.eval.patches import Patches

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DINO_INPUT = 224


class FeatureFn(Protocol):
    """Hàm trích đặc trưng: patch BGR uint8 [N, S, S, 3] → float [N, D]. `name` dùng làm khoá cache."""

    name: str

    def __call__(self, pixels_bgr: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------- KID


def _poly_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.shape[1]
    return (a @ b.T / d + 1.0) ** 3


def mmd2_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    """MMD² không chệch giữa hai tập đặc trưng (cần ≥ 2 mẫu mỗi tập)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        raise ValueError(f"KID cần ≥ 2 mẫu mỗi tập (có {m}, {n})")
    kxx = _poly_kernel(x, x)
    kyy = _poly_kernel(y, y)
    kxy = _poly_kernel(x, y)
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(term_x + term_y - 2.0 * kxy.mean())


def kid(
    x: np.ndarray,
    y: np.ndarray,
    n_subsets: int = 10,
    subset_size: int = 1000,
    seed: int = 0,
    n_boot: int = 2000,
) -> dict:
    """KID: trung bình MMD² không chệch trên `n_subsets` tập con **rời nhau** (hoán vị có seed rồi chia khúc), mỗi khúc
    `min(subset_size, n // n_subsets)` mẫu. Rời nhau để các giá trị độc lập → SE = std/√k trung thực (tập con chồng nhau
    chỉ đo nhiễu rút mẫu lại, SE bị đánh giá thấp). CI 95 % bootstrap của trung bình trên các giá trị tập con.
    """
    size = min(subset_size, len(x) // n_subsets, len(y) // n_subsets)
    if size < 2:
        raise ValueError(
            f"KID cần ≥ {2 * n_subsets} mẫu mỗi tập cho {n_subsets} tập con rời nhau (có {len(x)}, {len(y)})"
        )
    rng = np.random.default_rng(seed)
    px, py = rng.permutation(len(x)), rng.permutation(len(y))
    values = np.array(
        [mmd2_unbiased(x[px[k * size : (k + 1) * size]], y[py[k * size : (k + 1) * size]]) for k in range(n_subsets)]
    )
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    return {
        "mean": float(values.mean()),
        "se": se,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "subsets": values.tolist(),
        "subset_size": int(size),
        "subsets_disjoint": True,
        "n_x": len(x),
        "n_y": len(y),
    }


# ---------------------------------------------------------------- cache theo sha256 ảnh


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_features(
    patches: Patches,
    feature_fn: FeatureFn,
    cache_dir: str | Path | None = None,
    lock: Callable | None = None,
) -> tuple[np.ndarray, dict]:
    """Đặc trưng cho từng patch, theo thứ tự `patches`. Cache: `<cache_dir>/<feature_fn.name>/<sha[:2]>/<sha>.npz`,
    một khoá `x_y_size` mỗi patch. Chỉ patch chưa có trong cache mới chạy `feature_fn`, bên trong `lock()` nếu có.
    """
    n = len(patches.pixels)
    feats: list[np.ndarray | None] = [None] * n
    store: dict[int, tuple[Path, dict]] = {}
    if cache_dir is not None:
        root = Path(cache_dir) / feature_fn.name
        for img_idx in np.unique(patches.source):
            sha = _sha256(patches.paths[img_idx])
            f = root / sha[:2] / f"{sha}.npz"
            data = dict(np.load(f)) if f.is_file() else {}
            store[int(img_idx)] = (f, data)
            for i in np.flatnonzero(patches.source == img_idx):
                key = f"{patches.boxes[i, 0]}_{patches.boxes[i, 1]}_{patches.size}"
                if key in data:
                    feats[i] = data[key]
    todo = [i for i in range(n) if feats[i] is None]
    if todo:
        if lock is not None:
            with lock():
                new = np.asarray(feature_fn(patches.pixels[todo]), dtype=np.float32)
        else:
            new = np.asarray(feature_fn(patches.pixels[todo]), dtype=np.float32)
        for j, i in enumerate(todo):
            feats[i] = new[j]
            if cache_dir is not None:
                key = f"{patches.boxes[i, 0]}_{patches.boxes[i, 1]}_{patches.size}"
                store[int(patches.source[i])][1][key] = new[j]
        if cache_dir is not None:
            for f, data in (store[s] for s in {int(patches.source[i]) for i in todo}):
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez(f, **data)
    info = {"n_patches": n, "n_cached": n - len(todo), "n_computed": len(todo), "encoder": feature_fn.name}
    return np.stack(feats).astype(np.float32), info


# ---------------------------------------------------------------- DINOv2 ViT-L/14


class Dinov2Features:
    """DINOv2 ViT-L/14, token CLS. `backend`: `auto` (dgm-eval nếu import được, không thì torch.hub), `dgm-eval`, `torch.hub`."""

    def __init__(self, device: str = "cuda", batch_size: int = 32, backend: str = "auto", arch: str = "vitl14"):
        self.device = device
        self.batch_size = batch_size
        self.arch = arch
        self.requested_backend = backend
        self.backend: str | None = None
        self._model = None
        self._dgm_transform = None
        self.name = f"dinov2_{arch}_cls_bicubic224"

    def _load(self):
        import torch

        if self.requested_backend in ("auto", "dgm-eval"):
            try:
                from dgm_eval.models.dinov2 import DINOv2Encoder

                enc = DINOv2Encoder(arch=self.arch, clean_resize=False)
                self._model, self._dgm_transform, self.backend = enc, enc.transform, "dgm-eval"
            except ImportError:
                if self.requested_backend == "dgm-eval":
                    raise
        if self._model is None:
            hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            if hub_dir.is_dir():
                self._model = torch.hub.load(str(hub_dir), f"dinov2_{self.arch}", source="local")
            else:
                self._model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{self.arch}", trust_repo=True)
            self.backend = "torch.hub"
        self._model.eval().to(self.device)

    def _prep(self, pixels_bgr: np.ndarray):
        import torch
        from PIL import Image

        rgb = pixels_bgr[..., ::-1]
        if self._dgm_transform is not None:
            return torch.stack([self._dgm_transform(Image.fromarray(np.ascontiguousarray(p))) for p in rgb]).float()
        if rgb.shape[1:3] != (DINO_INPUT, DINO_INPUT):
            rgb = np.stack(
                [
                    np.asarray(Image.fromarray(np.ascontiguousarray(p)).resize((DINO_INPUT, DINO_INPUT), Image.BICUBIC))
                    for p in rgb
                ]
            )
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))

    def __call__(self, pixels_bgr: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()
        out = []
        with torch.inference_mode():
            for s in range(0, len(pixels_bgr), self.batch_size):
                x = self._prep(pixels_bgr[s : s + self.batch_size]).to(self.device)
                out.append(self._model(x).float().cpu().numpy())
        return np.concatenate(out, axis=0)
