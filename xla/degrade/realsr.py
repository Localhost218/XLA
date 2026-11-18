"""E3 (phase 07 bước 5, Deferred 7): nhiễu D2 lấy từ khung thật kiểu Real-SR (Ji 2020, "noise injection").

Thay nhiễu tổng hợp theo NLF/PSD của bước (d) bằng phần dư cao tần gom từ khung **dev** thật:

- Patch ``patch``×``patch`` px, đặt THẲNG lưới JPEG 8×8 của khung thật (toạ độ tuyệt đối chia hết cho 8), chỉ lấy ở vùng
  phẳng (độ lệch chuẩn của Y làm mịn σ 2 thuộc ``flat_quantile`` thấp nhất của khung đó), để phần dư là nhiễu chứ không
  phải cạnh.
- Phần dư = patch − trung bình từng khối 8×8 (theo từng kênh): bỏ bậc DC giữa các khối, là dấu khối JPEG chính ở vùng
  phẳng của khung q12. Phần AC trong khối còn lại.
- Khi dán: lưới ô dịch một độ lệch ngẫu nhiên KHÔNG chia hết cho 8, nên biên khối cũ không trùng lưới JPEG cuối; mỗi ô
  chọn ngẫu nhiên patch cùng bin độ sáng (8 bin Y) để giữ nhiễu phụ thuộc mức sáng. JPEG vẫn đúng một lần ở cuối
  (``d2_camera.encode_final``).

CLI: ``python -m xla.degrade.realsr build --split dev [--max-frames N] --json`` → ``<cache_dir>/degrade/realsr_pool.npz``
+ ``.json`` (danh sách khung đã dùng). Không dùng khung test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from xla import paths

POOL_REL = Path("degrade") / "realsr_pool.npz"
N_BINS = 8
LUMA_W = np.array([0.114, 0.587, 0.299], np.float32)  # BGR


def pool_path() -> Path:
    return paths.cache_dir() / POOL_REL


def block_mean_residual(patch: np.ndarray, block: int = 8) -> np.ndarray:
    """Patch [H, W, C] float (H, W chia hết cho ``block``) − trung bình từng khối ``block``×``block`` theo kênh."""
    h, w, c = patch.shape
    if h % block or w % block:
        raise ValueError(f"patch {h}x{w} không chia hết cho khối {block}")
    v = patch.reshape(h // block, block, w // block, block, c)
    return (v - v.mean(axis=(1, 3), keepdims=True)).reshape(h, w, c)


def extract_patches(img_bgr: np.ndarray, crop_left: int, patch: int, per_frame: int, flat_quantile: float,
                    rng) -> tuple[np.ndarray, np.ndarray]:
    """Phần dư (float32, thang 0–1) + Y trung bình (0–255) của ≤ ``per_frame`` patch phẳng thẳng lưới 8 px."""
    if patch % 8 or crop_left % 8:
        raise ValueError("patch và crop_left phải chia hết cho 8 (thẳng lưới JPEG)")
    f = img_bgr.astype(np.float32) / 255.0
    y = f @ LUMA_W
    ys = cv2.GaussianBlur(y, (0, 0), 2.0)
    h, w = y.shape
    cand = [(x0, y0) for y0 in range(0, h - patch + 1, patch) for x0 in range(crop_left, w - patch + 1, patch)]
    std = np.array([ys[y0:y0 + patch, x0:x0 + patch].std() for x0, y0 in cand])
    keep = np.flatnonzero(std <= np.quantile(std, flat_quantile))
    keep = rng.permutation(keep)[:per_frame]
    res = np.stack([block_mean_residual(f[cand[i][1]:cand[i][1] + patch, cand[i][0]:cand[i][0] + patch])
                    for i in keep]) if len(keep) else np.empty((0, patch, patch, 3), np.float32)
    ymean = np.array([y[cand[i][1]:cand[i][1] + patch, cand[i][0]:cand[i][0] + patch].mean() * 255.0 for i in keep],
                     np.float32)
    return res.astype(np.float32), ymean


def build_pool(frames: list[Path], crop_left: int = 16, patch: int = 32, per_frame: int = 8,
               flat_quantile: float = 0.3, seed: int = 20261118) -> dict:
    rng = np.random.default_rng(seed)
    res, ym, src = [], [], []
    for i, p in enumerate(frames):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"không giải mã được {p}")
        r, m = extract_patches(img, crop_left, patch, per_frame, flat_quantile, rng)
        res.append(r)
        ym.append(m)
        src += [i] * len(m)
    # float16: pool nằm trong RAM của MỖI tiến trình con (16 × float32 ≈ 5 GB với 24 patch/khung)
    return {"residual": np.concatenate(res).astype(np.float16), "y_mean": np.concatenate(ym),
            "frame_index": np.array(src, np.int32)}


def bin_of(y255: np.ndarray | float) -> np.ndarray:
    return np.clip((np.asarray(y255, np.float32) / (256.0 / N_BINS)).astype(int), 0, N_BINS - 1)


def load_pool(path: Path | None = None) -> dict:
    p = path or pool_path()
    if not p.exists():
        raise FileNotFoundError(f"chưa có pool Real-SR {p}: chạy `python -m xla.degrade.realsr build --split dev`")
    z = np.load(p, allow_pickle=False)
    pool = {"residual": z["residual"], "y_mean": z["y_mean"]}
    bins = bin_of(pool["y_mean"])
    pool["by_bin"] = [np.flatnonzero(bins == b) for b in range(N_BINS)]
    if not any(len(b) for b in pool["by_bin"]):
        raise ValueError(f"pool Real-SR {p} rỗng")
    return pool


def _nearest_bin(by_bin: list[np.ndarray], b: int) -> np.ndarray:
    for d in range(N_BINS):
        for c in (b - d, b + d):
            if 0 <= c < N_BINS and len(by_bin[c]):
                return by_bin[c]
    raise ValueError("pool rỗng")


def inject(srgb: np.ndarray, rng, pool: dict, gain: float) -> tuple[np.ndarray, dict]:
    """Cộng phần dư thật vào ảnh sRGB 0–1 [H, W, 3]: lưới ô dịch lệch lưới 8 px, ô chọn patch cùng bin Y."""
    h, w = srgb.shape[:2]
    ps = pool["residual"].shape[1]
    off = int(rng.choice([v for v in range(1, ps) if v % 8]))
    offy = int(rng.choice([v for v in range(1, ps) if v % 8]))
    noise = np.zeros((h + 2 * ps, w + 2 * ps, 3), np.float32)
    ymap = srgb @ LUMA_W * 255.0
    for y0 in range(-offy, h, ps):
        for x0 in range(-off, w, ps):
            ya, xa = max(y0, 0), max(x0, 0)
            yb, xb = min(y0 + ps, h), min(x0 + ps, w)
            idx = _nearest_bin(pool["by_bin"], int(bin_of(ymap[ya:yb, xa:xb].mean())))
            noise[y0 + ps:y0 + 2 * ps, x0 + ps:x0 + 2 * ps] = pool["residual"][int(rng.choice(idx))]
    out = np.clip(srgb + gain * noise[ps:ps + h, ps:ps + w], 0.0, 1.0)
    return out, {"realsr_gain": round(float(gain), 4), "realsr_offset": [off, offy]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.degrade.realsr", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--split", default="dev", choices=["dev"], help="chỉ dev: test giữ riêng cho bước 4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--per-frame", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from xla.eval.patches import split_images, subsample

    split_file = paths.repo_root() / "data" / "real" / "split.json"
    frames = split_images(split_file, a.split)
    if a.max_frames:
        frames = subsample(frames, a.max_frames, a.seed)
    pool = build_pool(frames, per_frame=a.per_frame, seed=a.seed)
    out = pool_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **pool)
    meta = {"split": a.split, "split_file_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
            "frames": [p.name for p in frames], "n_patches": len(pool["y_mean"]), "seed": a.seed,
            "per_frame": a.per_frame, "bins": np.bincount(bin_of(pool["y_mean"]), minlength=N_BINS).tolist(),
            "out": out.as_posix()}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                        newline="\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "frames"} | {"n_frames": len(frames)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
