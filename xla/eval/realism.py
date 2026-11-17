"""Phép đo "ảnh tổng hợp có giống khung thật của camera này không" (phase 06).

Ba tầng, xếp theo độ tin: thống kê mức thấp tất định (`xla.eval.lowlevel`) → C2ST trên patch phần dư cao tần
(`xla.eval.c2st`) → KID trên đặc trưng DINOv2 (`xla.eval.kid`). Hai tập đi qua cùng một đường nạp (`xla.eval.patches`).

CLI (AX):
    python -m xla.eval.realism --real <dir|split:dev> --synth <dir> --json [--out results/realism/x.json]
Exit: 0 đo xong; 1 lỗi đầu vào/môi trường; 2 hai tập khác quy trình đọc (khác cỡ ảnh, ảnh không giải mã được).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from xla.eval.patches import (
    CROP_LEFT_DEFAULT,
    ReadProtocolError,
    check_same_protocol,
    list_images,
    load_patches,
    probe_protocol,
    split_images,
    subsample,
)

__all__ = ["compute_realism", "load_patches", "main", "resolve_set"]

TIERS = ("stats", "c2st", "kid")
DEFAULTS = {
    "seed": 20261118,
    "n_patches": 1000,
    "kid_patch": 224,
    "kid_subsets": 10,
    "kid_subset_size": 1000,
    "c2st_patch": 64,
    "c2st_epochs": 20,
}


def resolve_set(spec: str, split_file: str | None = None, frames_dir: str | None = None) -> list[Path]:
    """`split:<tên>` → khung của phần đó trong split.json; còn lại là thư mục ảnh."""
    if spec.startswith("split:"):
        if split_file is None:
            try:
                from xla.paths import repo_root
            except ImportError as e:
                raise FileNotFoundError("chưa có xla.paths (phase 01): truyền --split-file") from e
            split_file = str(repo_root() / "data" / "real" / "split.json")
        return split_images(split_file, spec.split(":", 1)[1], frames_dir)
    return list_images(spec)


def _cuda_peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def compute_realism(
    real_paths: list[Path],
    synth_paths: list[Path],
    tiers: tuple[str, ...] = TIERS,
    feature_fn=None,
    measure=None,
    weights: dict[str, float] | None = None,
    crop_left: int = CROP_LEFT_DEFAULT,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    lock=None,
    **params,
) -> dict:
    """Chạy các tầng được chọn; trả dict JSON-hoá được, có giây từng tầng và VRAM đỉnh (khi chạy CUDA).

    `feature_fn` None → DINOv2 ViT-L/14 (`kid.Dinov2Features`); `measure` None → `xla.camera.measure`;
    `lock` là callable trả context manager (vd. `lambda: gpu_lock("realism")`) bọc mọi việc trên GPU.
    """
    p = {**DEFAULTS, **params}
    lock = lock or nullcontext
    pr, ps = probe_protocol(real_paths), probe_protocol(synth_paths)
    warnings = check_same_protocol(pr, ps)
    out: dict = {
        "protocol": {
            "real": pr.as_dict(),
            "synth": ps.as_dict(),
            "crop_left": crop_left,
            "decoder": "cv2.imdecode IMREAD_COLOR",
            "warnings": warnings,
        },
        "params": p,
        "device": device,
        "timing_s": {},
        "vram_peak_mb": {},
    }
    if device.startswith("cuda"):
        import torch

        out["gpu"] = torch.cuda.get_device_name(0)

    if "stats" in tiers:
        from xla.eval.lowlevel import measure_set, stat_distance

        t = time.perf_counter()
        mr = measure_set(real_paths, crop_left, measure)
        ms = measure_set(synth_paths, crop_left, measure)
        out["stats"] = stat_distance(mr, ms, weights)
        out["timing_s"]["stats"] = round(time.perf_counter() - t, 2)

    if "c2st" in tiers:
        from xla.eval.c2st import c2st

        t = time.perf_counter()
        a = load_patches(real_paths, p["c2st_patch"], p["n_patches"], p["seed"], crop_left)
        b = load_patches(synth_paths, p["c2st_patch"], p["n_patches"], p["seed"] + 1, crop_left)
        out["timing_s"]["c2st_load"] = round(time.perf_counter() - t, 2)
        if device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats()
        with lock():
            out["c2st"] = c2st(a, b, seed=p["seed"], epochs=p["c2st_epochs"], device=device)
        out["timing_s"]["c2st"] = round(time.perf_counter() - t, 2)
        out["vram_peak_mb"]["c2st"] = _cuda_peak_mb(device)

    if "kid" in tiers:
        from xla.eval.kid import Dinov2Features, extract_features, kid

        feature_fn = feature_fn or Dinov2Features(device=device)
        t = time.perf_counter()
        a = load_patches(real_paths, p["kid_patch"], p["n_patches"], p["seed"], crop_left)
        b = load_patches(synth_paths, p["kid_patch"], p["n_patches"], p["seed"] + 1, crop_left)
        out["timing_s"]["kid_load"] = round(time.perf_counter() - t, 2)
        if device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats()
        fr, ir = extract_features(a, feature_fn, cache_dir, lock)
        fs, is_ = extract_features(b, feature_fn, cache_dir, lock)
        out["timing_s"]["kid_features"] = round(time.perf_counter() - t - out["timing_s"]["kid_load"], 2)
        res = kid(fr, fs, p["kid_subsets"], p["kid_subset_size"], p["seed"])
        res.update(
            {
                "encoder": feature_fn.name,
                "backend": getattr(feature_fn, "backend", None),
                "features_real": ir,
                "features_synth": is_,
            }
        )
        out["kid"] = res
        out["timing_s"]["kid"] = round(time.perf_counter() - t, 2)
        out["vram_peak_mb"]["kid"] = _cuda_peak_mb(device)
    return out


def _load_weights(path: str | None) -> dict | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    return {k: float(v) for k, v in (data.get("weights", data) or {}).items()}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m xla.eval.realism", description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", required=True, help="thư mục ảnh, hoặc split:<dev|test|neg>")
    ap.add_argument("--synth", required=True, help="thư mục ảnh tổng hợp (đệ quy, jpg/png)")
    ap.add_argument("--split-file", help="đường dẫn split.json (mặc định <repo>/data/real/split.json)")
    ap.add_argument("--frames-dir", help="thư mục khung (mặc định <thư mục split.json>/frames)")
    ap.add_argument("--tiers", default=",".join(TIERS), help="tập con của stats,c2st,kid")
    ap.add_argument("--max-images", type=int, help="lấy tối đa N ảnh mỗi tập (có seed)")
    ap.add_argument("--n-patches", type=int, default=DEFAULTS["n_patches"])
    ap.add_argument("--c2st-epochs", type=int, default=DEFAULTS["c2st_epochs"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--crop-left", type=int, default=CROP_LEFT_DEFAULT)
    ap.add_argument("--weights", help="yaml/json: {đại lượng: trọng số} hoặc {weights: {...}}")
    ap.add_argument("--device", default=None, help="cuda|cpu (mặc định cuda nếu có)")
    ap.add_argument("--cache-dir", help="cache đặc trưng (mặc định <xla.paths.cache_dir()>/realism)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--no-gpu-lock", action="store_true", help="không chờ xla.gpu_lock (chỉ khi chắc không job GPU nào khác)"
    )
    ap.add_argument("--json", action="store_true", help="in JSON ra stdout")
    ap.add_argument("--out", help="ghi kết quả JSON vào file")
    return ap


def _summary(res: dict) -> str:
    lines = [f"ảnh: thật {res['protocol']['real']['n_images']}, tổng hợp {res['protocol']['synth']['n_images']}"]
    if "stats" in res:
        s = res["stats"]
        lines.append(
            f"thống kê: tổng {s['total']:.4f} (W1/IQR), DQT khớp: {s['dqt']['match']}, bỏ (NaN): {s['excluded_nan']}"
        )
    if "c2st" in res:
        c = res["c2st"]
        lines.append(
            f"C2ST: acc {c['accuracy']:.3f} CI95 [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] (0,5 = không phân biệt được)"
        )
    if "kid" in res:
        k = res["kid"]
        lines.append(f"KID: {k['mean']:.5f} ± {k['se']:.5f} (SE), CI95 [{k['ci95'][0]:.5f}, {k['ci95'][1]:.5f}]")
    lines.append(f"giây: {res['timing_s']}, VRAM đỉnh MB: {res['vram_peak_mb']}")
    return "\n".join(lines)


def _clean(o):
    """NaN/inf → None (JSON hợp lệ), numpy → kiểu Python, Path → str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
        bad = set(tiers) - set(TIERS)
        if bad:
            raise ValueError(f"tầng không biết: {sorted(bad)}")
        real = subsample(resolve_set(args.real, args.split_file, args.frames_dir), args.max_images, args.seed)
        synth = subsample(resolve_set(args.synth), args.max_images, args.seed + 1)
        device = args.device
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        lock = None
        if device.startswith("cuda") and not args.no_gpu_lock:
            from xla.gpu_lock import gpu_lock

            lock = lambda: gpu_lock("xla.eval.realism")
        cache_dir = None
        if "kid" in tiers and not args.no_cache:
            if args.cache_dir:
                cache_dir = args.cache_dir
            else:
                from xla.paths import cache_dir as _cd

                cache_dir = _cd() / "realism"
        t0 = time.perf_counter()
        res = compute_realism(
            real,
            synth,
            tiers=tiers,
            weights=_load_weights(args.weights),
            crop_left=args.crop_left,
            device=device,
            cache_dir=cache_dir,
            lock=lock,
            seed=args.seed,
            n_patches=args.n_patches,
            c2st_epochs=args.c2st_epochs,
        )
        res["timing_s"]["total"] = round(time.perf_counter() - t0, 2)
        res["inputs"] = {
            "real": args.real,
            "synth": args.synth,
            "split_file": args.split_file,
            "frames_dir": args.frames_dir,
            "max_images": args.max_images,
        }
        res["env"] = {"python": platform.python_version(), "platform": platform.platform()}
        code = 0
    except ReadProtocolError as e:
        res, code = {"error": "read_protocol_mismatch", "detail": str(e)}, 2
    except Exception as e:  # noqa: BLE001 — CLI: mọi lỗi khác là exit 1, in nguyên văn
        res, code = {"error": type(e).__name__, "detail": str(e)}, 1
    text = json.dumps(_clean(res), ensure_ascii=False, indent=2, allow_nan=False)
    if args.out and code == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    if args.json:
        print(text)
    else:
        print(
            _summary(res) if code == 0 else f"lỗi ({code}): {res['detail']}",
            file=sys.stdout if code == 0 else sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
