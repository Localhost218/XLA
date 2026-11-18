"""Sinh tập D0 (sạch) / D1 (ngẫu nhiên Real-ESRGAN) / D2 (bản sao camera) 640×480 + nhãn YOLO + manifest.

Cùng ảnh nguồn + cùng cửa sổ cho cả ba tập: hình học lấy RNG theo (seed, "geom", đơn vị), không theo tập. Mỗi tập ghi
``<data_root>/<set>/{train,val}/{images,labels}``, ``manifest.jsonl`` (nguồn, seed, tham số, sha256 ảnh ra) và
``dataset.yaml`` cho Ultralytics. Dựng lại một tập là xoá thư mục tập đó rồi sinh lại (tái lập bằng seed).

Chịu ngắt: mỗi ảnh ghi xong là thêm một dòng vào ``manifest.partial.jsonl`` (dòng đầu = khoá dựng: tập, seed, config,
hồ sơ). Chạy lại cùng khoá thì ảnh đã có file đúng sha256 được giữ, chỉ sinh phần còn thiếu; khác khoá thì dựng lại.

    python -m xla.degrade.build --set d2 --seed 20261118 --limit 50 --json
    python -m xla.degrade.build --set all --profile D:/XLA-wt/data/configs/camera_profile.yaml --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from xla import paths
from xla.degrade import d1_random, d2_camera, geometry, sources

SETS = ("d0", "d1", "d2")
DEFAULT_SEED = 20261118
D0_JPEG_QUALITY = 95                     # D0 "sạch": Pillow q95, 4:4:4


def fnv1a(s: str) -> int:
    """Hash chuỗi ổn định (không phụ thuộc PYTHONHASHSEED), như ``src/data_augmentation.py::hash_str``."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def rng_for(seed: int, tag: str, unit: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), fnv1a(tag), fnv1a(unit)])


def _repo_cfg(name: str) -> Path:
    return paths.repo_root() / "configs" / name


def encode_d0(frame_bgr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_bgr[..., ::-1])).save(
        buf, format="JPEG", quality=D0_JPEG_QUALITY, subsampling=0)
    return buf.getvalue()


def pad_black(win_bgr: np.ndarray) -> np.ndarray:
    frame = np.zeros((geometry.FRAME_H, geometry.FRAME_W, 3), np.uint8)
    frame[:, geometry.CROP_LEFT:] = win_bgr
    return frame


# ----------------------------------------------------------------------------- một đơn vị

_W: dict = {}


def worker_env() -> None:
    """Đặt TRƯỚC khi tạo pool tiến trình con (spawn kế thừa biến môi trường): BLAS một luồng. Mặc định OpenBLAS giữ
    bộ đệm cho mọi lõi, ~750 MB commit mỗi tiến trình chỉ để import numpy; 16 tiến trình là ~12 GB (18/11: hết commit
    memory, tràn trang). Phép tính ở đây không dùng BLAS đa luồng nên kết quả không đổi."""
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def _init_worker(state: dict) -> None:
    if state.get("_single_thread"):
        cv2.setNumThreads(1)  # nhiều tiến trình: mỗi tiến trình một luồng cv2 (không đổi kết quả)
    _W.clear()
    _W.update(state)


def render_unit(set_name: str, item: dict, unit: str, seed: int, ctx: dict) -> dict:
    """Một đơn vị (ảnh nguồn, chỉ số cửa sổ) → bytes ảnh + nhãn + bản ghi manifest; ``None``-dict khi bỏ."""
    gcfg = geometry.GeomConfig(**ctx["geometry"])
    boxes = np.asarray(item["boxes"], np.float32).reshape(-1, 4)
    ignore = np.asarray(item["ignore"], np.float32).reshape(-1, 4)
    win = geometry.plan_window(tuple(item["wh"]), boxes, ignore, ctx["person_short_px"],
                               rng_for(seed, "geom", unit), gcfg)
    if win is None:
        return {"unit": unit, "skipped": "no_valid_window"}
    img = cv2.imread(item["image"], cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"không đọc được {item['image']}")
    if (img.shape[1], img.shape[0]) != tuple(item["wh"]):
        raise ValueError(f"{item['key']}: cỡ ảnh {img.shape[1]}x{img.shape[0]} khác chỉ mục {item['wh']}")
    window = geometry.apply_window(img, win)
    rng = rng_for(seed, "d2" if set_name == "d2sr" else set_name, unit)   # d2sr: cùng tham số D2, chỉ khác nhiễu
    extra: dict = {}
    if set_name == "d0":
        data, params = encode_d0(pad_black(window)), {"jpeg": {"quality": D0_JPEG_QUALITY, "subsampling": "4:4:4"}}
    elif set_name == "d1":
        frame01, q, params = d1_random.degrade(window, rng, ctx["d1"])
        data = d1_random.encode_final(frame01, q)
    else:
        frame, q, params = d2_camera.degrade(window, rng, ctx["d2"], ctx["profile"],
                                             ctx.get("noise_pool") if set_name == "d2sr" else None)
        data, verified = d2_camera.encode_final(frame, q, ctx["profile"])
        params["jpeg"]["qtable_verified"] = verified
        extra = {"profile_id": ctx["profile"].get("profile_id"), "profile_version": ctx["profile"].get("profile_version"),
                 "d2_config_version": ctx["d2"].get("version")}
    fboxes = geometry.to_frame_boxes(win.boxes)
    name = unit.replace("/", "__").replace("#", "_w")
    return {
        "unit": unit, "name": name, "split": item["split"], "data": data,
        "label": geometry.yolo_lines(fboxes),
        "record": {
            "name": f"{name}.jpg", "set": set_name, "split": item["split"], "seed": seed, "unit": unit,
            "source": item["source"], "source_key": item["key"], "source_image": item["image"],
            "alt_m": item.get("alt_m"),
            "window": {"kind": win.kind, "scale_x": round(win.scale_x, 6), "scale_y": round(win.scale_y, 6),
                       "x0": win.x0, "y0": win.y0, "target_short_px": None if win.target_short_px is None
                       else round(win.target_short_px, 3), "boxes_dropped_partial": win.dropped},
            "n_boxes": len(fboxes),
            "boxes_xyxy_640": [[round(float(v), 2) for v in b] for b in fboxes],
            "params": params, **extra,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        },
    }


def _render_chunk(args: tuple) -> list[dict]:
    set_name, chunk, seed = args
    return [render_unit(set_name, it, u, seed, _W) for it, u in chunk]


# ----------------------------------------------------------------------------- dựng tập

def plan_units(items: list[sources.SourceItem], windows_per_image: int, seed: int, limit: int | None) -> list:
    units = [(sources.asdict(it), f"{it.key}#{w}") for it in items for w in range(windows_per_image)]
    units.sort(key=lambda t: (fnv1a(f"{seed}:{t[1]}"), t[1]))       # thứ tự trộn tất định; --limit lấy đầu danh sách
    return units[:limit] if limit else units


PARTIAL = "manifest.partial.jsonl"
KEY_FILE = "build_key.json"


def build_key(set_name: str, seed: int, ctx: dict) -> str:
    """Khoá dựng: đổi tập/seed/config/hồ sơ/hình học → khoá khác → không dùng lại ảnh cũ."""
    parts = {"set": set_name, "seed": seed, "geometry": ctx.get("geometry"), "person": ctx.get("person_short_px")}
    if set_name == "d1":
        parts["d1"] = ctx.get("d1")
    if set_name in ("d2", "d2sr"):
        parts["d2"] = ctx.get("d2")
        parts["profile"] = [ctx["profile"].get("profile_id"), ctx["profile"].get("profile_version")]
    if set_name == "d2sr":
        parts["pool"] = ctx.get("noise_pool_sha256")
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _safe_clear(set_dir: Path) -> None:
    if set_dir.exists():
        if any(set_dir.iterdir()) and not ((set_dir / "manifest.jsonl").exists() or (set_dir / PARTIAL).exists()):
            raise RuntimeError(f"{set_dir} có dữ liệu nhưng không có manifest.jsonl — không tự xoá, kiểm tay")
        shutil.rmtree(set_dir)


def _load_done(set_dir: Path, key: str) -> dict[str, dict]:
    """Bản ghi đã xong của lần dựng trước cùng khoá (ảnh còn đúng sha256); khác khoá → {} (dựng lại từ đầu)."""
    kf = set_dir / KEY_FILE
    if not kf.exists() or json.loads(kf.read_text(encoding="utf-8")).get("key") != key:
        return {}
    done: dict[str, dict] = {}
    for src in (set_dir / "manifest.jsonl", set_dir / PARTIAL):
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                    # dòng cuối ghi dở lúc ngắt
            if "skipped" in r:
                done[r["unit"]] = r
                continue
            img = set_dir / r["split"] / "images" / r["name"]
            lab = set_dir / r["split"] / "labels" / (Path(r["name"]).stem + ".txt")
            if img.exists() and lab.exists() and hashlib.sha256(img.read_bytes()).hexdigest() == r["sha256"]:
                done[r["unit"]] = r
    return done


def build_set(set_name: str, units: list, seed: int, ctx: dict, root: Path, workers: int,
              limit: int | None = None) -> dict:
    """Sinh một tập từ danh sách đơn vị đã trộn tất định. ``limit``: dừng khi đủ ``limit`` ảnh ghi được (đơn vị bị bỏ
    vì không đặt được cửa sổ thì lấy đơn vị kế tiếp, nên cùng seed + limit luôn ra cùng tập). Chạy lại sau khi bị ngắt
    thì giữ ảnh đã sinh đúng sha256 (cùng khoá dựng), chỉ sinh phần thiếu."""
    set_dir = root / set_name
    key = build_key(set_name, seed, ctx)
    done = _load_done(set_dir, key) if set_dir.exists() else {}
    if not done:
        _safe_clear(set_dir)
    for sp in ("train", "val"):
        (set_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (set_dir / sp / "labels").mkdir(parents=True, exist_ok=True)
    (set_dir / KEY_FILE).write_text(json.dumps({"key": key, "set": set_name, "seed": seed}) + "\n", encoding="utf-8",
                                    newline="\n")
    partial = set_dir / PARTIAL
    with partial.open("w", encoding="utf-8", newline="\n") as f:   # viết lại phần đã xác minh
        f.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in done.values())
    reused_units = set(done)
    reused = len(reused_units)
    t0 = time.perf_counter()
    target = limit or len(units)
    chosen: list[dict] = []
    pos, n_ok = 0, 0
    if workers > 1:
        worker_env()
    ex = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=({**ctx, "_single_thread": True},)) if workers > 1 else None
    if ex is None:
        _init_worker(ctx)
    try:
        with partial.open("a", encoding="utf-8", newline="\n") as pf:
            while n_ok < target and pos < len(units):
                need = target - n_ok
                batch = units[pos:pos + (need if not limit else need + max(4, need // 5))]
                pos += len(batch)
                todo = [u for u in batch if u[1] not in done]
                chunk = max(1, min(32, len(todo) // max(1, workers * 4)))
                jobs = [(set_name, todo[i:i + chunk], seed) for i in range(0, len(todo), chunk)]
                for part in (map(_render_chunk, jobs) if ex is None else ex.map(_render_chunk, jobs)):
                    for r in part:
                        if "skipped" in r:
                            rec = {"unit": r["unit"], "skipped": r["skipped"]}
                        else:
                            d = set_dir / r["split"]
                            (d / "images" / f"{r['name']}.jpg").write_bytes(r["data"])
                            (d / "labels" / f"{r['name']}.txt").write_text(r["label"], encoding="utf-8", newline="\n")
                            rec = r["record"]
                        done[r["unit"]] = rec
                        pf.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    pf.flush()
                for _, u in batch:                       # chọn theo thứ tự đơn vị (tất định với --limit)
                    if n_ok >= target:
                        break
                    rec = done[u]
                    chosen.append(rec)
                    n_ok += "skipped" not in rec
    finally:
        if ex is not None:
            ex.shutdown()
    records = sorted((r for r in chosen if "skipped" not in r), key=lambda x: x["name"])
    skipped = sum("skipped" in r for r in chosen)
    keep = {(r["split"], r["name"]) for r in records}
    for sp in ("train", "val"):                          # ảnh thừa của lần trước (vd. --limit lớn hơn) thì bỏ
        for img in (set_dir / sp / "images").glob("*.jpg"):
            if (sp, img.name) not in keep:
                img.unlink()
                (set_dir / sp / "labels" / (img.stem + ".txt")).unlink(missing_ok=True)
    nbytes = sum(r["bytes"] for r in records)
    with (set_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    partial.unlink()
    (set_dir / "dataset.yaml").write_text(yaml.safe_dump({
        "path": set_dir.as_posix(), "train": "train/images", "val": "val/images", "names": {0: "person"},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    dt = time.perf_counter() - t0
    n = len(records)
    rendered = n - sum(1 for r in records if r["unit"] in reused_units)
    return {"set": set_name, "written": n, "skipped": skipped, "reused_from_previous_run": reused,
            "seconds": round(dt, 2), "sec_per_image_wall": round(dt / max(rendered, 1), 4), "workers": workers,
            "bytes": nbytes, "mean_bytes": round(nbytes / max(n, 1)), "out": set_dir.as_posix(),
            "persons": int(sum(r["n_boxes"] for r in records)),
            "negative_windows": int(sum(r["window"]["kind"] == "negative" for r in records)),
            "split_counts": {sp: sum(r["split"] == sp for r in records) for sp in ("train", "val")}}


def make_context(set_names: list[str], profile_path: Path, src_cfg: dict, d1_cfg: Path, d2_cfg: Path) -> dict:
    profile = d2_camera.load_profile(profile_path)
    ctx = {"geometry": src_cfg["geometry"]["window"],
           "person_short_px": d2_camera.get_field(profile, "person_box_px.short_side")}
    if "d1" in set_names:
        ctx["d1"] = yaml.safe_load(d1_cfg.read_text(encoding="utf-8"))
    if "d2" in set_names or "d2sr" in set_names:
        ctx["d2"] = d2_camera.load_config(d2_cfg)
        ctx["profile"] = profile
    if "d2sr" in set_names:
        from xla.degrade import realsr

        ctx["noise_pool"] = realsr.load_pool()
        ctx["noise_pool_sha256"] = hashlib.sha256(realsr.pool_path().read_bytes()).hexdigest()
    return ctx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=[*SETS, "d2sr", "all"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=0, help="số ảnh ra tối đa (0 = đủ); chạy thử nhỏ")
    ap.add_argument("--profile", type=Path, default=_repo_cfg("camera_profile.yaml"))
    ap.add_argument("--sources", type=Path, default=_repo_cfg("sources.yaml"))
    ap.add_argument("--d1-config", type=Path, default=_repo_cfg("degrade_d1.yaml"))
    ap.add_argument("--d2-config", type=Path, default=_repo_cfg("degrade_d2.yaml"))
    ap.add_argument("--out", type=Path, default=None, help="mặc định data_root() của configs/paths.yaml")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    set_names = list(SETS) if a.set == "all" else [a.set]
    src_cfg = sources.load_config(a.sources)
    items = sources.select(src_cfg)
    units = plan_units(items, int(src_cfg["geometry"]["windows_per_image"]), a.seed, None)
    ctx = make_context(set_names, a.profile, src_cfg, a.d1_config, a.d2_config)
    root = a.out or paths.data_root()
    summary = {"seed": a.seed, "source_images": len(items), "units": len(units), "profile": str(a.profile),
               "sets": [build_set(s, units, a.seed, ctx, root, a.workers, a.limit or None) for s in set_names]}
    print(json.dumps(summary, ensure_ascii=False, indent=None if a.json else 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
