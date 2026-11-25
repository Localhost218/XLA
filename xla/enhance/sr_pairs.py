"""Cặp huấn luyện SR ×2 (phase 11 AI v2): ảnh sạch VisDrone → (LR 640×480 làm xấu D2, HR 1248×960).

Lệnh (AX)::

    python -m xla.enhance.sr_pairs --config configs/sr_x2.yaml [--workers 4] [--limit N] --json

Mỗi ảnh nguồn đủ lớn (≥ ``hr_window``): cửa sổ HR 1248×960 (gốc ngẫu nhiên, seed theo ảnh) → hạ ×½ ``INTER_AREA`` →
cửa sổ 624×480 → ``xla.degrade.d2_camera.degrade`` (REUSE, ``params_version`` kiểm theo config) → khung 640×480 có dải
mép → ``encode_final`` (JPEG đúng camera). LR lưu nguyên byte JPEG đó (``lr/<split>/images``); HR lưu JPEG
``hr_jpeg_quality`` (``hr/<split>/images``). Cột ``x ≥ crop_left`` của LR ứng với cột ``(x − crop_left)·2`` của HR, đúng
quy ước ``xla.enhance.train_net.PairPatches(scale=2)``. ``manifest.jsonl`` ghi nguồn, seed, gốc cửa sổ, q.

Chạy lại bỏ qua cặp đã có (resume). Mã thoát: 0 = xong, 1 = lỗi khi chạy, 2 = sai cách gọi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from xla.paths import data_root, repo_root, src_datasets_dir

LR_W, LR_H = 624, 480  # cửa sổ D2 (xla.degrade.d2_camera.WIN_W/WIN_H)
_CTX: dict[str, Any] = {}


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else repo_root() / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def image_seed(base: int, key: str) -> int:
    """Seed tất định theo ảnh (không phụ thuộc thứ tự / số worker)."""
    return (int(base) * 1_000_003 + int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)) % (2**63)


def make_pair(img: np.ndarray, seed: int, d2_cfg: dict[str, Any], profile: dict[str, Any],
              hr_wh: tuple[int, int] = (2 * LR_W, 2 * LR_H)) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    """Ảnh sạch BGR → (byte JPEG LR 640×480, HR uint8 ``hr_wh``, bản ghi). Ảnh nhỏ hơn cửa sổ → ValueError."""
    from xla.degrade import d2_camera

    hw, hh = hr_wh
    if (hw, hh) != (2 * LR_W, 2 * LR_H):
        raise ValueError(f"hr_window phải là {[2 * LR_W, 2 * LR_H]} (×2 của cửa sổ D2), gặp {[hw, hh]}")
    h, w = img.shape[:2]
    if h < hh or w < hw:
        raise ValueError(f"ảnh {w}x{h} nhỏ hơn cửa sổ {hw}x{hh}")
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, w - hw + 1))
    y0 = int(rng.integers(0, h - hh + 1))
    hr = np.ascontiguousarray(img[y0 : y0 + hh, x0 : x0 + hw])
    win = cv2.resize(hr, (LR_W, LR_H), interpolation=cv2.INTER_AREA)
    frame, q, _params = d2_camera.degrade(win, rng, d2_cfg, profile)
    data, verified = d2_camera.encode_final(frame, q, profile)
    return data, hr, {"window_xy": [x0, y0], "q": int(q), "qtables_verified": bool(verified)}


def _init(d2_cfg: dict[str, Any], profile: dict[str, Any], out: str, hr_q: int) -> None:
    _CTX.update(d2=d2_cfg, profile=profile, out=Path(out), hr_q=hr_q)


def _work(job: tuple[str, str, str, int]) -> dict[str, Any]:
    key, image, split, seed = job
    name = key.replace("/", "_") + ".jpg"
    lr_p = _CTX["out"] / "lr" / split / "images" / name
    hr_p = _CTX["out"] / "hr" / split / "images" / name
    rec = {"name": name, "source": key, "image": image, "split": split, "seed": seed}
    if lr_p.is_file() and hr_p.is_file():
        return {**rec, "reused": True}
    img = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"không giải mã được {image}")
    data, hr, info = make_pair(img, seed, _CTX["d2"], _CTX["profile"])
    ok, buf = cv2.imencode(".jpg", hr, [cv2.IMWRITE_JPEG_QUALITY, int(_CTX["hr_q"])])
    if not ok:
        raise RuntimeError(f"không mã hoá được HR {name}")
    for p, b in ((hr_p, buf.tobytes()), (lr_p, data)):  # HR trước, LR sau: có LR ⇒ cặp đủ
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(b)
        tmp.replace(p)
    return {**rec, **info, "reused": False}


def pick_sources(cfg: dict[str, Any], limit: int | None = None) -> list[tuple[str, str, str, int]]:
    """Ảnh VisDrone đủ lớn, chọn cách đều (tất định) ``n_train`` / ``n_val`` ảnh."""
    from xla.degrade.sources import load_visdrone

    src = yaml.safe_load((repo_root() / "configs" / "sources.yaml").read_text(encoding="utf-8"))["visdrone"]
    items = load_visdrone(src_datasets_dir() / src["root"], src["subsets"])
    hw, hh = cfg["pairs"]["hr_window"]
    jobs: list[tuple[str, str, str, int]] = []
    for split, n in (("train", cfg["pairs"]["n_train"]), ("val", cfg["pairs"]["n_val"])):
        ok = sorted((it for it in items if it.split == split and it.wh[0] >= hw and it.wh[1] >= hh),
                    key=lambda it: it.key)
        n = min(int(n), len(ok)) if limit is None else min(int(n), len(ok), int(limit))
        step = max(1, len(ok) // max(n, 1))
        for it in ok[::step][:n]:
            jobs.append((it.key, it.image, split, image_seed(cfg["pairs"]["seed"], it.key)))
    return jobs


def build_pairs(cfg: dict[str, Any], workers: int, limit: int | None = None) -> dict[str, Any]:
    from xla.degrade import d2_camera

    pc = cfg["pairs"]
    d2_cfg = d2_camera.load_config(repo_root() / pc["d2_config"])
    if int(d2_cfg.get("params_version", -1)) != int(pc["d2_params_version"]):
        raise ValueError(f"D2 params_version {d2_cfg.get('params_version')} khác {pc['d2_params_version']} của config")
    profile = d2_camera.load_profile(repo_root() / pc["profile"])
    out = data_root() / pc["out"]
    jobs = pick_sources(cfg, limit)
    workers = max(1, min(4, int(workers)))
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(d2_cfg, profile, str(out), pc["hr_jpeg_quality"])) as pool:
        recs = list(pool.imap(_work, jobs, chunksize=4))
    dt = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    new = [r for r in recs if not r["reused"]]
    return {"out": out.as_posix(), "n": len(recs), "new": len(new), "reused": len(recs) - len(new),
            "split_counts": {s: sum(r["split"] == s for r in recs) for s in ("train", "val")},
            "d2_params_version": int(d2_cfg["params_version"]), "workers": workers, "seconds": round(dt, 1),
            "sec_per_pair_wall": round(dt / max(len(new), 1), 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.sr_pairs", description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/sr_x2.yaml")
    ap.add_argument("--workers", type=int, default=None, help="trần 4")
    ap.add_argument("--limit", type=int, default=None, help="tối đa N ảnh mỗi split (thử nhanh)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        rep = build_pairs(cfg, args.workers or cfg["pairs"]["workers"], args.limit)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"sr_pairs lỗi: {e}", file=sys.stderr)
        return 1
    print(json.dumps(rep, ensure_ascii=False) if args.json else json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
