"""Khai thác mẫu cho verifier (phase 07 bước 1): hộp detector-v1 trên ``mine`` + devpool + d02x, dương GT, âm nền.

CLI (AX-01, mã thoát 0 đạt / 1 lỗi dữ liệu / 2 sai cách gọi)::

    python -m xla.verify.mine detect --config configs/verifier.yaml --json   # GPU (gpu_lock): detector ONNX
    python -m xla.verify.mine build  --config configs/verifier.yaml --json   # CPU: manifest.jsonl + mine.json
    python -m xla.verify.mine sheet  --config configs/verifier.yaml --json   # trang soát 200 âm thật ngẫu nhiên

Luật dữ liệu (D5): khung ``cal`` / ``holdout`` / khung giữ lại của khối cal / test v1 **không bao giờ** vào manifest.
Detector chạy trên ``cal`` chỉ để chấm (``det_real.json`` của results), không làm mẫu.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xla import contract as C
from xla.verify import common as V

MANIFEST = "manifest.jsonl"
DET_REAL = "det_real.json"      # results/: hộp detector (score_floor) trên mine + devpool, để khai thác + chấm OOF
DET_CAL = "det_cal.json"        # runs/ (ngoài repo): hộp detector trên cal, CHỈ để chấm — không nằm trong results/
REVIEW_CSV = "neg_review.csv"   # results/: id,image,action(drop),reason — người sót agent thấy trên trang soát
# id = <nguồn 3 chữ>-<sha1(ảnh|hộp)[:10]>: ổn định khi luật lọc đổi (không theo thứ tự dòng)


def _d02x_manifest(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    p = Path(cfg["d02x_dir"]) / "manifest.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r["split"] == "train"]


def _even(rows: list[Any], key: Any, n: int, rng: random.Random) -> list[Any]:
    """Lấy ``n`` phần tử chia đều theo ``key`` (nhóm ít thì lấy hết, phần dư chia cho nhóm còn lại)."""
    groups: dict[Any, list[Any]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    out: list[Any] = []
    left = dict(groups)
    while left and len(out) < n:
        per = max(1, (n - len(out)) // len(left))
        for k in sorted(left):
            take, left[k] = left[k][:per], left[k][per:]
            out += take
        left = {k: v for k, v in left.items() if v}
    return out[:n]


def d02x_sample(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(cfg["seed"])
    return _even(_d02x_manifest(cfg), lambda r: r["source"], int(cfg["mine"]["d02x_images"]), rng)


def cmd_detect(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch  # noqa: F401 — nạp DLL CUDA/cuDNN trước session ORT CUDA

    from xla.detect import export_onnx as X
    from xla.gpu_lock import gpu_lock

    rs = V.real_sets()
    real = sorted(set(rs["mine"]) | set(rs["devpool"]) | set(rs["cal"]))
    d02 = d02x_sample(cfg)
    onnx_path = Path(cfg["detector_bundle"]) / "detector.onnx"
    t0 = time.time()
    with gpu_lock("v2 phase07 verifier mine detect", timeout_s=21600):
        tl = time.time()
        sess = X.ort_session(onnx_path, cuda=True)
        det_real = {f: X.run_detector(sess, C.decode_frame(V.frame_path(f))) for f in real}
        ddir = Path(cfg["d02x_dir"]) / "train" / "images"
        det_d02 = {r["name"]: X.run_detector(sess, C.decode_frame(ddir / r["name"])) for r in d02}
        gpu_s = time.time() - tl
    def enc(d: dict[str, Any]) -> dict[str, Any]:
        return {k: [[*(round(float(x), 2) for x in b), round(float(s), 5)] for b, s in zip(*v, strict=True)]
                for k, v in d.items()}

    cal = set(rs["cal"])
    head = {"detector": str(onnx_path), "score_floor": 0.05}
    V.write_json(V.results_dir(cfg) / DET_REAL, {**head, "boxes": enc({k: v for k, v in det_real.items() if k not in cal})})
    V.write_json(V.runs_dir(cfg) / DET_CAL, {**head, "boxes": enc({k: v for k, v in det_real.items() if k in cal})})
    V.write_json(V.runs_dir(cfg) / "det_d02x.json", enc(det_d02))
    out = {"real_frames": len(real), "d02x_images": len(d02), "gpu_s": round(gpu_s, 1), "wall_s": round(time.time() - t0, 1)}
    _log_gpu(cfg, "mine_detect", gpu_s)
    return out


def _log_gpu(cfg: dict[str, Any], what: str, sec: float) -> None:
    p = V.runs_dir(cfg) / "gpu_seconds.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"what": what, "gpu_s": round(sec, 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def _random_boxes(rng: random.Random, n: int, side: list[float], avoid: np.ndarray, x_lo: float = C.CROP_LEFT_V1,
                  w: float = C.FRAME_W, h: float = C.FRAME_H) -> list[list[float]]:
    out: list[list[float]] = []
    for _ in range(n * 20):
        if len(out) >= n:
            break
        s = rng.uniform(*side)
        ar = rng.uniform(0.5, 2.0)
        bw, bh = s * ar ** 0.5, s / ar ** 0.5
        x1, y1 = rng.uniform(x_lo, w - bw), rng.uniform(0, h - bh)
        b = [x1, y1, x1 + bw, y1 + bh]
        if len(avoid) and V.iou_matrix(np.asarray([b]), avoid).max() > 0:
            continue
        out.append([round(v, 2) for v in b])
    return out


def _ioa(d: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Phần diện tích hộp ``d`` nằm trong GT lớn nhất (hộp nằm gọn trong người = một phần người, không làm âm)."""
    if not len(d) or not len(gt):
        return np.zeros(len(d))
    ix = np.clip(np.minimum(d[:, None, 2], gt[None, :, 2]) - np.maximum(d[:, None, 0], gt[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(d[:, None, 3], gt[None, :, 3]) - np.maximum(d[:, None, 1], gt[None, :, 1]), 0, None)
    area = np.maximum((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]), 1e-9)
    return (ix * iy / area[:, None]).max(1)


def _row(i: int, src: str, image: str, box: list[float], label: int, typ: str, fold: int,
         score: float | None = None, wh: tuple[int, int] = (C.FRAME_W, C.FRAME_H), x_lo: int = C.CROP_LEFT_V1,
         ) -> dict[str, Any]:
    key = f"{image}|{','.join(f'{float(v):.2f}' for v in box)}"
    rid = f"{src[:3]}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    return {"id": rid, "src": src, "image": image, "box": [round(float(v), 2) for v in box],
            "label": label, "type": typ, "fold": fold, "score": score, "wh": list(wh), "x_lo": x_lo}


def build_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    m = cfg["mine"]
    rng = random.Random(cfg["seed"])
    rs = V.real_sets()
    det = json.loads((V.results_dir(cfg) / DET_REAL).read_text(encoding="utf-8"))["boxes"]
    rows: list[dict[str, Any]] = []
    add = lambda *a, **k: rows.append(_row(len(rows), *a, **k))
    # ---- thật: mine (không người) + devpool (có nhãn)
    for f in rs["mine"] + rs["devpool"]:
        src = "mine" if f in set(rs["mine"]) else "devpool"
        gt = np.asarray(rs["gt"].get(f, []), np.float32).reshape(-1, 4)
        fold = rs["fold_of"][f]
        hard = f in rs["hard"]
        d = np.asarray(det.get(f, []), np.float32).reshape(-1, 5)
        iou = V.iou_matrix(d[:, :4], gt).max(1) if len(gt) else np.zeros(len(d))
        ioa = _ioa(d[:, :4], gt)
        for k, b in enumerate(d):
            if iou[k] >= m["pos_iou"]:
                add(src, f, b[:4].tolist(), 1, "real_pos_det", fold, float(b[4]))
            elif iou[k] < m["neg_max_iou"] and ioa[k] < m["neg_max_ioa"] and not hard:
                add(src, f, b[:4].tolist(), 0, "real_neg_hard", fold, float(b[4]))
        for b in gt:
            add(src, f, b.tolist(), 1, "real_pos_gt", fold)
        if not hard:
            avoid = V.context_boxes(gt, 2.0, 0.0, 0, C.FRAME_W, C.FRAME_H) if len(gt) else gt
            for b in _random_boxes(rng, int(m["random_bg_per_frame"]), m["random_bg_side"], avoid):
                add(src, f, b, 0, "real_neg_bg", fold)
    # ---- d02x: hộp detector trên ảnh mẫu + dương GT lấy đều theo nguồn
    man = {r["name"]: r for r in _d02x_manifest(cfg)}
    ddet = json.loads((V.runs_dir(cfg) / "det_d02x.json").read_text(encoding="utf-8"))
    for name, boxes in sorted(ddet.items()):
        gt = np.asarray(man[name]["boxes_xyxy_640"], np.float32).reshape(-1, 4)
        d = np.asarray(boxes, np.float32).reshape(-1, 5)
        iou = V.iou_matrix(d[:, :4], gt).max(1) if len(gt) else np.zeros(len(d))
        ioa = _ioa(d[:, :4], gt)
        for k, b in enumerate(d):
            if iou[k] >= m["pos_iou"]:
                add("d02x", name, b[:4].tolist(), 1, "ext_pos_det", -1, float(b[4]))
            elif iou[k] < m["neg_max_iou"] and ioa[k] < m["neg_max_ioa"]:
                add("d02x", name, b[:4].tolist(), 0, "ext_neg_det", -1, float(b[4]))
    cand = [(name, b) for name, r in man.items() for b in r["boxes_xyxy_640"]
            if min(b[2] - b[0], b[3] - b[1]) >= m["d02x_gt_min_side"]]
    for name, b in sorted(_even(cand, lambda t: man[t[0]]["source"], int(m["d02x_gt_pos_max"]), rng)):
        add("d02x", name, b, 1, "ext_pos_gt", -1)
    # ---- TinyPerson: chỉ dương (D: verifier-only), ảnh gốc cỡ thật
    for it, b in _tinyperson(cfg, rng):
        add("tinyperson", it.image, b, 1, "ext_pos_gt", -1, wh=tuple(it.wh), x_lo=0)
    return rows


def _tinyperson(cfg: dict[str, Any], rng: random.Random) -> list[tuple[Any, list[float]]]:
    from xla.degrade import sources as S
    from xla.paths import repo_root

    scfg = S.load_config(repo_root() / cfg["sources_config"])
    m = cfg["mine"]
    items = [i for i in S.select_external(scfg, for_d02x=False) if i.source == "tinyperson" and i.split == "train"]
    cand = []
    for it in items:
        if it.crop:  # ô cắt: hộp theo toạ độ ô -> đưa về ảnh gốc, cỡ ảnh = cỡ ảnh gốc không biết -> bỏ ô
            continue
        cand += [(it, list(map(float, b))) for b in it.boxes if min(b[2] - b[0], b[3] - b[1]) >= m["tinyperson_min_side"]]
    rng.shuffle(cand)
    return cand[: int(m["tinyperson_pos_max"])]


def apply_review(rows: list[dict[str, Any]], review: Path) -> tuple[list[dict[str, Any]], int]:
    """Bỏ dòng có ``id`` mang ``action == drop`` trong file soát; trả (dòng còn lại, số dòng thật sự bị bỏ)."""
    if not review.is_file():
        return rows, 0
    with open(review, encoding="utf-8", newline="") as f:
        drop = {r["id"] for r in csv.DictReader(f) if r["action"].strip() == "drop"}
    kept = [r for r in rows if r["id"] not in drop]
    return kept, len(rows) - len(kept)


def check_manifest(rows: list[dict[str, Any]], rs: dict[str, Any] | None = None) -> list[str]:
    rs = rs or V.real_sets()
    bad = V.forbidden_frames(rs)
    return [f"{r['id']}: khung {r['image']} thuộc cal/holdout/giữ lại/test" for r in rows
            if r["src"] in ("mine", "devpool") and r["image"] in bad]


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "by_type": dict(sorted(Counter(r["type"] for r in rows).items())),
        "by_src_label": dict(sorted(Counter(f"{r['src']}:{r['label']}" for r in rows).items())),
        "real_by_fold_type": dict(sorted(Counter(f"f{r['fold']}:{r['type']}" for r in rows if r["fold"] >= 0).items())),
    }


def cmd_build(cfg: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    rows = build_rows(cfg)
    rows, n_drop = apply_review(rows, V.results_dir(cfg) / REVIEW_CSV)
    errs = check_manifest(rows)
    if errs:
        return 1, {"errors": errs[:20]}
    out = V.results_dir(cfg) / MANIFEST
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"manifest": str(out), "dropped_by_review": n_drop, **counts(rows)}
    V.write_json(V.results_dir(cfg) / "mine.json", summary)
    return 0, summary


def load_manifest(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    p = V.results_dir(cfg) / MANIFEST
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def cmd_sheet(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ảnh ghép 10×10 vùng cắt (ngữ cảnh 3×, 96 px) của ``review_n`` âm thật: ưu tiên mọi âm hộp detector, còn lại nền."""
    import cv2

    rows = build_rows(cfg)
    rng = random.Random(cfg["seed"] + 1)
    hard = [r for r in rows if r["type"] == "real_neg_hard"]
    bg = [r for r in rows if r["type"] == "real_neg_bg"]
    n = int(cfg["mine"]["review_n"])
    rng.shuffle(hard)
    rng.shuffle(bg)
    pick = (hard + bg)[:n]
    out_dir = V.runs_dir(cfg) / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    tile, cols = 96, 10
    pages = []
    for p0 in range(0, len(pick), cols * cols):
        part = pick[p0 : p0 + cols * cols]
        sheet = np.zeros((tile * ((len(part) + cols - 1) // cols), tile * cols, 3), np.uint8)
        for i, r in enumerate(part):
            im = C.decode_frame(V.frame_path(r["image"]))
            ctx = V.context_boxes(np.asarray([r["box"]], np.float32), 3.0, 40.0, 0)
            crop = V.roi_crops(im, ctx, tile)[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
            crop = np.ascontiguousarray(crop)
            c = tile / 2
            bw = (r["box"][2] - r["box"][0]) / (ctx[0, 2] - ctx[0, 0]) * tile
            bh = (r["box"][3] - r["box"][1]) / (ctx[0, 3] - ctx[0, 1]) * tile
            cv2.rectangle(crop, (int(c - bw / 2), int(c - bh / 2)), (int(c + bw / 2), int(c + bh / 2)), (0, 255, 255), 1)
            cv2.putText(crop, f"{p0 + i:03d}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1)
            y, x = divmod(i, cols)
            sheet[y * tile : (y + 1) * tile, x * tile : (x + 1) * tile] = crop
        path = out_dir / f"neg_sheet_{p0 // (cols * cols):02d}.png"
        cv2.imwrite(str(path), sheet)
        pages.append(str(path))
    V.write_json(out_dir / "neg_sheet_ids.json", [r["id"] for r in pick])
    return {"pages": pages, "n": len(pick), "hard": sum(r["type"] == "real_neg_hard" for r in pick)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.verify.mine")
    ap.add_argument("cmd", choices=["detect", "build", "sheet"])
    ap.add_argument("--config", default=str(V.CONFIG_REL))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = V.load_cfg(a.config)
    except OSError as e:
        V.emit({"error": str(e)}, a.json)
        return 2
    if a.cmd == "detect":
        res, code = cmd_detect(cfg), 0
    elif a.cmd == "build":
        code, res = cmd_build(cfg)
    else:
        res, code = cmd_sheet(cfg), 0
    V.emit(res, a.json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
