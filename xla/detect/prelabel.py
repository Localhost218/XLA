"""Gán nhãn người trước cho khung test / dev-val (phase 08 bước 2).

Hai model khác họ với YOLO11s được chấm (thiết kế §7): **RT-DETR-x COCO** làm chính, **YOLO11x** làm phiếu thứ hai.
Mỗi model chạy SAHI (``get_sliced_prediction``): khung 640×480 phóng 2× (bicubic) thành 1280×960, lát 640×640 chồng 0,2,
cộng một lượt cả khung; gộp trong một model bằng NMS theo IOS 0,5 (lát cắt ngang người để lại hộp nửa người nằm trọn
trong hộp đủ, IOS bắt được, IoU thì không). RT-DETR suy luận ở 640 (cỡ nó được huấn luyện), YOLO11x ở 1280 (lượt cả khung
= "YOLO11x @1280"). Hai model hợp bằng WBF (``ensemble-boxes``, trọng số 2:1, IoU 0,55). Chỉ lớp person (COCO 0); bỏ hộp
nằm trọn trong dải tối ``EDGE_PX`` px mép trái (x2 ≤ 16). Mỗi hộp ghi nguồn: ``both`` / ``rtdetr`` / ``yolo`` (một model
có hộp IoU ≥ 0,55 với hộp hợp thì tính là có góp).

Giữ hộp: ``both`` mọi điểm; chỉ một model khi điểm của model đó ≥ ``SINGLE_KEEP_CONF``. Đây là nhãn **thô**, agent soát
từng khung sau (``results/labels/review_log.csv``), chủ dự án duyệt test trên Label Studio.

Dùng::

    python -m xla.detect.prelabel --split test --json      # test; ghi results/labels/prelabel/{prelabel.json,
                                                           #   labelstudio_tasks_test.json}
    python -m xla.detect.prelabel --split all --json       # test + dev-val
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from xla.detect.devval import load_manifest
from xla.paths import downloads_dir, repo_root

FRAMES_REL = Path("data/real/frames")
OUT_REL = Path("results/labels/prelabel")
W, H = 640, 480
UPSCALE = 2
SLICE = 640
OVERLAP = 0.2
EDGE_PX = 16
PERSON = 0
MODEL_CONF = 0.20  # ngưỡng mỗi model trước khi hợp
SINGLE_KEEP_CONF = 0.30  # hộp chỉ một model thấy: giữ khi điểm model đó ≥ ngưỡng này
WBF_IOU = 0.55
MATCH_IOU = 0.55
MODELS = (
    {"name": "rtdetr-x", "key": "rtdetr", "type": "rtdetr", "weights": "yolo-weights/rtdetr-x.pt", "imgsz": 640,
     "wbf_weight": 2},
    {"name": "yolo11x", "key": "yolo", "type": "ultralytics", "weights": "yolo-weights/yolo11x.pt", "imgsz": 1280,
     "wbf_weight": 1},
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def in_left_strip(b: list[float], edge_px: float = EDGE_PX) -> bool:
    """Hộp nằm trọn trong dải mép trái (px khung gốc)."""
    return b[2] <= edge_px


def clip_box(b: list[float]) -> list[float]:
    return [min(max(b[0], 0.0), W), min(max(b[1], 0.0), H), min(max(b[2], 0.0), W), min(max(b[3], 0.0), H)]


def fuse(per_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """WBF hai model → hộp hợp có nguồn + điểm từng model (px khung gốc). Thuần, không GPU."""
    from ensemble_boxes import weighted_boxes_fusion

    keys = [m["key"] for m in MODELS]
    boxes, scores, labels = [], [], []
    for k in keys:
        bs = per_model.get(k, [])
        boxes.append([[b["x1"] / W, b["y1"] / H, b["x2"] / W, b["y2"] / H] for b in bs])
        scores.append([b["conf"] for b in bs])
        labels.append([0] * len(bs))
    if not any(boxes):
        return []
    fb, fs, _ = weighted_boxes_fusion(boxes, scores, labels, weights=[m["wbf_weight"] for m in MODELS],
                                      iou_thr=WBF_IOU, skip_box_thr=MODEL_CONF, conf_type="avg")
    out = []
    for b, s in zip(fb.tolist(), fs.tolist(), strict=True):
        px = clip_box([b[0] * W, b[1] * H, b[2] * W, b[3] * H])
        conf_by, best_by = {}, {}
        for k in keys:
            best = max(((iou(px, [m["x1"], m["y1"], m["x2"], m["y2"]]), m["conf"]) for m in per_model.get(k, [])),
                       default=(0.0, 0.0))
            best_by[k] = best
            conf_by[k] = round(best[1], 4) if best[0] >= MATCH_IOU else None
        present = [k for k in keys if conf_by[k] is not None]
        if not present:  # tâm cụm WBF lệch khỏi mọi hộp gốc: gán cho model khớp nhất
            k = max(keys, key=lambda k: best_by[k][0])
            conf_by[k] = round(best_by[k][1], 4)
            present = [k]
        source = "both" if len(present) == 2 else present[0]
        out.append({"x1": round(px[0], 2), "y1": round(px[1], 2), "x2": round(px[2], 2), "y2": round(px[3], 2),
                    "score": round(s, 4), "source": source,
                    **{f"conf_{k}": conf_by[k] for k in keys}})
    return out


def keep(b: dict[str, Any]) -> bool:
    if in_left_strip([b["x1"], b["y1"], b["x2"], b["y2"]]):
        return False
    if b["source"] == "both":
        return True
    c = max((b.get(f"conf_{m['key']}") or 0.0) for m in MODELS)
    return c >= SINGLE_KEEP_CONF


def frames_for(split: str, root: Path) -> list[tuple[str, str]]:
    m = load_manifest(root)
    if not m["test_frames"] or not m["devval_frames"]:
        raise SystemExit("manifest chưa có test_frames/devval_frames: chạy `python -m xla.detect.devval select` trước")
    out: list[tuple[str, str]] = []
    if split in ("test", "all"):
        out += [(f, "test") for f in m["test_frames"]]
    if split in ("devval", "all"):
        out += [(f, "devval") for f in m["devval_frames"]]
    return out


def _predict(model: Any, img_rgb_up: Any, conf: float) -> list[dict[str, Any]]:
    from sahi.predict import get_sliced_prediction

    model.confidence_threshold = conf
    res = get_sliced_prediction(img_rgb_up, model, slice_height=SLICE, slice_width=SLICE,
                                overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, perform_standard_pred=True,
                                postprocess_type="NMS", postprocess_match_metric="IOS",
                                postprocess_match_threshold=0.5, postprocess_class_agnostic=True, verbose=0)
    out = []
    for op in res.object_prediction_list:
        if op.category.id != PERSON:
            continue
        x1, y1, x2, y2 = op.bbox.to_xyxy()
        b = clip_box([x1 / UPSCALE, y1 / UPSCALE, x2 / UPSCALE, y2 / UPSCALE])
        out.append({"x1": round(b[0], 2), "y1": round(b[1], 2), "x2": round(b[2], 2), "y2": round(b[3], 2),
                    "conf": round(float(op.score.value), 4)})
    return out


def to_ls_result(boxes: list[dict[str, Any]], from_name: str = "label", to_name: str = "image") -> list[dict]:
    """Hộp px khung gốc → ``result`` Label Studio (phần trăm 0–100)."""
    res = []
    for i, b in enumerate(boxes):
        res.append({
            "id": f"b{i + 1}", "from_name": from_name, "to_name": to_name, "type": "rectanglelabels",
            "original_width": W, "original_height": H, "image_rotation": 0,
            "value": {"x": b["x1"] / W * 100, "y": b["y1"] / H * 100, "width": (b["x2"] - b["x1"]) / W * 100,
                      "height": (b["y2"] - b["y1"]) / H * 100, "rotation": 0, "rectanglelabels": ["person"]},
        })
    return res


def run(split: str, lock_timeout: float) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from sahi import AutoDetectionModel

    from xla.gpu_lock import gpu_lock

    root = repo_root()
    todo = frames_for(split, root)
    models_meta = []
    for m in MODELS:
        p = downloads_dir() / m["weights"]
        if not p.exists():
            raise SystemExit(f"thiếu trọng số {p}")
        models_meta.append({"name": m["name"], "weights": str(p), "sha256": sha256_file(p)})
    raw: dict[str, dict[str, list[dict[str, Any]]]] = {f: {} for f, _ in todo}
    t_wait = time.monotonic()
    with gpu_lock(f"phase08 prelabel rtdetr-x+yolo11x sahi ({split})", timeout_s=lock_timeout):
        t0 = time.monotonic()
        waited = t0 - t_wait
        for m in MODELS:
            dm = AutoDetectionModel.from_pretrained(model_type=m["type"], model_path=str(downloads_dir() / m["weights"]),
                                                    confidence_threshold=MODEL_CONF, device="cuda:0",
                                                    image_size=m["imgsz"])
            for f, _ in todo:
                im = Image.open(root / FRAMES_REL / f).convert("RGB")
                if im.size != (W, H):
                    raise SystemExit(f"{f}: cỡ {im.size}, cần {(W, H)}")
                up = np.asarray(im.resize((W * UPSCALE, H * UPSCALE), Image.BICUBIC))
                raw[f][m["key"]] = _predict(dm, up, MODEL_CONF)
            del dm
            import torch

            torch.cuda.empty_cache()
        gpu_s = time.monotonic() - t0
    frames_out = {}
    counts = {"both": 0, "rtdetr": 0, "yolo": 0}
    dropped = {"left_strip": 0, "single_low_conf": 0}
    for f, sp in todo:
        fused = fuse(raw[f])
        kept = []
        for b in fused:
            if keep(b):
                kept.append(b)
                counts[b["source"]] += 1
            elif in_left_strip([b["x1"], b["y1"], b["x2"], b["y2"]]):
                dropped["left_strip"] += 1
            else:
                dropped["single_low_conf"] += 1
        for i, b in enumerate(kept):
            b["id"] = i + 1
        frames_out[f] = {"split": sp, "boxes": kept, "raw": raw[f]}
    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": "prelabel-v1",
        "models": models_meta,
        "params": {"upscale": UPSCALE, "slice": SLICE, "overlap": OVERLAP, "model_conf": MODEL_CONF,
                   "single_keep_conf": SINGLE_KEEP_CONF, "wbf_iou": WBF_IOU, "wbf_weights": [m["wbf_weight"] for m in MODELS],
                   "match_iou": MATCH_IOU, "edge_px": EDGE_PX, "imgsz": {m["key"]: m["imgsz"] for m in MODELS},
                   "sahi_postprocess": "NMS/IOS 0.5"},
        "frames": frames_out,
    }
    name = "prelabel.json" if split == "all" else f"prelabel_{split}.json"
    (out_dir / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    tasks = [{"data": {"image": f"/data/local-files/?d=frames/{f}", "frame": f},
              "predictions": [{"model_version": "prelabel-v1", "result": to_ls_result(v["boxes"])}]}
             for f, v in frames_out.items() if v["split"] == "test"]
    if tasks:
        (out_dir / "labelstudio_tasks_prelabel_test.json").write_text(
            json.dumps(tasks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return {"split": split, "frames": len(todo), "boxes_kept": sum(counts.values()), "by_source": counts,
            "dropped": dropped, "frames_with_box": sum(1 for v in frames_out.values() if v["boxes"]),
            "gpu_s": round(gpu_s, 1), "lock_wait_s": round(waited, 1), "out": str(OUT_REL / name), "models": models_meta}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.prelabel", description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("test", "devval", "all"), default="all")
    ap.add_argument("--lock-timeout", type=float, default=6 * 3600.0, help="giây chờ khoá GPU")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("YOLO_CONFIG_DIR", r"D:\DevCache\ultralytics")
    summary = run(args.split, args.lock_timeout)
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
