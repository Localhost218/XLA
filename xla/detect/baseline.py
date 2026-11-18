"""Mốc YOLO11s COCO có sẵn trên khung bay thật → CSV dự đoán theo hợp đồng §7 (phase 09 bước 2).

Ba biến thể, cùng trọng số ``yolo11s.pt`` COCO, imgsz 1280, conf 0,001, chỉ lớp người (0):

- ``full``        — khung nguyên.
- ``edge16``      — như ``full`` nhưng bỏ hộp có ``x1 < 16`` (chạm dải tối mép trái, cùng luật ``CROP_LEFT_PX`` của
  đếm sơ bộ phase 03). Lọc sau suy luận nên ms/khung bằng ``full``.
- ``sahi_edge16`` — SAHI (lát 320×320, chồng 0,2, mỗi lát suy luận ở imgsz 1280, cộng một lượt cả khung) rồi bỏ mép như
  ``edge16``. Conf < 0,1 nên SAHI tự đổi hậu xử lý sang NMS/IoU 0,5 (ghi trong ``run_meta.json``).

Chạy trên khung ``test`` + ``neg`` + toàn bộ ``dev`` của ``split.json`` (dev-val phase 08 chưa chốt; chạy hết dev thì
chọn dev-val kiểu nào cũng có dự đoán). GPU trong ``xla.gpu_lock`` (chờ khoá là bình thường). ``ms`` = thời gian suy luận
một khung (gồm tiền/hậu xử lý, đưa hộp về CPU; không gồm đọc file), sau ``WARMUP`` khung khởi động không tính.

AX::

    python -m xla.detect.baseline run --json                 # ghi results/baseline/pred/<biến thể>.csv + run_meta.json
    python -m xla.detect.baseline run --limit 10 --out-dir <tmp> --json   # chạy thử
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera.frames import CROP_LEFT_PX, FRAMES_REL
from xla.eval.detection import BOOT_N, BOOT_SEED
from xla.paths import downloads_dir, repo_root

WEIGHTS_REL = Path("yolo-weights/yolo11s.pt")
"""Tương đối ``xla.paths.downloads_dir()``."""
IMGSZ = 1280
CONF = 0.001
CLASS_PERSON = 0
N_COCO_CLASSES = 80
MAX_DET = 300
SAHI_SLICE = 320
SAHI_OVERLAP = 0.2
SAHI_BATCH = 8
WARMUP = 5
VARIANTS = ("full", "edge16", "sahi_edge16")
RUN_SPLITS = ("test", "neg", "dev")
SPLIT_REL = Path("data/real/split.json")
OUT_REL = Path("results/baseline/pred")
CSV_FIELDS = ("frame", "x1", "y1", "x2", "y2", "conf", "ms")

Row = tuple[float, float, float, float, float]  # x1, y1, x2, y2, conf


def select_frames(split_json: Path, splits: tuple[str, ...]) -> dict[str, list[str]]:
    frames = json.loads(split_json.read_text(encoding="utf-8"))["frames"]
    return {sp: sorted(f["frame"] for f in frames if f["split"] == sp) for sp in splits}


def drop_edge(rows: list[Row], crop_left: float = CROP_LEFT_PX) -> list[Row]:
    """Bỏ hộp chạm dải mép trái (``x1 < crop_left``)."""
    return [r for r in rows if r[0] >= crop_left]


def write_csv(path: Path, per_frame: dict[str, tuple[list[Row], float]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_FIELDS)
        for frame, (rows, ms) in per_frame.items():
            for x1, y1, x2, y2, conf in rows:
                w.writerow((frame, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(conf, 5), round(ms, 2)))
                n += 1
    return n


def _ms_stats(per_frame: dict[str, tuple[list[Row], float]]) -> dict[str, float]:
    v = np.asarray([ms for _, ms in per_frame.values()], dtype=float)
    return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
            "p95": round(float(np.percentile(v, 95)), 2), "n_frames": int(v.size),
            "frames_without_box": sum(1 for rows, _ in per_frame.values() if not rows)}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(frames: list[str], frames_dir: Path, out_dir: Path, lock_timeout: float) -> dict[str, Any]:
    import cv2
    import torch
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO

    from xla.gpu_lock import gpu_lock

    weights = downloads_dir() / WEIGHTS_REL
    full: dict[str, tuple[list[Row], float]] = {}
    sahi: dict[str, tuple[list[Row], float]] = {}
    lock_wait0 = time.perf_counter()
    with gpu_lock(f"phase09 baseline yolo11s@{IMGSZ} ({len(frames)} khung)", timeout_s=lock_timeout):
        lock_wait_s = time.perf_counter() - lock_wait0
        model = YOLO(str(weights))
        sahi_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                        confidence_threshold=CONF, device="cuda:0", image_size=IMGSZ)
        images = {f: cv2.imread(str(frames_dir / f), cv2.IMREAD_COLOR) for f in frames}
        bad = [f for f, im in images.items() if im is None]
        if bad:
            raise FileNotFoundError(f"không đọc được {len(bad)} khung, vd. {bad[:3]}")
        first = images[frames[0]]
        for _ in range(WARMUP):
            model.predict(first, imgsz=IMGSZ, conf=CONF, classes=[CLASS_PERSON], max_det=MAX_DET, device=0, verbose=False)
            _sahi_predict(get_sliced_prediction, sahi_model, first[:, :, ::-1])
        for f in frames:
            im = images[f]
            t0 = time.perf_counter()
            res = model.predict(im, imgsz=IMGSZ, conf=CONF, classes=[CLASS_PERSON], max_det=MAX_DET, device=0,
                                verbose=False)[0]
            data = res.boxes.data.cpu().numpy()
            full[f] = ([(float(x1), float(y1), float(x2), float(y2), float(c)) for x1, y1, x2, y2, c, _ in data],
                       (time.perf_counter() - t0) * 1000)
            rgb = np.ascontiguousarray(im[:, :, ::-1])
            t0 = time.perf_counter()
            rows = _sahi_predict(get_sliced_prediction, sahi_model, rgb)
            sahi[f] = (rows, (time.perf_counter() - t0) * 1000)
        gpu_name = torch.cuda.get_device_name(0)

    outs = {"full": full, "edge16": {f: (drop_edge(r), ms) for f, (r, ms) in full.items()},
            "sahi_edge16": {f: (drop_edge(r), ms) for f, (r, ms) in sahi.items()}}
    summary: dict[str, Any] = {}
    for name, per_frame in outs.items():
        n_rows = write_csv(out_dir / f"{name}.csv", per_frame)
        summary[name] = {"csv": (out_dir / f"{name}.csv").as_posix(), "rows": n_rows, "ms": _ms_stats(per_frame)}
    return {"variants": summary, "gpu": gpu_name, "gpu_lock_wait_s": round(lock_wait_s, 1),
            "weights": {"path": WEIGHTS_REL.as_posix(), "sha256": _sha256(weights)}}


def _sahi_predict(get_sliced_prediction: Any, sahi_model: Any, rgb: np.ndarray) -> list[Row]:
    res = get_sliced_prediction(rgb, sahi_model, slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                                overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
                                perform_standard_pred=True, verbose=0, batch_size=SAHI_BATCH,
                                exclude_classes_by_id=list(range(1, N_COCO_CLASSES)))
    rows = []
    for op in res.object_prediction_list:
        if op.category.id != CLASS_PERSON:
            continue
        x1, y1, x2, y2 = op.bbox.to_xyxy()
        rows.append((float(x1), float(y1), float(x2), float(y2), float(op.score.value)))
    return rows


def cmd_run(a: argparse.Namespace) -> int:
    import sahi
    import torch
    import ultralytics

    root = repo_root()
    by_split = select_frames(a.split_json or root / SPLIT_REL, RUN_SPLITS)
    frames = [f for sp in RUN_SPLITS for f in by_split[sp]]
    if a.limit:
        frames = frames[: a.limit]
    out_dir = a.out_dir or root / OUT_REL
    t0 = time.time()
    res = run(frames, a.frames_dir or root / FRAMES_REL, out_dir, a.lock_timeout)
    meta = {
        "phase": "09", "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "frames": {sp: len(set(frames) & set(by_split[sp])) for sp in RUN_SPLITS} | {"total": len(frames)},
        "imgsz": IMGSZ, "conf": CONF, "classes": [CLASS_PERSON], "max_det": MAX_DET, "warmup_frames": WARMUP,
        "edge_rule": f"bỏ hộp x1 < {CROP_LEFT_PX} px (CROP_LEFT_PX, phase 03)",
        "sahi": {"slice": SAHI_SLICE, "overlap": SAHI_OVERLAP, "batch": SAHI_BATCH, "perform_standard_pred": True,
                 "postprocess": "NMS/IOU 0.5 (SAHI tự đổi từ GREEDYNMM/IOS vì conf < 0.1)"},
        "ms_note": "suy luận + tiền/hậu xử lý + đưa hộp về CPU, không gồm đọc JPEG; edge16 dùng lại ms của full",
        "versions": {"ultralytics": ultralytics.__version__, "sahi": sahi.__version__, "torch": torch.__version__,
                     "python": platform.python_version()},
        "wall_s": None,
    } | res
    meta["wall_s"] = round(time.time() - t0, 1)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                           newline="\n")
    print(json.dumps(meta, ensure_ascii=False) if a.json else meta)
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    """Chấm ba CSV bằng ``xla.eval.detection`` → một JSON (``baseline.json`` khi nhãn đã duyệt, không thì bản tạm)."""
    from xla.eval.detection import EvalInputError, TestLeakError, evaluate

    root = repo_root()
    pred_dir = a.pred_dir or root / OUT_REL
    meta_p = pred_dir / "run_meta.json"
    doc: dict[str, Any] = {"phase": "09", "created": datetime.now(UTC).isoformat(timespec="seconds"),
                           "run_meta": json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else None,
                           "variants": {}}
    try:
        for v in VARIANTS:
            doc["variants"][v] = evaluate(pred_dir / f"{v}.csv", "test", a.labels_dir, a.manifest, a.split_json,
                                          a.thr, a.boot, a.seed, a.allow_missing_labels,
                                          devval_labels_dir=a.devval_labels_dir)
    except (EvalInputError, TestLeakError, FileNotFoundError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    official = all(r["official"] for r in doc["variants"].values())
    note = None if official else "SỐ TẠM — nhãn chưa được chủ dự án duyệt hoặc thiếu nhãn; không phải số chính thức"
    doc = {"official": official, "note": note} | doc  # đặt lên đầu file cho dễ thấy
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    brief = {v: {k: r["metrics"][k] for k in ("mAP50", "mAP50_95", "AP_small", "AR_small")} | {"n": r["n"]}
             for v, r in doc["variants"].items()}
    print(json.dumps({"out": a.out.as_posix(), "official": doc["official"], "variants": brief}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.baseline", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="chạy ba biến thể trên test + neg + dev (GPU, trong gpu_lock)")
    r.add_argument("--split-json", type=Path, default=None, help=f"mặc định {SPLIT_REL}")
    r.add_argument("--frames-dir", type=Path, default=None, help=f"mặc định {FRAMES_REL}")
    r.add_argument("--out-dir", type=Path, default=None, help=f"mặc định {OUT_REL}")
    r.add_argument("--limit", type=int, default=0, help="chỉ N khung đầu (chạy thử)")
    r.add_argument("--lock-timeout", type=float, default=6 * 3600.0, help="giây chờ khoá GPU")
    r.add_argument("--json", action="store_true")
    s = sub.add_parser("score", help="chấm ba CSV trên tập test → JSON")
    s.add_argument("--pred-dir", type=Path, default=None, help=f"mặc định {OUT_REL}")
    s.add_argument("--labels-dir", type=Path, default=None, help="mặc định data/real/labels")
    s.add_argument("--manifest", type=Path, default=None, help="mặc định <labels-dir>/manifest.json")
    s.add_argument("--devval-labels-dir", type=Path, default=None, help="mặc định = --labels-dir")
    s.add_argument("--split-json", type=Path, default=None, help=f"mặc định {SPLIT_REL}")
    s.add_argument("--thr", type=float, action="append", default=None, help="ngưỡng vận hành, mặc định 0.25")
    s.add_argument("--boot", type=int, default=BOOT_N)
    s.add_argument("--seed", type=int, default=BOOT_SEED)
    s.add_argument("--allow-missing-labels", action="store_true")
    s.add_argument("--out", type=Path, required=True, help="vd. results/baseline/baseline.json")
    a = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "score": cmd_score}[a.cmd](a)
    except (FileNotFoundError, TimeoutError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
