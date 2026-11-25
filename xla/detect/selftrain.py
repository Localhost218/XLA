"""Phase 11 — thí nghiệm E2: self-training trên khung bay dev thật (nhãn giả + tracker, SimROD rút gọn).

Ba bước, mỗi bước một tiến trình, giữ ``xla.gpu_lock`` riêng (lane khác chen được giữa các bước):

1. **Thầy** (``--teacher``): YOLO11m COCO @1280 fine-tune trên tập nguồn thắng E1 (D2), chọn checkpoint bằng mAP50
   dev-val mỗi epoch (``xla.detect.train.EpochTracker``, giống E1).
2. **Nhãn giả** (bước đầu của ``--round K``): thầy vòng K (K = 1: thầy; K > 1: trò vòng K−1) + TTA dự đoán mọi khung dev,
   theo **từng chuỗi khung dev liên tiếp** (``xla.eval.temporal.consecutive_runs``, khoảng cách < ``max_gap_s``), qua
   ByteTrack của ultralytics với trạng thái giữ suốt chuỗi (đúng ``model.track(persist=True)``: cùng lớp ``BYTETracker``,
   cùng ``update(boxes, img)``; gọi thẳng để giữ cả hộp thô cho luật khung mơ hồ), tracker mới cho mỗi chuỗi. Luật:
   giữ hộp ≥ τ_cao thuộc track dài ≥ L khung; cứu hộp ≥ τ_thấp trong track có hộp ≥ τ_cao; bỏ phát hiện lẻ; bỏ hộp chạm
   mép trái (D11). τ_cao = ngưỡng nhỏ nhất trên lưới cho **precision nhãn giả ≥ 0,9 trên khung dev-val** (nhãn thật).
   Khung dev-val (và khung dev cách nó < ``devval_buffer_s``) không bao giờ vào tập train; khung test / khối test không
   bao giờ được đọc (D10: chỉ split ``dev``); khung ``neg`` vào train với nhãn rỗng.
3. **Trò** (phần còn lại của ``--round K``): YOLO11s @1280 khởi từ YOLO11s D2 của E1, train trên D2 + khung đích nhãn
   giả (lặp ``target_repeat`` lần trong danh sách ảnh để mosaic trộn nguồn/đích), chọn checkpoint bằng dev-val.

``--score`` (CPU): chấm test + neg của mốc, E1 D2, thầy và mọi vòng bằng ``xla.eval.detection`` (ngưỡng F1 dev-val),
so ghép cặp, chọn model ship theo dev-val → ``results/e2/e2.json``. ``--onnx-score <bundle>``: chấm đường ONNX của
bundle (ORT, chính ``detector.onnx``) → ``results/e2/onnx_<tên>.json`` + ``metrics_bundle.json`` (5 khoá §4).

Chạy lại cùng lệnh sau khi máy sập: train tiếp từ ``last.pt``; bước đã xong (``train_done.json``,
``pseudo.json``) thì bỏ qua.

AX::

    python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --teacher --smoke --json   # → results/e2/smoke_teacher.json
    python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --teacher --json
    python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round K --smoke --json   # nhãn giả + smoke trò
    python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round K --json
    python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --score --json             # → results/e2/e2.json
    python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --onnx-score models/release/detector-v1.0.0 --json
    python -m xla.detect.selftrain teacher --config configs/teacher_v2.yaml --smoke --json   # AI v2 phase 05 (cuối file)
    python -m xla.detect.selftrain teacher --config configs/teacher_v2.yaml --json
    python -m xla.detect.selftrain pseudo --config configs/teacher_v2.yaml --json
    # exit 0 xong; 1 lỗi dữ liệu / thiếu run / hết giờ chờ khoá / rò rỉ test; 2 sai tham số
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xla.camera.frames import CROP_LEFT_PX, FRAMES_REL
from xla.detect.baseline import drop_edge, select_frames, write_csv
from xla.detect.train import (
    CSV_MAX_BYTES,
    SYNTH_VAL_KEY,
    EpochTracker,
    _dump,
    _fit,
    _git_commit,
    _gpu_temp,
    _load,
    _now,
    _sha256,
    devval_frames,
    predict,
    read_history,
    smoke_report,
)
from xla.paths import repo_root

CONFIG_REL = Path("configs/selftrain_e2.yaml")
RESULTS_REL = Path("results/e2")
E1_REL = Path("results/e1/e1.json")
SPLIT_REL = Path("data/real/split.json")
LABELS_REL = Path("data/real/labels")
IMG_W, IMG_H = 640, 480


class LeakError(ValueError):
    """Khung test / dev-val lọt vào tập train (D10)."""


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def e1_best(root: Path) -> dict[str, Any]:
    return json.loads((root / E1_REL).read_text(encoding="utf-8"))["best"]


def _split_rows(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))["frames"]


def _role_cfg(cfg: dict[str, Any], role: str, weights: str) -> dict[str, Any]:
    """Cấu hình dạng ``xla.detect.train`` (``weights``, ``train``, ``smoke``) cho ``_fit`` / ``smoke_report``."""
    r = cfg[role]
    return {"weights": weights, "train": dict(cfg["train"]) | {"epochs": r["epochs"], "batch": r["batch"]},
            "smoke": dict(cfg["smoke"]) | {"budget_min_per_run": r["budget_min_per_run"]}, "predict": cfg["predict"]}


def run_dir_of(cfg: dict[str, Any], name: str, smoke: bool = False) -> Path:
    return Path(cfg["runs_dir"]) / (f"{name}_smoke" if smoke else name)


def best_weights(cfg: dict[str, Any], name: str) -> Path:
    return run_dir_of(cfg, name) / "weights" / "best_devval.pt"


# ---------------------------------------------------------------- train một model (thầy / trò)


def free_virtual_gb() -> float | None:
    """Bộ nhớ ảo trống (GB) theo ``Win32_OperatingSystem.FreeVirtualMemory``; không đọc được → None."""
    import subprocess

    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                       capture_output=True, text=True, check=False)
    try:
        return round(int(r.stdout.strip()) / 2**20, 2)
    except ValueError:
        return None


def choose_workers(cfg: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """``workers_max`` khi bộ nhớ ảo trống ≥ ``workers_min_free_virtual_gb`` ngay trước train, không thì ``train.workers``
    (lệnh Owner 18/11 23:00: 4 worker từng làm hết commit memory ở E1)."""
    base, hi = int(cfg["train"]["workers"]), int(cfg.get("workers_max", cfg["train"]["workers"]))
    need = float(cfg.get("workers_min_free_virtual_gb", 10.0))
    free = free_virtual_gb()
    w = hi if free is not None and free >= need else base
    return w, {"free_virtual_gb": free, "need_gb": need, "workers": w, "time": _now()}


def _is_memory_error(e: BaseException) -> bool:
    msg = f"{type(e).__name__}: {e}"
    keys = ("MemoryError", "_ArrayMemoryError", "exception set", "DataLoader worker", "paging file", "out of memory")
    return isinstance(e, MemoryError) or any(k in msg for k in keys)


def train_model(cfg: dict[str, Any], name: str, role: str, weights: str, ds: dict[str, Any],
                smoke: bool) -> dict[str, Any]:
    """Một lần train ``name`` (``teacher`` / ``r<K>``), giữ ``gpu_lock`` suốt train + dự đoán CSV, rồi nhả."""
    from ultralytics.utils.torch_utils import strip_optimizer

    from xla.gpu_lock import gpu_lock

    root = repo_root()
    rcfg = _role_cfg(cfg, role, weights)
    run_dir = run_dir_of(cfg, name, smoke)
    if smoke and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dv = devval_frames(root)
    imgsz = int(cfg["train"]["imgsz"])
    info = _load(run_dir / "run_info.json", {}) or {"name": name, "role": role, "smoke": smoke, "sessions": []}
    info |= {"argv": sys.argv, "git": _git_commit(root), "config": rcfg, "dataset": ds}
    done_p = run_dir / "train_done.json"
    t_wait = time.perf_counter()
    with gpu_lock(f"phase11 e2 {'smoke ' if smoke else ''}train {name}", timeout_s=float(cfg["lock_timeout_s"])):
        t0 = time.perf_counter()
        session: dict[str, Any] = {"start": _now(), "lock_wait_s": round(t0 - t_wait, 1), "gpu_temp_c_start": _gpu_temp()}
        if smoke or not done_p.is_file():
            rcfg["train"]["workers"], session["workers_choice"] = choose_workers(cfg)
            session["workers"] = [rcfg["train"]["workers"]]
            for attempt in range(2):
                tracker = EpochTracker(run_dir, dv, root / FRAMES_REL, cfg["predict"], imgsz)
                try:
                    fit = _fit(rcfg, ds, run_dir, tracker, smoke)
                    break
                except Exception as e:  # chỉ lỗi bộ nhớ của dataloader mới train tiếp, còn lại ném tiếp
                    if attempt or smoke or not _is_memory_error(e) or not (run_dir / "weights" / "last.pt").is_file():
                        raise
                    print(f"[selftrain] lỗi bộ nhớ ({type(e).__name__}: {e}); train tiếp từ last.pt với workers "
                          f"{cfg['train']['workers']}", file=sys.stderr)
                    rcfg["train"]["workers"] = int(cfg["train"]["workers"])
                    session["workers"].append(rcfg["train"]["workers"])
                    session["memory_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            best = run_dir / "weights" / "best_devval.pt"
            strip_optimizer(best)
            _dump(done_p, fit | {"finished": _now(), "best_devval_sha256": _sha256(best)})
            session["resumed"] = fit["resumed"]
        session["train_s"] = round(time.perf_counter() - t0, 1)
        if not smoke:
            session["predict"] = predict_all(cfg, run_dir, name, root, dv)
        session["gpu_s"] = round(time.perf_counter() - t0, 1)
        session["gpu_temp_c_end"] = _gpu_temp()
    import platform

    import torch
    import ultralytics

    info["sessions"].append(session | {"end": _now()})
    info["versions"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0)}
    _dump(run_dir / "run_info.json", info)
    return {"run_dir": run_dir.as_posix(), "session": session, "history": read_history(run_dir / "devval_history.jsonl"),
            "train_done": _load(done_p), "dataset": ds}


def predict_all(cfg: dict[str, Any], run_dir: Path, name: str, root: Path, dv: list[str]) -> dict[str, Any]:
    """CSV dự đoán test / neg / dev-val bằng ``best_devval.pt`` (D11, bỏ hộp chạm mép như E1) → run + repo."""
    import psutil
    from ultralytics import YOLO

    pcfg, imgsz = cfg["predict"], int(cfg["train"]["imgsz"])
    by_split = select_frames(root / SPLIT_REL, tuple(pcfg["splits"])) | {"devval": dv}
    model = YOLO(str(run_dir / "weights" / "best_devval.pt"))
    psutil.cpu_percent(interval=None)
    preds = predict(model, [f for fs in by_split.values() for f in fs], root / FRAMES_REL, pcfg, imgsz)
    out: dict[str, Any] = {"cpu_percent_during": psutil.cpu_percent(interval=None), "csv": {}}
    for sp, fs in by_split.items():
        p = run_dir / "preds" / f"{sp}.csv"
        write_csv(p, {f: preds[f] for f in fs})
        dst = root / RESULTS_REL / "preds" / f"{name}_{sp}.csv"
        if p.stat().st_size < CSV_MAX_BYTES:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        out["csv"][sp] = {"run": p.as_posix(), "repo": dst.as_posix() if dst.is_file() else None,
                          "bytes": p.stat().st_size, "sha256": _sha256(p)}
    return out


def source_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    from xla.detect.train import dataset_info

    return dataset_info(cfg["source_set"], repo_root())  # kiểm luôn: tập nguồn không có khung thật


# ---------------------------------------------------------------- nhãn giả


def teacher_candidates(cfg: dict[str, Any], k: int) -> dict[str, Path]:
    """Thầy vòng ``k``: vòng 1 = các ứng viên ``pseudo.round1_teachers`` (``teacher`` = YOLO11m, ``e1_d2`` = YOLO11s D2
    của E1), chọn bằng chất lượng nhãn giả trên dev-val; vòng k > 1 = trò vòng k − 1."""
    if k > 1:
        return {f"r{k - 1}": best_weights(cfg, f"r{k - 1}")}
    names = cfg["pseudo"].get("round1_teachers", ["teacher"])
    return {n: Path(e1_best(repo_root())["weights"]) if n == "e1_d2" else best_weights(cfg, n) for n in names}


def _quality_key(q: dict[str, Any], pmin: float) -> tuple[float, float, float]:
    """Xếp thầy: đạt precision_min trước, rồi recall nhãn giả, rồi precision (trên khung dev-val)."""
    p, r = q["precision"] or 0.0, q["recall"] or 0.0
    return (1.0 if p >= pmin else 0.0, r, p)


def dev_sequences(rows: list[dict[str, Any]], max_gap_s: float) -> list[list[str]]:
    """Chuỗi khung dev liên tiếp theo giờ (mọi độ dài ≥ 1); CHỈ split ``dev`` (D10)."""
    from xla.eval.temporal import consecutive_runs

    return consecutive_runs(rows, "dev", max_gap_s=max_gap_s, min_len=1)


def _bytetrack_without_lap() -> Any:
    """``BYTETracker`` của ultralytics mà không cần gói ``lap`` (không nằm trong ``uv.lock``; ultralytics tự cài nó khi
    thiếu — luật lane: không cài gói mới). Ghép Hungarian bằng ``use_lap=False`` (``scipy``/``ultralytics.utils.ops``),
    cùng kết quả gán tối ưu."""
    import types

    try:
        import lap  # noqa: F401
    except ImportError:
        sys.modules["lap"] = types.SimpleNamespace(__version__="stub (xla.detect.selftrain: use_lap=False)")
    import functools

    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.utils import matching

    if not getattr(matching.linear_assignment, "_xla_no_lap", False):
        f = functools.partial(matching.linear_assignment, use_lap=False)
        f._xla_no_lap = True  # type: ignore[attr-defined]
        matching.linear_assignment = f
    return BYTETracker


def detect_sequences(weights: Path, seqs: list[list[str]], frames_dir: Path, pcfg: dict[str, Any],
                     imgsz: int) -> dict[str, dict[str, Any]]:
    """Phát hiện thô của thầy (+ TTA) trên mọi khung của các chuỗi (GPU): ``{frame: {"seq", "dets": [[x1,y1,x2,y2,conf]]}}``.

    Tách khỏi phần track (CPU) để chỉnh tham số tracker / luật mà không chạy lại GPU."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[str, dict[str, Any]] = {}
    for si, seq in enumerate(seqs):
        for f in seq:
            im = cv2.imread(str(frames_dir / f), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"không đọc được {frames_dir / f}")
            res = model.predict(im, imgsz=imgsz, conf=pcfg["det_conf"], augment=bool(pcfg["tta"]), device=0,
                                verbose=False, max_det=300)[0]
            out[f] = {"seq": si, "dets": [[float(v) for v in b[:5]] for b in res.boxes.data.cpu().numpy()]}
    return out


def track_sequences(dets: dict[str, dict[str, Any]], seqs: list[list[str]], pcfg: dict[str, Any],
                    ) -> dict[str, dict[str, Any]]:
    """ByteTrack trên phát hiện thô, tracker mới mỗi chuỗi, giữ trạng thái suốt chuỗi (như ``track(persist=True)``);
    thêm ``"tracks": [[x1,y1,x2,y2,tid,conf]...]`` vào từng khung (CPU)."""
    from ultralytics.engine.results import Boxes
    from ultralytics.utils import IterableSimpleNamespace

    BYTETracker = _bytetrack_without_lap()
    targs = IterableSimpleNamespace(**pcfg["tracker"])
    out: dict[str, dict[str, Any]] = {}
    for si, seq in enumerate(seqs):
        tracker = BYTETracker(targs)
        for f in seq:
            d = np.asarray(dets[f]["dets"], dtype=np.float32).reshape(-1, 5)
            boxes = Boxes(np.concatenate([d, np.zeros((len(d), 1), np.float32)], axis=1), (IMG_H, IMG_W))
            tr = np.asarray(tracker.update(boxes)).reshape(-1, 8)
            out[f] = {"seq": si, "dets": dets[f]["dets"],
                      "tracks": [[float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4]), float(t[5])]
                                 for t in tr]}
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (len(a), len(b)) của hộp xyxy."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _clip(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [min(max(x1, 0.0), IMG_W), min(max(y1, 0.0), IMG_H), min(max(x2, 0.0), IMG_W), min(max(y2, 0.0), IMG_H)]


def apply_rules(raw: dict[str, dict[str, Any]], tau_high: float, pcfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Luật nhãn giả → ``{frame: {"boxes": [[x1,y1,x2,y2,conf]...], "use": bool, "why": str}}``."""
    L, tau_low = int(pcfg["min_track_len"]), float(pcfg["tau_low"])
    tracks: dict[tuple[int, int], list[tuple[str, list[float]]]] = {}
    for f, r in raw.items():
        for t in r["tracks"]:
            tracks.setdefault((r["seq"], int(t[4])), []).append((f, t))
    good = {k for k, v in tracks.items() if len(v) >= L and max(t[5] for _, t in v) >= tau_high}
    seq_len: dict[int, int] = {}
    for r in raw.values():
        seq_len[r["seq"]] = seq_len.get(r["seq"], 0) + 1
    out: dict[str, dict[str, Any]] = {}
    for f, r in raw.items():
        boxes = [_clip(t[:4]) + [t[5]] for t in r["tracks"]
                 if (r["seq"], int(t[4])) in good and t[5] >= tau_low]
        boxes = [b for b in boxes if b[2] - b[0] > 1 and b[3] - b[1] > 1]
        boxes = drop_edge([tuple(b) for b in boxes], CROP_LEFT_PX)  # type: ignore[arg-type]
        boxes = [list(b) for b in boxes]
        use, why = True, "ok"
        if seq_len[r["seq"]] < L:
            use, why = False, "chuỗi ngắn hơn L"
        elif pcfg.get("drop_frames_with_unkept_high", True):
            hi = np.asarray([d[:4] for d in r["dets"] if d[4] >= tau_high and d[0] >= CROP_LEFT_PX]).reshape(-1, 4)
            kept = np.asarray([b[:4] for b in boxes]).reshape(-1, 4)
            if len(hi) and (not len(kept) or (_iou(hi, kept).max(axis=1) < float(pcfg["match_iou"])).any()):
                use, why = False, "có hộp ≥ τ_cao bị bỏ (lẻ / track ngắn)"
        out[f] = {"boxes": boxes, "use": use, "why": why}
    return out


def label_quality(labels: dict[str, dict[str, Any]], gt: dict[str, np.ndarray], iou_thr: float) -> dict[str, Any]:
    """Precision / recall nhãn giả trên khung dev-val được dùng (ghép tham lam theo điểm, IoU ≥ ``iou_thr``)."""
    tp = fp = n_gt = 0
    frames = [f for f in gt if f in labels and labels[f]["use"]]
    for f in frames:
        g = gt[f]
        n_gt += len(g)
        boxes = sorted(labels[f]["boxes"], key=lambda b: -b[4])
        used = np.zeros(len(g), bool)
        for b in boxes:
            if len(g):
                iou = _iou(np.asarray([b[:4]]), g)[0]
                iou[used] = -1
                j = int(np.argmax(iou))
                if iou[j] >= iou_thr:
                    used[j] = True
                    tp += 1
                    continue
            fp += 1
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / n_gt if n_gt else None, "tp": tp,
            "fp": fp, "gt": n_gt, "frames_used": len(frames), "frames_devval": len(gt)}


def select_tau_high(raw: dict[str, dict[str, Any]], gt: dict[str, np.ndarray], pcfg: dict[str, Any],
                    ) -> tuple[float, list[dict[str, Any]], str]:
    """τ_cao nhỏ nhất trên lưới có precision nhãn giả ≥ ``precision_min`` trên dev-val; không có → precision cao nhất."""
    grid = []
    for tau in pcfg["tau_high_grid"]:
        q = label_quality(apply_rules(raw, float(tau), pcfg), gt, float(pcfg["match_iou"]))
        grid.append({"tau_high": float(tau)} | q)
    ok = [g for g in grid if g["precision"] is not None and g["precision"] >= float(pcfg["precision_min"])
          and g["tp"] > 0]
    if ok:
        return ok[0]["tau_high"], grid, "nhỏ nhất đạt precision_min"
    best = max((g for g in grid if g["precision"] is not None), key=lambda g: (g["precision"], -g["tau_high"]))
    return best["tau_high"], grid, f"không τ nào đạt {pcfg['precision_min']}: lấy precision cao nhất"


def _yolo_line(b: list[float]) -> str:
    x1, y1, x2, y2 = b[:4]
    return f"0 {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} {(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}"


def build_pseudo_dataset(cfg: dict[str, Any], k: int, labels: dict[str, dict[str, Any]], rows: list[dict[str, Any]],
                         dv: list[str]) -> dict[str, Any]:
    """Ghi ``<datasets_dir>/e2_r<k>/`` (khung đích + nhãn YOLO, ``train.txt`` nguồn + đích, ``dataset.yaml``)."""
    root = repo_root()
    pcfg = cfg["pseudo"]
    split_of = {r["frame"]: r for r in rows}
    dv_t = np.asarray([split_of[f]["t_unix"] for f in dv])
    buf = float(pcfg["devval_buffer_s"])
    chosen, dropped = [], {"devval": 0, "devval_buffer": 0}
    for f, lab in sorted(labels.items()):
        if f in set(dv):
            dropped["devval"] += 1
            continue
        if np.abs(dv_t - split_of[f]["t_unix"]).min() < buf:
            dropped["devval_buffer"] += 1
            continue
        if lab["use"]:
            chosen.append(f)
    neg = sorted(r["frame"] for r in rows if r["split"] == "neg") if pcfg.get("neg_frames", True) else []
    bad = [f for f in chosen if split_of[f]["split"] != "dev"] + [f for f in neg if split_of[f]["split"] != "neg"]
    leak = sorted(set(chosen) & set(dv))
    if bad or leak:
        raise LeakError(f"tập train lẫn khung không phải dev/neg ({bad[:3]}) hoặc dev-val ({leak[:3]})")
    out = Path(cfg["datasets_dir"]) / f"e2_r{k}"
    if out.exists():
        shutil.rmtree(out)
    img_dir, lab_dir = out / "train" / "images", out / "train" / "labels"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    frames_dir = root / FRAMES_REL
    n_box = 0
    for f in chosen + neg:
        shutil.copyfile(frames_dir / f, img_dir / f)
        boxes = labels[f]["boxes"] if f in labels else []
        n_box += len(boxes)
        (lab_dir / f"{Path(f).stem}.txt").write_text("".join(_yolo_line(b) + "\n" for b in boxes), encoding="utf-8",
                                                     newline="\n")
    src = Path(cfg["datasets_dir"]) / cfg["source_set"]
    src_imgs = []
    for p in sorted((src / "train" / "images").iterdir()):  # tập nguồn chép vào (liên kết cứng): cache nhãn nằm ở
        dst = img_dir / p.name                              # đây, không ghi đè labels.cache của tập nguồn
        if dst.exists():
            raise ValueError(f"trùng tên ảnh nguồn / đích: {p.name}")
        try:
            dst.hardlink_to(p)
        except OSError:
            shutil.copyfile(p, dst)
        if (src / "train" / "labels" / f"{p.stem}.txt").is_file():  # không có file nhãn = ảnh nền
            shutil.copyfile(src / "train" / "labels" / f"{p.stem}.txt", lab_dir / f"{p.stem}.txt")
        src_imgs.append(dst)
    target = [img_dir / f for f in chosen]
    lst = [p.as_posix() for p in src_imgs] + [p.as_posix() for p in target] * int(pcfg["target_repeat"]) + \
          [(img_dir / f).as_posix() for f in neg]
    (out / "train.txt").write_text("\n".join(lst) + "\n", encoding="utf-8", newline="\n")
    yml = {"path": out.as_posix(), "train": (out / "train.txt").as_posix(),
           "val": (src / "val" / "images").as_posix(), "names": {0: "person"}}
    (out / "dataset.yaml").write_text(yaml.safe_dump(yml, sort_keys=False), encoding="utf-8", newline="\n")
    n_pos = sum(1 for f in chosen if labels[f]["boxes"])
    return {"set": f"{cfg['source_set']}+e2_r{k}", "yaml": (out / "dataset.yaml").as_posix(),
            "train_txt_sha256": _sha256(out / "train.txt"),
            "n_images": {"train": len(lst), "val": sum(1 for _ in (src / "val" / "images").iterdir())},
            "n_source": len(src_imgs), "n_target_frames": len(chosen), "n_target_with_box": n_pos,
            "n_target_boxes": n_box, "target_repeat": int(pcfg["target_repeat"]), "n_neg_frames": len(neg),
            "dropped_dev": dropped | {"rules": sum(1 for f, lab in labels.items() if not lab["use"])},
            "test_frames_in_train": 0, "devval_frames_in_train": 0}


def pseudo_label(cfg: dict[str, Any], k: int) -> dict[str, Any]:
    """Nhãn giả vòng ``k`` (GPU một lần giữ khoá cho phần dự đoán + track; phần luật chạy CPU). Đã có → đọc lại."""
    from xla.eval.detection import load_labels
    from xla.gpu_lock import gpu_lock

    root = repo_root()
    pdir = Path(cfg["runs_dir"]) / f"pseudo_r{k}"
    rep_p = pdir / "pseudo.json"
    if rep_p.is_file() and Path(_load(rep_p)["dataset"]["yaml"]).is_file():
        return _load(rep_p)
    pdir.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["pseudo"]
    rows = _split_rows(root)
    seqs = dev_sequences(rows, float(pcfg["max_gap_s"]))
    cands = teacher_candidates(cfg, k)
    for n, w in cands.items():
        if not w.is_file():
            raise FileNotFoundError(f"thiếu thầy {n} vòng {k}: {w}")
    todo = {n: w for n, w in cands.items() if not (pdir / f"raw_dets_{n}.json").is_file()}
    if todo:
        t_wait = time.perf_counter()
        with gpu_lock(f"phase11 e2 nhãn giả vòng {k} ({', '.join(todo)})", timeout_s=float(cfg["lock_timeout_s"])):
            wait = time.perf_counter() - t_wait
            for n, w in todo.items():
                t0 = time.perf_counter()
                dets = detect_sequences(w, seqs, root / FRAMES_REL, pcfg, int(cfg["train"]["imgsz"]))
                _dump(pdir / f"raw_dets_{n}.json", {"weights": w.as_posix(), "weights_sha256": _sha256(w),
                                                    "gpu_s": round(time.perf_counter() - t0, 1),
                                                    "lock_wait_s": round(wait, 1), "tta": bool(pcfg["tta"]),
                                                    "frames": dets})
    dv = devval_frames(root)
    gt, _, _ = load_labels(root / LABELS_REL, "devval", dv, allow_missing=False)
    per: dict[str, Any] = {}
    for n, w in cands.items():
        raw_doc = _load(pdir / f"raw_dets_{n}.json")
        if raw_doc["weights_sha256"] != _sha256(w):
            raise ValueError(f"raw_dets_{n}.json là của trọng số khác {w}: xoá {pdir} rồi chạy lại")
        raw = track_sequences(raw_doc["frames"], seqs, pcfg)
        tau, grid, why = select_tau_high(raw, gt, pcfg)
        labels = apply_rules(raw, tau, pcfg)
        per[n] = {"weights": w.as_posix(), "weights_sha256": raw_doc["weights_sha256"], "tau_high": tau,
                  "tau_high_rule": why, "devval_quality": label_quality(labels, gt, float(pcfg["match_iou"])),
                  "grid": grid, "gpu_s": raw_doc["gpu_s"], "_labels": labels}
    pmin = float(pcfg["precision_min"])
    pick = max(per, key=lambda n: _quality_key(per[n]["devval_quality"], pmin))
    labels = per[pick].pop("_labels")
    for v in per.values():
        v.pop("_labels", None)
    ds = build_pseudo_dataset(cfg, k, labels, rows, dv)
    rep = {"phase": "11", "round": k, "created": _now(), "teacher": pick, "teacher_weights": per[pick]["weights"],
           "teacher_sha256": per[pick]["weights_sha256"], "teacher_rule": "precision_min trước, rồi recall nhãn giả "
           "trên dev-val (_quality_key)", "candidates": per, "tta": bool(pcfg["tta"]), "n_sequences": len(seqs),
           "seq_len": {"min": min(map(len, seqs)), "median": float(np.median([len(s) for s in seqs])),
                       "max": max(map(len, seqs)), "n_frames": sum(map(len, seqs))},
           "tau_high": per[pick]["tau_high"], "tau_high_rule": per[pick]["tau_high_rule"], "tau_low": pcfg["tau_low"],
           "min_track_len": pcfg["min_track_len"], "devval_quality": per[pick]["devval_quality"],
           "gpu_s": round(sum(v["gpu_s"] for v in per.values()), 1), "dataset": ds, "config": pcfg}
    _dump(rep_p, rep)
    _dump(root / RESULTS_REL / f"pseudo_r{k}.json", {kk: v for kk, v in rep.items() if kk != "config"})
    return rep


# ---------------------------------------------------------------- lệnh


def cmd_teacher(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    ds = source_dataset(cfg)
    res = train_model(cfg, "teacher", "teacher", cfg["teacher"]["weights"], ds, smoke)
    return _finish(cfg, "teacher", "teacher", res, smoke)


def cmd_round(cfg: dict[str, Any], k: int, smoke: bool) -> dict[str, Any]:
    if not 1 <= k <= int(cfg["student"]["max_rounds"]):
        raise ValueError(f"--round {k} ngoài [1, {cfg['student']['max_rounds']}]")
    rep = pseudo_label(cfg, k)
    w = cfg["student"]["weights"]
    weights = e1_best(repo_root())["weights"] if w == "e1_best" else w
    res = train_model(cfg, f"r{k}", "student", weights, rep["dataset"], smoke)
    out = _finish(cfg, f"r{k}", "student", res, smoke)
    return out | {"pseudo": {kk: rep[kk] for kk in ("tau_high", "tau_high_rule", "devval_quality", "gpu_s")}}


def _finish(cfg: dict[str, Any], name: str, role: str, res: dict[str, Any], smoke: bool) -> dict[str, Any]:
    if smoke:
        rep = smoke_report(_role_cfg(cfg, role, ""), res) | {"phase": "11", "name": name}
        _dump(repo_root() / RESULTS_REL / f"smoke_{name}.json", rep)
        return rep
    return {k: res[k] for k in ("run_dir", "session", "train_done")} | {"history": res["history"]}


# ---------------------------------------------------------------- chấm (CPU) → e2.json

BASELINE_CSV_REL = Path("results/baseline/pred/edge16.csv")
E1_PREDS_REL = Path("results/e1/preds")
BUNDLE_KEYS = ("map50", "map50_95", "precision", "recall", "fp_per_neg_frame")  # hop-dong-model.md §4


def _cached(kind: str, fn: Any, csvs: list[Path], boot: int, seed: int, cache_dir: Path) -> dict[str, Any]:
    """``evaluate`` / ``compare`` nhớ theo sha256 CSV + nhãn + boot + seed (chấm lại e2.json khi thêm vòng không phải
    chạy lại bootstrap của model cũ)."""
    import hashlib

    h = hashlib.sha256(f"{kind}|{boot}|{seed}".encode())
    for p in csvs:
        h.update(_sha256(p).encode())
    for p in sorted((repo_root() / LABELS_REL).rglob("*")):
        if p.is_file():
            h.update(p.name.encode() + p.read_bytes())
    c = cache_dir / f"{kind}_{h.hexdigest()[:24]}.json"
    if c.is_file():
        return _load(c)
    r = fn(*csvs, "test", boot_n=boot, seed=seed)
    _dump(c, r)
    return r


def _combine(parts: list[Path], out: Path) -> Path:
    lines: list[str] = []
    for p in parts:
        rows = p.read_text(encoding="utf-8").splitlines()
        lines += rows if not lines else rows[1:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def rounds_done(cfg: dict[str, Any]) -> list[int]:
    return [k for k in range(1, int(cfg["student"]["max_rounds"]) + 1)
            if (run_dir_of(cfg, f"r{k}") / "train_done.json").is_file()]


def bundle_metrics(summ: dict[str, Any]) -> dict[str, float]:
    """5 khoá §4 từ ``xla.detect.train.summarize`` (ngưỡng vận hành = ngưỡng F1 dev-val)."""
    m = summ["metrics"]
    return {"map50": m["mAP50"]["value"], "map50_95": m["mAP50_95"]["value"], "precision": m["precision@op"]["value"],
            "recall": m["recall@op"]["value"], "fp_per_neg_frame": m["fp_per_neg_frame@op"]["value"]}


def _model_entry(cfg: dict[str, Any], name: str, root: Path, ev: dict[str, Any]) -> dict[str, Any]:
    from xla.detect.train import summarize

    run_dir = run_dir_of(cfg, name)
    hist = read_history(run_dir / "devval_history.jsonl")
    sel = max(hist, key=lambda h: (h["devval_mAP50"], -h["epoch"]))
    info, done = _load(run_dir / "run_info.json"), _load(run_dir / "train_done.json")
    summ = summarize(ev)
    return {"weights": (run_dir / "weights" / "best_devval.pt").resolve().as_posix(),
            "weights_sha256": done["best_devval_sha256"], "selected_by": "devval", "selected_epoch": sel["epoch"],
            "devval_map50": sel["devval_mAP50"],
            "devval_map50_last3_mean": float(np.mean([h["devval_mAP50"] for h in hist[-3:]])),
            "gpu_hours": round(sum(x["gpu_s"] for x in info["sessions"]) / 3600, 4), "epochs": done["epochs"],
            "batch_effective": done["batch_effective"],
            "workers": [w for x in info["sessions"] for w in x.get("workers", [info["config"]["train"]["workers"]])],
            "train_data": info["dataset"], "init_weights":
            info["config"]["weights"], "devval_history": hist, "test": summ, "test_bundle_keys": bundle_metrics(summ),
            "path": "pt: YOLO.predict khung nguyên 640×480, bỏ hộp x1 < 16 (như mốc / E1)"}


def score(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    """Chấm mốc, E1 D2, thầy, mọi vòng (đường ``.pt``) + so ghép cặp; chọn model ship theo dev-val → e2.json."""
    from xla.detect.train import criteria, summarize
    from xla.eval.detection import compare, evaluate

    root = repo_root()
    res_dir, tmp = root / RESULTS_REL, Path(cfg["runs_dir"]) / "score"
    e1 = _load(root / E1_REL)
    base_csv = root / BASELINE_CSV_REL
    csvs = {"edge16": base_csv,
            "e1_d2": _combine([root / E1_PREDS_REL / f"d2_{sp}.csv" for sp in ("test", "neg", "devval")],
                              tmp / "e1_d2_all.csv")}
    names = ["teacher"] + [f"r{k}" for k in rounds_done(cfg)]
    for n in names:
        csvs[n] = _combine([res_dir / "preds" / f"{n}_{sp}.csv" for sp in ("test", "neg", "devval")], tmp / f"{n}_all.csv")
    cache = tmp / "cache"
    evs = {n: _cached("evaluate", evaluate, [p], boot, seed, cache) for n, p in csvs.items()}
    for n, ev in evs.items():
        _dump(res_dir / "eval" / f"{n}_test.json", ev)
    models: dict[str, Any] = {n: _model_entry(cfg, n, root, evs[n]) for n in names}
    e1d2 = e1["sets"]["d2"]
    models["e1_d2"] = {k: e1d2[k] for k in ("weights", "weights_sha256", "selected_by", "selected_epoch",
                                            "devval_map50", "devval_map50_last3_mean", "gpu_hours", "epochs")} | {
        "test": summarize(evs["e1_d2"]), "test_bundle_keys": bundle_metrics(summarize(evs["e1_d2"])),
        "path": "pt: YOLO.predict khung nguyên 640×480, bỏ hộp x1 < 16 (như mốc / E1)", "source": E1_REL.as_posix()}
    cmp = {f"edge16_vs_{n}": _cached("compare", compare, [base_csv, csvs[n]], boot, seed, cache)
           for n in ["e1_d2", *names]}
    cmp |= {f"e1_d2_vs_{n}": _cached("compare", compare, [csvs["e1_d2"], csvs[n]], boot, seed, cache) for n in names}
    for n in ["e1_d2", *names]:
        models[n]["criteria"] = criteria(cmp[f"edge16_vs_{n}"], models[n]["test"], None)
    cands = ["e1_d2"] + [n for n in names if n != "teacher"]
    ship = max(cands, key=lambda n: (models[n]["devval_map50"], -cands.index(n)))
    ship_last3 = max(cands, key=lambda n: (models[n]["devval_map50_last3_mean"], -cands.index(n)))
    pseudo = {f"r{k}": _load(res_dir / f"pseudo_r{k}.json") for k in rounds_done(cfg)}
    smoke = {p.stem: _load(p) for p in sorted(res_dir.glob("smoke_*.json"))}
    gpu_h = sum(m["gpu_hours"] for n, m in models.items() if n != "e1_d2")
    gpu_h += sum(s["gpu_s"] for s in smoke.values()) / 3600 + sum(p["gpu_s"] for p in pseudo.values()) / 3600
    doc = {
        "phase": "11", "created": _now(), "config": cfg,
        "baseline": {"variant": "edge16", "csv": BASELINE_CSV_REL.as_posix(), "test": summarize(evs["edge16"])},
        "models": models, "pseudo": pseudo, "smoke": smoke,
        "compare": {k: {kk: v[kk] for kk in ("pred_a", "pred_b", "op_threshold", "n", "metrics", "bootstrap")}
                    for k, v in cmp.items()},
        "ship": {"name": ship, "weights": models[ship]["weights"], "devval_map50": models[ship]["devval_map50"],
                 "conf_threshold_pt": models[ship]["test"]["conf_threshold"], "threshold_source": "devval",
                 "rule": "devval_map50 (epoch tốt nhất) cao nhất trong {E1 D2, E2 các vòng}; thầy YOLO11m không dự",
                 "candidates": {n: {"devval_map50": models[n]["devval_map50"],
                                    "devval_map50_last3_mean": models[n]["devval_map50_last3_mean"]} for n in cands},
                 "by_last3_mean": ship_last3, "agree_with_last3_mean": ship == ship_last3},
        "gpu_hours_total": round(gpu_h, 4),
    }
    old = _load(res_dir / "e2.json") or {}
    if old.get("bundle"):
        doc["bundle"] = old["bundle"]
    _dump(res_dir / "e2.json", doc)
    return {"e2": (res_dir / "e2.json").as_posix(), "ship": doc["ship"], "gpu_hours_total": doc["gpu_hours_total"]}


def onnx_score(cfg: dict[str, Any], bundle: Path, boot: int, seed: int) -> dict[str, Any]:
    """Chấm đường ONNX của bundle (chính ``detector.onnx``, ORT CPU EP, khung nguyên uint8 BGR, cắt mép trong graph)
    trên test + neg + dev-val; ngưỡng vận hành = ngưỡng F1 dev-val của đường này. Ghi ``bundle`` vào e2.json và
    ``metrics_bundle.json`` (5 khoá §4, đầu vào ``--metrics`` của ``xla.detect.export_onnx``)."""
    from xla import contract as C
    from xla.detect import export_onnx as X
    from xla.detect.train import criteria, summarize
    from xla.eval.detection import compare, evaluate

    root = repo_root()
    res_dir = root / RESULTS_REL
    onnx_path = bundle / "detector.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"thiếu {onnx_path}")
    e2 = _load(res_dir / "e2.json")
    if e2 is None:
        raise FileNotFoundError(f"thiếu {res_dir / 'e2.json'} (chạy --score trước)")
    sha = C.sha256_file(onnx_path)
    by_split = select_frames(root / SPLIT_REL, tuple(cfg["predict"]["splits"])) | {"devval": devval_frames(root)}
    sess = X.ort_session(onnx_path)
    preds: dict[str, tuple[list[Any], float]] = {}
    t0 = time.perf_counter()
    for f in [f for fs in by_split.values() for f in fs]:
        im = C.decode_frame(root / FRAMES_REL / f)
        t1 = time.perf_counter()
        boxes, scores = X.run_detector(sess, im)
        ms = (time.perf_counter() - t1) * 1000  # ORT CPU EP, chỉ để tham khảo (ms bundle: meta.ms_per_frame, CUDA)
        preds[f] = ([(*map(float, b), float(s)) for b, s in zip(boxes, scores, strict=True)], ms)
    wall = time.perf_counter() - t0
    name = bundle.name
    csv_p = res_dir / "preds" / f"onnx_{name}_all.csv"
    write_csv(csv_p, preds)
    ev = evaluate(csv_p, "test", boot_n=boot, seed=seed)
    _dump(res_dir / "eval" / f"onnx_{name}_test.json", ev)
    summ = summarize(ev)
    metrics = bundle_metrics(summ)
    cmp_base = compare(root / BASELINE_CSV_REL, csv_p, "test", boot_n=boot, seed=seed)
    meta = _load(bundle / C.META_FILE) or {}
    ms_p95 = (meta.get("ms_per_frame") or {}).get("p95")
    crit = criteria(cmp_base, summ, None)
    crit["B3"] = {"pass": ms_p95 is not None and ms_p95 <= 30.0 and "CUDA" in meta["ms_per_frame"]["runtime"],
                  "ms_p50": (meta.get("ms_per_frame") or {}).get("p50"), "ms_p95": ms_p95,
                  "runtime": (meta.get("ms_per_frame") or {}).get("runtime"),
                  "cpu_perf_pct": (meta.get("ms_per_frame") or {}).get("cpu_perf_pct"),
                  "rule": "≤ 30 ms p95 trên ORT CUDA EP (meta.ms_per_frame của bundle)"}
    ship = e2["ship"]["name"]
    pt_csv = Path(cfg["runs_dir"]) / "score" / f"{ship}_all.csv"
    cmp_pt = compare(pt_csv, csv_p, "test", boot_n=boot, seed=seed) if pt_csv.is_file() else None
    entry = {"bundle": bundle.as_posix(), "onnx_sha256": sha, "model": ship, "conf_threshold": summ["conf_threshold"],
             "threshold_source": "devval (đường ONNX)", "metrics": metrics, "test": summ, "criteria": crit,
             "path": "onnx: detector.onnx (ORT CPU EP) trên khung uint8 BGR 640×480 nguyên, cắt mép trong graph, "
                     "letterbox vuông 1280, score_floor 0,05, NMS trong graph",
             "is_bundle_metrics": True, "predict_wall_s": round(wall, 1), "csv": csv_p.as_posix(),
             "compare_vs_edge16": {kk: cmp_base[kk] for kk in ("op_threshold", "n", "metrics")},
             "compare_pt_vs_onnx": ({kk: cmp_pt[kk] for kk in ("op_threshold", "n", "metrics")} if cmp_pt else None)}
    e2["bundle"] = entry
    _dump(res_dir / "e2.json", e2)
    _dump(res_dir / "metrics_bundle.json", {"metrics": metrics, "conf_threshold": summ["conf_threshold"],
                                             "onnx_sha256": sha, "model": ship})
    return {k: entry[k] for k in ("bundle", "onnx_sha256", "model", "conf_threshold", "metrics", "predict_wall_s")}


# ================================================================ AI v2 (phase 05): thầy @1280 trên d02x + nhãn giả
#
# ``python -m xla.detect.selftrain teacher|pseudo --config configs/teacher_v2.yaml [--smoke] --json``. Khác E2: tập
# nguồn và trọng số thầy cấu hình được (``source_set``, ``teacher.candidates``), chọn checkpoint bằng mAP50 val của tập
# nguồn (không khung thật có nhãn nào), khung thật vào train theo danh sách cho phép (``train.check_real_frames``), τ_cao
# của nhãn giả chọn theo fold, mọi khung nhãn giả mang ``session`` + ``fold`` (``folds.json``).

V2_RESULTS_REL = Path("results/v2/teacher")
FOLDS_REL = Path("data/real/labels/folds.json")
DEVPOOL_LABELS_REL = Path("data/real/labels/devpool")
NEG_SPLIT_REL = Path("data/real/neg_flight_split.json")
V1_BUNDLE = Path("D:/XLA/models/release/detector-v1.0.0")  # chỉ đọc: .onnx không có trong worktree (gitignore)


def resolve_weights(w: str | Path) -> Path:
    from xla.paths import downloads_dir

    p = Path(w)
    return p if p.is_absolute() else downloads_dir() / p


def machine_info() -> dict[str, Any]:
    """Khối ``machine`` kèm mọi số đo giờ/ms (GPU có hạ xung không, nhiệt, CPU)."""
    import os
    import platform
    import subprocess

    q = "name,driver_version,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,pstate,memory.used,memory.total"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    gpu = dict(zip(q.split(","), [v.strip() for v in r.stdout.strip().split(",")], strict=False)) if r.stdout else {}
    try:
        import psutil

        cpu = {"percent": psutil.cpu_percent(interval=0.5), "freq_mhz": getattr(psutil.cpu_freq(), "current", None),
               "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1)}
    except Exception:  # noqa: BLE001 — psutil hỏng không chặn phép đo
        cpu = {}
    return {"time": _now(), "host": platform.node(), "os": platform.platform(), "cpu_count": os.cpu_count(),
            "gpu": gpu, "cpu": cpu}


def source_rows(set_name: str) -> list[dict[str, Any]]:
    from xla.paths import data_root

    p = data_root() / set_name / "manifest.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sample_by_source(rows: list[dict[str, Any]], split: str, n: int, seed: int) -> list[str]:
    """``n`` ảnh của ``split`` lấy đều theo tỉ lệ từng ``source`` (seed cố định); ``n`` ≥ cỡ tập → lấy hết."""
    by: dict[str, list[str]] = {}
    for r in rows:
        if r["split"] == split:
            by.setdefault(r.get("source", "?"), []).append(r["name"])
    total = sum(map(len, by.values()))
    if n >= total:
        return sorted(x for v in by.values() for x in v)
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for src in sorted(by):
        names = sorted(by[src])
        k = min(len(names), max(1, round(n * len(names) / total)))
        out += [names[i] for i in sorted(rng.choice(len(names), k, replace=False))]
    return sorted(out)


def split_rows_by_frame(root: Path) -> dict[str, dict[str, Any]]:
    return {r["frame"]: r for r in _split_rows(root)}


def neg_split_doc(root: Path) -> dict[str, Any]:
    return json.loads((root / NEG_SPLIT_REL).read_text(encoding="utf-8"))


def frame_sessions(root: Path) -> dict[str, str]:
    """Phiên của mọi khung bay đã nhập (``manifest.csv`` + ``sessions.json``)."""
    from xla.camera.sessions import flight_rows

    return {r["frame"]: r["session"] for r in flight_rows(root) if r["session"]}


def fold_sessions(root: Path, sess: dict[str, str]) -> list[set[str]]:
    """Phiên của từng fold (``folds.json`` group = session): fold i loại mọi khung thật của các phiên này."""
    doc = json.loads((root / FOLDS_REL).read_text(encoding="utf-8"))
    if doc.get("groups") != "session":
        raise ValueError(f"folds.json groups = {doc.get('groups')!r}, cần 'session'")
    out = [{sess[f] for f in fold} for fold in doc["folds"]]
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[i] & out[j]:
                raise ValueError(f"fold {i} và {j} chung phiên {sorted(out[i] & out[j])}")
    return out


def build_v2_dataset(out: Path, source_set: str, train_names: list[str], val_names: list[str],
                     real: dict[str, list[list[float]]], pseudo_sessions: dict[str, str], root: Path,
                     excluded_sessions: set[str] | frozenset[str] = frozenset(), repeat_real: int = 1,
                     mine_max_frac: float = 0.10) -> dict[str, Any]:
    """Tập huấn luyện ``out/`` = ảnh nguồn (liên kết cứng) + khung thật ``real`` (``{frame: hộp px}``; rỗng = ảnh nền).

    Khung thật kiểm bằng danh sách cho phép (``train.check_real_frames``) TRƯỚC khi ghi gì. Cùng khoá (danh sách + nhãn)
    thì dùng lại tập đã dựng (train tiếp sau khi máy sập không đổi dữ liệu)."""
    import hashlib

    from xla.detect.train import check_real_frames
    from xla.paths import data_root

    allowed = check_real_frames(sorted(real), root, pseudo=pseudo_sessions, excluded_sessions=excluded_sessions,
                                n_total=len(train_names) + len(real) * repeat_real, mine_max_frac=mine_max_frac)
    key = hashlib.sha256(json.dumps([source_set, train_names, val_names, sorted(real.items()), repeat_real],
                                    sort_keys=True).encode()).hexdigest()
    info_p = out / "dataset_info.json"
    old = _load(info_p)
    if old and old.get("key") == key and (out / "dataset.yaml").is_file():
        return old
    if out.exists():
        shutil.rmtree(out)
    src = data_root() / source_set
    lists: dict[str, list[str]] = {"train": [], "val": []}
    for sp, names in (("train", train_names), ("val", val_names)):
        img_dir, lab_dir = out / sp / "images", out / sp / "labels"
        img_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)
        for n in names:
            dst = img_dir / n
            try:
                dst.hardlink_to(src / sp / "images" / n)
            except OSError:
                shutil.copyfile(src / sp / "images" / n, dst)
            lab = src / sp / "labels" / f"{Path(n).stem}.txt"
            if lab.is_file():
                shutil.copyfile(lab, lab_dir / lab.name)
            lists[sp].append(dst.as_posix())
    img_dir, lab_dir = out / "train" / "images", out / "train" / "labels"
    n_box = 0
    for f, boxes in sorted(real.items()):
        if (img_dir / f).exists():
            raise ValueError(f"trùng tên ảnh nguồn / khung thật: {f}")
        shutil.copyfile(root / FRAMES_REL / f, img_dir / f)
        (lab_dir / f"{Path(f).stem}.txt").write_text("".join(_yolo_line(b) + "\n" for b in boxes), encoding="utf-8",
                                                     newline="\n")
        n_box += len(boxes)
        lists["train"] += [(img_dir / f).as_posix()] * repeat_real
    for sp, paths in lists.items():
        (out / f"{sp}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
    yml = {"path": out.as_posix(), "train": (out / "train.txt").as_posix(), "val": (out / "val.txt").as_posix(),
           "names": {0: "person"}}
    (out / "dataset.yaml").write_text(yaml.safe_dump(yml, sort_keys=False), encoding="utf-8", newline="\n")
    info = {"key": key, "yaml": (out / "dataset.yaml").as_posix(), "source_set": source_set,
            "n_images": {"train": len(lists["train"]), "val": len(lists["val"])}, "n_source_train": len(train_names),
            "n_real_frames": len(real), "n_real_with_box": sum(1 for b in real.values() if b), "n_real_boxes": n_box,
            "repeat_real": repeat_real, "real_allowlist": allowed, "excluded_sessions": sorted(excluded_sessions),
            "train_txt_sha256": _sha256(out / "train.txt"), "created": _now()}
    _dump(info_p, info)
    return info


class ValTracker:
    """Callback ultralytics: giờ từng epoch, VRAM đỉnh, mAP50 val (tập nguồn) → ``best_val50.pt``; dừng sớm theo mAP50."""

    def __init__(self, run_dir: Path, patience: int, cap_s: float | None = None) -> None:
        self.run_dir, self.patience, self.cap_s = run_dir, patience, cap_s
        self.hist_path = run_dir / "val_history.jsonl"
        self.history = read_history(self.hist_path)
        self.best = max((h["val_mAP50"] for h in self.history), default=-1.0)
        self.t_call = time.perf_counter()
        self.batch_t: list[float] = []
        self.t0 = self.t_train = 0.0
        self.vram_gb = 0.0
        self.pending = False
        self.stopped: str | None = None

    def on_train_epoch_start(self, tr: Any) -> None:
        self.t0 = time.perf_counter()
        self.batch_t = []

    def on_train_batch_end(self, tr: Any) -> None:
        self.batch_t.append(time.perf_counter())

    def on_train_epoch_end(self, tr: Any) -> None:
        import torch

        self.t_train = time.perf_counter()
        self.pending = True
        self.vram_gb = torch.cuda.max_memory_reserved(0) / 2**30

    def on_fit_epoch_end(self, tr: Any) -> None:
        if not self.pending:
            return
        self.pending = False
        m50 = float((tr.metrics or {}).get(SYNTH_VAL_KEY, 0.0) or 0.0)
        gaps = np.diff(self.batch_t) if len(self.batch_t) > 1 else np.asarray([np.nan])
        rec = {"epoch": int(tr.epoch) + 1, "val_mAP50": m50,
               "val_mAP50_95": float((tr.metrics or {}).get("metrics/mAP50-95(B)", 0.0) or 0.0),
               "train_s": round(self.t_train - self.t0, 2), "val_save_s": round(time.perf_counter() - self.t_train, 2),
               "batches": len(self.batch_t), "median_batch_s": round(float(np.nanmedian(gaps)), 4),
               "batch": int(tr.batch_size), "vram_peak_gb": round(self.vram_gb, 2), "time": _now()}
        self.history = [h for h in self.history if h["epoch"] != rec["epoch"]] + [rec]
        with self.hist_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if m50 > self.best:
            self.best = m50
            shutil.copyfile(Path(tr.last), self.run_dir / "weights" / "best_val50.pt")
        best_ep = max(self.history, key=lambda h: (h["val_mAP50"], -h["epoch"]))["epoch"]
        if rec["epoch"] - best_ep >= self.patience:
            tr.stop, self.stopped = True, f"mAP50 val không tăng {self.patience} epoch (tốt nhất epoch {best_ep})"
        elif self.cap_s is not None and time.perf_counter() - self.t_call > self.cap_s:
            tr.stop, self.stopped = True, f"chạm trần giờ {self.cap_s / 3600:.2f} h"


TRACK_EVENTS = ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end", "on_fit_epoch_end")


def train_v2(cfg: dict[str, Any], name: str, weights: Path, ds: dict[str, Any], targs: dict[str, Any],
             patience: int, label: str, cap_s: float | None = None, select: str = "val") -> dict[str, Any]:
    """Một lần train (thầy / trò v2), giữ ``gpu_lock`` suốt train rồi nhả; train tiếp từ ``last.pt`` nếu máy sập.

    ``select="val"``: ``best.pt`` = ``best_val50.pt`` (mAP50 val tập nguồn); ``"last"``: epoch cuối."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import strip_optimizer

    from xla.gpu_lock import gpu_lock

    root = repo_root()
    run_dir = Path(cfg["runs_dir"]) / name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    done_p = run_dir / "train_done.json"
    info = _load(run_dir / "run_info.json", {}) or {"name": name, "sessions": []}
    info |= {"argv": sys.argv, "git": _git_commit(root), "weights_init": weights.as_posix(), "dataset": ds,
             "train_args": targs}
    if done_p.is_file():
        return {"run_dir": run_dir.as_posix(), "train_done": _load(done_p), "skipped": True,
                "history": read_history(run_dir / "val_history.jsonl")}
    t_wait = time.perf_counter()
    with gpu_lock(label, timeout_s=float(cfg["lock_timeout_s"])):
        t0 = time.perf_counter()
        session: dict[str, Any] = {"start": _now(), "lock_wait_s": round(t0 - t_wait, 1),
                                   "machine_start": machine_info()}
        last = run_dir / "weights" / "last.pt"
        tracker = ValTracker(run_dir, patience, cap_s)
        resumed = last.is_file()
        model = YOLO(str(last if resumed else weights))
        for ev in TRACK_EVENTS:
            model.add_callback(ev, getattr(tracker, ev))
        if resumed:
            model.train(resume=True, workers=int(targs["workers"]))
        else:
            model.train(data=ds["yaml"], project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **targs)
        best_src = run_dir / "weights" / ("best_val50.pt" if select == "val" else "last.pt")
        if not best_src.is_file():
            best_src = last
        best = run_dir / "weights" / "best.pt"
        if best_src != best:
            shutil.copyfile(best_src, best)
        strip_optimizer(best)
        session |= {"resumed": resumed, "gpu_s": round(time.perf_counter() - t0, 1), "machine_end": machine_info(),
                    "stopped": tracker.stopped}
    import torch
    import ultralytics

    hist = read_history(run_dir / "val_history.jsonl")
    done = {"finished": _now(), "select": select, "selected_from": best_src.name, "best_sha256": _sha256(best),
            "epochs_run": len(hist), "stopped": tracker.stopped,
            "best_epoch": (max(hist, key=lambda h: (h["val_mAP50"], -h["epoch"]))["epoch"] if hist else None)}
    _dump(done_p, done)
    info["sessions"].append(session | {"end": _now()})
    info["versions"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0)}
    _dump(run_dir / "run_info.json", info)
    return {"run_dir": run_dir.as_posix(), "train_done": done, "session": session, "history": hist}


def gpu_hours_of(run_dir: Path) -> float:
    info = _load(run_dir / "run_info.json") or {}
    return round(sum(s.get("gpu_s", 0.0) for s in info.get("sessions", [])) / 3600, 4)


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def teacher_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Đo VRAM đỉnh + giây/ảnh mỗi cỡ thầy (vài chục bước, không phải một epoch), rồi chọn cỡ theo Deferred 8 + trần giờ."""
    from xla.gpu_lock import gpu_lock

    root, sm, tc = repo_root(), cfg["smoke"], cfg["teacher"]
    rows = source_rows(cfg["source_set"])
    tr_names = sample_by_source(rows, "train", int(sm["n_train"]), int(cfg["seed"]))
    va_names = sample_by_source(rows, "val", int(sm["n_val"]), int(cfg["seed"]))
    ds = build_v2_dataset(Path(cfg["datasets_dir"]) / "v2_teacher_smoke", cfg["source_set"], tr_names, va_names, {}, {},
                          root)
    out: dict[str, Any] = {"phase": "05", "created": _now(), "n_train": len(tr_names), "n_val": len(va_names),
                           "batch": int(tc["batch"]), "imgsz": int(cfg["train"]["imgsz"]), "sizes": {}}
    t_wait = time.perf_counter()
    with gpu_lock("v2 phase05 smoke thầy", timeout_s=float(cfg["lock_timeout_s"])):
        t_lock = time.perf_counter()
        out["lock_wait_s"] = round(t_lock - t_wait, 1)
        out["machine"] = machine_info()
        for size in sm["sizes"]:
            import torch

            run_dir = Path(cfg["runs_dir"]) / f"teacher_smoke_{size}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            (run_dir / "weights").mkdir(parents=True)
            if torch.cuda.is_initialized():  # VRAM đỉnh đo riêng từng cỡ
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)
            t0 = time.perf_counter()
            tracker = ValTracker(run_dir, 99)
            from ultralytics import YOLO

            model = YOLO(str(resolve_weights(tc["candidates"][size])))
            for ev in TRACK_EVENTS:
                model.add_callback(ev, getattr(tracker, ev))
            targs = dict(cfg["train"]) | {"batch": int(tc["batch"]), "epochs": 1, "close_mosaic": 0}
            try:
                model.train(data=ds["yaml"], project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **targs)
                h = tracker.history[-1]
                s_img = h["median_batch_s"] / h["batch"]
                val_s_img = h["val_save_s"] / len(va_names)
                n_tr = int(cfg["data"]["n_source_train"]) + 85  # + khung mine
                epoch_s = s_img * n_tr + val_s_img * int(cfg["data"]["n_source_val"])
                fits = h["vram_peak_gb"] <= float(tc["vram_max_gb"]) and h["batch"] >= 2
                img_iters = float(tc["time_h"]) * 3600 / epoch_s * n_tr
                rec = {"fits_8gb": fits, "vram_peak_gb": h["vram_peak_gb"], "batch_effective": h["batch"],
                       "median_batch_s": h["median_batch_s"], "s_per_img_train": round(s_img, 4),
                       "s_per_img_val": round(val_s_img, 4), "batches": h["batches"],
                       "hours_per_epoch_est": round(epoch_s / 3600, 4), "epochs_in_time_est": round(
                           float(tc["time_h"]) * 3600 / epoch_s, 2), "img_iters_in_time_est": int(img_iters)}
            except Exception as e:  # OOM = không vừa; lỗi khác ném tiếp
                if not _is_oom(e):
                    raise
                rec = {"fits_8gb": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rec |= {"weights": str(resolve_weights(tc["candidates"][size])), "gpu_s": round(time.perf_counter() - t0, 1)}
            out["sizes"][size] = rec
            del model
            torch.cuda.empty_cache()
        out["gpu_s"] = round(time.perf_counter() - t_lock, 1)
    out["machine_end"] = machine_info()
    lrec = out["sizes"].get("l", {})
    use_l = bool(lrec.get("fits_8gb")) and lrec.get("img_iters_in_time_est", 0) >= int(tc["min_img_iters"])
    out["chosen"] = "l" if use_l else "m"
    out["rule"] = (f"l nếu vừa batch ≥ 2 @1280 + AMP (VRAM ≤ {tc['vram_max_gb']} GB) VÀ trong time_h {tc['time_h']} h "
                   f"học được ≥ {tc['min_img_iters']} ảnh (trần 1 giờ GPU của chủ dự án), không thì m")
    _dump(repo_root() / V2_RESULTS_REL / "smoke.json", out)
    return out


def teacher_size(cfg: dict[str, Any]) -> str:
    size = cfg["teacher"]["size"]
    if size == "auto":
        sm = _load(repo_root() / V2_RESULTS_REL / "smoke.json")
        if not sm:
            raise FileNotFoundError("teacher.size: auto cần results/v2/teacher/smoke.json (chạy teacher --smoke)")
        size = sm["chosen"]
    if size not in cfg["teacher"]["candidates"]:
        raise ValueError(f"cỡ thầy {size!r} không có trong teacher.candidates")
    return size


def teacher_dataset(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    rows = source_rows(cfg["source_set"])
    d = cfg["data"]
    tr = sample_by_source(rows, "train", int(d["n_source_train"]), int(cfg["seed"]))
    va = sample_by_source(rows, "val", int(d["n_source_val"]), int(cfg["seed"]))
    mine = {f["frame"]: [] for f in neg_split_doc(root)["frames"] if f["split"] == "mine"} if d["mine_background"] else {}
    return build_v2_dataset(Path(cfg["datasets_dir"]) / "v2_teacher", cfg["source_set"], tr, va, mine, {}, root,
                            mine_max_frac=float(d["mine_max_frac"]))


def teacher_train(cfg: dict[str, Any]) -> dict[str, Any]:
    root, tc = repo_root(), cfg["teacher"]
    size = teacher_size(cfg)
    ds = teacher_dataset(cfg, root)
    targs = dict(cfg["train"]) | {"batch": int(tc["batch"]), "epochs": int(tc["epochs"]), "time": float(tc["time_h"])}
    res = train_v2(cfg, "teacher", resolve_weights(tc["candidates"][size]), ds, targs, int(tc["early_stop_patience"]),
                   f"v2 phase05 train thầy YOLO11{size}")
    return {"size": size} | {k: res[k] for k in ("run_dir", "train_done", "history")}


def _write_preds_csv(path: Path, preds: dict[str, tuple[list[Any], float]]) -> Path:
    write_csv(path, preds)
    return path


def onnx_predict(onnx_path: Path, frames: list[str], root: Path, cuda: bool = False) -> dict[str, tuple[list, float]]:
    """Dự đoán bằng chính ``detector.onnx`` (ORT, khung uint8 BGR nguyên, cắt mép trong graph) → ``{frame: (hộp, ms)}``."""
    from xla import contract as C
    from xla.detect import export_onnx as X

    sess = X.ort_session(onnx_path, cuda=cuda)
    out = {}
    for f in frames:
        im = C.decode_frame(root / FRAMES_REL / f)
        t1 = time.perf_counter()
        boxes, scores = X.run_detector(sess, im)
        out[f] = ([(*map(float, b), float(s)) for b, s in zip(boxes, scores, strict=True)],
                  (time.perf_counter() - t1) * 1000)
    return out


def map50_by_fold(csv_p: Path, root: Path) -> list[float | None]:
    """mAP50 điểm của từng fold (khung devpool của fold; báo độ lệch giữa fold, không CI)."""
    from xla.eval.detection import _coco_eval, accumulate, load_labels, read_pred_csv

    folds = json.loads((root / FOLDS_REL).read_text(encoding="utf-8"))["folds"]
    known = set(split_rows_by_frame(root))
    preds, _ = read_pred_csv(csv_p, known)
    out: list[float | None] = []
    for fold in folds:
        gt, _, _ = load_labels(root / LABELS_REL, "devpool", sorted(fold), allow_missing=False)
        evals, _ = _coco_eval(sorted(gt), gt, preds)
        p, _ = accumulate(evals, "all")
        out.append(None if not np.isfinite(p[0]) else round(float(p[0]), 4))
    return out


def devpool_summary(ev: dict[str, Any]) -> dict[str, Any]:
    op = round(ev["f1_threshold_from_devval"]["thr"], 6)
    m = ev["metrics"]
    keys = ("mAP50", "mAP50_95", "AP_small", "AR_small")
    out = {k: m[k] for k in keys}
    for k in ("precision", "recall", "f1", "fp_per_neg_flight", "fp_per_neg_indoor"):
        nk = {"fp_per_neg_flight": "fp_per_devpool_empty_frame", "fp_per_neg_indoor": "fp_per_mine_frame"}.get(k, k)
        out[f"{nk}@op"] = m[f"{k}@{op:g}"]
    return {"op_threshold": op, "threshold_source": "F1 tối đa trên devpool OOF", "metrics": out, "n": ev["n"]}


def pseudo_v2(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Nhãn giả thầy v2 + TTA trên khung dev v1 cần cho C2 (không test/cal/holdout/mine), làm sạch ByteTrack, τ_cao theo
    fold; chấm thầy và detector-v1 trên devpool (OOF với thầy). Ghi ``pseudo.out_dir`` + ``results/v2/teacher``."""
    from ultralytics import YOLO

    from xla.eval.detection import BOOT_N, BOOT_SEED, compare, evaluate, load_labels
    from xla.gpu_lock import gpu_lock

    root, pcfg = repo_root(), cfg["pseudo"]
    res_dir = root / V2_RESULTS_REL
    tw = Path(cfg["runs_dir"]) / "teacher" / "weights" / "best.pt"
    if not tw.is_file():
        raise FileNotFoundError(f"thiếu thầy {tw} (chạy teacher trước)")
    split = split_rows_by_frame(root)
    neg = neg_split_doc(root)["frames"]
    reserved = {f["frame"] for f in neg if f["split"] in ("cal", "holdout")}
    mine = sorted(f["frame"] for f in neg if f["split"] == "mine")
    sess = frame_sessions(root)
    fsess = fold_sessions(root, sess)
    fold_of = {s: i for i, ss in enumerate(fsess) for s in ss}
    targets = sorted(f for f, r in split.items() if r["split"] == "dev" and f not in reserved and f not in set(mine))
    lost = [f for f in targets if sess.get(f) not in fold_of]
    if lost:
        raise ValueError(f"{len(lost)} khung dev không thuộc phiên nào của folds.json (vd. {lost[:3]})")
    seqs = dev_sequences([split[f] for f in targets], float(pcfg["max_gap_s"]))
    if smoke:
        seqs = seqs[:5]
    dp = json.loads((root / FOLDS_REL).read_text(encoding="utf-8"))["folds"]
    dp_all = sorted(f for fold in dp for f in fold)
    pdir = Path(cfg["runs_dir"]) / ("pseudo_smoke" if smoke else "pseudo")
    pdir.mkdir(parents=True, exist_ok=True)
    raw_p, t_csv = pdir / "raw_dets.json", res_dir / "preds" / "teacher_devpool.csv"
    raw_doc = _load(raw_p)
    gpu_s = 0.0
    if not raw_doc or raw_doc["weights_sha256"] != _sha256(tw) or not t_csv.is_file():
        t_wait = time.perf_counter()
        with gpu_lock("v2 phase05 nhãn giả + chấm devpool thầy", timeout_s=float(cfg["lock_timeout_s"])):
            t0 = time.perf_counter()
            mach = machine_info()
            dets = detect_sequences(tw, seqs, root / FRAMES_REL, pcfg, int(cfg["train"]["imgsz"]))
            t1 = time.perf_counter()
            preds = predict(YOLO(str(tw)), dp_all + mine, root / FRAMES_REL, cfg["predict"], int(cfg["train"]["imgsz"]),
                            warmup=3)
            t_csv.parent.mkdir(parents=True, exist_ok=True)
            write_csv(t_csv, preds)
            gpu_s = time.perf_counter() - t0
            raw_doc = {"weights": tw.as_posix(), "weights_sha256": _sha256(tw), "tta": bool(pcfg["tta"]),
                       "gpu_s": round(gpu_s, 1), "gpu_s_tta": round(t1 - t0, 1), "lock_wait_s": round(t0 - t_wait, 1),
                       "machine": mach, "frames": dets}
            _dump(raw_p, raw_doc)
    raw = track_sequences(raw_doc["frames"], seqs, pcfg)
    gt, _, _ = load_labels(root / LABELS_REL, "devpool", dp_all, allow_missing=False)
    per_fold, labels_by_fold = [], []
    for j, fold in enumerate(dp):
        gt_j = {f: gt[f] for f in fold if f in raw}
        tau, grid, why = select_tau_high(raw, gt_j, pcfg)
        lab = apply_rules(raw, tau, pcfg)
        labels_by_fold.append(lab)
        per_fold.append({"fold": j, "sessions": sorted(fsess[j]), "tau_high": tau, "tau_high_rule": why,
                         "quality_on_own_devpool": label_quality(lab, gt_j, float(pcfg["match_iou"])), "grid": grid})
    out_dir = Path(pcfg["out_dir"] + ("_smoke" if smoke else ""))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "labels").mkdir(parents=True)
    rows, n_box = [], 0
    for f in sorted(raw):
        j = fold_of[sess[f]]
        lab = labels_by_fold[j][f]
        if split[f]["split"] != "dev" or f in reserved:
            raise LeakError(f"khung nhãn giả không phải dev hoặc là cal/holdout: {f}")
        if lab["use"]:
            (out_dir / "labels" / f"{Path(f).stem}.txt").write_text(
                "".join(_yolo_line(b) + "\n" for b in lab["boxes"]), encoding="utf-8", newline="\n")
            n_box += len(lab["boxes"])
        rows.append({"frame": f, "session": sess[f], "fold": j, "use": lab["use"], "why": lab["why"],
                     "boxes": [[round(v, 2) for v in b] for b in lab["boxes"]]})
    man = {"version": "pseudo_v2", "created": _now(), "teacher": tw.as_posix(), "teacher_sha256": raw_doc["weights_sha256"],
           "tta": raw_doc["tta"], "rule": "τ_cao của fold j = nhỏ nhất đạt precision ≥ 0,9 trên devpool fold j "
           "(model fold i ≠ j mới huấn luyện trên nhãn này); khung test/cal/holdout/mine không bao giờ ở đây",
           "n_sequences": len(seqs), "frames": rows, "per_fold": per_fold}
    _dump(out_dir / "manifest.json", man)
    if not smoke:
        _dump(res_dir / "pseudo_manifest.json", man)  # bản trong repo (test Accept đọc bản này)
    summary: dict[str, Any] = {"n_frames": len(rows), "n_used": sum(r["use"] for r in rows),
                               "n_used_with_box": sum(1 for r in rows if r["use"] and r["boxes"]), "n_boxes": n_box,
                               "by_fold": [{"fold": j, "frames": sum(1 for r in rows if r["fold"] == j),
                                            "used": sum(1 for r in rows if r["fold"] == j and r["use"]),
                                            "boxes": sum(len(r["boxes"]) for r in rows if r["fold"] == j and r["use"])}
                                           for j in range(len(dp))],
                               "dropped": {w: sum(1 for r in rows if r["why"] == w) for w in {r["why"] for r in rows}}}
    doc: dict[str, Any] = {"phase": "05", "created": _now(), "teacher_weights": tw.as_posix(),
                           "teacher_sha256": raw_doc["weights_sha256"], "pseudo_dir": out_dir.as_posix(),
                           "pseudo": summary, "per_fold": [{k: v for k, v in p.items() if k != "grid"} for p in per_fold],
                           "gpu_s_pseudo_and_devpool": raw_doc["gpu_s"], "machine": raw_doc.get("machine")}
    if not smoke:
        v1_onnx = V1_BUNDLE / "detector.onnx"
        v1_csv = res_dir / "preds" / "v1_devpool.csv"
        write_csv(v1_csv, onnx_predict(v1_onnx, dp_all + mine, root))
        ev_t = evaluate(t_csv, "devpool", boot_n=BOOT_N, seed=BOOT_SEED)
        ev_v = evaluate(v1_csv, "devpool", boot_n=BOOT_N, seed=BOOT_SEED)
        cmp = compare(v1_csv, t_csv, "devpool", boot_n=BOOT_N, seed=BOOT_SEED)
        doc["devpool"] = {"teacher": devpool_summary(ev_t) | {"map50_by_fold": map50_by_fold(t_csv, root)},
                          "v1": devpool_summary(ev_v) | {"map50_by_fold": map50_by_fold(v1_csv, root),
                                                         "onnx": v1_onnx.as_posix()},
                          "teacher_minus_v1": {k: cmp["metrics"][k] for k in ("mAP50", "AR_small", "recall@op")},
                          "note": "thầy không thấy khung devpool nào (OOF); v1 đã học nhãn giả khung dev (E2, trừ dev-val "
                                  "+ đệm 0,3 s) nên với 74 khung devpool mới v1 KHÔNG phải OOF — lợi thế cho v1"}
        sm = _load(res_dir / "smoke.json") or {}
        aborted = [_load(p) for p in sorted(Path(cfg["runs_dir"]).glob("teacher_*_aborted/aborted.json"))]
        gh = {"smoke": round(sm.get("gpu_s", 0.0) / 3600, 4), "teacher": gpu_hours_of(Path(cfg["runs_dir"]) / "teacher"),
              "teacher_aborted": round(sum(a["gpu_s"] for a in aborted) / 3600, 4),
              "pseudo_and_devpool": round(raw_doc["gpu_s"] / 3600, 4), "cap": float(cfg["gpu_cap_h"]),
              "cap_note": "trần 1 giờ cả phase 05 do chủ dự án cắt 25/11 (plan gốc 6 giờ)"}
        gh["total"] = round(gh["smoke"] + gh["teacher"] + gh["teacher_aborted"] + gh["pseudo_and_devpool"], 4)
        doc["teacher_aborted"] = aborted
        doc["gpu_hours"] = gh
        doc["teacher_train"] = _load(Path(cfg["runs_dir"]) / "teacher" / "train_done.json")
        doc["teacher_val_history"] = read_history(Path(cfg["runs_dir"]) / "teacher" / "val_history.jsonl")
        _dump(res_dir / "teacher.json", doc)
    return doc


def _v2_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.selftrain", description="AI v2 phase 05: thầy + nhãn giả")
    ap.add_argument("cmd", choices=("teacher", "pseudo"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="teacher: đo VRAM + giây/ảnh mỗi cỡ; pseudo: 5 chuỗi đầu")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from xla.eval.detection import EvalInputError

    cfg = load_config(a.config)
    try:
        if a.cmd == "teacher":
            out = teacher_smoke(cfg) if a.smoke else teacher_train(cfg)
        else:
            out = pseudo_v2(cfg, a.smoke)
            out = {k: v for k, v in out.items() if k in ("pseudo", "per_fold", "devpool", "gpu_s_pseudo_and_devpool")}
    except (FileNotFoundError, TimeoutError, ValueError, EvalInputError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, default=str) if a.json else json.dumps(out, ensure_ascii=False, indent=1,
                                                                                    default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("teacher", "pseudo"):  # AI v2 (phase 05)
        return _v2_main(argv)
    ap = argparse.ArgumentParser(prog="python -m xla.detect.selftrain", description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None, help=f"mặc định {CONFIG_REL}")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--teacher", action="store_true", help="train thầy (GPU)")
    g.add_argument("--round", type=int, help="nhãn giả + train trò vòng K (GPU)")
    g.add_argument("--score", action="store_true", help="chấm + so ghép cặp → results/e2/e2.json (CPU)")
    g.add_argument("--onnx-score", type=Path, metavar="BUNDLE", help="chấm đường ONNX của bundle (ORT)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch trên smoke.fraction ảnh train")
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from xla.eval.detection import BOOT_N, BOOT_SEED, EvalInputError

    cfg = load_config(a.config or repo_root() / CONFIG_REL)
    try:
        if a.teacher:
            out = cmd_teacher(cfg, a.smoke)
        elif a.round is not None:
            out = cmd_round(cfg, a.round, a.smoke)
        elif a.score:
            out = score(cfg, a.boot if a.boot is not None else BOOT_N, a.seed if a.seed is not None else BOOT_SEED)
        else:
            out = onnx_score(cfg, a.onnx_score, a.boot if a.boot is not None else BOOT_N,
                             a.seed if a.seed is not None else BOOT_SEED)
    except (FileNotFoundError, TimeoutError, ValueError, EvalInputError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False) if a.json else json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
