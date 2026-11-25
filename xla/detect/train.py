"""Phase 10 — thí nghiệm E1: fine-tune YOLO11s COCO trên D0/D1/D2/D2+RealSR, chọn checkpoint bằng dev-val thật.

Một lần chạy = một tập, một tiến trình, giữ ``xla.gpu_lock`` suốt lúc train + dự đoán rồi nhả (lane khác chen được giữa
các lần). Run và trọng số ở ``<runs_dir>/<set>/`` (ngoài repo). Mỗi epoch, callback dự đoán 100 khung dev-val bằng
``last.pt``, chấm mAP50 bằng ``xla.eval.detection`` và giữ checkpoint tốt nhất ở ``weights/best_devval.pt``; val tổng hợp
của ultralytics chỉ được ghi lại, không dùng để chọn (D10: khung test không bao giờ dùng để chọn gì). Sau train: CSV dự
đoán test / neg / dev-val (D11, conf 0,001, bỏ hộp ``x1 < CROP_LEFT_PX`` như mốc ``edge16``) + ms/khung.

Chạy lại cùng lệnh sau khi máy sập: train tiếp từ ``last.pt`` (``resume``); train xong rồi thì chỉ dự đoán.

AX::

    python -m xla.detect.train --config configs/train_e1.yaml --set d2 --smoke --json   # 1 epoch / 10 % D2 → smoke.json
    python -m xla.detect.train --config configs/train_e1.yaml --set d0 --json           # lần đủ (GPU, gpu_lock)
    python -m xla.detect.train --config configs/train_e1.yaml --time --json             # đo lại ms/khung (B3, GPU)
    python -m xla.detect.train --config configs/train_e1.yaml --score --json            # chấm + so ghép cặp → e1.json
    # exit 0 xong; 1 lỗi dữ liệu / thiếu run / hết giờ chờ khoá; 2 sai tham số
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xla.camera.frames import CROP_LEFT_PX, FRAMES_REL
from xla.detect.baseline import WARMUP, Row, drop_edge, select_frames, write_csv
from xla.paths import data_root, downloads_dir, repo_root

CONFIG_REL = Path("configs/train_e1.yaml")
RESULTS_REL = Path("results/e1")
SPLIT_REL = Path("data/real/split.json")
LABELS_REL = Path("data/real/labels")
BASELINE_CSV_REL = Path("results/baseline/pred/edge16.csv")
CSV_MAX_BYTES = 5 * 1024 * 1024  # CSV < 5 MB thì commit (thuc-thi-phase-10-14.md §5.5)
REAL_FRAME_RE = re.compile(r"^\d{8}-\d{6}_\d{3}_\d{5}\.jpg$")  # tên khung bay thật (xla.camera.frames)
SYNTH_VAL_KEY = "metrics/mAP50(B)"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def _load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def devval_frames(root: Path) -> list[str]:
    return sorted(json.loads((root / LABELS_REL / "manifest.json").read_text(encoding="utf-8"))["devval_frames"])


NEG_SPLIT_REL = Path("data/real/neg_flight_split.json")
MINE_MAX_FRAC = 0.10  # ảnh nền mine ≤ 10 % tập train (phase 05 AI v2 bước 2)


def check_real_frames(names: list[str], root: Path, pseudo: dict[str, str] | None = None,
                      excluded_sessions: set[str] | frozenset[str] = frozenset(), n_total: int | None = None,
                      mine_max_frac: float = MINE_MAX_FRAC, session_of: dict[str, str] | None = None,
                      ) -> dict[str, int]:
    """Danh sách CHO PHÉP khung thật vào huấn luyện (hợp đồng §14.2, D5) — thay danh sách cấm cũ.

    Khung thật (tên khớp khung bay, hay có trong ``split.json`` / ``neg_flight_split.json``) chỉ được vào nếu:

    - là khung ``mine`` của ``neg_flight_split.json`` (ảnh nền, nhãn rỗng), tổng ≤ ``mine_max_frac`` × ``n_total``, hoặc
    - có trong ``pseudo`` (``{frame: session}``: khung mang nhãn giả), là ``dev`` của ``split.json``, không phải ``cal`` /
      ``holdout`` và phiên của nó không thuộc ``excluded_sessions`` (fold đang loại).

    ``session_of`` (tuỳ chọn) ghi đè phiên của khung mine khi lọc fold (mặc định lấy từ ``neg_flight_split.json``).
    Mọi khung thật khác (test, cal, holdout, dev chưa gán nhãn giả, khung ngoài split) → ValueError."""
    split = {f["frame"]: f["split"] for f in json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))["frames"]}
    neg_doc = json.loads((root / NEG_SPLIT_REL).read_text(encoding="utf-8"))
    neg = {f["frame"]: f for f in neg_doc["frames"]}
    pseudo = pseudo or {}
    sess = {f: r["session"] for f, r in neg.items()} | (session_of or {})
    bad: dict[str, list[str]] = {}
    n_mine = n_pseudo = 0
    for n in names:
        if not (n in split or n in neg or REAL_FRAME_RE.match(n)):
            continue
        kind = neg.get(n, {}).get("split")
        if kind == "mine":
            if sess.get(n) in excluded_sessions:
                bad.setdefault("mine của fold đang loại", []).append(n)
            n_mine += 1
            continue
        if kind in ("cal", "holdout"):
            bad.setdefault(kind, []).append(n)
        elif split.get(n) == "test":
            bad.setdefault("test", []).append(n)
        elif n not in pseudo:
            bad.setdefault("chưa gán (không mine, không nhãn giả)", []).append(n)
        elif split.get(n) != "dev":
            bad.setdefault(f"nhãn giả trên khung {split.get(n) or 'ngoài split'}", []).append(n)
        elif pseudo[n] in excluded_sessions:
            bad.setdefault("nhãn giả của fold đang loại", []).append(n)
        else:
            n_pseudo += 1
    if bad:
        msg = "; ".join(f"{k}: {len(v)} (vd. {v[:2]})" for k, v in bad.items())
        raise ValueError(f"khung thật ngoài danh sách cho phép (§14.2): {msg}")
    total = n_total if n_total is not None else len(names)
    if n_mine and n_mine > mine_max_frac * total:
        raise ValueError(f"ảnh nền mine {n_mine}/{total} > {mine_max_frac:.0%} tập train")
    return {"mine": n_mine, "pseudo": n_pseudo, "real_other": 0}


def dataset_info(set_name: str, root: Path) -> dict[str, Any]:
    """``dataset.yaml`` + sha256 manifest của tập sinh; khung thật ngoài danh sách cho phép → ValueError (§14.2)."""
    d = data_root() / set_name
    manifest = d / "manifest.jsonl"
    names = [json.loads(line)["name"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    real = check_real_frames(names, root, mine_max_frac=1.0)  # tập sinh: mọi khung thật đều là lỗi (dưới)
    if real["mine"] or real["pseudo"]:
        raise ValueError(f"{set_name}: tập sinh không được chứa khung thật (mine/nhãn giả chỉ thêm qua train.txt)")
    n_img = {sp: sum(1 for _ in (d / sp / "images").iterdir()) for sp in ("train", "val")}
    return {"set": set_name, "yaml": (d / "dataset.yaml").as_posix(), "manifest_sha256": _sha256(manifest),
            "build_key": _load(d / "build_key.json"), "n_images": n_img, "n_manifest": len(names),
            "real_frames_in_train": 0}


# ---------------------------------------------------------------- dự đoán (D11)


def predict(model: Any, frames: list[str], frames_dir: Path, pcfg: dict[str, Any], imgsz: int,
            warmup: int = WARMUP) -> dict[str, tuple[list[Row], float]]:
    """``{frame: (hộp đã bỏ mép trái, ms)}``; ms = suy luận + tiền/hậu xử lý, không gồm đọc JPEG (như mốc)."""
    import cv2

    images = {f: cv2.imread(str(frames_dir / f), cv2.IMREAD_COLOR) for f in frames}
    bad = [f for f, im in images.items() if im is None]
    if bad:
        raise FileNotFoundError(f"không đọc được {len(bad)} khung, vd. {bad[:3]}")

    def one(im: np.ndarray) -> list[Row]:
        res = model.predict(im, imgsz=imgsz, conf=pcfg["conf"], max_det=pcfg["max_det"], device=0, verbose=False)[0]
        return [(float(x1), float(y1), float(x2), float(y2), float(c))
                for x1, y1, x2, y2, c, _ in res.boxes.data.cpu().numpy()]

    for f in frames[:warmup]:
        one(images[f])
    out = {}
    for f in frames:
        t0 = time.perf_counter()
        rows = one(images[f])
        out[f] = (drop_edge(rows, CROP_LEFT_PX), (time.perf_counter() - t0) * 1000)
    return out


def devval_score(csv_path: Path) -> dict[str, float]:
    from xla.eval.detection import evaluate

    r = evaluate(csv_path, "devval", boot_n=0, f1_from_devval=False)
    return {"mAP50": r["metrics"]["mAP50"]["value"], "mAP50_95": r["metrics"]["mAP50_95"]["value"]}


class EpochTracker:
    """Callback ultralytics: đo giờ từng phần của epoch, VRAM đỉnh, chấm dev-val và giữ ``best_devval.pt``."""

    def __init__(self, run_dir: Path, frames: list[str], frames_dir: Path, pcfg: dict[str, Any], imgsz: int) -> None:
        self.run_dir, self.frames, self.frames_dir, self.pcfg, self.imgsz = run_dir, frames, frames_dir, pcfg, imgsz
        self.hist_path = run_dir / "devval_history.jsonl"
        self.history = read_history(self.hist_path)
        self.best = max((h["devval_mAP50"] for h in self.history), default=-1.0)
        self.t_call = time.perf_counter()
        self.t_first_epoch: float | None = None
        self.batch_t: list[float] = []
        self.t0 = self.t_train = 0.0
        self.vram_gb = 0.0
        self.pending = False  # on_fit_epoch_end còn được final_eval gọi lại lần nữa sau epoch cuối: bỏ lần đó

    def on_train_epoch_start(self, tr: Any) -> None:
        self.t0 = time.perf_counter()
        self.t_first_epoch = self.t_first_epoch or self.t0
        self.batch_t = []

    def on_train_batch_end(self, tr: Any) -> None:
        self.batch_t.append(time.perf_counter())

    def on_train_epoch_end(self, tr: Any) -> None:
        import torch

        self.t_train = time.perf_counter()
        self.pending = True
        self.vram_gb = torch.cuda.max_memory_reserved(0) / 2**30

    def on_fit_epoch_end(self, tr: Any) -> None:
        import torch
        from ultralytics import YOLO

        if not self.pending:
            return
        self.pending = False
        t_val = time.perf_counter()
        last = Path(tr.last)
        m = YOLO(str(last))
        preds = predict(m, self.frames, self.frames_dir, self.pcfg, self.imgsz, warmup=1)
        csv_p = self.run_dir / "devval_epoch.csv"
        write_csv(csv_p, preds)
        sc = {k: (v if v is not None else 0.0) for k, v in devval_score(csv_p).items()}  # NaN (không hộp) → 0
        del m
        torch.cuda.empty_cache()
        gaps = np.diff(self.batch_t) if len(self.batch_t) > 1 else np.asarray([np.nan])
        rec = {"epoch": int(tr.epoch) + 1, "devval_mAP50": sc["mAP50"], "devval_mAP50_95": sc["mAP50_95"],
               "val_synth_mAP50": (tr.metrics or {}).get(SYNTH_VAL_KEY), "train_s": round(self.t_train - self.t0, 2),
               "val_save_s": round(t_val - self.t_train, 2), "devval_s": round(time.perf_counter() - t_val, 2),
               "batches": len(self.batch_t), "median_batch_s": round(float(np.median(gaps)), 4),
               "batch": int(tr.batch_size), "vram_peak_gb": round(self.vram_gb, 2),
               "startup_s": round((self.t_first_epoch or self.t0) - self.t_call, 1), "time": _now()}
        self.history = [h for h in self.history if h["epoch"] != rec["epoch"]] + [rec]
        with self.hist_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if sc["mAP50"] > self.best:
            self.best = sc["mAP50"]
            shutil.copyfile(last, self.run_dir / "weights" / "best_devval.pt")


def read_history(path: Path) -> list[dict[str, Any]]:
    """Lịch sử dev-val, mỗi epoch một dòng (dòng sau đè dòng trước cùng epoch, khi train tiếp sau sập)."""
    by: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by[r["epoch"]] = r
    return [by[k] for k in sorted(by)]


# ---------------------------------------------------------------- một lần train


def _git_commit(root: Path) -> dict[str, Any]:
    def g(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def _fit(cfg: dict[str, Any], ds: dict[str, Any], run_dir: Path, tracker: EpochTracker, smoke: bool) -> dict[str, Any]:
    from ultralytics import YOLO

    last = run_dir / "weights" / "last.pt"
    resumed = last.is_file() and not smoke
    model = YOLO(str(last if resumed else downloads_dir() / cfg["weights"]))
    for ev in ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end", "on_fit_epoch_end"):
        model.add_callback(ev, getattr(tracker, ev))
    targs = dict(cfg["train"])
    if smoke:
        targs |= {"epochs": cfg["smoke"]["epochs"], "fraction": cfg["smoke"]["fraction"],
                  "close_mosaic": 0}  # smoke đo epoch CÓ mosaic (đắt hơn), không thì 1 epoch rơi vào close_mosaic
    if resumed:
        model.train(resume=True, workers=int(cfg["train"]["workers"]))  # workers đổi được khi train tiếp (ultralytics cho phép)
    else:
        model.train(data=ds["yaml"], project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **targs)
    return {"resumed": resumed, "batch_effective": int(model.trainer.batch_size), "epochs": int(model.trainer.epochs),
            "args": targs}


def train_set(cfg: dict[str, Any], set_name: str, smoke: bool) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import strip_optimizer

    from xla.gpu_lock import gpu_lock

    root = repo_root()
    run_dir = Path(cfg["runs_dir"]) / ("smoke" if smoke else set_name)
    if smoke and run_dir.exists():
        shutil.rmtree(run_dir)  # smoke luôn chạy mới
    run_dir.mkdir(parents=True, exist_ok=True)
    ds = dataset_info(set_name, root)
    frames_dir = root / FRAMES_REL
    dv = devval_frames(root)
    imgsz, pcfg = int(cfg["train"]["imgsz"]), cfg["predict"]
    info = _load(run_dir / "run_info.json", {}) or {"set": set_name, "smoke": smoke, "sessions": []}
    info |= {"argv": sys.argv, "git": _git_commit(root), "config": cfg, "dataset": ds}
    done_p = run_dir / "train_done.json"
    t_wait = time.perf_counter()
    with gpu_lock(f"phase10 e1 {'smoke ' if smoke else ''}train {set_name}", timeout_s=float(cfg["lock_timeout_s"])):
        t0 = time.perf_counter()
        session: dict[str, Any] = {"start": _now(), "lock_wait_s": round(t0 - t_wait, 1)}
        if smoke or not done_p.is_file():
            fit = _fit(cfg, ds, run_dir, EpochTracker(run_dir, dv, frames_dir, pcfg, imgsz), smoke)
            best = run_dir / "weights" / "best_devval.pt"
            strip_optimizer(best)
            _dump(done_p, fit | {"finished": _now(), "best_devval_sha256": _sha256(best)})
            session["resumed"] = fit["resumed"]
        session["train_s"] = round(time.perf_counter() - t0, 1)
        if not smoke:
            session["predict"] = _predict_all(cfg, run_dir, set_name, root, dv)
        session["gpu_s"] = round(time.perf_counter() - t0, 1)
    import torch
    import ultralytics

    info["sessions"].append(session | {"end": _now()})
    info["versions"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0)}
    _dump(run_dir / "run_info.json", info)
    return {"run_dir": run_dir.as_posix(), "session": session, "history": read_history(run_dir / "devval_history.jsonl"),
            "train_done": _load(done_p), "dataset": ds}


def _predict_all(cfg: dict[str, Any], run_dir: Path, set_name: str, root: Path, dv: list[str]) -> dict[str, Any]:
    import psutil
    from ultralytics import YOLO

    pcfg, imgsz = cfg["predict"], int(cfg["train"]["imgsz"])
    by_split = select_frames(root / SPLIT_REL, tuple(pcfg["splits"])) | {"devval": dv}
    model = YOLO(str(run_dir / "weights" / "best_devval.pt"))
    psutil.cpu_percent(interval=None)
    preds = predict(model, [f for fs in by_split.values() for f in fs], root / FRAMES_REL, pcfg, imgsz)
    cpu = psutil.cpu_percent(interval=None)
    out: dict[str, Any] = {"cpu_percent_during": cpu, "csv": {}}
    for sp, fs in by_split.items():
        p = run_dir / "preds" / f"{sp}.csv"
        write_csv(p, {f: preds[f] for f in fs})
        dst = root / RESULTS_REL / "preds" / f"{set_name}_{sp}.csv"
        if p.stat().st_size < CSV_MAX_BYTES:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        out["csv"][sp] = {"run": p.as_posix(), "repo": dst.as_posix() if dst.is_file() else None,
                          "bytes": p.stat().st_size, "sha256": _sha256(p)}
    ms = np.asarray([preds[f][1] for sp in pcfg["splits"] for f in by_split[sp]], dtype=float)
    out["ms"] = {"mean": round(float(ms.mean()), 2), "median": round(float(np.median(ms)), 2),
                 "p95": round(float(np.percentile(ms, 95)), 2), "n_frames": int(ms.size),
                 "note": "PyTorch FP32, conf 0,001, test + neg, sau WARMUP khung; máy có thể không rảnh"}
    return out


def smoke_report(cfg: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    """Ước giờ một epoch đủ từ 1 epoch trên ``fraction`` ảnh train → số epoch vừa trần ``budget_min_per_run``."""
    h = res["history"][-1]
    n_train = res["dataset"]["n_images"]["train"]
    nb_full = math.ceil(n_train / h["batch"])
    est_median = h["median_batch_s"] * nb_full + h["val_save_s"] + h["devval_s"]
    est_naive = h["train_s"] / cfg["smoke"]["fraction"] + h["val_save_s"] + h["devval_s"]
    epoch_s = max(est_median, est_naive)  # lấy số lớn hơn: an toàn với trần giờ
    budget_s = cfg["smoke"]["budget_min_per_run"] * 60
    overhead_s = h["startup_s"] + 120  # khởi động + dự đoán ~600 khung + final val
    return {
        "phase": "10", "created": _now(), "set": res["dataset"]["set"], "fraction": cfg["smoke"]["fraction"],
        "imgsz": cfg["train"]["imgsz"], "batch": cfg["train"]["batch"], "batch_effective": h["batch"],
        "workers": cfg["train"]["workers"], "n_train_full": n_train, "batches_smoke": h["batches"],
        "batches_full": nb_full, "measured": h, "epoch_s_est_median_batch": round(est_median, 1),
        "epoch_s_est_naive": round(est_naive, 1), "hours_per_epoch": round(epoch_s / 3600, 4),
        "vram_peak_gb": h["vram_peak_gb"], "budget_min_per_run": cfg["smoke"]["budget_min_per_run"],
        "epochs_fit_budget": int((budget_s - overhead_s) // epoch_s),
        "gpu_s": res["session"]["gpu_s"], "run_dir": res["run_dir"],
    }


def _gpu_temp() -> int | None:
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False)
    return int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout.strip() else None


def time_sets(cfg: dict[str, Any]) -> dict[str, Any]:
    """Đo lại ms/khung trên test + neg bằng ``best_devval.pt`` mỗi set, tiến trình riêng, một lần giữ khoá (B3).

    ms ghi lúc dự đoán ngay sau train bị phồng (cùng tiến trình với train: cuDNN ``deterministic``, bộ nhớ GPU còn
    giữ), nên B3 dùng số đo này. Ghi ``<run_dir>/ms_idle.json``; kèm % CPU toàn máy lúc đo."""
    import psutil
    import torch
    from ultralytics import YOLO

    from xla.gpu_lock import gpu_lock

    root = repo_root()
    by_split = select_frames(root / SPLIT_REL, tuple(cfg["predict"]["splits"]))
    frames = [f for fs in by_split.values() for f in fs]
    imgsz, out = int(cfg["train"]["imgsz"]), {}
    with gpu_lock("phase10 e1 đo ms best_devval (4 set)", timeout_s=float(cfg["lock_timeout_s"])):
        for s in cfg["sets"]:
            run_dir = Path(cfg["runs_dir"]) / s
            model = YOLO(str(run_dir / "weights" / "best_devval.pt"))
            psutil.cpu_percent(interval=None)
            temp0 = _gpu_temp()
            preds = predict(model, frames, root / FRAMES_REL, cfg["predict"], imgsz)
            cpu = psutil.cpu_percent(interval=None)
            ms = np.asarray([v[1] for v in preds.values()], dtype=float)
            out[s] = {"measured": _now(), "mean": round(float(ms.mean()), 2), "median": round(float(np.median(ms)), 2),
                      "p95": round(float(np.percentile(ms, 95)), 2), "n_frames": int(ms.size), "warmup_frames": WARMUP,
                      "cpu_percent_during": cpu, "gpu": torch.cuda.get_device_name(0),
                      "gpu_temp_c": {"start": temp0, "end": _gpu_temp()},
                      "note": "PyTorch FP32, conf 0,001, test + neg, tiến trình riêng sau train (giống baseline time)"}
            _dump(run_dir / "ms_idle.json", out[s])
            del model
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- chấm + so ghép cặp → e1.json (CPU)

B1_MAP50 = 0.227  # results/baseline/tieu-chi-sau-m2.md §2
B2_FP_FLIGHT = 0.05
B2_CI_HIGH_MAX = 0.1  # §3: cận trên CI quá ~0,1 → "chưa đủ dữ liệu"
B3_MS_P95 = 30.0
SUMMARY_KEYS = ("mAP50", "mAP50_95", "AP_small", "AR_small", "precision", "recall", "f1", "fp_per_neg_flight",
                "fp_per_neg_indoor", "fp_per_neg_frame")


def _combined_csv(set_name: str, root: Path, out: Path) -> Path:
    """Gộp ``<set>_{test,neg,devval}.csv`` của repo thành một CSV (bộ chấm cần cả ba trong một file)."""
    lines: list[str] = []
    for sp in ("test", "neg", "devval"):
        rows = (root / RESULTS_REL / "preds" / f"{set_name}_{sp}.csv").read_text(encoding="utf-8").splitlines()
        lines += rows if not lines else rows[1:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def summarize(ev: dict[str, Any]) -> dict[str, Any]:
    op = round(ev["f1_threshold_from_devval"]["thr"], 6)
    m = ev["metrics"]
    out = {}
    for k in SUMMARY_KEYS:
        src = m[k] if k in m else m[f"{k}@{op:g}"]
        out[k if k in m else f"{k}@op"] = {"value": src["value"], "ci95": src["ci95"]}
    return {"conf_threshold": op, "threshold_source": "devval", "metrics": out, "n": ev["n"],
            "official": ev["official"]}


def criteria(cmp_vs_base: dict[str, Any], summ: dict[str, Any], ms_p95: float | None) -> dict[str, Any]:
    m50 = cmp_vs_base["metrics"]["mAP50"]
    b1 = m50["b"] >= B1_MAP50 and m50["ci95"][0] > 0
    fp = summ["metrics"]["fp_per_neg_flight@op"]
    if fp["value"] <= B2_FP_FLIGHT and fp["ci95"][1] <= B2_CI_HIGH_MAX:
        b2 = "đạt"
    elif fp["value"] <= B2_FP_FLIGHT:
        b2 = "chưa đủ dữ liệu"
    else:
        b2 = "không đạt"
    return {"B1": {"pass": b1, "mAP50": m50["b"], "diff_vs_edge16": m50["diff"], "diff_ci95": m50["ci95"],
                   "rule": f"mAP50 ≥ {B1_MAP50} và cận dưới CI hiệu ghép cặp > 0"},
            "B2": {"verdict": b2, "fp_per_neg_flight": fp, "recall@op": summ["metrics"]["recall@op"],
                   "rule": f"fp_per_neg_flight@op ≤ {B2_FP_FLIGHT} và cận trên CI ≤ {B2_CI_HIGH_MAX}"},
            "B3": {"pass_pytorch": ms_p95 is not None and ms_p95 <= B3_MS_P95, "ms_p95_pytorch": ms_p95,
                   "rule": f"≤ {B3_MS_P95} ms p95 trên ORT CUDA EP (đo ở phase 11/16); ở đây chỉ số PyTorch"}}


def score(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    from xla.eval.detection import compare, evaluate

    root = repo_root()
    res_dir = root / RESULTS_REL
    smoke = _load(res_dir / "smoke.json")
    if smoke is None:
        raise FileNotFoundError(f"thiếu {res_dir / 'smoke.json'} (chạy --smoke trước)")
    base_csv = root / BASELINE_CSV_REL
    tmp = Path(cfg["runs_dir"]) / "score"
    ev_base = evaluate(base_csv, "test", boot_n=boot, seed=seed)
    _dump(res_dir / "eval" / "edge16_test.json", ev_base)
    sets, csvs = {}, {}
    for s in cfg["sets"]:
        run_dir = Path(cfg["runs_dir"]) / s
        info, done = _load(run_dir / "run_info.json"), _load(run_dir / "train_done.json")
        if not info or not done:
            raise FileNotFoundError(f"{s}: chưa có run hoàn tất ở {run_dir}")
        csvs[s] = _combined_csv(s, root, tmp / f"{s}_all.csv")
        ev = evaluate(csvs[s], "test", boot_n=boot, seed=seed)
        _dump(res_dir / "eval" / f"{s}_test.json", ev)
        hist = read_history(run_dir / "devval_history.jsonl")
        sel = max(hist, key=lambda h: (h["devval_mAP50"], -h["epoch"]))
        last3 = [h["devval_mAP50"] for h in hist[-3:]]
        pred = [x["predict"] for x in info["sessions"] if "predict" in x][-1]
        sets[s] = {
            "weights": (run_dir / "weights" / "best_devval.pt").resolve().as_posix(),
            "weights_sha256": done["best_devval_sha256"], "selected_by": "devval", "selected_epoch": sel["epoch"],
            "devval_map50": sel["devval_mAP50"], "devval_map50_last3_mean": float(np.mean(last3)),
            "smoke_hours_per_epoch": smoke["hours_per_epoch"],
            "hours_per_epoch": round(float(np.mean([h["train_s"] + h["val_save_s"] + h["devval_s"] for h in hist]))
                                     / 3600, 4),
            "gpu_hours": round(sum(x["gpu_s"] for x in info["sessions"]) / 3600, 4),
            "epochs": done["epochs"], "batch_effective": done["batch_effective"], "train_data": info["dataset"],
            "ms_per_frame": _load(run_dir / "ms_idle.json") or pred["ms"],
            "ms_per_frame_after_train": pred["ms"] | {"cpu_percent_during": pred["cpu_percent_during"]},
            "devval_history": hist, "test": summarize(ev),
        }
    compares = {f"edge16_vs_{s}": compare(base_csv, csvs[s], "test", boot_n=boot, seed=seed) for s in cfg["sets"]}
    compares["d1_vs_d2"] = compare(csvs["d1"], csvs["d2"], "test", boot_n=boot, seed=seed)
    compares["d1_vs_d2sr"] = compare(csvs["d1"], csvs["d2sr"], "test", boot_n=boot, seed=seed)
    for s in cfg["sets"]:
        sets[s]["criteria"] = criteria(compares[f"edge16_vs_{s}"], sets[s]["test"], sets[s]["ms_per_frame"]["p95"])
    best_set = max(cfg["sets"], key=lambda s: sets[s]["devval_map50"])  # chọn trên dev-val, không trên test (D10)
    b = sets[best_set]
    base_ms = (_load(root / "results/baseline/baseline.json") or {}).get("variants", {}).get("edge16", {})
    doc = {
        "phase": "10", "created": _now(), "config": cfg, "smoke": smoke,
        "baseline": {"variant": "edge16", "csv": BASELINE_CSV_REL.as_posix(), "test": summarize(ev_base),
                     "ms_per_frame_idle": base_ms.get("ms_per_frame_idle")},
        "sets": sets,
        "compare": {k: {kk: v[kk] for kk in ("pred_a", "pred_b", "op_threshold", "n", "metrics", "bootstrap")}
                    for k, v in compares.items()},
        "best": {"set": best_set, "weights": b["weights"], "devval_map50": b["devval_map50"],
                 "conf_threshold": b["test"]["conf_threshold"], "threshold_source": "devval"},
        "best_rule": "set có devval_map50 cao nhất (dev-val thật, không dùng khung test)",
        "gpu_hours_total": round(smoke["gpu_s"] / 3600 + sum(s["gpu_hours"] for s in sets.values()), 4),
    }
    _dump(res_dir / "e1.json", doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.train", description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None, help=f"mặc định {CONFIG_REL}")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", choices=("d0", "d1", "d2", "d2sr"))
    g.add_argument("--score", action="store_true", help="chấm mọi set + so ghép cặp → results/e1/e1.json (CPU)")
    g.add_argument("--time", action="store_true", help="đo lại ms/khung best_devval.pt mỗi set (GPU, gpu_lock)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch trên smoke.fraction ảnh train → results/e1/smoke.json")
    ap.add_argument("--boot", type=int, default=None, help="số mẫu bootstrap (mặc định của xla.eval.detection)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from xla.eval.detection import BOOT_N, BOOT_SEED, EvalInputError

    cfg = load_config(a.config or repo_root() / CONFIG_REL)
    try:
        if a.time:
            out = time_sets(cfg)
        elif a.score:
            doc = score(cfg, a.boot if a.boot is not None else BOOT_N, a.seed if a.seed is not None else BOOT_SEED)
            out: dict[str, Any] = {"e1": (repo_root() / RESULTS_REL / "e1.json").as_posix(), "best": doc["best"],
                                   "gpu_hours_total": doc["gpu_hours_total"]}
        else:
            res = train_set(cfg, a.set, a.smoke)
            if a.smoke:
                out = smoke_report(cfg, res)
                _dump(repo_root() / RESULTS_REL / "smoke.json", out)
            else:
                out = {k: res[k] for k in ("run_dir", "session", "train_done")} | {"history": res["history"]}
    except (FileNotFoundError, TimeoutError, ValueError, EvalInputError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False) if a.json else json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
