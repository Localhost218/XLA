"""Phase 06 AI v2 — trò YOLO11s @1280 chọn bằng OOF k-fold theo phiên trên ``devpool``, xuất ``detector-v2.0.0``.

Ứng viên (``configs/student_v2.yaml``, Deferred 9): C1 = d02x + nền mine; C3 = d0 + nền mine; C2 = C1 + nhãn giả dev
(phase 05). C1/C3 không thấy khung devpool nào → một lần huấn luyện, dự đoán toàn devpool là OOF. C2: mỗi fold i huấn
luyện bỏ MỌI khung thật thuộc phiên của fold i (nhãn giả + mine), dự đoán fold i → OOF ghép. Mọi ứng viên cùng epoch,
imgsz, batch, seed, trọng số khởi đầu; mỗi lần một khoá GPU (D6).

``select`` chấm OOF (``xla.eval.detection`` split ``devpool``, bootstrap theo khối 30 s — khối nằm gọn trong một phiên
nên trong một fold), so ghép cặp với ``detector-v1.0.0``, chọn cận dưới CI mAP50 OOF cao nhất; CI chồng nhau thì chọn
đơn giản hơn (C3 < C1 < C2). Kill (b): ứng viên chọn không hơn v1 (cận dưới CI hiệu ghép cặp ≤ 0) → không xuất v2.
Ngưỡng = F1 tối đa trên OOF (``select_f1_threshold`` chặn test + cal + holdout). Test chỉ được chấm, không chọn gì.

AX::

    python -m xla.detect.kfold run --config configs/student_v2.yaml --json            # mọi lần huấn luyện còn thiếu (GPU)
    python -m xla.detect.kfold run --config configs/student_v2.yaml --cand C1 --json
    python -m xla.detect.kfold select --config configs/student_v2.yaml --json         # OOF + chọn + Kill (b) (CPU)
    python -m xla.detect.kfold export --config configs/student_v2.yaml --json         # bundle + validate + ms (GPU)
    python -m xla.detect.kfold score-test --config configs/student_v2.yaml --json     # chấm test (chỉ chấm)
    python -m xla.detect.kfold tiling --config configs/student_v2.yaml --json         # SAHI ô 640 trên test (GPU)
    # exit 0 xong; 1 lỗi dữ liệu / rò rỉ / thiếu run / hết giờ khoá / Kill (b) khi export; 2 sai tham số
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xla.camera.frames import FRAMES_REL
from xla.detect import selftrain as S
from xla.detect.baseline import write_csv
from xla.detect.train import _dump, _load, _now, _sha256, predict
from xla.paths import repo_root

CONFIG_REL = Path("configs/student_v2.yaml")
RESULTS_REL = Path("results/v2/detect")
SIMPLER_ORDER = ("C3", "C1", "C2")  # hoà (CI chồng) → chọn cái đứng trước
BASELINE_CSV_REL = Path("results/baseline/pred/edge16.csv")
V1_TEST_CSV_REL = Path("results/e2/preds/onnx_detector-v1.0.0_all.csv")


class OOFLeakError(ValueError):
    """CSV OOF chứa khung test / cal / holdout / ngoài devpool + mine."""


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _res(root: Path | None = None) -> Path:
    return (root or repo_root()) / RESULTS_REL


def gpu_used_h(cfg: dict[str, Any]) -> float:
    """Giờ GPU đã dùng trong phase 06 (mọi run + lần dự đoán/xuất/đo ghi ở ``gpu_log.jsonl``)."""
    p = Path(cfg["runs_dir"]) / "gpu_log.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []
    runs = sum(S.gpu_hours_of(d) for d in Path(cfg["runs_dir"]).iterdir() if (d / "run_info.json").is_file()) \
        if Path(cfg["runs_dir"]).is_dir() else 0.0
    return round(runs + sum(r["gpu_s"] for r in rows) / 3600, 4)


def log_gpu(cfg: dict[str, Any], what: str, gpu_s: float, **extra: Any) -> None:
    p = Path(cfg["runs_dir"]) / "gpu_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"what": what, "gpu_s": round(gpu_s, 1), "time": _now()} | extra, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- dữ liệu thật: devpool, mine, fold, nhãn giả


def real_context(root: Path) -> dict[str, Any]:
    neg = S.neg_split_doc(root)["frames"]
    sess = S.frame_sessions(root)
    folds = json.loads((root / S.FOLDS_REL).read_text(encoding="utf-8"))["folds"]
    fsess = S.fold_sessions(root, sess)
    mine = sorted(f["frame"] for f in neg if f["split"] == "mine")
    mine_sess = {f["frame"]: f["session"] for f in neg if f["split"] == "mine"}
    return {"folds": folds, "fold_sessions": fsess, "mine": mine, "mine_session": mine_sess,
            "devpool": sorted(f for fold in folds for f in fold), "session": sess,
            "reserved": {f["frame"] for f in neg if f["split"] in ("cal", "holdout")}}


def pseudo_labels(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Khung nhãn giả dùng được của phase 05: ``{frame: {session, fold, boxes}}``."""
    p = Path(cfg["pseudo_dir"]) / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"thiếu nhãn giả {p} (phase 05 pseudo)")
    man = json.loads(p.read_text(encoding="utf-8"))
    return {r["frame"]: {"session": r["session"], "fold": r["fold"], "boxes": r["boxes"]} for r in man["frames"]
            if r["use"]}


def run_names(cfg: dict[str, Any]) -> list[tuple[str, str, int | None]]:
    """(tên run, ứng viên, fold loại) theo thứ tự chạy; C2 chỉ khi cổng C2 mở."""
    out = [("C1", "C1", None), ("C3", "C3", None)]
    if c2_gate(cfg)["run_c2"]:
        out += [(f"C2_fold{i}", "C2", i) for i in range(len(json.loads(
            (repo_root() / S.FOLDS_REL).read_text(encoding="utf-8"))["folds"]))]
    return out


def c2_gate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Chủ dự án 25/11: thầy không rõ ràng hơn v1 trên devpool (cận dưới CI hiệu mAP50 ghép cặp ≤ 0) → bỏ C2."""
    t = _load(repo_root() / S.V2_RESULTS_REL / "teacher.json")
    if not t or "devpool" not in t:
        raise FileNotFoundError("thiếu results/v2/teacher/teacher.json (phase 05) để quyết C2")
    d = t["devpool"]["teacher_minus_v1"]["mAP50"]
    run = cfg["candidates"]["C2"].get("enabled", True) and d["ci95"][0] is not None and d["ci95"][0] > 0
    return {"run_c2": bool(run), "teacher_map50": d["b"], "v1_map50": d["a"], "diff": d["diff"], "diff_ci95": d["ci95"],
            "rule": "chạy C2 chỉ khi cận dưới CI 95 % hiệu mAP50 (thầy − v1) trên devpool > 0 (chủ dự án 25/11)"}


def build_candidate(cfg: dict[str, Any], cand: str, fold: int | None, root: Path) -> dict[str, Any]:
    """Tập huấn luyện của một run. ``fold`` = fold bị loại (bỏ mọi khung thật thuộc phiên của nó: nhãn giả + mine);
    C2 không fold = bản cuối (nhãn giả mọi fold, chỉ sau khi C2 thắng ở ``select``)."""
    c, d, ctx = cfg["candidates"][cand], cfg["data"], real_context(root)
    excl = ctx["fold_sessions"][fold] if fold is not None else set()
    rows = S.source_rows(c["source_set"])
    tr = S.sample_by_source(rows, "train", int(d["n_source_train"]), int(cfg["seed"]))
    va = S.sample_by_source(rows, "val", int(d["n_source_val"]), int(cfg["seed"]))
    real: dict[str, list[list[float]]] = {}
    if c.get("mine", True):
        real |= {f: [] for f in ctx["mine"] if ctx["mine_session"][f] not in excl}
    pseudo_sess: dict[str, str] = {}
    if c.get("pseudo", False):
        if fold is None and (_load(_res(root) / "select.json") or {}).get("winner") != cand:
            raise OOFLeakError(f"{cand} không fold (bản cuối, thấy nhãn giả mọi fold) chỉ được chạy khi {cand} thắng OOF")
        for f, r in pseudo_labels(cfg).items():
            if r["session"] in excl:
                continue
            real[f] = [b[:4] for b in r["boxes"]]
            pseudo_sess[f] = r["session"]
    name = (f"{cand}_fold{fold}" if fold is not None else f"{cand}_final") if c.get("pseudo") else cand
    return S.build_v2_dataset(Path(cfg["datasets_dir"]) / f"v2_student_{name}", c["source_set"], tr, va, real,
                              pseudo_sess, root, excluded_sessions=excl, mine_max_frac=float(d["mine_max_frac"])) | {
        "excluded_fold": fold, "candidate": cand}


# ---------------------------------------------------------------- huấn luyện + dự đoán OOF (GPU)


def train_args(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg["student"]
    return dict(cfg["train"]) | {"batch": int(t["batch"]), "epochs": int(t["epochs"])}


def student_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Vài chục bước YOLO11s @1280: VRAM + giây/ảnh → số epoch vừa ngân sách một run (ghi smoke.json)."""
    import shutil

    from ultralytics import YOLO

    from xla.gpu_lock import gpu_lock

    root, sm = repo_root(), cfg["smoke"]
    rows = S.source_rows("d02x")
    tr = S.sample_by_source(rows, "train", int(sm["n_train"]), int(cfg["seed"]))
    va = S.sample_by_source(rows, "val", int(sm["n_val"]), int(cfg["seed"]))
    ds = S.build_v2_dataset(Path(cfg["datasets_dir"]) / "v2_student_smoke", "d02x", tr, va, {}, {}, root)
    run_dir = Path(cfg["runs_dir"]).parent / "student_smoke"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "weights").mkdir(parents=True)
    with gpu_lock("v2 phase06 smoke trò", timeout_s=float(cfg["lock_timeout_s"])):
        t0 = time.perf_counter()
        mach = S.machine_info()
        tracker = S.ValTracker(run_dir, 99)
        model = YOLO(str(S.resolve_weights(cfg["student"]["init_weights"])))
        for ev in S.TRACK_EVENTS:
            model.add_callback(ev, getattr(tracker, ev))
        model.train(data=ds["yaml"], project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                    **(train_args(cfg) | {"epochs": 1, "close_mosaic": 0}))
        gpu_s = time.perf_counter() - t0
    h = tracker.history[-1]
    s_img, v_img = h["median_batch_s"] / h["batch"], h["val_save_s"] / len(va)
    d = cfg["data"]
    epoch_s = s_img * (int(d["n_source_train"]) + 85) + v_img * int(d["n_source_val"])
    out = {"phase": "06", "created": _now(), "machine": mach, "vram_peak_gb": h["vram_peak_gb"], "batch": h["batch"],
           "median_batch_s": h["median_batch_s"], "s_per_img_train": round(s_img, 4), "s_per_img_val": round(v_img, 4),
           "hours_per_epoch_est": round(epoch_s / 3600, 4), "gpu_s": round(gpu_s, 1),
           "epochs_in_run_budget": round(float(cfg["student"]["run_budget_min"]) * 60 / epoch_s, 2)}
    log_gpu(cfg, "smoke trò", gpu_s)
    _dump(_res(root) / "smoke.json", out)
    return out


def predict_frames(cfg: dict[str, Any], weights: Path, frames: list[str], label: str) -> dict[str, tuple[list, float]]:
    from ultralytics import YOLO

    from xla.gpu_lock import gpu_lock

    with gpu_lock(label, timeout_s=float(cfg["lock_timeout_s"])):
        t0 = time.perf_counter()
        out = predict(YOLO(str(weights)), frames, repo_root() / FRAMES_REL, cfg["predict"], int(cfg["train"]["imgsz"]),
                      warmup=3)
        log_gpu(cfg, label, time.perf_counter() - t0, n_frames=len(frames))
    return out


def run_one(cfg: dict[str, Any], name: str, cand: str, fold: int | None) -> dict[str, Any]:
    """Một lần huấn luyện + dự đoán: fold i → devpool fold i + mine của phiên fold i (OOF); không fold → devpool + mine
    + test + neg (test chỉ để chấm ở bước 7, không chọn gì)."""
    root = repo_root()
    cap = float(cfg["gpu_cap_h"]) + float(cfg.get("extension_h", 0.0))
    used = gpu_used_h(cfg)
    run_dir = Path(cfg["runs_dir"]) / name
    if used >= cap and not (run_dir / "train_done.json").is_file():
        raise ValueError(f"trần GPU phase 06 {cap} h đã dùng {used} h: không chạy {name}")
    ds = build_candidate(cfg, cand, fold, root)
    w0 = S.resolve_weights(cfg["student"]["init_weights"])
    cap_s = float(cfg["student"]["run_budget_min"]) * 60 * float(cfg["student"].get("budget_slack", 1.15))
    res = S.train_v2(cfg, name, w0, ds, train_args(cfg), int(cfg["student"]["early_stop_patience"]),
                     f"v2 phase06 train {name}", cap_s=cap_s, select="val")
    ctx = real_context(root)
    preds_p = run_dir / "preds.csv"
    if not preds_p.is_file():
        if fold is None:
            from xla.detect.baseline import select_frames

            sp = select_frames(root / S.SPLIT_REL, ("test", "neg"))
            frames = ctx["devpool"] + ctx["mine"] + sp["test"] + sp["neg"]
        else:
            frames = sorted(ctx["folds"][fold]) + [f for f in ctx["mine"]
                                                   if ctx["mine_session"][f] in ctx["fold_sessions"][fold]]
        write_csv(preds_p, predict_frames(cfg, run_dir / "weights" / "best.pt", frames, f"v2 phase06 dự đoán {name}"))
    return {"name": name, "candidate": cand, "fold": fold, "train_done": res["train_done"],
            "gpu_h": S.gpu_hours_of(run_dir), "preds": preds_p.as_posix()}


def cmd_run(cfg: dict[str, Any], only: str | None) -> dict[str, Any]:
    plan = [r for r in run_names(cfg) if only is None or only in (r[0], r[1])]
    if not plan:
        raise ValueError(f"không có run nào khớp {only!r} (C2 chỉ chạy khi cổng C2 mở)")
    done = [run_one(cfg, *r) for r in plan]
    return {"runs": done, "gpu_used_h": gpu_used_h(cfg), "c2_gate": c2_gate(cfg)}


# ---------------------------------------------------------------- OOF + chọn (CPU)


def check_oof_frames(csv_p: Path, root: Path) -> set[str]:
    """Khung trong CSV OOF chỉ được là devpool + mine; test / cal / holdout / khác → OOFLeakError."""
    import csv as _csv

    ctx = real_context(root)
    allowed = set(ctx["devpool"]) | set(ctx["mine"])
    split = {f: r["split"] for f, r in S.split_rows_by_frame(root).items()}
    with csv_p.open(encoding="utf-8", newline="") as fh:
        frames = {r["frame"] for r in _csv.DictReader(fh)}
    bad = sorted(f for f in frames if f not in allowed or f in ctx["reserved"] or split.get(f) == "test")
    if bad:
        kinds = {("test" if split.get(f) == "test" else "cal/holdout" if f in ctx["reserved"] else "ngoài devpool")
                 for f in bad}
        raise OOFLeakError(f"{csv_p.name}: {len(bad)} khung {sorted(kinds)} trong OOF (vd. {bad[:3]})")
    return frames


def _filter_csv(src: Path, keep: set[str], out: Path) -> Path:
    lines = src.read_text(encoding="utf-8").splitlines()
    rows = [lines[0]] + [x for x in lines[1:] if x.split(",", 1)[0] in keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return out


def oof_csv(cfg: dict[str, Any], cand: str, root: Path) -> Path:
    """CSV OOF của ứng viên (devpool + mine) ghép từ run của nó → ``results/v2/detect/preds/<cand>_oof.csv``."""
    ctx = real_context(root)
    keep = set(ctx["devpool"]) | set(ctx["mine"])
    out = _res(root) / "preds" / f"{cand}_oof.csv"
    runs = Path(cfg["runs_dir"])
    if cand == "C2":
        parts = []
        for i in range(len(ctx["folds"])):
            p = runs / f"C2_fold{i}" / "preds.csv"
            if not p.is_file():
                raise FileNotFoundError(f"thiếu {p}")
            parts.append(p.read_text(encoding="utf-8").splitlines())
        rows = [parts[0][0]] + [x for p in parts for x in p[1:]]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    else:
        p = runs / cand / "preds.csv"
        if not p.is_file():
            raise FileNotFoundError(f"thiếu {p}")
        _filter_csv(p, keep, out)
    check_oof_frames(out, root)
    return out


def pick_winner(table: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Cận dưới CI mAP50 OOF cao nhất; ứng viên đơn giản hơn có CI chồng với nó → chọn đơn giản hơn (plan bước 3)."""
    cands = [c for c in SIMPLER_ORDER if c in table]
    lo = {c: table[c]["metrics"]["mAP50"]["ci95"][0] for c in cands}
    best = max(cands, key=lambda c: (lo[c] if lo[c] is not None else -1.0, -SIMPLER_ORDER.index(c)))
    for c in cands:
        if SIMPLER_ORDER.index(c) >= SIMPLER_ORDER.index(best):
            break
        hi = table[c]["metrics"]["mAP50"]["ci95"][1]
        if hi is not None and lo[best] is not None and hi >= lo[best]:
            return c, f"CI mAP50 OOF của {c} chồng với {best} (cận dưới CI cao nhất) → chọn đơn giản hơn"
    return best, "cận dưới CI 95 % mAP50 OOF cao nhất, không ứng viên đơn giản hơn nào có CI chồng"


def floor_threshold(csv_p: Path, root: Path) -> dict[str, Any]:
    """Ngưỡng F1 tối đa trên OOF trong miền ``≥ score_floor`` của graph (§2: graph bỏ mọi hộp < 0,05, nên ngưỡng vận
    hành không thể thấp hơn). F1 ở ngưỡng t chỉ phụ thuộc hộp ≥ t, nên lọc CSV ở floor rồi chọn là đúng phép chọn có ràng
    buộc. Vẫn qua ``select_f1_threshold`` (chặn test + cal + holdout)."""
    from xla.detect.export_onnx import SCORE_FLOOR
    from xla.eval.detection import evaluate

    lines = csv_p.read_text(encoding="utf-8").splitlines()
    out = _res(root) / "preds" / f"{csv_p.stem}_floor.csv"
    rows = [lines[0]] + [x for x in lines[1:] if float(x.split(",")[5]) >= SCORE_FLOOR]
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    ev = evaluate(out, "devpool", boot_n=0)
    f1 = ev["f1_threshold_from_devval"]
    out.unlink()
    return {"thr": round(max(f1["thr"], SCORE_FLOOR), 6), "score_floor": SCORE_FLOOR,
            "f1_oof": round(f1["f1_devval"], 4), "precision_oof": round(f1["precision_devval"], 4),
            "recall_oof": round(f1["recall_devval"], 4), "rule": "F1 tối đa trên OOF devpool, ngưỡng ≥ score_floor"}


def v1_pt_csv(cfg: dict[str, Any], root: Path) -> Path:
    """Dự đoán devpool + mine của v1 bằng ``.pt`` của nó (cùng đường ``predict`` với trò) — tạo một lần (GPU ngắn)."""
    p = _res(root) / "preds" / "v1_pt_devpool.csv"
    if not p.is_file():
        ctx = real_context(root)
        write_csv(p, predict_frames(cfg, Path(cfg["v1"]["pt"]), ctx["devpool"] + ctx["mine"],
                                    "v2 phase06 v1 .pt devpool (đối chứng)"))
    return p


def cmd_select(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    from xla.eval.detection import compare, evaluate

    root = repo_root()
    v1_onnx_csv = root / S.V2_RESULTS_REL / "preds" / "v1_devpool.csv"
    v1_csv = v1_pt_csv(cfg, root)  # cùng đường .pt với ứng viên (so ghép cặp công bằng; đường ONNX: ``finalize``)
    check_oof_frames(v1_csv, root)
    check_oof_frames(v1_onnx_csv, root)
    runs = Path(cfg["runs_dir"])
    cands = [c for c in SIMPLER_ORDER if (runs / (f"{c}_fold0" if c == "C2" else c) / "preds.csv").is_file()]
    if not cands:
        raise FileNotFoundError("chưa có ứng viên nào chạy xong (kfold run)")
    table, csvs = {}, {}
    for c in cands:
        csvs[c] = oof_csv(cfg, c, root)
        ev = evaluate(csvs[c], "devpool", boot_n=boot, seed=seed)
        cmp_v1 = compare(v1_csv, csvs[c], "devpool", boot_n=boot, seed=seed)
        table[c] = S.devpool_summary(ev) | {
            "map50_by_fold": S.map50_by_fold(csvs[c], root), "f1_threshold": ev["f1_threshold_from_devval"],
            "vs_v1": {k: cmp_v1["metrics"][k] for k in ("mAP50", "AR_small", "recall@op", "fp_per_neg_flight@op")},
            "train_done": _load(runs / (f"{c}_fold0" if c == "C2" else c) / "train_done.json")}
    v1 = S.devpool_summary(evaluate(v1_csv, "devpool", boot_n=boot, seed=seed)) | {
        "map50_by_fold": S.map50_by_fold(v1_csv, root)}
    pick, why = pick_winner(table)
    pair = {}
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            r = compare(csvs[a], csvs[b], "devpool", boot_n=boot, seed=seed)
            pair[f"{b}_minus_{a}"] = {k: r["metrics"][k] for k in ("mAP50", "AR_small")}
    final, ext = pick, None
    if (runs / f"{pick}_ext" / "preds.csv").is_file():  # nới giờ một lần (chủ dự án 25/11): model ship = bản nới
        final = f"{pick}_ext"
        p_ext = oof_csv(cfg, final, root)
        ev = evaluate(p_ext, "devpool", boot_n=boot, seed=seed)
        cmp_v1 = compare(v1_csv, p_ext, "devpool", boot_n=boot, seed=seed)
        cmp_base = compare(csvs[pick], p_ext, "devpool", boot_n=boot, seed=seed)
        ext = S.devpool_summary(ev) | {
            "map50_by_fold": S.map50_by_fold(p_ext, root), "f1_threshold": ev["f1_threshold_from_devval"],
            "vs_v1": {k: cmp_v1["metrics"][k] for k in ("mAP50", "AR_small", "recall@op", "fp_per_neg_flight@op")},
            "vs_base": {k: cmp_base["metrics"][k] for k in ("mAP50", "AR_small", "recall@op")},
            "train_done": _load(runs / final / "train_done.json"), "extend": _load(runs / final / "extend.json")}
    fin = ext or table[pick]
    cmp_onnx = compare(v1_onnx_csv, oof_csv(cfg, final, root) if ext else csvs[pick], "devpool", boot_n=boot,
                       seed=seed)["metrics"]["mAP50"]
    thr_info = floor_threshold(oof_csv(cfg, final, root) if ext else csvs[pick], root)
    d = fin["vs_v1"]["mAP50"]
    kill_b = not (d["ci95"][0] is not None and d["ci95"][0] > 0)
    doc = {"phase": "06", "created": _now(), "candidates": table, "v1": v1, "pairwise": pair, "winner": pick,
           "final_model": final, "extension": ext, "winner_rule": why, "ranking_key": "cận dưới CI mAP50 OOF; CI chồng → C3 < C1 < C2",
           "kill_b": {"triggered": kill_b, "winner_minus_v1_map50": d, "v1_path": "pt (cùng đường với trò)",
                      "info_winner_pt_minus_v1_onnx_map50": cmp_onnx,
                      "rule": "xuất v2 chỉ khi cận dưới CI 95 % hiệu mAP50 OOF ghép cặp (trò − v1, cùng đường .pt) > 0; "
                              "`finalize` kiểm lại trên đường ONNX của bundle",
                      "note": "v1 (.pt E2 r1 best_devval) đã học nhãn giả khung dev và được chọn checkpoint trên dev-val "
                              "v1 (86/160 khung devpool) nên trên devpool v1 KHÔNG phải OOF — lợi thế cho v1"},
           "oof_threshold": thr_info["thr"], "oof_threshold_info": thr_info | {
               "unconstrained_thr": round(fin["f1_threshold"]["thr"], 6)}, "c2_gate": c2_gate(cfg),
           "bootstrap": {"n": boot, "seed": seed, "unit": "khối 30 s split.json (mỗi khối nằm trong một phiên = một fold)"},
           "gpu_used_h": gpu_used_h(cfg)}
    _dump(_res(root) / "select.json", doc)
    return {k: doc[k] for k in ("winner", "final_model", "winner_rule", "kill_b", "oof_threshold")} | {
        "oof_map50": {c: table[c]["metrics"]["mAP50"] for c in cands}, "v1_map50": v1["metrics"]["mAP50"]}


# ---------------------------------------------------------------- bản cuối + xuất + đo (GPU)


def final_dir(cfg: dict[str, Any], sel: dict[str, Any]) -> Path:
    fm = sel.get("final_model", sel["winner"])
    return Path(cfg["runs_dir"]) / ("C2_final" if fm == "C2" else fm)


def cmd_extend(cfg: dict[str, Any]) -> dict[str, Any]:
    """Nới giờ MỘT lần cho trò thắng (chủ dự án 25/11): chỉ khi mAP50 val (tập nguồn) còn đang lên ở epoch cuối (epoch
    tốt nhất = epoch cuối và tăng so với epoch trước). Huấn luyện tiếp từ ``last.pt`` của run thắng thành ``<thắng>_ext``,
    cùng dữ liệu; model ship = bản nới. Quyết định chỉ dựa val tập nguồn, không nhìn OOF hay test."""
    root = repo_root()
    sel = _load(_res(root) / "select.json")
    if not sel:
        raise FileNotFoundError("thiếu select.json (kfold select trước)")
    w, ecfg = sel["winner"], cfg["extension"]
    if w == "C2":
        raise ValueError("nới giờ cho C2 chưa hỗ trợ (bản cuối C2 huấn luyện lại trên mọi fold)")
    base = Path(cfg["runs_dir"]) / w
    hist = S.read_history(base / "val_history.jsonl")
    last, prev = hist[-1], hist[-2]
    best_ep = max(hist, key=lambda h: (h["val_mAP50"], -h["epoch"]))["epoch"]
    rising = best_ep == last["epoch"] and last["val_mAP50"] > prev["val_mAP50"]
    evidence = {"winner": w, "last_epochs": [{k: h[k] for k in ("epoch", "val_mAP50")} for h in hist[-3:]],
                "rising": rising, "rule": "epoch tốt nhất = epoch cuối và mAP50 val tăng ở epoch cuối"}
    if not rising:
        raise ValueError(f"không nới: mAP50 val của {w} không còn lên ({evidence['last_epochs']})")
    if gpu_used_h(cfg) + float(ecfg["max_h"]) > float(cfg["gpu_cap_h"]) + float(cfg["extension_h"]) + 1e-6:
        raise ValueError(f"không đủ trần: đã dùng {gpu_used_h(cfg)} h, nới {ecfg['max_h']} h > trần "
                         f"{cfg['gpu_cap_h']} + {cfg['extension_h']}")
    name = f"{w}_ext"
    (Path(cfg["runs_dir"]) / name).mkdir(parents=True, exist_ok=True)
    _dump(Path(cfg["runs_dir"]) / name / "extend.json", evidence | {"created": _now(), "init": (base / "weights" /
                                                                                                 "last.pt").as_posix()})
    ds = build_candidate(cfg, w, None, root)
    targs = train_args(cfg) | {"epochs": int(ecfg["epochs"]), "lr0": float(ecfg["lr0"]),
                               "warmup_epochs": float(ecfg["warmup_epochs"])}
    res = S.train_v2(cfg, name, base / "weights" / "last.pt", ds, targs, int(cfg["student"]["early_stop_patience"]),
                     f"v2 phase06 nới giờ {name}", cap_s=float(ecfg["max_h"]) * 3600 * 0.93, select="val")
    preds_p = Path(cfg["runs_dir"]) / name / "preds.csv"
    if not preds_p.is_file():
        ctx = real_context(root)
        frames = ctx["devpool"] + ctx["mine"] + _test_neg(root)
        write_csv(preds_p, predict_frames(cfg, Path(cfg["runs_dir"]) / name / "weights" / "best.pt", frames,
                                          f"v2 phase06 dự đoán {name}"))
    return {"name": name, "evidence": evidence, "train_done": res["train_done"], "gpu_used_h": gpu_used_h(cfg)}


def winner_run(cfg: dict[str, Any], sel: dict[str, Any]) -> Path:
    """Run của trò thắng: C1/C3 dùng luôn run bước 1; C2 huấn luyện lại trên mọi fold (``C2_final``)."""
    w = sel["winner"]
    if sel.get("final_model", w) != w:
        return Path(cfg["runs_dir"]) / sel["final_model"]
    if w == "C2":
        run_one(cfg, "C2_final", "C2", None)
        return Path(cfg["runs_dir"]) / "C2_final"
    return Path(cfg["runs_dir"]) / w


def test_metrics(csv_p: Path, thr: float, boot: int, seed: int) -> dict[str, Any]:
    """Chỉ số test ở ngưỡng OOF (chỉ chấm, không chọn gì): 5 khoá §4 + AR-small, kèm CI."""
    from xla.eval.detection import evaluate

    ev = evaluate(csv_p, "test", thresholds=[thr], boot_n=boot, seed=seed, f1_from_devval=False)
    m, k = ev["metrics"], f"{thr:g}"
    return {"bundle_keys": {"map50": m["mAP50"]["value"], "map50_95": m["mAP50_95"]["value"],
                            "precision": m[f"precision@{k}"]["value"], "recall": m[f"recall@{k}"]["value"],
                            "fp_per_neg_frame": m[f"fp_per_neg_frame@{k}"]["value"]},
            "mAP50": m["mAP50"], "AR_small": m["AR_small"], "AP_small": m["AP_small"], f"recall@{k}": m[f"recall@{k}"],
            f"precision@{k}": m[f"precision@{k}"], f"fp_per_neg_flight@{k}": m[f"fp_per_neg_flight@{k}"],
            f"fp_per_neg_indoor@{k}": m[f"fp_per_neg_indoor@{k}"], "n": ev["n"], "threshold": thr,
            "threshold_source": "F1 tối đa trên OOF devpool"}


def cmd_export(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    from xla import contract as C
    from xla.detect import export_onnx as X

    root = repo_root()
    res = _res(root)
    sel = _load(res / "select.json")
    if not sel:
        raise FileNotFoundError("thiếu results/v2/detect/select.json (kfold select)")
    if sel["kill_b"]["triggered"]:
        _dump(res / "export.json", {"exported": False, "reason": "Kill (b)", "kill_b": sel["kill_b"], "created": _now()})
        raise ValueError(f"Kill (b): {sel['winner']} không hơn v1 trên OOF {sel['kill_b']['winner_minus_v1_map50']}")
    run_dir = winner_run(cfg, sel)
    thr = float(sel["oof_threshold"])
    tm = test_metrics(run_dir / "preds.csv", thr, boot, seed)
    out_dir = root / cfg["export"]["out"]
    t0 = time.perf_counter()
    rep = X.export_bundle(run_dir / "weights" / "best.pt", out_dir, version=cfg["export"]["version"],
                          conf_threshold=thr, metrics=tm["bundle_keys"], train_run=f"{RESULTS_REL.as_posix()}/{run_dir.name}",
                          imgsz=int(cfg["train"]["imgsz"]), parity_out=res / "parity.json")
    log_gpu(cfg, f"export {out_dir.name} (parity + ms)", time.perf_counter() - t0)
    errors = rep["errors"] + C.validate_bundle(out_dir)
    folds_out = []
    if sel["winner"] == "C2":  # Deferred 30: ONNX từng fold cho phase 09 đo recall OOF thật
        fdir = Path(cfg["runs_dir"]) / "folds"
        for i in range(len(real_context(root)["folds"])):
            t0 = time.perf_counter()
            r = X.export_bundle(Path(cfg["runs_dir"]) / f"C2_fold{i}" / "weights" / "best.pt", fdir / f"fold{i}" /
                                f"detector-v2.0.{i + 100}", version=f"2.0.{i + 100}", conf_threshold=thr,
                                metrics=tm["bundle_keys"], train_run=f"C2_fold{i}", imgsz=int(cfg["train"]["imgsz"]))
            log_gpu(cfg, f"export fold {i}", time.perf_counter() - t0)
            onnx = fdir / f"detector-fold{i}.onnx"
            onnx.write_bytes((fdir / f"fold{i}" / f"detector-v2.0.{i + 100}" / "detector.onnx").read_bytes())
            folds_out.append({"fold": i, "onnx": onnx.as_posix(), "sha256": _sha256(onnx), "errors": r["errors"]})
    doc = {"exported": not errors, "created": _now(), "bundle": out_dir.as_posix(), "winner": sel["winner"],
           "weights": (run_dir / "weights" / "best.pt").as_posix(), "conf_threshold": thr, "errors": errors,
           "warnings": rep["warnings"], "ms_per_frame": rep["ms_per_frame"], "onnx_sha256": rep["onnx_sha256"],
           "parity_pass": rep["parity"]["pass"], "test_pt": tm, "fold_onnx": folds_out,
           "ms_p95_ok": rep["ms_per_frame"]["p95"] <= 30.0, "machine": S.machine_info()}
    _dump(res / "export.json", doc)
    if errors:
        raise ValueError(f"bundle không qua validate: {errors[:3]}")
    return {k: doc[k] for k in ("bundle", "winner", "conf_threshold", "ms_per_frame", "onnx_sha256", "parity_pass")}


def cmd_finalize(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    """Đường ONNX của bundle (cái thật sự ship, ``score_floor`` 0,05 trong graph) trên OOF devpool + mine, so với v1 cũng
    đường ONNX — Kill (b) chính thức và ngưỡng ship = F1 tối đa trên OOF ONNX. Chọn C1/C3 ở ``select`` vẫn trên đường
    ``.pt`` (cùng đường cho mọi ứng viên). Xuất lại bundle với ngưỡng + chỉ số test đường ONNX (5 khoá §4)."""
    import shutil

    from xla import contract as C
    from xla.detect import export_onnx as X
    from xla.eval.detection import compare, evaluate
    from xla.gpu_lock import gpu_lock

    root = repo_root()
    res = _res(root)
    sel = _load(res / "select.json")
    bundle = root / cfg["export"]["out"]
    onnx = bundle / "detector.onnx"
    if not sel or not onnx.is_file():
        raise FileNotFoundError("cần select.json + bundle đã xuất (kfold export)")
    ctx = real_context(root)
    oof_p = res / "preds" / "detector-v2.0.0_oof_onnx.csv"
    test_p = res / "preds" / "detector-v2.0.0_test_onnx.csv"
    with gpu_lock("v2 phase06 OOF + test đường ONNX v2", timeout_s=float(cfg["lock_timeout_s"])):
        t0 = time.perf_counter()
        write_csv(oof_p, S.onnx_predict(onnx, ctx["devpool"] + ctx["mine"], root, cuda=True))
        write_csv(test_p, S.onnx_predict(onnx, _test_neg(root), root, cuda=True))
        log_gpu(cfg, "OOF + test đường ONNX v2", time.perf_counter() - t0)
    check_oof_frames(oof_p, root)
    v1_csv = root / S.V2_RESULTS_REL / "preds" / "v1_devpool.csv"
    ev = evaluate(oof_p, "devpool", boot_n=boot, seed=seed)
    cmp_v1 = compare(v1_csv, oof_p, "devpool", boot_n=boot, seed=seed)
    d = cmp_v1["metrics"]["mAP50"]
    kill_b = not (d["ci95"][0] is not None and d["ci95"][0] > 0)
    thr = round(max(float(ev["f1_threshold_from_devval"]["thr"]), X.SCORE_FLOOR), 6)
    doc: dict[str, Any] = {"created": _now(), "model": sel["final_model"], "path": "ONNX (ORT CUDA), khung nguyên uint8",
                           "oof_onnx": S.devpool_summary(ev) | {"map50_by_fold": S.map50_by_fold(oof_p, root)},
                           "vs_v1_onnx": {k: cmp_v1["metrics"][k] for k in ("mAP50", "AR_small", "recall@op",
                                                                            "fp_per_neg_flight@op")},
                           "kill_b": {"triggered": kill_b, "rule": "cận dưới CI 95 % hiệu mAP50 OOF ghép cặp ONNX v2 − "
                                      "ONNX v1 > 0"}, "conf_threshold": thr,
                           "threshold_rule": "F1 tối đa trên OOF devpool đường ONNX (≥ score_floor)"}
    if kill_b:
        shutil.rmtree(bundle)
        doc["exported"] = False
        _dump(res / "finalize.json", doc)
        raise ValueError(f"Kill (b) trên đường ONNX: {d}; đã xoá {bundle}")
    tm = test_metrics(test_p, thr, boot, seed)
    t0 = time.perf_counter()
    rep = X.export_bundle(final_dir(cfg, sel) / "weights" / "best.pt", bundle, version=cfg["export"]["version"],
                          conf_threshold=thr, metrics=tm["bundle_keys"],
                          train_run=f"{RESULTS_REL.as_posix()}/{sel['final_model']}", imgsz=int(cfg["train"]["imgsz"]),
                          parity_out=res / "parity.json")
    log_gpu(cfg, "xuất lại bundle (ngưỡng ONNX)", time.perf_counter() - t0)
    errors = rep["errors"] + C.validate_bundle(bundle)
    doc |= {"exported": not errors, "errors": errors, "onnx_sha256": rep["onnx_sha256"],
            "ms_per_frame": rep["ms_per_frame"], "test_onnx": tm, "machine": S.machine_info()}
    _dump(res / "finalize.json", doc)
    if errors:
        raise ValueError(f"bundle không qua validate: {errors[:3]}")
    return {"conf_threshold": thr, "kill_b": doc["kill_b"], "oof_onnx_map50": doc["oof_onnx"]["metrics"]["mAP50"],
            "vs_v1_onnx_map50": doc["vs_v1_onnx"]["mAP50"], "ms": rep["ms_per_frame"], "onnx_sha256": rep["onnx_sha256"]}


def cmd_score_test(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    """Chấm test (chỉ chấm): trò thắng (đường ONNX của bundle nếu có + đường .pt) so edge16 và v1; mọi ứng viên để báo
    tương quan OOF ↔ test."""
    from xla.eval.detection import compare, evaluate
    from xla.gpu_lock import gpu_lock

    root = repo_root()
    res = _res(root)
    sel = _load(res / "select.json")
    if not sel:
        raise FileNotFoundError("thiếu select.json")
    fin = _load(res / "finalize.json")
    thr = float(fin["conf_threshold"]) if fin else float(sel["oof_threshold"])
    v1_meta = json.loads((S.V1_BUNDLE / "meta.json").read_text(encoding="utf-8"))
    v1_csv, base_csv = root / V1_TEST_CSV_REL, root / BASELINE_CSV_REL
    keep = set(_test_neg(root))
    runs = Path(cfg["runs_dir"])
    win = final_dir(cfg, sel)
    pt_csv = _filter_csv(win / "preds.csv", keep, res / "preds" / f"{sel['winner']}_test_pt.csv")
    doc: dict[str, Any] = {"created": _now(), "winner": sel["winner"], "threshold": thr, "note": "test chỉ chấm"}
    bundle = root / cfg["export"]["out"]
    paths = {"pt": pt_csv}
    trial = res / "preds" / "detector-v2.0.0_test_onnx.csv"
    if not (bundle / "detector.onnx").is_file() and trial.is_file():  # bundle thử đã xoá do Kill (b): vẫn chấm CSV
        paths["onnx_trial"] = trial
    if (bundle / "detector.onnx").is_file():
        with gpu_lock("v2 phase06 chấm test ONNX v2", timeout_s=float(cfg["lock_timeout_s"])):
            t0 = time.perf_counter()
            preds = S.onnx_predict(bundle / "detector.onnx", sorted(keep), root, cuda=True)
            log_gpu(cfg, "chấm test ONNX v2", time.perf_counter() - t0)
        paths["onnx"] = res / "preds" / "detector-v2.0.0_test_onnx.csv"
        write_csv(paths["onnx"], preds)
    for k, p in paths.items():
        doc[k] = test_metrics(p, thr, boot, seed) | {
            "vs_edge16": {m: v for m, v in compare(base_csv, p, "test", thr_b=thr, boot_n=boot, seed=seed)[
                "metrics"].items() if m in ("mAP50", "AR_small", "recall@op", "fp_per_neg_flight@op")},
            "vs_v1": {m: v for m, v in compare(v1_csv, p, "test", thr_a=float(v1_meta["conf_threshold"]), thr_b=thr,
                                               boot_n=boot, seed=seed)["metrics"].items()
                      if m in ("mAP50", "AR_small", "recall@op", "fp_per_neg_flight@op")}}
    doc["v1"] = test_metrics(v1_csv, float(v1_meta["conf_threshold"]), boot, seed) | {
        "threshold_source": "meta.json v1 (F1 dev-val)"}
    doc["edge16"] = {"mAP50": evaluate(base_csv, "test", boot_n=boot, seed=seed)["metrics"]["mAP50"]}
    corr = {}
    for c, e in (sel["candidates"]).items():
        rd = runs / (f"{c}_fold0" if c == "C2" else c)
        src = runs / "C2_final" / "preds.csv" if c == "C2" else rd / "preds.csv"
        if src.is_file():
            p = _filter_csv(src, keep, res / "preds" / f"{c}_test_pt.csv")
            corr[c] = {"oof_map50": e["metrics"]["mAP50"]["value"],
                       "test_map50": evaluate(p, "test", boot_n=0, f1_from_devval=False)["metrics"]["mAP50"]["value"]}
    doc["oof_vs_test"] = corr
    _dump(res / "test.json", doc)
    def brief(v: dict[str, Any]) -> dict[str, Any]:
        return {kk: (vv["value"] if isinstance(vv, dict) and "value" in vv else vv) for kk, vv in v.items()
                if kk.startswith(("mAP50", "AR_small", "recall@"))}

    return {"winner": doc["winner"], "threshold": thr, "oof_vs_test": corr} | {
        k: brief(doc[k]) for k in ("pt", "onnx", "onnx_trial", "v1") if k in doc}


def _test_neg(root: Path) -> list[str]:
    from xla.detect.baseline import select_frames

    sp = select_frames(root / S.SPLIT_REL, ("test", "neg"))
    return sp["test"] + sp["neg"]


def cmd_tiling(cfg: dict[str, Any], boot: int, seed: int) -> dict[str, Any]:
    """SAHI (Deferred 15, chỉ đo) với trò thắng trên test + neg: mỗi cỡ ô một lượt, mAP50 + ms; ô ≥ khung 640×480 nghĩa
    là một ô duy nhất (= không cắt), nên đo thêm ô nhỏ hơn ghi ở ``tiling.slices``."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as e:  # D8: không cài gói
        raise FileNotFoundError(f"SAHI không có trong venv ({e}): bỏ bước 8 (D8)") from e
    import cv2

    from xla.camera.frames import CROP_LEFT_PX
    from xla.detect.baseline import drop_edge
    from xla.eval.detection import evaluate
    from xla.gpu_lock import gpu_lock

    root = repo_root()
    res = _res(root)
    sel = _load(res / "select.json")
    tcfg = cfg["tiling"]
    w = final_dir(cfg, sel) / "weights" / "best.pt"
    frames = _test_neg(root)
    thr = float(sel["oof_threshold"])
    out: dict[str, Any] = {"created": _now(), "winner": sel["winner"], "weights": w.as_posix(), "results": {}}
    with gpu_lock("v2 phase06 tiling SAHI", timeout_s=float(cfg["lock_timeout_s"])):
        t_all = time.perf_counter()
        out["machine"] = S.machine_info()
        dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(w),
                                                confidence_threshold=float(cfg["predict"]["conf"]), device="cuda:0",
                                                image_size=int(cfg["train"]["imgsz"]))
        for sl in tcfg["slices"]:
            preds, ms = {}, []
            for f in frames:
                im = cv2.imread(str(root / FRAMES_REL / f), cv2.IMREAD_COLOR)[:, :, ::-1].copy()  # SAHI cần RGB
                t0 = time.perf_counter()
                r = get_sliced_prediction(im, dm, slice_height=int(sl), slice_width=int(sl),
                                          overlap_height_ratio=float(tcfg["overlap"]),
                                          overlap_width_ratio=float(tcfg["overlap"]), verbose=0)
                dt = (time.perf_counter() - t0) * 1000
                rows = [(*o.bbox.to_xyxy(), float(o.score.value)) for o in r.object_prediction_list]
                preds[f] = (drop_edge(rows, CROP_LEFT_PX), dt)
                ms.append(dt)
            p = res / "preds" / f"tiling_{sl}.csv"
            write_csv(p, preds)
            m = evaluate(p, "test", thresholds=[thr], boot_n=boot, seed=seed, f1_from_devval=False)["metrics"]
            out["results"][str(sl)] = {"slice": int(sl), "overlap": float(tcfg["overlap"]), "mAP50": m["mAP50"],
                                       "AR_small": m["AR_small"], f"recall@{thr:g}": m[f"recall@{thr:g}"],
                                       "ms_p50": round(float(np.median(ms)), 1),
                                       "ms_p95": round(float(np.percentile(ms, 95)), 1),
                                       "note": "PyTorch + SAHI (không phải ORT), gồm đường chuẩn khung nguyên"}
            if time.perf_counter() - t_all > float(tcfg["max_s"]):
                out["stopped"] = f"chạm trần {tcfg['max_s']} s sau ô {sl}"
                break
        log_gpu(cfg, "tiling SAHI", time.perf_counter() - t_all)
    _dump(res / "tiling.json", out)
    return {k: {kk: (vv["value"] if isinstance(vv, dict) and "value" in vv else vv) for kk, vv in v.items()}
            for k, v in out["results"].items()}


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.kfold", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("smoke", "run", "select", "extend", "export", "finalize", "score-test", "tiling",
                                    "gpu"))
    ap.add_argument("--config", type=Path, default=None, help=f"mặc định {CONFIG_REL}")
    ap.add_argument("--cand", default=None, help="run: chỉ ứng viên / run này (C1, C3, C2, C2_fold0, ...)")
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from xla.eval.detection import BOOT_N, BOOT_SEED, EvalInputError

    cfg = load_config(a.config or repo_root() / CONFIG_REL)
    boot, seed = (a.boot if a.boot is not None else BOOT_N), (a.seed if a.seed is not None else BOOT_SEED)
    try:
        out: Any = {"smoke": lambda: student_smoke(cfg), "run": lambda: cmd_run(cfg, a.cand),
                    "select": lambda: cmd_select(cfg, boot, seed),
                    "extend": lambda: cmd_extend(cfg), "finalize": lambda: cmd_finalize(cfg, boot, seed), "export": lambda: cmd_export(cfg, boot, seed),
                    "score-test": lambda: cmd_score_test(cfg, boot, seed), "tiling": lambda: cmd_tiling(cfg, boot, seed),
                    "gpu": lambda: {"gpu_used_h": gpu_used_h(cfg), "cap_h": cfg["gpu_cap_h"]}}[a.cmd]()
    except (FileNotFoundError, TimeoutError, ValueError, EvalInputError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, default=str) if a.json else
          json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
