"""Phiên bay + chia khung bay không người ``mine`` / ``cal`` / ``holdout`` theo phiên (AI v2 phase 03, hợp đồng §14.2).

Phiên bay = cửa sổ chuyến (``bay: có`` + video) của ``python -m backend.flights --events`` (IOT-CV) cắt theo từng lần
ghi, id ``<rec>#<i>`` (``xla.camera.frames.flight_sessions``). Ba phần của khung bay không người:

- ``mine``: ảnh nền huấn luyện detector + lớp âm verifier;
- ``cal``: chỉ hiệu chỉnh ngưỡng verifier, k/n, hàng rào ảnh offline;
- ``holdout``: chỉ chấm B2 một lần (phase 09), khung chỉ tính khi chủ dự án duyệt (``approved: true``).

Lệnh (mã thoát 0 xong / 1 dữ liệu thiếu hoặc sai / 2 sai cách gọi)::

    python -m xla.camera.sessions index --json             # backend.flights --events -> data/real/sessions.json
    python -m xla.camera.sessions prelabel --json          # RT-DETR-x + YOLO11x + SAHI conf 0,05 (GPU, gpu_lock D6)
    python -m xla.camera.sessions sheets --json            # ảnh ghép 4×4 khung neg dự kiến (agent soát)
    python -m xla.camera.sessions split --json             # -> data/real/neg_flight_split.json
    python -m xla.camera.sessions review --json            # trang duyệt holdout results/v2/neg-holdout/review/
    python -m xla.camera.sessions apply-review <json> --json   # áp file "Xuất JSON" của chủ dự án
    python -m xla.camera.sessions weak-holdout --sheets --json   # holdout YẾU cùng buổi (agent duyệt) -> khối holdout_weak
    python -m xla.camera.sessions score-weak --iotcv D:/IOTCV-wt/iotcv --json   # B2 trên holdout yếu -> holdout_weak.json

Gán phiên (seed ``SEED``): phiên chứa khung dev/test v1 không bao giờ là ``holdout``/``cal``; phiên còn lại xáo theo seed,
lấy ``holdout`` tới khi đủ ``HOLDOUT_MIN_SESSIONS`` phiên và ``HOLDOUT_MIN_FRAMES`` khung, rồi ``CAL_MIN_SESSIONS``
phiên ``cal``, còn lại ``mine``. Không còn phiên cho ``cal`` → ``cal_mode: "block"``: ``cal`` = các khối 30 s nguyên của
``split.json`` v1 (khối dev, mỗi phiên tối đa một khối, không lấy khối dev cuối của phiên) xáo theo seed tới khi đủ
``CAL_MIN_FRAMES`` khung không người; mọi khung của khối ``cal`` bị giữ khỏi ``mine`` và khỏi ``devpool`` (phase 04).

Khung "không người": nhãn trước (RT-DETR-x + YOLO11x + SAHI, ngưỡng ``PRELABEL_CONF``) hụt nhiều người nhìn thẳng từ
trên xuống (đo trên dev-val v1, README), nên **mọi** khung neg dự kiến được agent soát từng khung trên ảnh ghép 4×4
(``sessions sheets``), vùng nghi ngờ phóng bằng ``xla.detect.review crop``; kết quả ở ``AGENT_REVIEW_REL``
(``frame,verdict,reason``), khung ``person`` bị loại khỏi neg. ``mine``/``cal`` chỉ lấy khung ``dev`` của ``split.json``
(không lấy khung đệm cạnh khối test); khung ``devpool`` (phase 04) không vào ``mine``.

Holdout **yếu** (``weak-holdout``, quyết định chủ dự án 26/11/2026, không thay holdout chặt): khung bay không thuộc
``split.json`` v1, cách mọi khối dev/test v1 ≥ ``WEAK_MIN_DIST_S``, agent soát từng khung; nằm ở khoá gốc ``holdout_weak``
(``kind: "weak_same_day"``, ``approved_by: "agent"``, ``owner_approved: false``), ngoài ``frames``, nên đường chặt không đổi.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera import frames as F
from xla.paths import iotcv_root, repo_root

SEED = 20261125
SOURCE = "flights-events"
SPLITS = ("mine", "cal", "holdout")
HOLDOUT_MIN_SESSIONS = 2
HOLDOUT_MIN_FRAMES = 300
CAL_MIN_SESSIONS = 1
CAL_MIN_FRAMES = 150
"""``cal_mode: block``: dừng lấy khối khi số khung cal không người đạt mốc này."""
CAL_BLOCK_MIN_FRAMES = 20
"""Khối dev ít khung hơn thế không làm khối cal (khối vụn ở biên chuyến)."""
KILL_A_MIN_HOLDOUT = 150
"""Kill (a) của plan: holdout đã duyệt dưới mốc này → "không đủ chứng minh"."""
PRELABEL_CONF = 0.05
PRELABEL_CHECKPOINT_EVERY = 25
IOTCV_TIMEOUT_S = 900

SESSIONS_REL = Path("data/real/sessions.json")
NEG_SPLIT_REL = Path("data/real/neg_flight_split.json")
SPLIT_V1_REL = Path("data/real/split.json")
SPLIT_V1_SHA256 = "c1b0172840972b3951ffcd2060b482e87670015172f337830f8883815991192c"
DEVVAL_MANIFEST_REL = Path("data/real/labels/manifest.json")
DEVPOOL_JSON_REL = Path("data/real/labels/devpool/devpool.json")
OUT_REL = Path("results/v2/neg-holdout")
EVENTS_OUT_REL = OUT_REL / "flights_events.txt"
SESSIONS_REPORT_REL = OUT_REL / "sessions_report.json"
SPLIT_REPORT_REL = OUT_REL / "split_report.json"
PRELABEL_RAW_REL = OUT_REL / "prelabel" / "raw.json"
AGENT_REVIEW_REL = OUT_REL / "agent_review.csv"
REVIEW_DIR_REL = OUT_REL / "review"
REVIEW_APPLIED_REL = OUT_REL / "review_applied.json"
REVIEW_EXPORT_VERSION = "neg-holdout-review-v1"
REVIEW_STORAGE_KEY = "xla.neg-holdout-review.v1"
VERDICTS = ("person", "no_person")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(path: Path, doc: Any, indent: int | None = 1) -> None:
    """Ghi JSON (LF, UTF-8) qua file tạm rồi đổi tên: máy sập giữa chừng không để file dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def local_hms(t: float) -> str:
    return datetime.fromtimestamp(t, F.LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- index


def iotcv_child_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Env cho subprocess ``uv`` sang IOT-CV: **bỏ** ``UV_PROJECT_ENVIRONMENT``/``VIRTUAL_ENV`` (không thì uv xoá venv XLA)."""
    env = {k: v for k, v in (os.environ if environ is None else environ).items()
           if k not in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")}
    env["PYTHONUTF8"] = "1"
    return env


def run_flights_events(root: Path | None = None) -> str:
    """Đầu ra nguyên văn (stdout) của ``uv run --no-sync python -m backend.flights --events`` ở gốc IOT-CV."""
    cwd = root or iotcv_root()
    p = subprocess.run(["uv", "run", "--no-sync", "python", "-m", "backend.flights", "--events"], cwd=cwd,
                       env=iotcv_child_env(), capture_output=True, text=True, encoding="utf-8",
                       timeout=IOTCV_TIMEOUT_S, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"backend.flights --events exit {p.returncode}: {p.stderr.strip()[-800:]}")
    return p.stdout


def build_sessions_doc(windows: list[tuple[str, str, F.FlightWindow]], rows: list[dict[str, Any]]) -> dict:
    """Nội dung ``sessions.json`` (§14.2): mỗi phiên ``{id, rec, t_start, t_end, n_frames, source}``."""
    n = Counter(r["session"] for r in rows)
    return {"sessions": [{"id": sid, "rec": rec, "t_start": round(w.t0, 3), "t_end": round(w.t1, 3),
                          "n_frames": n.get(sid, 0), "source": SOURCE} for sid, rec, w in windows]}


def load_sessions(root: Path | None = None) -> list[dict[str, Any]]:
    return _load((root or repo_root()) / SESSIONS_REL)["sessions"]


def session_of(rec: str, t: float, sessions: list[dict[str, Any]]) -> str:
    """Phiên chứa khung (lần ghi, giờ Unix); chuỗi rỗng nếu ngoài mọi phiên."""
    for s in sessions:
        if s["rec"] == rec and s["t_start"] <= t <= s["t_end"]:
            return s["id"]
    return ""


def frame_brightness(path: Path) -> float:
    """Độ sáng = trung bình kênh xám (0–255) của cả khung."""
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"không đọc được {path}")
    return float(im.mean())


# ---------------------------------------------------------------- dữ liệu v1 (chỉ đọc)


def load_split_v1(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """``split.json`` v1 theo tên khung; kiểm sha256 (§14.2) để không ai lặng lẽ chia lại."""
    p = (root or repo_root()) / SPLIT_V1_REL
    sha = sha256_file(p)
    if sha != SPLIT_V1_SHA256:
        raise RuntimeError(f"{SPLIT_V1_REL} sha256 {sha} khác hợp đồng §14.2 {SPLIT_V1_SHA256}")
    return {f["frame"]: f for f in _load(p)["frames"]}


def flight_rows(root: Path | None = None) -> list[dict[str, Any]]:
    """Khung bay của ``manifest.csv`` (đã nhập) kèm ``session`` theo ``sessions.json``."""
    root = root or repo_root()
    sessions = load_sessions(root)
    rows = []
    for r in F.read_manifest(root / F.MANIFEST_REL):
        if r["kind"] != "flight":
            continue
        rows.append(r | {"session": session_of(r["rec"], float(r["t_unix"]), sessions)})
    return rows


def locked_sessions(rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]]) -> set[str]:
    """Phiên có khung dev/test v1 → không bao giờ ``holdout``/``cal``."""
    return {r["session"] for r in rows if split_v1.get(r["frame"], {}).get("split") in ("dev", "test")}


def neg_candidate_frames(rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]]) -> list[str]:
    """Khung có thể vào neg (mine/cal/holdout): khung ``dev`` v1 + mọi khung của phiên không bị khoá."""
    locked = locked_sessions(rows, split_v1)
    out = [r["frame"] for r in rows
           if split_v1.get(r["frame"], {}).get("split") == "dev" or (r["session"] and r["session"] not in locked)]
    return sorted(out)


# ---------------------------------------------------------------- nhãn trước (GPU)


def prelabel_params(conf: float) -> dict[str, Any]:
    from xla.detect import prelabel as P

    return {"conf": conf, "upscale": P.UPSCALE, "slice": P.SLICE, "overlap": P.OVERLAP, "edge_px": P.EDGE_PX,
            "imgsz": {m["key"]: m["imgsz"] for m in P.MODELS}, "sahi_postprocess": "NMS/IOS 0.5"}


def run_prelabel(frames: list[str], out: Path, conf: float = PRELABEL_CONF, label: str = "phase03 neg prelabel",
                 lock_timeout: float = 6 * 3600.0, max_frames: int | None = None) -> dict[str, Any]:
    """Hộp người thô của từng model (RT-DETR-x @640, YOLO11x @1280, SAHI như ``xla.detect.prelabel``) ở ngưỡng ``conf``.

    Ghi ``out`` = ``{version, params, models, frames: {khung: {rtdetr: [...], yolo: [...]}}, runs}``; có rồi thì chạy
    tiếp phần thiếu (checkpoint mỗi ``PRELABEL_CHECKPOINT_EVERY`` khung). Khoá GPU giữ suốt một lần chạy, nhả ngay sau.
    """
    import torch
    from PIL import Image
    from sahi import AutoDetectionModel

    from xla.detect import prelabel as P
    from xla.gpu_lock import gpu_lock
    from xla.paths import downloads_dir

    root = repo_root()
    params = prelabel_params(conf)
    doc = _load(out) if out.exists() else {"version": "neg-prelabel-v2", "params": params, "models": [], "frames": {},
                                           "runs": []}
    if doc["params"] != params:
        raise RuntimeError(f"{out}: params khác lần chạy trước ({doc['params']} ≠ {params}); xoá file để chạy lại")
    models_meta = []
    for m in P.MODELS:
        p = downloads_dir() / m["weights"]
        if not p.exists():
            raise RuntimeError(f"thiếu trọng số {p}")
        models_meta.append({"name": m["name"], "key": m["key"], "weights": p.name, "sha256": sha256_file(p)})
    if doc["models"] and doc["models"] != models_meta:
        raise RuntimeError(f"{out}: trọng số khác lần chạy trước")
    doc["models"] = models_meta
    todo = {m["key"]: [f for f in frames if m["key"] not in doc["frames"].get(f, {})] for m in P.MODELS}
    if max_frames is not None:
        todo = {k: v[:max_frames] for k, v in todo.items()}
    done_now = 0
    t_wait = time.monotonic()
    gpu_s = 0.0
    if any(todo.values()):
        with gpu_lock(label, timeout_s=lock_timeout):
            t0 = time.monotonic()
            for m in P.MODELS:
                if not todo[m["key"]]:
                    continue
                dm = AutoDetectionModel.from_pretrained(model_type=m["type"], model_path=str(downloads_dir() / m["weights"]),
                                                        confidence_threshold=conf, device="cuda:0",
                                                        image_size=m["imgsz"])
                for i, f in enumerate(todo[m["key"]], 1):
                    im = Image.open(root / F.FRAMES_REL / f).convert("RGB")
                    if im.size != (P.W, P.H):
                        raise RuntimeError(f"{f}: cỡ {im.size}, cần {(P.W, P.H)}")
                    up = np.asarray(im.resize((P.W * P.UPSCALE, P.H * P.UPSCALE), Image.BICUBIC))
                    doc["frames"].setdefault(f, {})[m["key"]] = P._predict(dm, up, conf)
                    done_now += 1
                    if i % PRELABEL_CHECKPOINT_EVERY == 0:
                        _dump(out, doc, indent=None)
                        print(f"[prelabel] {m['key']} {i}/{len(todo[m['key']])}", file=sys.stderr, flush=True)
                _dump(out, doc, indent=None)
                del dm
                torch.cuda.empty_cache()
            gpu_s = time.monotonic() - t0
            waited = t0 - t_wait
        doc["runs"].append({"finished": datetime.now().isoformat(timespec="seconds"), "frame_model_pairs": done_now,
                            "gpu_s": round(gpu_s, 1), "lock_wait_s": round(waited, 1)})
        _dump(out, doc, indent=None)
    missing = sorted(f for f in frames if any(m["key"] not in doc["frames"].get(f, {}) for m in P.MODELS))
    return {"frames": len(frames), "frame_model_pairs_done_now": done_now, "missing": len(missing),
            "gpu_s": round(gpu_s, 1), "gpu_s_total": round(sum(r["gpu_s"] for r in doc["runs"]), 1),
            "out": str(out.relative_to(root)) if out.is_relative_to(root) else str(out)}


# ---------------------------------------------------------------- lệnh index / prelabel


def cmd_index(args: argparse.Namespace) -> int:
    root = repo_root()
    if args.events:
        events = Path(args.events).read_text(encoding="utf-8")
    else:
        events = run_flights_events()
    (root / EVENTS_OUT_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / EVENTS_OUT_REL).write_text(events, encoding="utf-8", newline="\n")
    ranges = F.all_recording_ranges()
    rows, counts = F.extract_flight(None, events, ranges)
    windows = F.flight_sessions(F.parse_flight_events(events), ranges)
    manifest = {r["frame"]: r for r in F.read_manifest(root / F.MANIFEST_REL) if r["kind"] == "flight"}
    names = {r["frame"] for r in rows}
    not_ingested = sorted(r["frame"] for r in rows if r["frame"] not in manifest
                          or manifest[r["frame"]]["sha256"] != r["sha256"]
                          or not (root / F.FRAMES_REL / r["frame"]).is_file())
    stale = sorted(f for f in manifest if f not in names)
    doc = build_sessions_doc(windows, rows)
    _dump(root / SESSIONS_REL, doc)
    split_v1 = load_split_v1(root)
    by_sess: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sess[r["session"]].append(r)
    devval = set(_load(root / DEVVAL_MANIFEST_REL)["devval_frames"])
    per_session = []
    for s in doc["sessions"]:
        rs = by_sess.get(s["id"], [])
        v1 = Counter(split_v1.get(r["frame"], {}).get("split", "none") for r in rs)
        light = [frame_brightness(root / F.FRAMES_REL / r["frame"]) for r in rs if r["frame"] in manifest]
        per_session.append({
            "id": s["id"], "flight_id": next((w.flight_id for sid, _, w in windows if sid == s["id"]), ""),
            "start_local": local_hms(s["t_start"]), "duration_s": round(s["t_end"] - s["t_start"], 1),
            "n_frames": s["n_frames"], "alt_median_m": round(float(np.median([r["alt_m"] for r in rs])), 2) if rs else None,
            "brightness_median": round(float(np.median(light)), 1) if light else None,
            "v1_split": dict(sorted(v1.items())), "v1_devval": sum(r["frame"] in devval for r in rs)})
    per_rec = {}
    for rec, (lo, hi) in sorted(ranges.items()):
        n = sum(1 for s in doc["sessions"] if s["rec"] == rec)
        alt = [F._float_or_nan(r.get("rel_alt_m")) for r in F.read_rows(F.recordings_dir() / rec)]
        alt = [a for a in alt if np.isfinite(a)]
        per_rec[rec] = {"start_local": local_hms(lo), "end_local": local_hms(hi), "sessions": n,
                        "frames_kept": sum(s["n_frames"] for s in doc["sessions"] if s["rec"] == rec),
                        "rel_alt_max_m": round(max(alt), 2) if alt else None}
    report = {"counts": counts, "recordings": per_rec, "sessions": per_session, "not_ingested": not_ingested[:20],
              "n_not_ingested": len(not_ingested), "stale_in_manifest": stale[:20], "n_stale_in_manifest": len(stale),
              "events_source": args.events or "uv run --no-sync python -m backend.flights --events (IOT-CV)"}
    _dump(root / SESSIONS_REPORT_REL, report)
    summary = {"sessions": len(doc["sessions"]), "frames": len(rows),
               "per_recording": {k: v["sessions"] for k, v in per_rec.items()},
               "n_not_ingested": len(not_ingested), "n_stale_in_manifest": len(stale)}
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    if not_ingested or stale:
        print("LỖI: khung bay chưa nhập / manifest lệch: chạy `python -m xla.camera.frames extract` rồi index lại",
              file=sys.stderr)
        return 1
    return 0


def cmd_prelabel(args: argparse.Namespace) -> int:
    root = repo_root()
    rows = flight_rows(root)
    frames = neg_candidate_frames(rows, load_split_v1(root))
    if args.extra_frames:
        frames = sorted(set(frames) | set(Path(args.extra_frames).read_text(encoding="utf-8").split()))
    summary = run_prelabel(frames, root / PRELABEL_RAW_REL, conf=args.conf, lock_timeout=args.lock_timeout,
                           max_frames=args.max_frames)
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    return 0 if summary["missing"] == 0 or args.max_frames else 1


# ---------------------------------------------------------------- ứng viên của nhãn trước


def fuse_raw(raw_frame: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Hộp hợp hai model theo đúng luật ``xla.detect.prelabel`` (lọc ``MODEL_CONF`` → WBF → ``keep``), có ``id``."""
    from xla.detect import prelabel as P

    per = {k: [b for b in raw_frame.get(k, []) if b["conf"] >= P.MODEL_CONF] for k in (m["key"] for m in P.MODELS)}
    kept = [b for b in P.fuse(per) if P.keep(b)]
    for i, b in enumerate(kept):
        b["id"] = i + 1
    return kept


def fused_candidate_frames(frames: list[str], raw: dict[str, Any]) -> set[str]:
    """Khung có ít nhất một hộp hợp (luật nhãn trước v1): ước lượng "có người" **trước** khi soát."""
    return {f for f in frames if fuse_raw(raw[f])}


# ---------------------------------------------------------------- soát của agent (từng khung)

NEG_SHEET_SCALE = 0.5
"""Ô ảnh ghép soát neg: khung thu 0,5× (320×240), 4×4 ô mỗi tờ, theo thứ tự giờ trong từng phiên."""
SHEETS_INDEX_REL = OUT_REL / "agent_review_sheets.csv"


def read_agent_review(path: Path) -> dict[str, dict[str, str]]:
    """``frame,verdict,reason`` → ``{khung: dòng}``; ``verdict`` ∈ ``VERDICTS``; khung lặp → lỗi."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r = {k: (v or "").strip() for k, v in r.items()}
            if r["verdict"] not in VERDICTS:
                raise ValueError(f"{path}: verdict lạ {r['verdict']!r} ở {r['frame']}")
            if r["frame"] in out:
                raise ValueError(f"{path}: khung {r['frame']} có hai dòng")
            out[r["frame"]] = r
    return out


def neg_sheets(frames: list[dict[str, Any]], out_dir: Path, cols: int = 4, rows: int = 4) -> list[dict[str, Any]]:
    """Ảnh ghép 4×4 của các khung neg dự kiến (``[{frame, session, split}]`` đã sắp) để agent soát từng khung."""
    import cv2

    from xla.detect.prelabel import FRAMES_REL

    w, h = round(640 * NEG_SHEET_SCALE), round(480 * NEG_SHEET_SCALE)
    per = cols * rows
    index = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for si in range(0, len(frames), per):
        tiles = []
        for ci, f in enumerate(frames[si:si + per]):
            im = cv2.imread(str(repo_root() / FRAMES_REL / f["frame"]), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            cap = f"{ci:02d} {f['split']} {f['frame'][16:-4]}"
            pad = cv2.copyMakeBorder(im, 16, 2, 1, 1, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            cv2.putText(pad, cap, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(pad)
            index.append({"sheet": si // per, "cell": ci, **f})
        tiles += [np.zeros_like(tiles[0])] * (per - len(tiles))
        sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(out_dir / f"neg_{si // per:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return index


# ---------------------------------------------------------------- chia mine / cal / holdout

MINE_MIN_GAP_S = 0.5
"""Khung ``mine`` trong phiên có dev/test v1 cách nhau ≥ 0,5 s (khung kề 0,2 s gần như trùng; giảm việc soát)."""


def assign_sessions(n_frames: dict[str, int], locked: set[str], rng: np.random.Generator,
                    ) -> tuple[dict[str, str], str]:
    """Vai của từng phiên + ``cal_mode`` (``"session"`` | ``"block"``). Phiên khoá luôn ``mine``."""
    roles = {s: "mine" for s in n_frames if s in locked}
    free = sorted(s for s in n_frames if s not in locked)
    free = [free[i] for i in rng.permutation(len(free))]
    holdout: list[str] = []
    while free:
        enough = len(holdout) >= HOLDOUT_MIN_SESSIONS and sum(n_frames[h] for h in holdout) >= HOLDOUT_MIN_FRAMES
        if enough or (len(holdout) >= HOLDOUT_MIN_SESSIONS and len(free) <= CAL_MIN_SESSIONS):
            break
        holdout.append(free.pop(0))
    cal, mine = free[:CAL_MIN_SESSIONS], free[CAL_MIN_SESSIONS:]
    roles |= {s: "holdout" for s in holdout} | {s: "cal" for s in cal} | {s: "mine" for s in mine}
    return dict(sorted(roles.items())), "session" if cal else "block"


def choose_cal_blocks(rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]], roles: dict[str, str],
                      lost: set[str], rng: np.random.Generator) -> tuple[list[int], list[dict[str, Any]]]:
    """``cal_mode: block``: khối dev v1 nguyên làm ``cal``.

    Ứng viên: khối có ≥ ``CAL_BLOCK_MIN_FRAMES`` khung ``dev`` trong phiên ``mine``. Thứ tự: ít khung "mất" nhất trước
    (``lost`` = dev-val + khung có hộp hợp của nhãn trước — biết **trước** khi soát nên chọn khối không phụ thuộc kết quả
    soát; chúng bị giữ khỏi devpool cùng khối), hoà thì theo ``rng``. Mỗi phiên tối đa một khối và luôn giữ ≥ 1 khối dev
    cho ``mine``; dừng khi số khung không "mất" của các khối đã chọn ≥ ``CAL_MIN_FRAMES``.
    """
    dev_of_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        v = split_v1.get(r["frame"])
        if v and v["split"] == "dev" and roles.get(r["session"]) == "mine":
            dev_of_block[v["block"]].append(r)
    blocks_of_session: dict[str, set[int]] = defaultdict(set)
    for b, rs in dev_of_block.items():
        blocks_of_session[rs[0]["session"]].add(b)
    cands = sorted(b for b, rs in dev_of_block.items() if len(rs) >= CAL_BLOCK_MIN_FRAMES)
    rank = {b: i for i, b in enumerate(cands[i] for i in rng.permutation(len(cands)))}
    info = {b: {"block": b, "session": dev_of_block[b][0]["session"], "n_dev": len(dev_of_block[b]),
                "n_lost": sum(r["frame"] in lost for r in dev_of_block[b]),
                "n_neg": sum(r["frame"] not in lost for r in dev_of_block[b])} for b in cands}
    order = sorted(cands, key=lambda b: (info[b]["n_lost"], rank[b]))
    chosen: list[int] = []
    for b in order:
        s = info[b]["session"]
        if sum(info[c]["n_neg"] for c in chosen) >= CAL_MIN_FRAMES:
            break
        if any(info[c]["session"] == s for c in chosen) or len(blocks_of_session[s]) <= 1:
            continue
        chosen.append(b)
    return sorted(chosen), [info[b] | {"chosen": b in chosen} for b in order]


def plan_neg(rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]], lost: set[str], devpool: set[str],
             seed: int = SEED) -> dict[str, Any]:
    """Cấu trúc chia (vai phiên, ``cal_mode``, khối cal) + khung neg **dự kiến** trước khi bỏ khung có người.

    Không phụ thuộc kết quả soát, nên khung cần soát không đổi khi soát xong. Khung dự kiến: ``holdout``/``cal`` theo
    phiên = mọi khung của phiên đó; phiên có dev/test v1: ``cal`` = mọi khung ``dev`` của khối cal, ``mine`` = khung
    ``dev`` còn lại không thuộc ``devpool``, cách nhau ≥ ``MINE_MIN_GAP_S`` trong từng phiên. Thuần, tất định.
    """
    rng = np.random.default_rng(seed)
    n_frames = Counter(r["session"] for r in rows if r["session"])
    locked = locked_sessions(rows, split_v1)
    roles, cal_mode = assign_sessions(dict(n_frames), locked, rng)
    cal_blocks: list[int] = []
    table: list[dict[str, Any]] = []
    if cal_mode == "block":
        cal_blocks, table = choose_cal_blocks(rows, split_v1, roles, lost, rng)
    planned, skipped = [], Counter()
    last: dict[str, float] = {}
    for r in sorted(rows, key=lambda r: (r["session"], float(r["t_unix"]), r["frame"])):
        f, s = r["frame"], r["session"]
        v = split_v1.get(f)
        if s not in roles:
            skipped["outside_session"] += 1
            continue
        if s in locked:
            if not v or v["split"] != "dev":
                skipped["locked_not_dev" if not v else f"locked_{v['split']}"] += 1
                continue
            split = "cal" if v["block"] in cal_blocks else "mine"
        else:
            split = roles[s]
        if split == "mine":
            if f in devpool:
                skipped["devpool"] += 1
                continue
            if float(r["t_unix"]) - last.get(s, -1e18) < MINE_MIN_GAP_S:
                skipped["mine_spacing"] += 1
                continue
            last[s] = float(r["t_unix"])
        planned.append({"frame": f, "session": s, "split": split})
    return {"seed": seed, "cal_mode": cal_mode, "roles": roles, "locked": sorted(locked), "cal_blocks": cal_blocks,
            "cal_block_candidates": table, "planned": planned, "skipped": dict(sorted(skipped.items()))}


def build_neg_split(plan: dict[str, Any], person: set[str], devval: set[str],
                    split_v1: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Nội dung ``neg_flight_split.json`` (§14.2, thêm ``cal_mode``) + báo cáo: khung dự kiến trừ khung có người."""
    frames = [x | {"approved": False} for x in sorted(plan["planned"], key=lambda x: x["frame"])
              if x["frame"] not in person]
    doc = {"seed": plan["seed"], "cal_mode": plan["cal_mode"], "sessions": plan["roles"], "frames": frames}
    by = Counter((x["session"], x["split"]) for x in frames)
    planned = Counter(x["split"] for x in plan["planned"])
    removed = Counter(x["split"] for x in plan["planned"] if x["frame"] in person)
    report = {
        "seed": plan["seed"], "cal_mode": plan["cal_mode"], "locked_sessions": plan["locked"],
        "counts": dict(Counter(x["split"] for x in frames)), "planned": dict(planned),
        "removed_person": dict(removed),
        "per_session": {s: {"role": r, **{k: by[(s, k)] for k in SPLITS if by[(s, k)]}} for s, r in plan["roles"].items()},
        "cal_blocks": plan["cal_blocks"], "cal_block_candidates": plan["cal_block_candidates"],
        "skipped": plan["skipped"],
        "devval_in_cal_blocks": sorted(f for f in devval if split_v1.get(f, {}).get("block") in plan["cal_blocks"]),
    }
    return doc, report


def load_neg_split(root: Path | None = None) -> dict[str, Any]:
    return _load((root or repo_root()) / NEG_SPLIT_REL)


def reserved_frames(neg_doc: dict[str, Any], rows: list[dict[str, Any]],
                    split_v1: dict[str, dict[str, Any]]) -> set[str]:
    """Khung không bao giờ được vào devpool / huấn luyện: mọi khung của phiên ``cal``/``holdout`` và mọi khung của khối
    ``cal`` (``cal_mode: block``), kể cả khung có người của khối đó."""
    roles = neg_doc["sessions"]
    cal_blocks = {split_v1[x["frame"]]["block"] for x in neg_doc["frames"]
                  if x["split"] == "cal" and x["frame"] in split_v1}
    out = {r["frame"] for r in rows if roles.get(r["session"]) in ("cal", "holdout")}
    if neg_doc.get("cal_mode") == "block":
        out |= {r["frame"] for r in rows if split_v1.get(r["frame"], {}).get("block") in cal_blocks}
    return out | {x["frame"] for x in neg_doc["frames"] if x["split"] in ("cal", "holdout")}


def check_split(neg_doc: dict[str, Any], rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]],
                ) -> list[str]:
    """Lỗi hợp đồng của ``neg_flight_split.json`` (rỗng = đạt): khoá, enum, phiên/khối cal + holdout sạch dev/test."""
    errs = []
    if set(neg_doc) - {"seed", "cal_mode", "sessions", "frames", WEAK_KEY} or neg_doc.get("seed") != SEED:
        errs.append(f"khoá/seed sai: {sorted(neg_doc)} seed={neg_doc.get('seed')}")
    if neg_doc.get("cal_mode") not in ("session", "block"):
        errs.append(f"cal_mode lạ {neg_doc.get('cal_mode')!r}")
    roles = neg_doc["sessions"]
    if any(v not in SPLITS for v in roles.values()):
        errs.append("vai phiên ngoài mine/cal/holdout")
    sess_of = {r["frame"]: r["session"] for r in rows}
    v1_by_session: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["frame"] in split_v1:
            v1_by_session[r["session"]].add(split_v1[r["frame"]]["split"])
    for s, role in roles.items():
        if role in ("cal", "holdout") and v1_by_session[s] & {"dev", "test"}:
            errs.append(f"phiên {role} {s} có khung {sorted(v1_by_session[s] & {'dev', 'test'})} v1")
    seen = set()
    cal_blocks = set()
    for x in neg_doc["frames"]:
        f = x["frame"]
        if set(x) != {"frame", "session", "split", "approved"} or x["split"] not in SPLITS:
            errs.append(f"dòng khung sai dạng: {x}")
            continue
        if f in seen:
            errs.append(f"khung lặp {f}")
        seen.add(f)
        if sess_of.get(f) != x["session"]:
            errs.append(f"{f}: session {x['session']!r} khác {sess_of.get(f)!r}")
        v = split_v1.get(f)
        if v and v["split"] == "test":
            errs.append(f"{f}: khung test v1 trong neg ({x['split']})")
        if x["split"] == "holdout" and v is not None:
            errs.append(f"{f}: khung holdout có trong split.json v1 ({v['split']})")
        if x["split"] == "cal" and neg_doc.get("cal_mode") == "block":
            if not v or v["split"] != "dev":
                errs.append(f"{f}: khung cal (block) không phải dev v1")
            else:
                cal_blocks.add(v["block"])
        if x["split"] == "holdout" and x["approved"] not in (True, False):
            errs.append(f"{f}: approved không phải bool")
    for x in neg_doc["frames"]:
        v = split_v1.get(x["frame"])
        if x["split"] == "mine" and v and v["block"] in cal_blocks:
            errs.append(f"{x['frame']}: khung mine nằm trong khối cal {v['block']} (khối không nguyên)")
    return errs


def current_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(kế hoạch chia từ dữ liệu trên đĩa, ngữ cảnh ``{rows, v1, devval}``)."""
    rows = flight_rows(root)
    split_v1 = load_split_v1(root)
    raw = _load(root / PRELABEL_RAW_REL)["frames"]
    missing = [f for f in neg_candidate_frames(rows, split_v1) if f not in raw]
    if missing:
        raise ValueError(f"{len(missing)} khung chưa có nhãn trước (chạy `sessions prelabel`): {missing[:3]}")
    devval = set(_load(root / DEVVAL_MANIFEST_REL)["devval_frames"])
    devpool_p = root / DEVPOOL_JSON_REL
    devpool = set(_load(devpool_p)["frames"]) if devpool_p.exists() else set()
    lost = devval | fused_candidate_frames([r["frame"] for r in rows if r["frame"] in raw], raw)
    return plan_neg(rows, split_v1, lost, devpool), {"rows": rows, "v1": split_v1, "devval": devval,
                                                     "devpool": devpool}


def cmd_split(args: argparse.Namespace) -> int:
    root = repo_root()
    plan, ctx = current_plan(root)
    review = read_agent_review(root / AGENT_REVIEW_REL)
    planned = [x["frame"] for x in plan["planned"]]
    unreviewed = [f for f in planned if f not in review]
    if unreviewed and not args.allow_unreviewed:
        print(json.dumps({"unreviewed": len(unreviewed), "first": unreviewed[:10]}, ensure_ascii=False))
        print(f"LỖI: {len(unreviewed)} khung neg dự kiến chưa soát ({AGENT_REVIEW_REL}; `sessions sheets`)",
              file=sys.stderr)
        return 1
    person = {f for f, r in review.items() if r["verdict"] == "person"}
    doc, report = build_neg_split(plan, person, ctx["devval"], ctx["v1"])
    errs = check_split(doc, ctx["rows"], ctx["v1"])
    holdout = sum(1 for x in doc["frames"] if x["split"] == "holdout")
    report |= {"unreviewed": len(unreviewed), "review_rows_not_planned": len(set(review) - set(planned)),
               "devpool_frames": len(ctx["devpool"]), "errors": errs,
               "kill_a": {"holdout_frames_pre_approval": holdout, "holdout_approved": 0, "min": KILL_A_MIN_HOLDOUT,
                          "verdict": "không đủ chứng minh" if holdout < KILL_A_MIN_HOLDOUT else "chờ duyệt"}}
    if errs:
        print(json.dumps(report, ensure_ascii=False))
        return 1
    old = root / NEG_SPLIT_REL
    if old.exists() and WEAK_KEY in (prev := _load(old)):  # khối holdout yếu do `weak-holdout` quản, giữ nguyên
        doc[WEAK_KEY] = prev[WEAK_KEY]
    _dump(root / NEG_SPLIT_REL, doc)
    _dump(root / SPLIT_REPORT_REL, report)
    out = {k: report[k] for k in ("cal_mode", "counts", "planned", "removed_person", "cal_blocks", "kill_a",
                                  "unreviewed")}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_sheets(args: argparse.Namespace) -> int:
    """Ảnh ghép 4×4 mọi khung neg dự kiến (cal trước, rồi mine/holdout; theo giờ trong phiên) + chỉ mục ô → khung."""
    from xla.paths import cache_dir

    root = repo_root()
    plan, ctx = current_plan(root)
    t = {r["frame"]: float(r["t_unix"]) for r in ctx["rows"]}
    rank = {"cal": 0, "holdout": 1, "mine": 2}
    frames = sorted(plan["planned"], key=lambda x: (rank[x["split"]], x["session"], t[x["frame"]]))
    if args.only_unreviewed:
        review = read_agent_review(root / AGENT_REVIEW_REL)
        frames = [x for x in frames if x["frame"] not in review]
    out_dir = Path(args.out) if args.out else cache_dir() / "xla-neg-review"
    index = neg_sheets(frames, out_dir)
    p = root / SHEETS_INDEX_REL
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("sheet", "cell", "frame", "session", "split"), lineterminator="\n")
        w.writeheader()
        w.writerows(index)
    out = {"frames": len(frames), "sheets": -(-len(frames) // 16), "dir": str(out_dir), "index": str(SHEETS_INDEX_REL),
           "planned": dict(Counter(x["split"] for x in plan["planned"])), "cal_blocks": plan["cal_blocks"]}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


# ---------------------------------------------------------------- trang duyệt holdout (cổng người)

GRID = 16
"""Khung mỗi lưới (4×4)."""

_REVIEW_HTML = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duyệt neg_flight giữ kín (holdout)</title>
<style>
:root { --bg: #f6f7f9; --card: #fff; --ink: #1d2330; --muted: #5b6475; --line: #d9dde5; --accent: #1f6feb;
        --warn: #c2410c; --ok: #15803d; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 system-ui, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
header { position: sticky; top: 0; z-index: 2; background: var(--card); border-bottom: 1px solid var(--line);
         padding: 10px 18px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
header h1 { font-size: 16px; margin: 0 12px 0 0; }
header .stat { color: var(--muted); }
header .stat b { color: var(--ink); }
button { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line); background: var(--card);
         cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
main { padding: 16px 18px 60px; max-width: 1500px; margin: 0 auto; }
.help { color: var(--muted); margin: 0 0 16px; }
.empty { padding: 40px; text-align: center; background: var(--card); border: 1px dashed var(--line); border-radius: 8px; }
section.grid { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 18px; padding: 10px; }
section.grid.seen { border-color: var(--ok); }
section.grid h2 { font-size: 14px; margin: 0 0 8px; display: flex; gap: 12px; align-items: center; }
.cells { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.cell { position: relative; border: 2px solid transparent; border-radius: 4px; }
.cell.person { border-color: var(--warn); }
.cell img { width: 100%; aspect-ratio: 4 / 3; display: block; background: #222; }
.cell label { display: flex; gap: 6px; align-items: center; padding: 3px 4px; font-size: 12px; color: var(--muted); }
.cell.person label { color: var(--warn); font-weight: 600; }
</style>
</head>
<body>
<header>
  <h1>Duyệt neg_flight giữ kín</h1>
  <span class="stat">Khung: <b id="n-frames">0</b></span>
  <span class="stat">Có người: <b id="n-person">0</b></span>
  <span class="stat">Lưới đã xem: <b id="n-seen">0</b> / <b id="n-grids">0</b></span>
  <button class="primary" id="export">Xuất JSON</button>
  <button id="reset">Xoá trạng thái</button>
</header>
<main>
  <p class="help">Đánh dấu <b>có người</b> cho mọi khung thấy người (kể cả rất nhỏ, mờ, một phần). Khung không đánh dấu
  được tính là <b>không người</b>. Xem hết một lưới thì tích <b>Đã xem lưới này</b>. Bấm ảnh để mở cỡ gốc. Trạng thái lưu
  trong trình duyệt (localStorage); xong thì bấm <b>Xuất JSON</b> và đưa file cho agent chạy
  <code>python -m xla.camera.sessions apply-review &lt;file&gt; --json</code>.</p>
  <div id="grids"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("data").textContent);
  var KEY = DATA.storage_key + ":" + DATA.split_sha256.slice(0, 12);
  var state = { person: {}, seen: {} };
  try { var s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.person && s.seen) { state = s; } }
  catch (e) { console.warn("localStorage hỏng, bắt đầu lại", e); }
  function save() { localStorage.setItem(KEY, JSON.stringify(state)); render(); }
  var grids = [];
  for (var i = 0; i < DATA.frames.length; i += DATA.grid) { grids.push(DATA.frames.slice(i, i + DATA.grid)); }
  var root = document.getElementById("grids");
  if (!grids.length) {
    root.innerHTML = '<div class="empty">Chưa có khung holdout nào (0 khung). Xem ' +
      '<code>results/v2/neg-holdout/README.md</code>: không có phiên bay nào ngoài các phiên chứa khung dev/test v1.</div>';
  }
  grids.forEach(function (g, gi) {
    var sec = document.createElement("section");
    sec.className = "grid"; sec.dataset.grid = gi;
    var h = document.createElement("h2");
    h.innerHTML = "Lưới " + (gi + 1) + " / " + grids.length + ' <label><input type="checkbox" class="seen"> Đã xem lưới này</label>';
    sec.appendChild(h);
    var cells = document.createElement("div"); cells.className = "cells";
    g.forEach(function (f) {
      var c = document.createElement("div"); c.className = "cell"; c.dataset.frame = f.frame;
      var a = document.createElement("a"); a.href = DATA.img_prefix + f.frame; a.target = "_blank"; a.rel = "noopener";
      var img = document.createElement("img"); img.loading = "lazy"; img.src = DATA.img_prefix + f.frame; img.alt = f.frame;
      a.appendChild(img); c.appendChild(a);
      var lab = document.createElement("label");
      lab.innerHTML = '<input type="checkbox" class="person"> có người <span>' + f.frame.slice(16, -4) + " · " + f.session + "</span>";
      c.appendChild(lab); cells.appendChild(c);
    });
    sec.appendChild(cells); root.appendChild(sec);
  });
  root.addEventListener("change", function (e) {
    var t = e.target;
    if (t.classList.contains("person")) {
      var f = t.closest(".cell").dataset.frame;
      if (t.checked) { state.person[f] = true; } else { delete state.person[f]; }
      save();
    } else if (t.classList.contains("seen")) {
      var gi = t.closest("section").dataset.grid;
      if (t.checked) { state.seen[gi] = true; } else { delete state.seen[gi]; }
      save();
    }
  });
  function render() {
    document.querySelectorAll(".cell").forEach(function (c) {
      var on = !!state.person[c.dataset.frame];
      c.classList.toggle("person", on); c.querySelector("input.person").checked = on;
    });
    document.querySelectorAll("section.grid").forEach(function (s) {
      var on = !!state.seen[s.dataset.grid];
      s.classList.toggle("seen", on); s.querySelector("input.seen").checked = on;
    });
    document.getElementById("n-frames").textContent = DATA.frames.length;
    document.getElementById("n-person").textContent = Object.keys(state.person).length;
    document.getElementById("n-seen").textContent = Object.keys(state.seen).length;
    document.getElementById("n-grids").textContent = grids.length;
  }
  document.getElementById("export").addEventListener("click", function () {
    var out = {
      version: DATA.version, split_sha256: DATA.split_sha256, exported_at: new Date().toISOString(),
      grids: grids.length, grids_seen: Object.keys(state.seen).map(Number).sort(function (a, b) { return a - b; }),
      frames: DATA.frames.map(function (f) { return { frame: f.frame, person: !!state.person[f.frame] }; })
    };
    var blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = "neg-holdout-review-" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a); a.click(); a.remove();
  });
  document.getElementById("reset").addEventListener("click", function () {
    if (confirm("Xoá mọi đánh dấu trên trang này?")) { state = { person: {}, seen: {} }; save(); }
  });
  render();
})();
</script>
</body>
</html>
"""


def build_review_page(neg_doc: dict[str, Any], split_sha256: str, out_html: Path, frames_dir: Path) -> dict[str, Any]:
    """Trang duyệt mọi khung ``holdout``: lưới 4×4, ô "có người", Xuất JSON, trạng thái localStorage. Mở thẳng từ đĩa."""
    holdout = [{"frame": x["frame"], "session": x["session"]} for x in neg_doc["frames"] if x["split"] == "holdout"]
    try:
        prefix = Path(os.path.relpath(frames_dir, out_html.parent)).as_posix().rstrip("/") + "/"
    except ValueError:  # khác ổ đĩa (Windows): đường dẫn tuyệt đối
        prefix = frames_dir.resolve().as_uri().rstrip("/") + "/"
    data = {"version": REVIEW_EXPORT_VERSION, "storage_key": REVIEW_STORAGE_KEY, "split_sha256": split_sha256,
            "grid": GRID, "img_prefix": prefix, "frames": holdout}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(_REVIEW_HTML.replace("__DATA__", blob), encoding="utf-8", newline="\n")
    return {"frames": len(holdout), "grids": -(-len(holdout) // GRID), "html": str(out_html)}


def apply_review(neg_doc: dict[str, Any], export: dict[str, Any], split_sha256: str,
                 allow_unseen: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Áp file Xuất JSON: khung holdout ``person: false`` → ``approved: true``; ``person: true`` → loại khỏi neg.

    Lỗi (``ValueError``) khi: sai ``version``; ``split_sha256`` khác file hiện tại (trang dựng từ bản chia khác); thiếu
    hoặc thừa khung so với holdout; còn lưới chưa "Đã xem" (trừ khi ``allow_unseen``); holdout đã có khung duyệt rồi.
    """
    if export.get("version") != REVIEW_EXPORT_VERSION:
        raise ValueError(f"version {export.get('version')!r} ≠ {REVIEW_EXPORT_VERSION!r}")
    if export.get("split_sha256") != split_sha256:
        raise ValueError("trang duyệt dựng từ neg_flight_split.json khác bản hiện tại: dựng lại trang (`sessions review`)")
    holdout = [x for x in neg_doc["frames"] if x["split"] == "holdout"]
    if any(x["approved"] for x in holdout):
        raise ValueError("holdout đã có khung approved: true (đã áp một lần); chạy lại `sessions split` trước")
    marks: dict[str, bool] = {}
    for e in export.get("frames", []):
        if not isinstance(e.get("person"), bool) or e.get("frame") in marks:
            raise ValueError(f"dòng xuất sai dạng hoặc lặp: {e}")
        marks[e["frame"]] = e["person"]
    names = {x["frame"] for x in holdout}
    missing, extra = sorted(names - set(marks)), sorted(set(marks) - names)
    if missing or extra:
        raise ValueError(f"thiếu {len(missing)} / thừa {len(extra)} khung so với holdout: {(missing + extra)[:5]}")
    n_grids = -(-len(holdout) // GRID)
    unseen = sorted(set(range(n_grids)) - set(export.get("grids_seen", [])))
    if unseen and not allow_unseen:
        raise ValueError(f"còn {len(unseen)} lưới chưa đánh dấu 'Đã xem' (lưới {[g + 1 for g in unseen[:10]]})")
    frames = []
    for x in neg_doc["frames"]:
        if x["split"] != "holdout":
            frames.append(x)
        elif not marks[x["frame"]]:
            frames.append(x | {"approved": True})
    removed = sorted(f for f, p in marks.items() if p)
    approved = len(names) - len(removed)
    report = {"holdout_before": len(names), "removed_person": removed, "approved": approved, "unseen_grids": unseen,
              "kill_a": {"holdout_approved": approved, "min": KILL_A_MIN_HOLDOUT,
                         "verdict": "không đủ chứng minh" if approved < KILL_A_MIN_HOLDOUT else "đủ"}}
    return neg_doc | {"frames": frames}, report


def cmd_review(args: argparse.Namespace) -> int:
    root = repo_root()
    p = root / NEG_SPLIT_REL
    out = build_review_page(_load(p), sha256_file(p), root / REVIEW_DIR_REL / "index.html", root / F.FRAMES_REL)
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_apply_review(args: argparse.Namespace) -> int:
    root = repo_root()
    p = root / NEG_SPLIT_REL
    export_p = Path(args.export)
    if not export_p.is_file():
        print(f"LỖI: không có file {export_p}", file=sys.stderr)
        return 2
    new_doc, report = apply_review(_load(p), _load(export_p), sha256_file(p), allow_unseen=args.allow_unseen)
    errs = check_split(new_doc, flight_rows(root), load_split_v1(root))
    if errs:
        raise ValueError(f"neg_flight_split.json sau khi áp lệch hợp đồng: {errs[:3]}")
    if not args.dry_run:
        _dump(p, new_doc)
        _dump(root / REVIEW_APPLIED_REL, report | {"export": export_p.name,
                                                   "applied": datetime.now().isoformat(timespec="seconds")})
    print(json.dumps(report | {"dry_run": args.dry_run}, ensure_ascii=False) if args.json else report)
    return 0


# ---------------------------------------------------------------- holdout yếu "cùng buổi" (quyết định chủ dự án 26/11)

WEAK_KEY = "holdout_weak"
"""Khoá gốc của ``neg_flight_split.json`` chứa holdout yếu. Không nằm trong ``frames``: đường holdout chặt (``split`` =
``holdout``, ``approved``) và mọi bên đọc ``frames`` (system, train, verify, detection) không thấy nó."""
WEAK_KIND = "weak_same_day"
WEAK_LABEL = "holdout yếu: cùng buổi bay với test, không độc lập; agent duyệt"
WEAK_MIN_DIST_S = 30.0
"""Khung holdout yếu cách mọi khối ``dev``/``test`` v1 ít nhất 30 s (khoảng giờ thật của các khung trong khối)."""
WEAK_MIN_GAP_S = 0.5
WEAK_DIST_STEPS = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
WEAK_BLOCK_S = 30.0
"""Cụm bootstrap theo khối: ``<phiên>/<floor((t - t_start) / 30 s)>``."""
WEAK_DIR_REL = OUT_REL / "weak"
WEAK_PLAN_REL = WEAK_DIR_REL / "plan.json"
WEAK_REVIEW_REL = WEAK_DIR_REL / "agent_review.csv"
WEAK_SHEETS_INDEX_REL = WEAK_DIR_REL / "agent_review_sheets.csv"
WEAK_SCORE_REL = Path("results/v2/system/holdout_weak.json")
WEAK_FRAME_KEYS = {"frame", "session", "t_unix", "dist_s", "block", "approved", "approved_by"}


def v1_block_intervals(split_v1: dict[str, dict[str, Any]]) -> list[tuple[float, float]]:
    """``[(t đầu, t cuối)]`` của mỗi khối có khung ``dev``/``test`` v1 (theo ``t_unix`` của chính các khung đó)."""
    ts: dict[int, list[float]] = defaultdict(list)
    for v in split_v1.values():
        if v["split"] in ("dev", "test") and v.get("t_unix") is not None:
            ts[v["block"]].append(float(v["t_unix"]))
    return sorted((min(x), max(x)) for x in ts.values())


def dist_to_blocks(t: float, intervals: list[tuple[float, float]]) -> float:
    """Giây tới khối dev/test v1 gần nhất (0 nếu nằm trong khối); không có khối → vô cực."""
    return min((0.0 if a <= t <= b else min(abs(t - a), abs(t - b)) for a, b in intervals), default=float("inf"))


def _spaced(rows: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    out, last = [], {}
    for r in rows:
        t = float(r["t_unix"])
        if t - last.get(r["session"], -1e18) >= gap - 1e-9:
            out.append(r)
            last[r["session"]] = t
    return out


def plan_weak_holdout(rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]], neg_doc: dict[str, Any],
                      devpool: set[str], sessions: list[dict[str, Any]], min_dist_s: float = WEAK_MIN_DIST_S,
                      min_gap_s: float = WEAK_MIN_GAP_S) -> dict[str, Any]:
    """Ứng viên holdout yếu: khung bay (đã nhập, trong phiên) **không** thuộc ``split.json`` v1, không là mine/cal/
    devpool/khung giữ lại, cách mọi khối dev/test v1 ≥ ``min_dist_s``, cách nhau ≥ ``min_gap_s`` trong phiên. Thuần.

    ``dist_hist``: số khung còn lại (sau giãn cách) ở từng mức khoảng cách ``WEAK_DIST_STEPS`` (để báo cáo luật cứng tới
    đâu, không dùng để chọn)."""
    iv = v1_block_intervals(split_v1)
    start = {s["id"]: float(s["t_start"]) for s in sessions}
    taken = {x["frame"] for x in neg_doc["frames"]} | reserved_frames(neg_doc, rows, split_v1) | set(devpool)
    skipped: Counter[str] = Counter()
    pool = []
    for r in sorted(rows, key=lambda r: (r["session"], float(r["t_unix"]), r["frame"])):
        if not r["session"]:
            skipped["outside_session"] += 1
        elif r["frame"] in split_v1:
            skipped[f"v1_{split_v1[r['frame']]['split']}"] += 1
        elif r["frame"] in taken:
            skipped["neg_or_reserved"] += 1
        else:
            pool.append(r | {"dist_s": dist_to_blocks(float(r["t_unix"]), iv)})
    hist = {f"{d:g}": len(_spaced([r for r in pool if r["dist_s"] >= d], min_gap_s)) for d in WEAK_DIST_STEPS}
    hist_s = {f"{d:g}": dict(sorted(Counter(r["session"] for r in _spaced([r for r in pool if r["dist_s"] >= d],
                                                                           min_gap_s)).items()))
              for d in WEAK_DIST_STEPS}
    near = [r for r in pool if r["dist_s"] < min_dist_s]
    skipped["near_v1_block"] += len(near)
    far = [r for r in pool if r["dist_s"] >= min_dist_s]
    kept = _spaced(far, min_gap_s)
    skipped["spacing"] += len(far) - len(kept)
    cands = [{"frame": r["frame"], "session": r["session"], "t_unix": float(r["t_unix"]),
              "dist_s": round(r["dist_s"], 3),
              "block": f"{r['session']}/{int((float(r['t_unix']) - start[r['session']]) // WEAK_BLOCK_S)}"}
             for r in kept]
    return {"kind": WEAK_KIND, "rule": {"min_dist_s": min_dist_s, "min_gap_s": min_gap_s, "block_s": WEAK_BLOCK_S},
            "not_in_v1": len(pool), "candidates": cands, "by_session": dict(Counter(c["session"] for c in cands)),
            "dist_hist": hist, "dist_hist_by_session": hist_s, "skipped": dict(sorted(skipped.items()))}


def build_weak_section(plan: dict[str, Any], review: dict[str, dict[str, str]],
                       allow_unreviewed: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Khối ``holdout_weak`` của ``neg_flight_split.json`` + báo cáo. Chỉ khung agent soát ``no_person`` được giữ
    (``approved: true``, ``approved_by: "agent"``; KHÔNG phải chủ dự án duyệt). Ứng viên chưa soát → ``ValueError``."""
    cands = plan["candidates"]
    unreviewed = [c["frame"] for c in cands if c["frame"] not in review]
    if unreviewed and not allow_unreviewed:
        raise ValueError(f"{len(unreviewed)} khung holdout yếu chưa soát ({WEAK_REVIEW_REL}): {unreviewed[:3]}")
    kept = [c | {"approved": True, "approved_by": "agent"} for c in cands
            if review.get(c["frame"], {}).get("verdict") == "no_person"]
    dropped = [{"frame": c["frame"], "session": c["session"], "reason": review[c["frame"]].get("reason", "")}
               for c in cands if review.get(c["frame"], {}).get("verdict") == "person"]
    section = {"kind": WEAK_KIND, "label": WEAK_LABEL, "approved_by": "agent", "owner_approved": False,
               "rule": plan["rule"], "frames": sorted(kept, key=lambda x: x["frame"])}
    report = {"candidates": len(cands), "kept": len(kept), "dropped_person_or_doubt": dropped,
              "unreviewed": len(unreviewed),
              "per_session": {s: {"candidates": n, "kept": sum(k["session"] == s for k in kept)}
                              for s, n in sorted(plan["by_session"].items())}}
    return section, report


def check_weak_section(section: dict[str, Any], neg_doc: dict[str, Any], rows: list[dict[str, Any]],
                       split_v1: dict[str, dict[str, Any]], devpool: set[str]) -> list[str]:
    """Lỗi của khối ``holdout_weak`` (rỗng = đạt): nhãn, luật khoảng cách / giãn cách, không chạm v1/neg/devpool."""
    errs = []
    if section.get("kind") != WEAK_KIND or section.get("approved_by") != "agent" or section.get("owner_approved"):
        errs.append(f"kind/approved_by/owner_approved sai: {section.get('kind')!r} {section.get('approved_by')!r}")
    if section.get("label") != WEAK_LABEL:
        errs.append("thiếu nhãn holdout yếu")
    rule = section.get("rule", {})
    iv = v1_block_intervals(split_v1)
    sess_of = {r["frame"]: r["session"] for r in rows}
    taken = {x["frame"] for x in neg_doc["frames"]} | reserved_frames(neg_doc, rows, split_v1) | set(devpool)
    seen: set[str] = set()
    for x in section.get("frames", []):
        f = x.get("frame")
        if set(x) != WEAK_FRAME_KEYS or x.get("approved") is not True or x.get("approved_by") != "agent":
            errs.append(f"dòng holdout yếu sai dạng: {x}")
            continue
        if f in seen:
            errs.append(f"khung lặp {f}")
        seen.add(f)
        if f in split_v1:
            errs.append(f"{f}: có trong split.json v1 ({split_v1[f]['split']})")
        if f in taken:
            errs.append(f"{f}: là khung mine/cal/devpool/giữ lại")
        if sess_of.get(f) != x["session"] or not x["session"]:
            errs.append(f"{f}: session {x['session']!r} khác {sess_of.get(f)!r}")
        if dist_to_blocks(float(x["t_unix"]), iv) < float(rule.get("min_dist_s", WEAK_MIN_DIST_S)) - 1e-9:
            errs.append(f"{f}: gần khối dev/test v1 hơn {rule.get('min_dist_s')} s")
    by: dict[str, list[float]] = defaultdict(list)
    for x in section.get("frames", []):
        by[x.get("session", "")].append(float(x.get("t_unix", 0.0)))
    for s, ts in by.items():
        ts.sort()
        if any(b - a < float(rule.get("min_gap_s", WEAK_MIN_GAP_S)) - 1e-9 for a, b in itertools.pairwise(ts)):
            errs.append(f"phiên {s}: hai khung holdout yếu cách nhau < {rule.get('min_gap_s')} s")
    return errs


def weak_frames(neg_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Khung holdout yếu agent đã duyệt (rỗng khi chưa có khối)."""
    return [x for x in neg_doc.get(WEAK_KEY, {}).get("frames", []) if x.get("approved") is True]


def _cluster_ci(per: np.ndarray, groups: list[str], boot_n: int, rng: np.random.Generator) -> list[float | None]:
    keys = sorted(set(groups))
    if len(keys) < 2:
        return [None, None]
    idx = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    boots = []
    for _ in range(boot_n):
        ii = [i for k in rng.choice(keys, size=len(keys), replace=True) for i in idx[k]]
        boots.append(float(per[ii].mean()))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def weak_report(csvs: dict[str, str], frames: list[dict[str, Any]], boot_n: int = 2000,
                seed: int = SEED) -> dict[str, Any]:
    """B2 trên holdout yếu: báo nhầm/khung theo tầng, theo phiên; CI 95 % bootstrap theo phiên VÀ theo khối 30 s."""
    from xla.eval import system as Y

    base = {"kind": WEAK_KIND, "label": WEAK_LABEL, "approved_by": "agent", "owner_approved": False,
            "n_frames": len(frames), "n_sessions": len({x["session"] for x in frames}),
            "n_blocks": len({x["block"] for x in frames}), "target": Y.B2_MAX}
    if not frames:
        return base | {"verdict": "không đủ chứng minh (0 khung)", "by_stage": {s: None for s in Y.STAGES}}
    names = [x["frame"] for x in frames]
    sess = [x["session"] for x in frames]
    blocks = [x["block"] for x in frames]
    by_stage = {}
    for stage in Y.STAGES:
        rng = np.random.default_rng(seed)
        counts = Y.read_rows(Path(csvs[stage]))
        per = np.asarray([counts.get(f, 0) for f in names], float)
        by_stage[stage] = {
            "fp": int(per.sum()), "fp_per_frame": float(per.mean()),
            "ci95_session": _cluster_ci(per, sess, boot_n, rng), "ci95_block": _cluster_ci(per, blocks, boot_n, rng),
            "by_session": {s: {"n": sess.count(s), "fp": int(per[[i for i, g in enumerate(sess) if g == s]].sum()),
                               "fp_per_frame": float(per[[i for i, g in enumerate(sess) if g == s]].mean())}
                           for s in sorted(set(sess))}}
    fp = by_stage["track"]["fp_per_frame"]
    enough = len(frames) >= KILL_A_MIN_HOLDOUT
    return base | {"by_stage": by_stage, "verdict": ("đạt" if fp <= Y.B2_MAX else "không đạt")
                   + ("" if enough else f" (không đủ chứng minh: < {KILL_A_MIN_HOLDOUT} khung)") + " — " + WEAK_LABEL}


def score_weak(iotcv: Path, runs: Path, runner: Any = None, root: Path | None = None, boot_n: int = 2000,
               rec_frames: Any = None, out: Path | None = None) -> dict[str, Any]:
    """Chấm B2 trên holdout yếu bằng ĐÚNG đường holdout của ``xla.eval.system.score`` (``_replay_set`` ba tầng qua
    IOT-CV, ``--blank`` = test + cal + mine, meta chỉ đọc), không chấm lại test, không đụng ``score.json``. 0 khung →
    ghi "không đủ chứng minh", không gọi IOT-CV, không lấy khoá GPU. Có khung → giữ khoá D6 MỘT lần cho cả việc."""
    import contextlib

    from xla.eval import system as Y

    root = root or repo_root()
    runner = runner or Y.run_iotcv
    rec_frames = rec_frames or Y.recording_frames
    frames = weak_frames(load_neg_split(root))
    out = out or root / WEAK_SCORE_REL
    sets = Y.load_frame_sets(root)
    blank = Y.score_holdout_blank(sets)
    names = frozenset(x["frame"] for x in frames)
    Y.check_disjoint(names, blank | sets.devpool | sets.holdout, "holdout yếu")
    replays: dict[str, Any] = {}
    csvs: dict[str, str] = {}
    if frames:
        log = runs / "score_weak.log"
        det_onnx = Y.install_bundle(iotcv, root / Y.RELEASE_REL / Y.DETECTOR_SYSTEM, runner, log)
        ver_onnx = Y.install_bundle(iotcv, root / Y.RELEASE_REL / Y.VERIFIER_SYSTEM, runner, log)
        with Y.held_gpu("v2 holdout yếu score") if runner is Y.run_iotcv else contextlib.nullcontext():
            hold = Y._replay_set("holdout_weak", names, blank, iotcv, runs, det_onnx, ver_onnx, runner, log,
                                 runs / "preds", rec_frames)
        replays, csvs = hold["replays"], hold["csv"]
    det_meta = Y._meta(root, Y.DETECTOR_SYSTEM)
    ver_meta = Y._meta(root, Y.VERIFIER_SYSTEM)
    doc = {"created": Y.utc_now(), "what": "B2 trên holdout yếu (cùng buổi), không phải holdout chặt của score.json",
           "bundles": {"detector": Y.DETECTOR_SYSTEM, "detector_onnx_sha256": det_meta["onnx_sha256"],
                       "recommended_track": det_meta.get("recommended_track"), "verifier": Y.VERIFIER_SYSTEM,
                       "verifier_conf_threshold": ver_meta["conf_threshold"]},
           "blank": {"rule": "test + cal + mine", "n": len(blank)}, "rule": load_neg_split(root).get(WEAK_KEY, {}).get(
               "rule"), "replays": replays} | weak_report(csvs, frames, boot_n)
    Y.write_json(out, doc)
    return doc


def cmd_weak_holdout(args: argparse.Namespace) -> int:
    from xla.paths import cache_dir

    root = repo_root()
    rows = flight_rows(root)
    split_v1 = load_split_v1(root)
    p = root / NEG_SPLIT_REL
    neg_doc = _load(p)
    devpool_p = root / DEVPOOL_JSON_REL
    devpool = set(_load(devpool_p)["frames"]) if devpool_p.exists() else set()
    plan = plan_weak_holdout(rows, split_v1, neg_doc, devpool, load_sessions(root), args.min_dist_s, args.min_gap_s)
    names = [c["frame"] for c in plan["candidates"]]
    raw_p = root / PRELABEL_RAW_REL
    raw = _load(raw_p)["frames"] if raw_p.exists() else {}
    plan["prelabel_covered"] = sum(f in raw for f in names)
    if args.prelabel and plan["prelabel_covered"] < len(names):
        plan["prelabel_run"] = run_prelabel(names, raw_p, lock_timeout=args.lock_timeout, label="v2 holdout yếu prelabel")
    if args.sheets and names:
        out_dir = Path(args.out) if args.out else cache_dir() / "xla-neg-review-weak"
        index = neg_sheets([c | {"split": "weak"} for c in plan["candidates"]], out_dir)
        (root / WEAK_SHEETS_INDEX_REL).parent.mkdir(parents=True, exist_ok=True)
        with (root / WEAK_SHEETS_INDEX_REL).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=("sheet", "cell", "frame", "session", "split"), lineterminator="\n",
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(index)
        plan["sheets_dir"] = str(out_dir)
    _dump(root / WEAK_PLAN_REL, plan)
    review = read_agent_review(root / WEAK_REVIEW_REL)
    try:
        section, report = build_weak_section(plan, review, allow_unreviewed=False)
    except ValueError as e:
        print(json.dumps({k: plan[k] for k in ("not_in_v1", "by_session", "dist_hist")} | {"unreviewed": str(e)},
                         ensure_ascii=False))
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    errs = check_weak_section(section, neg_doc, rows, split_v1, devpool)
    if errs:
        raise ValueError(f"holdout yếu lệch luật: {errs[:3]}")
    if not args.dry_run:
        _dump(p, neg_doc | {WEAK_KEY: section})
    out = {"rule": plan["rule"], "not_in_v1": plan["not_in_v1"], "dist_hist": plan["dist_hist"]} | report | {
        "dry_run": args.dry_run, "label": WEAK_LABEL}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_score_weak(args: argparse.Namespace) -> int:
    from xla.eval.system import RUNS_DEFAULT

    runs = Path(args.runs) if args.runs else RUNS_DEFAULT.parent / "holdout-weak"
    doc = score_weak(Path(args.iotcv), runs, boot_n=args.boot_n)
    out = {k: doc[k] for k in ("n_frames", "n_sessions", "n_blocks", "verdict", "label")}
    out["fp_per_frame"] = {s: (v or {}).get("fp_per_frame") for s, v in doc["by_stage"].items()}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m xla.camera.sessions", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index", help="phiên bay từ backend.flights --events -> data/real/sessions.json")
    i.add_argument("--events", help="đọc đầu ra --events từ file thay vì gọi IOT-CV")
    i.add_argument("--json", action="store_true")
    p = sub.add_parser("prelabel", help="nhãn trước ngưỡng thấp trên khung neg ứng viên (GPU, gpu_lock)")
    p.add_argument("--conf", type=float, default=PRELABEL_CONF)
    p.add_argument("--extra-frames", help="file danh sách khung thêm (vd. devpool phase 04)")
    p.add_argument("--max-frames", type=int, help="chỉ chạy tối đa N khung mỗi model lần này (chạy tiếp sau)")
    p.add_argument("--lock-timeout", type=float, default=6 * 3600.0)
    p.add_argument("--json", action="store_true")
    c = sub.add_parser("sheets", help="ảnh ghép 4×4 khung neg dự kiến để agent soát từng khung")
    c.add_argument("--only-unreviewed", action="store_true")
    c.add_argument("--out", help="thư mục ảnh (mặc định cache ngoài repo)")
    c.add_argument("--json", action="store_true")
    s = sub.add_parser("split", help="chia mine/cal/holdout -> data/real/neg_flight_split.json")
    s.add_argument("--allow-unreviewed", action="store_true", help="ghi cả khi còn khung chưa soát (chỉ để lập kế hoạch)")
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("review", help="trang duyệt holdout cho chủ dự án")
    r.add_argument("--json", action="store_true")
    a = sub.add_parser("apply-review", help="áp file Xuất JSON của trang duyệt")
    a.add_argument("export", help="file JSON chủ dự án xuất từ trang duyệt")
    a.add_argument("--allow-unseen", action="store_true", help="chấp nhận lưới chưa đánh dấu Đã xem")
    a.add_argument("--dry-run", action="store_true", help="chỉ kiểm, không ghi")
    a.add_argument("--json", action="store_true")
    w = sub.add_parser("weak-holdout", help="holdout yếu cùng buổi (agent duyệt) -> khối holdout_weak")
    w.add_argument("--min-dist-s", type=float, default=WEAK_MIN_DIST_S, help="giây tối thiểu tới khối dev/test v1")
    w.add_argument("--min-gap-s", type=float, default=WEAK_MIN_GAP_S)
    w.add_argument("--prelabel", action="store_true", help="nhãn trước cho ứng viên chưa có (GPU, gpu_lock)")
    w.add_argument("--lock-timeout", type=float, default=6 * 3600.0)
    w.add_argument("--sheets", action="store_true", help="ảnh ghép 4×4 ứng viên để agent soát")
    w.add_argument("--out", help="thư mục ảnh ghép (mặc định cache ngoài repo)")
    w.add_argument("--dry-run", action="store_true", help="không ghi neg_flight_split.json")
    w.add_argument("--json", action="store_true")
    g = sub.add_parser("score-weak", help="B2 trên holdout yếu qua IOT-CV -> results/v2/system/holdout_weak.json")
    g.add_argument("--iotcv", required=True)
    g.add_argument("--runs", help="thư mục cache/CSV (mặc định D:/DevCache/runs/xla/v2/holdout-weak)")
    g.add_argument("--boot-n", type=int, default=2000)
    g.add_argument("--json", action="store_true")
    return ap


COMMANDS = {"index": "cmd_index", "prelabel": "cmd_prelabel", "sheets": "cmd_sheets", "split": "cmd_split",
            "review": "cmd_review", "apply-review": "cmd_apply_review", "weak-holdout": "cmd_weak_holdout",
            "score-weak": "cmd_score_weak"}


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return globals()[COMMANDS[args.cmd]](args)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
