"""Nhập khung bay thật từ bản ghi IOT-CV, giữ nguyên byte JPEG camera đã gửi (phase 03).

Nguồn: ``<iotcv_root>/logs/recordings/<rec>/`` (``video-NNN.avi`` MJPEG + ``frames.csv``). Khung được ghép với dòng
``frames.csv`` theo khoá (``segment``, ``video_frame``) qua ``Recording`` của IOT-CV (import, không chép mã): chunk AVI
không có dòng CSV là khung lấp chỗ mất hình, không có độ cao, bị loại và đếm.

Lệnh::

    python -m xla.camera.frames extract --json   # khung bay + mẫu âm -> data/real/frames/, manifest.csv
    python -m xla.camera.frames prelim --json    # YOLO11s COCO @1280 đếm người sơ bộ (GPU, trong gpu_lock)

``extract`` exit 1 khi số khung bay cuối lệch quá ``FLIGHT_FRAMES_TOL`` so với ``FLIGHT_FRAMES_EXPECTED``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from xla.paths import iotcv_root, repo_root

# ---- Hằng số của bộ dữ liệu (plan phase 03, Deferred 14/15) ----
REC_FLIGHT = "20261102-100941"
"""Lần ghi duy nhất có cảnh bay (6 chuyến sáng 02/11/2026)."""
NEG_RECS = ("20261102-033719", "20261102-040317", "20261102-042928", "20261102-043054")
"""Bốn lần ghi trong nhà 02/11 03:37–05:35: mẫu âm (không người)."""
LOCAL_TZ = timezone(timedelta(hours=7))
"""Giờ của id chuyến trong đầu ra ``backend.flights --events`` là giờ Việt Nam."""
ALT_MIN_M = 3.0
SPIKE_WINDOW_S = 5.0
"""Cửa sổ trung vị trượt (căn giữa) để bắt điểm vọt ``rel_alt_m``."""
SPIKE_DEV_M = 5.0
FLIGHT_FRAMES_EXPECTED = 2944
FLIGHT_FRAMES_TOL = 0.10
NEG_PERIOD_S = 2.0
NEG_MAX = 300
CROP_LEFT_PX = 16
"""Dải mép trái (đo ~13 px): hộp đếm sơ bộ chạm vào cột ``[0, 16)`` bị bỏ."""

# ---- Đếm người sơ bộ (chỉ để phân tầng, không phải nhãn) ----
PRELIM_WEIGHTS = "yolo-weights/yolo11s.pt"
"""Tương đối ``xla.paths.downloads_dir()``."""
PRELIM_IMGSZ = 1280
PRELIM_CONF = 0.25
PRELIM_CLASS_PERSON = 0

EVENTS_REL = Path("results/data/flights_events.txt")
FRAMES_REL = Path("data/real/frames")
MANIFEST_REL = Path("data/real/manifest.csv")
EXTRACT_REPORT_REL = Path("results/data/extract_report.json")
PRELIM_REL = Path("results/data/prelim_person_boxes.csv")
MANIFEST_FIELDS = ("frame", "kind", "rec", "segment", "video_frame", "t_unix", "alt_m", "flight_id", "bytes", "sha256")

_EVENT_LINE = re.compile(r"^(?P<id>\d{8}-\d{6})\s+(?P<dur>\d+)\s+(?P<fly>có|không)\s.*\[(?P<src>[^\]]+)\]\s*$")


@dataclass(frozen=True)
class FlightWindow:
    """Cửa sổ một chuyến: ``[t0, t1]`` giây Unix, lấy từ đầu ra ``backend.flights --events``."""

    flight_id: str
    t0: float
    t1: float


def frame_name(rec: str, segment: int, video_frame: int) -> str:
    """Tên file khung theo hợp đồng §7: ``<rec>_<segment>_<video_frame>.jpg`` (đệm số 0 để sắp được)."""
    return f"{rec}_{segment:03d}_{video_frame:05d}.jpg"


def parse_frame_name(name: str) -> tuple[str, int, int]:
    """Ngược của ``frame_name``."""
    stem = name.removesuffix(".jpg")
    rec, seg, vf = stem.rsplit("_", 2)
    return rec, int(seg), int(vf)


def parse_flight_events(text: str) -> list[FlightWindow]:
    """Chuyến "bay: có" và có video từ đầu ra ``python -m backend.flights --events`` (bảng ``id dài(s) bay ...``)."""
    out = []
    for line in text.splitlines():
        m = _EVENT_LINE.match(line.rstrip())
        if not m or m["fly"] != "có" or "video" not in m["src"]:
            continue
        t0 = datetime.strptime(m["id"], "%Y%m%d-%H%M%S").replace(tzinfo=LOCAL_TZ).timestamp()
        out.append(FlightWindow(m["id"], t0, t0 + float(m["dur"])))
    return out


def _import_iotcv_video() -> Any:
    """Import ``backend.flights.video`` của IOT-CV (chỉ đọc; không ghi ``__pycache__`` vào repo đó)."""
    root = str(iotcv_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from backend.flights import video  # type: ignore[import-not-found]
    finally:
        sys.dont_write_bytecode = old
    return video


def recordings_dir() -> Path:
    return iotcv_root() / "logs" / "recordings"


def read_rows(rec_dir: Path) -> list[dict[str, str]]:
    with (rec_dir / "frames.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_nan(s: str | None) -> float:
    try:
        return float(s) if s not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def centered_rolling_median(t: np.ndarray, v: np.ndarray, window_s: float) -> np.ndarray:
    """Trung vị của ``v`` trong ``[t_i - w/2, t_i + w/2]`` cho mỗi i (``t`` tăng dần, ``v`` hữu hạn)."""
    lo = np.searchsorted(t, t - window_s / 2, side="left")
    hi = np.searchsorted(t, t + window_s / 2, side="right")
    return np.array([np.median(v[a:b]) for a, b in zip(lo, hi, strict=True)])


def spike_mask(t: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """True ở điểm vọt: lệch trung vị trượt ``SPIKE_WINDOW_S`` quá ``SPIKE_DEV_M``. NaN không phải điểm vọt."""
    out = np.zeros(len(t), bool)
    ok = np.isfinite(alt)
    if ok.any():
        order = np.argsort(t[ok], kind="stable")
        tt, aa = t[ok][order], alt[ok][order]
        med = centered_rolling_median(tt, aa, SPIKE_WINDOW_S)
        bad = np.abs(aa - med) > SPIKE_DEV_M
        idx = np.flatnonzero(ok)[order]
        out[idx] = bad
    return out


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decodes(data: bytes) -> bool:
    import cv2

    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return im is not None and im.shape[:2] == (480, 640)


def _iter_rows_with_jpeg(rec_dir: Path) -> Iterator[tuple[dict[str, str], int, int, bytes | None]]:
    video = _import_iotcv_video()
    recording = video.Recording(rec_dir)
    rows = [r for r in read_rows(rec_dir) if r.get("segment") and r.get("video_frame")]
    if len(rows) != len(recording.rows):
        raise RuntimeError(f"{rec_dir.name}: số dòng hợp lệ {len(rows)} khác Recording.rows {len(recording.rows)}")
    for n, r in enumerate(rows):
        seg, vf = recording.rows[n]
        if (seg, vf) != (int(r["segment"]), int(r["video_frame"])):
            raise RuntimeError(f"{rec_dir.name}: dòng {n} lệch khoá ({seg},{vf}) với CSV")
        yield r, seg, vf, recording.frame(n)


def count_avi_chunks(rec_dir: Path) -> int:
    video = _import_iotcv_video()
    return sum(len(video.avi_chunk_index(p)) for p in sorted(rec_dir.glob("video-*.avi")))


def _write_frame(out_dir: Path, name: str, data: bytes) -> None:
    p = out_dir / name
    if p.exists() and p.read_bytes() == data:
        return
    p.write_bytes(data)


def extract_flight(out_dir: Path, events_text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Khung bay của ``REC_FLIGHT`` sau các bước lọc; trả (dòng manifest, số đếm từng bước)."""
    flights = parse_flight_events(events_text)
    if not flights:
        raise RuntimeError("không đọc được chuyến bay nào từ flights_events.txt")
    rec_dir = recordings_dir() / REC_FLIGHT
    counts: dict[str, int] = {"flights": len(flights), "avi_chunks_total": count_avi_chunks(rec_dir)}
    recs = list(_iter_rows_with_jpeg(rec_dir))
    counts["csv_rows"] = len(recs)
    counts["avi_chunks_without_csv_row"] = counts["avi_chunks_total"] - len(recs)
    t = np.array([float(r["unix_time"]) for r, *_ in recs])
    alt = np.array([_float_or_nan(r.get("rel_alt_m")) for r, *_ in recs])
    spike = spike_mask(t, alt)
    has_jpeg = np.array([d is not None for *_, d in recs])
    counts["csv_rows_with_jpeg"] = int(has_jpeg.sum())
    fid = np.array([next((f.flight_id for f in flights if f.t0 <= ti <= f.t1), "") for ti in t])
    in_win = has_jpeg & (fid != "")
    counts["in_flight_window"] = int(in_win.sum())
    alt_ok = in_win & np.isfinite(alt) & (alt > ALT_MIN_M)
    counts["alt_gt_3m"] = int(alt_ok.sum())
    keep = alt_ok & ~spike
    counts["not_spike"] = int(keep.sum())
    counts["spikes_removed"] = int((alt_ok & spike).sum())
    out: list[dict[str, Any]] = []
    undecodable = 0
    for i in np.flatnonzero(keep):
        _row, seg, vf, data = recs[i]
        assert data is not None
        if not _decodes(data):
            undecodable += 1
            continue
        name = frame_name(REC_FLIGHT, seg, vf)
        _write_frame(out_dir, name, data)
        out.append({"frame": name, "kind": "flight", "rec": REC_FLIGHT, "segment": seg, "video_frame": vf,
                    "t_unix": round(float(t[i]), 3), "alt_m": round(float(alt[i]), 3), "flight_id": str(fid[i]),
                    "bytes": len(data), "sha256": sha256_hex(data)})
    counts["undecodable"] = undecodable
    counts["final"] = len(out)
    return out, counts


def extract_neg(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Mẫu âm trong nhà: 1 khung / ``NEG_PERIOD_S`` giây mỗi lần ghi, rồi rải đều xuống tối đa ``NEG_MAX``."""
    cands = []
    for rec in NEG_RECS:
        last = -1e18
        for r, seg, vf, data in _iter_rows_with_jpeg(recordings_dir() / rec):
            ti = float(r["unix_time"])
            if data is None or ti - last < NEG_PERIOD_S:
                continue
            last = ti
            cands.append((rec, seg, vf, ti, _float_or_nan(r.get("rel_alt_m")), data))
    counts = {"candidates_1_per_2s": len(cands)}
    if len(cands) > NEG_MAX:
        idx = np.unique(np.round(np.linspace(0, len(cands) - 1, NEG_MAX)).astype(int))
        cands = [cands[i] for i in idx]
    out = []
    for rec, seg, vf, ti, a, data in cands:
        if not _decodes(data):
            continue
        name = frame_name(rec, seg, vf)
        _write_frame(out_dir, name, data)
        out.append({"frame": name, "kind": "neg", "rec": rec, "segment": seg, "video_frame": vf,
                    "t_unix": round(ti, 3), "alt_m": round(a, 3) if np.isfinite(a) else "", "flight_id": "",
                    "bytes": len(data), "sha256": sha256_hex(data)})
    counts["final"] = len(out)
    return out, counts


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["kind"], r["t_unix"])))


def read_manifest(path: Path | None = None) -> list[dict[str, str]]:
    p = path or repo_root() / MANIFEST_REL
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cmd_extract(args: argparse.Namespace) -> int:
    root = repo_root()
    events = (root / EVENTS_REL).read_text(encoding="utf-8")
    out_dir = root / FRAMES_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    fly, fly_counts = extract_flight(out_dir, events)
    neg, neg_counts = extract_neg(out_dir)
    names = {r["frame"] for r in fly + neg}
    stale = sorted(p.name for p in out_dir.glob("*.jpg") if p.name not in names)
    for n in stale:
        (out_dir / n).unlink()
    write_manifest(root / MANIFEST_REL, fly + neg)
    lo, hi = FLIGHT_FRAMES_EXPECTED * (1 - FLIGHT_FRAMES_TOL), FLIGHT_FRAMES_EXPECTED * (1 + FLIGHT_FRAMES_TOL)
    ok = lo <= fly_counts["final"] <= hi
    total_bytes = sum(int(r["bytes"]) for r in fly + neg)
    report = {
        "recording": REC_FLIGHT,
        "flight": fly_counts,
        "neg": neg_counts,
        "expected_flight_frames": FLIGHT_FRAMES_EXPECTED,
        "expected_range": [round(lo), round(hi)],
        "within_expected": ok,
        "stale_removed": len(stale),
        "frames_total_bytes": total_bytes,
        "per_flight": {f: sum(1 for r in fly if r["flight_id"] == f) for f in sorted({r["flight_id"] for r in fly})},
    }
    (root / EXTRACT_REPORT_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / EXTRACT_REPORT_REL).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        for k, v in fly_counts.items():
            print(f"bay {k}: {v}")
        print(f"âm: {neg_counts}; tổng {total_bytes / 1e6:.1f} MB")
    if not ok:
        print(f"LỖI: {fly_counts['final']} khung bay ngoài dải {round(lo)}–{round(hi)}", file=sys.stderr)
    return 0 if ok else 1


def cmd_prelim(args: argparse.Namespace) -> int:
    import cv2
    from ultralytics import YOLO

    from xla.gpu_lock import gpu_lock
    from xla.paths import downloads_dir

    root = repo_root()
    rows = read_manifest()
    weights = downloads_dir() / PRELIM_WEIGHTS
    out_rows = []
    dropped_edge = 0
    with gpu_lock("phase03 prelim yolo11s@1280", timeout_s=args.lock_timeout):
        model = YOLO(str(weights))
        for r in rows:
            im = cv2.imread(str(root / FRAMES_REL / r["frame"]), cv2.IMREAD_COLOR)
            res = model.predict(im, imgsz=PRELIM_IMGSZ, conf=PRELIM_CONF, classes=[PRELIM_CLASS_PERSON], device=0,
                                verbose=False)[0]
            for xyxy, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), strict=True):
                x1, y1, x2, y2 = xyxy
                if x1 < CROP_LEFT_PX:
                    dropped_edge += 1
                    continue
                out_rows.append((r["frame"], round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(conf, 4)))
    p = root / PRELIM_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(out_rows)
    kinds = {r["frame"]: r["kind"] for r in rows}
    with_person = {k: len({f for f, *_ in out_rows if kinds[f] == k}) for k in ("flight", "neg")}
    summary = {"frames": len(rows), "boxes": len(out_rows), "dropped_left_strip": dropped_edge,
               "frames_with_person": with_person, "weights": weights.name, "imgsz": PRELIM_IMGSZ, "conf": PRELIM_CONF}
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.camera.frames", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="chép khung bay + mẫu âm, ghi manifest")
    e.add_argument("--json", action="store_true")
    p = sub.add_parser("prelim", help="đếm người sơ bộ YOLO11s @1280 (GPU)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--lock-timeout", type=float, default=3600.0, help="giây chờ khoá GPU")
    args = ap.parse_args(argv)
    return {"extract": cmd_extract, "prelim": cmd_prelim}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
