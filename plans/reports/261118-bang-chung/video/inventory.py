"""Kiểm kê 11 lần ghi hình: thời lượng, số khung, fps, khoảng hở, cỡ JPEG, quality, khung lúc bay."""
import csv, json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

REPO = Path(r"d:\Coding\IOT-CV")
sys.path.insert(0, str(REPO))
from backend.flights.video import Recording  # noqa: E402

OUT = Path(__file__).parent
TZ = timezone(timedelta(hours=7))
rows_out = []
all_frames = []
for d in sorted((REPO / "logs/recordings").iterdir()):
    rec = Recording(d)
    with (d / "frames.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        rows_out.append({"rec": d.name, "frames": 0}); continue
    t = np.array([float(r["unix_time"]) for r in rows])
    dt = np.diff(t)
    sizes = []
    for i, (seg, vf) in enumerate(rec.rows):
        ch = rec._index(seg)
        sizes.append(ch[vf][1] if vf < len(ch) else 0)
    sizes = np.array(sizes)
    fid = np.array([int(r["frame_id"]) for r in rows if r.get("frame_id")])
    alt = np.array([float(r["rel_alt_m"]) if r.get("rel_alt_m") else np.nan for r in rows])
    armed = np.array([r.get("armed") == "True" for r in rows])
    q = [r.get("jpeg_quality") for r in rows if r.get("jpeg_quality")]
    flying = armed & (alt > 1.0)
    dur = t[-1] - t[0]
    # fps lúc bay
    fly_fps = None
    if flying.sum() > 1:
        tf = t[flying]
        # thời gian bay = tổng các đoạn liên tiếp
        span = sum(b - a for a, b in zip(tf[:-1], tf[1:]) if b - a < 30)
        fly_fps = flying.sum() / span if span > 0 else None
    fid_gap = int(np.sum(np.clip(np.diff(fid) - 1, 0, None))) if len(fid) > 1 else 0
    res = sorted({(r["width"], r["height"]) for r in rows})
    rows_out.append({
        "rec": d.name,
        "start_local": datetime.fromtimestamp(t[0], TZ).strftime("%d/%m %H:%M:%S"),
        "end_local": datetime.fromtimestamp(t[-1], TZ).strftime("%H:%M:%S"),
        "dur_s": round(dur, 1), "frames": len(rows), "segments": len({r["segment"] for r in rows}),
        "res": res, "fps_mean": round(len(rows) / dur, 2) if dur else None,
        "dt_median": round(float(np.median(dt)), 3) if len(dt) else None,
        "gaps_gt1s": int((dt > 1).sum()), "gaps_gt5s": int((dt > 5).sum()),
        "gap_total_s": round(float(dt[dt > 1].sum()), 1), "max_gap_s": round(float(dt.max()), 1) if len(dt) else None,
        "frame_id_skipped": fid_gap,
        "jpeg_kb_mean": round(sizes[sizes > 0].mean() / 1024, 1) if (sizes > 0).any() else None,
        "jpeg_kb_p10_p90": [round(float(np.percentile(sizes, p)) / 1024, 1) for p in (10, 90)],
        "zero_or_missing": int((sizes == 0).sum()),
        "quality_values": sorted(set(q))[:10] if q else "không có cột/trống",
        "alt_filled": int(np.isfinite(alt).sum()), "armed_frames": int(armed.sum()),
        "flying_frames": int(flying.sum()), "fly_fps": round(fly_fps, 2) if fly_fps else None,
        "alt_max": round(float(np.nanmax(alt)), 1) if np.isfinite(alt).any() else None,
    })
    for i, r in enumerate(rows):
        all_frames.append((d.name, i, t[i], alt[i], bool(armed[i]), int(sizes[i]), r.get("mode", "")))
(OUT / "inventory.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=1), encoding="utf-8")
with (OUT / "all_frames.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["rec", "n", "t", "alt", "armed", "size", "mode"]); w.writerows(all_frames)
for r in rows_out:
    print(json.dumps(r, ensure_ascii=False))
