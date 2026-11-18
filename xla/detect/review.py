"""Agent soát nhãn gán trước (phase 08 bước 3): dựng ảnh soát, áp các sửa, ghi nhãn đã soát.

Ảnh soát: khung phóng ``SCALE``× (mặc định 2), lưới toạ độ khung gốc (vạch mỗi 32 px, số mỗi 64 px), hộp đánh số
theo ``prelabel.json``; màu theo nguồn: xanh lá ``both``, cam chỉ RT-DETR, xanh dương chỉ YOLO11x. Ảnh sau sửa (``--edits``)
vẽ hộp cuối màu hồng.

File sửa (CSV, mỗi người soát một file, ``results/labels/review_parts/*.csv``) cột
``frame,action,box_id,x1,y1,x2,y2,reason``; ``action``:

- ``ok``      khung đã xem, không đổi gì (bắt buộc có ít nhất một dòng mỗi khung để chứng minh đã xem);
- ``drop``    bỏ hộp ``box_id`` (lý do: đốm hồng/trắng, dải mép, bóng, vật không phải người...);
- ``tighten`` thay hộp ``box_id`` bằng ``x1,y1,x2,y2`` (px khung gốc 640×480);
- ``add``     thêm hộp ``x1,y1,x2,y2`` (người sót);
- ``hard``    cờ khung khó (nhóm người dính nhau / người mờ).

Dùng::

    python -m xla.detect.review render --frames F1.jpg F2.jpg [--edits part.csv] [--out DIR]
    python -m xla.detect.review crop F.jpg x1 y1 x2 y2 [--scale 6]     # phóng một vùng để xem kỹ
    python -m xla.detect.review apply --json      # gộp review_parts → review_log.csv + nhãn dev-val + nhãn test tạm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from xla.detect.devval import LABELS_REL, load_manifest, save_manifest
from xla.detect.prelabel import EDGE_PX, FRAMES_REL, MODELS, OUT_REL, H, W, in_left_strip
from xla.paths import cache_dir, repo_root

SCALE = 2
PRELABEL_JSON = OUT_REL / "prelabel.json"
PARTS_REL = Path("results/labels/review_parts")
LOG_REL = Path("results/labels/review_log.csv")
AGENT_TEST_REL = Path("results/labels/agent_reviewed_test")
REVIEWED_JSON_REL = Path("results/labels/agent_reviewed.json")
ACTIONS = ("ok", "drop", "tighten", "add", "hard")
COLORS = {"both": (60, 200, 60), "rtdetr": (0, 140, 255), "yolo": (255, 120, 0), "final": (180, 60, 255)}
MIN_BOX_PX = 2.0


def review_dir() -> Path:
    return cache_dir() / "xla-labels-review"


def load_prelabel(root: Path | None = None) -> dict[str, Any]:
    return json.loads(((root or repo_root()) / PRELABEL_JSON).read_text(encoding="utf-8"))


def _grid(img: Any, s: int) -> None:
    import cv2

    for x in range(0, W + 1, 32):
        major = x % 64 == 0
        cv2.line(img, (x * s, 0), (x * s, 10 if major else 5), (255, 255, 0), 1)
        cv2.line(img, (x * s, H * s - (10 if major else 5)), (x * s, H * s), (255, 255, 0), 1)
        if major:
            cv2.putText(img, str(x), (x * s + 2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, H + 1, 32):
        major = y % 64 == 0
        cv2.line(img, (0, y * s), (10 if major else 5, y * s), (255, 255, 0), 1)
        cv2.line(img, (W * s - (10 if major else 5), y * s), (W * s, y * s), (255, 255, 0), 1)
        if major:
            cv2.putText(img, str(y), (12, y * s + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    # dải mép trái
    cv2.line(img, (EDGE_PX * s, 0), (EDGE_PX * s, H * s), (90, 90, 90), 1)


def _box(img: Any, b: list[float], color: tuple[int, int, int], label: str, s: int) -> None:
    import cv2

    p1 = (round(b[0] * s), round(b[1] * s))
    p2 = (round(b[2] * s), round(b[3] * s))
    cv2.rectangle(img, p1, p2, color, 1)
    ty = p1[1] - 4 if p1[1] > 14 else p2[1] + 13
    cv2.putText(img, label, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def read_edits(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for p in paths:
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                r = {k: (v or "").strip() for k, v in r.items()}
                if r["action"] not in ACTIONS:
                    raise ValueError(f"{p}: action lạ {r['action']!r} ở khung {r['frame']}")
                out[r["frame"]].append(r)
    return out


def _coords(r: dict[str, str]) -> list[float]:
    b = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    b = [min(max(b[0], 0), W), min(max(b[1], 0), H), min(max(b[2], 0), W), min(max(b[3], 0), H)]
    if b[2] - b[0] < MIN_BOX_PX or b[3] - b[1] < MIN_BOX_PX:
        raise ValueError(f"{r['frame']}: hộp quá nhỏ hoặc ngược {b}")
    return b


def apply_frame(boxes: list[dict[str, Any]], edits: list[dict[str, str]]) -> tuple[list[list[float]], bool]:
    """Hộp gán trước + sửa → (hộp cuối px gốc, hard)."""
    by_id = {b["id"]: [b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes}
    hard = False
    added: list[list[float]] = []
    for e in edits:
        a = e["action"]
        if a in ("drop", "tighten"):
            bid = int(e["box_id"])
            if bid not in by_id:
                raise ValueError(f"{e['frame']}: box_id {bid} không có (đã bỏ hai lần?)")
            if a == "drop":
                del by_id[bid]
            else:
                by_id[bid] = _coords(e)
        elif a == "add":
            added.append(_coords(e))
        elif a == "hard":
            hard = True
    final = [by_id[k] for k in sorted(by_id)] + added
    return [b for b in final if not in_left_strip(b)], hard


def render(frame: str, boxes: list[dict[str, Any]], out: Path, s: int = SCALE,
           final: list[list[float]] | None = None, title: str = "") -> Path:
    import cv2

    img = cv2.imread(str(repo_root() / FRAMES_REL / frame), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (W * s, H * s), interpolation=cv2.INTER_CUBIC)
    _grid(img, s)
    if final is None:
        for b in boxes:
            src = b["source"]
            conf = max((b.get(f"conf_{m['key']}") or 0.0) for m in MODELS)
            tag = {"both": "B", "rtdetr": "R", "yolo": "Y"}[src]
            _box(img, [b["x1"], b["y1"], b["x2"], b["y2"]], COLORS[src], f"{b['id']}{tag}{conf:.2f}", s)
    else:
        for i, b in enumerate(final):
            _box(img, b, COLORS["final"], str(i + 1), s)
    pad = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(pad, title or frame, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), pad, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def cmd_render(args: argparse.Namespace) -> int:
    pre = load_prelabel()["frames"]
    split_alt = {f["frame"]: f.get("alt_m") for f in
                 json.loads((repo_root() / "data/real/split.json").read_text(encoding="utf-8"))["frames"]}
    edits = read_edits([Path(p) for p in args.edits]) if args.edits else {}
    out_dir = Path(args.out) if args.out else review_dir()
    paths = []
    for f in args.frames:
        v = pre[f]
        title = f"{f}  [{v['split']}]  alt={split_alt.get(f)} m  boxes={len(v['boxes'])}"
        if args.edits:
            final, hard = apply_frame(v["boxes"], edits.get(f, []))
            p = render(f, v["boxes"], out_dir / f"{Path(f).stem}_after.jpg", args.scale, final,
                       title + f"  -> {len(final)}{' HARD' if hard else ''}")
        else:
            p = render(f, v["boxes"], out_dir / f"{Path(f).stem}.jpg", args.scale, title=title)
        paths.append(str(p))
    print("\n".join(paths))
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    import cv2

    img = cv2.imread(str(repo_root() / FRAMES_REL / args.frame), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = (max(0, int(args.x1)), max(0, int(args.y1)), min(W, int(args.x2)), min(H, int(args.y2)))
    c = cv2.resize(img[y1:y2, x1:x2], ((x2 - x1) * args.scale, (y2 - y1) * args.scale), interpolation=cv2.INTER_CUBIC)
    for x in range((x1 // 8 + 1) * 8, x2, 8):
        cv2.line(c, ((x - x1) * args.scale, 0), ((x - x1) * args.scale, 6 if x % 32 else 14), (255, 255, 0), 1)
        if x % 32 == 0:
            cv2.putText(c, str(x), ((x - x1) * args.scale + 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for y in range((y1 // 8 + 1) * 8, y2, 8):
        cv2.line(c, (0, (y - y1) * args.scale), (6 if y % 32 else 14, (y - y1) * args.scale), (255, 255, 0), 1)
        if y % 32 == 0:
            cv2.putText(c, str(y), (16, (y - y1) * args.scale + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    out = (Path(args.out) if args.out else review_dir()) / f"{Path(args.frame).stem}_crop_{x1}_{y1}_{x2}_{y2}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), c)
    print(out)
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """Một vùng của khung + cùng vùng ở các khung lân cận theo giờ (người đi lại, đốm/vật thì đứng yên hoặc nhấp nháy)."""
    import cv2
    import numpy as np

    with (repo_root() / "data/real/manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["kind"] == "flight"]
    t_of = {r["frame"]: float(r["t_unix"]) for r in rows}
    t0 = t_of[args.frame]
    names = sorted(t_of, key=lambda f: t_of[f])
    picks = []
    for off in args.offsets:
        best = min(names, key=lambda f: abs(t_of[f] - (t0 + off)))
        if abs(t_of[best] - (t0 + off)) <= max(0.3, abs(off) * 0.5) and best not in [p for p, _ in picks]:
            picks.append((best, t_of[best] - t0))
    x1, y1 = max(0, int(args.x1) - args.pad), max(0, int(args.y1) - args.pad)
    x2, y2 = min(W, int(args.x2) + args.pad), min(H, int(args.y2) + args.pad)
    tiles = []
    for f, dt in picks:
        img = cv2.imread(str(repo_root() / FRAMES_REL / f), cv2.IMREAD_COLOR)
        c = cv2.resize(img[y1:y2, x1:x2], ((x2 - x1) * args.scale, (y2 - y1) * args.scale),
                       interpolation=cv2.INTER_CUBIC)
        if f == args.frame:
            cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), (180, 60, 255), 2)
        c = cv2.copyMakeBorder(c, 20, 0, 0, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(c, f"{dt:+.1f}s", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(c)
    strip = np.hstack(tiles)
    out = (Path(args.out) if args.out else review_dir()) / f"{Path(args.frame).stem}_ctx_{x1}_{y1}_{x2}_{y2}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(out)
    return 0


def yolo_lines(boxes: list[list[float]]) -> str:
    """Hộp px gốc → dòng YOLO ``0 cx cy w h`` chuẩn hoá theo 640×480, 6 chữ số."""
    lines = []
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
        w, h = (b[2] - b[0]) / W, (b[3] - b[1]) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "".join(x + "\n" for x in lines)


def write_labels(dir_: Path, labels: dict[str, list[list[float]]]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for f, boxes in labels.items():
        (dir_ / f"{Path(f).stem}.txt").write_text(yolo_lines(boxes), encoding="utf-8", newline="\n")


def cmd_apply(args: argparse.Namespace) -> int:
    root = repo_root()
    pre = load_prelabel(root)["frames"]
    parts = sorted((root / PARTS_REL).glob("*.csv"))
    edits = read_edits(parts)
    m = load_manifest(root)
    expected = {f: "test" for f in m["test_frames"]} | {f: "devval" for f in m["devval_frames"]}
    missing = sorted(f for f in expected if f not in edits)
    extra = sorted(f for f in edits if f not in expected)
    final: dict[str, list[list[float]]] = {}
    hard: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    log_rows = []
    for f, sp in expected.items():
        if f not in edits:
            continue
        boxes, is_hard = apply_frame(pre[f]["boxes"], edits[f])
        final[f] = boxes
        if is_hard:
            hard.append(f)
        src = {b["id"]: b["source"] for b in pre[f]["boxes"]}
        for e in edits[f]:
            counts[e["action"]] += 1
            if e["action"] in ("drop", "tighten"):
                counts[f"{e['action']}_{src[int(e['box_id'])]}"] += 1
            log_rows.append([f, sp, e["action"], e["box_id"], e["x1"], e["y1"], e["x2"], e["y2"],
                             src.get(int(e["box_id"]), "") if e["box_id"] else "", e["reason"]])
    out = {"frames_expected": len(expected), "frames_reviewed": len(final), "missing": missing[:20],
           "n_missing": len(missing), "extra": extra, "actions": dict(sorted(counts.items())), "hard": len(hard),
           "prelabel_boxes": {sp: sum(len(pre[f]["boxes"]) for f, s in expected.items() if s == sp)
                              for sp in ("test", "devval")},
           "final_boxes": {sp: sum(len(final.get(f, [])) for f, s in expected.items() if s == sp)
                           for sp in ("test", "devval")}}
    if missing or extra:
        print(json.dumps(out, ensure_ascii=False))
        print(f"LỖI: {len(missing)} khung chưa soát, {len(extra)} khung lạ", file=sys.stderr)
        return 1
    log = root / LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["frame", "split", "action", "box_id", "x1", "y1", "x2", "y2", "prelabel_source", "reason"])
        w.writerows(sorted(log_rows, key=lambda r: (r[1], r[0])))
    write_labels(root / LABELS_REL / "devval", {f: b for f, b in final.items() if expected[f] == "devval"})
    write_labels(root / AGENT_TEST_REL, {f: b for f, b in final.items() if expected[f] == "test"})
    (root / REVIEWED_JSON_REL).write_text(json.dumps(
        {"frames": {f: {"split": expected[f], "hard": f in hard, "boxes": [[round(v, 2) for v in b] for b in bs]}
                    for f, bs in sorted(final.items())}}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    m["hard_frames"] = sorted(hard)
    m["n_boxes"] = {"test": m["n_boxes"].get("test", 0) if m.get("approved_by_owner") else 0,
                    "devval": out["final_boxes"]["devval"]}
    pre_doc = load_prelabel(root)
    m["prelabel_models"] = [{"name": x["name"], "weights": Path(x["weights"]).name, "sha256": x["sha256"]}
                            for x in pre_doc["models"]]
    save_manifest(m, root)
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.review", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="dựng ảnh soát (hoặc ảnh sau sửa với --edits)")
    r.add_argument("--frames", nargs="+", required=True)
    r.add_argument("--edits", nargs="*")
    r.add_argument("--scale", type=int, default=SCALE)
    r.add_argument("--out")
    c = sub.add_parser("crop", help="phóng một vùng khung (px gốc) để xem kỹ")
    c.add_argument("frame")
    for k in ("x1", "y1", "x2", "y2"):
        c.add_argument(k, type=float)
    c.add_argument("--scale", type=int, default=6)
    c.add_argument("--out")
    x = sub.add_parser("context", help="cùng một vùng ở các khung lân cận theo giờ")
    x.add_argument("frame")
    for k in ("x1", "y1", "x2", "y2"):
        x.add_argument(k, type=float)
    x.add_argument("--offsets", type=float, nargs="+", default=[-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0])
    x.add_argument("--pad", type=int, default=24)
    x.add_argument("--scale", type=int, default=3)
    x.add_argument("--out")
    a = sub.add_parser("apply", help="gộp các file sửa → review_log.csv + nhãn")
    a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return {"render": cmd_render, "crop": cmd_crop, "context": cmd_context, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
