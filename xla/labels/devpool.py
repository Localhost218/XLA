"""Tập nhãn dev mở rộng ``devpool`` + fold theo phiên cho chọn model k-fold (AI v2 phase 04, hợp đồng §14.2).

Lệnh (mã thoát 0 xong / 1 dữ liệu thiếu hoặc sai / 2 sai cách gọi)::

    python -m xla.labels.devpool select --json     # chọn khung -> data/real/labels/devpool/devpool.json
    python -m xla.labels.devpool prelabel --json   # hợp hai model (WBF như xla.detect.prelabel) -> devpool/review/
    python -m xla.labels.devpool sheets --json     # ảnh soát (xla.detect.review.render) ghép 2×2, ra cache ngoài repo
    python -m xla.labels.devpool apply --json      # review/edits.csv -> devpool/<stem>.txt (YOLO 640x480)
    python -m xla.labels.devpool folds --json      # -> data/real/labels/folds.json

Chọn (``select``): khung ``dev`` của ``split.json`` v1 trừ mọi khung bị giữ cho ``cal``/``holdout``
(``xla.camera.sessions.reserved_frames``: phiên cal/holdout, cả khối 30 s của ``cal_mode: block``); khung test không bao
giờ vào. Mọi khung dev-val 100 v1 còn lại vào trước, giữ nguyên byte nhãn cũ; phần còn lại phân tầng (phiên × bin độ cao
× có/không ứng viên người theo hộp hợp của nhãn trước), tầng có ứng viên trước (xoay vòng), rồi tầng không ứng viên chia
theo cỡ tầng; mọi cặp khung cách nhau ≥ ``gap``. ``gap`` = mốc lớn nhất trong ``CAPACITY_GAPS`` (bắt đầu từ 1 s của plan)
mà sức chứa khung có ứng viên đạt ``PERSON_MIN`` (``choose_gap``); đích ``TARGET_MIN``–``TARGET_MAX`` khung không đạt ở
mốc nào thì lấy hết sức chứa và báo thiếu (bảng sức chứa ở ``devpool.json``, README "Lệch so với plan").

Soát (``sheets`` → ``review/edits.csv`` → ``apply``): cùng định dạng sửa với ``xla.detect.review`` (``ok``, ``drop``,
``tighten``, ``add``, ``hard``), mọi khung mới phải có ≥ 1 dòng; dev-val v1 không soát lại (chép nhãn cũ).

Fold (``folds``): ``GroupKFold`` theo phiên (gán tham lam: phiên nhiều khung có người trước, vào fold ít người nhất) với
số fold lớn nhất ≤ ``N_FOLDS_MAX`` mà mọi fold có ≥ ``FOLD_PERSON_MIN`` khung có người, khi ≥ 3 phiên có khung người;
không thì ``block5``: khối 30 s v1 liền nhau theo giờ chia thành 5 đoạn liên tiếp, cân số khung có người.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera import sessions as S
from xla.camera.split import alt_bin
from xla.paths import cache_dir, repo_root

SEED = 20261125
MIN_GAP_S = 1.0
"""Khoảng cách của plan; chỉ nới theo ``CAPACITY_GAPS`` khi sức chứa không đạt ``PERSON_MIN`` (``choose_gap``)."""
CAPACITY_GAPS = (1.0, 0.5, 0.33, 0.2)
TARGET_MIN = 500
TARGET_MAX = 700
PERSON_MIN = 150
N_FOLDS_MAX = 5
FOLD_PERSON_MIN = 20
BLOCK5_FOLDS = 5
MIN_SESSIONS_FOR_GROUPS = 3
W, H = 640, 480

LABELS_REL = Path("data/real/labels")
DEVPOOL_REL = LABELS_REL / "devpool"
DEVPOOL_JSON_REL = S.DEVPOOL_JSON_REL
REVIEW_REL = DEVPOOL_REL / "review"
PRELABEL_REL = REVIEW_REL / "prelabel.json"
EDITS_REL = REVIEW_REL / "edits.csv"
REVIEW_LOG_REL = REVIEW_REL / "review_log.csv"
LABELS_JSON_REL = REVIEW_REL / "labels.json"
FOLDS_REL = LABELS_REL / "folds.json"
FOLD_INFO_REL = DEVPOOL_REL / "fold_info.json"
DEVVAL_DIR_REL = LABELS_REL / "devval"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Spacing:
    """Giờ đã chọn (sắp tăng), kiểm khoảng cách nhỏ nhất tới giờ mới."""

    def __init__(self, gap: float) -> None:
        self.gap = gap
        self.ts: list[float] = []

    def ok(self, t: float) -> bool:
        i = bisect.bisect_left(self.ts, t)
        return not ((i < len(self.ts) and self.ts[i] - t < self.gap) or (i > 0 and t - self.ts[i - 1] < self.gap))

    def add(self, t: float) -> None:
        bisect.insort(self.ts, t)


def capacity(ts: list[float], gap: float) -> int:
    """Số khung tối đa chọn được cách nhau ≥ ``gap`` (tham lam theo giờ là tối ưu)."""
    n, last = 0, -float("inf")
    for t in sorted(ts):
        if t - last >= gap:
            n, last = n + 1, t
    return n


def min_gap(ts: list[float]) -> float:
    s = sorted(ts)
    return min((b - a for a, b in itertools.pairwise(s)), default=float("inf"))


# ---------------------------------------------------------------- chọn khung


def pool_frames(rows: list[dict[str, Any]], split_v1: dict[str, dict[str, Any]], neg_doc: dict[str, Any],
                ) -> list[dict[str, Any]]:
    """Khung được phép vào devpool: ``dev`` v1 (và khung phiên ``mine`` tự do), trừ khung bị giữ cho cal/holdout."""
    reserved = S.reserved_frames(neg_doc, rows, split_v1)
    roles = neg_doc["sessions"]
    locked = S.locked_sessions(rows, split_v1)
    out = []
    for r in rows:
        v = split_v1.get(r["frame"])
        if r["frame"] in reserved or (v and v["split"] == "test"):
            continue
        if (v and v["split"] == "dev") or (r["session"] not in locked and roles.get(r["session"]) == "mine"):
            out.append({"frame": r["frame"], "session": r["session"], "t": float(r["t_unix"]),
                        "alt_m": float(r["alt_m"]), "block": v["block"] if v else None})
    return sorted(out, key=lambda f: f["frame"])


def capacity_table(pool: list[dict[str, Any]], cand: set[str]) -> list[dict[str, Any]]:
    """Sức chứa (số khung chọn được) của pool và của khung có ứng viên ở từng khoảng cách ``CAPACITY_GAPS``."""
    return [{"min_gap_s": g, "capacity": capacity([f["t"] for f in pool], g),
             "capacity_candidate_person": capacity([f["t"] for f in pool if f["frame"] in cand], g)}
            for g in CAPACITY_GAPS]


def choose_gap(pool: list[dict[str, Any]], cand: set[str]) -> float:
    """Mốc lớn nhất trong ``CAPACITY_GAPS`` có sức chứa khung ứng viên ≥ ``PERSON_MIN``; không mốc nào đạt → mốc nhỏ nhất."""
    for row in capacity_table(pool, cand):
        if row["capacity_candidate_person"] >= PERSON_MIN:
            return row["min_gap_s"]
    return CAPACITY_GAPS[-1]


def stratum(f: dict[str, Any], cand: set[str]) -> tuple[str, str, bool]:
    return f["session"], alt_bin(f["alt_m"]), f["frame"] in cand


def select(pool: list[dict[str, Any]], devval: list[str], cand: set[str], seed: int = SEED,
           gap: float = MIN_GAP_S, n_max: int = TARGET_MAX) -> tuple[list[str], dict[str, Any]]:
    """Chọn devpool: dev-val còn trong pool (ép vào trước) → tầng có ứng viên (xoay vòng, lấy hết sức chứa) → tầng không
    ứng viên (chia theo cỡ tầng, phần dư lớn nhất), mọi cặp cách nhau ≥ ``gap``. Thuần, tất định theo ``seed``."""
    from xla.detect.devval import allocate

    by_name = {f["frame"]: f for f in pool}
    rng = np.random.default_rng(seed)
    sp = Spacing(gap)
    chosen: list[str] = []
    forced = [f for f in devval if f in by_name]
    for f in sorted(forced, key=lambda x: by_name[x]["t"]):
        if not sp.ok(by_name[f]["t"]):
            raise ValueError(f"dev-val {f} cách khung dev-val khác < {gap} s")
        sp.add(by_name[f]["t"])
        chosen.append(f)
    strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for f in pool:
        if f["frame"] not in chosen:
            strata[stratum(f, cand)].append(f)
    order = {s: [strata[s][i] for i in rng.permutation(len(strata[s]))] for s in sorted(strata)}

    def take(s: tuple[str, str, bool], want: int) -> int:
        k = 0
        while k < want and order[s]:
            f = order[s].pop(0)
            if sp.ok(f["t"]):
                sp.add(f["t"])
                chosen.append(f["frame"])
                k += 1
        return k

    live = [s for s in sorted(order) if s[2]]
    while live and len(chosen) < n_max:
        live = [s for s in live if take(s, 1)]
    while len(chosen) < n_max:
        neg = {s: len(order[s]) for s in sorted(order) if not s[2] and order[s]}
        if not neg:
            break
        share = allocate(neg, n_max - len(chosen))
        if sum(take(s, share[s]) for s in sorted(share)) == 0:
            break
    names = sorted(chosen, key=lambda x: by_name[x]["t"])
    n_cand = sum(f in cand for f in names)
    c = Counter(stratum(by_name[f], cand) for f in names)
    report = {"seed": seed, "min_gap_s": gap, "n": len(names), "n_devval": len(forced),
              "n_candidate_person": n_cand, "target": [TARGET_MIN, TARGET_MAX], "person_min": PERSON_MIN,
              "met": TARGET_MIN <= len(names) <= TARGET_MAX and n_cand >= PERSON_MIN,
              "capacity_table": capacity_table(pool, cand),
              "by_stratum": {f"{s}|{b}|{'cand' if p else 'no_cand'}": v for (s, b, p), v in sorted(c.items())},
              "pool": len(pool), "pool_candidate_person": sum(f["frame"] in cand for f in pool)}
    return names, report


# ---------------------------------------------------------------- fold


def greedy_group_folds(groups: dict[str, int], k: int) -> list[list[str]]:
    """GroupKFold tham lam: nhóm nhiều khung người trước, vào fold ít người nhất (hoà: fold chỉ số nhỏ). Tất định."""
    folds: list[list[str]] = [[] for _ in range(k)]
    load = [0] * k
    for g in sorted(groups, key=lambda g: (-groups[g], g)):
        i = min(range(k), key=lambda i: (load[i], i))
        folds[i].append(g)
        load[i] += groups[g]
    return folds


def contiguous_folds(items: list[tuple[Any, int]], k: int) -> list[list[Any]]:
    """Chia dãy ``(khoá, số người)`` theo thứ tự thành ``k`` đoạn liên tiếp khác rỗng, cực đại số người nhỏ nhất của
    một đoạn (quy hoạch động; hoà thì giữ điểm cắt tìm thấy trước)."""
    n = len(items)
    if n < k:
        raise ValueError(f"chỉ có {n} khối, cần ≥ {k}")
    pre = [0]
    for _, p in items:
        pre.append(pre[-1] + p)
    best = [[-1.0] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = float("inf")
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for m in range(j - 1, i):
                v = min(best[j - 1][m], pre[i] - pre[m])
                if v > best[j][i]:
                    best[j][i], cut[j][i] = v, m
    out, i = [], n
    for j in range(k, 0, -1):
        m = cut[j][i]
        out.append([key for key, _ in items[m:i]])
        i = m
    return out[::-1]


def _fold_counts(folds: list[list[str]], person: set[str]) -> dict[str, Any]:
    return {"frames_per_fold": [len(f) for f in folds],
            "person_frames_per_fold": [sum(x in person for x in f) for f in folds]}


def build_folds(frames: list[dict[str, Any]], person: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``folds.json`` (§14.2: ``{groups, folds}``) + thông tin fold. ``frames`` = ``[{frame, session, t, block}]``."""
    by_s: dict[str, list[str]] = defaultdict(list)
    for f in frames:
        by_s[f["session"]].append(f["frame"])
    pers_s = {s: sum(x in person for x in fs) for s, fs in by_s.items()}
    info: dict[str, Any] = {"seed": SEED, "fold_person_min": FOLD_PERSON_MIN,
                            "sessions_with_person": sum(v > 0 for v in pers_s.values()),
                            "person_frames_per_session": dict(sorted(pers_s.items())), "tried": []}
    if info["sessions_with_person"] >= MIN_SESSIONS_FOR_GROUPS:
        for k in range(min(N_FOLDS_MAX, info["sessions_with_person"]), 1, -1):
            gf = greedy_group_folds(pers_s, k)
            load = [sum(pers_s[s] for s in g) for g in gf]
            info["tried"].append({"k": k, "person_per_fold": load})
            if min(load) >= FOLD_PERSON_MIN:
                folds = [sorted(x for s in g for x in by_s[s]) for g in gf]
                info |= {"groups": "session", "k": k, "fold_sessions": [sorted(g) for g in gf]}
                return {"groups": "session", "folds": folds}, info | _fold_counts(folds, person)
    blocks: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for f in frames:
        blocks[f["block"]].append(f)
    order = sorted(blocks, key=lambda b: min(f["t"] for f in blocks[b]))
    parts = contiguous_folds([(b, sum(f["frame"] in person for f in blocks[b])) for b in order], BLOCK5_FOLDS)
    folds = [sorted(f["frame"] for b in part for f in blocks[b]) for part in parts]
    info |= {"groups": "block5", "k": BLOCK5_FOLDS, "fold_blocks": parts}
    return {"groups": "block5", "folds": folds}, info | _fold_counts(folds, person)


# ---------------------------------------------------------------- nhãn


def read_edits(path: Path) -> dict[str, list[dict[str, str]]]:
    from xla.detect.review import read_edits as _read

    return dict(_read([path])) if path.exists() else {}


def final_labels(pre: dict[str, Any], edits: dict[str, list[dict[str, str]]], frames: list[str],
                 ) -> tuple[dict[str, list[list[float]]], list[str], list[str]]:
    """(hộp cuối mỗi khung, khung hard, khung thiếu dòng soát) qua ``xla.detect.review.apply_frame``."""
    from xla.detect.review import apply_frame

    out, hard, missing = {}, [], []
    for f in frames:
        if not edits.get(f):
            missing.append(f)
            continue
        out[f], h = apply_frame(pre[f]["boxes"], edits[f])
        if h:
            hard.append(f)
    return out, hard, missing


def valid_yolo(text: str) -> bool:
    """Nhãn YOLO hợp lệ 640×480: mỗi dòng ``0 cx cy w h``, cạnh trong (0, 1], hộp nằm trong khung."""
    for line in text.splitlines():
        p = line.split()
        if len(p) != 5 or p[0] != "0":
            return False
        cx, cy, w, h = (float(x) for x in p[1:])
        if not (0 < w <= 1 and 0 < h <= 1 and cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6
                and cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6):
            return False
    return True


def yolo_boxes(text: str) -> list[list[float]]:
    out = []
    for line in text.splitlines():
        _, cx, cy, w, h = (float(x) for x in line.split())
        out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return out


# ---------------------------------------------------------------- lệnh


def _context(root: Path) -> dict[str, Any]:
    return {"rows": S.flight_rows(root), "v1": S.load_split_v1(root), "neg": S.load_neg_split(root),
            "raw": _load(root / S.PRELABEL_RAW_REL)["frames"],
            "devval": _load(root / S.DEVVAL_MANIFEST_REL)["devval_frames"]}


def cmd_select(args: argparse.Namespace) -> int:
    root = repo_root()
    c = _context(root)
    pool = pool_frames(c["rows"], c["v1"], c["neg"])
    missing = [f["frame"] for f in pool if f["frame"] not in c["raw"]]
    if missing:
        raise ValueError(f"{len(missing)} khung pool chưa có nhãn trước (sessions prelabel): {missing[:3]}")
    cand = S.fused_candidate_frames([f["frame"] for f in pool], c["raw"])
    gap = args.min_gap if args.min_gap is not None else choose_gap(pool, cand)
    names, report = select(pool, c["devval"], cand, seed=args.seed, gap=gap, n_max=args.n_max)
    by = {f["frame"]: f for f in pool}
    report |= {"min_gap_observed_s": round(min_gap([by[f]["t"] for f in names]), 3),
               "devval_not_in_devpool": sorted(set(c["devval"]) - set(names)),
               "devval_not_in_devpool_reason": "nằm trong khối cal (cal_mode block), bị giữ khỏi devpool"}
    doc = {"version": "devpool-v1", "seed": args.seed, "min_gap_s": report["min_gap_s"], "frames": names,
           "sessions": {f: by[f]["session"] for f in names}, "report": report}
    S._dump(root / DEVPOOL_JSON_REL, doc)
    out = {k: report[k] for k in ("n", "n_devval", "n_candidate_person", "min_gap_s", "met", "pool")}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_prelabel(args: argparse.Namespace) -> int:
    from xla.detect import prelabel as P

    root = repo_root()
    names = _load(root / DEVPOOL_JSON_REL)["frames"]
    raw_doc = _load(root / S.PRELABEL_RAW_REL)
    missing = [f for f in names if f not in raw_doc["frames"]]
    gpu = None
    if missing:  # khung chưa có nhãn trước thô (vd. phiên mine mới): chạy GPU đúng như phase 03, trong gpu_lock
        gpu = S.run_prelabel(missing, root / S.PRELABEL_RAW_REL, label="phase04 devpool prelabel",
                             lock_timeout=args.lock_timeout)
        raw_doc = _load(root / S.PRELABEL_RAW_REL)
    frames = {f: {"split": "devpool", "boxes": S.fuse_raw(raw_doc["frames"][f])} for f in names}
    doc = {"version": "devpool-prelabel-v1", "source": S.PRELABEL_RAW_REL.as_posix(), "models": raw_doc["models"],
           "params": {"model_conf": P.MODEL_CONF, "single_keep_conf": P.SINGLE_KEEP_CONF, "wbf_iou": P.WBF_IOU,
                      "wbf_weights": [m["wbf_weight"] for m in P.MODELS], "match_iou": P.MATCH_IOU,
                      "raw": raw_doc["params"]},
           "frames": frames}
    S._dump(root / PRELABEL_REL, doc)
    src = Counter(b["source"] for v in frames.values() for b in v["boxes"])
    out = {"frames": len(names), "boxes": sum(src.values()), "by_source": dict(src),
           "frames_with_box": sum(bool(v["boxes"]) for v in frames.values()), "gpu": gpu}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_sheets(args: argparse.Namespace) -> int:
    """Ảnh soát của ``xla.detect.review.render`` (hộp gán trước theo nguồn, hoặc hộp cuối với ``--final``), ghép
    ``cols × rows`` khung mỗi tờ, theo thứ tự giờ."""
    import cv2

    from xla.detect import review as R

    root = repo_root()
    doc = _load(root / DEVPOOL_JSON_REL)
    pre = _load(root / PRELABEL_REL)["frames"]
    devval = set(_load(root / S.DEVVAL_MANIFEST_REL)["devval_frames"])
    edits = read_edits(root / EDITS_REL)
    names = [f for f in doc["frames"] if f not in devval]
    if args.frames:
        names = [f for f in names if f in set(args.frames)]
    final, _, _ = final_labels(pre, {f: edits.get(f) or [{"action": "ok"}] for f in names}, names)
    out_dir = Path(args.out) if args.out else cache_dir() / "xla-devpool-review"
    tiles = []
    for f in names:
        title = f"{f} pre={len(pre[f]['boxes'])} final={len(final.get(f, []))}"
        p = R.render(f, pre[f]["boxes"], out_dir / "frames" / f"{Path(f).stem}.jpg", args.scale,
                     final.get(f) if args.final else None, title)
        tiles.append(cv2.imread(str(p), cv2.IMREAD_COLOR))
    sheets = []
    per = args.cols * args.rows
    for i in range(0, len(tiles), per):
        chunk = tiles[i:i + per]
        chunk = chunk + [np.zeros_like(tiles[0])] * (per - len(chunk))
        sheet = np.vstack([np.hstack(chunk[r * args.cols:(r + 1) * args.cols]) for r in range(args.rows)])
        p = out_dir / f"{'final' if args.final else 'pre'}_{i // per:03d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheets.append(str(p))
    out = {"frames": len(names), "sheets": len(sheets), "dir": str(out_dir)}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from xla.detect.review import write_labels

    root = repo_root()
    doc = _load(root / DEVPOOL_JSON_REL)
    devval = set(_load(root / S.DEVVAL_MANIFEST_REL)["devval_frames"])
    pre = _load(root / PRELABEL_REL)["frames"]
    edits = read_edits(root / EDITS_REL)
    new = [f for f in doc["frames"] if f not in devval]
    final, hard, missing = final_labels(pre, edits, new)
    extra = sorted(set(edits) - set(new))
    if missing or extra:
        print(json.dumps({"missing": missing[:10], "n_missing": len(missing), "extra": extra[:10]}, ensure_ascii=False))
        print("LỖI: khung devpool chưa có dòng soát / dòng soát lạ", file=sys.stderr)
        return 1
    out_dir = root / DEVPOOL_REL
    keep = {f"{Path(f).stem}.txt" for f in doc["frames"]}
    for p in out_dir.glob("*.txt"):
        if p.name not in keep:
            p.unlink()
    write_labels(out_dir, final)
    for f in sorted(set(doc["frames"]) & devval):  # dev-val v1: chép nguyên byte nhãn cũ
        shutil.copyfile(root / DEVVAL_DIR_REL / f"{Path(f).stem}.txt", out_dir / f"{Path(f).stem}.txt")
    src = {f: {b["id"]: b["source"] for b in pre[f]["boxes"]} for f in new}
    log = [[f, e["action"], e["box_id"], e["x1"], e["y1"], e["x2"], e["y2"],
            src[f].get(int(e["box_id"]), "") if e["box_id"] else "", e["reason"]] for f in new for e in edits[f]]
    with (root / REVIEW_LOG_REL).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["frame", "action", "box_id", "x1", "y1", "x2", "y2", "prelabel_source", "reason"])
        w.writerows(log)
    labels = {f: [[round(v, 2) for v in b] for b in final[f]] for f in new}
    for f in sorted(set(doc["frames"]) & devval):
        labels[f] = [[round(v, 2) for v in b]
                     for b in yolo_boxes((out_dir / f"{Path(f).stem}.txt").read_text(encoding="utf-8"))]
    S._dump(root / LABELS_JSON_REL, {"hard": sorted(hard), "boxes": dict(sorted(labels.items()))})
    out = {"frames": len(doc["frames"]), "boxes": sum(len(v) for v in labels.values()),
           "frames_with_person": sum(bool(v) for v in labels.values()),
           "prelabel_boxes_new": sum(len(pre[f]["boxes"]) for f in new),
           "actions": dict(Counter(x[1] for x in log)),
           "actions_by_source": dict(Counter(f"{x[1]}_{x[7]}" for x in log if x[7])), "hard": len(hard)}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def devpool_frames_meta(root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """(``[{frame, session, t, block}]`` của devpool, khung có ≥ 1 hộp người theo nhãn trên đĩa)."""
    doc = _load(root / DEVPOOL_JSON_REL)
    v1 = S.load_split_v1(root)
    rows = {r["frame"]: r for r in S.flight_rows(root)}
    frames, person = [], set()
    for f in doc["frames"]:
        p = root / DEVPOOL_REL / f"{Path(f).stem}.txt"
        if not p.exists():
            raise ValueError(f"thiếu nhãn {p} (chạy `devpool apply`)")
        if p.read_text(encoding="utf-8").strip():
            person.add(f)
        frames.append({"frame": f, "session": rows[f]["session"], "t": float(rows[f]["t_unix"]),
                       "block": v1[f]["block"] if f in v1 else None})
    return frames, person


def cmd_folds(args: argparse.Namespace) -> int:
    root = repo_root()
    frames, person = devpool_frames_meta(root)
    folds, info = build_folds(frames, person)
    S._dump(root / FOLDS_REL, folds)
    S._dump(root / FOLD_INFO_REL, info)
    ok = min(info["person_frames_per_fold"]) >= FOLD_PERSON_MIN
    out = {k: info[k] for k in ("groups", "k", "frames_per_fold", "person_frames_per_fold")} | {"ok": ok}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    if not ok:
        print(f"LỖI: có fold < {FOLD_PERSON_MIN} khung có người", file=sys.stderr)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.labels.devpool", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="chọn khung devpool")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--n-max", type=int, default=TARGET_MAX)
    s.add_argument("--min-gap", type=float, help="ép khoảng cách (mặc định: choose_gap)")
    s.add_argument("--json", action="store_true")
    p = sub.add_parser("prelabel", help="hộp hợp hai model cho khung devpool")
    p.add_argument("--lock-timeout", type=float, default=6 * 3600.0)
    p.add_argument("--json", action="store_true")
    h = sub.add_parser("sheets", help="ảnh ghép soát ra cache")
    h.add_argument("--frames", nargs="*")
    h.add_argument("--scale", type=int, default=1)
    h.add_argument("--final", action="store_true", help="vẽ hộp cuối (sau edits.csv) thay hộp gán trước")
    h.add_argument("--cols", type=int, default=2)
    h.add_argument("--rows", type=int, default=2)
    h.add_argument("--out")
    h.add_argument("--json", action="store_true")
    a = sub.add_parser("apply", help="ghi nhãn YOLO devpool")
    a.add_argument("--json", action="store_true")
    f = sub.add_parser("folds", help="ghi folds.json")
    f.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cmds = {"select": cmd_select, "prelabel": cmd_prelabel, "sheets": cmd_sheets, "apply": cmd_apply,
            "folds": cmd_folds}
    try:
        return cmds[args.cmd](args)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
