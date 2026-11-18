"""Chọn tập dev-val (phase 08 bước 1) và đọc/ghi ``data/real/labels/manifest.json``.

Dev-val là 100 khung **dev** dùng chọn checkpoint / ngưỡng (D10: không chọn trên test). Luật chọn:

- chỉ khung ``split == "dev"`` trong ``data/real/split.json`` (file chỉ đọc, không sửa);
- phân tầng **như test**: tầng = (bin độ cao của khung theo ``xla.camera.split.ALT_BINS``, khung có người sơ bộ theo
  ``results/data/prelim_person_boxes.csv``); số khung mỗi tầng chia theo tỉ lệ khung test của tầng đó (phần dư lớn nhất);
- mọi cặp khung dev-val cách nhau ≥ ``DEVVAL_MIN_GAP_S`` giây (giờ ``t_unix``);
- trong mỗi tầng, thứ tự ngẫu nhiên theo seed; tầng hiếm (ít khung dev) chọn trước để không bị tầng khác chiếm chỗ.

Dùng::

    python -m xla.detect.devval select --json     # ghi danh sách vào manifest (giữ các trường khác)
    python -m xla.detect.devval show --json       # in số đếm theo tầng từ manifest hiện có
"""

from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera.split import alt_bin
from xla.paths import repo_root

DEVVAL_SEED = 20261118
DEVVAL_N = 100
DEVVAL_MIN_GAP_S = 1.0
MANIFEST_VERSION = "xla-test-v1"

SPLIT_REL = Path("data/real/split.json")
PRELIM_REL = Path("results/data/prelim_person_boxes.csv")
LABELS_REL = Path("data/real/labels")
MANIFEST_REL = LABELS_REL / "manifest.json"


def default_manifest() -> dict[str, Any]:
    """Manifest rỗng đúng định dạng chung của lane labels / baseline."""
    return {
        "version": MANIFEST_VERSION,
        "approved_by_owner": False,
        "approved_date": None,
        "test_frames": [],
        "devval_frames": [],
        "hard_frames": [],
        "n_boxes": {"test": 0, "devval": 0},
        "prelabel_models": [],
        "owner_edits": None,
        "devval_seed": DEVVAL_SEED,
        "devval_min_gap_s": DEVVAL_MIN_GAP_S,
    }


def load_manifest(root: Path | None = None) -> dict[str, Any]:
    p = (root or repo_root()) / MANIFEST_REL
    if not p.exists():
        return default_manifest()
    m = json.loads(p.read_text(encoding="utf-8"))
    base = default_manifest()
    base.update(m)
    return base


def save_manifest(m: dict[str, Any], root: Path | None = None) -> Path:
    p = (root or repo_root()) / MANIFEST_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return p


def load_split(root: Path | None = None) -> list[dict[str, Any]]:
    return json.loads(((root or repo_root()) / SPLIT_REL).read_text(encoding="utf-8"))["frames"]


def read_person_frames(root: Path | None = None) -> set[str]:
    with ((root or repo_root()) / PRELIM_REL).open(encoding="utf-8", newline="") as f:
        return {r["frame"] for r in csv.DictReader(f)}


def stratum(frame: dict[str, Any], person: set[str]) -> tuple[str, bool]:
    return alt_bin(float(frame["alt_m"])), frame["frame"] in person


def allocate(weights: dict[tuple[str, bool], int], n: int) -> dict[tuple[str, bool], int]:
    """Chia ``n`` theo tỉ lệ ``weights`` bằng phần dư lớn nhất (tất định: hoà thì theo khoá tăng dần)."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("không có khung test để lấy tỉ lệ tầng")
    raw = {k: n * w / total for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


class _Spacing:
    """Tập giờ đã chọn, kiểm nhanh khoảng cách nhỏ nhất tới giờ mới."""

    def __init__(self, gap: float) -> None:
        self.gap = gap
        self.ts: list[float] = []

    def ok(self, t: float) -> bool:
        i = bisect.bisect_left(self.ts, t)
        if i < len(self.ts) and self.ts[i] - t < self.gap:
            return False
        return not (i > 0 and t - self.ts[i - 1] < self.gap)

    def add(self, t: float) -> None:
        bisect.insort(self.ts, t)


def select_devval(frames: list[dict[str, Any]], person: set[str], n: int = DEVVAL_N, seed: int = DEVVAL_SEED,
                  min_gap_s: float = DEVVAL_MIN_GAP_S) -> tuple[list[str], dict[str, Any]]:
    """Trả (danh sách khung dev-val sắp theo giờ, báo cáo). Không đọc/ghi file; tất định theo ``seed``."""
    test_w = Counter(stratum(f, person) for f in frames if f["split"] == "test")
    quota = allocate(dict(test_w), n)
    pool: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for f in frames:
        if f["split"] == "dev":
            pool[stratum(f, person)].append(f)
    rng = np.random.default_rng(seed)
    spacing = _Spacing(min_gap_s)
    chosen: list[dict[str, Any]] = []
    got: dict[tuple[str, bool], int] = defaultdict(int)
    # sức chứa của tầng = số khung chọn được nếu chỉ tầng đó (tham lam theo giờ, cách ≥ min_gap_s)
    cap = {}
    for s in quota:
        sp = _Spacing(min_gap_s)
        for t in sorted(float(f["t_unix"]) for f in pool[s]):
            if sp.ok(t):
                sp.add(t)
        cap[s] = len(sp.ts)
    order = {s: [i for i in rng.permutation(len(pool[s]))] for s in sorted(quota)}
    cand = {s: sorted(pool[s], key=lambda f: f["frame"]) for s in quota}
    pos = defaultdict(int)

    def take(s: tuple[str, bool], want: int) -> int:
        k = 0
        while k < want and pos[s] < len(order[s]):
            f = cand[s][order[s][pos[s]]]
            pos[s] += 1
            t = float(f["t_unix"])
            if spacing.ok(t):
                spacing.add(t)
                chosen.append(f)
                k += 1
        got[s] += k
        return k

    # lượt 1: tầng có người trước (hiếm, nằm xen giữa khung không người cùng lúc), rồi tầng chật trước
    # (sức chứa / chỉ tiêu nhỏ nhất), khoá để hoà tất định
    for s in sorted(quota, key=lambda s: (not s[1], cap[s] / max(quota[s], 1), s)):
        take(s, quota[s])
    # lượt 2: tầng thiếu (cụm khung dày, cách 1 s không đủ) → bù phần thiếu cho các tầng còn khung, cùng trạng thái
    # có/không người trước, chia theo tỉ lệ test
    deficit = n - len(chosen)
    redistributed: dict[str, int] = {}
    for same_person_first in (True, False):
        while deficit > 0:
            short_person = {s[1] for s in quota if got[s] < quota[s]}
            live = {s: test_w[s] for s in quota if pos[s] < len(order[s])
                    and (not same_person_first or s[1] in short_person)}
            if not live:
                break
            share = allocate(live, deficit)
            added = 0
            for s in sorted(share):
                a = take(s, share[s])
                if a:
                    redistributed[f"{s[0]}|{'person' if s[1] else 'no_person'}"] = (
                        redistributed.get(f"{s[0]}|{'person' if s[1] else 'no_person'}", 0) + a)
                added += a
            deficit -= added
            if added == 0:
                break
    chosen.sort(key=lambda f: float(f["t_unix"]))
    names = [f["frame"] for f in chosen]

    def key(s: tuple[str, bool]) -> str:
        return f"{s[0]}|{'person' if s[1] else 'no_person'}"

    report = {
        "n": len(names),
        "seed": seed,
        "min_gap_s": min_gap_s,
        "quota_by_stratum": {key(s): quota[s] for s in sorted(quota)},
        "chosen_by_stratum": {key(s): got.get(s, 0) for s in sorted(quota)},
        "test_by_stratum": {key(s): test_w[s] for s in sorted(test_w)},
        "dev_pool_by_stratum": {key(s): len(pool[s]) for s in sorted(pool)},
        "dev_capacity_by_stratum": {key(s): cap[s] for s in sorted(cap)},
        "redistributed_extra": redistributed,
        "shortfall": n - len(names),
    }
    return names, report


def min_gap(frames_by_name: dict[str, dict[str, Any]], names: list[str]) -> float:
    ts = sorted(float(frames_by_name[x]["t_unix"]) for x in names)
    return min((b - a for a, b in itertools.pairwise(ts)), default=float("inf"))


def cmd_select(args: argparse.Namespace) -> int:
    root = repo_root()
    frames = load_split(root)
    person = read_person_frames(root)
    names, report = select_devval(frames, person, n=args.n, seed=args.seed, min_gap_s=args.min_gap)
    m = load_manifest(root)
    m["test_frames"] = sorted(f["frame"] for f in frames if f["split"] == "test")
    m["devval_frames"] = names
    m["devval_seed"] = args.seed
    m["devval_min_gap_s"] = args.min_gap
    save_manifest(m, root)
    report["min_gap_observed_s"] = round(min_gap({f["frame"]: f for f in frames}, names), 3)
    print(json.dumps(report, ensure_ascii=False) if args.json else report)
    if report["shortfall"]:
        print(f"LỖI: thiếu {report['shortfall']} khung dev-val", file=sys.stderr)
        return 1
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = repo_root()
    frames = {f["frame"]: f for f in load_split(root)}
    person = read_person_frames(root)
    m = load_manifest(root)
    c = Counter(stratum(frames[x], person) for x in m["devval_frames"])
    out = {"n": len(m["devval_frames"]),
           "by_stratum": {f"{s[0]}|{'person' if s[1] else 'no_person'}": v for s, v in sorted(c.items())},
           "min_gap_observed_s": round(min_gap(frames, m["devval_frames"]), 3) if m["devval_frames"] else None}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.devval", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="chọn dev-val, ghi vào manifest")
    s.add_argument("--n", type=int, default=DEVVAL_N)
    s.add_argument("--seed", type=int, default=DEVVAL_SEED)
    s.add_argument("--min-gap", type=float, default=DEVVAL_MIN_GAP_S, help="giây tối thiểu giữa hai khung dev-val")
    s.add_argument("--json", action="store_true")
    w = sub.add_parser("show", help="số đếm dev-val theo tầng")
    w.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return {"select": cmd_select, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
