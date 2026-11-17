"""Chia dev/test theo khối thời gian 30 s, phân tầng theo (bin độ cao, có người sơ bộ) — phase 03, Deferred 15, D10.

Chạy **một lần** cho cả dự án; sau phase 03, ``data/real/split.json`` là chỉ đọc (chỉ được chia lại trước phase 08).

Lệnh::

    python -m xla.camera.split build --seed 20261118 --json

Quy tắc:

- Khối = 30 s tính từ giờ bắt đầu chuyến (khung bay) hoặc từ khung đầu của lần ghi (mẫu âm); id khối đánh số toàn cục.
- Tầng của khối bay = (bin độ cao theo trung vị ``alt_m`` của khối, khối có ít nhất một khung có người sơ bộ).
- Gán khối test bằng ``numpy.random.default_rng(seed)``: trước hết mỗi bin độ cao có ≥ 2 khối có người đưa một khối có
  người vào test (luôn chừa ≥ 1 khối mỗi tầng cho dev); sau đó lần lượt lấy khối từ tầng có tỉ lệ khung đã vào test thấp
  nhất tới khi đủ ``TEST_MIN`` khung test, bỏ qua khối làm vượt ``TEST_MAX``.
- Khung test: trong khối test, chọn tham lam theo giờ, cách nhau ≥ ``TEST_SPACING_S``. Khung khối test không được chọn
  không thuộc tập nào.
- Khung dev: mọi khung của khối dev, trừ khung cách khoảng thời gian của một khối test < ``DEV_BUFFER_S``.
- Mẫu âm: ``split = "neg"``, ``alt_m = null`` khi lần ghi không có độ cao.

``--test-min/--test-max`` (mặc định 300/400 theo plan) chỉ để Owner chia lại có chủ đích; báo cáo ghi giá trị đã dùng
và trần khả thi (``feasibility``: số khung test tối đa nếu mọi khung bay vào test).

Exit 1 khi số khung test ngoài ``[TEST_MIN, TEST_MAX]`` hoặc số khung test có người sơ bộ < ``TEST_PERSON_MIN``
(file vẫn được ghi để Owner xem; không tự nới quy tắc).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera.frames import EVENTS_REL, PRELIM_REL, parse_flight_events, read_manifest
from xla.paths import repo_root

SEED_DEFAULT = 20261118
BLOCK_S = 30.0
ALT_BINS = ((3.0, 8.0, "3-8"), (8.0, 14.0, "8-14"), (14.0, math.inf, ">14"))
TEST_SPACING_S = 1.0
DEV_BUFFER_S = 5.0
TEST_MIN = 300
TEST_MAX = 400
TEST_PERSON_MIN = 150
MIN_DEV_BLOCKS_PER_STRATUM = 1
SPLIT_REL = Path("data/real/split.json")
REPORT_REL = Path("results/data/split_report.json")


def alt_bin(alt: float) -> str:
    """Nhãn bin độ cao (m); dưới 3 m (không có trong tập bay đã lọc) trả ``"<3"``."""
    for lo, hi, name in ALT_BINS:
        if lo <= alt < hi:
            return name
    return "<3"


def read_person_frames(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["frame"] for r in csv.DictReader(f)}


def greedy_spaced(times: list[float], spacing: float) -> list[int]:
    """Chỉ số (theo thứ tự ``times`` tăng dần) chọn tham lam, cách nhau ≥ ``spacing``."""
    out, last = [], -math.inf
    for i, t in enumerate(times):
        if t - last >= spacing:
            out.append(i)
            last = t
    return out


def _interval_distance(t: float, a: float, b: float) -> float:
    return 0.0 if a <= t < b else min(abs(t - a), abs(t - b))


def build(seed: int, manifest: list[dict[str, str]], person_frames: set[str], events_text: str,
          test_min: int = TEST_MIN, test_max: int = TEST_MAX) -> tuple[dict, dict]:
    """Trả (nội dung ``split.json`` theo hợp đồng §7, báo cáo). Không đọc/ghi file; tất định theo ``seed``."""
    flights = {f.flight_id: f for f in parse_flight_events(events_text)}
    fly = sorted((r for r in manifest if r["kind"] == "flight"), key=lambda r: float(r["t_unix"]))
    neg = sorted((r for r in manifest if r["kind"] == "neg"), key=lambda r: float(r["t_unix"]))

    # ---- khối bay
    key_of: dict[str, tuple[str, int]] = {}
    for r in fly:
        f = flights[r["flight_id"]]
        key_of[r["frame"]] = (f.flight_id, int((float(r["t_unix"]) - f.t0) // BLOCK_S))
    keys = sorted(set(key_of.values()))
    block_id = {k: i for i, k in enumerate(keys)}
    interval = {block_id[(fid, k)]: (flights[fid].t0 + k * BLOCK_S, flights[fid].t0 + (k + 1) * BLOCK_S)
                for fid, k in keys}
    members: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in fly:
        members[block_id[key_of[r["frame"]]]].append(r)
    info = {}
    for b, rs in members.items():
        med = float(np.median([float(r["alt_m"]) for r in rs]))
        n_test = len(greedy_spaced([float(r["t_unix"]) for r in rs], TEST_SPACING_S))
        info[b] = {"bin": alt_bin(med), "person": any(r["frame"] in person_frames for r in rs), "n": len(rs),
                   "n_test_cand": n_test, "alt_median": round(med, 2)}
    strata: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for b in sorted(info):
        strata[(info[b]["bin"], info[b]["person"])].append(b)

    # ---- gán khối test
    rng = np.random.default_rng(seed)
    order = {s: [bs[i] for i in rng.permutation(len(bs))] for s, bs in sorted(strata.items())}
    test_blocks: list[int] = []
    skipped: set[int] = set()

    def remaining(s: tuple[str, bool]) -> list[int]:
        return [b for b in order[s] if b not in test_blocks and b not in skipped]

    def n_test_frames() -> int:
        return sum(info[b]["n_test_cand"] for b in test_blocks)

    for lo, hi, name in ALT_BINS:
        s = (name, True)
        if s in order and len(order[s]) >= 2:
            test_blocks.append(order[s][0])
    stratum_frames = {s: sum(info[b]["n_test_cand"] for b in bs) for s, bs in order.items()}
    while n_test_frames() < test_min:
        cands = [s for s in sorted(order) if len(remaining(s)) > MIN_DEV_BLOCKS_PER_STRATUM]
        if not cands:
            break
        s = min(cands, key=lambda s: (sum(info[b]["n_test_cand"] for b in test_blocks if b in order[s])
                                      / max(stratum_frames[s], 1), s))
        b = remaining(s)[0]
        if n_test_frames() + info[b]["n_test_cand"] > test_max:
            skipped.add(b)
            continue
        test_blocks.append(b)
    test_set = set(test_blocks)

    # ---- chọn khung
    test_rows = [r for r in fly if block_id[key_of[r["frame"]]] in test_set]
    chosen = {test_rows[i]["frame"] for i in greedy_spaced([float(r["t_unix"]) for r in test_rows], TEST_SPACING_S)}
    test_iv = [interval[b] for b in sorted(test_set)]
    frames_out: list[dict[str, Any]] = []
    excluded = Counter()
    for r in fly:
        b = block_id[key_of[r["frame"]]]
        t = float(r["t_unix"])
        if b in test_set:
            if r["frame"] not in chosen:
                excluded["test_block_not_spaced"] += 1
                continue
            split = "test"
        else:
            if any(_interval_distance(t, a, z) < DEV_BUFFER_S for a, z in test_iv):
                excluded["dev_buffer_5s"] += 1
                continue
            split = "dev"
        frames_out.append({"frame": r["frame"], "t_unix": float(r["t_unix"]), "alt_m": float(r["alt_m"]),
                           "block": b, "split": split})
    # ---- mẫu âm: khối 30 s từ khung đầu mỗi lần ghi, id sau khối bay
    nb = len(keys)
    neg_keys: dict[tuple[str, int], int] = {}
    t0_rec: dict[str, float] = {}
    for r in neg:
        t = float(r["t_unix"])
        t0 = t0_rec.setdefault(r["rec"], t)
        k = (r["rec"], int((t - t0) // BLOCK_S))
        if k not in neg_keys:
            neg_keys[k] = nb + len(neg_keys)
        frames_out.append({"frame": r["frame"], "t_unix": t, "alt_m": float(r["alt_m"]) if r["alt_m"] else None,
                           "block": neg_keys[k], "split": "neg"})
    frames_out.sort(key=lambda d: (d["t_unix"], d["frame"]))
    split_doc = {"seed": seed, "block_s": int(BLOCK_S), "frames": frames_out}
    report = _report(seed, frames_out, person_frames, info, test_blocks, skipped, excluded, fly, test_min, test_max)
    return split_doc, report


def _feasibility(fly: list[dict[str, str]], person_frames: set[str]) -> dict:
    """Trần trên của tập test: nếu MỌI khung bay đều vào test (cách nhau ≥ ``TEST_SPACING_S``)."""
    t = [float(r["t_unix"]) for r in fly]
    idx = greedy_spaced(t, TEST_SPACING_S)
    person = Counter(alt_bin(float(fly[i]["alt_m"])) for i in idx if fly[i]["frame"] in person_frames)
    return {"max_test_frames_if_all_flight": len(idx), "max_test_person_frames_if_all_flight": sum(person.values()),
            "max_test_person_by_alt_bin": {b[2]: person.get(b[2], 0) for b in ALT_BINS},
            "max_frames_2s_apart_if_all_flight": len(greedy_spaced(t, 2.0))}


def _report(seed, frames_out, person_frames, info, test_blocks, skipped, excluded, fly, test_min, test_max) -> dict:
    fid = {r["frame"]: r["flight_id"] for r in fly}
    table: dict[str, Any] = {}
    cnt = Counter()
    for d in frames_out:
        if d["split"] == "neg":
            continue
        k = (fid[d["frame"]], d["split"], alt_bin(d["alt_m"]), d["frame"] in person_frames)
        cnt[k] += 1
    for (f, s, b, p), n in sorted(cnt.items()):
        table.setdefault(f, {}).setdefault(s, {}).setdefault(b, {})["person" if p else "no_person"] = n
    by_split = Counter(d["split"] for d in frames_out)
    test_person = Counter(alt_bin(d["alt_m"]) for d in frames_out
                          if d["split"] == "test" and d["frame"] in person_frames)
    dev_person = Counter(alt_bin(d["alt_m"]) for d in frames_out
                         if d["split"] == "dev" and d["frame"] in person_frames)
    n_test = by_split.get("test", 0)
    n_test_person = sum(test_person.values())
    checks = {
        "test_in_range": test_min <= n_test <= test_max,
        "test_person_ge_min": n_test_person >= TEST_PERSON_MIN,
        "test_person_bins_ge_2": sum(1 for v in test_person.values() if v > 0) >= 2,
    }
    return {
        "seed": seed,
        "rules": {"block_s": BLOCK_S, "alt_bins": [b[2] for b in ALT_BINS], "test_spacing_s": TEST_SPACING_S,
                  "dev_buffer_s": DEV_BUFFER_S, "test_target": [test_min, test_max],
                  "test_person_min": TEST_PERSON_MIN, "person": "YOLO11s COCO @1280 conf>=0.25, bỏ hộp x1<16"},
        "counts": dict(sorted(by_split.items())),
        "flight_frames_in_manifest": len(fly),
        "excluded": dict(sorted(excluded.items())),
        "flight_frames_with_person_prelim": sum(1 for r in fly if r["frame"] in person_frames),
        "feasibility": _feasibility(fly, person_frames),
        "test_person_frames_by_alt_bin": {b[2]: test_person.get(b[2], 0) for b in ALT_BINS},
        "dev_person_frames_by_alt_bin": {b[2]: dev_person.get(b[2], 0) for b in ALT_BINS},
        "test_person_frames": n_test_person,
        "checks": checks,
        "blocks": {"total_flight": len(info), "test": sorted(test_blocks), "skipped_over_max": sorted(skipped),
                   "by_stratum": {f"{b}|{'person' if p else 'no_person'}": n for (b, p), n in
                                  sorted(Counter((v["bin"], v["person"]) for v in info.values()).items())},
                   "detail": {str(k): v for k, v in sorted(info.items())}},
        "table_flight_split_bin_person": table,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def cmd_build(args: argparse.Namespace) -> int:
    root = repo_root()
    doc, report = build(args.seed, read_manifest(), read_person_frames(root / PRELIM_REL),
                        (root / EVENTS_REL).read_text(encoding="utf-8"), args.test_min, args.test_max)
    out = Path(args.out) if args.out else root / SPLIT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(doc), encoding="utf-8", newline="\n")
    rep = Path(args.report) if args.report else root / REPORT_REL
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(dumps(report), encoding="utf-8", newline="\n")
    summary = {k: report[k] for k in ("counts", "excluded", "test_person_frames_by_alt_bin", "test_person_frames",
                                      "checks")}
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    bad = [k for k, v in report["checks"].items() if not v]
    if bad:
        print(f"LỖI: kiểm không đạt {bad} — báo Owner, không tự đổi cách chia", file=sys.stderr)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.camera.split", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="dựng data/real/split.json + results/data/split_report.json")
    b.add_argument("--seed", type=int, default=SEED_DEFAULT)
    b.add_argument("--out", help="đường dẫn split.json (mặc định data/real/split.json)")
    b.add_argument("--report", help="đường dẫn báo cáo (mặc định results/data/split_report.json)")
    b.add_argument("--test-min", type=int, default=TEST_MIN, help="số khung test tối thiểu cần đạt (plan: 300)")
    b.add_argument("--test-max", type=int, default=TEST_MAX, help="số khung test tối đa (plan: 400)")
    b.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return {"build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
