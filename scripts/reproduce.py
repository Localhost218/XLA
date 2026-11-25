"""Một lệnh tái lập mọi bảng + hình của báo cáo (phase 18 v1; phase 17 AI v2 thêm t7–t9, f4).

    python scripts/reproduce.py --check --json   # chấm lại dự đoán đã lưu (CPU) + dựng bảng/hình, so sha256 manifest
    python scripts/reproduce.py --write          # dựng lại + ghi results/reproduce/manifest.json (sau khi kết quả đổi)
    python scripts/reproduce.py --full           # in rồi chạy các bước GPU (huấn luyện lại, ~9 giờ GPU), xong thì --write
    python scripts/reproduce.py --full --dry-run # chỉ in danh sách bước GPU kèm giờ ước

`--check` làm hai việc, không cần GPU:
  1. chấm lại mọi CSV dự đoán đã lưu bằng `xla.eval.detection.evaluate` (boot_n=0: chỉ số điểm, không CI) và so mAP50 /
     mAP50-95 với số đã ghi trong JSON kết quả (lệch > 1e-6 là FAIL);
  2. dựng mọi bảng (`results/reproduce/tables/*.md`) và hình (`results/reproduce/fig/*.svg`) từ JSON kết quả vào một thư
     mục tạm, so sha256 với manifest; ảnh mẫu trong báo cáo (`assets`) cũng so sha256.
Bảng và hình trong `BAO_CAO_DO_AN.md` chỉ được lấy từ đây, không dán tay. Mã thoát: 0 đạt, 1 lệch, 2 thiếu dữ liệu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_REL = Path("results/reproduce")
MANIFEST = ROOT / OUT_REL / "manifest.json"
TOL = 1e-6
SPLITS = ("test", "neg", "devval")

# (tên, các CSV dự đoán, JSON đã ghi, đường tới dict metrics trong JSON, chọn ngưỡng F1 trên dev-val)
RESCORE: list[tuple[str, list[str], str, list[str], bool]] = [
    ("baseline_edge16", ["results/baseline/pred/edge16.csv"], "results/baseline/baseline.json",
     ["variants", "edge16", "metrics"], True),
    *[(f"e1_{s}", [f"results/e1/preds/{s}_{sp}.csv" for sp in SPLITS], "results/e1/e1.json",
       ["sets", s, "test", "metrics"], True) for s in ("d0", "d1", "d2", "d2sr")],
    *[(f"e2_{m}", [f"results/e2/preds/{m}_{sp}.csv" for sp in SPLITS], f"results/e2/eval/{m}_test.json",
       ["metrics"], True) for m in ("teacher", "r1", "r2")],
    ("e2_bundle_onnx", ["results/e2/preds/onnx_detector-v1.0.0_all.csv"], "results/e2/e2.json",
     ["bundle", "test", "metrics"], True),
    *[(f"e4_{d}_{v}", [f"results/e4/preds/{d}_{v}.csv"], "results/e4/e4.json", ["cells", d, v], False)
      for d in ("coco", "bundle") for v in ("raw", "classic", "net")],
]

# AI v2 (phase 09): CSV cả hệ từ IOT-CV, ba tầng. Ngưỡng vận hành của từng tầng đọc từ `op_threshold` trong score.json,
# không chọn lại trên dev-val (cả hệ không có CSV dev-val). (tên, CSV, JSON đã ghi, đường tới dict của tầng)
SYSTEM_REL = "results/v2/system/score.json"
V2_STAGES = ("detector", "verifier", "track")
RESCORE_V2: list[tuple[str, str, str, list[str]]] = [
    (f"v2_system_{st}", f"results/v2/system/preds/system_{st}_test.csv", SYSTEM_REL, ["test", "stages", st])
    for st in V2_STAGES]

# Bước GPU của --full: (lệnh, giờ GPU ước theo lần chạy 18–19/11/2026 trên RTX 4060 Laptop)
FULL_STEPS: list[tuple[str, float]] = [
    ("python -m xla.camera.profile build --split dev --json", 0.0),
    ("python -m xla.degrade.calibrate --split dev --budget 155 --workers 4", 0.5),
    ("python -m xla.degrade.realism_compare --json", 0.1),
    ("python -m xla.detect.baseline time", 0.1),
    *[(f"python -m xla.detect.train --config configs/train_e1.yaml --set {s} --json", 0.7)
      for s in ("d0", "d1", "d2", "d2sr")],
    ("python -m xla.detect.train --config configs/train_e1.yaml --score --json", 0.0),
    ("python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --teacher --json", 0.8),
    ("python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round 1 --json", 0.6),
    ("python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round 2 --json", 0.6),
    ("python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --score --json", 0.0),
    (("python -m xla.detect.export_onnx --weights D:/DevCache/runs/xla/e2/r1/weights/best_devval.pt "
     "--out models/release/detector-v1.0.0 --json"), 0.1),
    ("python -m xla.enhance.train_net train --config configs/restore.yaml --json", 0.7),
    ("python -m xla.enhance.export_onnx --ckpt <best.pt của run trên> --out models/release/enhancer-v1.0.0 --json", 0.0),
    ("python -m xla.eval.enhance_eval all --json", 0.3),
]


# -- tiện ích ----------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def dig(d: Any, keys: list[str]) -> Any:
    for k in keys:
        d = d[k]
    return d


def f3(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}".replace(".", ",")


def f1(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def ci(m: dict) -> str:
    lo, hi = (m.get("ci95") or [None, None])[:2]
    return f"{f3(m['value'])} [{f3(lo)}; {f3(hi)}]" if lo is not None else f3(m["value"])


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# -- 1. chấm lại dự đoán ----------------------------------------------------------------------
def _combine(parts: list[Path], out: Path) -> Path:
    lines: list[str] = []
    for p in parts:
        rows = p.read_text(encoding="utf-8").splitlines()
        lines += rows if not lines else rows[1:]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def rescore(tmp: Path) -> list[dict]:
    from xla.eval.detection import evaluate

    res = []
    for name, parts, ref, keys, f1dev in RESCORE:
        paths = [ROOT / p for p in parts]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            res.append({"name": name, "pass": False, "error": f"thiếu {missing}"})
            continue
        csv = paths[0] if len(paths) == 1 else _combine(paths, tmp / f"{name}_all.csv")
        got = evaluate(csv, "test", boot_n=0, f1_from_devval=f1dev)["metrics"]
        want = dig(load(ref), keys)
        diff = {k: round(abs(got[k]["value"] - want[k]["value"]), 9) for k in ("mAP50", "mAP50_95")}
        res.append({"name": name, "mAP50": got["mAP50"]["value"], "ref": ref,
                    "max_abs_diff": max(diff.values()), "pass": max(diff.values()) <= TOL})
    for name, part, ref, keys in RESCORE_V2:
        path = ROOT / part
        if not path.is_file():
            res.append({"name": name, "pass": False, "error": f"thiếu {path}"})
            continue
        want = dig(load(ref), keys)
        thr = want["op_threshold"]
        got = evaluate(path, "test", thresholds=[thr], boot_n=0, f1_from_devval=False)["metrics"]
        # cả hệ chấm thêm recall + báo nhầm ở ngưỡng vận hành: so luôn hai số đó, không chỉ mAP
        ks = ("mAP50", "mAP50_95", f"recall@{thr:g}", f"fp_per_neg_flight@{thr:g}")
        diff = {k: round(abs(got[k]["value"] - want["metrics"][k]["value"]), 9) for k in ks}
        res.append({"name": name, "mAP50": got["mAP50"]["value"], "ref": ref,
                    "max_abs_diff": max(diff.values()), "pass": max(diff.values()) <= TOL})
    return res


# -- 2. bảng ---------------------------------------------------------------------------------
DET_ROWS = [
    ("Mốc: YOLO11s COCO @1280 + cắt mép 16 px", "results/baseline/baseline.json", ["variants", "edge16"]),
    *[(f"E1 {lab}", "results/e1/e1.json", ["sets", s, "test"])
      for s, lab in (("d0", "D0 ảnh sạch"), ("d1", "D1 làm xấu ngẫu nhiên"), ("d2", "D2 bản sao camera"),
                     ("d2sr", "D2 + siêu phân giải"))],
    *[(f"E2 {lab}", f"results/e2/eval/{m}_test.json", [])
      for m, lab in (("teacher", "thầy YOLO11m"), ("r1", "trò vòng 1"), ("r2", "trò vòng 2"))],
    ("Bundle detector-v1.0.0 (ONNX, ship)", "results/e2/e2.json", ["bundle", "test"]),
]


def op_fp(d: dict) -> tuple[float, float]:
    """(ngưỡng vận hành chọn bằng F1 trên dev-val, FP/khung bay không người ở ngưỡng đó)."""
    m = d["metrics"]
    if "fp_per_neg_flight@op" in m:
        return d["conf_threshold"], m["fp_per_neg_flight@op"]["value"]
    thr = d["f1_threshold_from_devval"]["thr"]
    key = next(k for k in m if k.startswith("fp_per_neg_flight@") and abs(float(k.split("@")[1]) - thr) < 1e-9)
    return thr, m[key]["value"]


def t_detect() -> str:
    rows = []
    for lab, ref, keys in DET_ROWS:
        d = dig(load(ref), keys)
        m = d["metrics"]
        thr, fp = op_fp(d)
        rows.append([lab, ci(m["mAP50"]), f3(m["mAP50_95"]["value"]), f3(m["AR_small"]["value"]), f3(fp), f3(thr)])
    return table(["Model", "mAP50 [CI 95 %]", "mAP50-95", "AR nhỏ", "FP/khung bay không người", "Ngưỡng (F1 dev-val)"], rows)


def t_realism() -> str:
    rows = []
    for r in load("results/realism/d1_vs_d2.json")["rows"]:
        c, k = r["c2st"], r["kid"]
        rows.append([r["what"], f3(r["stats"]["total"]), f"{f3(c['mean'])} [{f3(c['ci_low'])}; {f3(c['ci_high'])}]",
                     f"{f3(k['mean'])} [{f3(k['ci_low'])}; {f3(k['ci_high'])}]"])
    return table(["Bộ làm xấu", "Lệch thống kê (tổng, nhỏ = giống)", "C2ST (0,5 = không phân biệt được)",
                  "KID ×100 (nhỏ = giống)"], rows)


def t_e4() -> str:
    e4 = load("results/e4/e4.json")
    rows = []
    for d, lab in (("coco", "YOLO11s COCO (có sẵn)"), ("bundle", "detector-v1.0.0 (đã fine-tune)")):
        cells = e4["cells"][d]
        pd = e4["paired_diff_vs_raw"][d]
        rows.append([lab, f3(cells["raw"]["mAP50"]["value"]),
                     *[f"{f3(cells[v]['mAP50']['value'])} ({'+' if pd[v]['mAP50']['diff'] >= 0 else ''}"
                       f"{f3(pd[v]['mAP50']['diff'])} [{f3(pd[v]['mAP50']['ci95'][0])}; "
                       f"{f3(pd[v]['mAP50']['ci95'][1])}])" for v in ("classic", "net")]])
    return table(["Bộ nhận diện", "Khung gốc", "Sau cổ điển (chênh [CI])", "Sau mạng (chênh [CI])"], rows)


def t_enhance() -> str:
    crit = load("results/enhance/enhance_eval.json")["criteria_branch_a"]
    mark = {"dat": "đạt", "khong_dat": "không đạt"}
    rows = [[c["text"], (f3(c["value"]) + (f" [{f3(c['ci95'][0])}; {f3(c['ci95'][1])}]" if c.get("ci95") else "")),
             mark.get(c["status"], c["status"])] for c in crit]
    return table(["Tiêu chí (mạng enhancer-v1.0.0, khung test)", "Giá trị [CI 95 %]", "Kết luận"], rows)


def t_ab() -> str:
    rows = []
    for ref, lab in (("results/ab/ab_score.json", "Lần 1, cả khung"), ("results/ab-zoom/ab_score.json", "Lần 2, phóng to 2×")):
        for comp, vs in (("net_vs_raw", "gốc"), ("net_vs_classic", "cổ điển")):
            c = load(ref)["comparisons"][comp]
            lo, hi = c["ci95_wilson"]
            rows.append([lab, f"mạng vs {vs}", f"{c['net_wins']}/{c['n']}", f"{f3(c['win_rate'])} [{f3(lo)}; {f3(hi)}]",
                         f3(c["criterion_min"])])
    return table(["Lượt chấm", "So sánh", "Mạng thắng", "Tỉ lệ [Wilson 95 %]", "Ngưỡng đạt"], rows)


BENCH_REL = "results/iotcv/bench-20261119-022705.json"


def t_iotcv() -> str:
    b = load(BENCH_REL)
    rows = []
    for key, lab in (("realtime", "Nhịp thật, có người xem Ảnh rõ"), ("realtime_detector_only", "Nhịp thật, chỉ detector"),
                     ("maxrate", "Hết tốc lực, hai model")):
        ms = b[key]["ms"]
        fps = b[key].get("processed_fps", b[key].get("fps"))
        rows.append([lab, *[f"{f1(ms[s]['p50'])} / {f1(ms[s]['p95'])}" if s in ms else "–"
                            for s in ("detector", "enhancer", "total")], f1(fps)])
    m = b["machine"]
    note = (f"\nĐiều kiện: GPU {m['gpu_temp_c']:.0f} °C, xung lúc nghỉ {m['gpu_clock_mhz']:.0f}/{m['gpu_clock_max_mhz']:.0f} MHz; "
            f"bản ghi {b['recording']}, khung {b['start']}–{b['start'] + 630}; bỏ khung {f1(100 * b['realtime']['dropped_frac'])} %.\n")
    return table(["Chế độ", "Detector p50 / p95 ms", "Enhancer p50 / p95 ms", "Tổng p50 / p95 ms", "Khung/s xử lý"],
                 rows) + note


# -- AI v2 (plan 261125-ai-v2): t7 nhánh B ba tầng, t8 A/B v2, t9 ms + giờ offline ----------------
# Đích của chỉ số §5 (`plans/reports/261125-brainstorm-cai-thien-ai-v2.md` §5) không nằm trong file kết quả nào trừ B1 (score.json)
# và B2 (holdout_weak.json); recall đích chỉ có ở thiết kế, nên ghi là hằng có nguồn.
RECALL_TARGET = 0.6
AB_V2 = "results/ab-v2"
AB_V2_ZOOM = "results/ab-v2-zoom"
HOLDOUT_WEAK_REL = "results/v2/system/holdout_weak.json"
IOTCV_V2_REL = "results/v2/iotcv/bench.json"
SR_EXPORT_REL = "results/v2/sr/export.json"
SR_BENCH_REL = "results/v2/sr/bench.json"
CALIBRATE_REL = "results/v2/system/calibrate.json"
VARIANT = {"raw": "gốc", "classic": "classic", "offline": "offline ×2"}
V2_INPUTS = (HOLDOUT_WEAK_REL, f"{AB_V2}/ab_score.json", f"{AB_V2}/decision.json", f"{AB_V2}/guard_fp.json", IOTCV_V2_REL,
             SR_EXPORT_REL, SR_BENCH_REL, CALIBRATE_REL)


def sgn(x: float) -> str:
    return f"+{f3(x)}" if x >= 0 else f3(x)


def diff_ci(d: dict) -> str:
    return f"{sgn(d['diff'])} [{sgn(d['ci95'][0])}; {sgn(d['ci95'][1])}]"


def at(m: dict, prefix: str) -> dict:
    """Chỉ số ở ngưỡng vận hành: khoá dạng `recall@0.35` (một ngưỡng mỗi tầng)."""
    (key,) = [k for k in m if k.startswith(prefix + "@")]
    return m[key]


def mark(ok: bool) -> str:
    return "đạt" if ok else "**không đạt**"


def t_v2_detect() -> str:
    s = load(SYSTEM_REL)
    b = s["bundles"]
    tr = b["detector"]["recommended_track"]
    labs = {"detector": f"1 detector `{b['detector']['bundle']}` (ONNX của v1.0.0)",
            "verifier": f"2 + `{b['verifier']['bundle']}` @{f3(b['verifier']['conf_threshold'])}",
            "track": f"3 + lọc track k{tr['k']}/n{tr['n']} (cả hệ)"}
    st = s["test"]["stages"]
    n = st["track"]["n"]
    rows = []
    for k in V2_STAGES:
        m = st[k]["metrics"]
        rows.append([labs[k], ci(m["mAP50"]), ci(at(m, "recall")), ci(at(m, "fp_per_neg_flight")),
                     f3(at(m, "fp_per_neg_indoor")["value"]), f3(st[k]["op_threshold"])])
    out = table(["Tầng (CSV của IOT-CV)", "mAP50 [CI 95 %]", "Recall @ ngưỡng [CI]",
                 f"Báo nhầm / khung bay trống ({n['negative_frames_flight']}) [CI]",
                 f"/ khung trong nhà ({n['negative_frames_indoor']})", "Ngưỡng"], rows)
    out += (f"\nTest v1: {n['frames']} khung ({n['frames_with_person']} có người, {n['gt_boxes']} người). "
            "CI 95 % bootstrap theo khối 30 s.\n\n")
    cmp_rows = []
    for ref, lab in (("v1", "`detector-v1.0.0` (v1, chỉ detector)"), ("edge16", "mốc COCO + cắt mép 16 px")):
        c = st["track"]["compare"][ref]["metrics"]
        cmp_rows.append([lab, *[f"{f3(c[k]['a'])} → {f3(c[k]['b'])}: {diff_ci(c[k])}"
                                for k in ("mAP50", "recall@op", "fp_per_neg_flight@op")]])
    out += table(["Mốc → cả hệ (tầng 3): hiệu [CI 95 %]", "mAP50", "Recall @ ngưỡng", "Báo nhầm / khung bay trống"],
                 cmp_rows)
    b1, ho, hw = s["b1"], s["holdout"], load(HOLDOUT_WEAK_REL)
    rec = at(st["track"]["metrics"], "recall")
    vs = b1["vs_edge16_map50"]
    crit = [
        ["B1: mAP50 cả hệ", ci(b1["map50_system_verifier"]), f"≥ {f3(b1['target'])}",
         mark(b1["map50_system_verifier"]["value"] >= b1["target"])],
        ["B1 tối thiểu: mAP50 và hiệu với mốc edge16", f"{f3(b1['map50_system_verifier']['value'])}; {diff_ci(vs)}",
         f"≥ {f3(b1['min'])} và cận dưới hiệu > 0",
         mark(b1["map50_system_verifier"]["value"] >= b1["min"] and vs["ci95"][0] > 0)],
        ["B2: báo nhầm / khung, holdout giữ kín (tầng 3)", ho["verdict"], f"≤ {f3(hw['target'])}",
         "**không đủ chứng minh**" if ho["n_frames"] == 0 else "–"],
        ["B2 trên holdout yếu (cùng buổi, agent duyệt)", hw["verdict"], f"≤ {f3(hw['target'])}",
         "**không đủ chứng minh**" if hw["n_frames"] == 0 else "–"],
        ["Recall @ ngưỡng vận hành", ci(rec), f"≥ {f3(RECALL_TARGET)}", mark(rec["value"] >= RECALL_TARGET)],
    ]
    return out + "\n" + table(["Chỉ số §5 (nhánh B)", "v2", "Đích", "Kết luận"], crit)


def t_v2_ab() -> str:
    sc = load(f"{AB_V2}/ab_score.json")
    dec = load(f"{AB_V2}/decision.json")
    rows = []
    for name, c in sc["comparisons"].items():
        lo, hi = c["ci95_wilson"]
        rows.append(["cả khung", f"{VARIANT[c['candidate']]} vs {VARIANT[c['baseline']]}", f"{c['wins']}/{c['n']}",
                     f"{f3(c['win_rate'])} [{f3(lo)}; {f3(hi)}]", "đạt" if c["passes"] else "**không đạt**"])
    if not (ROOT / AB_V2_ZOOM / "ab_score.json").is_file():
        rows.append(["phóng to 2×", "cả ba so sánh", "chưa chấm", "–", "không quyết định (luật chốt trước khi chấm)"])
    cons = sc["consistency"]
    out = table(["Lượt", "So sánh", "Bên thứ nhất thắng", "Tỉ lệ [Wilson 95 %]", "Cận dưới > 0,5"], rows)
    out += f"\nCặp lặp đổi bên: {cons['agree']}/{cons['n_repeats']} chọn như lần đầu.\n\n"
    g = load(f"{AB_V2}/guard_fp.json")
    gd = g["diff"]
    out += table(["Hàng rào báo nhầm (detector trên khung cal không người)", "Gốc", "Offline ×2", "Hiệu [CI 95 %]",
                  "Kết luận"],
                 [[f"{g['cal_raw']['n_frames']} khung, {gd['n_blocks']} khối, {gd['n_sessions']} phiên",
                   f"{g['cal_raw']['n_boxes']} hộp = {f3(g['cal_raw']['fp_per_frame'])}",
                   f"{g['cal_offline']['n_boxes']} hộp = {f3(g['cal_offline']['fp_per_frame'])}",
                   f"{sgn(gd['diff'])} [{sgn(gd['ci95'][0])}; {sgn(gd['ci95'][1])}]",
                   "qua" if dec["guard_fp"]["decision"] == "pass" else "**loại**"]])
    ship = "có" if dec["offline_ship"] else "**không**"
    return out + f"\nShip `enhancer-offline-v1.0.0`: {ship} ({dec['offline_reason']}).\n"


def _p(v: float | None) -> str:
    return "–" if v is None else f1(v)


def t_v2_ms() -> str:
    b = load(IOTCV_V2_REL)
    bv = b["bundles"]
    rows = []
    for mode, lab in (("realtime", "có người xem Ảnh rõ"), ("realtime_detector_only", "chỉ nhận diện")):
        for step in ("detector", "verifier", "enhancer", "total"):
            s = b["steps"][mode][step]
            if s["n_v1"] == 0 and s["n_v2"] == 0:
                continue
            d = None if s["v1_p95"] is None else round(s["v2_p95"] - s["v1_p95"], 2)
            cl = b["ceiling"][mode] if step == "total" else None
            rows.append([lab if step == "detector" else "", "tổng" if step == "total" else step,
                         f"{_p(s['v1_p50'])} / {_p(s['v1_p95'])}", f"{_p(s['v2_p50'])} / {_p(s['v2_p95'])}",
                         "–" if d is None else f"{d:+.2f}".replace(".", ","),
                         f"+{f1(cl['max_delta_ms'])}" if cl else "", mark(cl["pass"]) if cl else ""])
    out = table(["Luồng, nhịp thật", "Bước", "v1 p50 / p95 ms", "v2 p50 / p95 ms", "Hiệu p95", "Trần", "Kết luận"], rows)
    clk = [r["machine_during"]["clock_mhz"]["median"] for r in b["runs"]]
    tmp = [r["machine_during"]["temp_c"]["median"] for r in b["runs"]]
    drop = {c: [f1(100 * r["realtime"]["dropped_frac"]) for r in b["runs"] if r["config"] == c] for c in ("v1", "v2")}
    out += (f"\nv1 = `{bv['v1']['detector']['bundle']}` + `{bv['v1']['enhancer']['bundle']}`; v2 = "
            f"`{bv['v2']['detector']['bundle']}` + `{bv['v2']['verifier']['bundle']}` + `{bv['v2']['enhancer']['bundle']}`. "
            f"Bản ghi {b['recording']}, khung {b['start']}–{b['start'] + b['frames']}, {len(b['runs'])} lượt xen kẽ "
            f"{'-'.join(b['order'])}. Máy: xung GPU trung vị mỗi lượt {min(clk):.0f}–{max(clk):.0f} MHz (tối đa "
            f"{b['machine_start']['clock_max_mhz']:.0f}), {min(tmp):.0f}–{max(tmp):.0f} °C; bỏ khung có Ảnh rõ: "
            f"v1 {' · '.join(drop['v1'])} %, v2 {' · '.join(drop['v2'])} %.\n\n")
    ex, be = load(SR_EXPORT_REL), load(SR_BENCH_REL)
    spf, fps, mins = ex["s_per_frame"], be["recording_fps"]["fps"], be["flight_minutes"]
    hours = spf["mean"] * fps * mins * 60 / 3600
    out += table(["Ảnh rõ offline ×2 (`enhancer-offline-v1.0.0`, không ship)", "s / khung TB (p95)",
                  "Khung / s của bản ghi", f"Giờ GPU cho chuyến {mins} phút"],
                 [[f"ORT CUDA fp16, {spf['n']} khung dev", f"{f3(spf['mean'])} ({f3(spf['p95'])})", f"{f1(fps)}",
                   f"{f3(hours)}"]])
    return out


TABLES = {"t1_nhan_dien.md": t_detect, "t2_do_giong_that.md": t_realism, "t3_tieu_chi_anh_ro.md": t_enhance,
          "t4_ab.md": t_ab, "t5_e4.md": t_e4, "t6_iotcv_ms.md": t_iotcv,
          "t7_v2_nhan_dien.md": t_v2_detect, "t8_v2_ab.md": t_v2_ab, "t9_v2_ms.md": t_v2_ms}


# -- 3. hình (SVG viết tay: tất định, không phụ thuộc font/phiên bản thư viện) ---------------------
def svg_bars(title: str, items: list[tuple[str, float, float, float]], vmax: float | None, ref: float | None = None,
             ref_label: str = "") -> str:
    """Thanh ngang có râu CI. items = (nhãn, giá trị, ci_lo, ci_hi)."""
    w, left, bar_h, gap, top = 760, 300, 22, 12, 46
    h = top + len(items) * (bar_h + gap) + 30
    vmax = vmax or max(hi for *_, hi in items) * 1.12  # chừa chỗ cho nhãn số sau râu CI
    sx = (w - left - 60) / vmax
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           'font-family="sans-serif" font-size="13">'),
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           f'<text x="12" y="24" font-size="15" font-weight="bold" fill="#111">{title}</text>']
    for i, (lab, v, lo, hi) in enumerate(items):
        y = top + i * (bar_h + gap)
        out += [f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" fill="#222">{lab}</text>',
                f'<rect x="{left}" y="{y}" width="{v * sx:.1f}" height="{bar_h}" fill="#4a6fa5"/>',
                (f'<line x1="{left + lo * sx:.1f}" y1="{y + bar_h / 2}" x2="{left + hi * sx:.1f}" y2="{y + bar_h / 2}" '
                'stroke="#111" stroke-width="1.5"/>'),
                f'<text x="{left + hi * sx + 6:.1f}" y="{y + 16}" fill="#111">{f3(v)}</text>']
    if ref is not None:
        x = left + ref * sx
        out += [f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{h - 22}" stroke="#c0392b" stroke-dasharray="5,4"/>',
                f'<text x="{x + 4:.1f}" y="{h - 8}" fill="#c0392b">{ref_label}</text>']
    out.append("</svg>")
    return "\n".join(out) + "\n"


def f_detect() -> str:
    items = []
    for lab, ref, keys in DET_ROWS:
        m = dig(load(ref), keys)["metrics"]["mAP50"]
        items.append((lab.split(" (")[0].replace("Mốc: ", "Mốc "), m["value"], *m["ci95"]))
    base = items[0][1]
    return svg_bars("mAP50 trên 198 khung test thật (râu = CI 95 % bootstrap theo khối)", items, 0.6, base, "mốc")


def f_ab() -> str:
    items = []
    for ref, lab in (("results/ab/ab_score.json", "L1"), ("results/ab-zoom/ab_score.json", "L2 phóng to")):
        for comp, vs in (("net_vs_raw", "gốc"), ("net_vs_classic", "cổ điển")):
            c = load(ref)["comparisons"][comp]
            items.append((f"{lab}: mạng vs {vs}", c["win_rate"], *c["ci95_wilson"]))
    return svg_bars("A/B mù: tỉ lệ lượt người chấm chọn ảnh của mạng", items, 1.0, 0.5, "50 %")


def f_e4() -> str:
    e4 = load("results/e4/e4.json")
    items = []
    for d, lab in (("coco", "COCO"), ("bundle", "fine-tune")):
        for v, vl in (("raw", "gốc"), ("classic", "cổ điển"), ("net", "mạng")):
            m = e4["cells"][d][v]["mAP50"]
            items.append((f"{lab} × {vl}", m["value"], *m["ci95"]))
    return svg_bars("E4: mAP50 khi đưa khung đã làm rõ vào bộ nhận diện", items, None)


def svg_curve(title: str, series: list[tuple[str, str, list[tuple[float, float]]]], xstep: float, ystep: float,
              xlab: str, ylab: str, vline: tuple[float, str] | None, point: tuple[float, float, str] | None) -> str:
    """Đường gấp khúc nhiều chuỗi trên trục tuyến tính, vạch chia đều `xstep`/`ystep`. series = (nhãn, màu, [(x, y)])."""
    import math

    nx = math.ceil(max(x for *_, pts in series for x, _ in pts) / xstep - 1e-9) + 1
    ny = math.ceil(max(y for *_, pts in series for _, y in pts) / ystep - 1e-9) + 1
    xmax, ymax = nx * xstep, ny * ystep
    w, h, left, right, top, bot = 760, 440, 70, 190, 46, 56
    pw, ph = w - left - right, h - top - bot
    sx, sy = pw / xmax, ph / ymax

    def px(x: float, y: float) -> tuple[float, float]:
        return left + x * sx, top + ph - y * sy

    def f(v: float) -> str:
        return f"{v:.2f}".replace(".", ",")

    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            'font-family="sans-serif" font-size="13">'),
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           f'<text x="12" y="24" font-size="15" font-weight="bold" fill="#111">{title}</text>',
           f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="#111"/>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="#111"/>']
    for i in range(nx + 1):
        x, _ = px(i * xstep, 0)
        out.append(f'<text x="{x:.1f}" y="{top + ph + 18}" text-anchor="middle" fill="#222">{f(i * xstep)}</text>')
    for i in range(ny + 1):
        _, y = px(0, i * ystep)
        out += [f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#222">{f(i * ystep)}</text>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#e4e4e4"/>']
    out += [f'<text x="{left + pw / 2:.1f}" y="{h - 12}" text-anchor="middle" fill="#111">{xlab}</text>',
            (f'<text x="16" y="{top + ph / 2:.1f}" text-anchor="middle" fill="#111" '
             f'transform="rotate(-90 16 {top + ph / 2:.1f})">{ylab}</text>')]
    if vline is not None:
        x, _ = px(vline[0], 0)
        out += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ph}" stroke="#c0392b" stroke-dasharray="5,4"/>',
                f'<text x="{x + 4:.1f}" y="{top + 12}" fill="#c0392b">{vline[1]}</text>']
    for i, (lab, color, pts) in enumerate(series):
        path = " ".join(f"{a:.1f},{b:.1f}" for a, b in (px(x, y) for x, y in pts))
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        out += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" fill="{color}"/>' for a, b in (px(x, y) for x, y in pts)]
        ly = top + 10 + i * 22
        out += [f'<line x1="{left + pw + 16}" y1="{ly}" x2="{left + pw + 40}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{left + pw + 46}" y="{ly + 4}" fill="#111">{lab}</text>']
    if point is not None:
        a, b = px(point[0], point[1])
        out += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" fill="none" stroke="#111" stroke-width="2"/>',
                f'<text x="{a + 10:.1f}" y="{b - 10:.1f}" fill="#111">{point[2]}</text>']
    out.append("</svg>")
    return "\n".join(out) + "\n"


F4_SERIES = ((1, 1, "#4a6fa5"), (2, 3, "#e08a2c"), (3, 3, "#5b8c5a"))


def f_v2_curve() -> str:
    """Báo nhầm trên cal ↔ recall devpool khi quét ngưỡng verifier, mỗi đường một cấu hình track (max_gap 1)."""
    c = load(CALIBRATE_REL)
    rows = c["rows"]
    series = []
    for k, n, color in F4_SERIES:
        sel = sorted((r for r in rows if (r["k"], r["n"], r["max_gap"]) == (k, n, 1)), key=lambda r: r["thr"])
        pts = [(r["cal_fp_per_frame"], r["recall"]) for r in sel]
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]  # ngưỡng liền nhau cùng điểm: vẽ một lần
        series.append((f"track k{k}/n{n}", color, pts))
    ch = c["selection"]["chosen"]
    from xla.eval.system import D7_CAL_MAX as lim  # SSOT của luật chọn D7 (cùng hằng vẽ calibrate_curve.png)
    return svg_curve(f"Quét ngưỡng verifier: báo nhầm trên {ch['n_cal']} khung cal ↔ recall {ch['n_gt']} người devpool",
                     series, 0.02, 0.05, "báo nhầm / khung cal (không người)", "recall devpool (lạc quan)",
                     (lim, f"cal ≤ {f3(lim)}"), (ch["cal_fp_per_frame"], ch["recall"],
                                                 f"chọn: ngưỡng {f3(ch['thr'])}, k{ch['k']}/n{ch['n']}"))


FIGS = {"f1_nhan_dien.svg": f_detect, "f2_ab.svg": f_ab, "f3_e4.svg": f_e4, "f4_v2_bao_nham_recall.svg": f_v2_curve}


def ab_example() -> dict[str, str]:
    """Ảnh mẫu cho báo cáo: khung đầu tiên (thứ tự khoá A/B) có người và có cả hai phép so."""
    key = load("results/ab/ab_key.json")
    by_frame: dict[str, dict[str, str]] = {}
    for p in key["pairs"]:
        if p["has_person"] and p["repeat_of"] is None:
            f = by_frame.setdefault(p["frame"], {})
            f[p["left"]], f[p["right"]] = p["left_img"], p["right_img"]
    # khoá A/B không ghép đủ ba biến thể trên một khung: lấy khung có nhiều biến thể nhất (thứ tự khoá khi hoà)
    frame = max(by_frame, key=lambda f: len(by_frame[f]))
    return {"frame": frame, **{k: f"results/ab/{v}" for k, v in sorted(by_frame[frame].items())}}


# -- dựng + so --------------------------------------------------------------------------------
def build(out_dir: Path) -> dict[str, str]:
    hashes = {}
    for sub, items in (("tables", TABLES), ("fig", FIGS)):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        for name, fn in items.items():
            p = out_dir / sub / name
            p.write_text(fn(), encoding="utf-8", newline="\n")
            hashes[(OUT_REL / sub / name).as_posix()] = sha256(p)
    return hashes


def assets() -> dict[str, str]:
    ex = ab_example()
    rels = [v for k, v in ex.items() if k != "frame"] + [BENCH_REL]
    return {r: sha256(ROOT / r) for r in rels}


REPORT = ROOT / "BAO_CAO_DO_AN.md"
BLOCK_RE = r"(<!-- bang: (results/reproduce/tables/[\w.-]+\.md) -->\n)(.*?)(<!-- /bang -->)"


def sync_report() -> int:
    """Chép bảng đã dựng vào các khối `<!-- bang: <file> -->...<!-- /bang -->` của báo cáo. Trả số khối."""
    import re

    if not REPORT.is_file():
        return 0
    text = REPORT.read_text(encoding="utf-8")
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        return m.group(1) + (ROOT / m.group(2)).read_text(encoding="utf-8") + m.group(4)

    REPORT.write_text(re.sub(BLOCK_RE, sub, text, flags=re.DOTALL), encoding="utf-8", newline="\n")
    return n


def manifest_doc(outputs: dict[str, str]) -> dict:
    return {
        "format": "xla-reproduce-v1",
        "note": "sha256 của mọi bảng/hình báo cáo dùng; dựng lại bằng scripts/reproduce.py --write",
        "inputs": sorted({p for _, parts, ref, _, _ in RESCORE for p in [*parts, ref]}
                         | {p for _, part, ref, _ in RESCORE_V2 for p in (part, ref)} | set(V2_INPUTS)),
        "rescore_tolerance": TOL,
        "outputs": outputs,
        "assets": assets(),
        "ab_example": ab_example(),
        "full_steps": [{"cmd": c, "gpu_hours": h} for c, h in FULL_STEPS],
    }


def check() -> dict:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scores = rescore(tmp)
        built = build(tmp / "out")
    if not MANIFEST.is_file():
        return {"pass": False, "error": f"thiếu {MANIFEST} (chạy --write)", "rescore": scores}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out_diff = sorted(k for k in set(built) | set(man["outputs"]) if built.get(k) != man["outputs"].get(k))
    on_disk = sorted(k for k, v in man["outputs"].items() if not (ROOT / k).is_file() or sha256(ROOT / k) != v)
    cur_assets = assets()
    asset_diff = sorted(k for k in set(cur_assets) | set(man["assets"]) if cur_assets.get(k) != man["assets"].get(k))
    ok = all(s["pass"] for s in scores) and not out_diff and not on_disk and not asset_diff
    return {"pass": ok, "rescore": scores, "outputs_mismatch": out_diff, "committed_stale": on_disk,
            "assets_mismatch": asset_diff, "n_outputs": len(built)}


def full(dry: bool) -> int:
    total = sum(h for _, h in FULL_STEPS)
    print(f"{len(FULL_STEPS)} bước GPU, ước {total:.1f} giờ GPU (RTX 4060 Laptop, 18–19/11/2026)")
    for cmd, h in FULL_STEPS:
        print(f"  [{h:.1f} h] {cmd}")
        if not dry:
            if "<" in cmd:
                print("    bỏ qua: cần điền checkpoint của bước trước", file=sys.stderr)
                return 2
            rc = subprocess.run(cmd.split(), cwd=ROOT, check=False).returncode
            if rc != 0:
                print(f"    lỗi mã {rc}, dừng", file=sys.stderr)
                return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="--full: chỉ in danh sách bước")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.full:
        return full(a.dry_run)
    if a.write:
        outputs = build(ROOT / OUT_REL)
        MANIFEST.write_text(json.dumps(manifest_doc(outputs), ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                            newline="\n")
        print(f"ghi {MANIFEST.relative_to(ROOT)}: {len(outputs)} bảng/hình; chép {sync_report()} bảng vào báo cáo")
        return 0
    res = check()
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        for s in res.get("rescore", []):
            print(f"{'OK ' if s['pass'] else 'LỆCH'} chấm lại {s['name']}: {s.get('mAP50', s.get('error'))}")
        print("ĐẠT" if res["pass"] else f"KHÔNG ĐẠT: {json.dumps({k: v for k, v in res.items() if k != 'rescore'}, ensure_ascii=False)}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
