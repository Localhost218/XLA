"""Phase 07: hiệu chỉnh tham số D2 cho thống kê mức thấp khớp khung **dev** thật.

Hàm mục tiêu = ``xla.eval.lowlevel.stat_distance(...)["total"]`` (tổng W1/IQR của phase 06 bước 2, trọng số bằng nhau)
giữa khung dev thật và D2 sinh từ ``sources_n`` đơn vị nguồn cố định. Mọi ứng viên dùng CÙNG đơn vị và CÙNG dòng số
ngẫu nhiên (``build.rng_for``), nên mục tiêu là hàm tất định của tham số. Tối ưu: CMA-ES (``cma``) trên không gian chuẩn
hoá [0, 1] với cận ghi ở khối ``calibration.params`` của ``configs/degrade_d2.yaml``; CPU, cả một thế hệ chạy song song.

CLI (AX):
    python -m xla.degrade.calibrate --split dev --budget N --json [--workers 16] [--dry-run]
    python -m xla.degrade.calibrate --realsr-gains 0.5,1,1.5,2 --json    # E3: chọn hệ số Real-SR, cùng mục tiêu
Exit: 0 khoảng cách giảm (đã ghi tham số tốt nhất vào config, tăng ``params_version``, trừ ``--dry-run``); 1 không giảm;
2 lỗi đầu vào. Đường hội tụ + khung đã dùng: ``results/realism/calibration.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from xla import paths
from xla.degrade import build, d2_camera, sources

CROP_LEFT = 16
NLF_BINS = 8
RESULT_REL = Path("results") / "realism" / "calibration.json"


# ----------------------------------------------------------------------------- không gian tham số

def get_key(cfg: dict, dotted: str):
    cur = cfg
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def set_key(cfg: dict, dotted: str, value) -> None:
    *head, last = dotted.split(".")
    cur = cfg
    for k in head:
        cur = cur[k]
    cur[last] = value


def resolve_space(cfg: dict, profile: dict) -> list[dict]:
    """Khối ``calibration.params`` → danh sách tham số có ``lo``/``hi``/``x0`` cụ thể.

    ``mode: center``: tham số là tâm của khoảng [a, b] trong config, nửa bề rộng giữ nguyên (cắt ≥ 0).
    ``lo_from_profile``: cận dưới lấy từ trường hồ sơ (hệ số WB ISP, v1.2.0: cận dưới thật). Điểm khởi đầu luôn là giá
    trị hiện có trong config, kẹp vào [lo, hi] (vòng 1, 18/11, config ghi 2,2 nhưng khởi đầu ở cận hồ sơ theo lệnh Owner:
    ghi ở ``rounds`` của ``calibration.json``)."""
    out = []
    for s in cfg["calibration"]["params"]:
        s = dict(s)
        cur = get_key(cfg, s["key"])
        if s.get("mode") == "center":
            s["half_width"] = (float(cur[1]) - float(cur[0])) / 2
            s["x0"] = (float(cur[0]) + float(cur[1])) / 2
        else:
            s["x0"] = float(cur)
        if "lo_from_profile" in s:
            s["lo"] = float(d2_camera.get_field(profile, s["lo_from_profile"]))
        s["x0"] = float(np.clip(s["x0"], s["lo"], s["hi"]))
        if not s["lo"] < s["hi"]:
            raise ValueError(f"tham số {s['name']}: cận {s['lo']}..{s['hi']} không hợp lệ")
        out.append(s)
    return out


def apply_values(cfg: dict, space: list[dict], values: list[float]) -> dict:
    new = copy.deepcopy(cfg)
    for s, v in zip(space, values, strict=True):
        v = float(v)
        if s.get("mode") == "center":
            hw = s["half_width"]
            set_key(new, s["key"], [round(max(v - hw, 0.0), 4), round(max(v + hw, 0.0), 4)])
        else:
            set_key(new, s["key"], round(v, 4))
    return new


def to_unit(space: list[dict], values) -> np.ndarray:
    return np.array([(v - s["lo"]) / (s["hi"] - s["lo"]) for s, v in zip(space, values, strict=True)])


def from_unit(space: list[dict], u) -> list[float]:
    return [s["lo"] + float(np.clip(x, 0, 1)) * (s["hi"] - s["lo"]) for s, x in zip(space, u, strict=True)]


# ----------------------------------------------------------------------------- đo (giống hệt xla.eval.lowlevel.measure_set)

def measure_bytes(data: bytes, is_jpeg: bool = True) -> dict:
    from xla.eval.lowlevel import default_measure

    m = default_measure()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("không giải mã được ảnh")
    return {"scalars": m.measure_image(img, crop_left=CROP_LEFT),
            "nlf": m.nlf(img, crop_left=CROP_LEFT, n_bins=NLF_BINS),
            "psd": m.psd_radial(img, crop_left=CROP_LEFT),
            "header": m.read_jpeg_header(data) if is_jpeg else None}


def aggregate(rows: list[dict]) -> dict:
    """Gộp số đo từng ảnh thành dạng ``measure_set`` trả về (để đưa vào ``stat_distance``)."""
    scalars: dict[str, list[float]] = {}
    nlf_vals: dict[int, list[float]] = {}
    psd_vals: dict[float, list[float]] = {}
    bin_w = 256.0 / NLF_BINS
    for r in rows:
        for k, v in r["scalars"].items():
            scalars.setdefault(k, []).append(float(v))
        for level, sigma in r["nlf"]:
            if math.isfinite(level) and math.isfinite(sigma):
                nlf_vals.setdefault(min(int(level // bin_w), NLF_BINS - 1), []).append(float(sigma))
        for f, power in r["psd"]:
            if math.isfinite(f) and math.isfinite(power) and power > 0:
                psd_vals.setdefault(round(float(f), 4), []).append(math.log10(power))
    return {"n_images": len(rows), "scalars": scalars, "nlf": nlf_vals, "psd": psd_vals,
            "headers": [r["header"] for r in rows]}


def _measure_file(p: str) -> dict:
    return measure_bytes(Path(p).read_bytes(), Path(p).suffix.lower() in (".jpg", ".jpeg"))


# ----------------------------------------------------------------------------- worker sinh + đo

_W: dict = {}


def _init(state: dict) -> None:
    cv2.setNumThreads(1)      # mỗi tiến trình một luồng: 16 tiến trình × luồng cv2 mặc định là tranh nhau CPU
    _W.clear()
    _W.update(state)
    if state.get("pool_needed"):
        from xla.degrade import realsr

        _W["noise_pool"] = realsr.load_pool()


def _render_measure(args: tuple) -> list[tuple[int, dict]]:
    cand, cfg, chunk, set_name = args
    ctx = dict(_W, d2=cfg)
    out = []
    for item, unit in chunk:
        r = build.render_unit(set_name, item, unit, _W["seed"], ctx)
        if "skipped" not in r:
            out.append((cand, measure_bytes(r["data"])))
    return out


class Evaluator:
    """Giữ pool tiến trình + số đo dev; ``__call__(list cfg)`` → list (tổng, chi tiết)."""

    def __init__(self, real_meas: dict, units: list, ctx: dict, workers: int, set_name: str = "d2"):
        self.real, self.units, self.set_name = real_meas, units, set_name
        state = dict(ctx, pool_needed=set_name == "d2sr")
        state.pop("noise_pool", None)
        build.worker_env()
        self.ex = ProcessPoolExecutor(workers, initializer=_init, initargs=(state,))
        self.chunk = max(1, len(units) // (workers * 2))

    def __call__(self, cfgs: list[dict]) -> list[dict]:
        from xla.eval.lowlevel import stat_distance

        jobs = [(c, cfg, self.units[i:i + self.chunk], self.set_name) for c, cfg in enumerate(cfgs)
                for i in range(0, len(self.units), self.chunk)]
        rows: list[list[dict]] = [[] for _ in cfgs]
        for part in self.ex.map(_render_measure, jobs):
            for c, m in part:
                rows[c].append(m)
        return [stat_distance(self.real, aggregate(r)) for r in rows]

    def close(self) -> None:
        self.ex.shutdown()


# ----------------------------------------------------------------------------- dev thật

def dev_frames(split: str) -> tuple[list[Path], dict]:
    from xla.eval.patches import split_images

    split_file = paths.repo_root() / "data" / "real" / "split.json"
    frames = split_images(split_file, split)
    test = {Path(p).name for p in split_images(split_file, "test")}
    leak = sorted({p.name for p in frames} & test)
    if leak:
        raise ValueError(f"khung hiệu chỉnh trùng test: {leak[:5]}")
    return frames, {"split_file": "data/real/split.json",
                    "split_file_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest()}


def measure_real(frames: list[Path], workers: int) -> dict:
    """Số đo khung thật, cache theo sha256 danh sách khung + mã đo (``xla/camera/measure.py``)."""
    code = (paths.repo_root() / "xla" / "camera" / "measure.py").read_bytes()
    key = hashlib.sha256("\n".join(sorted(p.name for p in frames)).encode() + code).hexdigest()[:16]
    cache = paths.cache_dir() / "degrade" / f"real_meas_{key}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        build.worker_env()
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_measure_file, [str(p) for p in frames], chunksize=8))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
    for r in rows:
        r["nlf"] = [tuple(x) for x in r["nlf"]]
        r["psd"] = [tuple(x) for x in r["psd"]]
    return aggregate(rows)


# ----------------------------------------------------------------------------- ghi config

def set_yaml_value(text: str, dotted: str, value) -> str:
    """Thay giá trị một khoá ``khối.khoá`` (hai cấp) trong YAML, giữ chú thích và thứ tự dòng."""
    block, key = dotted.split(".")
    lines = text.split("\n")
    in_block = False
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(block)}:", line):
            in_block = True
            continue
        if in_block and line and not line.startswith(" ") and not line.startswith("#"):
            break
        m = re.match(rf"^(  {re.escape(key)}:\s*)([^#]*?)(\s*(#.*)?)$", line) if in_block else None
        if m:
            v = "[" + ", ".join(_fmt(x) for x in value) + "]" if isinstance(value, list) else _fmt(value)
            lines[i] = m.group(1) + v + m.group(3)
            return "\n".join(lines)
    raise KeyError(f"không thấy {dotted} trong YAML")


def set_top_value(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf"^{re.escape(key)}:[ \t]*[^\s#]+", f"{key}: {value}", text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise KeyError(f"không thấy khoá gốc {key}")
    return new


def _fmt(x) -> str:
    return f"{float(x):.4g}" if isinstance(x, float) else str(x)


def write_config(path: Path, space: list[dict], best_cfg: dict, params_version: int) -> None:
    text = path.read_text(encoding="utf-8")
    for s in space:
        text = set_yaml_value(text, s["key"], get_key(best_cfg, s["key"]))
    text = set_top_value(text, "params_version", str(params_version))
    text = set_top_value(text, "version", f'"0.{params_version}.0"')
    path.write_text(text, encoding="utf-8", newline="\n")
    d2_camera.load_config(path)  # đọc lại được


# ----------------------------------------------------------------------------- chạy

def _context(profile_path: Path, sources_cfg: Path, d2_path: Path, seed: int, n_units: int) -> tuple:
    src = sources.load_config(sources_cfg)
    items = sources.select(src)
    units = build.plan_units(items, int(src["geometry"]["windows_per_image"]), seed, None)[:n_units]
    ctx = build.make_context(["d2"], profile_path, src, Path(), d2_path)
    ctx["seed"] = seed
    return ctx, units


def _summary_items(res: dict) -> dict:
    return {k: round(v["value"], 4) for k, v in res["items"].items() if math.isfinite(v["value"])}


def checkpoint_path(base: dict, space: list[dict], units: list, profile: dict, seed: int, split: str) -> Path:
    """Checkpoint CMA-ES theo khoá (config gốc, không gian, đơn vị, hồ sơ, seed, split): đổi gì là khoá mới."""
    key = json.dumps({"cfg": base, "space": space, "units": [u for _, u in units], "seed": seed, "split": split,
                      "profile": [profile.get("profile_id"), profile.get("profile_version")]}, sort_keys=True,
                     default=str)
    return paths.cache_dir() / "degrade" / f"calib_ckpt_{hashlib.sha256(key.encode()).hexdigest()[:16]}.pkl"


def _save_ckpt(path: Path, state: dict) -> None:
    import pickle

    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(state))
    tmp.replace(path)                      # thay nguyên tử: ngắt giữa chừng không làm hỏng checkpoint cũ


def run_cma(ev: Evaluator, base: dict, space: list[dict], budget: int, sigma0: float, seed: int,
            log, ckpt: Path | None = None, progress: Path | None = None) -> tuple[list[dict], dict, dict, float]:
    """CMA-ES; sau lượt "trước" và sau MỖI lượt ghi ``ckpt`` (trạng thái CMA, ứng viên của thế hệ đang dở, đường hội
    tụ) và ``progress`` (jsonl, một dòng mỗi lượt). Có ``ckpt`` cũ cùng khoá thì chạy tiếp từ đó, không đánh giá lại lượt
    đã xong. Trả (đường hội tụ, tốt nhất, trước, giây)."""
    import pickle

    import cma

    names = [s["name"] for s in space]
    if ckpt is not None and ckpt.exists():
        st = pickle.loads(ckpt.read_bytes())
        es, hist, best, before, gen, secs = st["es"], st["hist"], st["best"], st["before"], st["gen"], st["secs"]
        pending = st.get("pending")
        es.opts.set({"maxfevals": budget - 1})  # ngân sách có thể khác lần trước
        log(f"chạy tiếp từ checkpoint {ckpt.name}: {len(hist)} lượt, thế hệ {gen}, tốt nhất {best['total']:.4f}")
    else:
        t0 = time.perf_counter()
        x0 = [s["x0"] for s in space]
        before = ev([apply_values(base, space, x0)])[0]
        hist = [{"eval": 1, "gen": 0, "values": dict(zip(names, x0, strict=True)), "total": before["total"]}]
        best = {"total": before["total"], "values": x0, "res": before}
        es = cma.CMAEvolutionStrategy(to_unit(space, x0).tolist(), sigma0,
                                      {"bounds": [0, 1], "seed": seed, "maxfevals": budget - 1, "verbose": -9})
        gen, secs = 0, time.perf_counter() - t0
        pending = None
        log(f"trước: {before['total']:.4f}")
        if progress is not None:
            progress.write_text(json.dumps(hist[0]) + "\n", encoding="utf-8", newline="\n")

    def save() -> None:
        if ckpt is not None:
            _save_ckpt(ckpt, {"es": es, "hist": hist, "best": best, "before": before, "gen": gen, "secs": secs,
                              "pending": pending})

    save()
    while pending is not None or (not es.stop() and len(hist) < budget):
        if pending is None:                       # thế hệ mới: hỏi CMA, lưu ứng viên trước khi đánh giá
            gen += 1
            xs = es.ask()[:budget - len(hist)]
            pending = {"gen": gen, "xs": [[float(v) for v in x] for x in xs], "full": len(xs) == es.popsize,
                       "totals": []}
            save()
        gen = pending["gen"]
        for x in pending["xs"][len(pending["totals"]):]:   # chịu ngắt: checkpoint sau MỖI lượt
            t0 = time.perf_counter()
            v = from_unit(space, x)
            r = ev([apply_values(base, space, v)])[0]
            pending["totals"].append(r["total"])
            h = {"eval": len(hist) + 1, "gen": gen,
                 "values": {s["name"]: round(xv, 5) for s, xv in zip(space, v, strict=True)}, "total": r["total"]}
            hist.append(h)
            if r["total"] < best["total"]:
                best = {"total": r["total"], "values": v, "res": r}
            secs += time.perf_counter() - t0
            if progress is not None:
                with progress.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(h) + "\n")
            save()
        full, totals = pending["full"], pending["totals"]
        if full:
            es.tell([np.array(x) for x in pending["xs"]], totals)
        pending = None
        save()
        log(f"thế hệ {gen}: tốt nhất thế hệ {min(totals):.4f}, tốt nhất {best['total']:.4f} ({len(hist)} lượt)")
        if not full:
            break
    return hist, best, before, secs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.degrade.calibrate", description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="dev", choices=["dev"], help="khung thật để khớp; test giữ riêng cho bước 4")
    ap.add_argument("--budget", type=int, default=120, help="số lượt đánh giá mục tiêu (gồm lượt 'trước')")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sources-n", type=int, default=None, help="số đơn vị nguồn (mặc định calibration.sources_n)")
    ap.add_argument("--seed", type=int, default=build.DEFAULT_SEED)
    ap.add_argument("--profile", type=Path, default=paths.repo_root() / "configs" / "camera_profile.yaml")
    ap.add_argument("--sources", type=Path, default=paths.repo_root() / "configs" / "sources.yaml")
    ap.add_argument("--d2-config", type=Path, default=paths.repo_root() / "configs" / "degrade_d2.yaml")
    ap.add_argument("--out", type=Path, default=paths.repo_root() / RESULT_REL)
    ap.add_argument("--realsr-gains", default="", help="E3: đánh giá các hệ số Real-SR (cách dấu phẩy), ghi gain tốt nhất")
    ap.add_argument("--dry-run", action="store_true", help="không ghi config")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[calibrate] {msg}", file=sys.stderr, flush=True)

    try:
        base = d2_camera.load_config(a.d2_config)
        n_units = a.sources_n or int(base["calibration"]["sources_n"])
        ctx, units = _context(a.profile, a.sources, a.d2_config, a.seed, n_units)
        frames, split_meta = dev_frames(a.split)
        t0 = time.perf_counter()
        real = measure_real(frames, a.workers)
        log(f"đo {len(frames)} khung {a.split}: {time.perf_counter() - t0:.0f} s")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(json.dumps({"error": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return 2
    if a.realsr_gains:
        return _run_realsr(a, base, real, units, ctx, log)

    space = resolve_space(base, ctx["profile"])
    ckpt = checkpoint_path(base, space, units, ctx["profile"], a.seed, a.split)
    progress = ckpt.with_suffix(".progress.jsonl")
    log(f"checkpoint: {ckpt}")
    ev = Evaluator(real, units, ctx, a.workers)
    try:
        hist, best, before, secs = run_cma(ev, base, space, a.budget, float(base["calibration"]["sigma0"]), a.seed,
                                           log, ckpt, progress)
    finally:
        ev.close()
    best_cfg = apply_values(base, space, best["values"])
    improved = best["total"] < before["total"]
    old_pv = int(base.get("params_version", 1))
    new_pv = old_pv + 1 if improved and not a.dry_run else old_pv
    out = {
        "phase": "07",
        "measured": time.strftime("%Y-%m-%d"),
        "objective": "xla.eval.lowlevel.stat_distance total (W1/IQR thật, trọng số bằng nhau), D2 vs khung thật",
        "optimizer": {"name": "CMA-ES (cma)", "space": "chuẩn hoá [0,1], cận theo calibration.params",
                      "sigma0": base["calibration"]["sigma0"], "seed": a.seed, "budget": a.budget,
                      "evals": len(hist), "cpu_seconds_wall": round(secs, 1), "workers": a.workers},
        "profile": {"path": "configs/camera_profile.yaml", "profile_version": ctx["profile"].get("profile_version"),
                    "profile_id": ctx["profile"].get("profile_id")},
        "real": {"split": a.split, **split_meta, "n_frames": len(frames), "frames_used": [p.name for p in frames]},
        "synth": {"set": "d2", "seed": a.seed, "n_units": len(units), "units": [u for _, u in units]},
        "space": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()} for s in space],
        "before": {"params_version": old_pv, "values": {s["name"]: s["x0"] for s in space},
                   "total": before["total"], "items": _summary_items(before)},
        "after": {"params_version": new_pv, "values": {s["name"]: round(v, 5)
                                                         for s, v in zip(space, best["values"], strict=True)},
                  "total": best["total"], "items": _summary_items(best["res"])},
        "improved": improved,
        "written": improved and not a.dry_run,
        "history": hist,
    }
    if improved and not a.dry_run:
        write_config(a.d2_config, space, best_cfg, new_pv)
    _merge_write(a.out, out)
    brief = {k: out[k] for k in ("improved",)} | {"before": before["total"], "after": best["total"],
                                                    "evals": len(hist), "seconds": round(secs, 1),
                                                    "params_version": new_pv, "values": out["after"]["values"]}
    print(json.dumps(brief, ensure_ascii=False, indent=None if a.json else 2))
    return 0 if improved else 1


ROUND_KEEP = ("realsr", "previous_rounds", "rejected_rounds")


def _merge_write(path: Path, data: dict) -> None:
    """Ghi ``calibration.json``. Mức gốc = vòng hiệu chỉnh ĐANG DÙNG (khớp ``params_version`` của config); vòng cũ đã
    được thay chuyển vào ``previous_rounds``; vòng không giảm khoảng cách (không nhận) vào ``rejected_rounds``. Khối
    ``realsr`` (E3) giữ nguyên khi chỉ ghi một vòng hiệu chỉnh và ngược lại."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "history" not in data:                          # chỉ ghi khối realsr
        data = {**old, **data} if old else data
    elif "history" in old:                             # có vòng cũ
        old_round = {k: v for k, v in old.items() if k not in ROUND_KEEP}
        keep = {k: old[k] for k in ROUND_KEEP if k in old}
        keep.setdefault("previous_rounds", [])
        keep.setdefault("rejected_rounds", [])
        data["round"] = int(old_round.get("round", 1)) + 1 + len(keep["rejected_rounds"])
        if data["improved"] and data.get("written"):
            keep["previous_rounds"].append(old_round)
            data = {**data, **keep}
        else:
            keep["rejected_rounds"].append(data)
            data = {**old_round, **keep}
    else:
        data = {**{k: old[k] for k in ROUND_KEEP if k in old}, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def _run_realsr(a, base: dict, real: dict, units: list, ctx: dict, log) -> int:
    gains = [float(g) for g in a.realsr_gains.split(",") if g.strip()]
    ev_d2 = Evaluator(real, units, ctx, a.workers, "d2")
    try:
        d2_total = ev_d2([base])[0]["total"]
    finally:
        ev_d2.close()
    ev = Evaluator(real, units, ctx, a.workers, "d2sr")
    try:
        cfgs = []
        for g in gains:
            c = copy.deepcopy(base)
            c.setdefault("realsr", {})["gain"] = g
            cfgs.append(c)
        res = ev(cfgs)
    finally:
        ev.close()
    rows = [{"gain": g, "total": r["total"], "items": _summary_items(r)} for g, r in zip(gains, res, strict=True)]
    best = min(rows, key=lambda r: r["total"])
    log(f"D2 {d2_total:.4f}; Real-SR " + ", ".join(f"g{r['gain']}: {r['total']:.4f}" for r in rows))
    if not a.dry_run:
        p = a.d2_config
        p.write_text(set_yaml_value(p.read_text(encoding="utf-8"), "realsr.gain", best["gain"]), encoding="utf-8",
                     newline="\n")
    _merge_write(a.out, {"realsr": {"d2_total_same_units": d2_total, "grid": rows, "chosen_gain": best["gain"],
                                    "params_version_d2": base.get("params_version")}})
    print(json.dumps({"d2_total": d2_total, "realsr": [{"gain": r["gain"], "total": r["total"]} for r in rows],
                      "chosen_gain": best["gain"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
