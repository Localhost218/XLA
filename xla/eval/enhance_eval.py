"""Đánh giá nhánh A (ảnh rõ) trên khung bay thật — phase 14 bước 2–6 (không có ảnh chuẩn để so).

Ba biến thể ảnh trên cùng khung: ``raw`` (khung gốc), ``classic`` (đối chứng cổ điển, ``--classic module:func``),
``net`` (``enhancer.onnx`` của bundle ``--enhancer``, ORT). Lệnh con (mỗi lệnh ``--json``)::

    python -m xla.eval.enhance_eval physical --json   # bước 2: chỉ số vật lý + tỉ lệ đổi so với gốc, CI theo khối
    python -m xla.eval.enhance_eval flicker --json    # bước 3: nhấp nháy qua xla.eval.temporal
    python -m xla.eval.enhance_eval speed --json      # bước 4: ms/khung (cổ điển CPU, mạng ORT CUDA EP)
    python -m xla.eval.enhance_eval iqa --json        # bước 5: MUSIQ + NIQE (pyiqa), CHỈ mô tả
    python -m xla.eval.enhance_eval e4 --json         # bước 6: 2 detector × 3 biến thể → results/e4/e4.json
    python -m xla.eval.enhance_eval all --json        # tất cả → results/enhance/enhance_eval.json

Mã thoát (D7): 0 = đo xong (đạt/không đạt tiêu chí nằm trong JSON, không phải mã thoát); 1 = dữ liệu/bundle sai hợp
đồng/thiếu nhãn/không nạp được hàm cổ điển; 2 = sai cách gọi (cờ sai, thư mục bundle không tồn tại).

Quy ước (ghi lại để Owner chạy lại bằng model thật):

- Khung test = ``split == "test"`` của ``data/real/split.json`` (198 khung, cách nhau ≥ 1 s). CI 95 % = bootstrap theo
  khối 30 s (``block``), cùng một bộ mẫu khối cho mọi biến thể (ghép cặp), ``BOOT_N`` mẫu, seed ``BOOT_SEED``.
- Tỉ lệ đổi = ``thống_kê(biến thể) / thống_kê(gốc) − 1``; thống kê là trung bình, riêng dark channel, Y p1, Laplacian
  dùng trung vị (tiêu chí §8 ghi "dark channel trung vị"). "Giảm" = ``−tỉ lệ đổi``.
- Nhấp nháy: khung test cách nhau ≥ 1 s nên không có chuỗi liền kề trong ``split.json``. Lấy mọi khung bay của
  ``data/real/manifest.csv`` nằm trong khoảng thời gian của một khối test (khung bị bỏ khi chia test vẫn là khung giữ
  lại, không thuộc dev), nhóm bằng ``xla.eval.temporal.consecutive_runs`` (< 250 ms). Tỉ lệ gộp = trung bình có trọng số
  của ``flicker_ratio`` từng chuỗi, trọng số = Σ|Δ gốc| của chuỗi — đúng bằng ``flicker_ratio`` trên mọi chuỗi một lần.
- GPU (ORT CUDA EP, pyiqa, YOLO) chạy trong ``xla.gpu_lock`` — mỗi lệnh con giữ khoá một lần rồi nhả.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera.measure import measure_image
from xla.contract import FRAME_SHAPE, INPUT_NAME, decode_frame, load_profile, validate_bundle
from xla.eval.detection import BOOT_N, BOOT_SEED, EvalInputError, TestLeakError, _ci
from xla.eval.temporal import consecutive_runs, flicker_ratio, mean_abs_diff
from xla.paths import repo_root

Fn = Callable[[np.ndarray], np.ndarray]

VARIANTS = ("raw", "classic", "net")
DEFAULT_CLASSIC = "xla.enhance.classic:enhance"
ENHANCER_REL = Path("models/release/enhancer-v1.0.0")
DETECTOR_REL = Path("models/release/detector-v1.0.0")
SPLIT_REL = Path("data/real/split.json")
FRAMES_REL = Path("data/real/frames")
MANIFEST_CSV_REL = Path("data/real/manifest.csv")
PROFILE_REL = Path("configs/camera_profile.yaml")
CLASSIC_BENCH_REL = Path("results/enhance/classic_bench.json")
OUT_REL = Path("results/enhance/enhance_eval.json")
E4_OUT_REL = Path("results/e4/e4.json")
E4_PRED_REL = Path("results/e4/preds")
AB_RESULT_REL = Path("results/ab/ab_result.json")
AB_KEY_REL = Path("results/ab/ab_key.json")
LOCK_TIMEOUT_S = 6 * 3600.0
ORT_THREADS = 4  # máy lật bit RAM dưới tải: tối đa 4 luồng/worker (thuc-thi §3)
PROVIDERS = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider", "trt": "TensorrtExecutionProvider"}

PHYSICAL = {
    "pink_frac": "mean",
    "dark_channel_15px": "median",
    "y_p1": "median",
    "saturation_mean": "mean",
    "blocking_ratio_8x8": "mean",
    "laplacian_var": "median",
}
"""Chỉ số vật lý (khoá của ``xla.camera.measure.measure_image``) → thống kê gộp trên khung."""
_AGG: dict[str, Callable[[np.ndarray], float]] = {"mean": np.nanmean, "median": np.nanmedian}

# §8 nhánh A (thiết kế, sửa lại sau M2)
CRIT_PINK_DROP = 0.90
CRIT_DARK_DROP = 0.50
CRIT_FLICKER_MAX = 1.0
CRIT_MS_MAX = 30.0
CRIT_AB_VS_RAW = 0.70
CRIT_AB_VS_CLASSIC = 0.55

SPEED_N = 100
SPEED_WARMUP = 10
IQA_METRICS = ("musiq", "niqe")
IQA_LICENSE = ("pyiqa: PolyForm Noncommercial License 1.0.0 (+ NTU S-Lab License cho một số thành phần) — chỉ dùng "
               "trong đồ án, phi thương mại; số NR-IQA chỉ để mô tả, không chọn checkpoint/bộ khôi phục (Deferred 5)")
COCO_OP_THR = 0.25  # ngưỡng vận hành mốc COCO (DEFAULT_THRESHOLDS của xla.eval.detection)


class UsageError(Exception):
    """Sai cách gọi → exit 2."""


class EvalError(Exception):
    """Dữ liệu / bundle / hàm cổ điển sai → exit 1."""


# ---------------------------------------------------------------- tiện ích số


def _num(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return round(f, 6) if np.isfinite(f) else None


def _ci_list(values: Sequence[float]) -> list[float | None]:
    if not len(values):
        return [None, None]
    return [_num(x) for x in _ci(list(values))]


def _ratio_change(a: float, b: float) -> float:
    return a / b - 1.0 if b != 0 and np.isfinite(a) and np.isfinite(b) else float("nan")


def block_samples(blocks: Sequence[int], n: int, seed: int) -> list[np.ndarray]:
    """``n`` mẫu bootstrap theo khối: lấy lại (có hoàn lại) đúng số khối, trả chỉ số phần tử của các khối được chọn."""
    arr = np.asarray(blocks)
    uniq = np.unique(arr)
    if uniq.size == 0:
        return []
    members = {int(b): np.flatnonzero(arr == b) for b in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[int(b)] for b in rng.choice(uniq, uniq.size)]) for _ in range(n)]


def change_summary(values: dict[str, np.ndarray], how: str, samples: list[np.ndarray]) -> dict[str, Any]:
    """Thống kê ``how`` mỗi biến thể + CI, và tỉ lệ đổi so với ``raw`` (ghép cặp trên cùng mẫu khối) + CI."""
    agg = _AGG[how]
    base = values["raw"]
    out: dict[str, Any] = {}
    for var, v in values.items():
        entry: dict[str, Any] = {"value": _num(agg(v)), "ci95": _ci_list([agg(v[s]) for s in samples])}
        if var != "raw":
            entry["change_vs_raw"] = _num(_ratio_change(agg(v), agg(base)))
            entry["change_ci95"] = _ci_list([_ratio_change(agg(v[s]), agg(base[s])) for s in samples])
        out[var] = entry
    return {"stat": how} | out


def pooled_flicker(ratios: Sequence[float], weights: Sequence[float]) -> float:
    """Gộp ``flicker_ratio`` từng chuỗi theo trọng số Σ|Δ gốc| (= ``flicker_ratio`` trên mọi chuỗi cùng lúc)."""
    w = np.asarray(weights, float)
    if w.size == 0 or w.sum() <= 0:
        return float("nan")
    return float(np.dot(np.asarray(ratios, float), w) / w.sum())


def run_flicker(raw_run: Sequence[np.ndarray], proc_run: Sequence[np.ndarray], crop_left: int) -> tuple[float, float]:
    """(``flicker_ratio`` của một chuỗi, trọng số Σ|Δ gốc|). Mọi phép tính nhấp nháy đi qua ``xla.eval.temporal``."""
    return flicker_ratio(raw_run, proc_run, crop_left), mean_abs_diff(raw_run, crop_left) * (len(raw_run) - 1)


def d5_recommendation(diff_ci_lo: float | None) -> str:
    """D5: đường nhận diện đổi sang ảnh ``enhanced`` CHỈ khi CI 95 % của chênh mAP50 ghép cặp nằm trọn trên 0."""
    return "enhanced" if diff_ci_lo is not None and diff_ci_lo > 0 else "raw"


def _status(ok: bool | None) -> str:
    return "cho_do" if ok is None else ("dat" if ok else "khong_dat")


# ---------------------------------------------------------------- hàm cổ điển, bundle, ORT


def _check_out(name: str, out: Any) -> np.ndarray:
    if not isinstance(out, np.ndarray) or out.dtype != np.uint8 or out.shape != FRAME_SHAPE:
        got = f"{getattr(out, 'dtype', type(out).__name__)} {getattr(out, 'shape', '')}"
        raise EvalError(f"{name}: phải trả uint8 {list(FRAME_SHAPE)} BGR, gặp {got}")
    return out


def load_classic(spec: str, params_path: Path | None = None) -> tuple[Fn, dict[str, Any]]:
    """Nạp hàm cổ điển từ ``module:func`` hoặc ``đường/dẫn/file.py:func``.

    Module có ``load_params`` (hợp đồng §5.1 của ``xla.enhance.classic``) thì gọi một lần (``params_path`` hoặc mặc
    định của module) và truyền ``params`` vào ``func(frame, params)``; không có thì gọi ``func(frame)``.
    """
    mod_name, sep, func_name = spec.rpartition(":")
    if not sep or not mod_name or not func_name.isidentifier():
        raise UsageError(f"--classic phải dạng module:func hoặc file.py:func, gặp {spec!r}")
    try:
        if mod_name.endswith(".py"):
            path = Path(mod_name)
            if not path.is_file():
                raise EvalError(f"--classic: không có file {path}")
            mspec = importlib.util.spec_from_file_location(f"_xla_classic_{path.stem}", path)
            if mspec is None or mspec.loader is None:
                raise EvalError(f"--classic: không nạp được {path}")
            mod = importlib.util.module_from_spec(mspec)
            mspec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise EvalError(f"--classic: không import được {mod_name} ({e}); lane classic chưa gộp thì truyền --classic "
                        "<file.py:func> tạm") from e
    func = getattr(mod, func_name, None)
    if not callable(func):
        raise EvalError(f"--classic: {mod_name} không có hàm {func_name}")
    loader = getattr(mod, "load_params", None)
    if params_path is not None and not callable(loader):
        raise UsageError(f"--classic-params: {mod_name} không có load_params")
    params = loader(params_path) if callable(loader) else None
    fn: Fn = (lambda f: func(f, params)) if callable(loader) else (lambda f: func(f))
    info = {"spec": spec, "params_path": str(params_path) if params_path else None,
            "params": json.loads(json.dumps(params, default=str)) if params is not None else None}
    return (lambda f: _check_out("classic", fn(f))), info


def parse_providers(text: str) -> list[str]:
    names = [t.strip().lower() for t in text.split(",") if t.strip()]
    bad = [n for n in names if n not in PROVIDERS]
    if not names or bad:
        raise UsageError(f"--providers: chỉ nhận {sorted(PROVIDERS)} cách nhau dấu phẩy, gặp {text!r}")
    return [PROVIDERS[n] for n in names]


def ort_session(onnx_path: Path, providers: list[str]) -> tuple[Any, list[str]]:
    """Phiên ORT với ``providers`` (thứ tự ưu tiên); trả (phiên, provider thật sự dùng) — ghi vào JSON, không im lặng."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()  # nạp DLL CUDA/cuDNN của torch/nvidia-*; thất bại thì CUDA EP tự báo qua get_providers
        except Exception as e:  # noqa: BLE001 - chỉ ghi lại, provider thật được kiểm ngay dưới
            print(f"cảnh báo: ort.preload_dlls() lỗi: {e}", file=sys.stderr)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = ORT_THREADS
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    return sess, list(sess.get_providers())


@dataclass
class Bundle:
    path: Path
    model: str
    meta: dict[str, Any]

    @property
    def onnx(self) -> Path:
        return self.path / f"{self.model}.onnx"


def open_bundle(path: Path, model: str) -> Bundle:
    if not path.is_dir():
        raise UsageError(f"không có thư mục bundle {model}: {path}")
    errors = validate_bundle(path)
    if errors:
        raise EvalError(f"bundle {path} sai hợp đồng: {errors}")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("model") != model:
        raise EvalError(f"{path}: meta.model = {meta.get('model')!r}, cần {model!r}")
    return Bundle(path, model, meta)


def _bundle_info(b: Bundle, active: list[str]) -> dict[str, Any]:
    return {"path": str(b.path), "version": b.meta.get("version"), "onnx_sha256": b.meta.get("onnx_sha256"),
            "train_run": b.meta.get("train_run"), "providers_active": active}


class Enhancer:
    def __init__(self, bundle: Bundle, providers: list[str]) -> None:
        self.bundle = bundle
        self.sess, self.active = ort_session(bundle.onnx, providers)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return _check_out("enhancer", self.sess.run(None, {INPUT_NAME: frame})[0])


class BundleDetector:
    def __init__(self, bundle: Bundle, providers: list[str]) -> None:
        self.bundle = bundle
        self.sess, self.active = ort_session(bundle.onnx, providers)
        self.names = [o.name for o in self.sess.get_outputs()]

    def __call__(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        out = dict(zip(self.names, self.sess.run(None, {INPUT_NAME: frame}), strict=True))
        return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
                for b, s in zip(out["boxes"].reshape(-1, 4), out["scores"].reshape(-1), strict=True)]


class CocoDetector:
    """Mốc YOLO11s COCO như ``xla.detect.baseline`` (imgsz, conf, lớp người, max_det), suy luận trên mảng BGR."""

    def __init__(self, weights: Path, device: str) -> None:
        from ultralytics import YOLO

        if not weights.is_file():
            raise EvalError(f"không có trọng số mốc COCO {weights}")
        self.weights, self.device = weights, device
        self.model = YOLO(str(weights))

    def __call__(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        from xla.detect.baseline import CLASS_PERSON, CONF, IMGSZ, MAX_DET

        res = self.model.predict(frame, imgsz=IMGSZ, conf=CONF, classes=[CLASS_PERSON], max_det=MAX_DET,
                                 device=self.device, verbose=False)[0]
        return [(float(x1), float(y1), float(x2), float(y2), float(c))
                for x1, y1, x2, y2, c, _ in res.boxes.data.cpu().numpy()]


@dataclass
class VariantSet:
    fns: dict[str, Fn]
    crop_left: int
    info: dict[str, Any]
    uses_gpu: bool
    enhancer: Enhancer | None = None


def profile_crop_left(root: Path) -> int:
    prof = load_profile(root / PROFILE_REL)
    try:
        return int(prof["left_strip"]["crop_left_px"])
    except (KeyError, TypeError, ValueError) as e:
        raise EvalError(f"{PROFILE_REL}: thiếu left_strip.crop_left_px") from e


def build_variants(a: argparse.Namespace, root: Path) -> VariantSet:
    providers = parse_providers(a.providers)
    classic, classic_info = load_classic(a.classic, a.classic_params)
    enh = Enhancer(open_bundle(a.enhancer or root / ENHANCER_REL, "enhancer"), providers)
    crop_left = profile_crop_left(root)
    if enh.bundle.meta.get("crop_left") != crop_left:
        raise EvalError(f"crop_left của enhancer ({enh.bundle.meta.get('crop_left')}) khác hồ sơ camera ({crop_left})")
    info = {"raw": "khung gốc camera", "classic": classic_info, "net": _bundle_info(enh.bundle, enh.active),
            "providers_requested": providers}
    if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in enh.active:
        info["warning"] = "yêu cầu CUDA EP nhưng ORT chỉ chạy được " + ",".join(enh.active)
    return VariantSet({"raw": lambda f: f, "classic": classic, "net": enh}, crop_left, info,
                      "CUDAExecutionProvider" in enh.active, enh)


def progress(label: str, i: int, n: int, every: int = 25) -> None:
    """Một dòng tiến độ ra stderr (stdout chỉ có JSON kết quả)."""
    if i == 0 or (i + 1) % every == 0 or i + 1 == n:
        print(f"[{label}] {i + 1}/{n} {datetime.now().strftime('%H:%M:%S')}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def maybe_gpu(need: bool, label: str, timeout_s: float) -> Iterator[dict[str, Any]]:
    """Giữ ``xla.gpu_lock`` khi ``need``; yield bản ghi chờ khoá để đưa vào JSON."""
    rec: dict[str, Any] = {"gpu_lock": need, "wait_s": 0.0}
    if not need:
        yield rec
        return
    from xla.gpu_lock import gpu_lock

    t0 = time.perf_counter()
    with gpu_lock(f"phase14 abeval {label}", timeout_s=timeout_s):
        rec["wait_s"] = round(time.perf_counter() - t0, 1)
        yield rec


# ---------------------------------------------------------------- khung


@dataclass
class Ctx:
    root: Path
    split_rows: list[dict[str, Any]]
    frames_dir: Path
    boot: int
    seed: int
    limit: int
    lock_timeout: float
    meta: dict[str, Any] = field(default_factory=dict)

    def frames(self, split: str) -> list[dict[str, Any]]:
        rows = sorted((r for r in self.split_rows if r["split"] == split), key=lambda r: r["frame"])
        return rows[: self.limit] if self.limit else rows

    def load(self, frame: str) -> np.ndarray:
        p = self.frames_dir / frame
        if not p.is_file():
            raise EvalError(f"thiếu khung {p}")
        img = decode_frame(p)
        if img.shape != FRAME_SHAPE:
            raise EvalError(f"{frame}: cỡ {img.shape}, cần {FRAME_SHAPE}")
        return img


def make_ctx(a: argparse.Namespace) -> Ctx:
    root = repo_root()
    split_json = a.split_json or root / SPLIT_REL
    if not split_json.is_file():
        raise EvalError(f"không có {split_json}")
    rows = json.loads(split_json.read_text(encoding="utf-8"))["frames"]
    if a.boot < 0 or a.limit < 0:
        raise UsageError("--boot và --limit phải ≥ 0")
    return Ctx(root, rows, a.frames_dir or root / FRAMES_REL, a.boot, a.seed, a.limit, a.lock_timeout,
               {"split_json": str(split_json), "boot": {"n": a.boot, "seed": a.seed, "unit": "khối 30 s (block)"},
                "smoke_limit": a.limit or None})


# ---------------------------------------------------------------- bước 2: chỉ số vật lý


def cmd_physical(a: argparse.Namespace, ctx: Ctx, vs: VariantSet) -> dict[str, Any]:
    rows = ctx.frames("test")
    if not rows:
        raise EvalError("không có khung test")
    vals = {v: {k: np.empty(len(rows)) for k in PHYSICAL} for v in VARIANTS}
    with maybe_gpu(vs.uses_gpu, "physical", ctx.lock_timeout) as lock:
        for i, r in enumerate(rows):
            progress("physical", i, len(rows))
            img = ctx.load(r["frame"])
            for v in VARIANTS:
                m = measure_image(vs.fns[v](img), vs.crop_left)
                for k in PHYSICAL:
                    vals[v][k][i] = m[k]
    samples = block_samples([int(r["block"]) for r in rows], ctx.boot, ctx.seed)
    metrics = {k: change_summary({v: vals[v][k] for v in VARIANTS}, how, samples) for k, how in PHYSICAL.items()}
    drops = {}
    for key, crit in (("pink_frac", CRIT_PINK_DROP), ("dark_channel_15px", CRIT_DARK_DROP)):
        drops[key] = {}
        for v in ("classic", "net"):
            ch, (lo, hi) = metrics[key][v]["change_vs_raw"], metrics[key][v]["change_ci95"]
            drop = None if ch is None else -ch
            drops[key][v] = {"drop": _num(drop), "ci95": [_num(-hi) if hi is not None else None,
                                                          _num(-lo) if lo is not None else None],
                             "criterion_min": crit, "status": _status(None if drop is None else drop >= crit)}
    return {"n_frames": len(rows), "n_blocks": len({r["block"] for r in rows}), "crop_left": vs.crop_left,
            "metrics": metrics, "drops": drops, "gpu": lock,
            "note": "tỉ lệ đổi = thống_kê(biến thể)/thống_kê(gốc) − 1; drop = −tỉ lệ đổi"}


# ---------------------------------------------------------------- bước 3: nhấp nháy


def test_region_rows(split_rows: list[dict[str, Any]], manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Khung bay của ``manifest.csv`` nằm trong khoảng thời gian của một khối test → ``{"frame","t_unix","split","block"}``.

    Khung trong khoảng của một khối dev bị bỏ (không đo trên dữ liệu huấn luyện)."""
    spans: dict[tuple[str, int], list[float]] = {}
    for r in split_rows:
        if r["split"] in ("test", "dev"):
            s = spans.setdefault((r["split"], int(r["block"])), [np.inf, -np.inf])
            s[0], s[1] = min(s[0], float(r["t_unix"])), max(s[1], float(r["t_unix"]))
    test = [(b, lo, hi) for (sp, b), (lo, hi) in spans.items() if sp == "test"]
    dev = [(lo, hi) for (sp, _), (lo, hi) in spans.items() if sp == "dev"]
    out = []
    for m in manifest_rows:
        if m.get("kind") != "flight":
            continue
        t = float(m["t_unix"])
        hit = [b for b, lo, hi in test if lo <= t <= hi]
        if hit and not any(lo <= t <= hi for lo, hi in dev):
            out.append({"frame": m["frame"], "t_unix": t, "split": "test", "block": hit[0]})
    return out


def cmd_flicker(a: argparse.Namespace, ctx: Ctx, vs: VariantSet) -> dict[str, Any]:
    mpath = a.manifest_csv or ctx.root / MANIFEST_CSV_REL
    if not mpath.is_file():
        raise EvalError(f"không có {mpath}")
    with mpath.open(encoding="utf-8", newline="") as fh:
        mrows = list(csv.DictReader(fh))
    rows = test_region_rows(ctx.split_rows, mrows)
    block_of = {r["frame"]: r["block"] for r in rows}
    runs = consecutive_runs(rows, "test")
    if ctx.limit:
        runs = runs[: max(1, ctx.limit // 10)]
    if not runs:
        raise EvalError("không có chuỗi khung test liền kề (< 250 ms)")
    per: dict[str, list[tuple[float, float]]] = {"classic": [], "net": []}
    blocks, skipped = [], 0
    with maybe_gpu(vs.uses_gpu, "flicker", ctx.lock_timeout) as lock:
        for i, run in enumerate(runs):
            progress("flicker", i, len(runs), every=10)
            raw = [ctx.load(f) for f in run]
            if mean_abs_diff(raw, vs.crop_left) == 0:
                skipped += 1  # khung gốc đứng yên tuyệt đối: không chia được, ghi số chuỗi bỏ
                continue
            blocks.append(block_of[run[0]])
            for v, acc in per.items():
                acc.append(run_flicker(raw, [vs.fns[v](f) for f in raw], vs.crop_left))
    samples = block_samples(blocks, ctx.boot, ctx.seed)
    res: dict[str, Any] = {}
    for v, pairs in per.items():
        r = np.asarray([p[0] for p in pairs])
        w = np.asarray([p[1] for p in pairs])
        val = pooled_flicker(r, w)
        res[v] = {"flicker_ratio": _num(val), "ci95": _ci_list([pooled_flicker(r[s], w[s]) for s in samples]),
                  "criterion_max": CRIT_FLICKER_MAX, "status": _status(val <= CRIT_FLICKER_MAX)}
    return {"n_runs": len(blocks), "n_runs_skipped_static": skipped, "n_frames": sum(len(x) for x in runs),
            "n_blocks": len(set(blocks)), "max_gap_s": 0.25, "source": str(mpath), "crop_left": vs.crop_left,
            "variants": res, "gpu": lock,
            "note": "khung manifest.csv trong khoảng thời gian khối test (không thuộc khối dev); 1 = không đổi"}


# ---------------------------------------------------------------- bước 4: ms/khung


def time_fn(fn: Fn, frames: list[np.ndarray], warmup: int) -> dict[str, Any]:
    for f in frames[:warmup]:
        fn(f)
    ms = []
    for f in frames:
        t0 = time.perf_counter()
        fn(f)
        ms.append((time.perf_counter() - t0) * 1000)
    v = np.asarray(ms)
    return {"p50": _num(np.median(v)), "p95": _num(np.percentile(v, 95)), "mean": _num(v.mean()), "n": int(v.size),
            "warmup": warmup}


def cmd_speed(a: argparse.Namespace, ctx: Ctx, vs: VariantSet) -> dict[str, Any]:
    rows = ctx.frames("test")[: a.n_frames]
    if not rows:
        raise EvalError("không có khung test")
    frames = [ctx.load(r["frame"]) for r in rows]
    classic = time_fn(vs.fns["classic"], frames, a.warmup) | {"device": "CPU", "runtime": "python (hàm cổ điển)"}
    with maybe_gpu(vs.uses_gpu, "speed", ctx.lock_timeout) as lock:
        net = time_fn(vs.fns["net"], frames, a.warmup)
    import onnxruntime as ort

    net |= {"providers_active": vs.enhancer.active if vs.enhancer else None, "runtime": f"onnxruntime {ort.__version__}"}
    bench_p = ctx.root / CLASSIC_BENCH_REL
    recorded = {"classic_bench": json.loads(bench_p.read_text(encoding="utf-8")) if bench_p.is_file() else None,
                "enhancer_meta_ms_per_frame": vs.enhancer.bundle.meta.get("ms_per_frame") if vs.enhancer else None}
    p50 = net["p50"]
    return {"measured": {"classic": classic, "net": net}, "recorded": recorded, "gpu": lock,
            "criterion": {"net_p50_max_ms": CRIT_MS_MAX, "status": _status(None if p50 is None else p50 <= CRIT_MS_MAX)},
            "note": "ms = một lần gọi hàm trên khung đã giải mã (không gồm đọc JPEG); mạng = ORT, provider ở providers_active"}


# ---------------------------------------------------------------- bước 5: NR-IQA (chỉ mô tả)


def cmd_iqa(a: argparse.Namespace, ctx: Ctx, vs: VariantSet) -> dict[str, Any]:
    try:
        import cv2
        import pyiqa
        import torch
    except ImportError as e:
        raise EvalError(f"thiếu gói cho NR-IQA: {e}") from e
    names = [n.strip() for n in a.iqa_metrics.split(",") if n.strip()]
    device = a.iqa_device
    if device == "cuda" and not torch.cuda.is_available():
        raise EvalError("--iqa-device cuda nhưng torch không thấy CUDA; dùng --iqa-device cpu")
    rows = ctx.frames("test")
    if not rows:
        raise EvalError("không có khung test")
    scores = {n: {v: np.empty(len(rows)) for v in VARIANTS} for n in names}
    lower_better: dict[str, bool] = {}
    with maybe_gpu(vs.uses_gpu or device == "cuda", "iqa", ctx.lock_timeout) as lock:
        try:
            metrics = {n: pyiqa.create_metric(n, device=torch.device(device), as_loss=False) for n in names}
        except Exception as e:  # tải/khởi tạo trọng số lỗi: báo thành lỗi dữ liệu
            raise EvalError(f"pyiqa không tạo được metric {names}: {e}") from e
        lower_better = {n: bool(getattr(m, "lower_better", False)) for n, m in metrics.items()}
        with torch.no_grad():
            for i, r in enumerate(rows):
                progress("iqa", i, len(rows))
                img = ctx.load(r["frame"])
                for v in VARIANTS:
                    rgb = cv2.cvtColor(np.ascontiguousarray(vs.fns[v](img)[:, vs.crop_left:]), cv2.COLOR_BGR2RGB)
                    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(device)
                    for n, m in metrics.items():
                        scores[n][v][i] = float(m(x).item())
    samples = block_samples([int(r["block"]) for r in rows], ctx.boot, ctx.seed)
    out = {n: change_summary(scores[n], "mean", samples) | {"lower_better": lower_better[n]} for n in names}
    return {"n_frames": len(rows), "crop": f"cột [{vs.crop_left}, 640)", "device": device,
            "pyiqa_version": pyiqa.__version__, "license": IQA_LICENSE, "metrics": out, "gpu": lock,
            "note": "CHỈ mô tả (Deferred 5): không dùng để chọn gì"}


# ---------------------------------------------------------------- bước 6: E4


def _pick(m: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    for k, v in m.items():
        if k == prefix or k.startswith(prefix + "@"):
            return v
    return None


def cmd_e4(a: argparse.Namespace, ctx: Ctx, vs: VariantSet) -> dict[str, Any]:
    from xla.detect.baseline import WEIGHTS_REL, drop_edge, write_csv
    from xla.eval.detection import compare, evaluate
    from xla.paths import downloads_dir

    det_b = open_bundle(a.detector or ctx.root / DETECTOR_REL, "detector")
    if det_b.meta.get("input_variant", "raw") != "raw":
        raise EvalError("E4 cần detector chấm trên khung gốc (input_variant raw)")
    providers = parse_providers(a.providers)
    bundle_det = BundleDetector(det_b, providers)
    coco = CocoDetector(a.coco_weights or downloads_dir() / WEIGHTS_REL, a.coco_device)
    dets: dict[str, Callable[[np.ndarray], list[tuple[float, float, float, float, float]]]] = {
        "coco": coco, "bundle": bundle_det}
    ops = {"coco": COCO_OP_THR, "bundle": float(det_b.meta["conf_threshold"])}
    rows = ctx.frames("test") + ctx.frames("neg")
    per: dict[str, dict[str, dict[str, tuple[list[Any], float]]]] = {d: {v: {} for v in VARIANTS} for d in dets}
    need_gpu = vs.uses_gpu or "CUDAExecutionProvider" in bundle_det.active or a.coco_device != "cpu"
    with maybe_gpu(need_gpu, "e4", ctx.lock_timeout) as lock:
        for i, r in enumerate(rows):
            progress("e4", i, len(rows), every=50)
            img = ctx.load(r["frame"])
            for v in VARIANTS:
                x = vs.fns[v](img)
                for d, det in dets.items():
                    t0 = time.perf_counter()
                    boxes = drop_edge(det(x))
                    per[d][v][r["frame"]] = (boxes, (time.perf_counter() - t0) * 1000)
    pred_dir = a.pred_dir or ctx.root / E4_PRED_REL
    csvs = {d: {v: pred_dir / f"{d}_{v}.csv" for v in VARIANTS} for d in dets}
    for d in dets:
        for v in VARIANTS:
            write_csv(csvs[d][v], per[d][v])
    common = {"split": "test", "labels_dir": a.labels_dir, "manifest_path": a.manifest, "split_json": a.split_json,
              "boot_n": ctx.boot, "seed": ctx.seed, "allow_missing_labels": a.allow_missing_labels}
    cells: dict[str, Any] = {}
    diffs: dict[str, Any] = {}
    for d in dets:
        cells[d], diffs[d] = {}, {}
        for v in VARIANTS:
            print(f"[e4] chấm {d} × {v}", file=sys.stderr, flush=True)
            r = evaluate(csvs[d][v], thresholds=[ops[d]], f1_from_devval=False, **common)
            m = r["metrics"]
            cells[d][v] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "AR_small": m["AR_small"],
                           "fp_per_neg_frame@op": _pick(m, "fp_per_neg_frame"), "csv": str(csvs[d][v]),
                           "official": r["official"], "n": r["n"], "ms_per_frame": r["ms_per_frame"]}
        for v in ("classic", "net"):
            c = compare(csvs[d]["raw"], csvs[d][v], thr_a=ops[d], thr_b=ops[d], **common)
            diffs[d][v] = {k: c["metrics"][k] for k in ("mAP50", "mAP50_95", "fp_per_neg_frame@op")}
    lo = diffs["bundle"]["net"]["mAP50"]["ci95"][0]
    rec = d5_recommendation(lo)
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "test", "op_threshold": ops,
        "detectors": {"coco": {"weights": str(coco.weights), "device": a.coco_device,
                               "note": "như xla.detect.baseline full + bỏ hộp x1 < 16 (edge16)"},
                      "bundle": _bundle_info(det_b, bundle_det.active)},
        "variants": vs.info, "cells": cells, "paired_diff_vs_raw": diffs,
        "d5": {"recommended_input_variant": rec,
               "basis": "bundle × net: CI 95 % của chênh mAP50 ghép cặp (net − raw)",
               "diff_ci95": diffs["bundle"]["net"]["mAP50"]["ci95"],
               "rule": "đổi D5 sang enhanced CHỈ khi CI nằm trọn trên 0",
               "action": ("khuyến nghị — không tự sửa hợp đồng; enhanced thì Owner tăng contract_version 1.1"
                          if rec == "enhanced" else "giữ D5: nhận diện chạy khung gốc")},
        "gpu": lock,
    }


# ---------------------------------------------------------------- all


def _ab_summary(root: Path, ab_result: Path | None, ab_key: Path | None) -> dict[str, Any] | None:
    p = ab_result or root / AB_RESULT_REL
    if not p.is_file():
        return None
    from xla.eval.ab_score import AbScoreError, score

    try:
        return score(json.loads(p.read_text(encoding="utf-8")),
                     json.loads((ab_key or root / AB_KEY_REL).read_text(encoding="utf-8")))
    except (AbScoreError, OSError, ValueError) as e:
        raise EvalError(f"không chấm được A/B từ {p}: {e}") from e


def criteria(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Bảng đối chiếu §8 nhánh A (đạt / không đạt / chờ đo) cho biến thể ``net``; kết luận viết ở README."""
    ph, fl, sp, ab, e4 = (doc.get(k) for k in ("physical", "flicker", "speed", "ab", "e4"))
    out = []
    for cid, key, text in (("pink_drop", "pink_frac", "% điểm hồng giảm ≥ 90 %"),
                           ("dark_channel_drop", "dark_channel_15px", "dark channel trung vị giảm ≥ 50 %")):
        d = ph["drops"][key]["net"] if ph else None
        out.append({"id": cid, "text": text, "value": d and d["drop"], "ci95": d and d["ci95"],
                    "status": d["status"] if d else "cho_do"})
    f = fl["variants"]["net"] if fl else None
    out.append({"id": "flicker", "text": "nhấp nháy không tăng (≤ 1)", "value": f and f["flicker_ratio"],
                "ci95": f and f["ci95"], "status": f["status"] if f else "cho_do"})
    for cid, comp, text in (("ab_vs_raw", "net_vs_raw", "thắng ≥ 70 % lượt A/B so với ảnh gốc"),
                            ("ab_vs_classic", "net_vs_classic", "thắng ≥ 55 % lượt A/B so với cổ điển")):
        c = ab["comparisons"][comp] if ab else None
        out.append({"id": cid, "text": text, "value": c and c["win_rate"], "ci95": c and c["ci95_wilson"],
                    "status": _status(c["meets_criterion"]) if c else "cho_do"})
    out.append({"id": "ms", "text": "≤ 30 ms/khung (mạng, p50)", "value": sp and sp["measured"]["net"]["p50"],
                "status": sp["criterion"]["status"] if sp else "cho_do"})
    if e4:
        diff = e4["paired_diff_vs_raw"]["bundle"]["net"]["mAP50"]
        raw_path = e4["d5"]["recommended_input_variant"] == "raw"
        out.append({"id": "map_no_drop", "text": "không làm giảm mAP nhánh B", "value": diff["diff"],
                    "ci95": diff["ci95"], "status": "dat" if raw_path or (diff["ci95"][0] or -1) > 0 else "khong_dat",
                    "note": "D5 giữ khung gốc: detector không thấy ảnh enhancer" if raw_path else None})
    else:
        out.append({"id": "map_no_drop", "text": "không làm giảm mAP nhánh B", "value": None, "status": "cho_do"})
    return out


SUBCOMMANDS = ("physical", "flicker", "speed", "iqa", "e4")
_RUNNERS: dict[str, Callable[[argparse.Namespace, Ctx, VariantSet], dict[str, Any]]] = {
    "physical": cmd_physical, "flicker": cmd_flicker, "speed": cmd_speed, "iqa": cmd_iqa, "e4": cmd_e4}


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def run(a: argparse.Namespace) -> dict[str, Any]:
    ctx = make_ctx(a)
    vs = build_variants(a, ctx.root)
    head = {"phase": "14", "created": datetime.now(UTC).isoformat(timespec="seconds"), "variants": vs.info} | ctx.meta
    if a.cmd != "all":
        doc = head | {a.cmd: _RUNNERS[a.cmd](a, ctx, vs)}
        if a.cmd == "e4":
            _write(a.e4_out or ctx.root / E4_OUT_REL, doc["e4"])
        if a.out:
            _write(a.out, doc)
        return doc
    doc = dict(head)
    for name in SUBCOMMANDS:
        doc[name] = _RUNNERS[name](a, ctx, vs)
    _write(a.e4_out or ctx.root / E4_OUT_REL, doc["e4"])
    doc["e4"] = {k: doc["e4"][k] for k in ("cells", "paired_diff_vs_raw", "d5", "op_threshold")} | {
        "file": str(a.e4_out or ctx.root / E4_OUT_REL)}
    doc["ab"] = _ab_summary(ctx.root, a.ab_result, a.ab_key)
    doc["criteria_branch_a"] = criteria(doc)
    _write(a.out or ctx.root / OUT_REL, doc)
    return doc


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m xla.eval.enhance_eval", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in (*SUBCOMMANDS, "all"):
        p = sub.add_parser(name)
        add_variant_args(p)
        p.add_argument("--split-json", type=Path, default=None, help=f"mặc định {SPLIT_REL}")
        p.add_argument("--frames-dir", type=Path, default=None, help=f"mặc định {FRAMES_REL}")
        p.add_argument("--boot", type=int, default=BOOT_N, help="số mẫu bootstrap theo khối (0 = bỏ CI)")
        p.add_argument("--seed", type=int, default=BOOT_SEED)
        p.add_argument("--limit", type=int, default=0, help="chỉ N khung đầu mỗi tập (chạy thử, không chính thức)")
        p.add_argument("--lock-timeout", type=float, default=LOCK_TIMEOUT_S, help="giây chờ khoá GPU")
        p.add_argument("--out", type=Path, default=None,
                       help=f"ghi JSON ra file (all: mặc định {OUT_REL})")
        p.add_argument("--json", action="store_true", help="in JSON một dòng")
        if name in ("flicker", "all"):
            p.add_argument("--manifest-csv", type=Path, default=None, help=f"mặc định {MANIFEST_CSV_REL}")
        if name in ("speed", "all"):
            p.add_argument("--n-frames", type=int, default=SPEED_N)
            p.add_argument("--warmup", type=int, default=SPEED_WARMUP)
        if name in ("iqa", "all"):
            p.add_argument("--iqa-metrics", default=",".join(IQA_METRICS))
            p.add_argument("--iqa-device", choices=("cuda", "cpu"), default="cuda")
        if name in ("e4", "all"):
            p.add_argument("--detector", type=Path, default=None, help=f"bundle detector, mặc định {DETECTOR_REL}")
            p.add_argument("--coco-weights", type=Path, default=None, help="mặc định <downloads_dir>/yolo-weights/yolo11s.pt")
            p.add_argument("--coco-device", default="0", help="thiết bị ultralytics: 0 (GPU) hoặc cpu")
            p.add_argument("--labels-dir", type=Path, default=None, help="mặc định data/real/labels")
            p.add_argument("--manifest", type=Path, default=None, help="mặc định <labels-dir>/manifest.json")
            p.add_argument("--allow-missing-labels", action="store_true")
            p.add_argument("--pred-dir", type=Path, default=None, help=f"CSV dự đoán E4, mặc định {E4_PRED_REL}")
            p.add_argument("--e4-out", type=Path, default=None, help=f"mặc định {E4_OUT_REL}")
        if name == "all":
            p.add_argument("--ab-result", type=Path, default=None, help=f"mặc định {AB_RESULT_REL} (nếu có)")
            p.add_argument("--ab-key", type=Path, default=None, help=f"mặc định {AB_KEY_REL}")
    return ap


def add_variant_args(p: argparse.ArgumentParser) -> None:
    """Cờ chọn biến thể — dùng chung với ``xla.eval.ab_page``."""
    p.add_argument("--enhancer", type=Path, default=None, help=f"bundle enhancer, mặc định {ENHANCER_REL}")
    p.add_argument("--classic", default=DEFAULT_CLASSIC, help="hàm cổ điển module:func hoặc file.py:func")
    p.add_argument("--classic-params", type=Path, default=None, help="YAML tham số cho load_params của module cổ điển")
    p.add_argument("--providers", default="cuda,cpu", help="ORT provider theo thứ tự ưu tiên: cuda,cpu | cpu")


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    try:
        doc = run(a)
    except UsageError as e:
        print(f"LỖI cách gọi: {e}", file=sys.stderr)
        return 2
    except (EvalError, EvalInputError, TestLeakError, TimeoutError, FileNotFoundError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(doc, ensure_ascii=False) if a.json else json.dumps(doc, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
