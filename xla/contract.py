"""Hợp đồng model XLA ↔ IOT-CV (SSOT: ``docs/hop-dong-model.md``): hằng số, kiểm bundle, bundle giả, kiểm hồ sơ camera.

CLI (D7, AX-01)::

    python -m xla.contract validate <bundle-dir> [--json]
    python -m xla.contract make-dummy <out-dir> [--kind detector|enhancer|verifier|enhancer-offline] [--json]
    python -m xla.contract check-profile <camera_profile.yaml|.json> [--json]

Mã thoát: 0 = đạt, 1 = bundle/hồ sơ sai hợp đồng, 2 = sai cách gọi (thiếu tham số, thư mục không tồn tại).
``validate`` không ``--json`` in danh sách lỗi dạng JSON một dòng (``[]`` = đạt); có ``--json`` in object
``{"bundle", "ok", "errors": [{"section", "message"}]}``. Mỗi lỗi mở đầu bằng mục hợp đồng, ví dụ ``§2 chữ ký: ...``.
``make-dummy`` không ``--kind`` sinh cặp v1 (detector + enhancer); ``--kind`` sinh đúng một bundle.

Mục lỗi theo kind: detector §2, enhancer §3 (classic §12), verifier §9, enhancer-offline §11, ``recommended_track`` §10;
khoá meta chung và cấu trúc bundle §4.

Quy ước bổ sung không có trong hợp đồng (chỉ chi tiết hoá, không đổi nghĩa):

- ``expected.json`` của enhancer / enhancer-offline: ``sha256_out`` = sha256 của ``frame_out.tobytes()`` (uint8 HWC BGR,
  C-order), ``mean`` = trung bình từng kênh [b, g, r].
- ``expected.json`` của verifier: ``{"000.jpg": {"scores": [...]}}`` cùng thứ tự hộp trong ``golden/boxes.json``
  (``{"000.jpg": [[x1, y1, x2, y2], ...]}``; danh sách rỗng = N = 0).
- Golden của enhancer-offline với ``temporal_window`` T > 1: 3 khung golden là một chuỗi liên tiếp, cửa sổ của khung i
  là các khung ``clip(i - T//2 … i + T//2)`` (lặp khung biên, như đầu/cuối chuyến).
- ``meta.metrics``: đủ khoá theo loại model; giá trị là số, hoặc ``null`` khi chưa đo (bundle giả). Enhancer-offline:
  ``metrics`` tuỳ chọn (phase 14 ghi), có thì mỗi khoá đã biết phải là số hoặc ``null``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---- §1–§4, §9–§12: hằng số hợp đồng -----------------------------------------------------------------------------
CONTRACT_VERSION = "1.1"
CONTRACT_MAJOR = 1
FRAME_W = 640
FRAME_H = 480
FRAME_SHAPE = (FRAME_H, FRAME_W, 3)  # HWC, BGR, uint8
CROP_LEFT_V1 = 16
OPSET = 17
MODELS_V1 = ("detector", "enhancer")
MODELS = (*MODELS_V1, "verifier", "enhancer-offline")
# kind có từ minor nào của major 1 (bundle kind mới mà ghi contract_version cũ hơn là sai)
KIND_SINCE_MINOR = {"detector": 0, "enhancer": 0, "verifier": 1, "enhancer-offline": 1}
SECTION = {"detector": "2", "enhancer": "3", "verifier": "9", "enhancer-offline": "11"}

INPUT_NAME = "frame"
INPUT_DTYPE = "uint8"
BOXES_NAME = "boxes"
FRAMES_NAME = "frames"
DETECTOR_OUTPUTS = {"boxes": ("float32", ("N", 4)), "scores": ("float32", ("N",))}
ENHANCER_OUTPUTS = {"frame_out": ("uint8", FRAME_SHAPE)}

# §9 verifier
VERIFIER_INPUTS = {INPUT_NAME: ("uint8", FRAME_SHAPE), BOXES_NAME: ("float32", ("N", 4))}
VERIFIER_OUTPUTS = {"scores": ("float32", ("N",))}
VERIFIER_CROP_PX = (64, 96)
VERIFIER_ROI_MODE = "output_half_pixel"  # = torchvision.ops.roi_align(..., aligned=False)
VERIFIER_SAMPLING_RATIO = 2
VERIFIER_INPUT_VARIANT = "raw"
VERIFIER_PROBE_BOX = (100.0, 100.0, 140.0, 180.0)  # hộp chạy thử lúc IOT-CV nạp (§9)

# §11 enhancer-offline
OFFLINE_SCALE = 2
OFFLINE_OUTPUT_SIZE = [FRAME_W * OFFLINE_SCALE, FRAME_H * OFFLINE_SCALE]  # [width, height]
OFFLINE_SHAPE = (FRAME_H * OFFLINE_SCALE, FRAME_W * OFFLINE_SCALE, 3)
OFFLINE_OUTPUTS = {"frame_out": ("uint8", OFFLINE_SHAPE)}

OUTPUTS = {
    "detector": DETECTOR_OUTPUTS,
    "enhancer": ENHANCER_OUTPUTS,
    "verifier": VERIFIER_OUTPUTS,
    "enhancer-offline": OFFLINE_OUTPUTS,
}

GOLDEN_FRAMES = ("000.jpg", "001.jpg", "002.jpg")
EXPECTED_FILE = "expected.json"
BOXES_FILE = "boxes.json"
META_FILE = "meta.json"

META_COMMON = {
    "contract_version": str,
    "model": str,
    "version": str,
    "onnx_sha256": str,
    "frame_size": list,
    "crop_left": int,
    "camera_profile_version": str,
    "train_run": str,
    "metrics": dict,
    "ms_per_frame": dict,
    "created": str,
}
# khoá chung mà kind được bỏ (enhancer-offline không chạy trực tiếp: không có ngân sách ms, metrics phase 14 mới ghi)
META_COMMON_OPTIONAL = {"enhancer-offline": ("metrics", "ms_per_frame")}
META_DETECTOR = {"imgsz": int, "score_floor": float, "nms_iou": float, "max_det": int, "conf_threshold": float}
META_DETECTOR_OPTIONAL = {"input_variant": ("raw", "enhanced")}
# khoá riêng theo kind: bắt buộc / tuỳ chọn. Khoá riêng của kind khác xuất hiện → lỗi.
META_KIND: dict[str, dict[str, Any]] = {
    "detector": META_DETECTOR,
    "enhancer": {},
    "verifier": {
        "crop_px": int,
        "context_scale": float,
        "min_side_px": float,
        "conf_threshold": float,
        "input_variant": str,
        "backbone": str,
    },
    "enhancer-offline": {
        "scale": int,
        "output_size": list,
        "temporal_window": int,
        "network": (str, dict),
        "fp16": bool,
        "peak_vram_mb": float,
        "s_per_frame": float,
        "machine": (str, dict),
    },
}
META_KIND_OPTIONAL: dict[str, dict[str, Any]] = {
    "detector": {"input_variant": str, "recommended_track": dict},
    "enhancer": {"network": (str, dict), "classic_params_sha256": str},
    "verifier": {},
    "enhancer-offline": {},
}
METRICS_KEYS = {
    "detector": ("map50", "map50_95", "precision", "recall", "fp_per_neg_frame"),
    "enhancer": ("pink_frac_drop", "dark_channel_drop", "flicker_ratio"),
    "verifier": ("fp_per_cal_frame_before", "fp_per_cal_frame_after", "recall_oof_before", "recall_oof_after"),
    "enhancer-offline": ("fp_cal_raw", "fp_cal_offline", "fp_holdout_raw", "fp_holdout_offline"),
}
METRICS_OPTIONAL = {"enhancer-offline"}  # có metrics thì khoá nào có mặt mới kiểm
MS_PER_FRAME_KEYS = {"device": str, "runtime": str, "p50": float, "p95": float}
TRACK_KEYS = {"k": int, "n": int, "iou": float, "max_gap": int}  # §10 recommended_track

# §4 / §9 / §11 dung sai golden
DET_IOU_MIN = 0.95
DET_SCORE_ABS_MAX = 0.02
VER_SCORE_ABS_MAX = 0.02
ENH_MAE_MAX = 1.0
ENH_ABS_MAX = 8

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
BUNDLE_DIR_RE = re.compile(r"^(detector|enhancer|verifier|enhancer-offline)-v(\d+\.\d+\.\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ONNX_DTYPES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 10: "float16", 11: "float64"}

DUMMY_BOX = (100.0, 100.0, 150.0, 160.0)
DUMMY_SCORE = 0.9
DUMMY_VERIFIER_SCORE = 0.8
DUMMY_VERIFIER_BOXES = {
    "000.jpg": [list(VERIFIER_PROBE_BOX), [300.0, 200.0, 360.0, 330.0]],
    "001.jpg": [[20.0, 0.0, 70.0, 40.0]],
    "002.jpg": [],  # N = 0
}
DUMMY_VERSION = "0.0.0"


class _Errors:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, section: str, message: str) -> None:
        self.items.append((section, message))

    def as_strings(self) -> list[str]:
        return [f"§{s}: {m}" for s, m in self.items]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_ok(value: Any, typ: Any) -> bool:
    if isinstance(typ, tuple):
        return any(_type_ok(value, t) for t in typ)
    if typ is float:
        return _is_number(value)
    if typ is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, typ)


def _type_name(typ: Any) -> str:
    return " | ".join(t.__name__ for t in typ) if isinstance(typ, tuple) else typ.__name__


def offline_temporal_window(meta: dict[str, Any]) -> int:
    """``temporal_window`` của enhancer-offline nếu hợp lệ (nguyên dương, lẻ), không thì 1 (lỗi meta báo riêng)."""
    t = meta.get("temporal_window")
    return t if _type_ok(t, int) and t >= 1 and t % 2 == 1 else 1


def model_inputs(model: str, meta: dict[str, Any] | None = None) -> dict[str, tuple[str, tuple[Any, ...]]]:
    """Chữ ký input theo kind (thứ tự = thứ tự input của graph)."""
    if model == "verifier":
        return dict(VERIFIER_INPUTS)
    if model == "enhancer-offline":
        t = offline_temporal_window(meta or {})
        if t > 1:
            return {FRAMES_NAME: (INPUT_DTYPE, (t, *FRAME_SHAPE))}
    return {INPUT_NAME: (INPUT_DTYPE, FRAME_SHAPE)}


# ---- kiểm meta.json ---------------------------------------------------------------------------------------------
def _check_contract_version(meta: dict[str, Any], model: str, err: _Errors) -> None:
    cv = meta.get("contract_version")
    if not isinstance(cv, str):
        return
    m = re.match(r"^(\d+)\.(\d+)$", cv)
    if not m:
        err.add("4 contract_version", f"'{cv}' không đúng dạng MAJOR.MINOR")
    elif int(m.group(1)) != CONTRACT_MAJOR:
        err.add("4 contract_version", f"major {m.group(1)} khác major được hỗ trợ {CONTRACT_MAJOR}")
    elif int(m.group(2)) < KIND_SINCE_MINOR.get(model, 0):
        err.add(
            "4 contract_version",
            f"'{cv}' cũ hơn {CONTRACT_MAJOR}.{KIND_SINCE_MINOR[model]}, phiên bản đầu tiên có kind '{model}'",
        )


def _check_common(meta: dict[str, Any], model: str, dir_version: str | None, err: _Errors) -> None:
    if isinstance(meta.get("model"), str) and meta["model"] != model:
        err.add("4 meta.json", f"model '{meta['model']}' khác loại bundle '{model}'")
    ver = meta.get("version")
    if isinstance(ver, str):
        if not SEMVER_RE.match(ver):
            err.add("4 meta.json", f"version '{ver}' không phải semver")
        elif dir_version is not None and ver != dir_version:
            err.add("4 meta.json", f"version '{ver}' khác tên thư mục (v{dir_version})")
    sha = meta.get("onnx_sha256")
    if isinstance(sha, str) and not SHA256_RE.match(sha):
        err.add("4 meta.json", "onnx_sha256 không phải 64 ký tự hex thường")
    if isinstance(meta.get("frame_size"), list) and meta["frame_size"] != [FRAME_W, FRAME_H]:
        err.add("4 meta.json", f"frame_size {meta['frame_size']} khác [{FRAME_W}, {FRAME_H}]")
    cl = meta.get("crop_left")
    if _type_ok(cl, int) and not 0 <= cl < FRAME_W:
        err.add("4 meta.json", f"crop_left {cl} ngoài [0, {FRAME_W})")
    cpv = meta.get("camera_profile_version")
    if isinstance(cpv, str) and not SEMVER_RE.match(cpv):
        err.add("4 meta.json", f"camera_profile_version '{cpv}' không phải semver")
    created = meta.get("created")
    if isinstance(created, str):
        try:
            datetime.fromisoformat(created)
        except ValueError:
            err.add("4 meta.json", f"created '{created}' không phải ISO 8601")
    if isinstance(meta.get("metrics"), dict):
        for k in METRICS_KEYS[model]:
            if k not in meta["metrics"]:
                if model not in METRICS_OPTIONAL:
                    err.add("4 meta.json", f"metrics thiếu '{k}'")
            elif meta["metrics"][k] is not None and not _is_number(meta["metrics"][k]):
                err.add("4 meta.json", f"metrics.{k} phải là số hoặc null")
    if isinstance(meta.get("ms_per_frame"), dict):
        for k, typ in MS_PER_FRAME_KEYS.items():
            if not _type_ok(meta["ms_per_frame"].get(k), typ):
                err.add("4 meta.json", f"ms_per_frame.{k} thiếu hoặc sai kiểu ({_type_name(typ)})")


def _check_track(track: dict[str, Any], err: _Errors) -> None:
    sec = "10 recommended_track"
    for k, typ in TRACK_KEYS.items():
        if not _type_ok(track.get(k), typ):
            err.add(sec, f"'{k}' thiếu hoặc sai kiểu ({_type_name(typ)})")
    k, n, iou, gap = (track.get(x) for x in TRACK_KEYS)
    if _type_ok(k, int) and k < 1:
        err.add(sec, f"k {k} phải nguyên dương")
    if _type_ok(n, int) and n < 1:
        err.add(sec, f"n {n} phải nguyên dương")
    if _type_ok(k, int) and _type_ok(n, int) and k >= 1 and n >= 1 and k > n:
        err.add(sec, f"k {k} > n {n} (cần k ≤ n)")
    if _is_number(iou) and not 0.0 < iou <= 1.0:
        err.add(sec, f"iou {iou} ngoài (0, 1]")
    if _type_ok(gap, int) and gap < 0:
        err.add(sec, f"max_gap {gap} phải ≥ 0")


def _check_kind(meta: dict[str, Any], model: str, err: _Errors) -> None:
    if model == "detector":
        iv = meta.get("input_variant", "raw")
        if iv not in META_DETECTOR_OPTIONAL["input_variant"]:
            err.add("4 meta.json", f"input_variant '{iv}' không thuộc {META_DETECTOR_OPTIONAL['input_variant']}")
        for k in ("score_floor", "nms_iou", "conf_threshold"):
            if _is_number(meta.get(k)) and not 0.0 <= meta[k] <= 1.0:
                err.add("4 meta.json", f"{k} {meta[k]} ngoài [0, 1]")
        if "recommended_track" in meta and isinstance(meta["recommended_track"], dict):
            _check_track(meta["recommended_track"], err)
    elif model == "enhancer":
        sha = meta.get("classic_params_sha256")
        if meta.get("network") == "classic" and sha is None:
            err.add("12 meta.json", "network 'classic' cần classic_params_sha256")
        if isinstance(sha, str) and not SHA256_RE.match(sha):
            err.add("12 meta.json", "classic_params_sha256 không phải 64 ký tự hex thường")
    elif model == "verifier":
        cp = meta.get("crop_px")
        if _type_ok(cp, int) and cp not in VERIFIER_CROP_PX:
            err.add("9 meta.json", f"crop_px {cp} không thuộc {list(VERIFIER_CROP_PX)}")
        for k in ("context_scale", "min_side_px"):
            if _is_number(meta.get(k)) and meta[k] <= 0:
                err.add("9 meta.json", f"{k} {meta[k]} phải > 0")
        if _is_number(meta.get("conf_threshold")) and not 0.0 <= meta["conf_threshold"] <= 1.0:
            err.add("9 meta.json", f"conf_threshold {meta['conf_threshold']} ngoài [0, 1]")
        if isinstance(meta.get("input_variant"), str) and meta["input_variant"] != VERIFIER_INPUT_VARIANT:
            err.add("9 meta.json", f"input_variant '{meta['input_variant']}' phải là '{VERIFIER_INPUT_VARIANT}'")
    else:  # enhancer-offline
        if _type_ok(meta.get("scale"), int) and meta["scale"] != OFFLINE_SCALE:
            err.add("11 meta.json", f"scale {meta['scale']} khác {OFFLINE_SCALE}")
        if isinstance(meta.get("output_size"), list) and meta["output_size"] != OFFLINE_OUTPUT_SIZE:
            err.add("11 meta.json", f"output_size {meta['output_size']} khác {OFFLINE_OUTPUT_SIZE}")
        t = meta.get("temporal_window")
        if _type_ok(t, int) and (t < 1 or t % 2 == 0):
            err.add("11 meta.json", f"temporal_window {t} phải nguyên dương và lẻ")
        for k in ("peak_vram_mb", "s_per_frame"):
            if _is_number(meta.get(k)) and meta[k] < 0:
                err.add("11 meta.json", f"{k} {meta[k]} phải ≥ 0")


def _check_meta(meta: dict[str, Any], model: str, dir_version: str | None, err: _Errors) -> None:
    sec = "4" if model in MODELS_V1 else SECTION[model]
    optional_common = META_COMMON_OPTIONAL.get(model, ())
    for key, typ in META_COMMON.items():
        if key not in meta:
            if key not in optional_common:
                err.add("4 meta.json", f"thiếu khoá bắt buộc '{key}'")
        elif not _type_ok(meta[key], typ):
            err.add("4 meta.json", f"khoá '{key}' sai kiểu: cần {_type_name(typ)}, gặp {type(meta[key]).__name__}")
    own = {**META_KIND_OPTIONAL[model], **META_KIND[model]}
    for key, typ in own.items():
        if key not in meta:
            if key in META_KIND[model]:
                err.add(f"{sec} meta.json", f"thiếu khoá bắt buộc '{key}'")
        elif not _type_ok(meta[key], typ):
            err.add(f"{sec} meta.json", f"khoá '{key}' sai kiểu: cần {_type_name(typ)}, gặp {type(meta[key]).__name__}")
    foreign = {k for kind in MODELS for k in (*META_KIND[kind], *META_KIND_OPTIONAL[kind])} - set(own)
    for key in sorted(foreign & set(meta)):
        err.add(f"{sec} meta.json", f"khoá '{key}' không dành cho kind '{model}'")
    _check_contract_version(meta, model, err)
    _check_common(meta, model, dir_version, err)
    _check_kind(meta, model, err)


# ---- kiểm chữ ký graph -----------------------------------------------------------------------------------------
def _dims(value_info: Any) -> tuple[Any, ...]:
    out = []
    for d in value_info.type.tensor_type.shape.dim:
        out.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return tuple(out)


def _dtype(value_info: Any) -> str:
    t = value_info.type.tensor_type.elem_type
    return _ONNX_DTYPES.get(t, str(t))


def _shape_ok(dims: tuple[Any, ...], want: tuple[Any, ...]) -> bool:
    """Chiều cố định phải khớp; chiều ghi tên (``"N"``) trong hợp đồng là chiều động, nhận mọi giá trị."""
    return len(dims) == len(want) and all(isinstance(w, str) or g == w for g, w in zip(dims, want, strict=True))


def _check_signature(onnx_path: Path, model: str, err: _Errors, meta: dict[str, Any] | None = None) -> bool:
    import onnx

    sec = SECTION[model]
    try:
        m = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as e:  # noqa: BLE001 - file hỏng: báo đúng mục, không văng
        err.add(f"{sec} chữ ký", f"không đọc được {onnx_path.name}: {e}")
        return False
    n_before = len(err.items)
    opset = {o.domain: o.version for o in m.opset_import}.get("", None)
    if opset != OPSET:
        err.add(f"{sec} chữ ký", f"opset {opset} khác {OPSET}")
    want_in = model_inputs(model, meta)
    inits = {i.name for i in m.graph.initializer}
    inputs = [i for i in m.graph.input if i.name not in inits]
    if [i.name for i in inputs] != list(want_in):
        names = ", ".join(f"'{n}'" for n in want_in)
        err.add(f"{sec} chữ ký", f"input phải đúng [{names}] theo thứ tự, gặp {[i.name for i in inputs]}")
    else:
        for i in inputs:
            dtype, shape = want_in[i.name]
            if _dtype(i) != dtype:
                err.add(f"{sec} chữ ký", f"input '{i.name}' dtype {_dtype(i)}, cần {dtype}")
            if not _shape_ok(_dims(i), shape):
                err.add(f"{sec} chữ ký", f"input '{i.name}' shape {list(_dims(i))}, cần {list(shape)}")
    spec = OUTPUTS[model]
    outs = {o.name: o for o in m.graph.output}
    if sorted(outs) != sorted(spec):
        err.add(f"{sec} chữ ký", f"output phải là {sorted(spec)}, gặp {sorted(outs)}")
    else:
        for name, (dtype, shape) in spec.items():
            o = outs[name]
            if _dtype(o) != dtype:
                err.add(f"{sec} chữ ký", f"output '{name}' dtype {_dtype(o)}, cần {dtype}")
            if not _shape_ok(_dims(o), shape):
                err.add(f"{sec} chữ ký", f"output '{name}' shape {list(_dims(o))}, cần {list(shape)}")
    return len(err.items) == n_before


# ---- chạy golden bằng ORT CPU ------------------------------------------------------------------------------------
def decode_frame(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"không giải mã được {path}")
    return np.ascontiguousarray(img)


def _session(onnx_path: Path) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def run_feeds(onnx_path: Path, feeds: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """Chạy ORT CPU một lần mỗi khoá của ``feeds`` (mỗi giá trị là dict input → mảng); trả dict output theo tên."""
    sess = _session(onnx_path)
    names = [o.name for o in sess.get_outputs()]
    return {k: dict(zip(names, sess.run(None, v), strict=True)) for k, v in feeds.items()}


def run_model(onnx_path: Path, frames: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Model một input ``frame`` (detector, enhancer, enhancer-offline ảnh đơn)."""
    return run_feeds(onnx_path, {k: {INPUT_NAME: v} for k, v in frames.items()})


def temporal_stack(frames: list[np.ndarray], i: int, t: int) -> np.ndarray:
    """Cửa sổ ``[t, H, W, 3]`` quanh khung ``i`` (khung giữa = khung đích), lặp khung biên ở đầu/cuối chuỗi (§11)."""
    h = t // 2
    idx = [min(max(j, 0), len(frames) - 1) for j in range(i - h, i + h + 1)]
    return np.ascontiguousarray(np.stack([frames[j] for j in idx]))


def _golden_feeds(
    model: str, frames: dict[str, np.ndarray], boxes: dict[str, Any], meta: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    if model == "verifier":
        return {
            n: {INPUT_NAME: f, BOXES_NAME: np.asarray(boxes[n], np.float32).reshape(-1, 4)} for n, f in frames.items()
        }
    t = offline_temporal_window(meta) if model == "enhancer-offline" else 1
    if t > 1:
        seq = [frames[n] for n in GOLDEN_FRAMES]
        return {n: {FRAMES_NAME: temporal_stack(seq, i, t)} for i, n in enumerate(GOLDEN_FRAMES)}
    return {n: {INPUT_NAME: f} for n, f in frames.items()}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def compare_detector(got: dict[str, np.ndarray], exp: dict[str, Any], conf: float) -> list[str]:
    """So một khung với ``expected`` theo dung sai §4; trả danh sách sai khác (rỗng = khớp)."""
    gb, gs = np.asarray(got["boxes"], np.float32).reshape(-1, 4), np.asarray(got["scores"], np.float32).reshape(-1)
    eb = np.asarray(exp.get("boxes", []), np.float32).reshape(-1, 4)
    es = np.asarray(exp.get("scores", []), np.float32).reshape(-1)
    gk, ek = gs >= conf, es >= conf
    gb, gs, eb, es = gb[gk], gs[gk], eb[ek], es[ek]
    if len(gb) != len(eb):
        return [f"{len(gb)} hộp ≥ conf_threshold, golden có {len(eb)}"]
    used: set[int] = set()
    problems = []
    for i in range(len(eb)):
        best = max(((j, _iou(eb[i], gb[j])) for j in range(len(gb)) if j not in used), key=lambda t: t[1], default=None)
        if best is None or best[1] < DET_IOU_MIN or abs(float(gs[best[0]] - es[i])) > DET_SCORE_ABS_MAX:
            problems.append(f"hộp golden {i} không có hộp khớp (IoU ≥ {DET_IOU_MIN}, |Δscore| ≤ {DET_SCORE_ABS_MAX})")
        else:
            used.add(best[0])
    return problems


def compare_verifier(got: np.ndarray, exp: dict[str, Any]) -> list[str]:
    """§9: cùng số điểm, cùng thứ tự hộp, |Δscore| ≤ 0,02."""
    g = np.asarray(got, np.float32).reshape(-1)
    e = np.asarray(exp.get("scores", []), np.float32).reshape(-1)
    if g.shape != e.shape:
        return [f"{g.size} điểm, golden có {e.size}"]
    bad = np.flatnonzero(np.abs(g - e) > VER_SCORE_ABS_MAX)
    return [f"hộp {i}: score {g[i]:.4f}, golden {e[i]:.4f} (|Δ| ≤ {VER_SCORE_ABS_MAX})" for i in bad]


def compare_enhancer(got: np.ndarray, ref: np.ndarray) -> list[str]:
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    mae, mx = float(diff.mean()), int(diff.max())
    if mae > ENH_MAE_MAX or mx > ENH_ABS_MAX:
        return [f"sai khác trung bình {mae:.3f} (≤ {ENH_MAE_MAX}), lớn nhất {mx} (≤ {ENH_ABS_MAX})"]
    return []


def _check_strip(f: np.ndarray, cl: Any, sec: str, frame: str, err: _Errors) -> None:
    """Cột ``[0, cl)`` của ảnh ra phải lặp lại cột ``cl`` (§3; §11 với ``cl`` = 2·crop_left)."""
    if _type_ok(cl, int) and 0 < cl < f.shape[1] and not np.array_equal(f[:, :cl], np.repeat(f[:, cl : cl + 1], cl, 1)):
        err.add(f"{sec} output", f"{frame}: cột [0, {cl}) không lặp lại cột {cl}")


def _check_runtime_output(
    model: str, out: dict[str, np.ndarray], meta: dict[str, Any], frame: str, n_boxes: int, err: _Errors
) -> None:
    if model == "detector":
        b, s = out["boxes"], out["scores"]
        if (
            b.dtype != np.float32
            or s.dtype != np.float32
            or b.ndim != 2
            or b.shape[1:] != (4,)
            or s.shape != (b.shape[0],)
        ):
            err.add("2 output", f"{frame}: boxes {b.dtype}{list(b.shape)}, scores {s.dtype}{list(s.shape)} sai")
            return
        if _type_ok(meta.get("max_det"), int) and b.shape[0] > meta["max_det"]:
            err.add("2 output", f"{frame}: {b.shape[0]} hộp > max_det {meta['max_det']}")
        if len(s) and (np.any(np.diff(s) > 0) or s.min() < 0 or s.max() > 1):
            err.add("2 output", f"{frame}: scores phải trong [0,1] và giảm dần")
        if len(b) and (
            np.any(b[:, 0] >= b[:, 2])
            or np.any(b[:, 1] >= b[:, 3])
            or b.min() < 0
            or np.any(b[:, [0, 2]] > FRAME_W)
            or np.any(b[:, [1, 3]] > FRAME_H)
        ):
            err.add("2 output", f"{frame}: hộp phải x1<x2, y1<y2 và nằm trong [0,{FRAME_W}]×[0,{FRAME_H}]")
    elif model == "verifier":
        s = out["scores"]
        if s.dtype != np.float32 or s.shape != (n_boxes,):
            err.add("9 output", f"{frame}: scores {s.dtype}{list(s.shape)}, cần float32[{n_boxes}]")
            return
        if len(s) and (s.min() < 0 or s.max() > 1):
            err.add("9 output", f"{frame}: scores phải trong [0, 1]")
    else:
        shape = OFFLINE_SHAPE if model == "enhancer-offline" else FRAME_SHAPE
        sec = SECTION[model]
        f = out["frame_out"]
        if f.dtype != np.uint8 or f.shape != shape:
            err.add(f"{sec} output", f"{frame}: frame_out {f.dtype}{list(f.shape)}, cần uint8{list(shape)}")
            return
        cl = meta.get("crop_left")
        scale = OFFLINE_SCALE if model == "enhancer-offline" else 1
        _check_strip(f, cl * scale if _type_ok(cl, int) else None, sec, frame, err)


def _load_golden_boxes(gdir: Path, err: _Errors) -> dict[str, Any] | None:
    try:
        boxes = json.loads((gdir / BOXES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        err.add("9 golden", f"{BOXES_FILE} không đọc được: {e}")
        return None
    if not isinstance(boxes, dict) or sorted(boxes) != sorted(GOLDEN_FRAMES):
        err.add("9 golden", f"{BOXES_FILE} phải có đúng khoá {list(GOLDEN_FRAMES)}")
        return None
    for name, lst in boxes.items():
        ok = isinstance(lst, list) and all(
            isinstance(b, list) and len(b) == 4 and all(_is_number(x) for x in b) for b in lst
        )
        if not ok:
            err.add("9 golden", f"{BOXES_FILE}: {name} phải là danh sách [x1, y1, x2, y2] số")
            return None
    return boxes


def _check_golden(bundle: Path, model: str, onnx_path: Path, meta: dict[str, Any], run: bool, err: _Errors) -> None:
    gsec = {"verifier": "9 golden", "enhancer-offline": "11 golden"}.get(model, "4 golden")
    gdir = bundle / "golden"
    wanted = [*GOLDEN_FRAMES, EXPECTED_FILE]
    if model in ("enhancer", "enhancer-offline"):
        wanted += [n.replace(".jpg", ".out.png") for n in GOLDEN_FRAMES]
    if model == "verifier":
        wanted.append(BOXES_FILE)
    missing = [n for n in wanted if not (gdir / n).is_file()]
    if missing:
        err.add(gsec, f"thiếu {missing} trong golden/")
        return
    try:
        expected = json.loads((gdir / EXPECTED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        err.add(gsec, f"expected.json không đọc được: {e}")
        return
    if not isinstance(expected, dict) or sorted(expected) != sorted(GOLDEN_FRAMES):
        err.add(gsec, f"expected.json phải có đúng khoá {list(GOLDEN_FRAMES)}")
        return
    boxes: dict[str, Any] = {}
    if model == "verifier":
        loaded = _load_golden_boxes(gdir, err)
        if loaded is None:
            return
        boxes = loaded
    if not run:
        return
    try:
        frames = {n: decode_frame(gdir / n) for n in GOLDEN_FRAMES}
        bad = {n: f.shape for n, f in frames.items() if f.shape != FRAME_SHAPE}
        if bad:
            err.add("1 khung", f"khung golden sai cỡ {bad}, cần {list(FRAME_SHAPE)}")
            return
        outs = run_feeds(onnx_path, _golden_feeds(model, frames, boxes, meta))
    except Exception as e:  # noqa: BLE001 - ORT/giải mã lỗi: báo đúng mục
        err.add(gsec, f"chạy ORT CPU trên golden lỗi: {e}")
        return
    conf = meta.get("conf_threshold") if _is_number(meta.get("conf_threshold")) else 0.0
    shape = OFFLINE_SHAPE if model == "enhancer-offline" else FRAME_SHAPE
    for name in GOLDEN_FRAMES:
        n_before = len(err.items)
        _check_runtime_output(model, outs[name], meta, name, len(boxes.get(name, [])), err)
        if len(err.items) > n_before and model == "verifier":
            continue
        if model == "detector":
            problems = compare_detector(outs[name], expected[name], conf)
        elif model == "verifier":
            problems = compare_verifier(outs[name]["scores"], expected[name])
        else:
            ref = decode_png(gdir / name.replace(".jpg", ".out.png"))
            exp = expected[name]
            if ref.shape != shape or hashlib.sha256(ref.tobytes()).hexdigest() != exp.get("sha256_out"):
                err.add(gsec, f"{name}: .out.png không khớp sha256_out trong expected.json")
                continue
            if outs[name]["frame_out"].shape != shape:
                continue  # đã báo ở _check_runtime_output
            problems = compare_enhancer(outs[name]["frame_out"], ref)
        for p in problems:
            err.add(gsec, f"{name}: {p}")


def decode_png(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"không giải mã được {path}")
    return img


def validate_bundle(bundle_dir: str | Path, run_golden: bool = True) -> list[str]:
    """Kiểm một bundle theo hợp đồng §2–§4, §9–§12. Trả danh sách lỗi, rỗng = đạt."""
    bundle = Path(bundle_dir)
    err = _Errors()
    m = BUNDLE_DIR_RE.match(bundle.name)
    meta: dict[str, Any] = {}
    meta_path = bundle / META_FILE
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                err.add("4 meta.json", "meta.json phải là object")
        except ValueError as e:
            err.add("4 meta.json", f"không phải JSON hợp lệ: {e}")
    else:
        err.add("4 bundle", "thiếu meta.json")
    if m:
        model, dir_version = m.group(1), m.group(2)
    else:
        err.add("4 bundle", f"tên thư mục '{bundle.name}' không đúng dạng <{'|'.join(MODELS)}>-vMAJOR.MINOR.PATCH")
        model, dir_version = meta.get("model"), None
        if model not in MODELS:
            return err.as_strings()
    if meta:
        _check_meta(meta, model, dir_version, err)
    onnx_path = bundle / f"{model}.onnx"
    sig_ok = False
    if not onnx_path.is_file():
        err.add("4 bundle", f"thiếu {onnx_path.name}")
    else:
        sha = meta.get("onnx_sha256")
        if isinstance(sha, str) and SHA256_RE.match(sha) and sha256_file(onnx_path) != sha:
            err.add("4 sha256", f"sha256 của {onnx_path.name} khác meta.onnx_sha256")
        sig_ok = _check_signature(onnx_path, model, err, meta)
    # chữ ký sai thì không chạy golden (lỗi dây chuyền chỉ lặp lại cùng một lỗi)
    _check_golden(bundle, model, onnx_path, meta, run_golden and sig_ok, err)
    return err.as_strings()


# ---- bundle giả ------------------------------------------------------------------------------------------------
def _const(name: str, dtype: int, dims: list[int], vals: list[Any]) -> Any:
    from onnx import helper

    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, dims, vals))


def _strip_nodes(src: str, out: str, crop_left: int) -> list[Any]:
    """Nút ONNX: ``out`` = ``src`` [H, W, 3] với cột ``[0, crop_left)`` thay bằng lặp cột ``crop_left``."""
    from onnx import TensorProto, helper

    i64 = TensorProto.INT64
    return [
        _const(f"{out}_st", i64, [1], [crop_left]),
        _const(f"{out}_st1", i64, [1], [crop_left + 1]),
        _const(f"{out}_end", i64, [1], [FRAME_W]),
        _const(f"{out}_ax1", i64, [1], [1]),
        _const(f"{out}_rep", i64, [3], [1, crop_left, 1]),
        helper.make_node("Slice", [src, f"{out}_st", f"{out}_st1", f"{out}_ax1"], [f"{out}_col"]),
        helper.make_node("Tile", [f"{out}_col", f"{out}_rep"], [f"{out}_strip"]),
        helper.make_node("Slice", [src, f"{out}_st", f"{out}_end", f"{out}_ax1"], [f"{out}_body"]),
        helper.make_node("Concat", [f"{out}_strip", f"{out}_body"], [out], axis=1),
    ]


def roi_align_nodes(frame: str, boxes: str, out: str, crop_px: int) -> list[Any]:
    """Nút ONNX cắt ``out`` float32 ``[N, 3, crop_px, crop_px]`` (BGR 0–255) từ ``frame`` uint8 HWC theo ``boxes``
    ``[N, 4]`` xyxy px, đúng chế độ RoiAlign của §9 (``output_half_pixel``, ``sampling_ratio`` 2, ``avg``)."""
    from onnx import TensorProto, helper

    return [
        helper.make_node("Cast", [frame], [f"{out}_f"], to=TensorProto.FLOAT),
        _const(f"{out}_ax0", TensorProto.INT64, [1], [0]),
        helper.make_node("Unsqueeze", [f"{out}_f", f"{out}_ax0"], [f"{out}_nhwc"]),
        helper.make_node("Transpose", [f"{out}_nhwc"], [f"{out}_nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Shape", [boxes], [f"{out}_n"], start=0, end=1),
        helper.make_node(
            "ConstantOfShape", [f"{out}_n"], [f"{out}_bi"], value=helper.make_tensor("bi0", TensorProto.INT64, [1], [0])
        ),
        helper.make_node(
            "RoiAlign",
            [f"{out}_nchw", boxes, f"{out}_bi"],
            [out],
            mode="avg",
            output_height=crop_px,
            output_width=crop_px,
            sampling_ratio=VERIFIER_SAMPLING_RATIO,
            spatial_scale=1.0,
            coordinate_transformation_mode=VERIFIER_ROI_MODE,
        ),
    ]


def _finish_model(nodes: list[Any], name: str, inputs: list[Any], outputs: list[Any]) -> Any:
    import onnx
    from onnx import helper

    graph = helper.make_graph(nodes, name, inputs, outputs)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="xla.contract.dummy")
    m.ir_version = 8  # IR của opset 17
    onnx.checker.check_model(m)
    return m


def roi_align_probe_model(crop_px: int) -> Any:
    """Graph chỉ có phần cắt RoiAlign của §9 (input ``frame``, ``boxes``; output ``crops``), để đo parity với PyTorch."""
    from onnx import TensorProto, helper

    ins = [
        helper.make_tensor_value_info(INPUT_NAME, TensorProto.UINT8, list(FRAME_SHAPE)),
        helper.make_tensor_value_info(BOXES_NAME, TensorProto.FLOAT, ["N", 4]),
    ]
    outs = [helper.make_tensor_value_info("crops", TensorProto.FLOAT, ["N", 3, crop_px, crop_px])]
    return _finish_model(roi_align_nodes(INPUT_NAME, BOXES_NAME, "crops", crop_px), "roi_align_probe", ins, outs)


def _dummy_graph(model: str, crop_left: int, crop_px: int = VERIFIER_CROP_PX[0]) -> Any:
    from onnx import TensorProto, helper

    f32 = TensorProto.FLOAT
    frame = helper.make_tensor_value_info(INPUT_NAME, TensorProto.UINT8, list(FRAME_SHAPE))
    inputs = [frame]
    if model == "detector":
        nodes = [
            helper.make_node("Cast", [INPUT_NAME], ["f"], to=f32),
            helper.make_node("ReduceMax", ["f"], ["fmax"], keepdims=0),
            _const("zero", f32, [], [0.0]),
            helper.make_node("Mul", ["fmax", "zero"], ["dep"]),  # 0 * max(frame): output phụ thuộc input
            _const("s0", f32, [1], [DUMMY_SCORE]),
            helper.make_node("Add", ["s0", "dep"], ["scores"]),
            _const("b0", f32, [1, 4], list(DUMMY_BOX)),
            helper.make_node("Add", ["b0", "dep"], ["boxes"]),
        ]
        outputs = [
            helper.make_tensor_value_info("boxes", f32, ["N", 4]),
            helper.make_tensor_value_info("scores", f32, ["N"]),
        ]
    elif model == "enhancer":
        nodes = _strip_nodes(INPUT_NAME, "frame_out", crop_left)
        outputs = [helper.make_tensor_value_info("frame_out", TensorProto.UINT8, list(FRAME_SHAPE))]
    elif model == "verifier":
        # RoiAlign thật (đúng chế độ §9) + điểm hằng số: 0 * mean(crop) + DUMMY_VERIFIER_SCORE, shape [N]
        inputs.append(helper.make_tensor_value_info(BOXES_NAME, f32, ["N", 4]))
        nodes = [
            *roi_align_nodes(INPUT_NAME, BOXES_NAME, "crops", crop_px),
            helper.make_node("ReduceMean", ["crops"], ["cmean"], axes=[1, 2, 3], keepdims=0),
            _const("zero", f32, [], [0.0]),
            helper.make_node("Mul", ["cmean", "zero"], ["dep"]),
            _const("s0", f32, [], [DUMMY_VERIFIER_SCORE]),
            helper.make_node("Add", ["dep", "s0"], ["scores"]),
        ]
        outputs = [helper.make_tensor_value_info("scores", f32, ["N"])]
    else:  # enhancer-offline: lặp dải mép ở đầu vào rồi Resize nearest ×2 → cột [0, 2·crop_left) lặp cột 2·crop_left
        nodes = [
            *_strip_nodes(INPUT_NAME, "stripped", crop_left),
            _const("scales", f32, [3], [float(OFFLINE_SCALE), float(OFFLINE_SCALE), 1.0]),
            helper.make_node(
                "Resize",
                ["stripped", "", "scales"],
                ["frame_out"],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            ),
        ]
        outputs = [helper.make_tensor_value_info("frame_out", TensorProto.UINT8, list(OFFLINE_SHAPE))]
    return _finish_model(nodes, f"dummy_{model.replace('-', '_')}", inputs, outputs)


def _golden_frames() -> dict[str, np.ndarray]:
    """3 khung xám có hoạ tiết (gradient + ô cờ), khác nhau mỗi khung."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    out = {}
    for i, name in enumerate(GOLDEN_FRAMES):
        g = (xx * 255 // (FRAME_W - 1) + ((xx // (16 + 8 * i) + yy // (16 + 8 * i)) % 2) * 40 + i * 30) % 256
        out[name] = np.repeat(g.astype(np.uint8)[:, :, None], 3, axis=2)
    return out


def _dummy_meta_kind(model: str) -> dict[str, Any]:
    if model == "detector":
        return {
            "imgsz": 640,
            "score_floor": 0.05,
            "nms_iou": 0.7,
            "max_det": 100,
            "conf_threshold": 0.25,
            "input_variant": "raw",
        }
    if model == "verifier":
        return {
            "crop_px": VERIFIER_CROP_PX[0],
            "context_scale": 1.5,
            "min_side_px": 32,
            "conf_threshold": 0.5,
            "input_variant": VERIFIER_INPUT_VARIANT,
            "backbone": "dummy-roialign-const",
        }
    if model == "enhancer-offline":
        return {
            "scale": OFFLINE_SCALE,
            "output_size": list(OFFLINE_OUTPUT_SIZE),
            "temporal_window": 1,
            "network": "dummy-nearest-x2",
            "fp16": False,
            "peak_vram_mb": 0.0,
            "s_per_frame": 0.0,
            "machine": "none (bundle giả)",
        }
    return {}


def make_dummy_bundle(
    out_dir: str | Path, model: str, version: str = DUMMY_VERSION, crop_left: int = CROP_LEFT_V1
) -> Path:
    """Dựng bundle giả đúng hợp đồng ở ``out_dir/<model>-v<version>/`` và trả đường dẫn thư mục đó."""
    import cv2
    import onnx
    import onnxruntime as ort

    if model not in MODELS:
        raise ValueError(f"model phải thuộc {MODELS}")
    bundle = Path(out_dir) / f"{model}-v{version}"
    gdir = bundle / "golden"
    gdir.mkdir(parents=True, exist_ok=True)
    onnx_path = bundle / f"{model}.onnx"
    onnx.save(_dummy_graph(model, crop_left), str(onnx_path))
    for name, img in _golden_frames().items():
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"không mã hoá được {name}")
        (gdir / name).write_bytes(buf.tobytes())
    kind_meta = _dummy_meta_kind(model)
    boxes = DUMMY_VERIFIER_BOXES if model == "verifier" else {}
    if boxes:
        (gdir / BOXES_FILE).write_text(json.dumps(boxes, indent=2), encoding="utf-8")
    frames = {n: decode_frame(gdir / n) for n in GOLDEN_FRAMES}
    outs = run_feeds(onnx_path, _golden_feeds(model, frames, boxes, kind_meta))
    expected: dict[str, Any] = {}
    for name, o in outs.items():
        if model == "detector":
            expected[name] = {"boxes": o["boxes"].tolist(), "scores": o["scores"].tolist()}
        elif model == "verifier":
            expected[name] = {"scores": o["scores"].tolist()}
        else:
            f = o["frame_out"]
            ok, buf = cv2.imencode(".png", f)
            if not ok:
                raise RuntimeError(f"không mã hoá được PNG cho {name}")
            (gdir / name.replace(".jpg", ".out.png")).write_bytes(buf.tobytes())
            expected[name] = {
                "sha256_out": hashlib.sha256(f.tobytes()).hexdigest(),
                "mean": f.reshape(-1, 3).mean(0).tolist(),
            }
    (gdir / EXPECTED_FILE).write_text(json.dumps(expected, indent=2), encoding="utf-8")
    meta: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model": model,
        "version": version,
        "onnx_sha256": sha256_file(onnx_path),
        "frame_size": [FRAME_W, FRAME_H],
        "crop_left": crop_left,
        "camera_profile_version": "0.0.0",
        "train_run": "dummy",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if model not in METRICS_OPTIONAL:
        meta["metrics"] = {k: None for k in METRICS_KEYS[model]}
    if "ms_per_frame" not in META_COMMON_OPTIONAL.get(model, ()):
        meta["ms_per_frame"] = {
            "device": "none (bundle giả)",
            "runtime": f"onnxruntime {ort.__version__} CPU EP",
            "p50": 0.0,
            "p95": 0.0,
        }
    meta.update(kind_meta)
    (bundle / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


# ---- §6 hồ sơ camera ---------------------------------------------------------------------------------------------
def camera_profile_schema() -> dict[str, Any]:
    from xla.paths import repo_root

    return json.loads((repo_root() / "configs" / "camera_profile.schema.json").read_text(encoding="utf-8"))


def _ordered_violations(node: Any, schema: dict[str, Any], root: dict[str, Any], path: str, out: list[str]) -> None:
    """Kiểm khoá mở rộng ``x-ordered`` (JSON Schema không so được hai trường): các khoá liệt kê phải tăng dần."""
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        schema = root["$defs"][ref]
    if not isinstance(schema, dict):
        return
    if isinstance(node, dict):
        order = schema.get("x-ordered")
        if order and all(_is_number(node.get(k)) for k in order):
            vals = [node[k] for k in order]
            if any(a > b for a, b in itertools.pairwise(vals)):
                out.append(f"{path or '/'}: phải {' ≤ '.join(order)}, gặp {vals}")
        for k, sub in schema.get("properties", {}).items():
            if k in node:
                _ordered_violations(node[k], sub, root, f"{path}/{k}", out)


def validate_camera_profile(profile: dict[str, Any]) -> list[str]:
    """Kiểm hồ sơ camera theo ``configs/camera_profile.schema.json`` + ràng buộc ``x-ordered``. Rỗng = đạt."""
    import jsonschema

    schema = camera_profile_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = [f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(profile)]
    _ordered_violations(profile, schema, schema, "", errors)
    return sorted(errors)


def load_profile(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


# ---- CLI -------------------------------------------------------------------------------------------------------
def _structured(errors: list[str]) -> list[dict[str, str]]:
    out = []
    for e in errors:
        head, _, msg = e.partition(": ")
        out.append({"section": head, "message": msg})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.contract", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="kiểm một bundle; in [] khi đạt")
    v.add_argument("bundle_dir")
    v.add_argument("--json", action="store_true", help="in object có cấu trúc")
    v.add_argument("--no-run", action="store_true", help="không chạy ORT trên golden")
    d = sub.add_parser("make-dummy", help="sinh bundle giả; không --kind = detector-v0.0.0/ và enhancer-v0.0.0/")
    d.add_argument("out_dir")
    d.add_argument("--kind", choices=MODELS, help="chỉ sinh bundle giả của kind này")
    d.add_argument("--json", action="store_true")
    c = sub.add_parser("check-profile", help="kiểm camera_profile.yaml theo schema §6")
    c.add_argument("profile")
    c.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "validate":
        bundle = Path(args.bundle_dir)
        if not bundle.is_dir():
            print(f"không có thư mục bundle: {bundle}", file=sys.stderr)
            return 2
        errors = validate_bundle(bundle, run_golden=not args.no_run)
        if args.json:
            print(
                json.dumps({"bundle": str(bundle), "ok": not errors, "errors": _structured(errors)}, ensure_ascii=False)
            )
        else:
            print(json.dumps(errors, ensure_ascii=False))
        return 0 if not errors else 1
    if args.cmd == "make-dummy":
        kinds = (args.kind,) if args.kind else MODELS_V1
        made = [str(make_dummy_bundle(args.out_dir, mdl)) for mdl in kinds]
        print(json.dumps({"bundles": made}, ensure_ascii=False) if args.json else "\n".join(made))
        return 0
    path = Path(args.profile)
    if not path.is_file():
        print(f"không có file hồ sơ: {path}", file=sys.stderr)
        return 2
    try:
        profile = load_profile(path)
    except Exception as e:  # noqa: BLE001 - hồ sơ hỏng: báo thành lỗi, exit 1
        errors = [f"/: không đọc được hồ sơ: {e}"]
    else:
        errors = validate_camera_profile(profile) if isinstance(profile, dict) else ["/: hồ sơ phải là mapping"]
    print(
        json.dumps(
            {"profile": str(path), "ok": not errors, "errors": errors} if args.json else errors, ensure_ascii=False
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
