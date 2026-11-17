"""Hợp đồng model XLA ↔ IOT-CV (SSOT: ``docs/hop-dong-model.md``): hằng số, kiểm bundle, bundle giả, kiểm hồ sơ camera.

CLI (D7)::

    python -m xla.contract validate <bundle-dir> [--json]
    python -m xla.contract make-dummy <out-dir> [--json]
    python -m xla.contract check-profile <camera_profile.yaml|.json> [--json]

Mã thoát: 0 = đạt, 1 = bundle/hồ sơ sai hợp đồng, 2 = sai cách gọi (thiếu tham số, thư mục không tồn tại).
``validate`` không ``--json`` in danh sách lỗi dạng JSON một dòng (``[]`` = đạt); có ``--json`` in object
``{"bundle", "ok", "errors": [{"section", "message"}]}``. Mỗi lỗi mở đầu bằng mục hợp đồng, ví dụ ``§2 chữ ký: ...``.

Quy ước bổ sung không có trong hợp đồng (chỉ chi tiết hoá, không đổi nghĩa):

- ``expected.json`` của enhancer: ``sha256_out`` = sha256 của ``frame_out.tobytes()`` (uint8 HWC BGR, C-order),
  ``mean`` = trung bình từng kênh [b, g, r].
- ``meta.metrics``: đủ khoá theo loại model; giá trị là số, hoặc ``null`` khi chưa đo (bundle giả).
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

# ---- §1–§4: hằng số hợp đồng -------------------------------------------------------------------------------------
CONTRACT_VERSION = "1.0"
CONTRACT_MAJOR = 1
FRAME_W = 640
FRAME_H = 480
FRAME_SHAPE = (FRAME_H, FRAME_W, 3)  # HWC, BGR, uint8
CROP_LEFT_V1 = 16
OPSET = 17
MODELS = ("detector", "enhancer")

INPUT_NAME = "frame"
INPUT_DTYPE = "uint8"
DETECTOR_OUTPUTS = {"boxes": ("float32", ("N", 4)), "scores": ("float32", ("N",))}
ENHANCER_OUTPUTS = {"frame_out": ("uint8", FRAME_SHAPE)}
OUTPUTS = {"detector": DETECTOR_OUTPUTS, "enhancer": ENHANCER_OUTPUTS}

GOLDEN_FRAMES = ("000.jpg", "001.jpg", "002.jpg")
EXPECTED_FILE = "expected.json"
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
META_DETECTOR = {"imgsz": int, "score_floor": float, "nms_iou": float, "max_det": int, "conf_threshold": float}
META_DETECTOR_OPTIONAL = {"input_variant": ("raw", "enhanced")}
METRICS_KEYS = {
    "detector": ("map50", "map50_95", "precision", "recall", "fp_per_neg_frame"),
    "enhancer": ("pink_frac_drop", "dark_channel_drop", "flicker_ratio"),
}
MS_PER_FRAME_KEYS = {"device": str, "runtime": str, "p50": float, "p95": float}

# §4 dung sai golden
DET_IOU_MIN = 0.95
DET_SCORE_ABS_MAX = 0.02
ENH_MAE_MAX = 1.0
ENH_ABS_MAX = 8

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
BUNDLE_DIR_RE = re.compile(r"^(detector|enhancer)-v(\d+\.\d+\.\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ONNX_DTYPES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 10: "float16", 11: "float64"}

DUMMY_BOX = (100.0, 100.0, 150.0, 160.0)
DUMMY_SCORE = 0.9
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


def _type_ok(value: Any, typ: type) -> bool:
    if typ is float:
        return _is_number(value)
    if typ is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, typ)


# ---- kiểm meta.json ---------------------------------------------------------------------------------------------
def _check_meta(meta: dict[str, Any], model: str, dir_version: str | None, err: _Errors) -> None:
    required = dict(META_COMMON)
    if model == "detector":
        required.update(META_DETECTOR)
    for key, typ in required.items():
        if key not in meta:
            err.add("4 meta.json", f"thiếu khoá bắt buộc '{key}'")
        elif not _type_ok(meta[key], typ):
            err.add("4 meta.json", f"khoá '{key}' sai kiểu: cần {typ.__name__}, gặp {type(meta[key]).__name__}")

    cv = meta.get("contract_version")
    if isinstance(cv, str):
        m = re.match(r"^(\d+)\.(\d+)$", cv)
        if not m:
            err.add("4 contract_version", f"'{cv}' không đúng dạng MAJOR.MINOR")
        elif int(m.group(1)) != CONTRACT_MAJOR:
            err.add("4 contract_version", f"major {m.group(1)} khác major được hỗ trợ {CONTRACT_MAJOR}")
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
                err.add("4 meta.json", f"metrics thiếu '{k}'")
            elif meta["metrics"][k] is not None and not _is_number(meta["metrics"][k]):
                err.add("4 meta.json", f"metrics.{k} phải là số hoặc null")
    if isinstance(meta.get("ms_per_frame"), dict):
        for k, typ in MS_PER_FRAME_KEYS.items():
            if not _type_ok(meta["ms_per_frame"].get(k), typ):
                err.add("4 meta.json", f"ms_per_frame.{k} thiếu hoặc sai kiểu ({typ.__name__})")
    if model == "detector":
        iv = meta.get("input_variant", "raw")
        if iv not in META_DETECTOR_OPTIONAL["input_variant"]:
            err.add("4 meta.json", f"input_variant '{iv}' không thuộc {META_DETECTOR_OPTIONAL['input_variant']}")
        for k in ("score_floor", "nms_iou", "conf_threshold"):
            if _is_number(meta.get(k)) and not 0.0 <= meta[k] <= 1.0:
                err.add("4 meta.json", f"{k} {meta[k]} ngoài [0, 1]")
    else:
        for k in (*META_DETECTOR, "input_variant"):
            if k in meta:
                err.add("4 meta.json", f"khoá '{k}' chỉ dành cho detector")


# ---- kiểm chữ ký graph -----------------------------------------------------------------------------------------
def _dims(value_info: Any) -> tuple[Any, ...]:
    out = []
    for d in value_info.type.tensor_type.shape.dim:
        out.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return tuple(out)


def _check_signature(onnx_path: Path, model: str, err: _Errors) -> bool:
    import onnx

    sec = "2" if model == "detector" else "3"
    try:
        m = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as e:  # noqa: BLE001 - file hỏng: báo đúng mục, không văng
        err.add(f"{sec} chữ ký", f"không đọc được {onnx_path.name}: {e}")
        return False
    n_before = len(err.items)
    opset = {o.domain: o.version for o in m.opset_import}.get("", None)
    if opset != OPSET:
        err.add(f"{sec} chữ ký", f"opset {opset} khác {OPSET}")
    inits = {i.name for i in m.graph.initializer}
    inputs = [i for i in m.graph.input if i.name not in inits]
    if [i.name for i in inputs] != [INPUT_NAME]:
        err.add(f"{sec} chữ ký", f"input phải đúng một tensor tên '{INPUT_NAME}', gặp {[i.name for i in inputs]}")
    else:
        dt = _ONNX_DTYPES.get(inputs[0].type.tensor_type.elem_type, str(inputs[0].type.tensor_type.elem_type))
        if dt != INPUT_DTYPE:
            err.add(f"{sec} chữ ký", f"input '{INPUT_NAME}' dtype {dt}, cần {INPUT_DTYPE}")
        if _dims(inputs[0]) != FRAME_SHAPE:
            err.add(f"{sec} chữ ký", f"input '{INPUT_NAME}' shape {list(_dims(inputs[0]))}, cần {list(FRAME_SHAPE)}")
    spec = OUTPUTS[model]
    outs = {o.name: o for o in m.graph.output}
    if sorted(outs) != sorted(spec):
        err.add(f"{sec} chữ ký", f"output phải là {sorted(spec)}, gặp {sorted(outs)}")
    else:
        for name, (dtype, shape) in spec.items():
            o = outs[name]
            dt = _ONNX_DTYPES.get(o.type.tensor_type.elem_type, str(o.type.tensor_type.elem_type))
            if dt != dtype:
                err.add(f"{sec} chữ ký", f"output '{name}' dtype {dt}, cần {dtype}")
            dims = _dims(o)
            fixed_ok = len(dims) == len(shape) and all(
                isinstance(want, str) or got == want for got, want in zip(dims, shape, strict=True)
            )
            if not fixed_ok:
                err.add(f"{sec} chữ ký", f"output '{name}' shape {list(dims)}, cần {list(shape)}")
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


def run_model(onnx_path: Path, frames: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    sess = _session(onnx_path)
    names = [o.name for o in sess.get_outputs()]
    return {k: dict(zip(names, sess.run(None, {INPUT_NAME: v}), strict=True)) for k, v in frames.items()}


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


def compare_enhancer(got: np.ndarray, ref: np.ndarray) -> list[str]:
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    mae, mx = float(diff.mean()), int(diff.max())
    if mae > ENH_MAE_MAX or mx > ENH_ABS_MAX:
        return [f"sai khác trung bình {mae:.3f} (≤ {ENH_MAE_MAX}), lớn nhất {mx} (≤ {ENH_ABS_MAX})"]
    return []


def _check_runtime_output(
    model: str, out: dict[str, np.ndarray], meta: dict[str, Any], frame: str, err: _Errors
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
    else:
        f = out["frame_out"]
        if f.dtype != np.uint8 or f.shape != FRAME_SHAPE:
            err.add("3 output", f"{frame}: frame_out {f.dtype}{list(f.shape)}, cần uint8{list(FRAME_SHAPE)}")
            return
        cl = meta.get("crop_left")
        if (
            _type_ok(cl, int)
            and 0 < cl < FRAME_W
            and not np.array_equal(f[:, :cl], np.repeat(f[:, cl : cl + 1], cl, axis=1))
        ):
            err.add("3 output", f"{frame}: cột [0, {cl}) không lặp lại cột {cl}")


def _check_golden(bundle: Path, model: str, onnx_path: Path, meta: dict[str, Any], run: bool, err: _Errors) -> None:
    gdir = bundle / "golden"
    missing = [n for n in (*GOLDEN_FRAMES, EXPECTED_FILE) if not (gdir / n).is_file()]
    if model == "enhancer":
        missing += [
            n.replace(".jpg", ".out.png") for n in GOLDEN_FRAMES if not (gdir / n.replace(".jpg", ".out.png")).is_file()
        ]
    if missing:
        err.add("4 golden", f"thiếu {missing} trong golden/")
        return
    try:
        expected = json.loads((gdir / EXPECTED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        err.add("4 golden", f"expected.json không đọc được: {e}")
        return
    if not isinstance(expected, dict) or sorted(expected) != sorted(GOLDEN_FRAMES):
        err.add("4 golden", f"expected.json phải có đúng khoá {list(GOLDEN_FRAMES)}")
        return
    if not run:
        return
    try:
        frames = {n: decode_frame(gdir / n) for n in GOLDEN_FRAMES}
        bad = {n: f.shape for n, f in frames.items() if f.shape != FRAME_SHAPE}
        if bad:
            err.add("1 khung", f"khung golden sai cỡ {bad}, cần {list(FRAME_SHAPE)}")
            return
        outs = run_model(onnx_path, frames)
    except Exception as e:  # noqa: BLE001 - ORT/giải mã lỗi: báo đúng mục
        err.add("4 golden", f"chạy ORT CPU trên golden lỗi: {e}")
        return
    conf = meta.get("conf_threshold") if _is_number(meta.get("conf_threshold")) else 0.0
    for name in GOLDEN_FRAMES:
        _check_runtime_output(model, outs[name], meta, name, err)
        if model == "detector":
            problems = compare_detector(outs[name], expected[name], conf)
        else:
            ref = decode_png(gdir / name.replace(".jpg", ".out.png"))
            exp = expected[name]
            if ref.shape != FRAME_SHAPE or hashlib.sha256(ref.tobytes()).hexdigest() != exp.get("sha256_out"):
                err.add("4 golden", f"{name}: .out.png không khớp sha256_out trong expected.json")
                continue
            problems = compare_enhancer(outs[name]["frame_out"], ref)
        for p in problems:
            err.add("4 golden", f"{name}: {p}")


def decode_png(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"không giải mã được {path}")
    return img


def validate_bundle(bundle_dir: str | Path, run_golden: bool = True) -> list[str]:
    """Kiểm một bundle theo hợp đồng §2–§4. Trả danh sách lỗi, rỗng = đạt."""
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
        err.add("4 bundle", f"tên thư mục '{bundle.name}' không đúng dạng <detector|enhancer>-vMAJOR.MINOR.PATCH")
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
        sig_ok = _check_signature(onnx_path, model, err)
    # chữ ký sai thì không chạy golden (lỗi dây chuyền chỉ lặp lại cùng một lỗi)
    _check_golden(bundle, model, onnx_path, meta, run_golden and sig_ok, err)
    return err.as_strings()


# ---- bundle giả ------------------------------------------------------------------------------------------------
def _dummy_graph(model: str, crop_left: int) -> Any:
    import onnx
    from onnx import TensorProto, helper

    frame = helper.make_tensor_value_info(INPUT_NAME, TensorProto.UINT8, list(FRAME_SHAPE))
    if model == "detector":
        nodes = [
            helper.make_node("Cast", [INPUT_NAME], ["f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["f"], ["fmax"], keepdims=0),
            helper.make_node("Constant", [], ["zero"], value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])),
            helper.make_node("Mul", ["fmax", "zero"], ["dep"]),  # 0 * max(frame): output phụ thuộc input
            helper.make_node(
                "Constant", [], ["s0"], value=helper.make_tensor("s", TensorProto.FLOAT, [1], [DUMMY_SCORE])
            ),
            helper.make_node("Add", ["s0", "dep"], ["scores"]),
            helper.make_node(
                "Constant", [], ["b0"], value=helper.make_tensor("b", TensorProto.FLOAT, [1, 4], list(DUMMY_BOX))
            ),
            helper.make_node("Add", ["b0", "dep"], ["boxes"]),
        ]
        outputs = [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N"]),
        ]
    else:

        def const_i64(name: str, vals: list[int]) -> Any:
            return helper.make_node(
                "Constant", [], [name], value=helper.make_tensor(name + "_v", TensorProto.INT64, [len(vals)], vals)
            )

        nodes = [
            const_i64("c_st", [crop_left]),
            const_i64("c_st1", [crop_left + 1]),
            const_i64("c_end", [FRAME_W]),
            const_i64("ax1", [1]),
            const_i64("rep", [1, crop_left, 1]),
            helper.make_node("Slice", [INPUT_NAME, "c_st", "c_st1", "ax1"], ["col"]),
            helper.make_node("Tile", ["col", "rep"], ["strip"]),
            helper.make_node("Slice", [INPUT_NAME, "c_st", "c_end", "ax1"], ["body"]),
            helper.make_node("Concat", ["strip", "body"], ["frame_out"], axis=1),
        ]
        outputs = [helper.make_tensor_value_info("frame_out", TensorProto.UINT8, list(FRAME_SHAPE))]
    graph = helper.make_graph(nodes, f"dummy_{model}", [frame], outputs)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="xla.contract.dummy")
    m.ir_version = 8  # IR của opset 17
    onnx.checker.check_model(m)
    return m


def _golden_frames() -> dict[str, np.ndarray]:
    """3 khung xám có hoạ tiết (gradient + ô cờ), khác nhau mỗi khung."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    out = {}
    for i, name in enumerate(GOLDEN_FRAMES):
        g = (xx * 255 // (FRAME_W - 1) + ((xx // (16 + 8 * i) + yy // (16 + 8 * i)) % 2) * 40 + i * 30) % 256
        out[name] = np.repeat(g.astype(np.uint8)[:, :, None], 3, axis=2)
    return out


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
    outs = run_model(onnx_path, {n: decode_frame(gdir / n) for n in GOLDEN_FRAMES})
    expected: dict[str, Any] = {}
    for name, o in outs.items():
        if model == "detector":
            expected[name] = {"boxes": o["boxes"].tolist(), "scores": o["scores"].tolist()}
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
        "metrics": {k: None for k in METRICS_KEYS[model]},
        "ms_per_frame": {
            "device": "none (bundle giả)",
            "runtime": f"onnxruntime {ort.__version__} CPU EP",
            "p50": 0.0,
            "p95": 0.0,
        },
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if model == "detector":
        meta.update(imgsz=640, score_floor=0.05, nms_iou=0.7, max_det=100, conf_threshold=0.25, input_variant="raw")
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
    d = sub.add_parser("make-dummy", help="sinh detector-v0.0.0/ và enhancer-v0.0.0/ giả")
    d.add_argument("out_dir")
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
        made = [str(make_dummy_bundle(args.out_dir, mdl)) for mdl in MODELS]
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
