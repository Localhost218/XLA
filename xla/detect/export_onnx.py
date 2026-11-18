"""Xuất ``detector.onnx`` + bundle ``detector-v<ver>/`` đúng hợp đồng §2, §4 (phase 11 bước 5–7, D1).

AX (giao diện chốt ở ``thuc-thi-phase-10-14.md`` §5.4)::

    python -m xla.detect.export_onnx --weights <pt> --out models/release/detector-v1.0.0 --version 1.0.0 \
        --conf <float> --metrics <json> --train-run <str> [--imgsz 1280] [--parity-out results/e2/parity.json] \
        [--skip-cuda] --json
    # exit 0 = bundle qua xla.contract.validate + parity đạt; 1 = lệch hợp đồng / parity / lỗi lúc chạy; 2 = sai cách gọi

Dây chuyền:

1. ``ultralytics export`` opset 17, tĩnh ``[1,3,imgsz,imgsz]``, không NMS (graph lõi giữ nguyên, ``output0`` =
   ``[1, 4+nc, A]``: ``cx,cy,w,h`` px letterbox + điểm lớp đã sigmoid). Chạy trên CPU, trong thư mục tạm dưới
   ``xla.paths.cache_dir()`` (ultralytics ghi ``.onnx`` cạnh ``.pt``, nên ``.pt`` được chép sang đó trước).
2. Bọc graph (``build_detector_model``): ``frame`` uint8 [480,640,3] BGR → cắt cột ``[crop_left, 640)`` → BGR→RGB →
   letterbox như ``ultralytics.data.augment.LetterBox`` (``auto=False``: ``r = min(imgsz/480, imgsz/(640−crop_left))``,
   cỡ mới ``round(w·r)``, ``round(h·r)``; nội suy bilinear **bit-exact** với ``cv2.resize`` INTER_LINEAR trên uint8 —
   hệ số nguyên 11 bit và các phép dịch cắt cụt của OpenCV, viết bằng op số nguyên (``_resize_nodes``); lề
   ``round(d/2 ∓ 0,1)`` giá trị 114) → /255 → NCHW → lõi → chỉ lớp người
   (mô hình nhiều lớp: hộp chỉ tính là người khi ``argmax`` lớp = 0, đúng ``non_max_suppression(classes=[0])``) →
   ``score ≥ score_floor`` → ``NonMaxSuppression`` (``center_point_box=1``, ``nms_iou``, tối đa ``max_det``) →
   ``x = (x_lb − pad_x)/r + crop_left``, ``y = (y_lb − pad_y)/r`` → kẹp vào ``[crop_left,640]×[0,480]`` (nằm trong
   ``[0,640]×[0,480]`` của §2; ultralytics trên khung đã cắt cũng kẹp như vậy) — làm cho mọi anchor, và anchor có hộp
   rỗng sau khi kẹp (§2 đòi ``x1<x2, y1<y2``) bị loại trước NMS → sắp ``scores`` giảm dần (``TopK``). Không dùng
   ``Compress``/``NonZero``: ``Compress`` của ORT CUDA 1.24 hỏng khi đầu vào rỗng.
3. ``onnxsim`` khi nó trả ``check=True`` **và** ORT CPU cho cùng kết quả trên khung thật (lệch > 1e-3 → giữ bản chưa rút
   gọn, ghi vào báo cáo).
4. Tương đương trên dev-val (100 khung có nhãn trong ``data/real/labels/manifest.json``), chấm bằng
   ``xla.eval.detection.evaluate``: chuẩn là ``.pt`` qua ``YOLO.predict`` mặc định trên **cùng khung đã cắt**
   ``[:, crop_left:]``, cùng ``conf = score_floor``, ``iou = nms_iou``, ``max_det`` (cộng ``crop_left`` vào x). Cổng:
   ``|mAP50(ORT CPU) − mAP50(.pt)| ≤ 0,005`` và như vậy cho ORT CUDA EP. Lưu ý: ``Model.predict`` ép ``rect=True`` nên
   ``.pt`` letterbox chữ nhật tối thiểu (1280×992, lề 3 px) còn graph letterbox vuông theo §2 (lề 147 px = 4,5 bước
   stride, lưới P5 lệch nửa ô) — phần lệch còn lại của cổng đến từ đây. Kèm số tham khảo (không chặn):
   ``.pt`` ``rect=False`` (cùng tiền xử lý với graph — lệch ở đây mới là lỗi xuất) và ``.pt`` trên khung nguyên rồi bỏ
   hộp ``x1 < crop_left`` (đường của mốc phase 09 / E1 tạo ra ``meta.metrics``).
5. ``ms_per_frame``: ORT CUDA EP, warmup 30 + đo 300 khung dev (gồm chép host→GPU), p50/p95. Ngay trước khi đo, đọc
   ``% Processor Performance`` 60 s (``cpu_perf_pct`` trong ``meta.ms_per_frame``); min < 80 % → cảnh báo
   "đo lúc hạ xung" trong báo cáo (không chặn: số vẫn ghi, người phát hành đo lại). Cả bước 4 (``.pt`` chạy
   GPU, ORT CUDA) lẫn bước 5 nằm trong một lần giữ ``xla.gpu_lock`` (D9). ``import torch`` trước khi tạo session CUDA
   (DLL CUDA của torch, xem ``pyproject.toml``).
6. ``golden/``: 3 khung **dev** ngoài dev-val (chép nguyên byte JPEG), ở vị trí 1/4, 1/2, 3/4 danh sách dev; mỗi vị trí dò
   tới khung đầu tiên có hộp ``≥ conf_threshold``. ``expected.json`` từ ORT CPU. ``meta.json`` đủ khoá §4, thêm
   ``golden_sources`` và ``export`` (truy vết). Cuối cùng ``xla.contract.validate_bundle`` phải rỗng.

``--metrics``: JSON object có 5 khoá ``map50, map50_95, precision, recall, fp_per_neg_frame`` (số hoặc null), hoặc object
có khoá ``metrics`` chứa 5 khoá đó. ``--skip-cuda``: không dùng GPU (``.pt`` chạy CPU, không parity CUDA, ms đo trên
ORT CPU EP và báo cảnh báo) — chỉ để thử khi GPU bận/không có; bundle phát hành phải đo CUDA.

Hằng số vận hành (ghi vào ``meta.json``): ``SCORE_FLOOR`` 0,05 (§2 mặc định), ``NMS_IOU`` 0,5 = ``PREDICT_IOU`` của
ultralytics (mốc phase 09 gọi ``predict`` không truyền ``iou``), ``MAX_DET`` 100 = ``maxDets`` của bộ chấm.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla import contract as C
from xla.camera.frames import FRAMES_REL
from xla.paths import cache_dir, repo_root

SCORE_FLOOR = 0.05
NMS_IOU = 0.5
MAX_DET = 100
IMGSZ = 1280
STRIDE = 32
PAD_VALUE = 114.0
RESIZE_COEF_SCALE = 2048  # INTER_RESIZE_COEF_SCALE của OpenCV
CLASS_PERSON = 0
PARITY_TOL = 0.005
SIMPLIFY_TOL = 1e-3
WARMUP = 30
TIMED_FRAMES = 300
CPU_THREADS = 4  # máy lật bit RAM dưới tải nặng: tối đa 4 luồng CPU (thuc-thi-phase-10-14.md §3)
GOLDEN_POSITIONS = (0.25, 0.5, 0.75)
GOLDEN_SCAN = 60
GPU_LOCK_TIMEOUT_S = 6 * 3600.0
CPU_PERF_MIN_PCT = 80.0  # dưới mức này CPU đang hạ xung (pin yếu, tiết kiệm điện): ms không đại diện
CPU_PERF_WINDOW_S = 60
CPU_PERF_COUNTER = r"\Processor Information(_Total)\% Processor Performance"
CORE_INPUT = "images"
CORE_PREFIX = "core/"
PROFILE_REL = Path("configs/camera_profile.yaml")
SPLIT_REL = Path("data/real/split.json")
LABELS_REL = Path("data/real/labels")
CSV_FIELDS = ("frame", "x1", "y1", "x2", "y2", "conf")

Row = tuple[float, float, float, float, float]  # x1, y1, x2, y2, conf (px khung gốc)


class UsageError(ValueError):
    """Tham số sai (CLI exit 2)."""


# ---- hình học letterbox ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Letterbox:
    """Tham số letterbox của khung đã cắt mép, tính đúng như ``LetterBox.get_params`` (auto=False, center=True)."""

    imgsz: int
    crop_left: int

    @property
    def src_w(self) -> int:
        return C.FRAME_W - self.crop_left

    @property
    def src_h(self) -> int:
        return C.FRAME_H

    @property
    def r(self) -> float:
        return min(self.imgsz / self.src_h, self.imgsz / self.src_w)

    @property
    def new_w(self) -> int:
        return max(round(self.src_w * self.r), 1)

    @property
    def new_h(self) -> int:
        return max(round(self.src_h * self.r), 1)

    @property
    def pads(self) -> tuple[int, int, int, int]:
        """(top, bottom, left, right)."""
        dw, dh = (self.imgsz - self.new_w) / 2, (self.imgsz - self.new_h) / 2
        return round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)


# ---- dựng graph -------------------------------------------------------------------------------------------------
def _const(name: str, value: Any, dtype: Any = np.int64) -> Any:
    from onnx import helper, numpy_helper

    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.asarray(value, dtype), name + "_v"))


def resize_taps(src: int, dst: int, clamp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chỉ số + hệ số nguyên (thang 2048) của ``cv2.resize`` INTER_LINEAR trên uint8 cho một trục.

    Đúng như ``resizeGeneric`` của OpenCV: ``f = float((d+0,5)·src/dst − 0,5)``, hệ số ``cvRound((1−f)·2048)``,
    ``cvRound(f·2048)``; trục x kẹp ``f = 0`` ở hai biên (``clamp=True``), trục y không kẹp ``f`` mà chỉ kẹp hàng nguồn."""
    scale = 1.0 / (dst / src)
    f = ((np.arange(dst) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int64)
    f = (f - s).astype(np.float32)
    if clamp:
        f[s < 0], s[s < 0] = 0, 0
        f[s >= src - 1], s[s >= src - 1] = 0, src - 1
    a0 = np.rint((np.float32(1) - f) * np.float32(RESIZE_COEF_SCALE)).astype(np.int32)
    a1 = np.rint(f * np.float32(RESIZE_COEF_SCALE)).astype(np.int32)
    return np.clip(s, 0, src - 1), np.clip(s + 1, 0, src - 1), a0, a1


def _resize_nodes(src: str, lb: Letterbox, out: str) -> list[Any]:
    """Resize HWC int32 bit-exact với ``cv2.resize(..., INTER_LINEAR)`` uint8 (đường SIMD của OpenCV 4.x).

    Ngang: ``h = S[x0]·a0 + S[x1]·a1``. Dọc: ``((b0·(h0>>4))>>16 + (b1·(h1>>4))>>16 + 2) >> 2`` — các phép dịch phải
    cắt cụt của ``VResizeLinearVec_32s8u``. Một ``Resize`` float + ``Round`` lệch +1 mức ở ~12 % điểm ảnh (luôn cao hơn),
    nên không dùng. Mọi giá trị không âm nên ``Div`` số nguyên (cắt về 0) = dịch phải."""
    from onnx.helper import make_node as N

    i32 = np.int32
    xs0, xs1, a0, a1 = resize_taps(lb.src_w, lb.new_w, clamp=True)
    ys0, ys1, b0, b1 = resize_taps(lb.src_h, lb.new_h, clamp=False)
    return [
        _const("pre/xs0", xs0), _const("pre/xs1", xs1), _const("pre/ys0", ys0), _const("pre/ys1", ys1),
        _const("pre/a0", a0.reshape(1, -1, 1), i32), _const("pre/a1", a1.reshape(1, -1, 1), i32),
        _const("pre/b0", b0.reshape(-1, 1, 1), i32), _const("pre/b1", b1.reshape(-1, 1, 1), i32),
        _const("pre/d16", 16, i32), _const("pre/d65536", 1 << 16, i32), _const("pre/two", 2, i32),
        _const("pre/four", 4, i32),
        N("Gather", [src, "pre/xs0"], ["pre/hx0"], axis=1), N("Gather", [src, "pre/xs1"], ["pre/hx1"], axis=1),
        N("Mul", ["pre/hx0", "pre/a0"], ["pre/hm0"]), N("Mul", ["pre/hx1", "pre/a1"], ["pre/hm1"]),
        N("Add", ["pre/hm0", "pre/hm1"], ["pre/h"]),
        N("Div", ["pre/h", "pre/d16"], ["pre/h4"]),
        N("Gather", ["pre/h4", "pre/ys0"], ["pre/v0"], axis=0), N("Gather", ["pre/h4", "pre/ys1"], ["pre/v1"], axis=0),
        N("Mul", ["pre/v0", "pre/b0"], ["pre/vm0"]), N("Mul", ["pre/v1", "pre/b1"], ["pre/vm1"]),
        N("Div", ["pre/vm0", "pre/d65536"], ["pre/vh0"]), N("Div", ["pre/vm1", "pre/d65536"], ["pre/vh1"]),
        N("Add", ["pre/vh0", "pre/vh1"], ["pre/vs"]), N("Add", ["pre/vs", "pre/two"], ["pre/vr"]),
        N("Div", ["pre/vr", "pre/four"], [out]),
    ]


def _pre_nodes(lb: Letterbox, out: str) -> list[Any]:
    from onnx import TensorProto
    from onnx.helper import make_node as N

    top, bottom, left, right = lb.pads
    nodes = [
        _const("pre/c0", [lb.crop_left]), _const("pre/c1", [C.FRAME_W]), _const("pre/ax1", [1]),
        N("Slice", [C.INPUT_NAME, "pre/c0", "pre/c1", "pre/ax1"], ["pre/crop"]),
        _const("pre/rgb_idx", [2, 1, 0]),
        N("Gather", ["pre/crop", "pre/rgb_idx"], ["pre/rgb"], axis=2),
        N("Cast", ["pre/rgb"], ["pre/i"], to=TensorProto.INT32),
    ]
    cur = "pre/i"
    if (lb.new_h, lb.new_w) != (lb.src_h, lb.src_w):
        nodes += _resize_nodes(cur, lb, "pre/rs")
        cur = "pre/rs"
    nodes += [
        N("Cast", [cur], ["pre/f"], to=TensorProto.FLOAT),
        _const("pre/lo", 0.0, np.float32), _const("pre/hi", 255.0, np.float32),
        N("Clip", ["pre/f", "pre/lo", "pre/hi"], ["pre/fc"]),  # v_pack_u của OpenCV bão hoà về [0, 255]
        N("Transpose", ["pre/fc"], ["pre/chw"], perm=[2, 0, 1]),
        _const("pre/ax0", [0]),
        N("Unsqueeze", ["pre/chw", "pre/ax0"], ["pre/nchw"]),
    ]
    cur = "pre/nchw"
    if any((top, bottom, left, right)):
        nodes += [
            _const("pre/pads", [0, 0, top, left, 0, 0, bottom, right]),
            _const("pre/padv", PAD_VALUE, np.float32),
            N("Pad", [cur, "pre/pads", "pre/padv"], ["pre/pad"], mode="constant"),
        ]
        cur = "pre/pad"
    nodes += [_const("pre/255", 255.0, np.float32), N("Div", [cur, "pre/255"], [out])]
    return nodes


def _post_nodes(core_out: str, nc: int, lb: Letterbox, score_floor: float, nms_iou: float, max_det: int) -> list[Any]:
    from onnx.helper import make_node as N

    f32 = np.float32
    pad_x, pad_y = lb.pads[2], lb.pads[0]
    cl = float(lb.crop_left)
    nodes = [
        _const("post/i0", [0]), _const("post/i2", [2]), _const("post/i4", [4]), _const("post/i5", [5]),
        _const("post/ax0", [0]), _const("post/ax1", [1]),
        N("Slice", [core_out, "post/i0", "post/i4", "post/ax1"], ["post/xywh"]),
        N("Transpose", ["post/xywh"], ["post/boxes_nms"], perm=[0, 2, 1]),  # [1, A, 4]
        N("Slice", [core_out, "post/i4", "post/i5", "post/ax1"], ["post/person"]),  # [1, 1, A]
        _const("post/floor", score_floor, f32),
        N("GreaterOrEqual", ["post/person", "post/floor"], ["post/keep_floor"]),
    ]
    keep = "post/keep_floor"
    if nc > 1:  # ultralytics: lớp tốt nhất trước, lọc classes=[0] sau
        nodes += [
            _const("post/iend", [4 + nc]),
            N("Slice", [core_out, "post/i4", "post/iend", "post/ax1"], ["post/cls"]),
            N("ArgMax", ["post/cls"], ["post/am"], axis=1, keepdims=1),
            _const("post/cls_person", CLASS_PERSON),
            N("Equal", ["post/am", "post/cls_person"], ["post/is_person"]),
            N("And", ["post/keep_floor", "post/is_person"], ["post/keep"]),
        ]
        keep = "post/keep"
    # Hộp của MỌI anchor → toạ độ khung gốc, kẹp, rồi hộp rỗng sau khi kẹp bị loại TRƯỚC NMS (gộp vào điểm), để graph
    # không cần Compress/NonZero: Compress của ORT CUDA 1.24 hỏng khi đầu vào rỗng (cudaMemcpyAsync invalid argument).
    nodes += [
        N("Squeeze", ["post/boxes_nms", "post/ax0"], ["post/b_a4"]),  # [A, 4] cx, cy, w, h
        N("Slice", ["post/b_a4", "post/i0", "post/i2", "post/ax1"], ["post/cxy"]),
        N("Slice", ["post/b_a4", "post/i2", "post/i4", "post/ax1"], ["post/wh"]),
        _const("post/half", 0.5, f32),
        N("Mul", ["post/wh", "post/half"], ["post/hwh"]),
        N("Sub", ["post/cxy", "post/hwh"], ["post/xy1"]),
        N("Add", ["post/cxy", "post/hwh"], ["post/xy2"]),
        N("Concat", ["post/xy1", "post/xy2"], ["post/xyxy_lb"], axis=1),
        _const("post/pad", [pad_x, pad_y, pad_x, pad_y], f32),
        _const("post/r", lb.r, f32),
        _const("post/shift", [cl, 0.0, cl, 0.0], f32),
        _const("post/hi", [C.FRAME_W, C.FRAME_H, C.FRAME_W, C.FRAME_H], f32),
        N("Sub", ["post/xyxy_lb", "post/pad"], ["post/t1"]),
        N("Div", ["post/t1", "post/r"], ["post/t2"]),
        N("Add", ["post/t2", "post/shift"], ["post/t3"]),
        N("Max", ["post/t3", "post/shift"], ["post/t4"]),  # cận dưới = [crop_left, 0, crop_left, 0]
        N("Min", ["post/t4", "post/hi"], ["post/b_clip"]),
        N("Slice", ["post/b_clip", "post/i0", "post/i2", "post/ax1"], ["post/p1"]),
        N("Slice", ["post/b_clip", "post/i2", "post/i4", "post/ax1"], ["post/p2"]),
        N("Sub", ["post/p2", "post/p1"], ["post/dwh"]),
        N("ReduceMin", ["post/dwh"], ["post/dmin"], axes=[1], keepdims=0),  # [A]
        _const("post/zero", 0.0, f32),
        N("Greater", ["post/dmin", "post/zero"], ["post/valid_a"]),
        _const("post/shape_1_1_a", [1, 1, -1]),
        N("Reshape", ["post/valid_a", "post/shape_1_1_a"], ["post/valid"]),
        N("And", [keep, "post/valid"], ["post/keep_all"]),
        _const("post/neg", -1.0, f32),
        N("Where", ["post/keep_all", "post/person", "post/neg"], ["post/eff"]),
        _const("post/max_out", [max_det]), _const("post/iou", [nms_iou], f32), _const("post/sthr", [0.0], f32),
        N("NonMaxSuppression", ["post/boxes_nms", "post/eff", "post/max_out", "post/iou", "post/sthr"],
          ["post/sel"], center_point_box=1),
        _const("post/col2", 2),
        N("Gather", ["post/sel", "post/col2"], ["post/idx"], axis=1),  # [K]
        N("Gather", ["post/b_clip", "post/idx"], ["post/b_k"], axis=0),
        _const("post/flat", [-1]),
        N("Reshape", ["post/person", "post/flat"], ["post/s_a"]),
        N("Gather", ["post/s_a", "post/idx"], ["post/s_k"], axis=0),
        N("Shape", ["post/s_k"], ["post/n"]),
        N("TopK", ["post/s_k", "post/n"], ["scores", "post/order"], axis=0, largest=1, sorted=1),
        N("Gather", ["post/b_k", "post/order"], ["post/boxes_f"], axis=0),
        N("Identity", ["post/boxes_f"], ["boxes"]),
    ]
    return nodes


def _core_io(core: Any, imgsz: int) -> tuple[str, str, int]:
    """Kiểm graph lõi của ultralytics; trả (tên input, tên output, số lớp)."""
    inits = {i.name for i in core.graph.initializer}
    inputs = [i for i in core.graph.input if i.name not in inits]
    if len(inputs) != 1 or C._dims(inputs[0]) != (1, 3, imgsz, imgsz):
        raise ValueError(f"graph lõi phải có một input [1,3,{imgsz},{imgsz}], gặp "
                         f"{[(i.name, C._dims(i)) for i in inputs]}")
    if len(core.graph.output) != 1:
        raise ValueError(f"graph lõi phải có một output, gặp {[o.name for o in core.graph.output]}")
    dims = C._dims(core.graph.output[0])
    if len(dims) != 3 or dims[0] != 1 or not isinstance(dims[1], int) or dims[1] < 5:
        raise ValueError(f"output lõi phải là [1, 4+nc, A], gặp {list(dims)}")
    return inputs[0].name, core.graph.output[0].name, dims[1] - 4


def build_preprocess_model(imgsz: int, crop_left: int) -> Any:
    """Chỉ phần tiền xử lý: ``frame`` → ``images`` [1,3,imgsz,imgsz] float32 (để kiểm khớp ``LetterBox``)."""
    import onnx
    from onnx import TensorProto, helper

    lb = Letterbox(imgsz, crop_left)
    graph = helper.make_graph(
        _pre_nodes(lb, CORE_INPUT), "xla_detector_pre",
        [helper.make_tensor_value_info(C.INPUT_NAME, TensorProto.UINT8, list(C.FRAME_SHAPE))],
        [helper.make_tensor_value_info(CORE_INPUT, TensorProto.FLOAT, [1, 3, imgsz, imgsz])])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", C.OPSET)])
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=True)
    return m


def build_detector_model(core: Any, *, imgsz: int, crop_left: int, score_floor: float = SCORE_FLOOR,
                         nms_iou: float = NMS_IOU, max_det: int = MAX_DET) -> Any:
    """Bọc graph lõi ultralytics (``[1,3,S,S]`` → ``[1,4+nc,A]``) thành ``detector.onnx`` đúng §2."""
    import onnx
    from onnx import TensorProto, helper

    opset = {o.domain: o.version for o in core.opset_import}
    if opset.get("", opset.get("ai.onnx")) != C.OPSET or set(opset) - {"", "ai.onnx"}:
        raise ValueError(f"graph lõi phải opset {C.OPSET}, chỉ miền mặc định; gặp {opset}")
    core = onnx.compose.add_prefix(core, CORE_PREFIX)
    core_in, core_out, nc = _core_io(core, imgsz)
    lb = Letterbox(imgsz, crop_left)
    nodes = _pre_nodes(lb, core_in) + list(core.graph.node) + _post_nodes(core_out, nc, lb, score_floor, nms_iou,
                                                                           max_det)
    graph = helper.make_graph(
        nodes, "xla_detector",
        [helper.make_tensor_value_info(C.INPUT_NAME, TensorProto.UINT8, list(C.FRAME_SHAPE))],
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", 4]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N"])],
        initializer=list(core.graph.initializer))
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", C.OPSET)],
                          producer_name="xla.detect.export_onnx")
    m.ir_version = core.ir_version
    for p in core.metadata_props:
        m.metadata_props.add(key=p.key, value=p.value)
    onnx.checker.check_model(m, full_check=True)
    return m


# ---- ORT --------------------------------------------------------------------------------------------------------
def ort_session(model: Any, cuda: bool = False) -> Any:
    """Session từ đường dẫn hoặc ``ModelProto``. CUDA không lên được thì lỗi (không lặng lẽ chạy CPU)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    src = str(model) if isinstance(model, (str, Path)) else model.SerializeToString()
    if cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        so.intra_op_num_threads = CPU_THREADS
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(src, sess_options=so, providers=providers)
    if cuda and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ORT không bật được CUDAExecutionProvider (có {sess.get_providers()}); "
                           "import torch trước khi tạo session chưa?")
    return sess


def run_detector(sess: Any, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes, scores = sess.run(["boxes", "scores"], {C.INPUT_NAME: frame})
    return boxes, scores


def _rows(boxes: np.ndarray, scores: np.ndarray) -> list[Row]:
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)) for b, s in zip(boxes, scores, strict=True)]


def simplify_checked(model: Any, probes: list[np.ndarray]) -> tuple[Any, bool, str]:
    """``onnxsim`` nếu nó không làm vỡ graph; trả (model, đã rút gọn?, ghi chú)."""
    try:
        import onnxsim

        simp, ok = onnxsim.simplify(model)
    except Exception as e:  # noqa: BLE001 - onnxsim hỏng thì giữ graph gốc, có ghi chú
        return model, False, f"onnxsim lỗi ({type(e).__name__}: {e}); giữ graph chưa rút gọn"
    if not ok:
        return model, False, "onnxsim check=False; giữ graph chưa rút gọn"
    a, b = ort_session(model), ort_session(simp)
    for i, fr in enumerate(probes):
        (ba, sa), (bb, sb) = run_detector(a, fr), run_detector(b, fr)
        if ba.shape != bb.shape or (ba.size and max(np.abs(ba - bb).max(), np.abs(sa - sb).max()) > SIMPLIFY_TOL):
            return model, False, f"onnxsim đổi kết quả ở khung thử {i} ({ba.shape} vs {bb.shape}); giữ graph gốc"
    return simp, True, f"onnxsim {onnxsim.__version__}: đã rút gọn, khớp ORT CPU trên {len(probes)} khung thử"


def cpu_performance_pct(window_s: int = CPU_PERF_WINDOW_S) -> dict[str, Any] | None:
    """``% Processor Performance`` của Windows trong ``window_s`` giây (mỗi giây một mẫu); None khi không đo được.

    Đo ms lúc CPU hạ xung (vd. pin vừa sạc lại từ mức thấp: ~22 %) cho số chậm hơn thật; báo cáo ghi kèm số này."""
    import subprocess

    if sys.platform != "win32":
        return None
    cmd = (f"(Get-Counter '{CPU_PERF_COUNTER}' -SampleInterval 1 -MaxSamples {int(window_s)})"
           ".CounterSamples.CookedValue")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=window_s + 60, check=True).stdout
        vals = [float(v.replace(",", ".")) for v in out.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not vals:
        return None
    return {"counter": CPU_PERF_COUNTER, "min": round(min(vals), 1), "mean": round(float(np.mean(vals)), 1),
            "window_s": int(window_s), "throttled": min(vals) < CPU_PERF_MIN_PCT}


def _cpu_perf_warning(perf: dict[str, Any] | None) -> str | None:
    if perf is None:
        return "không đọc được % Processor Performance trước khi đo ms (không biết CPU có hạ xung không)"
    if perf["throttled"]:
        return (f"đo lúc hạ xung: % Processor Performance min {perf['min']} / tb {perf['mean']} < "
                f"{CPU_PERF_MIN_PCT} trong {perf['window_s']} s — ms_per_frame không đại diện, đo lại khi CPU đủ xung")
    return None


def measure_ms(sess: Any, frames: list[np.ndarray], warmup: int = WARMUP) -> dict[str, float]:
    for fr in frames[:warmup]:
        run_detector(sess, fr)
    t = []
    for fr in frames:
        t0 = time.perf_counter()
        run_detector(sess, fr)
        t.append((time.perf_counter() - t0) * 1000.0)
    v = np.asarray(t)
    return {"p50": round(float(np.percentile(v, 50)), 3), "p95": round(float(np.percentile(v, 95)), 3),
            "n_frames": int(v.size), "warmup": warmup}


# ---- ultralytics -------------------------------------------------------------------------------------------------
def export_core(weights: Path, work: Path, imgsz: int) -> Path:
    """``ultralytics export`` opset 17 trên CPU, trong ``work`` (chép ``.pt`` sang trước)."""
    from ultralytics import YOLO

    pt = work / weights.name
    shutil.copyfile(weights, pt)
    out = YOLO(str(pt)).export(format="onnx", imgsz=imgsz, opset=C.OPSET, dynamic=False, simplify=False, nms=False,
                               batch=1, device="cpu", verbose=False)
    path = Path(out)
    if not path.is_file():
        raise RuntimeError(f"ultralytics export không tạo {path}")
    return path


def predict_pt(weights: Path, frames: dict[str, np.ndarray], crop_left: int, imgsz: int, device: Any,
               ) -> dict[str, dict[str, list[Row]]]:
    """``.pt`` qua ``YOLO.predict``, ba đường:

    - ``crop_rect``: khung đã cắt, mặc định của ``Model.predict`` (``rect=True`` → letterbox chữ nhật tối thiểu, với
      1280: 1280×992, lề trên 3 px) — chuẩn của cổng parity;
    - ``crop_square``: khung đã cắt, ``rect=False`` → letterbox vuông imgsz×imgsz như graph (§2) — tách lỗi xuất khỏi
      ảnh hưởng của hình dạng letterbox;
    - ``full_edge``: khung nguyên rồi bỏ hộp ``x1 < crop_left`` (đường của mốc phase 09 / E1)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    classes = [CLASS_PERSON] if len(model.names) > 1 else None

    def pred(img: np.ndarray, dx: float, rect: bool) -> list[Row]:
        res = model.predict(img, imgsz=imgsz, conf=SCORE_FLOOR, iou=NMS_IOU, max_det=MAX_DET, classes=classes,
                            device=device, rect=rect, verbose=False)[0]
        return [(float(x1) + dx, float(y1), float(x2) + dx, float(y2), float(c))
                for x1, y1, x2, y2, c, _ in res.boxes.data.cpu().numpy()]

    crops = {f: np.ascontiguousarray(im[:, crop_left:]) for f, im in frames.items()}
    return {"crop_rect": {f: pred(im, float(crop_left), True) for f, im in crops.items()},
            "crop_square": {f: pred(im, float(crop_left), False) for f, im in crops.items()},
            "full_edge": {f: [r for r in pred(im, 0.0, True) if r[0] >= crop_left] for f, im in frames.items()}}


# ---- chấm dev-val ------------------------------------------------------------------------------------------------
def write_pred_csv(path: Path, preds: dict[str, list[Row]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_FIELDS)
        for frame, rows in preds.items():
            for x1, y1, x2, y2, c in rows:
                w.writerow((frame, round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3), round(c, 6)))


def score_devval(preds: dict[str, list[Row]], csv_path: Path) -> dict[str, Any]:
    from xla.eval.detection import evaluate

    write_pred_csv(csv_path, preds)
    r = evaluate(csv_path, split="devval", boot_n=0)
    return {"mAP50": r["metrics"]["mAP50"]["value"], "mAP50_95": r["metrics"]["mAP50_95"]["value"],
            "n_frames": r["n"]["frames"], "labels_official": r["official"]}


def parity_entry(cand: dict[str, Any], ref: dict[str, Any], gate: bool = True) -> dict[str, Any]:
    d = cand["mAP50"] - ref["mAP50"]
    out = cand | {"delta_map50": round(d, 6), "delta_map50_95": round(cand["mAP50_95"] - ref["mAP50_95"], 6)}
    return out | ({"pass": abs(d) <= PARITY_TOL} if gate else {})


# ---- bundle ------------------------------------------------------------------------------------------------------
def load_metrics(path: str | Path) -> dict[str, Any]:
    """Đọc ``--metrics``: object 5 khoá §4, hoặc object có khoá ``metrics`` chứa chúng."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("metrics"), dict):
        doc = doc["metrics"]
    if not isinstance(doc, dict):
        raise UsageError(f"{path}: metrics phải là JSON object")
    return check_metrics(doc)


def check_metrics(doc: dict[str, Any]) -> dict[str, Any]:
    keys = C.METRICS_KEYS["detector"]
    missing = [k for k in keys if k not in doc]
    bad = [k for k in keys if k in doc and doc[k] is not None and not C._is_number(doc[k])]
    if missing or bad:
        raise UsageError(f"metrics thiếu {missing} / không phải số {bad} (cần {list(keys)})")
    return {k: doc[k] for k in keys}


def _check_args(weights: Path, out_dir: Path, version: str, conf: float, imgsz: int) -> None:
    if not weights.is_file():
        raise UsageError(f"không có file trọng số {weights}")
    if not C.SEMVER_RE.match(version):
        raise UsageError(f"--version '{version}' không phải semver MAJOR.MINOR.PATCH")
    if out_dir.name != f"detector-v{version}":
        raise UsageError(f"--out phải là thư mục tên 'detector-v{version}' (§4), gặp '{out_dir.name}'")
    if not SCORE_FLOOR <= conf <= 1.0:
        raise UsageError(f"--conf {conf} phải trong [score_floor {SCORE_FLOOR}, 1]")
    if imgsz <= 0 or imgsz % STRIDE:
        raise UsageError(f"--imgsz {imgsz} phải là bội dương của {STRIDE}")


def _load_frames(names: list[str], frames_dir: Path) -> dict[str, np.ndarray]:
    out = {f: C.decode_frame(frames_dir / f) for f in names}
    bad = {f: im.shape for f, im in out.items() if im.shape != C.FRAME_SHAPE}
    if bad:
        raise ValueError(f"khung sai cỡ (cần {list(C.FRAME_SHAPE)}): {list(bad.items())[:3]}")
    return out


def pick_golden(dev: list[str], frames_dir: Path, sess: Any, conf: float) -> list[str]:
    """3 khung dev ở vị trí 1/4, 1/2, 3/4; mỗi vị trí lấy khung đầu tiên (trong ``GOLDEN_SCAN``) có hộp ≥ conf."""
    picked: list[str] = []
    for pos in GOLDEN_POSITIONS:
        start = int(pos * len(dev))
        window = [f for f in dev[start:start + GOLDEN_SCAN] if f not in picked] or [f for f in dev if f not in picked]
        choice = window[0]
        for f in window:
            _, scores = run_detector(sess, C.decode_frame(frames_dir / f))
            if (scores >= conf).any():
                choice = f
                break
        picked.append(choice)
    return picked


def _write_golden(bundle: Path, onnx_path: Path, sources: list[str], frames_dir: Path) -> None:
    gdir = bundle / "golden"
    if gdir.exists():
        shutil.rmtree(gdir)
    gdir.mkdir(parents=True)
    for name, src in zip(C.GOLDEN_FRAMES, sources, strict=True):
        shutil.copyfile(frames_dir / src, gdir / name)  # nguyên byte JPEG camera
    outs = C.run_model(onnx_path, {n: C.decode_frame(gdir / n) for n in C.GOLDEN_FRAMES})
    expected = {n: {"boxes": o["boxes"].tolist(), "scores": o["scores"].tolist()} for n, o in outs.items()}
    (gdir / C.EXPECTED_FILE).write_text(json.dumps(expected, indent=1), encoding="utf-8", newline="\n")


def _camera_profile() -> tuple[int, str]:
    prof = C.load_profile(repo_root() / PROFILE_REL)
    return int(prof["left_strip"]["crop_left_px"]), str(prof["profile_version"])


def _dev_frames() -> tuple[list[str], list[str]]:
    """(dev-val có nhãn từ manifest, khung dev còn lại) — đều sắp theo tên."""
    from xla.eval.detection import read_manifest, read_split

    split = read_split(repo_root() / SPLIT_REL)
    manifest = read_manifest(repo_root() / LABELS_REL / "manifest.json") or {}
    devval = sorted(manifest.get("devval_frames", []))
    if not devval:
        raise ValueError(f"manifest {LABELS_REL / 'manifest.json'} không có devval_frames")
    dev = sorted(f for f, r in split.items() if r["split"] == "dev")
    not_dev = [f for f in devval if split.get(f, {}).get("split") != "dev"]
    if not_dev:
        raise ValueError(f"{len(not_dev)} khung dev-val không thuộc split dev (vd. {not_dev[:3]})")
    return devval, [f for f in dev if f not in set(devval)]


def export_bundle(weights: str | Path, out_dir: str | Path, *, version: str, conf_threshold: float,
                  metrics: dict, train_run: str, imgsz: int = IMGSZ, parity_out: str | Path | None = None,
                  skip_cuda: bool = False) -> dict:
    """Xuất bundle detector; trả báo cáo ``{errors, warnings, parity, ms_per_frame, onnx_sha256, ...}``.

    ``errors`` rỗng = bundle qua ``validate_bundle`` và parity đạt. Tham số sai → ``UsageError``."""
    import onnx

    t_start = time.time()
    weights, bundle = Path(weights), Path(out_dir)
    _check_args(weights, bundle, version, conf_threshold, imgsz)
    metrics = check_metrics(dict(metrics))
    import onnxruntime as ort
    import torch  # import trước khi tạo session ORT CUDA: nạp DLL CUDA của torch (pyproject.toml)
    import ultralytics

    crop_left, profile_version = _camera_profile()
    devval, dev_rest = _dev_frames()
    frames_dir = repo_root() / FRAMES_REL
    dv_frames = _load_frames(devval, frames_dir)
    warnings: list[str] = []
    errors: list[str] = []
    bundle.mkdir(parents=True, exist_ok=True)
    onnx_path = bundle / "detector.onnx"
    work_root = cache_dir() / "xla-detect-export"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        work = Path(tmp)
        core = onnx.load(str(export_core(weights, work, imgsz)))
        wrapped = build_detector_model(core, imgsz=imgsz, crop_left=crop_left)
        model, simplified, simp_note = simplify_checked(wrapped, [dv_frames[f] for f in devval[:2]])
        if not simplified:
            warnings.append(simp_note)
        onnx.save(model, str(onnx_path))
        onnx_sha = C.sha256_file(onnx_path)

        cpu = ort_session(onnx_path)
        ort_cpu = {f: _rows(*run_detector(cpu, im)) for f, im in dv_frames.items()}
        timed = [C.decode_frame(frames_dir / dev_rest[int(i)])
                 for i in np.linspace(0, len(dev_rest) - 1, TIMED_FRAMES)]
        lock_wait = 0.0
        if skip_cuda:
            torch.set_num_threads(CPU_THREADS)
            pt = predict_pt(weights, dv_frames, crop_left, imgsz, "cpu")
            ort_cuda = None
            cpu_perf = cpu_performance_pct()
            ms = measure_ms(cpu, timed)
            ms_meta = {"device": platform.processor() or "CPU", "runtime": f"onnxruntime {ort.__version__} CPU EP"}
            warnings.append("--skip-cuda: không parity CUDA, ms đo trên ORT CPU EP (bundle phát hành cần CUDA EP)")
        else:
            from xla.gpu_lock import gpu_lock

            t_wait = time.perf_counter()
            with gpu_lock(f"phase11 export_onnx {bundle.name} (parity + ms)", timeout_s=GPU_LOCK_TIMEOUT_S):
                lock_wait = time.perf_counter() - t_wait
                pt = predict_pt(weights, dv_frames, crop_left, imgsz, 0)
                torch.cuda.empty_cache()
                gpu = ort_session(onnx_path, cuda=True)
                ort_cuda = {f: _rows(*run_detector(gpu, im)) for f, im in dv_frames.items()}
                cpu_perf = cpu_performance_pct()
                ms = measure_ms(gpu, timed)
                ms_meta = {"device": torch.cuda.get_device_name(0),
                           "runtime": f"onnxruntime-gpu {ort.__version__} CUDA EP"}
                del gpu

        ref = score_devval(pt["crop_rect"], work / "pt_crop_rect.csv")
        parity: dict[str, Any] = {
            "metric": "mAP50", "split": "devval", "tolerance": PARITY_TOL, "n_frames": ref["n_frames"],
            "reference": {"name": "pt", "pipeline": f"YOLO.predict mặc định (rect=True) trên khung[:, {crop_left}:], "
                                                    f"x += {crop_left}"} | ref,
            "candidates": {"ort_cpu": parity_entry(score_devval(ort_cpu, work / "ort_cpu.csv"), ref),
                           "ort_cuda": (parity_entry(score_devval(ort_cuda, work / "ort_cuda.csv"), ref)
                                        if ort_cuda is not None else None)},
        }
        cpu_map = parity["candidates"]["ort_cpu"]["mAP50"]
        square = score_devval(pt["crop_square"], work / "pt_crop_square.csv")
        edge = score_devval(pt["full_edge"], work / "pt_full_edge.csv")
        parity["info"] = {
            "pt_crop_square": square | {
                "pipeline": f"YOLO.predict rect=False (letterbox vuông {imgsz}×{imgsz} như graph) trên khung[:, "
                            f"{crop_left}:]",
                "delta_map50_ort_cpu_minus_this": round(cpu_map - square["mAP50"], 6),
                "note": "tham khảo, không chặn: cùng tiền xử lý với graph, lệch ở đây mới là lỗi xuất"},
            "pt_full_edge16": edge | {
                "pipeline": f"YOLO.predict trên khung nguyên, bỏ hộp x1 < {crop_left} (đường của mốc / E1)",
                "delta_map50_ort_cpu_minus_this": round(cpu_map - edge["mAP50"], 6),
                "note": "tham khảo, không chặn: khác đầu vào (cắt mép trong graph vs lọc hộp sau)"}}
    perf_warn = _cpu_perf_warning(cpu_perf)
    if perf_warn:
        warnings.append(perf_warn)
    gated = [v for v in parity["candidates"].values() if v is not None]
    parity["pass"] = all(v["pass"] for v in gated)
    for k, v in parity["candidates"].items():
        if v is not None and not v["pass"]:
            errors.append(f"parity {k}: |ΔmAP50| = {abs(v['delta_map50']):.4f} > {PARITY_TOL} so với .pt")

    golden = pick_golden(dev_rest, frames_dir, cpu, conf_threshold)
    del cpu
    _write_golden(bundle, onnx_path, golden, frames_dir)
    meta: dict[str, Any] = {
        "contract_version": C.CONTRACT_VERSION, "model": "detector", "version": version, "onnx_sha256": onnx_sha,
        "frame_size": [C.FRAME_W, C.FRAME_H], "crop_left": crop_left, "camera_profile_version": profile_version,
        "imgsz": imgsz, "score_floor": SCORE_FLOOR, "nms_iou": NMS_IOU, "max_det": MAX_DET,
        "conf_threshold": float(conf_threshold), "input_variant": "raw", "train_run": train_run, "metrics": metrics,
        "ms_per_frame": ms_meta | {"p50": ms["p50"], "p95": ms["p95"], "n_frames": ms["n_frames"],
                                   "warmup": ms["warmup"], "cpu_perf_pct": cpu_perf},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "golden_sources": golden,
        "export": {"tool": "xla.detect.export_onnx", "weights": weights.name, "weights_sha256": C.sha256_file(weights),
                   "ultralytics": ultralytics.__version__, "onnx": onnx.__version__, "opset": C.OPSET,
                   "onnxsim": simplified, "nms": "ONNX NonMaxSuppression", "parity_pass": parity["pass"]},
    }
    (bundle / C.META_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    errors += C.validate_bundle(bundle)
    parity |= {"onnx_sha256": onnx_sha, "weights_sha256": meta["export"]["weights_sha256"],
               "created": meta["created"], "bundle": bundle.as_posix()}
    if parity_out is not None:
        p = Path(parity_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(parity, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"bundle": bundle.as_posix(), "errors": errors, "warnings": warnings, "parity": parity,
            "ms_per_frame": meta["ms_per_frame"], "onnx_sha256": onnx_sha, "onnxsim": simp_note,
            "golden_sources": golden, "gpu_lock_wait_s": round(lock_wait, 1), "wall_s": round(time.time() - t_start, 1)}


# ---- CLI ---------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _ultralytics_log_to_stderr() -> Iterator[None]:
    """Log của ultralytics in ra stdout; dời sang stderr để stdout chỉ còn báo cáo (``--json``)."""
    from ultralytics.utils import LOGGER

    moved = [(h, h.stream) for h in LOGGER.handlers if isinstance(h, logging.StreamHandler)]
    for h, _ in moved:
        h.setStream(sys.stderr)
    try:
        yield
    finally:
        for h, stream in moved:
            h.setStream(stream)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.export_onnx", description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True, help="checkpoint .pt (ultralytics)")
    ap.add_argument("--out", type=Path, required=True, help="thư mục bundle, tên detector-v<version>")
    ap.add_argument("--version", required=True, help="semver, trùng tên thư mục")
    ap.add_argument("--conf", type=float, required=True, help="conf_threshold (chọn trên dev-val)")
    ap.add_argument("--metrics", type=Path, required=True, help="JSON chỉ số test (5 khoá §4)")
    ap.add_argument("--train-run", required=True, help="id lần huấn luyện trong results/")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--parity-out", type=Path, default=None, help="ghi parity JSON, vd. results/e2/parity.json")
    ap.add_argument("--skip-cuda", action="store_true", help="không dùng GPU (thử); bundle phát hành phải đo CUDA")
    ap.add_argument("--json", action="store_true", help="in báo cáo JSON một dòng")
    a = ap.parse_args(argv)
    try:
        if not a.metrics.is_file():
            raise UsageError(f"không có file metrics {a.metrics}")
        metrics = load_metrics(a.metrics)
        _check_args(a.weights, a.out, a.version, a.conf, a.imgsz)
    except (UsageError, ValueError) as e:
        print(f"LỖI tham số: {e}", file=sys.stderr)
        return 2
    try:
        with _ultralytics_log_to_stderr():
            rep = export_bundle(a.weights, a.out, version=a.version, conf_threshold=a.conf, metrics=metrics,
                                train_run=a.train_run, imgsz=a.imgsz, parity_out=a.parity_out,
                                skip_cuda=a.skip_cuda)
    except Exception as e:  # noqa: BLE001 - lỗi lúc chạy: báo rõ, exit 1 (không phải lỗi cách gọi)
        print(f"LỖI: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(rep, ensure_ascii=False))
    else:
        c = rep["parity"]["candidates"]
        print(f"bundle {rep['bundle']}: {'ĐẠT' if not rep['errors'] else 'LỖI'}; ΔmAP50 cpu "
              f"{c['ort_cpu']['delta_map50']:+.4f}, cuda "
              f"{c['ort_cuda']['delta_map50'] if c['ort_cuda'] else None}; ms p50/p95 "
              f"{rep['ms_per_frame']['p50']}/{rep['ms_per_frame']['p95']}")
        for e in rep["errors"] + rep["warnings"]:
            print(f"  - {e}")
    return 0 if not rep["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())

