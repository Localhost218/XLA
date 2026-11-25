"""Phần chung của verifier (phase 07): cấu hình, hình học vùng cắt §9, dữ liệu thật (fold, cal, test), IoU, JSON ra.

Hình học vùng cắt ở đây là **bản tham chiếu PyTorch** của graph ONNX (``export_onnx``): cùng phép nới hộp, kẹp và
``torchvision.ops.roi_align(aligned=False, sampling_ratio=2)`` = ``RoiAlign output_half_pixel`` của §9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xla import contract as C
from xla.paths import repo_root

CONFIG_REL = Path("configs/verifier.yaml")
FRAMES_REL = Path("data/real/frames")
SPLIT_REL = Path("data/real/split.json")
NEG_SPLIT_REL = Path("data/real/neg_flight_split.json")
FOLDS_REL = Path("data/real/labels/folds.json")
DEVPOOL_LABELS_REL = Path("data/real/labels/devpool/review/labels.json")
FOLD_INFO_REL = Path("data/real/labels/devpool/fold_info.json")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_cfg(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else repo_root() / CONFIG_REL
    if not p.is_absolute():
        p = repo_root() / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def runs_dir(cfg: dict[str, Any]) -> Path:
    d = Path(cfg["runs_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(cfg: dict[str, Any]) -> Path:
    d = repo_root() / cfg["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def emit(obj: dict[str, Any], as_json: bool) -> None:
    print(json.dumps(obj, ensure_ascii=False) if as_json else json.dumps(obj, indent=1, ensure_ascii=False))
    sys.stdout.flush()


# ---- dữ liệu thật --------------------------------------------------------------------------------------------------
def real_sets(root: Path | None = None) -> dict[str, Any]:
    """Khung mine / cal / holdout (theo phiên), fold devpool, nhãn devpool, khung test v1, khung hard, khung giữ lại."""
    root = root or repo_root()
    neg = json.loads((root / NEG_SPLIT_REL).read_text(encoding="utf-8"))
    folds = json.loads((root / FOLDS_REL).read_text(encoding="utf-8"))["folds"]
    info = json.loads((root / FOLD_INFO_REL).read_text(encoding="utf-8"))
    lab = json.loads((root / DEVPOOL_LABELS_REL).read_text(encoding="utf-8"))
    split = {f["frame"]: f for f in json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))["frames"]}
    by: dict[str, list[str]] = {"mine": [], "cal": [], "holdout": []}
    session = {}
    for f in neg["frames"]:
        by.setdefault(f["split"], []).append(f["frame"])
        session[f["frame"]] = f["session"]
    fold_of = {fr: i for i, fs in enumerate(folds) for fr in fs}
    sess_fold = {s: i for i, ss in enumerate(info["fold_sessions"]) for s in ss}
    for fr in by["mine"]:  # âm mine chia theo phiên của nó -> fold chứa phiên đó
        if session[fr] not in sess_fold:
            raise ValueError(f"phiên {session[fr]} của khung mine {fr} không thuộc fold nào")
        fold_of[fr] = sess_fold[session[fr]]
    # khối 30 s của cal (cal_mode block): mọi khung của khối, kể cả khung có người, bị giữ lại (phase 03)
    cal_blocks = {split[f]["block"] for f in by["cal"] + by["holdout"] if f in split}
    reserved = {f for f, r in split.items() if r.get("block") in cal_blocks and neg.get("cal_mode") == "block"}
    return {
        "mine": sorted(by["mine"]),
        "cal": sorted(by["cal"]),
        "holdout": sorted(by["holdout"]),
        "reserved": sorted(reserved | set(by["cal"]) | set(by["holdout"])),
        "devpool": sorted(f for fs in folds for f in fs),
        "fold_of": fold_of,
        "n_folds": len(folds),
        "session_of": session,
        "gt": {k: [list(map(float, b)) for b in v] for k, v in lab["boxes"].items()},
        "hard": set(lab.get("hard", [])),
        "test": sorted(f for f, r in split.items() if r["split"] == "test"),
    }


def forbidden_frames(rs: dict[str, Any]) -> set[str]:
    """Khung không bao giờ vào manifest khai thác / huấn luyện: cal, holdout, mọi khung giữ lại, test v1."""
    return set(rs["cal"]) | set(rs["holdout"]) | set(rs["reserved"]) | set(rs["test"])


def frame_path(name: str, root: Path | None = None) -> Path:
    return (root or repo_root()) / FRAMES_REL / name


# ---- hình học ------------------------------------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def context_boxes(boxes: Any, context_scale: float, min_side: float, x_lo: float = C.CROP_LEFT_V1,
                  w: float = C.FRAME_W, h: float = C.FRAME_H) -> Any:
    """§9: tâm giữ, cạnh = max(max(w, h) * context_scale, min_side), kẹp vào [x_lo, w] x [0, h]. numpy hoặc torch."""
    is_t = not isinstance(boxes, np.ndarray)
    if is_t:
        import torch

        mx, cl = torch.maximum, torch.clamp
    else:
        mx, cl = np.maximum, np.clip
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    side = mx(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * context_scale
    side = mx(side, side * 0 + min_side)
    half = side * 0.5
    x1, x2 = cl(cx - half, x_lo, w), cl(cx + half, x_lo, w)
    y1, y2 = cl(cy - half, 0, h), cl(cy + half, 0, h)
    if is_t:
        import torch

        return torch.stack([x1, y1, x2, y2], 1)
    return np.stack([x1, y1, x2, y2], 1).astype(np.float32)


def roi_crops(frame_bgr: np.ndarray, ctx_boxes: np.ndarray, crop_px: int) -> np.ndarray:
    """float32 [N, 3, c, c] BGR 0–255: ``roi_align(aligned=False, sampling_ratio=2)`` trên khung HWC uint8."""
    import torch
    from torchvision.ops import roi_align

    if not len(ctx_boxes):
        return np.zeros((0, 3, crop_px, crop_px), np.float32)
    x = torch.from_numpy(np.ascontiguousarray(frame_bgr)).permute(2, 0, 1)[None].float()
    b = torch.from_numpy(np.asarray(ctx_boxes, np.float32).reshape(-1, 4))
    rois = torch.cat([torch.zeros(len(b), 1), b], 1)
    out = roi_align(x, rois, (crop_px, crop_px), spatial_scale=1.0, sampling_ratio=C.VERIFIER_SAMPLING_RATIO,
                    aligned=False)
    return out.numpy()
