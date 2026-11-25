"""Bộ chấm nhận diện người trên khung bay thật — một bộ chấm cho mọi model của dự án (phase 09, hợp đồng §7, D10/D11).

Đầu vào:

- Dự đoán: CSV ``frame,x1,y1,x2,y2,conf`` (+ cột tuỳ chọn ``ms``), px khung gốc 640×480, không lọc ngưỡng.
- Nhãn: ``<labels-dir>/test/<stem>.txt``, ``<labels-dir>/devval/<stem>.txt`` (YOLO ``0 cx cy w h`` chuẩn hoá theo
  640×480; rỗng = không người). Thư mục con không có thì đọc phẳng ``<labels-dir>/<stem>.txt`` (nhãn tạm).
- ``<labels-dir>/manifest.json`` (phase 08): ``devval_frames``, ``hard_frames``, ``approved_by_owner``.
- ``data/real/split.json``: tập (test lấy từ đây, D10), khối 30 s, độ cao.

Số đo: mAP50, mAP50-95, AP-small (area < 32²), AR-small, AR (pycocotools, maxDets 100); P/R/F1 ở ngưỡng truyền vào và ở
ngưỡng tối ưu F1 **chọn trên dev-val** (``select_f1_threshold`` từ chối khung test); báo nhầm trung bình mỗi khung âm
(khung của tập không người + mọi khung ``neg``); ms/khung từ cột ``ms``. CI 95 % bằng bootstrap theo khối 30 s
(khối bay và khối âm lấy lại riêng, có hoàn lại). Tách theo bin độ cao (``xla.camera.split.ALT_BINS``) và cờ ``hard``.

Ghép hộp dùng ``COCOeval.evaluate`` của pycocotools một lần; phần cộng dồn (accumulate) viết lại dạng vectơ để bootstrap
nhanh, và mỗi lần chạy tự đối chiếu với ``COCOeval.accumulate/summarize`` trên toàn tập (lệch là lỗi, không im lặng).

AX::

    python -m xla.eval.detection --pred results/baseline/pred/full.csv --split test --json
    python -m xla.eval.detection compare --pred-a <csv A> --pred-b <csv B> --split test --json
    # exit 0 xong; 1 CSV có khung không có trong split.json / thiếu nhãn / dữ liệu sai; 2 tham số sai (D7)

``compare``: hiệu B − A ghép cặp — cùng một bộ mẫu khối bootstrap cho cả hai model — với CI 95 % và ``p_b_better``;
ngưỡng vận hành mỗi model là ``--thr-a``/``--thr-b`` hoặc ngưỡng F1 tối đa của chính model đó trên dev-val.
Báo nhầm/khung âm tách ``fp_per_neg_flight`` (khung bay không người của tập chấm) và ``fp_per_neg_indoor`` (khung
``neg`` trong nhà) bên cạnh số gộp ``fp_per_neg_frame``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera.split import alt_bin
from xla.paths import repo_root

IMG_W, IMG_H = 640, 480
SMALL_AREA = 32.0**2
MAX_DET = 100
REC_THRS = np.linspace(0.0, 1.0, 101)
N_IOU = 10  # 0.50:0.05:0.95, mặc định pycocotools
EVAL_SPLITS = ("test", "devval", "devpool")
RESERVED_KINDS = ("cal", "holdout")  # khung neg_flight_split.json không bao giờ chọn/đếm âm (§14.2, D5)
DEFAULT_THRESHOLDS = (0.25,)
BOOT_N = 2000
BOOT_SEED = 20261118
CI_PCT = 95.0
LABELS_REL = Path("data/real/labels")
SPLIT_REL = Path("data/real/split.json")
NEG_SPLIT_REL = Path("data/real/neg_flight_split.json")
FOLDS_REL = Path("data/real/labels/folds.json")
PRED_FIELDS = ("frame", "x1", "y1", "x2", "y2", "conf")
_AREA_INDEX = {"all": 0, "small": 1}  # thứ tự areaRng mặc định của COCOeval: all, small, medium, large
_EPS = np.spacing(1)


class EvalInputError(Exception):
    """Dữ liệu vào sai (khung lạ trong CSV, thiếu nhãn, nhãn hỏng) — CLI trả exit 1."""


class TestLeakError(ValueError):
    """Chọn ngưỡng/checkpoint trên khung test (D10)."""


# ---------------------------------------------------------------- đọc dữ liệu


def read_split(path: Path) -> dict[str, dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {f["frame"]: f for f in doc["frames"]}


def read_yolo_label(path: Path) -> np.ndarray:
    """Trả mảng (N, 4) xyxy px khung gốc 640×480."""
    boxes = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or parts[0] != "0":
            raise EvalInputError(f"{path}:{n}: cần '0 cx cy w h', gặp {line!r}")
        cx, cy, w, h = (float(v) for v in parts[1:])
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            raise EvalInputError(f"{path}:{n}: toạ độ ngoài [0, 1] hoặc hộp rỗng: {line!r}")
        boxes.append(((cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H, (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H))
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def load_labels(labels_dir: Path, sub: str, frames: list[str], allow_missing: bool,
                ) -> tuple[dict[str, np.ndarray], list[str], str]:
    """Trả (gt, khung thiếu nhãn, sha256 của các file nhãn đã đọc — tên + nội dung, theo thứ tự khung)."""
    base = labels_dir / sub if (labels_dir / sub).is_dir() else labels_dir
    gt: dict[str, np.ndarray] = {}
    missing = []
    digest = hashlib.sha256()
    for f in frames:
        p = base / (Path(f).stem + ".txt")
        if p.is_file():
            gt[f] = read_yolo_label(p)
            digest.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
        else:
            missing.append(f)
    if missing and not allow_missing:
        raise EvalInputError(f"thiếu {len(missing)} file nhãn trong {base} (vd. {missing[:3]}); "
                             "nhãn tạm chưa đủ thì thêm --allow-missing-labels")
    return gt, missing, digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_pred_csv(path: Path, known: set[str]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Trả ``{frame: (boxes (N,4), scores (N,))}`` và ``{frame: ms}``; khung không có trong ``known`` → EvalInputError."""
    rows: dict[str, list[tuple[float, ...]]] = {}
    ms: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not set(PRED_FIELDS) <= set(reader.fieldnames):
            raise EvalInputError(f"{path}: cần cột {','.join(PRED_FIELDS)}, gặp {reader.fieldnames}")
        has_ms = "ms" in reader.fieldnames
        for r in reader:
            rows.setdefault(r["frame"], []).append(tuple(float(r[k]) for k in PRED_FIELDS[1:]))
            if has_ms and r["ms"] not in ("", None):
                ms[r["frame"]] = float(r["ms"])
    unknown = sorted(set(rows) - known)
    if unknown:
        raise EvalInputError(f"{path}: {len(unknown)} khung không có trong split.json (vd. {unknown[:3]})")
    out = {}
    for f, rs in rows.items():
        a = np.asarray(rs, dtype=float)
        out[f] = (a[:, :4], a[:, 4])
    return out, ms


# ---------------------------------------------------------------- ghép hộp (pycocotools) và cộng dồn


@dataclass
class FrameEval:
    """Kết quả ghép của một khung: mỗi vùng diện tích (``all``/``small``) một bộ mảng, hoặc None khi không gt không dt."""

    frame: str
    n_gt: int
    areas: dict[str, dict[str, np.ndarray] | None] = field(default_factory=dict)


def _coco_eval(frames: list[str], gt: dict[str, np.ndarray],
               preds: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[list[FrameEval], np.ndarray]:
    """Chạy ``COCOeval.evaluate`` một lần; trả kết quả từng khung + ``stats`` của ``summarize`` (đối chiếu)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = [{"id": i + 1, "width": IMG_W, "height": IMG_H, "file_name": f} for i, f in enumerate(frames)]
    cats = [{"id": 1, "name": "person"}]
    gt_anns, dt_anns = [], []
    for i, f in enumerate(frames):
        for x1, y1, x2, y2 in gt.get(f, np.empty((0, 4))):
            w, h = x2 - x1, y2 - y1
            gt_anns.append({"id": len(gt_anns) + 1, "image_id": i + 1, "category_id": 1, "bbox": [x1, y1, w, h],
                            "area": w * h, "iscrowd": 0})
        boxes, scores = preds.get(f, (np.empty((0, 4)), np.empty(0)))
        for (x1, y1, x2, y2), s in zip(boxes, scores, strict=True):
            w, h = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
            dt_anns.append({"id": len(dt_anns) + 1, "image_id": i + 1, "category_id": 1, "bbox": [x1, y1, w, h],
                            "area": w * h, "score": float(s), "iscrowd": 0})
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt, coco_dt = COCO(), COCO()
        coco_gt.dataset = {"images": images, "categories": cats, "annotations": gt_anns}
        coco_dt.dataset = {"images": images, "categories": cats, "annotations": dt_anns}
        coco_gt.createIndex()
        coco_dt.createIndex()
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = [im["id"] for im in images]
        ev.params.catIds = [1]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    n_img = len(images)
    out = []
    for i, f in enumerate(frames):
        fe = FrameEval(frame=f, n_gt=len(gt.get(f, ())))
        for name, a in _AREA_INDEX.items():
            e = ev.evalImgs[a * n_img + i]
            if e is None:
                fe.areas[name] = None
                continue
            fe.areas[name] = {
                "scores": np.asarray(e["dtScores"][:MAX_DET], dtype=float),
                "tp": np.asarray(e["dtMatches"][:, :MAX_DET] > 0),
                "ig": np.asarray(e["dtIgnore"][:, :MAX_DET], dtype=bool),
                "npig": np.int64(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)),
            }
        out.append(fe)
    return out, np.asarray(ev.stats, dtype=float)


def _concat(evals: list[FrameEval], area: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    parts = [fe.areas[area] for fe in evals if fe.areas.get(area) is not None]
    npig = int(sum(p["npig"] for p in parts))
    if not parts:
        return np.empty(0), np.empty((N_IOU, 0), bool), np.empty((N_IOU, 0), bool), 0
    return (np.concatenate([p["scores"] for p in parts]), np.concatenate([p["tp"] for p in parts], axis=1),
            np.concatenate([p["ig"] for p in parts], axis=1), npig)


def accumulate(evals: list[FrameEval], area: str) -> tuple[np.ndarray, np.ndarray]:
    """Như ``COCOeval.accumulate``: trả (precision trung bình theo 101 mức recall, recall cuối) cho từng IoU (T,).

    Không có gt nào (sau ignore) → NaN (pycocotools ghi -1)."""
    scores, tp_m, ig, npig = _concat(evals, area)
    if npig == 0:
        return np.full(N_IOU, np.nan), np.full(N_IOU, np.nan)
    order = np.argsort(-scores, kind="mergesort")
    tp_m, ig = tp_m[:, order], ig[:, order]
    tps = np.cumsum(tp_m & ~ig, axis=1, dtype=float)
    fps = np.cumsum(~tp_m & ~ig, axis=1, dtype=float)
    nd = scores.size
    prec, rec = np.zeros(N_IOU), np.zeros(N_IOU)
    for t in range(N_IOU):
        if nd == 0:
            continue
        rc = tps[t] / npig
        pr = tps[t] / (fps[t] + tps[t] + _EPS)
        rec[t] = rc[-1]
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, REC_THRS, side="left")
        q = np.where(idx < nd, pr[np.minimum(idx, nd - 1)], 0.0)
        prec[t] = q.mean()
    return prec, rec


def prf_at(evals: list[FrameEval], thr: float) -> dict[str, float]:
    """P/R/F1 ở ngưỡng ``thr``, IoU 0,5, mọi diện tích (ghép tham lam theo điểm của COCO)."""
    scores, tp_m, ig, npig = _concat(evals, "all")
    keep = (scores >= thr) & ~ig[0]
    tp = int(np.count_nonzero(tp_m[0] & keep))
    fp = int(np.count_nonzero(~tp_m[0] & keep))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / npig if npig else float("nan")
    f1 = 2 * p * r / (p + r) if (tp and npig) else (0.0 if npig else float("nan"))
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": npig - tp}


def select_f1_threshold(evals: list[FrameEval], split_of: dict[str, str],
                        reserved: set[str] | frozenset[str] = frozenset()) -> dict[str, float]:
    """Ngưỡng tối đa F1 (IoU 0,5) — CHỈ trên khung dev; khung thuộc ``test`` (hoặc không rõ tập) hay thuộc ``reserved``
    (``cal`` / ``holdout`` của ``neg_flight_split.json``, §14.2) → TestLeakError."""
    leak = [fe.frame for fe in evals if split_of.get(fe.frame) != "dev" or fe.frame in reserved]
    if leak:
        raise TestLeakError(f"chọn ngưỡng chỉ được trên khung dev không phải cal/holdout (D5, D10); {len(leak)} khung "
                            f"vi phạm, vd. {leak[:3]}")
    scores, tp_m, ig, npig = _concat(evals, "all")
    keep = ~ig[0]
    scores, tp = scores[keep], tp_m[0][keep]
    if npig == 0 or scores.size == 0:
        raise EvalInputError("dev-val không có hộp người hoặc không có dự đoán: không chọn được ngưỡng F1")
    order = np.argsort(-scores, kind="mergesort")
    scores, tp = scores[order], tp[order]
    ctp = np.cumsum(tp)
    n = np.arange(1, scores.size + 1)
    last_of_tie = np.r_[scores[1:] != scores[:-1], True]  # chỉ cắt ở cuối nhóm điểm bằng nhau
    f1 = np.where(last_of_tie, 2 * ctp / (n + npig), -1.0)
    k = int(np.argmax(f1))
    return {"thr": float(scores[k]), "f1_devval": float(f1[k]), "precision_devval": float(ctp[k] / n[k]),
            "recall_devval": float(ctp[k] / npig), "n_frames_devval": len(evals)}


# ---------------------------------------------------------------- chỉ số + bootstrap

NEG_KINDS = (("all", "fp_per_neg_frame"), ("flight", "fp_per_neg_flight"), ("indoor", "fp_per_neg_indoor"))
"""Báo nhầm/khung âm: gộp; khung bay không người của tập chấm; khung ``neg`` (trong nhà) — với ``devpool`` là khung
``mine``. Khung ``cal`` / ``holdout`` (``RESERVED_KINDS``) không bao giờ là khung âm của bất kỳ tập chấm nào."""


def reserved_frames(path: Path | None = None) -> set[str]:
    """Khung ``cal`` + ``holdout`` của ``neg_flight_split.json`` (không có file → rỗng, như repo v1)."""
    p = path or repo_root() / NEG_SPLIT_REL
    if not p.is_file():
        return set()
    return {f["frame"] for f in json.loads(p.read_text(encoding="utf-8"))["frames"] if f["split"] in RESERVED_KINDS}


def mine_frames(path: Path | None = None) -> list[str]:
    p = path or repo_root() / NEG_SPLIT_REL
    return sorted(f["frame"] for f in json.loads(p.read_text(encoding="utf-8"))["frames"] if f["split"] == "mine")


def devpool_frames(path: Path | None = None) -> list[str]:
    """Khung devpool = hợp các fold của ``folds.json`` (§14.2)."""
    p = path or repo_root() / FOLDS_REL
    return sorted(f for fold in json.loads(p.read_text(encoding="utf-8"))["folds"] for f in fold)


def _metrics(evals: list[FrameEval], neg_fp: dict[float, dict[str, list[int]]], thresholds: list[float],
             ) -> dict[str, float]:
    """``neg_fp[thr][kind]`` = số hộp ≥ thr trên từng khung âm loại ``kind`` trong mẫu (đã nhân theo bootstrap)."""
    p_all, r_all = accumulate(evals, "all")
    p_small, r_small = accumulate(evals, "small")
    m = {"mAP50": p_all[0], "mAP50_95": np.mean(p_all), "AP_small": np.mean(p_small), "AR_small": np.mean(r_small),
         "AR": np.mean(r_all)}
    for thr in thresholds:
        prf = prf_at(evals, thr)
        m |= {f"precision@{thr:g}": prf["precision"], f"recall@{thr:g}": prf["recall"], f"f1@{thr:g}": prf["f1"]}
        for kind, key in NEG_KINDS:
            fp = neg_fp[thr][kind]
            m[f"{key}@{thr:g}"] = float(np.mean(fp)) if fp else float("nan")
    return {k: float(v) for k, v in m.items()}


@dataclass
class EvalSet:
    evals: dict[str, FrameEval]
    neg_frames: list[str]  # khung âm: khung của tập không có gt + khung ``neg``
    neg_indoor: set[str]  # khung ``neg`` (trong nhà); phần còn lại của neg_frames là khung bay không người
    block_of: dict[str, int]
    raw: dict[str, np.ndarray]  # điểm thô mỗi khung (đếm báo nhầm, không cắt maxDets)
    flight_blocks: list[int]
    neg_blocks: list[int]
    neg_set: set[str] = field(init=False)
    _fp_cache: dict[tuple[float, str], int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.neg_set = set(self.neg_frames)

    def fp_count(self, thr: float, frame: str) -> int:
        key = (thr, frame)
        if key not in self._fp_cache:
            self._fp_cache[key] = int(np.count_nonzero(self.raw.get(frame, np.empty(0)) >= thr))
        return self._fp_cache[key]


def _neg_counts(s: EvalSet, frames: list[str], thresholds: list[float]) -> dict[float, dict[str, list[int]]]:
    negs = [f for f in frames if f in s.neg_set]
    out = {}
    for thr in thresholds:
        c = [(f in s.neg_indoor, s.fp_count(thr, f)) for f in negs]
        out[thr] = {"all": [n for _, n in c], "flight": [n for ind, n in c if not ind],
                    "indoor": [n for ind, n in c if ind]}
    return out


def _subset_metrics(s: EvalSet, frames: list[str], keep: set[str] | None, thresholds: list[float]) -> dict[str, float]:
    fs = frames if keep is None else [f for f in frames if f in keep]
    return _metrics([s.evals[f] for f in fs if f in s.evals], _neg_counts(s, fs, thresholds), thresholds)


def resample(s: EvalSet, n: int, seed: int) -> list[list[str]]:
    """``n`` mẫu bootstrap theo khối: mỗi mẫu lấy lại (có hoàn lại) đúng số khối bay và số khối âm, trả danh sách khung.

    Mẫu chỉ phụ thuộc tập khối + seed, nên hai model chấm trên cùng nhãn nhận đúng cùng mẫu (so sánh ghép cặp)."""
    rng = np.random.default_rng(seed)
    members: dict[int, list[str]] = {}
    for f, b in s.block_of.items():
        members.setdefault(b, []).append(f)
    fb, nb = np.asarray(s.flight_blocks), np.asarray(s.neg_blocks)
    out = []
    for _ in range(n):
        picked = list(rng.choice(fb, fb.size)) if fb.size else []
        picked += list(rng.choice(nb, nb.size)) if nb.size else []
        out.append([f for b in picked for f in members[int(b)]])
    return out


def _ci(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = (100 - CI_PCT) / 2, 100 - (100 - CI_PCT) / 2
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (np.nan, np.nan)


def bootstrap(s: EvalSet, subsets: dict[str, set[str] | None], thresholds: list[float], samples: list[list[str]],
              ) -> dict[str, dict[str, tuple[float, float]]]:
    """CI 95 % theo khối trên các mẫu của ``resample``."""
    rows: dict[str, list[dict[str, float]]] = {k: [] for k in subsets}
    for frames in samples:
        for name, keep in subsets.items():
            rows[name].append(_subset_metrics(s, frames, keep, thresholds))
    return {name: {k: _ci([r[k] for r in rs]) for k in rs[0]} for name, rs in rows.items() if rs}


# ---------------------------------------------------------------- chuẩn bị một model


@dataclass
class Prepared:
    """Mọi thứ của một CSV trên một tập: kết quả ghép, khung âm, tập con, ngưỡng (đã gồm ngưỡng F1 dev-val nếu có)."""

    s: EvalSet
    frames: list[str]
    neg_split: list[str]
    gt: dict[str, np.ndarray]
    missing: list[str]
    labels_sha: str
    subsets: dict[str, set[str] | None]
    thresholds: list[float]
    f1_info: dict[str, Any] | None
    stats: np.ndarray
    preds: dict[str, tuple[np.ndarray, np.ndarray]]
    ms: dict[str, float]
    manifest: dict[str, Any] | None
    manifest_path: Path
    labels_dir: Path
    devval_labels_dir: Path

    @property
    def all_frames(self) -> list[str]:
        return self.frames + self.neg_split

    @property
    def official(self) -> bool:
        return bool((self.manifest or {}).get("approved_by_owner", False)) and not self.missing


def prepare(pred_csv: Path, split: str = "test", labels_dir: Path | None = None, manifest_path: Path | None = None,
            split_json: Path | None = None, thresholds: list[float] | None = None, allow_missing_labels: bool = False,
            f1_from_devval: bool = True, devval_labels_dir: Path | None = None) -> Prepared:
    if split not in EVAL_SPLITS:
        raise ValueError(f"split phải là một trong {EVAL_SPLITS}")
    root = repo_root()
    labels_dir = labels_dir or root / LABELS_REL
    devval_labels_dir = devval_labels_dir or labels_dir
    manifest_path = manifest_path or labels_dir / "manifest.json"
    split_rows = read_split(split_json or root / SPLIT_REL)
    split_of = {f: r["split"] for f, r in split_rows.items()}
    manifest = read_manifest(manifest_path)
    preds, ms = read_pred_csv(pred_csv, set(split_rows))
    thresholds = list(thresholds or [])

    reserved = reserved_frames()
    devval = list((manifest or {}).get("devval_frames", []))
    bad_dv = [f for f in devval if split_of.get(f) != "dev"]
    if bad_dv:
        raise EvalInputError(f"manifest: {len(bad_dv)} khung dev-val không thuộc dev (vd. {bad_dv[:3]})")
    test_frames = sorted(f for f, sp in split_of.items() if sp == "test")
    if manifest and manifest.get("test_frames") and set(manifest["test_frames"]) != set(test_frames):
        raise EvalInputError("manifest.test_frames khác tập test của split.json (D10: split.json là nguồn sự thật)")
    frames = {"test": test_frames, "devval": sorted(devval), "devpool": devpool_frames()}[split]
    if not frames:
        raise EvalInputError(f"tập {split} rỗng (dev-val lấy từ manifest {manifest_path})")
    # v2: devpool không bao giờ chứa cal/holdout. dev-val v1 (lịch sử) giữ nguyên 100 khung, kể cả 14 khung nay thuộc
    # khối cal — chỉ để tái lập số v1 (scripts/reproduce.py); mọi lựa chọn v2 chạy trên devpool.
    bad = [f for f in frames if split == "devpool" and (f in reserved or split_of.get(f) != "dev")]
    if bad:
        raise EvalInputError(f"tập {split}: {len(bad)} khung cal/holdout hoặc không thuộc dev (vd. {bad[:3]})")
    gt, missing, labels_sha = load_labels(labels_dir if split == "test" else devval_labels_dir, split, frames,
                                          allow_missing_labels)
    frames = [f for f in frames if f in gt]

    f1_info = None
    if f1_from_devval and split == "devpool":  # OOF: ngưỡng F1 tối đa trên chính devpool (dự đoán ngoài fold)
        dp_evals, _ = _coco_eval(frames, gt, preds)
        f1_info = select_f1_threshold(dp_evals, split_of, reserved) | {"source": "devpool OOF",
                                                                       "devval_labels_sha256": labels_sha}
        if round(f1_info["thr"], 6) not in thresholds:
            thresholds.append(round(f1_info["thr"], 6))
    elif f1_from_devval and devval:
        dv_gt, dv_missing, dv_sha = load_labels(devval_labels_dir, "devval", sorted(devval), allow_missing_labels)
        dv_evals, _ = _coco_eval(sorted(dv_gt), dv_gt, preds)
        f1_info = select_f1_threshold(dv_evals, split_of) | {"devval_labels_missing": len(dv_missing),
                                                             "devval_labels_sha256": dv_sha}
        if round(f1_info["thr"], 6) not in thresholds:
            thresholds.append(round(f1_info["thr"], 6))

    evals_list, stats = _coco_eval(frames, gt, preds)
    neg_split = (mine_frames() if split == "devpool" else
                 sorted(f for f, sp in split_of.items() if sp == "neg" and f not in reserved))
    s = EvalSet(evals={fe.frame: fe for fe in evals_list}, neg_frames=[f for f in frames if len(gt[f]) == 0] + neg_split,
                neg_indoor=set(neg_split), block_of={f: int(split_rows[f]["block"]) for f in frames + neg_split},
                raw={f: v[1] for f, v in preds.items()},
                flight_blocks=sorted({int(split_rows[f]["block"]) for f in frames}),
                neg_blocks=sorted({int(split_rows[f]["block"]) for f in neg_split}))

    hard = set((manifest or {}).get("hard_frames", []))
    bins = {f: alt_bin(float(split_rows[f]["alt_m"])) for f in frames if split_rows[f].get("alt_m") is not None}
    subsets: dict[str, set[str] | None] = {"all": None}
    for b in sorted(set(bins.values())):
        subsets[f"alt:{b}"] = {f for f, v in bins.items() if v == b} | set(neg_split)
    if hard:
        subsets["hard"] = (hard & set(frames)) | set(neg_split)
        subsets["not_hard"] = (set(frames) - hard) | set(neg_split)
    return Prepared(s, frames, neg_split, gt, missing, labels_sha, subsets, thresholds, f1_info, stats, preds, ms,
                    manifest, manifest_path, labels_dir, devval_labels_dir)


# ---------------------------------------------------------------- chấm một model


def evaluate(pred_csv: Path, split: str = "test", labels_dir: Path | None = None, manifest_path: Path | None = None,
             split_json: Path | None = None, thresholds: list[float] | None = None, boot_n: int = BOOT_N,
             seed: int = BOOT_SEED, allow_missing_labels: bool = False, f1_from_devval: bool = True,
             devval_labels_dir: Path | None = None) -> dict[str, Any]:
    """Chấm một CSV; ``devval_labels_dir`` mặc định = ``labels_dir`` (nhãn tạm test và dev-val có thể ở hai nơi)."""
    pr = prepare(pred_csv, split, labels_dir, manifest_path, split_json, list(thresholds or DEFAULT_THRESHOLDS),
                 allow_missing_labels, f1_from_devval, devval_labels_dir)
    s, thresholds = pr.s, pr.thresholds
    point = {name: _subset_metrics(s, pr.all_frames, keep, thresholds) for name, keep in pr.subsets.items()}
    _check_against_pycocotools(point["all"], pr.stats)
    ci = bootstrap(s, pr.subsets, thresholds, resample(s, boot_n, seed)) if boot_n > 0 else {}

    def _pack(name: str) -> dict[str, Any]:
        return {k: {"value": _nan_none(v), "ci95": [_nan_none(x) for x in ci.get(name, {}).get(k, (np.nan, np.nan))]}
                for k, v in point[name].items()}

    def _counts(keep: set[str] | None) -> dict[str, int]:
        fs = [f for f in pr.frames if keep is None or f in keep]
        return {"frames": len(fs), "frames_with_person": sum(len(pr.gt[f]) > 0 for f in fs),
                "gt_boxes": int(sum(len(pr.gt[f]) for f in fs))}

    timed = [pr.ms[f] for f in pr.all_frames if f in pr.ms]
    return {
        "pred": str(pred_csv), "split": split, "labels_dir": str(pr.labels_dir), "labels_sha256": pr.labels_sha,
        "devval_labels_dir": str(pr.devval_labels_dir),
        "manifest": str(pr.manifest_path) if pr.manifest else None,
        "labels_approved_by_owner": bool((pr.manifest or {}).get("approved_by_owner", False)),
        "official": pr.official,
        "note": None if pr.official else "nhãn chưa được chủ dự án duyệt hoặc thiếu nhãn — chưa phải số chính thức",
        "n": _counts(None) | {"neg_split_frames": len(pr.neg_split), "negative_frames": len(s.neg_frames),
                              "negative_frames_flight": len(s.neg_frames) - len(pr.neg_split),
                              "negative_frames_indoor": len(pr.neg_split), "labels_missing": len(pr.missing),
                              "pred_frames_other_splits": len(set(pr.preds) - set(pr.all_frames))},
        "thresholds": thresholds, "f1_threshold_from_devval": pr.f1_info,
        "metrics": _pack("all"),
        "by_alt_bin": {k[4:]: _counts(pr.subsets[k]) | {"metrics": _pack(k)} for k in pr.subsets if k.startswith("alt:")},
        "by_hard": {k: _counts(pr.subsets[k]) | {"metrics": _pack(k)} for k in ("hard", "not_hard") if k in pr.subsets},
        "ms_per_frame": ({"mean": float(np.mean(timed)), "median": float(np.median(timed)), "n_frames": len(timed)}
                         if timed else None),
        "bootstrap": _boot_info(s, boot_n, seed) | {"note": "subset alt/hard giữ mọi khung neg cho báo nhầm/khung âm"},
    }


def _boot_info(s: EvalSet, boot_n: int, seed: int) -> dict[str, Any]:
    return {"n": boot_n, "seed": seed, "unit": "khối 30 s của split.json", "ci_pct": CI_PCT,
            "n_blocks": {"flight": len(s.flight_blocks), "neg": len(s.neg_blocks)}}


# ---------------------------------------------------------------- so sánh ghép cặp hai model

COMPARE_KEYS = (("mAP50", True), ("mAP50_95", True), ("AP_small", True), ("AR_small", True), ("precision@op", True),
                ("recall@op", True), ("f1@op", True), ("fp_per_neg_frame@op", False), ("fp_per_neg_flight@op", False),
                ("fp_per_neg_indoor@op", False))
"""(chỉ số, cao hơn là tốt hơn). ``@op`` = ở ngưỡng vận hành của từng model."""


def _op_metrics(pr: Prepared, frames: list[str], op: float) -> dict[str, float]:
    m = _subset_metrics(pr.s, frames, None, [op])
    return {k.replace(f"@{op:g}", "@op"): v for k, v in m.items()}


def _op_threshold(pr: Prepared, thr: float | None, name: str) -> tuple[float, str]:
    if thr is not None:
        return thr, "tham số"
    if pr.f1_info is None:
        raise EvalInputError(f"model {name}: không có dev-val để chọn ngưỡng; truyền --thr-{name}")
    return round(pr.f1_info["thr"], 6), "F1 tối đa trên dev-val"


def compare(pred_a: Path, pred_b: Path, split: str = "test", labels_dir: Path | None = None,
            manifest_path: Path | None = None, split_json: Path | None = None, thr_a: float | None = None,
            thr_b: float | None = None, boot_n: int = BOOT_N, seed: int = BOOT_SEED, allow_missing_labels: bool = False,
            devval_labels_dir: Path | None = None) -> dict[str, Any]:
    """Hiệu B − A ghép cặp: cùng một bộ mẫu khối bootstrap cho cả hai model; CI 95 % của hiệu và ``p_b_better``.

    ``p_diff_gt_0`` = tỉ lệ mẫu có hiệu > 0; ``p_b_better`` = tỉ lệ mẫu B tốt hơn A theo chiều của chỉ số (báo nhầm thấp
    hơn là tốt hơn, nên với báo nhầm đó là tỉ lệ hiệu < 0)."""
    preps, ops = {}, {}
    for name, pred, thr in (("a", pred_a, thr_a), ("b", pred_b, thr_b)):
        preps[name] = prepare(pred, split, labels_dir, manifest_path, split_json, [thr] if thr is not None else [],
                              allow_missing_labels, thr is None, devval_labels_dir)
        ops[name] = _op_threshold(preps[name], thr, name)
    pa, pb = preps["a"], preps["b"]
    if pa.all_frames != pb.all_frames or pa.s.block_of != pb.s.block_of:
        raise EvalInputError("hai model không chấm trên cùng tập khung/khối — không ghép cặp được")
    point = {n: _op_metrics(p, p.all_frames, ops[n][0]) for n, p in preps.items()}
    _check_against_pycocotools(_op_metrics(pa, pa.all_frames, ops["a"][0]), pa.stats)
    _check_against_pycocotools(_op_metrics(pb, pb.all_frames, ops["b"][0]), pb.stats)
    diffs: dict[str, list[float]] = {k: [] for k, _ in COMPARE_KEYS}
    for frames in resample(pa.s, boot_n, seed):
        ma, mb = _op_metrics(pa, frames, ops["a"][0]), _op_metrics(pb, frames, ops["b"][0])
        for k, _ in COMPARE_KEYS:
            diffs[k].append(mb[k] - ma[k])
    metrics = {}
    for k, higher in COMPARE_KEYS:
        d = np.asarray(diffs[k], dtype=float)
        d = d[np.isfinite(d)]
        gt0 = float(np.mean(d > 0)) if d.size else float("nan")
        better = float(np.mean(d > 0 if higher else d < 0)) if d.size else float("nan")
        metrics[k] = {"a": _nan_none(point["a"][k]), "b": _nan_none(point["b"][k]),
                      "diff": _nan_none(point["b"][k] - point["a"][k]),
                      "ci95": [_nan_none(x) for x in _ci(list(d))], "p_diff_gt_0": _nan_none(gt0),
                      "p_b_better": _nan_none(better), "higher_is_better": higher, "n_valid_samples": int(d.size)}
    return {
        "pred_a": str(pred_a), "pred_b": str(pred_b), "split": split, "official": pa.official and pb.official,
        "labels_sha256": pa.labels_sha,
        "op_threshold": {n: {"thr": ops[n][0], "source": ops[n][1]} for n in ops},
        "n": {"frames": len(pa.frames), "frames_with_person": sum(len(pa.gt[f]) > 0 for f in pa.frames),
              "gt_boxes": int(sum(len(pa.gt[f]) for f in pa.frames)),
              "negative_frames_flight": len(pa.s.neg_frames) - len(pa.neg_split),
              "negative_frames_indoor": len(pa.neg_split)},
        "metrics": metrics,
        "bootstrap": _boot_info(pa.s, boot_n, seed) | {"paired": True,
                                                       "note": "hiệu = B − A trên cùng mẫu khối cho cả hai model"},
    }


# ---------------------------------------------------------------- tiện ích + CLI


def _nan_none(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


def _check_against_pycocotools(m: dict[str, float], stats: np.ndarray) -> None:
    """Đối chiếu cộng dồn tự viết với ``COCOeval.summarize`` (stats[1]=AP50, [0]=AP, [3]=AP-small, [9]=AR-small)."""
    for key, idx in (("mAP50", 1), ("mAP50_95", 0), ("AP_small", 3), ("AR_small", 9)):
        ref = float(stats[idx])
        mine = m[key]
        if ref < 0 and not np.isfinite(mine):
            continue
        if not np.isclose(mine, ref, atol=1e-9):
            raise AssertionError(f"{key}: cộng dồn tự viết {mine} ≠ pycocotools {ref}")


def _print_text(r: dict[str, Any]) -> None:
    def fmt(d: dict[str, Any]) -> str:
        lo, hi = d["ci95"]
        v = d["value"]
        return "n/a" if v is None else (f"{v:.3f}" + (f" [{lo:.3f}–{hi:.3f}]" if lo is not None else ""))

    print(f"{r['pred']} — tập {r['split']}: {r['n']['frames']} khung, {r['n']['frames_with_person']} có người, "
          f"{r['n']['gt_boxes']} hộp; khung âm {r['n']['negative_frames']}")
    for k, d in r["metrics"].items():
        print(f"  {k:<26} {fmt(d)}")
    if r["ms_per_frame"]:
        print(f"  ms/khung trung bình {r['ms_per_frame']['mean']:.1f} ({r['ms_per_frame']['n_frames']} khung)")
    if r["note"]:
        print(f"  CHÚ Ý: {r['note']}")


def _print_compare(r: dict[str, Any]) -> None:
    print(f"B − A: B={r['pred_b']} A={r['pred_a']} (ngưỡng A {r['op_threshold']['a']['thr']:g}, "
          f"B {r['op_threshold']['b']['thr']:g})")
    for k, d in r["metrics"].items():
        lo, hi = d["ci95"]
        ci = f"[{lo:+.3f} … {hi:+.3f}]" if lo is not None else "[n/a]"
        diff = "n/a" if d["diff"] is None else f"{d['diff']:+.3f}"
        print(f"  {k:<22} A {d['a']}  B {d['b']}  hiệu {diff} {ci}  P(B tốt hơn) {d['p_b_better']}")


def _common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--split", choices=EVAL_SPLITS, default="test")
    ap.add_argument("--labels-dir", type=Path, default=None, help=f"mặc định {LABELS_REL}")
    ap.add_argument("--manifest", type=Path, default=None, help="mặc định <labels-dir>/manifest.json")
    ap.add_argument("--devval-labels-dir", type=Path, default=None, help="mặc định = --labels-dir")
    ap.add_argument("--split-json", type=Path, default=None, help=f"mặc định {SPLIT_REL}")
    ap.add_argument("--boot", type=int, default=BOOT_N, help="số lần bootstrap (0 = bỏ CI)")
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--allow-missing-labels", action="store_true", help="chấm trên khung có nhãn (kết quả không chính thức)")
    ap.add_argument("--out", type=Path, default=None, help="ghi JSON ra file")
    ap.add_argument("--json", action="store_true")


def _emit(r: dict[str, Any], a: argparse.Namespace, printer: Any) -> None:
    text = json.dumps(r, ensure_ascii=False, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    if a.json:
        print(text)
    else:
        printer(r)


def main_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.eval.detection compare",
                                 description="so sánh ghép cặp hai CSV dự đoán (hiệu B − A, CI bootstrap khối)")
    ap.add_argument("--pred-a", type=Path, required=True)
    ap.add_argument("--pred-b", type=Path, required=True)
    ap.add_argument("--thr-a", type=float, default=None, help="ngưỡng vận hành A (mặc định: F1 tối đa trên dev-val)")
    ap.add_argument("--thr-b", type=float, default=None, help="ngưỡng vận hành B (mặc định: F1 tối đa trên dev-val)")
    _common_args(ap)
    a = ap.parse_args(argv)
    try:
        r = compare(a.pred_a, a.pred_b, a.split, a.labels_dir, a.manifest, a.split_json, a.thr_a, a.thr_b, a.boot,
                    a.seed, a.allow_missing_labels, a.devval_labels_dir)
    except (EvalInputError, TestLeakError, FileNotFoundError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    _emit(r, a, _print_compare)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compare"]:
        return main_compare(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m xla.eval.detection", description=__doc__.split("\n")[0])
    ap.add_argument("--pred", type=Path, required=True, help="CSV frame,x1,y1,x2,y2,conf[,ms]")
    ap.add_argument("--thr", type=float, action="append", default=None, help="ngưỡng vận hành (lặp được), mặc định 0.25")
    ap.add_argument("--no-f1-devval", action="store_true", help="không chọn ngưỡng F1 trên dev-val")
    _common_args(ap)
    a = ap.parse_args(argv)
    try:
        r = evaluate(a.pred, a.split, a.labels_dir, a.manifest, a.split_json, a.thr, a.boot, a.seed,
                     a.allow_missing_labels, not a.no_f1_devval, a.devval_labels_dir)
    except (EvalInputError, TestLeakError, FileNotFoundError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    _emit(r, a, _print_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
