"""``meta.metrics`` của bundle enhancer trên tập **test** thật (hợp đồng §4: ``pink_frac_drop``, ``dark_channel_drop``,
``flicker_ratio``), ORT CPU, không GPU.

AX::

    python -m xla.enhance.split_metrics --bundle models/release/enhancer-v2.0.0 --json [--write-meta]
    # exit 0 đo xong; 1 bundle / dữ liệu sai; 2 sai cách gọi

Cùng định nghĩa với ``metrics_dev`` của ``enhancer-v1.0.0`` và ``enhancer-offline-v1.0.0`` (``train_net.select``), không
định nghĩa lại:

- ``pink_frac_drop``, ``dark_channel_drop`` = ``train_net.score_of(frame_stats(gốc), frame_stats(ra))`` — trung bình theo
  khung của ``xla.camera.measure.measure_image`` (``pink_frac``, ``dark_channel_15px``), trên **mọi** 198 khung ``test`` của
  ``split.json`` (test đã thưa ≥ 1 s/khung nên không lấy mỗi khung thứ 3 như dev).
- ``flicker_ratio`` = ``xla.eval.temporal.flicker_ratio`` gộp mọi chuỗi liền kề (< 250 ms). Khung ``test`` cách nhau ≥ 1 s
  nên không có chuỗi nào; chuỗi lấy từ **mọi khung bay trong các khối 30 s test** (khối dựng lại đúng như
  ``xla.camera.split.build``: ``(flight_id, ⌊(t − t0)/30⌋)``). Các khung đó không thuộc dev, không dùng chỉnh tham số.

Ghi thêm (khoá ngoài hợp đồng ``metrics_test``): số khung, thống kê gốc/ra, và hai chỉ số theo định nghĩa
``export_classic.dev_quality`` (trung vị dark channel, % điểm hồng của ``classic``) để so được với ``metrics_dev`` của
bundle classic. ``onnx`` không đổi, sha256 kiểm trước khi đo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla import contract
from xla.paths import repo_root

SPLIT_REL = Path("data/real/split.json")
FRAMES_REL = Path("data/real/frames")
CPU_THREADS = 4  # luật máy: job CPU tối đa 4 worker
TEST_BLOCK_SPLIT = "test_block"


class MetricsError(Exception):
    """Bundle / dữ liệu sai → exit 1."""


def test_frame_list(split_doc: dict[str, Any]) -> list[str]:
    """Khung ``test`` của ``split.json``, theo thời gian."""
    rows = sorted((f for f in split_doc["frames"] if f["split"] == "test"), key=lambda f: float(f["t_unix"]))
    return [f["frame"] for f in rows]


def block_rows(split_doc: dict[str, Any], manifest: list[dict[str, str]], flight_t0: dict[str, float],
                    block_s: float) -> list[dict[str, Any]]:
    """Mọi khung bay của manifest nằm trong một khối test (khoá ``(flight_id, ⌊(t − t0)/block_s⌋)`` như
    ``xla.camera.split.build``), dạng ``{"frame", "t_unix", "split": TEST_BLOCK_SPLIT}`` cho ``consecutive_runs``."""
    fly = {r["frame"]: r for r in manifest if r["kind"] == "flight"}

    def key(r: dict[str, str]) -> tuple[str, int]:
        return r["flight_id"], int((float(r["t_unix"]) - flight_t0[r["flight_id"]]) // block_s)

    names = test_frame_list(split_doc)
    missing = [n for n in names if n not in fly]
    if missing:
        raise MetricsError(f"{len(missing)} khung test không có trong manifest (vd. {missing[0]})")
    blocks = {key(fly[n]) for n in names}
    return [{"frame": r["frame"], "t_unix": float(r["t_unix"]), "split": TEST_BLOCK_SPLIT}
            for r in fly.values() if key(r) in blocks]


def measure(fn: Callable[[np.ndarray], np.ndarray], read: Callable[[str], np.ndarray], stat_names: list[str],
            runs: list[list[str]], crop_left: int) -> dict[str, Any]:
    """Ba chỉ số hợp đồng + chi tiết cho hàm ``khung → khung``; ``read(tên) → khung BGR``."""
    from xla.enhance.export_classic import dev_quality
    from xla.enhance.train_net import frame_stats, score_of
    from xla.eval.temporal import mean_abs_diff

    if not stat_names or not runs:
        raise MetricsError(f"thiếu khung: {len(stat_names)} khung test, {len(runs)} chuỗi liền kề")
    raw = [read(n) for n in stat_names]
    out = [fn(f) for f in raw]
    raw_st, out_st = frame_stats(raw, crop_left), frame_stats(out, crop_left)
    sc = score_of(raw_st, out_st)
    num = den = 0.0  # gộp như flicker_ratio nhiều chuỗi, từng chuỗi một (không giữ cả nghìn khung)
    n_run_frames = 0
    for run in runs:
        r = [read(n) for n in run]
        pairs = len(r) - 1
        den += mean_abs_diff(r, crop_left) * pairs
        num += mean_abs_diff([fn(f) for f in r], crop_left) * pairs
        n_run_frames += len(r)
    if den <= 0:
        raise MetricsError("ảnh gốc đứng yên tuyệt đối trên mọi chuỗi (mẫu số flicker 0)")
    flicker = num / den
    return {
        "metrics": {"pink_frac_drop": sc["pink_frac_drop"], "dark_channel_drop": sc["dark_channel_drop"],
                    "flicker_ratio": flicker},
        "detail": {"n_frames_stats": len(stat_names), "n_runs_flicker": len(runs), "n_frames_flicker": n_run_frames,
                   "raw_stats": raw_st, "out_stats": out_st, "blocking_excess_drop": sc["blocking_excess_drop"],
                   "classic_def": dev_quality(raw, out, crop_left)},
    }


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def open_bundle(bundle: Path) -> tuple[dict[str, Any], Path]:
    meta_p, onnx_p = bundle / contract.META_FILE, bundle / "enhancer.onnx"
    if not meta_p.is_file() or not onnx_p.is_file():
        raise MetricsError(f"bundle {bundle} thiếu {contract.META_FILE} hoặc enhancer.onnx")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("model") != "enhancer":
        raise MetricsError(f"{bundle}: meta.model = {meta.get('model')!r}, cần 'enhancer'")
    if contract.sha256_file(onnx_p) != meta.get("onnx_sha256"):
        raise MetricsError(f"{onnx_p}: sha256 khác meta.onnx_sha256")
    return meta, onnx_p


def run(bundle: Path) -> dict[str, Any]:
    import cv2
    import onnxruntime as ort

    from xla.camera.frames import EVENTS_REL, parse_flight_events, read_manifest
    from xla.eval.temporal import consecutive_runs

    cv2.setNumThreads(CPU_THREADS)
    root = repo_root()
    meta, onnx_p = open_bundle(bundle)
    crop_left = int(meta["crop_left"])
    split_doc = json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))
    t0 = {f.flight_id: f.t0 for f in parse_flight_events((root / EVENTS_REL).read_text(encoding="utf-8"))}
    rows = block_rows(split_doc, read_manifest(), t0, float(split_doc["block_s"]))
    runs = consecutive_runs(rows, TEST_BLOCK_SPLIT)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = CPU_THREADS
    sess = ort.InferenceSession(str(onnx_p), sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    fdir = root / FRAMES_REL
    res = measure(lambda f: sess.run(None, {name: f})[0], lambda n: contract.decode_frame(fdir / n),
                  test_frame_list(split_doc), runs, crop_left)
    res["detail"].update(n_test_block_frames=len(rows), runtime=f"onnxruntime CPU EP {ort.__version__}, "
                         f"{CPU_THREADS} luồng", measured=datetime.now(UTC).isoformat(timespec="seconds"),
                         command=f"python -m xla.enhance.split_metrics --bundle {_rel(bundle)} --write-meta")
    return res


def write_meta(bundle: Path, res: dict[str, Any]) -> list[str]:
    """Ghi ``metrics`` + ``metrics_test`` vào ``meta.json`` (giữ nguyên mọi khoá khác), rồi ``validate_bundle``."""
    p = bundle / contract.META_FILE
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["metrics"] = res["metrics"]
    meta["metrics_test"] = {"split": "test", **res["detail"]}
    meta["metrics_note"] = ("metrics = tập test thật (198 khung test; flicker trên chuỗi liền kề của các khối test), "
                            "cùng định nghĩa metrics_dev của enhancer-v1.0.0 (train_net.score_of); "
                            "đo bằng xla.enhance.split_metrics")
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return contract.validate_bundle(bundle)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.split_metrics", description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True, help="thư mục enhancer-vX.Y.Z")
    ap.add_argument("--write-meta", action="store_true", help="ghi metrics vào meta.json của bundle")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    bundle = Path(args.bundle)
    bundle = bundle if bundle.is_absolute() else repo_root() / bundle
    try:
        res = run(bundle)
        if args.write_meta:
            res["validate_errors"] = write_meta(bundle, res)
    except MetricsError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res, ensure_ascii=False, indent=None if args.json else 2))
    return 1 if res.get("validate_errors") else 0


if __name__ == "__main__":
    sys.exit(main())
