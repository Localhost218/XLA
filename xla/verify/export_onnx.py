"""Xuất ``verifier.onnx`` §9 + bundle ``verifier-v1.0.0`` (phase 07 bước 5–6).

CLI (AX-01, mã thoát 0 đạt / 1 parity, ``validate`` hoặc Kill (c) chặn / 2 sai cách gọi)::

    python -m xla.verify.export_onnx --config configs/verifier.yaml --json [--force]

Graph (một model, N động, N = 0 hợp lệ): nới hộp (tâm giữ, cạnh = max(w, h) · ``context_scale``, tối thiểu
``min_side_px``) → kẹp ``[crop_left, 640] × [0, 480]`` → ``RoiAlign`` (``output_half_pixel`` = ``roi_align(aligned=False)``,
``sampling_ratio`` 2) → BGR→RGB, /255, chuẩn hoá ImageNet → backbone → sigmoid. Parity ORT CPU + CUDA với PyTorch
eager cùng trọng số, ``|Δ| ≤ 0,02`` (§9). Kill (c) ở ``select.json`` → không xuất (mã 1).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla import contract as C
from xla.verify import common as V

VERSION = "1.0.0"
MS_NS = (1, 5, 20)


def graph_module(backbone: Any, crop_px: int, context_scale: float, min_side_px: float,
                 crop_left: int = C.CROP_LEFT_V1) -> Any:
    """``nn.Module`` (frame uint8 [480, 640, 3] BGR, boxes float32 [N, 4]) → scores float32 [N]."""
    import torch
    from torchvision.ops import roi_align

    from xla.verify.train import normalize_bgr

    class VerifierGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, frame: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
            x = frame.permute(2, 0, 1).unsqueeze(0).float()
            ctx = V.context_boxes(boxes, context_scale, min_side_px, float(crop_left), float(C.FRAME_W),
                                  float(C.FRAME_H))
            rois = torch.cat([torch.zeros_like(ctx[:, :1]), ctx], 1)
            crops = roi_align(x, rois, (crop_px, crop_px), spatial_scale=1.0,
                              sampling_ratio=C.VERIFIER_SAMPLING_RATIO, aligned=False)
            return torch.sigmoid(self.backbone(normalize_bgr(crops))).reshape(-1)

    return VerifierGraph().eval()


def export(module: Any, onnx_path: Path) -> None:
    import onnx
    import torch

    frame = torch.zeros(C.FRAME_SHAPE, dtype=torch.uint8)
    boxes = torch.tensor([list(C.VERIFIER_PROBE_BOX), [300.0, 200.0, 360.0, 330.0]], dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, (frame, boxes), str(onnx_path), input_names=[C.INPUT_NAME, C.BOXES_NAME],
                      output_names=["scores"], dynamic_axes={C.BOXES_NAME: {0: "N"}, "scores": {0: "N"}},
                      opset_version=C.OPSET, do_constant_folding=True, dynamo=False)
    m = onnx.load(str(onnx_path))
    modes = [a.s.decode() for n in m.graph.node if n.op_type == "RoiAlign" for a in n.attribute
             if a.name == "coordinate_transformation_mode"]
    if modes != [C.VERIFIER_ROI_MODE]:
        raise RuntimeError(f"RoiAlign trong graph có coordinate_transformation_mode {modes}, cần [{C.VERIFIER_ROI_MODE}]")
    onnx.checker.check_model(m)


def ort_session(onnx_path: Path, cuda: bool) -> Any:
    import onnxruntime as ort

    if cuda:
        import torch  # noqa: F401 — nạp DLL CUDA/cuDNN của torch trước session ORT CUDA
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    s = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
    s.run(None, {C.INPUT_NAME: np.zeros(C.FRAME_SHAPE, np.uint8),
                 C.BOXES_NAME: np.asarray([C.VERIFIER_PROBE_BOX], np.float32)})
    if cuda and s.get_providers()[0] != "CUDAExecutionProvider":  # đọc sau một lần chạy (luật chung)
        raise RuntimeError(f"ORT không chạy CUDA EP (có {s.get_providers()})")
    return s


def parity(module: Any, onnx_path: Path, feeds: dict[str, tuple[np.ndarray, np.ndarray]], cuda: bool) -> dict[str, Any]:
    """|Δ| lớn nhất giữa ORT (CPU, CUDA nếu ``cuda``) và PyTorch eager fp32 CPU trên ``feeds``."""
    import torch

    ref = {}
    with torch.no_grad():
        for k, (f, b) in feeds.items():
            ref[k] = module(torch.from_numpy(f), torch.from_numpy(b)).numpy()
    out: dict[str, Any] = {}
    for dev in ["cpu"] + (["cuda"] if cuda else []):
        s = ort_session(onnx_path, dev == "cuda")
        d = 0.0
        for k, (f, b) in feeds.items():
            got = s.run(None, {C.INPUT_NAME: f, C.BOXES_NAME: b})[0]
            if got.shape != ref[k].shape:
                raise RuntimeError(f"{k}: ORT {dev} shape {got.shape}, PyTorch {ref[k].shape}")
            d = max(d, float(np.abs(got - ref[k]).max()) if got.size else 0.0)
        out[dev] = {"max_abs_diff": round(d, 6), "pass": d <= C.VER_SCORE_ABS_MAX}
    out["n_boxes"] = int(sum(len(b) for _, b in feeds.values()))
    return out


def bench(onnx_path: Path, frame: np.ndarray, boxes_pool: np.ndarray, warmup: int = 30, n: int = 300) -> dict[str, Any]:
    s = ort_session(onnx_path, cuda=True)
    out = {}
    for k in MS_NS:
        b = np.ascontiguousarray(np.resize(boxes_pool, (k, 4)).astype(np.float32))
        feed = {C.INPUT_NAME: frame, C.BOXES_NAME: b}
        for _ in range(warmup):
            s.run(None, feed)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            s.run(None, feed)
            ts.append((time.perf_counter() - t0) * 1000)
        out[str(k)] = {"p50": round(float(np.percentile(ts, 50)), 3), "p95": round(float(np.percentile(ts, 95)), 3)}
    return out


def golden_feeds(cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[list[float]]]]:
    """3 khung golden: 2 khung devpool có người (hộp detector ở score_floor + GT), 1 khung mine với N = 0."""
    rs = V.real_sets()
    det = json.loads((V.results_dir(cfg) / "det_real.json").read_text(encoding="utf-8"))["boxes"]
    fold0 = [f for f in rs["devpool"] if rs["fold_of"][f] == 0 and rs["gt"].get(f) and det.get(f)]
    fold1 = [f for f in rs["devpool"] if rs["fold_of"][f] == 1 and rs["gt"].get(f) and det.get(f)]
    srcs = {"000.jpg": fold0[len(fold0) // 2], "001.jpg": fold1[len(fold1) // 2], "002.jpg": rs["mine"][0]}
    boxes = {
        "000.jpg": [b[:4] for b in det[srcs["000.jpg"]]] + rs["gt"][srcs["000.jpg"]][:3],
        "001.jpg": [b[:4] for b in det[srcs["001.jpg"]]] + rs["gt"][srcs["001.jpg"]][:3] + [list(C.VERIFIER_PROBE_BOX)],
        "002.jpg": [],
    }
    return srcs, {k: [[round(float(x), 2) for x in b] for b in v] for k, v in boxes.items()}


def cmd_export(cfg: dict[str, Any], force: bool) -> tuple[int, dict[str, Any]]:
    import onnxruntime as ort
    import torch

    from xla.detect.export_onnx import cpu_performance_pct
    from xla.detect.selftrain import machine_info
    from xla.gpu_lock import gpu_lock
    from xla.paths import repo_root
    from xla.verify import train as T

    sel = json.loads((V.results_dir(cfg) / "select.json").read_text(encoding="utf-8"))
    if sel["kill_c"] and not force:
        return 1, {"error": "Kill (c) kích hoạt ở select.json: không xuất verifier-v1.0.0", "select": sel["kill_c_rule"]}
    cand = sel["chosen"]
    c = cfg["train"]["candidates"][cand]
    r = sel["candidates"][cand]
    op = cfg["select"].get("operating", "d7")
    thr = (r["d7"] if op == "d7" and r["d7"] else r["at_recall_drop"])["thr"]
    at = r["d7"] if op == "d7" and r["d7"] else r["at_recall_drop"]
    bb = T.load_model(cfg, cand, f"{cand}_final")
    module = graph_module(bb, int(c["crop_px"]), float(cfg["crop"]["context_scale"]), float(cfg["crop"]["min_side_px"]))
    bundle = repo_root() / "models" / "release" / f"verifier-v{VERSION}"
    if bundle.exists():
        shutil.rmtree(bundle)
    gdir = bundle / "golden"
    gdir.mkdir(parents=True)
    onnx_path = bundle / "verifier.onnx"
    export(module, onnx_path)
    srcs, gboxes = golden_feeds(cfg)
    for k, f in srcs.items():
        shutil.copyfile(V.frame_path(f), gdir / k)
    (gdir / C.BOXES_FILE).write_text(json.dumps(gboxes, indent=1), encoding="utf-8")
    frames = {k: C.decode_frame(gdir / k) for k in srcs}
    feeds = {k: (frames[k], np.asarray(gboxes[k], np.float32).reshape(-1, 4)) for k in srcs}
    rng = np.random.default_rng(cfg["seed"])
    feeds["random_20"] = (frames["000.jpg"], np.asarray(
        [[x, y, x + w, y + h] for x, y, w, h in zip(rng.uniform(0, 600, 20), rng.uniform(0, 440, 20),
                                                    rng.uniform(4, 80, 20), rng.uniform(4, 80, 20), strict=True)],
        np.float32))
    perf = cpu_performance_pct()
    with gpu_lock("v2 phase07 verifier export parity+ms", timeout_s=21600):
        t0 = time.time()
        par = parity(module, onnx_path, feeds, cuda=True)
        ms = bench(onnx_path, frames["000.jpg"], feeds["random_20"][1])
        mach = machine_info()
        gpu_s = time.time() - t0
    from xla.verify.mine import _log_gpu

    _log_gpu(cfg, "export parity+ms", gpu_s)
    s_cpu = ort_session(onnx_path, cuda=False)
    expected = {k: {"scores": [float(x) for x in s_cpu.run(None, {C.INPUT_NAME: frames[k],
                                                                C.BOXES_NAME: feeds[k][1]})[0]]} for k in srcs}
    (gdir / C.EXPECTED_FILE).write_text(json.dumps(expected, indent=1), encoding="utf-8")
    meta = {
        "contract_version": C.CONTRACT_VERSION, "model": "verifier", "version": VERSION,
        "onnx_sha256": C.sha256_file(onnx_path), "frame_size": [C.FRAME_W, C.FRAME_H], "crop_left": C.CROP_LEFT_V1,
        "camera_profile_version": json.loads((Path(cfg["detector_bundle"]) / "meta.json").read_text("utf-8"))[
            "camera_profile_version"],
        "train_run": f"v2/verifier/{cand}_final",
        "metrics": {
            "fp_per_cal_frame_before": round(r["before"]["cal"]["fp_per_frame"], 5),
            "fp_per_cal_frame_after": round(at["cal"]["fp_per_frame"], 5),
            "recall_oof_before": round(r["before"]["oof"]["recall"], 5),
            "recall_oof_after": round(at["oof"]["recall"], 5),
        },
        "ms_per_frame": {"device": mach.get("gpu", {}).get("name", "cuda"), "runtime": f"onnxruntime-gpu {ort.__version__} CUDA EP",
                         "p50": ms["5"]["p50"], "p95": ms["5"]["p95"], "n_boxes": 5, "by_n": ms,
                         "cpu_perf_pct": perf, "machine": mach},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "crop_px": int(c["crop_px"]), "context_scale": float(cfg["crop"]["context_scale"]),
        "min_side_px": float(cfg["crop"]["min_side_px"]), "conf_threshold": float(thr), "input_variant": "raw",
        "backbone": c["backbone"],
        "detector": {"bundle": "detector-v1.0.0", "conf_threshold": T.det_conf(cfg)},
        "operating_point": op if (op != "d7" or r["d7"]) else "recall_drop",
        "golden_sources": [srcs[k] for k in sorted(srcs)],
        "export": {"torch": torch.__version__, "opset": C.OPSET, "roi_align": "torchvision aligned=False", "parity": par},
    }
    (bundle / C.META_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    errs = C.validate_bundle(bundle)
    res = {"bundle": str(bundle), "chosen": cand, "conf_threshold": thr, "parity": par, "ms": ms, "gpu_s": round(gpu_s, 1),
           "validate": errs}
    V.write_json(V.results_dir(cfg) / "export.json", {**res, "machine": mach, "cpu_perf_pct": perf})
    ok = not errs and par["cpu"]["pass"] and par["cuda"]["pass"]
    return (0 if ok else 1), res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.verify.export_onnx")
    ap.add_argument("--config", default=str(V.CONFIG_REL))
    ap.add_argument("--force", action="store_true", help="xuất dù Kill (c) (chỉ để thử, không ship)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = V.load_cfg(a.config)
    except OSError as e:
        V.emit({"error": str(e)}, a.json)
        return 2
    code, res = cmd_export(cfg, a.force)
    V.emit(res, a.json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
