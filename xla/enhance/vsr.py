"""Mạng video BasicVSR++ cho ảnh rõ offline (phase 12 AI v2, tuỳ chọn, có tiêu chí dừng).

Lệnh (AX-01)::

    python -m xla.enhance.vsr probe  --config configs/vsr_x2.yaml --json
    python -m xla.enhance.vsr train  --config configs/vsr_x2.yaml --json
    python -m xla.enhance.vsr export --config configs/vsr_x2.yaml --json

``probe`` (plan bước 1): nạp trọng số REDS4 công bố (kiểm sha256, strict) vào kiến trúc vendor (deformable conv qua
``torchvision.ops.deform_conv2d``, không mmcv); đo PyTorch fp16 với ``temporal_window`` T khung 640×480 (s/khung, VRAM
đỉnh); xuất graph §11 biến thể video (``frames`` uint8 [T,480,640,3] → ``frame_out`` [960,1280,3] của khung giữa) ở
opset 19 (``DeformConv``); tạo phiên ORT CPU và CUDA rồi chạy thử. Dừng (``killed: onnx|vram|speed``) theo
``probe.kill``. Luôn ghi ``results/v2/vsr/status.json`` (kết quả hoặc lý do dừng kèm số đo), không có kết quả im lặng.

``train`` / ``export`` chỉ chạy khi ``status.json`` không ghi ``killed``. Mã thoát: 0 = probe đạt / lệnh xong,
1 = probe dừng (killed) hoặc lệnh bị chặn vì đã dừng, 2 = sai cách gọi.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from xla import contract
from xla.enhance import sr

KILL_REASONS = ("onnx", "vram", "speed")


def build_model(cfg: dict[str, Any]) -> nn.Module:
    from xla.enhance.arch.basicvsrpp import BasicVSRPlusPlus, load_reds4

    mc = cfg["model"]
    path = Path(cfg["weights_dir"]) / mc["weights"]
    if not path.is_file():
        raise FileNotFoundError(f"thiếu trọng số {path} (tải {mc['url']})")
    if sr.sha256_of(path) != mc["sha256"]:
        raise ValueError(f"sha256 của {path.name} khác config")
    return load_reds4(BasicVSRPlusPlus(**mc["kwargs"]), str(path), strict=True).eval()


class VSRGraph(nn.Module):
    """Graph §11 video: ``frames`` uint8 [T,480,640,3] BGR → ``frame_out`` uint8 [960,1280,3] BGR của khung giữa.

    Cắt cột ``[crop_left, 640)`` → RGB /255 → BasicVSR++ (×4) → khung giữa → avg-pool 2 (về ×2) → kẹp, làm tròn → BGR →
    cột ``[0, 2·crop_left)`` lặp cột ``2·crop_left``.
    """

    def __init__(self, net: nn.Module, crop_left: int, native_scale: int = 4, half: bool = False) -> None:
        super().__init__()
        self.net = net
        self.crop_left = int(crop_left)
        self.pool = native_scale // sr.SCALE
        self.use_half = half
        self.register_buffer("perm", torch.tensor([2, 1, 0], dtype=torch.long), persistent=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        t = frames.shape[0]
        x = frames[:, :, self.crop_left :, :].index_select(3, self.perm).permute(0, 3, 1, 2).unsqueeze(0).float() / 255
        if self.use_half:
            x = x.half()
        y = self.net(x)[:, t // 2].float()
        if self.pool > 1:
            y = F.avg_pool2d(y, self.pool)
        y = torch.round(y.clamp(0.0, 1.0) * 255.0)[0].permute(1, 2, 0).index_select(2, self.perm).to(torch.uint8)
        strip = y[:, :1, :].expand(-1, sr.SCALE * self.crop_left, -1)
        return torch.cat([strip, y], dim=1)


def verdict(m: dict[str, Any], kill: dict[str, Any]) -> str | None:
    """Lý do dừng theo plan bước 1 (thứ tự: onnx, vram, speed) hoặc ``None`` khi probe đạt.

    PyTorch tràn VRAM ngay ở bước đo (trước cả xuất ONNX) → ``vram``: xuất lỗi lúc đó là hệ quả, không phải lỗi op.
    """
    if "out of memory" in str(m.get("torch_fp16_error", "")).lower():
        return "vram"
    if not (m.get("onnx_export_ok") and m.get("ort_cpu_ok") and m.get("ort_cuda_ok")):
        return "onnx"
    vram = m.get("ort_cuda_peak_vram_gb", m.get("torch_fp16_peak_vram_gb"))
    if vram is None or vram > float(kill["max_vram_gb"]):
        return "vram"
    s = m.get("ort_cuda_s_per_frame", m.get("torch_fp16_s_per_frame"))
    if s is None or s > float(kill["max_s_per_frame"]):
        return "speed"
    return None


def no_grad_call(g: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return g(x)


def register_deform_conv_onnx(opset: int = 19) -> bool:
    """Ánh xạ ``torchvision::deform_conv2d`` → ONNX ``DeformConv`` (opset ≥ 19) cho exporter TorchScript.

    torchvision 0.21 không đăng ký symbolic này (xuất báo ``UnsupportedOperatorError``). Bố cục offset của torchvision
    (cặp (dy, dx) theo vị trí kernel, nhóm offset) trùng đặc tả ``DeformConv``. Trả ``True`` khi đăng ký được.
    """
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx import symbolic_helper as sh

    def _sym(g: Any, inp: Any, weight: Any, offset: Any, mask: Any, bias: Any, stride_h: Any, stride_w: Any,
             pad_h: Any, pad_w: Any, dil_h: Any, dil_w: Any, n_weight_grps: Any, n_offset_grps: Any,
             use_mask: Any) -> Any:
        i = [sh._get_const(v, "i", "int") for v in (stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps,
                                                     n_offset_grps)]
        kh, kw = sh._get_tensor_sizes(weight)[2:]
        args = [inp, weight, offset, bias] + ([mask] if sh._get_const(use_mask, "b", "use_mask") else [])
        return g.op("DeformConv", *args, strides_i=[i[0], i[1]], pads_i=[i[2], i[3], i[2], i[3]],
                    dilations_i=[i[4], i[5]], group_i=i[6], offset_group_i=i[7], kernel_shape_i=[kh, kw])

    register_custom_op_symbolic("torchvision::deform_conv2d", _sym, opset)
    return True


def _try(fn: Any) -> tuple[bool, Any, str | None]:
    try:
        return True, fn(), None
    except Exception as e:  # noqa: BLE001 - probe ghi đúng lỗi của ORT/exporter làm số đo, không văng
        return False, None, f"{type(e).__name__}: {str(e)[:400]}"


def probe(cfg: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort

    from xla.detect.selftrain import machine_info
    from xla.gpu_lock import gpu_lock

    pc = cfg["probe"]
    t = int(pc["temporal_window"])
    crop_left, _ = sr.camera_crop_left()
    names = sr.dev_frames(int(pc["n_frames"]))
    fr = [contract.decode_frame(sr.frames_dir() / n) for n in names]
    clip = np.ascontiguousarray(np.stack([fr[min(i, len(fr) - 1)] for i in range(t)]))
    m: dict[str, Any] = {"temporal_window": t, "frame_size": pc["frame_size"], "opset": int(pc["opset"]),
                         "onnxruntime": ort.__version__, "torch": torch.__version__,
                         "contract_opset": contract.OPSET}
    net = build_model(cfg)
    m["weights_sha256_ok"] = True
    m["params"] = int(sum(p.numel() for p in net.parameters()))
    onnx_path = sr.runs_dir(cfg) / "probe" / f"basicvsrpp_t{t}_opset{pc['opset']}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    t_wait = time.time()
    with gpu_lock("v2 phase12 vsr probe", timeout_s=sr.GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        m["machine_before"] = machine_info()
        g = VSRGraph(net, crop_left, int(cfg["model"]["native_scale"]), half=True).half().cuda().eval()
        x = torch.from_numpy(clip).cuda()
        ok, _, err = _try(lambda gg=g: no_grad_call(gg, x))
        if not ok:
            m["torch_fp16_error"] = err
        else:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            with torch.no_grad():
                for i in range(int(pc["warmup"]) + int(pc["iters"])):
                    t1 = time.perf_counter()
                    out = g(x).cpu()
                    torch.cuda.synchronize()
                    if i >= int(pc["warmup"]):
                        ts.append(time.perf_counter() - t1)
            m["torch_fp16_s_per_frame"] = float(np.mean(ts))  # một khung ra = một lần chạy cửa sổ T khung
            m["torch_fp16_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            m["torch_out_shape"] = list(out.shape)

        def _export(gg: nn.Module = g) -> None:
            with torch.no_grad():
                torch.onnx.export(gg, (x,), str(onnx_path), opset_version=int(pc["opset"]),
                                  input_names=[contract.FRAMES_NAME], output_names=["frame_out"], dynamo=False,
                                  do_constant_folding=True)

        m["deformconv_symbolic"] = "đăng ký riêng (xla.enhance.vsr.register_deform_conv_onnx; torchvision không có)"
        register_deform_conv_onnx(int(pc["opset"]))
        m["onnx_export_ok"], _, m["onnx_export_error"] = _try(_export)
        del g
        torch.cuda.empty_cache()
        if m["onnx_export_ok"]:
            import onnx

            mo = onnx.load(str(onnx_path), load_external_data=False)
            m["onnx_ops"] = sorted({n.op_type for n in mo.graph.node})
            m["onnx_has_deformconv"] = "DeformConv" in m["onnx_ops"]
            for prov, key in ((["CPUExecutionProvider"], "ort_cpu"), (["CUDAExecutionProvider"], "ort_cuda")):
                def _run(p: list[str] = prov) -> Any:
                    so = ort.SessionOptions()
                    so.log_severity_level = 3
                    so.intra_op_num_threads = sr.ORT_CPU_THREADS
                    s = ort.InferenceSession(str(onnx_path), sess_options=so, providers=p)
                    t1 = time.perf_counter()
                    y = s.run(None, {contract.FRAMES_NAME: clip})[0]
                    return {"providers": s.get_providers(), "s": time.perf_counter() - t1, "shape": list(y.shape)}

                with sr.DeviceVramPeak() as vp:
                    m[f"{key}_ok"], res, m[f"{key}_error"] = _try(_run)
                if res:
                    m[f"{key}_run"] = res
                    if key == "ort_cuda":
                        m["ort_cuda_s_per_frame"] = res["s"]
                        m["ort_cuda_peak_vram_gb"] = round(vp.delta_mb / 1024, 3)
                        m["ort_cuda_ok"] = res["providers"][0] == "CUDAExecutionProvider"
        else:
            m["ort_cpu_ok"] = m["ort_cuda_ok"] = False
            m["ort_cpu_error"] = m["ort_cuda_error"] = "không có file ONNX (xuất lỗi)"
        m["machine_after"] = machine_info()
        gpu_s = time.time() - t0
    sr.record_gpu(cfg, "probe", gpu_s, f"T={t}")
    reason = verdict(m, pc["kill"])
    return {"phase": "12", "step": "probe", "created": sr.now(), "killed": reason,
            "kill_criteria": pc["kill"], "measurements": m, "gpu_seconds": round(gpu_s, 1),
            "wait_lock_seconds": round(t0 - t_wait, 1), "gpu_hours_used_total": round(sr.gpu_hours_used(cfg), 3),
            "onnx": onnx_path.as_posix() if m.get("onnx_export_ok") else None}


def status_path(cfg: dict[str, Any]) -> Path:
    return sr.results_dir(cfg) / "status.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.vsr", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "train", "export"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/vsr_x2.yaml")
        p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = sr.load_config(args.config)
    if args.cmd == "probe":
        if sr.budget_left_s(cfg) <= 0:
            print(f"trần GPU {cfg['gpu_budget_hours']} h của phase 12 đã hết", file=sys.stderr)
            return 1
        try:
            rep = probe(cfg)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"vsr probe lỗi: {e}", file=sys.stderr)
            return 1
        prev = json.loads(status_path(cfg).read_text(encoding="utf-8")) if status_path(cfg).is_file() else {}
        sr.write_json(status_path(cfg), {**prev, **rep})
        print(json.dumps(rep, ensure_ascii=False) if args.json else json.dumps(rep, indent=1, ensure_ascii=False))
        return 0 if rep["killed"] is None else 1
    st = json.loads(status_path(cfg).read_text(encoding="utf-8")) if status_path(cfg).is_file() else None
    if st is None or st.get("killed"):
        why = "chưa probe" if st is None else f"probe đã dừng: killed {st['killed']}"
        print(json.dumps({"cmd": args.cmd, "blocked": why}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(f"{args.cmd}: probe đạt nhưng bước này chưa làm trong lane (plan phase 12 bước 2–3)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
