"""Huấn luyện mạng khôi phục trên cặp D2 → D0 và chọn checkpoint trên khung dev thật (phase 13, Deferred 11).

Lệnh::

    python -m xla.enhance.train_net train --config configs/restore.yaml [--smoke] [--run-id ID] [--max-steps N] --json
    python -m xla.enhance.train_net select --config configs/restore.yaml --run-id ID --json

``train``: patch ``patch``×``patch`` cùng cửa sổ từ D2 (vào) / D0 (đích), bỏ cột ``[0, crop_left)``, gốc patch trên lưới
8 px; loss L1 + w·(1 − SSIM), không GAN; khởi từ trọng số công bố; AMP bf16; ``last.pt`` mỗi ``last_every`` bước (resume
tự động nếu có), ``step_XXXXXX.pt`` mỗi ``ckpt_every`` bước; val PSNR/SSIM trên cặp D2 val (**số tổng hợp**). Giữ
``xla.gpu_lock`` suốt lần chạy. ``--smoke`` = ``smoke_steps`` bước, ghi giờ/1 000 bước + VRAM đỉnh.

``select``: mỗi checkpoint chạy trên các chuỗi khung dev liền kề (``xla.eval.temporal.consecutive_runs``), đo
``flicker_ratio`` và thống kê mức thấp (``xla.camera.measure.measure_image``: ``pink_frac``, ``dark_channel_15px``,
``blocking_ratio_8x8``) so với khung gốc; loại checkpoint có flicker > ``flicker_max``; chọn điểm cao nhất.

Mã thoát: 0 = xong, 1 = lỗi khi chạy / không checkpoint nào qua, 2 = sai cách gọi.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from xla.enhance.net import (
    RUNS_DIR,
    EnhancerGraph,
    build,
    enhance_frames,
    load_pretrained,
    n_params,
    resolve,
    variant_init,
)
from xla.paths import data_root, repo_root

GPU_LOCK_TIMEOUT_S = 6 * 3600
TRAIN_JSON_REL = Path("results/restore/train.json")


# ---- cấu hình ------------------------------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else repo_root() / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def camera_crop_left() -> tuple[int, str]:
    from xla.contract import load_profile

    prof = load_profile(repo_root() / "configs" / "camera_profile.yaml")
    return int(prof["left_strip"]["crop_left_px"]), str(prof["profile_version"])


# ---- dữ liệu ---------------------------------------------------------------------------------------------------
def _read(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"không giải mã được {path}")
    return img


def pair_names(split: str, cfg: dict[str, Any]) -> list[str]:
    root = data_root()
    a = {p.name for p in (root / cfg["data"]["input_set"] / split / "images").glob("*.jpg")}
    b = {p.name for p in (root / cfg["data"]["target_set"] / split / "images").glob("*.jpg")}
    names = sorted(a & b)
    if not names:
        raise FileNotFoundError(f"không có cặp {split} trong {root}")
    return names


def to_tensor_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_bgr[..., ::-1])).permute(2, 0, 1).float() / 255.0


class PairPatches(Dataset):
    """Patch cùng cửa sổ (D2 vào, D0 đích), gốc trên lưới ``grid`` tính từ ``crop_left``, có lật ngẫu nhiên."""

    def __init__(self, names: list[str], cfg: dict[str, Any], crop_left: int, length: int) -> None:
        root = data_root()
        self.src = root / cfg["data"]["input_set"] / "train" / "images"
        self.dst = root / cfg["data"]["target_set"] / "train" / "images"
        self.names = names
        self.crop_left = crop_left
        self.patch = int(cfg["data"]["patch"])
        self.grid = int(cfg["data"]["grid"])
        self.flip = bool(cfg["data"]["flip"])
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random((torch.initial_seed() + i * 7919) % (2**32))
        n = self.names[rng.randrange(len(self.names))]
        a, b = _read(self.src / n), _read(self.dst / n)
        if a.shape != b.shape:
            raise ValueError(f"cặp lệch cỡ {n}: {a.shape} / {b.shape}")
        h, w = a.shape[:2]
        p, g, cl = self.patch, self.grid, self.crop_left
        x0 = cl + g * rng.randrange((w - cl - p) // g + 1)
        y0 = g * rng.randrange((h - p) // g + 1)
        a, b = a[y0 : y0 + p, x0 : x0 + p], b[y0 : y0 + p, x0 : x0 + p]
        if self.flip and rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        if self.flip and rng.random() < 0.5:
            a, b = a[::-1], b[::-1]
        return to_tensor_rgb(a), to_tensor_rgb(b)


# ---- loss / đo -------------------------------------------------------------------------------------------------
def _gauss(win: int = 11, sigma: float = 1.5, device: Any = None) -> torch.Tensor:
    g = torch.exp(-((torch.arange(win, dtype=torch.float32, device=device) - win // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    return (g @ g.t()).expand(3, 1, win, win).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SSIM trung bình (cửa sổ Gauss 11, σ 1,5, từng kênh, ảnh 0–1, không đệm)."""
    x, y = x.float(), y.float()
    k = _gauss(device=x.device)
    c1, c2 = 0.01**2, 0.03**2
    mx, my = F.conv2d(x, k, groups=3), F.conv2d(y, k, groups=3)
    sxx = F.conv2d(x * x, k, groups=3) - mx**2
    syy = F.conv2d(y * y, k, groups=3) - my**2
    sxy = F.conv2d(x * y, k, groups=3) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean()


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = float(F.mse_loss(x.float(), y.float()))
    return 10 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def synthetic_val(model: torch.nn.Module, cfg: dict[str, Any], crop_left: int, n: int | None) -> dict[str, Any]:
    """PSNR/SSIM trên khung đủ 480×(640−crop_left) của cặp val — SỐ TỔNG HỢP (ảnh D2 tự sinh), không dùng kết luận."""
    root = data_root()
    names = pair_names("val", cfg)
    if n is not None and n < len(names):
        names = names[:: max(1, len(names) // n)][:n]
    model.eval()
    ps, ss, ps0, ss0 = [], [], [], []
    for nm in names:
        a = to_tensor_rgb(_read(root / cfg["data"]["input_set"] / "val" / "images" / nm)[:, crop_left:])[None].cuda()
        b = to_tensor_rgb(_read(root / cfg["data"]["target_set"] / "val" / "images" / nm)[:, crop_left:])[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(a)
        y = torch.round(y.float().clamp(0, 1) * 255) / 255
        ps.append(psnr(y, b))
        ss.append(float(ssim(y, b)))
        ps0.append(psnr(a, b))
        ss0.append(float(ssim(a, b)))
    model.train()
    return {
        "kind": "số tổng hợp (cặp D2→D0 val, ảnh tự sinh)",
        "n": len(names),
        "psnr": float(np.mean(ps)),
        "ssim": float(np.mean(ss)),
        "psnr_input": float(np.mean(ps0)),
        "ssim_input": float(np.mean(ss0)),
    }


# ---- huấn luyện ------------------------------------------------------------------------------------------------
def _lr_at(step: int, cfg_t: dict[str, Any], total: int) -> float:
    if step < cfg_t["warmup_steps"]:
        return cfg_t["lr"] * (step + 1) / cfg_t["warmup_steps"]
    t = (step - cfg_t["warmup_steps"]) / max(1, total - cfg_t["warmup_steps"])
    return cfg_t["lr_min"] + 0.5 * (cfg_t["lr"] - cfg_t["lr_min"]) * (1 + math.cos(math.pi * min(t, 1.0)))


def _save(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(cfg: dict[str, Any], run_id: str, smoke: bool, max_steps: int | None) -> dict[str, Any]:
    from xla.gpu_lock import gpu_lock

    ct = cfg["train"]
    total = int(ct["smoke_steps"] if smoke else (max_steps or ct["max_steps"]))
    crop_left, profile_version = camera_crop_left()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    torch.manual_seed(ct["seed"])
    random.seed(ct["seed"])
    np.random.seed(ct["seed"] % (2**32))

    model = build(cfg["candidate"], cfg.get("arch_kwargs"))
    if cfg.get("init", "pretrained") == "pretrained":
        load_pretrained(model, cfg["candidate"], strict=not cfg.get("arch_kwargs"))
    step = 0
    gpu_s_prev = 0.0
    last = run_dir / "last.pt"
    ck: dict[str, Any] | None = None
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        step = int(ck["step"])
        gpu_s_prev = float(ck.get("gpu_seconds", 0.0))
        torch.manual_seed(ct["seed"] + step)
    names = pair_names("train", cfg)
    batch = int(ct["batch"])
    ds = PairPatches(names, cfg, crop_left, length=(total - step) * batch)
    dl = DataLoader(ds, batch_size=batch, num_workers=min(4, int(ct["workers"])), pin_memory=True, drop_last=True,
                    persistent_workers=False)
    meta = {"candidate": cfg["candidate"], "arch_kwargs": cfg.get("arch_kwargs") or {}, "crop_left": crop_left,
            "init": cfg.get("init", "pretrained"), "variant": cfg.get("variant"),
            "camera_profile_version": profile_version, "config_version": cfg["version"], "run_id": run_id}
    vals: list[dict[str, Any]] = []
    from xla.enhance.net import wait_cpu_perf

    cpu_gate = wait_cpu_perf()  # CPU hạ xung làm chậm dataloader → sai giờ/1 000 bước và phí giờ GPU
    t_wait = time.time()
    with gpu_lock(f"phase13 restore {'smoke' if smoke else 'train'} {run_id}", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        wait_s = t0 - t_wait
        model.cuda().train()
        opt = torch.optim.AdamW(
            model.parameters(), lr=ct["lr"], betas=tuple(ct["betas"]), weight_decay=ct["weight_decay"]
        )
        if ck is not None:
            opt.load_state_dict(ck["opt"])  # tham số đã ở CUDA: trạng thái Adam theo về đúng thiết bị
            del ck
        torch.cuda.reset_peak_memory_stats()
        if step == 0:
            vals.append({"step": 0, **synthetic_val(model, cfg, crop_left, ct["val_n"])})
        t_steps = time.time()
        steps_done = 0
        it = iter(dl)
        while step < total:
            a, b = next(it)
            a, b = a.cuda(non_blocking=True), b.cuda(non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, ct, total)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(a)
            y = y.float()
            l1 = F.l1_loss(y, b)
            ls = 1 - ssim(y.clamp(0, 1), b)
            loss = l1 + ct["ssim_weight"] * ls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ct["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), ct["grad_clip"])
            opt.step()
            step += 1
            steps_done += 1
            if not math.isfinite(float(loss)):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}")
            if step % 50 == 0 or step == total:
                rec = {"step": step, "loss": float(loss), "l1": float(l1), "1-ssim": float(ls),
                       "lr": opt.param_groups[0]["lr"], "s_per_step": (time.time() - t_steps) / steps_done,
                       "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            gpu_s = gpu_s_prev + time.time() - t0
            if step % ct["last_every"] == 0 or step == total:
                _save(last, {**meta, "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                             "gpu_seconds": gpu_s})
            if not smoke and (step % ct["ckpt_every"] == 0 or step == total):
                _save(run_dir / f"step_{step:06d}.pt", {**meta, "model": model.state_dict(), "step": step})
            if step % ct["val_every"] == 0 and step != total:
                v = {"step": step, **synthetic_val(model, cfg, crop_left, ct["val_n"])}
                vals.append(v)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"val": v}) + "\n")
        s_per_step = (time.time() - t_steps) / max(1, steps_done)
        final_val = {"step": step, **synthetic_val(model, cfg, crop_left, None if not smoke else ct["val_n"])}
        vals.append(final_val)
        vram = torch.cuda.max_memory_allocated() / 2**30
        gpu_s = gpu_s_prev + time.time() - t0
    out = {
        "run_id": run_id,
        "smoke": smoke,
        "candidate": cfg["candidate"],
        "params": n_params(model),
        "steps": step,
        "batch": batch,
        "lr": ct["lr"],
        "patch": cfg["data"]["patch"],
        "s_per_step": s_per_step,
        "hours_per_1000_steps": s_per_step * 1000 / 3600,
        "vram_peak_gb": vram,
        "gpu_seconds": gpu_s,
        "wait_lock_seconds": wait_s,
        "cpu_perf_gate": cpu_gate,
        "val_synthetic": vals,
        "run_dir": str(run_dir),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / ("smoke.json" if smoke else "train_summary.json")).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


# ---- chọn checkpoint trên dev thật -------------------------------------------------------------------------
SELECT_KEYS = ("pink_frac", "dark_channel_15px", "blocking_ratio_8x8")


def dev_runs(split: str) -> list[list[str]]:
    from xla.eval.temporal import consecutive_runs

    s = json.loads((repo_root() / "data" / "real" / "split.json").read_text(encoding="utf-8"))
    return consecutive_runs(s["frames"], split)


def frame_stats(frames: list[np.ndarray], crop_left: int) -> dict[str, float]:
    from xla.camera.measure import measure_image

    ms = [measure_image(f, crop_left) for f in frames]
    return {k: float(np.nanmean([m[k] for m in ms])) for k in SELECT_KEYS}


def score_of(raw: dict[str, float], out: dict[str, float]) -> dict[str, float]:
    pink = (raw["pink_frac"] - out["pink_frac"]) / max(raw["pink_frac"], 1e-9)
    dark = (raw["dark_channel_15px"] - out["dark_channel_15px"]) / max(raw["dark_channel_15px"], 1e-9)
    blk = (raw["blocking_ratio_8x8"] - out["blocking_ratio_8x8"]) / max(raw["blocking_ratio_8x8"] - 1.0, 1e-9)
    return {"pink_frac_drop": pink, "dark_channel_drop": dark, "blocking_excess_drop": blk, "score": pink + dark + blk}


SELECT_RULE = (
    "điểm = pink_frac_drop + dark_channel_drop + blocking_excess_drop (cao hơn tốt hơn). Có checkpoint huấn luyện qua "
    "flicker_ratio ≤ flicker_max → điểm cao nhất trong số đó. Không checkpoint nào qua → checkpoint huấn luyện có "
    "flicker_ratio thấp nhất (hoà thì điểm cao hơn), ghi rõ ngưỡng không đạt"
)


def choose(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Áp ``SELECT_RULE`` lên các dòng checkpoint (bỏ dòng ``pretrained``: không có file checkpoint)."""
    trained = [r for r in rows if r["path"]]
    ok = [r for r in trained if r["pass_flicker"]]
    if ok:
        return max(ok, key=lambda r: (r["score"], r["checkpoint"])), "qua flicker_max, điểm cao nhất"
    if trained:
        best = min(trained, key=lambda r: (r["flicker_ratio"], -r["score"]))
        top = max(trained, key=lambda r: r["score"])
        note = "cũng là điểm cao nhất" if top is best else f"điểm cao nhất là {top['checkpoint']}"
        return best, f"không checkpoint nào qua flicker_max: chọn flicker_ratio thấp nhất ({note})"
    return None, "không có checkpoint huấn luyện"


def rescore(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Áp lại ``SELECT_RULE`` lên kết quả ``select`` đã lưu (không chạy GPU lại)."""
    p = repo_root() / TRAIN_JSON_REL
    sel = json.loads(p.read_text(encoding="utf-8"))["select"]
    if sel.get("run_id") != run_id:
        raise ValueError(f"select đã lưu là của {sel.get('run_id')}, không phải {run_id}")
    best, how = choose(sel["checkpoints"])
    sel.update(rule=SELECT_RULE, chosen_how=how, chosen=best["checkpoint"] if best else None,
               chosen_path=best["path"] if best else None)
    return sel


def select(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    from xla.eval.temporal import flicker_ratio
    from xla.gpu_lock import gpu_lock

    cs = cfg["select"]
    crop_left, _ = camera_crop_left()
    run_dir = RUNS_DIR / run_id
    ckpts = sorted(run_dir.glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"không có checkpoint step_*.pt trong {run_dir}")
    runs = dev_runs(cs["split"])
    fdir = repo_root() / "data" / "real" / "frames"
    raw_runs = [[_read(fdir / n) for n in r] for r in runs]
    every = int(cs["stats_every"])
    raw_flat = [f for r in raw_runs for f in r]
    raw_stats = frame_stats(raw_flat[::every], crop_left)
    pre = {"name": "pretrained", "path": None}
    rows: list[dict[str, Any]] = []
    kept: list[tuple[dict[str, Any], float, list[np.ndarray]]] = []
    t_wait = time.time()
    with gpu_lock(f"phase13 restore select {run_id}", timeout_s=GPU_LOCK_TIMEOUT_S):
        t0 = time.time()
        # kiến trúc lấy từ chính checkpoint (biến thể thu nhỏ có arch_kwargs riêng), không từ config
        ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        cand, akw, init = ck0["candidate"], ck0.get("arch_kwargs") or {}, ck0.get("init", "pretrained")
        del ck0
        items = [*([pre] if init == "pretrained" else []), *({"name": p.stem, "path": p} for p in ckpts)]
        for item in items:
            model = build(cand, akw)
            if item["path"] is None:
                load_pretrained(model, cand, strict=not akw)
            else:
                model.load_state_dict(torch.load(item["path"], map_location="cpu", weights_only=False)["model"])
            graph = EnhancerGraph(model.eval(), crop_left, half=True).half().cuda().eval()
            out_runs = [enhance_frames(graph, r) for r in raw_runs]
            del graph, model
            torch.cuda.empty_cache()
            fl = flicker_ratio(raw_runs, out_runs, crop_left)
            kept.append((item, fl, [f for r in out_runs for f in r][::every]))
            del out_runs
        gpu_s = time.time() - t0
    for item, fl, sub_frames in kept:  # thống kê CPU sau khi nhả khoá GPU
        st = frame_stats(sub_frames, crop_left)
        sc = score_of(raw_stats, st)
        rows.append({"checkpoint": item["name"], "path": str(item["path"]) if item["path"] else None,
                     "flicker_ratio": fl, "stats": st, **sc, "pass_flicker": fl <= cs["flicker_max"]})
    best, how = choose(rows)
    return {
        "run_id": run_id,
        "split": cs["split"],
        "n_runs": len(runs),
        "n_frames_flicker": len(raw_flat),
        "n_frames_stats": len(raw_flat[::every]),
        "raw_stats": raw_stats,
        "flicker_max": cs["flicker_max"],
        "rule": SELECT_RULE,
        "chosen_how": how,
        "checkpoints": rows,
        "chosen": best["checkpoint"] if best else None,
        "chosen_path": best["path"] if best else None,
        "gpu_seconds": gpu_s,
        "wait_lock_seconds": t0 - t_wait,
    }


def _write_train_json(update: dict[str, Any], run_id: str | None = None) -> None:
    """Ghi khoá ``smoke``/``train``/``select`` (lần mới nhất) và bản sao dưới ``runs.<run_id>`` để giữ mọi vòng."""
    p = repo_root() / TRAIN_JSON_REL
    cur = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    cur.update(update)
    if run_id:
        cur.setdefault("runs", {}).setdefault(run_id, {}).update(update)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.enhance.train_net", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/restore.yaml")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--run-id", default=None)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--candidate", default=None, help="ghi đè candidate của config")
    t.add_argument("--batch", type=int, default=None, help="ghi đè train.batch")
    t.add_argument("--lr", type=float, default=None, help="ghi đè train.lr")
    t.add_argument("--ckpt-every", type=int, default=None, help="ghi đè train.ckpt_every và val_every")
    t.add_argument("--workers", type=int, default=None, help="ghi đè train.workers (trần 4)")
    t.add_argument("--json", action="store_true")
    s = sub.add_parser("select")
    s.add_argument("--config", default="configs/restore.yaml")
    s.add_argument("--run-id", required=True)
    s.add_argument("--rescore", action="store_true", help="áp lại luật chọn lên kết quả select đã lưu, không chạy GPU")
    s.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "train":
        if args.batch:
            cfg["train"]["batch"] = args.batch
        if args.lr:
            cfg["train"]["lr"] = args.lr
        if args.workers:
            cfg["train"]["workers"] = args.workers
        if args.ckpt_every:
            cfg["train"]["ckpt_every"] = cfg["train"]["val_every"] = args.ckpt_every
        if args.candidate:  # ứng viên hoặc biến thể thu nhỏ (xla.enhance.net.VARIANTS)
            base, kw = resolve(args.candidate)
            cfg["candidate"] = base
            cfg["init"] = variant_init(args.candidate)
            cfg["variant"] = args.candidate
            cfg["arch_kwargs"] = {k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()}
        run_id = args.run_id or f"{cfg['candidate']}-{'smoke' if args.smoke else 'train'}"
        out = train(cfg, run_id, args.smoke, args.max_steps)
        _write_train_json({"smoke" if args.smoke else "train": out}, run_id)
    else:
        out = rescore(cfg, args.run_id) if args.rescore else select(cfg, args.run_id)
        _write_train_json({"select": out}, args.run_id)
        if out["chosen"] is None:
            print(json.dumps(out, ensure_ascii=False) if args.json else "không checkpoint nào qua", file=sys.stderr)
            return 1
    print(json.dumps(out, ensure_ascii=False) if args.json else json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
