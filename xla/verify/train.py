"""Huấn luyện verifier (phase 07 bước 2–4): vùng cắt, k-fold theo phiên, OOF, ngưỡng D7, Kill (c).

CLI (AX-01, mã thoát 0 đạt / 1 lỗi hoặc Kill (c) kích hoạt ở ``select`` / 2 sai cách gọi)::

    python -m xla.verify.train prep   --config configs/verifier.yaml --json   # CPU: vùng cắt rung + JPEG (D2) -> runs/
    python -m xla.verify.train run    --config configs/verifier.yaml --json [--only C ...]  # GPU (gpu_lock): fold + bản cuối
    python -m xla.verify.train select --config configs/verifier.yaml --json   # CPU: OOF, chọn, ngưỡng, Kill (c)

Fold: mẫu thật mang ``fold`` (devpool theo ``folds.json``, âm mine theo phiên); mẫu ngoài (d02x, TinyPerson) ``fold -1``
luôn vào train. Dừng sớm theo loss của 10 % ảnh ngoài giữ lại (không theo fold OOF: OOF chỉ để chấm và chọn).
Chấm: mọi hộp detector-v1 ≥ ``conf_threshold`` của nó trên devpool + mine (điểm OOF từ model fold kia) và trên cal
(model cuối, cal chưa từng vào manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from xla import contract as C
from xla.verify import common as V
from xla.verify import mine as M


# ---- vùng cắt -------------------------------------------------------------------------------------------------------
def _load_image(row: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    if row["src"] in ("mine", "devpool"):
        return C.decode_frame(V.frame_path(row["image"]))
    if row["src"] == "d02x":
        return C.decode_frame(Path(cfg["d02x_dir"]) / "train" / "images" / row["image"])
    return C.decode_frame(Path(row["image"]))


def _jitter(box: np.ndarray, rng: np.random.Generator, shift: float, scale: float) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    s = max(w, h)
    cx = (box[0] + box[2]) / 2 + rng.uniform(-shift, shift) * s
    cy = (box[1] + box[3]) / 2 + rng.uniform(-shift, shift) * s
    k = 1 + rng.uniform(-scale, scale)
    return np.asarray([cx - w * k / 2, cy - h * k / 2, cx + w * k / 2, cy + h * k / 2], np.float32)


def _prep_image(args: tuple[list[dict[str, Any]], dict[str, Any], int]) -> list[tuple[int, np.ndarray]]:
    import cv2

    rows, cfg, crop_px = args
    t, cr = cfg["train"], cfg["crop"]
    im = _load_image(rows[0], cfg)
    hh, ww = im.shape[:2]
    out = []
    for r in rows:
        rng = np.random.default_rng(int(hashlib.sha1(f"{cfg['seed']}|{r['id']}|{crop_px}".encode()).hexdigest()[:8], 16))
        base = np.asarray(r["box"], np.float32)
        boxes = np.stack([base] + [_jitter(base, rng, t["jitter_shift"], t["jitter_scale"])
                                   for _ in range(int(t["variants"]) - 1)])
        ctx = V.context_boxes(boxes, cr["context_scale"], cr["min_side_px"], r["x_lo"], ww, hh)
        crops = V.roi_crops(im, ctx, crop_px).round().clip(0, 255).astype(np.uint8)  # [V, 3, c, c] BGR
        for v in range(len(crops)):
            if rng.random() < t["jpeg_prob"]:
                q = int(rng.integers(t["jpeg_quality"][0], t["jpeg_quality"][1] + 1))
                hwc = np.ascontiguousarray(crops[v].transpose(1, 2, 0))
                ok, buf = cv2.imencode(".jpg", hwc, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    crops[v] = cv2.imdecode(buf, cv2.IMREAD_COLOR).transpose(2, 0, 1)
        out.append((r["_i"], crops))
    return out


def eval_boxes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Hộp detector ≥ ``conf_threshold`` detector trên devpool + mine + cal (kèm vai: ``oof`` theo fold / ``cal``)."""
    rs = V.real_sets()
    det = json.loads((V.results_dir(cfg) / M.DET_REAL).read_text(encoding="utf-8"))["boxes"]
    det.update(json.loads((V.runs_dir(cfg) / M.DET_CAL).read_text(encoding="utf-8"))["boxes"])
    thr = det_conf(cfg)
    out = []
    for f in rs["mine"] + rs["devpool"] + rs["cal"]:
        role = "cal" if f in set(rs["cal"]) else "oof"
        for b in det.get(f, []):
            if b[4] >= thr:
                out.append({"image": f, "box": b[:4], "score": b[4], "role": role,
                            "fold": rs["fold_of"].get(f, -1) if role == "oof" else -1})
    return out


def det_conf(cfg: dict[str, Any]) -> float:
    return float(json.loads((Path(cfg["detector_bundle"]) / "meta.json").read_text(encoding="utf-8"))["conf_threshold"])


def _prep_eval(args: tuple[str, list[list[float]], dict[str, Any], int]) -> np.ndarray:
    f, boxes, cfg, crop_px = args
    im = C.decode_frame(V.frame_path(f))
    cr = cfg["crop"]
    ctx = V.context_boxes(np.asarray(boxes, np.float32).reshape(-1, 4), cr["context_scale"], cr["min_side_px"])
    return V.roi_crops(im, ctx, crop_px).astype(np.float32)  # chính xác như graph (không làm tròn)


def cmd_prep(cfg: dict[str, Any]) -> dict[str, Any]:
    rows = M.load_manifest(cfg)
    for i, r in enumerate(rows):
        r["_i"] = i
    by_img: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_img.setdefault(f"{r['src']}|{r['image']}", []).append(r)
    ev = eval_boxes(cfg)
    ev_by: dict[str, list[int]] = {}
    for i, e in enumerate(ev):
        ev_by.setdefault(e["image"], []).append(i)
    out: dict[str, Any] = {"rows": len(rows), "images": len(by_img), "eval_boxes": len(ev)}
    sizes = sorted({int(c["crop_px"]) for c in cfg["train"]["candidates"].values()})
    workers = min(4, int(cfg["train"]["workers"]))
    for cp in sizes:
        t0 = time.time()
        arr = np.zeros((len(rows), int(cfg["train"]["variants"]), 3, cp, cp), np.uint8)
        with Pool(workers) as pool:
            for res in pool.imap_unordered(_prep_image, [(g, cfg, cp) for g in by_img.values()], chunksize=8):
                for i, c in res:
                    arr[i] = c
            ev_arr = np.zeros((len(ev), 3, cp, cp), np.float32)
            jobs = [(f, [ev[i]["box"] for i in idx], cfg, cp) for f, idx in ev_by.items()]
            for (f, _, _, _), c in zip(jobs, pool.imap(_prep_eval, jobs, chunksize=8), strict=True):
                ev_arr[ev_by[f]] = c
        np.save(V.runs_dir(cfg) / f"patches_{cp}.npy", arr)
        np.save(V.runs_dir(cfg) / f"eval_{cp}.npy", ev_arr)
        out[f"prep_s_{cp}"] = round(time.time() - t0, 1)
    V.write_json(V.runs_dir(cfg) / "eval_boxes.json", ev)
    return out


# ---- model ----------------------------------------------------------------------------------------------------------
def build_backbone(name: str, pretrained: bool = True) -> Any:
    import torchvision.models as tvm
    from torch import nn

    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(f"backbone lạ {name!r} (chỉ mobilenet_v3_small, efficientnet_b0 — Deferred 10)")
    return m


def normalize_bgr(x: Any) -> Any:
    """[N, 3, c, c] BGR 0–255 (float) -> RGB /255 chuẩn hoá ImageNet (như graph)."""
    import torch

    mean = torch.tensor(V.IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(V.IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.flip(1) / 255.0 - mean) / std


def _ext_val(rows: list[dict[str, Any]], seed: int) -> np.ndarray:
    """10 % ảnh ngoài (theo ảnh, tất định) giữ lại để dừng sớm."""
    return np.asarray([r["fold"] < 0 and int(hashlib.sha1(f"{seed}|{r['image']}".encode()).hexdigest()[:4], 16) % 10 == 0
                       for r in rows])


def train_one(cfg: dict[str, Any], cand: str, fold: int | None, log: Any) -> dict[str, Any]:
    """Huấn luyện một model (``fold`` = fold giữ lại; None = bản cuối dùng mọi mẫu). Checkpoint mỗi epoch, chạy tiếp được."""
    import torch
    import torch.nn.functional as F

    c = cfg["train"]["candidates"][cand]
    t = cfg["train"]
    cp = int(c["crop_px"])
    tag = f"{cand}_f{fold}" if fold is not None else f"{cand}_final"
    rd = V.runs_dir(cfg) / "runs" / tag
    rd.mkdir(parents=True, exist_ok=True)
    done = rd / "done.json"
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    rows = M.load_manifest(cfg)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    patches = torch.from_numpy(np.load(V.runs_dir(cfg) / f"patches_{cp}.npy")).cuda()  # uint8 [M, V, 3, c, c]
    y = torch.tensor([r["label"] for r in rows], dtype=torch.float32, device="cuda")
    wmap = t["weights"]

    def wkey(r: dict[str, Any]) -> str:
        if r["fold"] < 0:
            return "ext_pos" if r["label"] else "ext_neg"
        return "real_pos" if r["label"] else ("real_neg_hard" if r["type"] == "real_neg_hard" else "real_neg_bg")

    w = torch.tensor([wmap[wkey(r)] for r in rows], dtype=torch.float32, device="cuda")
    ev = _ext_val(rows, cfg["seed"])
    fold_arr = np.asarray([r["fold"] for r in rows])
    tr = ~ev & ((fold_arr != fold) if fold is not None else True)
    tr_idx, va_idx = np.flatnonzero(tr), np.flatnonzero(ev)
    model = build_backbone(c["backbone"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    epochs = int(t["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=epochs * (len(tr_idx) // t["batch"] + 1))
    scaler = torch.amp.GradScaler()
    state_p = rd / "last.pt"
    start, best, best_ep, bad, hist = 0, float("inf"), -1, 0, []
    if state_p.is_file():
        st = torch.load(state_p, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, best_ep, bad, hist = st["epoch"] + 1, st["best"], st["best_ep"], st["bad"], st["hist"]
        rng = np.random.default_rng(cfg["seed"] + start)
    nv = patches.shape[1]
    t0 = time.time()
    for ep in range(start, epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), t["batch"]):
            b = torch.from_numpy(perm[i : i + t["batch"]]).cuda()
            v = torch.randint(0, nv, (len(b),), device="cuda")
            x = patches[b, v].float()
            flip = torch.rand(len(b), device="cuda") < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            gain = 1 + (torch.rand(len(b), 1, 1, 1, device="cuda") - 0.5) * 0.4  # sáng / tương phản nhẹ
            bias = (torch.rand(len(b), 1, 1, 1, device="cuda") - 0.5) * 40
            x = (x * gain + bias).clamp(0, 255)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(normalize_bgr(x)).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y[b], reduction="none") * w[b]).sum() / w[b].sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(b)
        vl = _val_loss(model, patches, y, va_idx)
        hist.append({"epoch": ep, "train_loss": round(tot / len(perm), 5), "val_loss": round(vl, 5),
                     "s": round(time.time() - t0, 1)})
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            torch.save(model.state_dict(), rd / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "best_ep": best_ep, "bad": bad, "hist": hist}, state_p)
        log(f"{tag} ep {ep} train {hist[-1]['train_loss']} val {vl:.5f} best_ep {best_ep} {hist[-1]['s']}s")
        if bad >= int(t["patience"]):
            break
    res = {"tag": tag, "best_epoch": best_ep, "best_val_loss": round(best, 5), "hist": hist,
           "train_n": len(tr_idx), "val_n": len(va_idx), "s": round(time.time() - t0, 1)}
    V.write_json(done, res)
    return res


def _val_loss(model: Any, patches: Any, y: Any, idx: np.ndarray) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(idx), 1024):
            b = torch.from_numpy(idx[i : i + 1024]).cuda()
            logit = model(normalize_bgr(patches[b, 0].float())).squeeze(1).float()
            tot += float(F.binary_cross_entropy_with_logits(logit, y[b], reduction="sum"))
    return tot / max(len(idx), 1)


def predict(model: Any, crops: np.ndarray) -> np.ndarray:
    """Điểm P(người) fp32 cho vùng cắt float BGR [N, 3, c, c] (đúng đường chấm, không autocast)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 512):
            x = torch.from_numpy(crops[i : i + 512]).cuda()
            out.append(torch.sigmoid(model(normalize_bgr(x)).squeeze(1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def load_model(cfg: dict[str, Any], cand: str, tag: str) -> Any:
    import torch

    m = build_backbone(cfg["train"]["candidates"][cand]["backbone"], pretrained=False)
    m.load_state_dict(torch.load(V.runs_dir(cfg) / "runs" / tag / "best.pt", map_location="cpu", weights_only=True))
    return m.eval()


def cmd_run(cfg: dict[str, Any], only: list[str] | None) -> dict[str, Any]:
    from xla.gpu_lock import gpu_lock

    rs = V.real_sets()
    logp = V.runs_dir(cfg) / "logs" / "train.log"
    logp.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(logp, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    ev = json.loads((V.runs_dir(cfg) / "eval_boxes.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    cands = only or list(cfg["train"]["candidates"])
    with gpu_lock("v2 phase07 verifier train", timeout_s=21600):
        t0 = time.time()
        for cand in cands:
            cp = int(cfg["train"]["candidates"][cand]["crop_px"])
            crops = np.load(V.runs_dir(cfg) / f"eval_{cp}.npy")
            res = {"runs": []}
            scores = np.full(len(ev), np.nan, np.float32)
            for k in range(rs["n_folds"]):
                res["runs"].append(train_one(cfg, cand, k, log))
                m = load_model(cfg, cand, f"{cand}_f{k}").cuda()
                idx = [i for i, e in enumerate(ev) if e["role"] == "oof" and e["fold"] == k]
                scores[idx] = predict(m, crops[idx])
            res["runs"].append(train_one(cfg, cand, None, log))
            m = load_model(cfg, cand, f"{cand}_final").cuda()
            idx = [i for i, e in enumerate(ev) if e["role"] == "cal"]
            scores[idx] = predict(m, crops[idx])
            V.write_json(V.runs_dir(cfg) / "scores" / f"{cand}.json", [None if np.isnan(s) else round(float(s), 6)
                                                                         for s in scores])
            out[cand] = {r["tag"]: {"best_epoch": r["best_epoch"], "s": r["s"]} for r in res["runs"]}
            log(f"{cand} xong")
        gpu_s = time.time() - t0
    M._log_gpu(cfg, f"train {','.join(cands)}", gpu_s)
    (V.runs_dir(cfg) / "logs" / "train.done").write_text(json.dumps(out), encoding="utf-8")
    return {"gpu_s": round(gpu_s, 1), "runs": out}


# ---- chấm, chọn, ngưỡng, Kill (c) --------------------------------------------------------------------------------
def curve(ev: list[dict[str, Any]], scores: list[float | None], role: str, thr: float) -> dict[str, Any]:
    """Báo nhầm/khung và recall (GT devpool trúng ≥ 1 hộp giữ lại, IoU ≥ 0,5) khi giữ hộp có điểm verifier ≥ thr."""
    rs = V.real_sets()
    frames = rs["cal"] if role == "cal" else rs["mine"] + rs["devpool"]
    keep: dict[str, list[list[float]]] = {}
    fp = 0
    for e, s in zip(ev, scores, strict=True):
        if e["role"] != role or s is None or s < thr:
            continue
        gt = np.asarray(rs["gt"].get(e["image"], []), np.float32).reshape(-1, 4)
        if not len(gt) or V.iou_matrix(np.asarray([e["box"]]), gt).max() < 0.5:
            fp += 1
        keep.setdefault(e["image"], []).append(e["box"])
    out = {"thr": thr, "fp": fp, "frames": len(frames), "fp_per_frame": fp / len(frames)}
    if role == "oof":
        hit, tot, by_size = 0, 0, {"<16": [0, 0], "16-32": [0, 0], ">=32": [0, 0]}
        for f in rs["devpool"]:
            gt = np.asarray(rs["gt"].get(f, []), np.float32).reshape(-1, 4)
            kb = np.asarray(keep.get(f, []), np.float32).reshape(-1, 4)
            m = V.iou_matrix(gt, kb).max(1) >= 0.5 if len(kb) else np.zeros(len(gt), bool)
            for g, ok in zip(gt, m, strict=True):
                side = max(g[2] - g[0], g[3] - g[1])
                key = "<16" if side < 16 else ("16-32" if side < 32 else ">=32")
                by_size[key][0] += int(ok)
                by_size[key][1] += 1
            hit += int(m.sum())
            tot += len(gt)
        out.update({"recall": hit / tot, "gt": tot, "recall_by_size": {k: [a, b] for k, (a, b) in by_size.items()}})
    return out


def _grid() -> list[float]:
    return [0.0] + [round(x, 3) for x in np.arange(0.005, 1.0, 0.005)]


def cmd_select(cfg: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    s = cfg["select"]
    ev = json.loads((V.runs_dir(cfg) / "eval_boxes.json").read_text(encoding="utf-8"))
    res: dict[str, Any] = {"det_conf": det_conf(cfg), "candidates": {}}
    for cand in cfg["train"]["candidates"]:
        p = V.runs_dir(cfg) / "scores" / f"{cand}.json"
        if not p.is_file():
            continue
        sc = json.loads(p.read_text(encoding="utf-8"))
        oof = [curve(ev, sc, "oof", t) for t in _grid()]
        cal = [curve(ev, sc, "cal", t) for t in _grid()]
        b_oof, b_cal = oof[0], cal[0]
        # điểm OOF: ngưỡng lớn nhất còn recall giảm ≤ max_recall_drop
        ok = [i for i, o in enumerate(oof) if b_oof["recall"] - o["recall"] <= s["max_recall_drop"] + 1e-9]
        i_k = max(ok, key=lambda i: (-oof[i]["fp"], i))
        red_oof = 1 - oof[i_k]["fp"] / max(b_oof["fp"], 1)
        red_cal = 1 - cal[i_k]["fp"] / max(b_cal["fp"], 1)
        # D7: cal fp/khung ≤ đích, trong đó recall OOF lớn nhất; hoà recall thì ít báo nhầm OOF hơn, rồi ngưỡng cao hơn
        d7 = [i for i, c in enumerate(cal) if c["fp_per_frame"] <= s["cal_fp_target"] + 1e-12]
        i_d7 = max(d7, key=lambda i: (oof[i]["recall"], -oof[i]["fp"], i)) if d7 else None
        res["candidates"][cand] = {
            "before": {"oof": b_oof, "cal": b_cal},
            "at_recall_drop": {"thr": oof[i_k]["thr"], "oof": oof[i_k], "cal": cal[i_k],
                               "fp_reduction_oof": round(red_oof, 4), "fp_reduction_cal": round(red_cal, 4)},
            "d7": None if i_d7 is None else {"thr": cal[i_d7]["thr"], "oof": oof[i_d7], "cal": cal[i_d7]},
        }
    if not res["candidates"]:
        return 1, {"error": "chưa có điểm của ứng viên nào (chạy run trước)"}
    best = max(res["candidates"], key=lambda c: res["candidates"][c]["at_recall_drop"]["fp_reduction_oof"])
    r = res["candidates"][best]
    kill = r["at_recall_drop"]["fp_reduction_cal"] < s["kill_min_fp_reduction"]
    res.update({"chosen": best, "kill_c": kill,
                "kill_c_rule": f"giảm báo nhầm cal < {s['kill_min_fp_reduction']} ở recall OOF giảm ≤ {s['max_recall_drop']}"})
    V.write_json(V.results_dir(cfg) / "select.json", res)
    return (1 if kill else 0), {"chosen": best, "kill_c": kill,
                                "table": {c: {"fp_red_oof": v["at_recall_drop"]["fp_reduction_oof"],
                                              "fp_red_cal": v["at_recall_drop"]["fp_reduction_cal"],
                                              "thr": v["at_recall_drop"]["thr"],
                                              "d7_thr": v["d7"] and v["d7"]["thr"],
                                              "d7_recall": v["d7"] and round(v["d7"]["oof"]["recall"], 4)}
                                          for c, v in res["candidates"].items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.verify.train")
    ap.add_argument("cmd", choices=["prep", "run", "select"])
    ap.add_argument("--config", default=str(V.CONFIG_REL))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = V.load_cfg(a.config)
    except OSError as e:
        V.emit({"error": str(e)}, a.json)
        return 2
    if a.only and set(a.only) - set(cfg["train"]["candidates"]):
        V.emit({"error": f"ứng viên lạ {sorted(set(a.only) - set(cfg['train']['candidates']))}"}, a.json)
        return 2
    random.seed(cfg["seed"])
    if a.cmd == "prep":
        res, code = cmd_prep(cfg), 0
    elif a.cmd == "run":
        res, code = cmd_run(cfg, a.only), 0
    else:
        code, res = cmd_select(cfg)
    V.emit(res, a.json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
