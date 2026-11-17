"""Dữ liệu nguồn sạch có nhãn người cho D0/D1/D2: VisDrone2019-DET + tập con NOMAD, lọc ảnh nhìn thẳng xuống.

- VisDrone: category 1 (pedestrian) + 2 (people) → lớp người; category 0 (ignored regions) → vùng cửa sổ cắt không
  được chồng lên (``geometry.plan_window``). Định dạng dòng: ``x,y,w,h,score,category,truncation,occlusion``.
- NOMAD: nhãn ``annotations/annotations.json`` (Detectron2, ``bbox_mode`` 1 = XYWH tuyệt đối, theo ảnh GỐC 5472×3078),
  quy về ảnh đã hạ mẫu theo ``fetch_manifest.jsonl``; khoảng cách theo thư mục ``ActorXXX_aNN`` (NN mét).
  Hộp có ``visibility`` < ``min_visibility`` coi là vùng bỏ qua (người gần như khuất: không làm dương, không làm âm).
- Lọc nadir (Deferred 3): metadata khi có; còn lại heuristic không-có-trời + điểm CLIP zero-shot; ngưỡng ở
  ``configs/sources.yaml``, chọn trên 100 ảnh soát tay (``review-sheet``).

    python -m xla.degrade.sources score   --config configs/sources.yaml      # tính sky_frac + clip_nadir → cache
    python -m xla.degrade.sources review-sheet --n 100 --out <thư mục>        # lưới ảnh nhỏ để soát tay
    python -m xla.degrade.sources seq-sheet --per-sheet 24 --out <thư mục>   # lưới soát nhãn chuỗi VisDrone
    python -m xla.degrade.sources fit-probe        # probe CLIP trên nhãn soát tay → probe_scores_file
    python -m xla.degrade.sources stats   --config configs/sources.yaml --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from xla import paths

VISDRONE_PERSON = (1, 2)
VISDRONE_IGNORED = 0
NOMAD_DIST_RE = re.compile(r"_a(\d+)$")
NADIR_PROMPTS = [
    "a top-down aerial photo looking straight down at the ground",
    "an overhead nadir view from a drone camera pointing straight down",
    "a bird's eye view of the ground seen from directly above",
]
OBLIQUE_PROMPTS = [
    "an oblique aerial photo taken at an angle, with the horizon or distant buildings visible",
    "a drone photo looking forward over a street in perspective",
    "a photo taken from a tall building looking down at an angle",
]


@dataclass
class SourceItem:
    key: str                     # "<source>/<stem>" duy nhất
    source: str                  # visdrone | nomad
    image: str                   # đường dẫn tuyệt đối
    split: str                   # train | val
    wh: tuple[int, int]
    boxes: list = field(default_factory=list)    # [[x1,y1,x2,y2], ...] người, px ảnh nguồn
    ignore: list = field(default_factory=list)   # vùng bỏ qua
    alt_m: float | None = None                    # độ cao/khoảng cách theo metadata (NOMAD)
    sky_frac: float | None = None
    clip_nadir: float | None = None
    probe_nadir: float | None = None              # xác suất "nhận" của probe tuyến tính trên đặc trưng CLIP

    def boxes_np(self) -> np.ndarray:
        return np.asarray(self.boxes, np.float32).reshape(-1, 4)

    def ignore_np(self) -> np.ndarray:
        return np.asarray(self.ignore, np.float32).reshape(-1, 4)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else paths.src_datasets_dir() / q


# ----------------------------------------------------------------------------- VisDrone

def parse_visdrone_annotation(text: str) -> tuple[list, list]:
    """Một file nhãn VisDrone → (hộp người xyxy, vùng ignored xyxy)."""
    boxes, ignore = [], []
    for line in text.splitlines():
        parts = [p for p in line.strip().split(",") if p != ""]
        if len(parts) < 6:
            continue
        x, y, w, h = (float(v) for v in parts[:4])
        cat = int(parts[5])
        if w <= 0 or h <= 0:
            continue
        b = [x, y, x + w, y + h]
        if cat in VISDRONE_PERSON:
            boxes.append(b)
        elif cat == VISDRONE_IGNORED:
            ignore.append(b)
    return boxes, ignore


def load_visdrone(root: Path, subsets: dict[str, str]) -> list[SourceItem]:
    items = []
    for sub, split in subsets.items():
        d = root / sub
        if not (d / "images").is_dir():
            raise FileNotFoundError(f"thiếu VisDrone {d / 'images'} (giải nén theo configs/sources.yaml)")
        for img in sorted((d / "images").glob("*.jpg")):
            ann = d / "annotations" / (img.stem + ".txt")
            boxes, ignore = parse_visdrone_annotation(ann.read_text(encoding="utf-8")) if ann.exists() else ([], [])
            h, w = _image_hw(img)
            items.append(SourceItem(f"visdrone/{img.stem}", "visdrone", str(img), split, (w, h), boxes, ignore))
    return items


def _image_hw(path: Path) -> tuple[int, int]:
    from PIL import Image  # chỉ đọc header

    with Image.open(path) as im:
        return im.height, im.width


# ----------------------------------------------------------------------------- NOMAD

def load_nomad(root: Path, min_visibility: int, val_actor_mod: int) -> list[SourceItem]:
    man = root / "fetch_manifest.jsonl"
    if not man.exists():
        return []
    fetched = {}
    for line in man.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            fetched[r["name"]] = r
    anns = json.loads((root / "annotations" / "annotations.json").read_text(encoding="utf-8"))
    items = []
    for rec in anns:
        name = rec["file_name"]
        if name not in fetched:
            continue
        f = fetched[name]
        sx = f["saved_wh"][0] / f["orig_wh"][0]
        sy = f["saved_wh"][1] / f["orig_wh"][1]
        boxes, ignore = [], []
        for a in rec.get("annotations", []):
            if int(a.get("bbox_mode", 1)) != 1:
                raise ValueError(f"NOMAD {name}: bbox_mode {a.get('bbox_mode')} chưa hỗ trợ (chỉ XYWH_ABS)")
            x, y, w, h = a["bbox"]
            b = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
            (boxes if int(a.get("visibility", 100)) >= min_visibility else ignore).append(b)
        stem = Path(name).stem
        folder = stem.rsplit("_f", 1)[0]
        m = NOMAD_DIST_RE.search(folder)
        actor = int(folder[5:8])
        split = "val" if actor % val_actor_mod == 0 else "train"   # tách theo diễn viên, tránh rò cảnh
        img = Path(f["rel"])
        items.append(SourceItem(f"nomad/{stem}", "nomad", str(img), split, tuple(f["saved_wh"]), boxes, ignore,
                                alt_m=float(m.group(1)) if m else None))
    return items


# ----------------------------------------------------------------------------- lọc nadir

def sky_fraction(img_bgr: np.ndarray, top_frac: float = 0.3) -> float:
    """Tỉ lệ điểm "giống trời" ở dải trên cùng: sáng, ít bão hoà hoặc ngả xanh dương, gần như không có vân."""
    small = cv2.resize(img_bgr, (320, round(320 * img_bgr.shape[0] / img_bgr.shape[1])), interpolation=cv2.INTER_AREA)
    band = small[: max(1, int(small.shape[0] * top_frac))]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    b, g, r = (band[..., i].astype(np.float32) for i in range(3))
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad = cv2.blur(grad, (5, 5))
    bright_flat = (v > 0.55) & (grad < 12.0)
    skyish = ((b >= r) & (b >= g - 5) & (s < 0.6)) | (s < 0.12)
    return float((bright_flat & skyish).mean())


class ClipScorer:
    """Điểm zero-shot P(nadir) bằng open_clip (ViT-B-32, laion2b), so hai nhóm câu mô tả."""

    def __init__(self, model_name: str, pretrained: str, device: str):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(device)
        tok = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            t = self.model.encode_text(tok(NADIR_PROMPTS + OBLIQUE_PROMPTS).to(device)).float()
            t = t / t.norm(dim=-1, keepdim=True)
        n = len(NADIR_PROMPTS)
        self.text = torch.stack([t[:n].mean(0), t[n:].mean(0)])
        self.text = self.text / self.text.norm(dim=-1, keepdim=True)

    def score(self, imgs_bgr: list[np.ndarray]) -> list[float]:
        from PIL import Image

        torch = self.torch
        batch = torch.stack([self.preprocess(Image.fromarray(i[..., ::-1])) for i in imgs_bgr]).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(batch).float()
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100.0 * f @ self.text.T
            return logits.softmax(dim=-1)[:, 0].cpu().tolist()


VIEW_CLASSES = ("nadir", "near", "oblique", "mixed")


def visdrone_seq(item: SourceItem) -> str:
    """Mã chuỗi VisDrone theo split: ``train/0000002`` (số chuỗi trùng giữa train và val nên kèm split)."""
    return f"{item.split}/{Path(item.image).stem.split('_')[0]}"


def load_view_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """Nhãn góc nhìn soát tay (``configs/visdrone_view.yaml``): ``{"sequences": {"train/0000002": lớp},
    "images": {"visdrone/<stem>": lớp}}``. Nhãn ảnh thắng nhãn chuỗi; ảnh không mang lớp ``mixed``."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {"sequences": {str(k): str(v) for k, v in (data.get("sequences") or {}).items()},
           "images": {str(k): str(v) for k, v in (data.get("images") or {}).items()}}
    bad = {k: v for k, v in out["sequences"].items() if v not in VIEW_CLASSES}
    bad |= {k: v for k, v in out["images"].items() if v not in VIEW_CLASSES[:3]}
    if bad:
        raise ValueError(f"{path}: nhãn không hợp lệ (chuỗi: {VIEW_CLASSES}, ảnh: {VIEW_CLASSES[:3]}): {bad}")
    return out


def view_of(item: SourceItem, labels: dict[str, dict[str, str]]) -> str:
    """Lớp góc nhìn soát tay của một ảnh VisDrone: nhãn ảnh nếu có, không thì nhãn chuỗi (thiếu → lỗi)."""
    if item.key in labels["images"]:
        return labels["images"][item.key]
    seq = visdrone_seq(item)
    if seq not in labels["sequences"]:
        raise KeyError(f"chuỗi {seq} chưa có nhãn góc nhìn (configs/visdrone_view.yaml)")
    return labels["sequences"][seq]


def _score_pass(item: SourceItem, rule: dict) -> bool:
    if item.sky_frac is None or item.clip_nadir is None:
        raise ValueError(f"{item.key}: chưa có điểm nadir (chạy `python -m xla.degrade.sources score`)")
    return item.sky_frac <= rule["max_sky_frac"] and item.clip_nadir >= rule["min_clip_nadir"]


def is_nadir(item: SourceItem, rule: dict, view_labels: dict | None = None) -> bool:
    """Thứ tự quyết: (1) nguồn bị loại cả bộ (``exclude_sources``); (2) metadata nguồn; (3) nhãn soát tay VisDrone
    (ảnh, rồi chuỗi): lớp trong ``accept_views`` → nhận; ``mixed`` → probe (``probe_nadir ≥ min_probe_nadir``) nếu
    cấu hình có ngưỡng probe, không thì sky + CLIP; lớp khác → loại; (4) không có bảng nhãn → sky + CLIP.
    Chuỗi VisDrone thiếu nhãn khi đã có bảng nhãn là lỗi (bảng phải đủ)."""
    if item.source in rule.get("exclude_sources", {}):
        return False
    meta = rule.get("metadata", {}).get(item.source)
    if meta == "nadir":
        return True
    if meta == "oblique":
        return False
    if view_labels is not None and item.source == "visdrone":
        view = view_of(item, view_labels)
        if view != "mixed":
            return view in rule["accept_views"]
        if rule.get("min_probe_nadir") is not None:
            if item.probe_nadir is None:
                raise ValueError(f"{item.key}: chưa có điểm probe (chạy `python -m xla.degrade.sources fit-probe`)")
            return item.probe_nadir >= float(rule["min_probe_nadir"])
    return _score_pass(item, rule)


def view_labels_for(cfg: dict) -> dict | None:
    f = cfg["nadir"].get("view_labels_file")
    if not f:
        return None
    p = Path(f)
    return load_view_labels(p if p.is_absolute() else paths.repo_root() / p)


# ----------------------------------------------------------------------------- chỉ mục

def collect(cfg: dict) -> list[SourceItem]:
    items: list[SourceItem] = []
    vd = cfg["visdrone"]
    items += load_visdrone(_resolve(vd["root"]), vd["subsets"])
    nm = cfg.get("nomad")
    if nm and nm.get("enabled", True):
        items += load_nomad(_resolve(nm["root"]), int(nm["min_visibility"]), int(nm["val_actor_mod"]))
    return items


def scores_path(cfg: dict) -> Path:
    return _resolve(cfg["nadir"]["scores_file"])


def attach_scores(items: list[SourceItem], cfg: dict) -> None:
    p = scores_path(cfg)
    if not p.exists():
        return
    sc = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            sc[r["key"]] = r
    for it in items:
        if it.key in sc:
            it.sky_frac = sc[it.key]["sky_frac"]
            it.clip_nadir = sc[it.key]["clip_nadir"]
    pf = cfg["nadir"].get("probe_scores_file")
    if pf and _resolve(pf).exists():
        pr = {}
        for line in _resolve(pf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pr[r["key"]] = r["probe_nadir"]
        for it in items:
            it.probe_nadir = pr.get(it.key)


def select(cfg: dict) -> list[SourceItem]:
    """Ảnh nguồn dùng được: có điểm nadir đạt ngưỡng; ảnh không người vẫn giữ (cho cửa sổ âm)."""
    items = collect(cfg)
    attach_scores(items, cfg)
    rule = cfg["nadir"]
    views = view_labels_for(cfg)
    out = [it for it in items if is_nadir(it, rule, views)]
    if not cfg.get("keep_images_without_person", True):
        out = [it for it in out if it.boxes]
    return sorted(out, key=lambda i: i.key)


# ----------------------------------------------------------------------------- CLI

def cmd_score(cfg: dict, batch: int, device: str) -> int:
    items = collect(cfg)
    out = scores_path(cfg)
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [it for it in items if it.key not in done]
    print(f"[sources] chấm nadir {len(todo)} ảnh (đã có {len(done)})", flush=True)
    if not todo:
        return 0
    nd = cfg["nadir"]
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    from contextlib import nullcontext

    from xla.gpu_lock import gpu_lock

    ctx = gpu_lock("degrade:clip-nadir") if device == "cuda" else nullcontext()
    with ctx:
        scorer = ClipScorer(nd["clip_model"], nd["clip_pretrained"], device)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8", newline="\n") as f:
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                imgs = []
                for it in chunk:
                    im = cv2.imread(it.image, cv2.IMREAD_REDUCED_COLOR_2)
                    if im is None:
                        raise RuntimeError(f"không đọc được {it.image}")
                    imgs.append(im)
                probs = scorer.score(imgs)
                for it, im, pr in zip(chunk, imgs, probs):
                    f.write(json.dumps({"key": it.key, "sky_frac": round(sky_fraction(im), 5),
                                        "clip_nadir": round(float(pr), 5)}) + "\n")
                f.flush()
                if (i // batch) % 20 == 0:
                    print(f"[sources] {i + len(chunk)}/{len(todo)}", flush=True)
    return 0


def cmd_review_sheet(cfg: dict, n: int, seed: int, out_dir: Path, per_sheet: int) -> int:
    """Lưới ảnh nhỏ đánh số (n ảnh ngẫu nhiên có người, có điểm) + CSV điểm để soát tay chọn ngưỡng."""
    items = [it for it in collect(cfg) if it.boxes]
    attach_scores(items, cfg)
    items = [it for it in items if it.clip_nadir is not None]
    rng = np.random.default_rng(seed)
    pick = [items[i] for i in sorted(rng.choice(len(items), size=min(n, len(items)), replace=False))]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["idx,key,sky_frac,clip_nadir"]
    tw, th, cols = 300, 200, 5
    for s in range(0, len(pick), per_sheet):
        tiles = []
        for j, it in enumerate(pick[s:s + per_sheet]):
            idx = s + j
            im = cv2.imread(it.image, cv2.IMREAD_REDUCED_COLOR_4)
            t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (tw, 18), (0, 0, 0), -1)
            cv2.putText(t, f"#{idx} sky{it.sky_frac:.2f} clip{it.clip_nadir:.2f}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
            rows.append(f"{idx},{it.key},{it.sky_frac},{it.clip_nadir}")
        while len(tiles) % cols:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
        cv2.imwrite(str(out_dir / f"sheet_{s // per_sheet:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out_dir / "review.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"[sources] {len(pick)} ảnh → {out_dir}")
    return 0


def clip_embeddings(items: list[SourceItem], model_name: str, pretrained: str, cache: Path,
                    batch: int = 64) -> dict[str, np.ndarray]:
    """Đặc trưng ảnh CLIP (chuẩn hoá L2), cache ``.npz`` theo khoá; chỉ tính ảnh chưa có."""
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["keys"].tolist(), z["emb"], strict=True))
    todo = [it for it in items if it.key not in have]
    if todo:
        from contextlib import nullcontext

        import torch
        from PIL import Image

        from xla.gpu_lock import gpu_lock

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with gpu_lock("degrade:clip-emb") if device == "cuda" else nullcontext():
            scorer = ClipScorer(model_name, pretrained, device)
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                imgs = [cv2.imread(it.image, cv2.IMREAD_REDUCED_COLOR_2) for it in chunk]
                with torch.no_grad():
                    b = torch.stack([scorer.preprocess(Image.fromarray(np.ascontiguousarray(im[..., ::-1])))
                                     for im in imgs]).to(device)
                    f = scorer.model.encode_image(b).float()
                    f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
                have.update(zip((it.key for it in chunk), f, strict=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(have)
        np.savez(cache, keys=np.array(keys), emb=np.stack([have[k] for k in keys]))
    return have


def cmd_fit_probe(cfg: dict, cache: Path) -> int:
    """Probe tuyến tính (logistic, C=1, cân lớp) trên đặc trưng CLIP: nhãn = lớp soát tay (chuỗi nadir/near/oblique +
    nhãn từng ảnh), dương = ``accept_views``. Chấm mọi ảnh VisDrone → ``probe_scores_file``; in kiểm chéo 5 phần trên
    ảnh có nhãn riêng (ảnh chuỗi mixed) để chọn ``min_probe_nadir``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    nd = cfg["nadir"]
    labels = view_labels_for(cfg)
    if labels is None:
        raise ValueError("fit-probe cần nadir.view_labels_file")
    items = [it for it in collect(cfg) if it.source == "visdrone"]
    emb = clip_embeddings(items, nd["clip_model"], nd["clip_pretrained"], cache)
    accept = set(nd["accept_views"])
    seq_tr = [(it.key, labels["sequences"][visdrone_seq(it)] in accept) for it in items
              if it.key not in labels["images"] and labels["sequences"][visdrone_seq(it)] != "mixed"]
    img_tr = [(k, v in accept) for k, v in sorted(labels["images"].items())]
    xs = np.stack([emb[k] for k, _ in seq_tr])
    ys = np.array([v for _, v in seq_tr])
    xi = np.stack([emb[k] for k, _ in img_tr])
    yi = np.array([v for _, v in img_tr])
    cv = np.zeros(len(img_tr))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(xi, yi):
        c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        c.fit(np.vstack([xs, xi[tr]]), np.concatenate([ys, yi[tr]]))
        cv[te] = c.predict_proba(xi[te])[:, 1]
    table = []
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        sel = cv >= t
        tp = int((sel & yi).sum())
        table.append({"t": t, "selected": int(sel.sum()), "precision": round(tp / max(int(sel.sum()), 1), 3),
                      "recall": round(tp / max(int(yi.sum()), 1), 3)})
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    clf.fit(np.vstack([xs, xi]), np.concatenate([ys, yi]))
    keys = sorted(emb)
    prob = clf.predict_proba(np.stack([emb[k] for k in keys]))[:, 1]
    out = _resolve(nd["probe_scores_file"])
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for k, p in zip(keys, prob, strict=True):
            f.write(json.dumps({"key": k, "probe_nadir": round(float(p), 5)}) + "\n")
    print(json.dumps({"accept_views": sorted(accept), "train_seq_images": len(seq_tr), "train_image_labels": len(img_tr),
                      "positives_image_labels": int(yi.sum()), "cv_on_image_labels": table, "out": str(out)},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_seq_sheet(cfg: dict, out_dir: Path, per_sheet: int) -> int:
    """Lưới soát nhãn góc nhìn theo chuỗi VisDrone: mỗi hàng một chuỗi, 3 khung (đầu / giữa / cuối) có ghi mã chuỗi."""
    by_seq: dict[str, list[SourceItem]] = {}
    for it in collect(cfg):
        if it.source == "visdrone":
            by_seq.setdefault(visdrone_seq(it), []).append(it)
    seqs = sorted(by_seq)
    out_dir.mkdir(parents=True, exist_ok=True)
    tw, th, per_row = 260, 160, 2
    rows_csv = ["seq,n_images,frames"]
    for s in range(0, len(seqs), per_sheet):
        tiles = []
        for seq in seqs[s:s + per_sheet]:
            its = sorted(by_seq[seq], key=lambda i: i.key)
            pick = [its[0], its[len(its) // 2], its[-1]]
            rows_csv.append(f"{seq},{len(its)},{'|'.join(Path(i.image).stem for i in pick)}")
            for k, it in enumerate(pick):
                im = cv2.imread(it.image, cv2.IMREAD_REDUCED_COLOR_4)
                t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
                cv2.rectangle(t, (0, 0), (tw, 16), (0, 0, 0), -1)
                cv2.putText(t, f"{seq} n{len(its)} [{k}]", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(t)
        sep = np.full((th, 8, 3), 255, np.uint8)
        groups = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        while len(groups) % per_row:
            groups.append(np.zeros_like(groups[0]))
        grid = np.vstack([np.hstack([groups[r], sep, groups[r + 1]]) for r in range(0, len(groups), per_row)])
        cv2.imwrite(str(out_dir / f"seq_{s // per_sheet:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out_dir / "sequences.csv").write_text("\n".join(rows_csv) + "\n", encoding="utf-8", newline="\n")
    print(f"[sources] {len(seqs)} chuỗi → {out_dir}")
    return 0


def cmd_stats(cfg: dict, as_json: bool) -> int:
    items = collect(cfg)
    attach_scores(items, cfg)
    rule = cfg["nadir"]
    views = view_labels_for(cfg)
    st: dict = {}
    for it in items:
        d = st.setdefault(f"{it.source}/{it.split}", {"images": 0, "with_person": 0, "persons": 0, "scored": 0,
                                                      "nadir_images": 0, "nadir_with_person": 0, "nadir_persons": 0})
        d["images"] += 1
        d["with_person"] += bool(it.boxes)
        d["persons"] += len(it.boxes)
        if it.clip_nadir is not None:
            d["scored"] += 1
        scored = it.source in rule.get("exclude_sources", {}) or it.clip_nadir is not None
        if scored and is_nadir(it, rule, views):
            d["nadir_images"] += 1
            d["nadir_with_person"] += bool(it.boxes)
            d["nadir_persons"] += len(it.boxes)
    print(json.dumps(st, ensure_ascii=False, indent=None if as_json else 2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "review-sheet", "seq-sheet", "fit-probe", "stats"])
    ap.add_argument("--config", default=str(paths.repo_root() / "configs" / "sources.yaml"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--per-sheet", type=int, default=25)
    ap.add_argument("--out", type=Path, default=paths.cache_dir() / "degrade" / "nadir_review")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "score":
        return cmd_score(cfg, a.batch, a.device)
    if a.cmd == "review-sheet":
        return cmd_review_sheet(cfg, a.n, a.seed, a.out, a.per_sheet)
    if a.cmd == "seq-sheet":
        return cmd_seq_sheet(cfg, a.out, a.per_sheet)
    if a.cmd == "fit-probe":
        return cmd_fit_probe(cfg, paths.cache_dir() / "degrade" / f"clip_emb_{cfg['nadir']['clip_model']}.npz")
    return cmd_stats(cfg, a.json)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SourceItem", "asdict", "collect", "is_nadir", "parse_visdrone_annotation", "select", "sky_fraction"]
