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


def is_nadir(item: SourceItem, rule: dict) -> bool:
    """Metadata thắng; không có thì cần cả hai: ``sky_frac ≤ max_sky_frac`` và ``clip_nadir ≥ min_clip_nadir``."""
    meta = rule.get("metadata", {}).get(item.source)
    if meta == "nadir":
        return True
    if meta == "oblique":
        return False
    if item.sky_frac is None or item.clip_nadir is None:
        raise ValueError(f"{item.key}: chưa có điểm nadir (chạy `python -m xla.degrade.sources score`)")
    return item.sky_frac <= rule["max_sky_frac"] and item.clip_nadir >= rule["min_clip_nadir"]


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


def select(cfg: dict) -> list[SourceItem]:
    """Ảnh nguồn dùng được: có điểm nadir đạt ngưỡng; ảnh không người vẫn giữ (cho cửa sổ âm)."""
    items = collect(cfg)
    attach_scores(items, cfg)
    rule = cfg["nadir"]
    out = [it for it in items if is_nadir(it, rule)]
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


def cmd_stats(cfg: dict, as_json: bool) -> int:
    items = collect(cfg)
    attach_scores(items, cfg)
    rule = cfg["nadir"]
    st: dict = {}
    for it in items:
        d = st.setdefault(f"{it.source}/{it.split}", {"images": 0, "with_person": 0, "persons": 0, "scored": 0,
                                                      "nadir_images": 0, "nadir_with_person": 0, "nadir_persons": 0})
        d["images"] += 1
        d["with_person"] += bool(it.boxes)
        d["persons"] += len(it.boxes)
        if it.clip_nadir is not None:
            d["scored"] += 1
            if is_nadir(it, rule):
                d["nadir_images"] += 1
                d["nadir_with_person"] += bool(it.boxes)
                d["nadir_persons"] += len(it.boxes)
    print(json.dumps(st, ensure_ascii=False, indent=None if as_json else 2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "review-sheet", "stats"])
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
    return cmd_stats(cfg, a.json)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SourceItem", "asdict", "collect", "is_nadir", "parse_visdrone_annotation", "select", "sky_fraction"]
