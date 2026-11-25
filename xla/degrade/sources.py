"""Dữ liệu nguồn sạch có nhãn người cho D0/D1/D2: VisDrone2019-DET + tập con NOMAD, lọc ảnh nhìn thẳng xuống.

- VisDrone: category 1 (pedestrian) + 2 (people) → lớp người; category 0 (ignored regions) → vùng cửa sổ cắt không
  được chồng lên (``geometry.plan_window``). Định dạng dòng: ``x,y,w,h,score,category,truncation,occlusion``.
- NOMAD: nhãn ``annotations/annotations.json`` (Detectron2, ``bbox_mode`` 1 = XYWH tuyệt đối, theo ảnh GỐC 5472×3078),
  quy về ảnh đã hạ mẫu theo ``fetch_manifest.jsonl``; khoảng cách theo thư mục ``ActorXXX_aNN`` (NN mét).
  Hộp có ``visibility`` < ``min_visibility`` coi là vùng bỏ qua (người gần như khuất: không làm dương, không làm âm).
- Lọc nadir (Deferred 3): metadata khi có; còn lại heuristic không-có-trời + điểm CLIP zero-shot; ngưỡng ở
  ``configs/sources.yaml``, chọn trên 100 ảnh soát tay (``review-sheet``).
- Nguồn ngoài (AI v2 phase 02, khối ``external``): HERIDAL (VOC XML, cắt ô trước khi làm xấu), TinyPerson (COCO),
  WiSARD VIS (YOLO). Chỉ vào tập trộn ``d02x``; d0/d1/d2 v1 vẫn chỉ VisDrone. Lọc góc: nhãn soát tay
  (``configs/view_external.yaml``) rồi probe CLIP đã fit trên nhãn VisDrone, cùng ngưỡng.

    python -m xla.degrade.sources score   --config configs/sources.yaml      # tính sky_frac + clip_nadir → cache
    python -m xla.degrade.sources review-sheet --n 100 --out <thư mục>        # lưới ảnh nhỏ để soát tay
    python -m xla.degrade.sources seq-sheet --per-sheet 24 --out <thư mục>   # lưới soát nhãn chuỗi VisDrone
    python -m xla.degrade.sources fit-probe        # probe CLIP trên nhãn soát tay → probe_scores_file
    python -m xla.degrade.sources stats   --config configs/sources.yaml --json
    python -m xla.degrade.sources extract [--set heridal] [--verify]   # giải nén nguồn ngoài + manifest sha256
    python -m xla.degrade.sources probe-external   # chấm probe góc nhìn cho ảnh nguồn ngoài (GPU, gpu_lock)
    python -m xla.degrade.sources seq-sheet --source heridal --n 100 --out <thư mục>   # soát tay nguồn ngoài
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

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
    group: str | None = None                      # chuỗi / cảnh (nguồn ngoài): chia train/val theo nhóm, không theo ảnh
    crop: list | None = None                      # [x0, y0, x1, y1] ô cắt trong ảnh gốc (HERIDAL); ``wh`` = cỡ ô

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


def load_val_split(cfg: dict) -> dict | None:
    """Danh sách chuỗi VisDrone train chuyển sang val (``val_split_file``, sinh bằng lệnh ``pick-val``)."""
    f = cfg.get("val_split_file")
    if not f:
        return None
    p = Path(f) if Path(f).is_absolute() else paths.repo_root() / f
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    bad = [s for s in data["train_to_val"] if not str(s).startswith("train/")]
    if bad:
        raise ValueError(f"{p}: train_to_val chỉ nhận chuỗi train/…: {bad}")
    return data


def _filtered(cfg: dict) -> list[SourceItem]:
    items = collect(cfg)
    attach_scores(items, cfg)
    rule = cfg["nadir"]
    views = view_labels_for(cfg)
    out = [it for it in items if is_nadir(it, rule, views)]
    if not cfg.get("keep_images_without_person", True):
        out = [it for it in out if it.boxes]
    return sorted(out, key=lambda i: i.key)


def select(cfg: dict, external: bool = False) -> list[SourceItem]:
    """Ảnh nguồn dùng được: có điểm nadir đạt ngưỡng; ảnh không người vẫn giữ (cho cửa sổ âm). Chuỗi train nằm trong
    ``val_split_file`` chuyển sang val (chia theo chuỗi, cùng chia cho D0/D1/D2). ``external``: thêm nguồn ngoài đã
    lọc + cắt ô (chỉ tập trộn ``d02x``; d0/d1/d2 v1 giữ nguyên VisDrone)."""
    out = _filtered(cfg)
    vs = load_val_split(cfg)
    if vs:
        move = set(vs["train_to_val"])
        for it in out:
            if it.source == "visdrone" and visdrone_seq(it) in move:
                it.split = "val"
    if external:
        out += select_external(cfg)
    return out


def pick_val_sequences(items: list[SourceItem], seed: int, min_val_with_person: int, max_extra: int) -> list[str]:
    """Chọn chuỗi VisDrone train chuyển sang val: trộn có seed, lấy lần lượt chuỗi nào không làm số ảnh có người thêm
    vượt ``max_extra``, dừng khi val đạt ``min_val_with_person`` ảnh có người."""
    per: dict[str, int] = {}
    have = 0
    for it in items:
        if it.source != "visdrone":
            continue
        s = visdrone_seq(it)
        if it.split == "val":
            have += bool(it.boxes)
        else:
            per[s] = per.get(s, 0) + bool(it.boxes)
    need = min_val_with_person - have
    order = sorted(per)
    np.random.default_rng(seed).shuffle(order)
    chosen, extra = [], 0
    for s in order:
        if extra >= need:
            break
        if per[s] and extra + per[s] <= max_extra:
            chosen.append(s)
            extra += per[s]
    if extra < need:
        raise ValueError(f"không đủ chuỗi train để val đạt {min_val_with_person} ảnh có người (thêm được {extra})")
    return sorted(chosen)


def cmd_pick_val(cfg: dict, seed: int, min_val: int, max_extra: int, out: Path) -> int:
    items = _filtered(cfg)
    chosen = pick_val_sequences(items, seed, min_val, max_extra)
    ch = set(chosen)
    tr_seq = {visdrone_seq(i) for i in items if i.source == "visdrone" and i.split == "train"}
    moved = [i for i in items if i.source == "visdrone" and visdrone_seq(i) in ch]
    val_p = sum(bool(i.boxes) for i in items if i.split == "val") + sum(bool(i.boxes) for i in moved)
    head = (f"# Chuỗi VisDrone train chuyển sang val của tập sinh D0/D1/D2 (phase 07, lệnh Owner 18/11: val gốc chỉ "
            f"{val_p - sum(bool(i.boxes) for i in moved)} ảnh\n"
            f"# có người). Sinh bằng: python -m xla.degrade.sources pick-val --seed {seed} --min-val {min_val} "
            f"--max-extra {max_extra}\n"
            "# Chia theo CHUỖI (không theo ảnh) để không rò cảnh giữa train và val. Không sửa tay.\n")
    body = {"seed": seed, "min_val_with_person": min_val, "max_extra": max_extra,
            "train_sequences_total": len(tr_seq), "moved_sequences": len(chosen),
            "moved_images": len(moved), "moved_with_person": sum(bool(i.boxes) for i in moved),
            "val_with_person_after": val_p, "train_to_val": chosen}
    out.write_text(head + yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in body.items() if k != "train_to_val"} | {"out": str(out)}, ensure_ascii=False))
    return 0


# ----------------------------------------------------------------------------- nguồn ngoài: giải nén + manifest

EXTERNAL_SETS = ("heridal", "tinyperson", "wisard")
HERIDAL_STEM_RE = re.compile(r"^(train|test)_([A-Z]+)_\d+$")     # train_BRA_0001: mã nơi bay = cảnh
TINYPERSON_GROUP_RE = re.compile(r"^(.+?_V\d+)_I\d+$")            # bb_V0032_I0001640: khung của video V0032
TINYPERSON_PERSON = ("sea_person", "earth_person")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
EXTRACT_MANIFEST = "manifest.jsonl"
EXTRACT_INFO = "extract.json"
GPU_LOCK_TIMEOUT_S = 21600                                         # D6: chờ khoá GPU tới 6 giờ
_CHUNK = 1 << 20


def external_cfg(cfg: dict) -> dict | None:
    ext = cfg.get("external")
    return ext if ext and ext.get("sets") else None


def external_root(ext: dict) -> Path:
    """Gốc giải nén nguồn ngoài: ``external.root`` tương đối ``data_root`` (configs/paths.yaml)."""
    p = Path(ext["root"])
    return p if p.is_absolute() else paths.data_root() / p


def external_sets(ext: dict) -> list[str]:
    names = [n for n, s in ext["sets"].items() if s.get("enabled", True)]
    bad = [n for n in names if n not in EXTERNAL_SETS]
    if bad:
        raise ValueError(f"external.sets: bộ chưa có trình đọc {bad} (có: {EXTERNAL_SETS})")
    return names


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def safe_member(name: str) -> PurePosixPath:
    """Tên thành viên file nén → đường dẫn tương đối an toàn; tuyệt đối / ``..`` / ổ đĩa → lỗi (file nén là dữ liệu
    ngoài, không tin)."""
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or not p.parts or any(x in ("..", "", ".") for x in p.parts) or ":" in p.parts[0]:
        raise ValueError(f"tên thành viên nguy hiểm trong file nén: {name!r}")
    return p


def _members(archive: Path) -> Iterator[tuple[str, int, Callable]]:
    """(tên, kích thước, hàm mở luồng) của từng file thường trong zip / tar(.gz); file lẻ (json) là chính nó.
    Liên kết (symlink/hardlink) trong tar → lỗi."""
    low = archive.name.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not info.is_dir():
                    yield info.filename, info.file_size, (lambda i=info: z.open(i))
    elif low.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            for m in t:
                if m.isfile():
                    yield m.name, m.size, (lambda m=m: t.extractfile(m))
                elif not m.isdir():
                    raise ValueError(f"{archive.name}: thành viên không phải file thường {m.name!r} (type {m.type!r})")
    else:
        yield archive.name, archive.stat().st_size, (lambda: archive.open("rb"))


def _write_hashed(src, dest: Path) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h, n = hashlib.sha256(), 0
    with tmp.open("wb") as out:
        while chunk := src.read(_CHUNK):
            h.update(chunk)
            out.write(chunk)
            n += len(chunk)
    tmp.replace(dest)
    return h.hexdigest(), n


def _wanted(rel: str, scfg: dict) -> bool:
    if any(fnmatch.fnmatchcase(rel, p) for p in scfg.get("exclude", [])):
        return False
    return any(fnmatch.fnmatchcase(rel, p) for p in scfg["include"])


def read_manifest(set_dir: Path) -> list[dict]:
    p = set_dir / EXTRACT_MANIFEST
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_manifest(set_dir: Path, full: bool = True) -> list[str]:
    """Đối chiếu file trên đĩa với ``manifest.jsonl`` (``full``: băm lại sha256, không thì chỉ cỡ). Trả danh sách lỗi."""
    problems = []
    for r in read_manifest(set_dir):
        p = set_dir / r["rel"]
        if not p.is_file():
            problems.append(f"thiếu {r['rel']}")
        elif p.stat().st_size != r["bytes"]:
            problems.append(f"lệch cỡ {r['rel']}")
        elif full and file_sha256(p) != r["sha256"]:
            problems.append(f"lệch sha256 {r['rel']}")
    return problems


def extract_set(name: str, scfg: dict, files_meta: list[dict], archive_dir: Path, dest: Path,
                force: bool = False) -> dict:
    """Kiểm sha256 từng file nén theo ``downloads.<bộ>.files``, giải nén phần ``include`` (trừ ``exclude``) vào
    ``dest``, ghi ``manifest.jsonl`` (rel, sha256, bytes, archive) + ``extract.json``. Đã giải nén cùng file nén và
    file còn đủ cỡ → dùng lại (``force`` để làm lại)."""
    expected = {f["name"]: f for f in files_meta}
    archives = []
    for a in scfg["archives"]:
        if a not in expected:
            raise KeyError(f"{name}: {a} chưa có sha256 trong downloads.{name}.files (configs/sources.yaml)")
        p = archive_dir / a
        if not p.is_file():
            raise FileNotFoundError(f"thiếu file nén {p}")
        sha = file_sha256(p)
        if sha != expected[a]["sha256"]:
            raise ValueError(f"{p}: sha256 {sha} khác configs/sources.yaml {expected[a]['sha256']}")
        archives.append({"name": a, "sha256": sha, "bytes": p.stat().st_size})
    info_p = dest / EXTRACT_INFO
    if info_p.exists() and not force:
        old = json.loads(info_p.read_text(encoding="utf-8"))
        if old.get("archives") == archives and not verify_manifest(dest, full=False):
            return old | {"reused": True}
    rows, seen = [], set()
    for a in archives:
        for mname, size, opener in _members(archive_dir / a["name"]):
            rel = safe_member(mname).as_posix()
            if not _wanted(rel, scfg):
                continue
            if rel in seen:
                raise ValueError(f"{name}: {rel} xuất hiện hai lần trong các file nén")
            seen.add(rel)
            out = dest / rel
            if out.is_file() and out.stat().st_size == size and not force:
                sha = file_sha256(out)                  # lần trước đã ghi xong (đổi tên nguyên tử) → băm lại
            else:
                with opener() as f:
                    sha, n = _write_hashed(f, out)
                if n != size:
                    raise ValueError(f"{name}: {rel} ghi {n} byte, file nén ghi {size}")
            rows.append({"rel": rel, "sha256": sha, "bytes": size, "archive": a["name"]})
    if not rows:
        raise ValueError(f"{name}: không file nào khớp include {scfg['include']}")
    rows.sort(key=lambda r: r["rel"])
    man = dest / EXTRACT_MANIFEST
    man.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8",
                   newline="\n")
    info = {"set": name, "archives": archives, "files": len(rows), "bytes": sum(r["bytes"] for r in rows),
            "manifest_sha256": file_sha256(man), "extracted": date.today().isoformat()}
    info_p.write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return info


def cmd_extract(cfg: dict, only: str | None, force: bool, verify: bool) -> int:
    ext = external_cfg(cfg)
    if ext is None:
        raise ValueError("configs/sources.yaml chưa có khối external.sets")
    archive_dir = paths.downloads_dir() / ext["archive_dir"]
    out = {}
    for name in external_sets(ext):
        if only and name != only:
            continue
        dest = external_root(ext) / name
        if verify:
            bad = verify_manifest(dest, full=True)
            out[name] = {"files": len(read_manifest(dest)), "problems": bad[:20], "n_problems": len(bad)}
            continue
        print(f"[sources] extract {name} → {dest}", flush=True)
        out[name] = extract_set(name, ext["sets"][name], cfg["downloads"][name]["files"], archive_dir, dest, force)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 1 if verify and any(v["n_problems"] for v in out.values()) else 0


# ----------------------------------------------------------------------------- nguồn ngoài: trình đọc

def parse_voc(text: str, classes: tuple[str, ...]) -> tuple[list, tuple[int, int] | None]:
    """VOC XML → (hộp xyxy của các lớp ``classes`` → người, (w, h) theo ``<size>`` nếu có). Lớp lạ → lỗi."""
    root = ElementTree.fromstring(text)
    boxes = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in classes:
            raise ValueError(f"VOC: lớp lạ {name!r} (nhận {classes})")
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height"))) if size is not None else None
    return boxes, wh


def load_heridal(set_dir: Path, scfg: dict) -> list[SourceItem]:
    """HERIDAL VOC: ảnh ``JPEGImages/*.jpg``, nhãn ``Annotations/<stem>.xml`` (thiếu XML = ảnh nền không người).
    Nhóm = mã nơi bay trong tên (``train_BRA_0001`` → ``BRA``); nhóm trong ``val_groups`` → val, còn lại train."""
    base = set_dir / scfg.get("subdir", "")
    img_dir, ann_dir = base / "JPEGImages", base / "Annotations"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"thiếu {img_dir} (chạy `python -m xla.degrade.sources extract --set heridal`)")
    classes = tuple(c.lower() for c in scfg.get("classes", ["person"]))
    val_groups = set(scfg.get("val_groups", []))
    items = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        m = HERIDAL_STEM_RE.match(img.stem)
        if not m:
            raise ValueError(f"HERIDAL: tên ảnh lạ {img.name} (chờ train|test_<NƠI>_<số>)")
        ann = ann_dir / f"{img.stem}.xml"
        boxes, wh_xml = parse_voc(ann.read_text(encoding="utf-8"), classes) if ann.exists() else ([], None)
        h, w = _image_hw(img)
        # 4 ảnh test_MED 4000×2250 có <size> 4000×3000 (XML mẫu) nhưng hộp đều nằm trong ảnh: tin cỡ ảnh thật;
        # hộp vượt ảnh thật thì nhãn không khớp ảnh → lỗi.
        if wh_xml is not None and wh_xml != (w, h) and any(b[2] > w or b[3] > h for b in boxes):
            raise ValueError(f"HERIDAL {img.name}: <size> {wh_xml} khác cỡ ảnh {(w, h)} và có hộp vượt ảnh")
        group = m.group(2)
        items.append(SourceItem(f"heridal/{img.stem}", "heridal", str(img), "val" if group in val_groups else "train",
                                (w, h), boxes, [], group=group))
    return items


def load_tinyperson(set_dir: Path, scfg: dict) -> list[SourceItem]:
    """TinyPerson COCO ``tiny_set_<k>.json``, ảnh ở ``<k>/<file_name>``; ``splits`` ánh xạ k → train/val.
    ``sea_person``/``earth_person`` → người; hộp có cờ trong ``ignore_flags`` → vùng bỏ qua. Nhóm = video
    (``bb_V0032``), ảnh rời (bing…) là nhóm riêng."""
    flags = tuple(scfg.get("ignore_flags", ["ignore", "uncertain"]))
    items, keys = [], set()
    for k, split in scfg["splits"].items():
        jp = set_dir / f"tiny_set_{k}.json"
        if not jp.exists():
            raise FileNotFoundError(f"thiếu {jp} (chạy `python -m xla.degrade.sources extract --set tinyperson`)")
        data = json.loads(jp.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        bad = sorted(set(cats.values()) - set(TINYPERSON_PERSON))
        if bad:
            raise ValueError(f"{jp.name}: lớp lạ {bad}")
        anns: dict[int, list] = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            path = set_dir / k / im["file_name"]
            if not path.is_file():
                raise FileNotFoundError(f"TinyPerson: thiếu ảnh {path}")
            boxes, ignore = [], []
            for a in anns.get(im["id"], []):
                x, y, bw, bh = (float(v) for v in a["bbox"])
                if bw <= 0 or bh <= 0:
                    continue
                (ignore if any(a.get(f) for f in flags) else boxes).append([x, y, x + bw, y + bh])
            stem = Path(im["file_name"]).stem
            key = f"tinyperson/{stem}"
            if key in keys:
                raise ValueError(f"TinyPerson: trùng tên ảnh {stem}")
            keys.add(key)
            m = TINYPERSON_GROUP_RE.match(stem)
            items.append(SourceItem(key, "tinyperson", str(path), split, (int(im["width"]), int(im["height"])), boxes,
                                    ignore, group=m.group(1) if m else stem))
    return items


def parse_yolo(text: str, wh: tuple[int, int]) -> list:
    """Nhãn YOLO chuẩn hoá ``c cx cy w h`` → hộp xyxy px; lớp nào cũng là người. Dòng sai định dạng → lỗi."""
    w, h = wh
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"YOLO: dòng sai định dạng {line!r}")
        int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        if bw > 0 and bh > 0:
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def load_wisard(set_dir: Path, scfg: dict) -> list[SourceItem]:
    """WiSARD: chỉ thư mục ``*_VIS_*`` (RGB), mỗi ảnh ``.jpeg`` một ``.txt`` YOLO cùng tên; ``count.txt`` bỏ qua.
    Nhóm = thư mục (một chuyến bay)."""
    dirs = sorted(d for d in set_dir.iterdir() if d.is_dir() and "_VIS_" in d.name) if set_dir.is_dir() else []
    if not dirs:
        raise FileNotFoundError(f"thiếu thư mục *_VIS_* trong {set_dir} (chạy `... sources extract --set wisard`)")
    items = []
    for d in dirs:
        for img in sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            lab = img.with_suffix(".txt")
            if not lab.exists():
                raise FileNotFoundError(f"WiSARD: ảnh {img.name} thiếu nhãn {lab.name}")
            h, w = _image_hw(img)
            items.append(SourceItem(f"wisard/{img.stem}", "wisard", str(img), scfg.get("split", "train"), (w, h),
                                    parse_yolo(lab.read_text(encoding="utf-8"), (w, h)), [], group=d.name))
    return items


EXTERNAL_LOADERS = {"heridal": load_heridal, "tinyperson": load_tinyperson, "wisard": load_wisard}


def check_group_split(items: list[SourceItem]) -> None:
    """Một nhóm (chuỗi/cảnh) chỉ được nằm ở một split; rò → lỗi."""
    seen: dict[tuple[str, str], set] = {}
    for it in items:
        if it.group is not None:
            seen.setdefault((it.source, it.group), set()).add(it.split)
    leak = sorted(f"{s}/{g}" for (s, g), sp in seen.items() if len(sp) > 1)
    if leak:
        raise ValueError(f"nhóm nằm ở cả train lẫn val: {leak[:10]}")


def collect_external(cfg: dict, only: str | None = None) -> list[SourceItem]:
    """Ảnh nguồn ngoài (mức ẢNH, chưa cắt ô) của các bộ bật trong ``external.sets``."""
    ext = external_cfg(cfg)
    if ext is None:
        return []
    items = []
    for name in external_sets(ext):
        if only is None or name == only:
            items += EXTERNAL_LOADERS[name](external_root(ext) / name, ext["sets"][name])
    check_group_split(items)
    return items


# ----------------------------------------------------------------------------- nguồn ngoài: cắt ô HERIDAL

def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Gốc các ô dọc một trục: bước ``tile − overlap``, ô cuối áp sát mép (phủ hết ảnh)."""
    if length <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError(f"overlap {overlap} ≥ tile {tile}")
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_item(item: SourceItem, tcfg: dict) -> list[SourceItem]:
    """Cắt ảnh lớn thành ô ``w×h`` chồng mép ``overlap``. Trong mỗi ô: hộp còn ≥ ``min_keep_area_frac`` diện tích →
    người; còn ít hơn (nhưng > 0) → vùng bỏ qua (người bị cắt cụt không làm dương cũng không làm âm). Mỗi hộp thuộc
    đúng một ô "chủ" (tâm ô gần tâm hộp nhất); chỉ giữ ô làm chủ ít nhất một hộp, cộng ``neg_tiles_per_image`` ô
    không chạm người nào (chọn tất định theo khoá ảnh)."""
    w, h = item.wh
    tw, th = min(int(tcfg["w"]), w), min(int(tcfg["h"]), h)
    ov = int(tcfg["overlap"])
    keep_frac = float(tcfg["min_keep_area_frac"])
    tiles = [(x, y, x + tw, y + th) for y in tile_starts(h, th, ov) for x in tile_starts(w, tw, ov)]
    boxes, ignore = item.boxes_np(), item.ignore_np()
    t = np.asarray(tiles, np.float64)
    owner = np.zeros(len(boxes), int)
    if len(boxes):
        bc = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        tc = np.stack([(t[:, 0] + t[:, 2]) / 2, (t[:, 1] + t[:, 3]) / 2], 1)
        owner = ((bc[:, None, :] - tc[None]) ** 2).sum(-1).argmin(1)
    out, empty = [], []
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        kept, cut = [], []
        for b in boxes:
            c = [max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)]
            vis = max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])
            if vis <= 0:
                continue
            rel = [c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0]
            (kept if vis >= keep_frac * (b[2] - b[0]) * (b[3] - b[1]) else cut).append(rel)
        ign = [[max(g[0], x0) - x0, max(g[1], y0) - y0, min(g[2], x1) - x0, min(g[3], y1) - y0]
               for g in ignore if g[0] < x1 and g[2] > x0 and g[1] < y1 and g[3] > y0]
        sub = SourceItem(f"{item.key}_t{ti:02d}", item.source, item.image, item.split, (tw, th),
                         [[float(v) for v in b] for b in kept], [[float(v) for v in b] for b in cut + ign],
                         item.alt_m, item.sky_frac, item.clip_nadir, item.probe_nadir, item.group, [x0, y0, x1, y1])
        if bool(np.any(owner == ti)) and len(boxes):
            out.append(sub)
        elif not kept and not cut:
            empty.append(sub)
    k = min(int(tcfg.get("neg_tiles_per_image", 0)), len(empty))
    if k:
        rng = np.random.default_rng(zlib.crc32(item.key.encode("utf-8")))
        out += [empty[i] for i in sorted(rng.choice(len(empty), size=k, replace=False))]
    return sorted(out, key=lambda s: s.key)


# ----------------------------------------------------------------------------- nguồn ngoài: lọc góc nhìn

def load_external_view_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """``configs/view_external.yaml``: ``groups`` ``"<bộ>/<nhóm>"`` → lớp (có ``mixed``), ``images``
    ``"<bộ>/<stem>"`` → lớp (nadir/near/oblique; soát tay, thắng nhãn nhóm và probe)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = {"groups": {str(k): str(v) for k, v in (data.get("groups") or {}).items()},
           "images": {str(k): str(v) for k, v in (data.get("images") or {}).items()}}
    bad = {k: v for k, v in out["groups"].items() if v not in VIEW_CLASSES}
    bad |= {k: v for k, v in out["images"].items() if v not in VIEW_CLASSES[:3]}
    if bad:
        raise ValueError(f"{path}: nhãn không hợp lệ: {bad}")
    return out


def external_view_labels(ext: dict) -> dict[str, dict[str, str]]:
    f = ext.get("view_labels_file")
    if not f:
        return {"groups": {}, "images": {}}
    p = Path(f)
    return load_external_view_labels(p if p.is_absolute() else paths.repo_root() / p)


def external_view(item: SourceItem, labels: dict) -> str | None:
    """Nhãn soát tay của ảnh ngoài: nhãn ảnh, rồi nhãn nhóm; không có → ``None`` (đi qua probe)."""
    if item.key in labels["images"]:
        return labels["images"][item.key]
    return labels["groups"].get(f"{item.source}/{item.group}")


def is_nadir_external(item: SourceItem, ext: dict, labels: dict) -> bool:
    """Nhãn soát tay (ảnh → nhóm) khác ``mixed`` quyết; còn lại probe CLIP ≥ ``min_probe_nadir`` (cùng ngưỡng
    VisDrone). Thiếu điểm probe → lỗi."""
    v = external_view(item, labels)
    if v is not None and v != "mixed":
        return v in ext["accept_views"]
    if item.probe_nadir is None:
        raise ValueError(f"{item.key}: chưa có điểm probe (chạy `python -m xla.degrade.sources probe-external`)")
    return item.probe_nadir >= float(ext["min_probe_nadir"])


def external_probe_path(ext: dict) -> Path:
    return external_root(ext) / ext["probe_scores_file"]


def attach_external_probe(items: list[SourceItem], ext: dict) -> None:
    p = external_probe_path(ext)
    if not p.exists():
        return
    sc = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            sc[r["key"]] = r["probe_nadir"]
    for it in items:
        it.probe_nadir = sc.get(it.key)


def select_external(cfg: dict, for_d02x: bool = True) -> list[SourceItem]:
    """Nguồn ngoài đã lọc góc nhìn, bộ có ``tile`` thì cắt ô (ô kế thừa split/nhóm/điểm của ảnh). ``for_d02x``:
    chỉ bộ có ``d02x: true`` (mặc định)."""
    ext = external_cfg(cfg)
    if ext is None:
        return []
    items = collect_external(cfg)
    attach_external_probe(items, ext)
    labels = external_view_labels(ext)
    out = []
    for it in items:
        scfg = ext["sets"][it.source]
        if for_d02x and not scfg.get("d02x", True):
            continue
        if not is_nadir_external(it, ext, labels):
            continue
        out += tile_item(it, scfg["tile"]) if scfg.get("tile") else [it]
    return sorted(out, key=lambda i: i.key)


def windows_by_source(cfg: dict) -> dict[str, int]:
    """Số cửa sổ mỗi ảnh/ô theo nguồn ngoài (``external.sets.<bộ>.windows_per_image``)."""
    ext = external_cfg(cfg)
    if ext is None:
        return {}
    return {n: int(s["windows_per_image"]) for n, s in ext["sets"].items() if s.get("windows_per_image") is not None}


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


CLIP_DECODE_THREADS = 4                  # ≤ 4 worker (máy lật bit RAM dưới tải)
CLIP_SHORT_SIDE = 224                    # cạnh ngắn đầu vào CLIP ViT-B-32 (open_clip Resize(224) bicubic + CenterCrop)


def _clip_thumb(path: str):
    """Ảnh cho CLIP: giải mã 1/2 rồi thu cạnh ngắn về 224 bằng PIL bicubic — cùng phép Resize đầu tiên của
    preprocess open_clip, nên preprocess sau đó chỉ còn cắt giữa + chuẩn hoá."""
    from PIL import Image

    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if im is None:
        raise RuntimeError(f"không đọc được {path}")
    pil = Image.fromarray(np.ascontiguousarray(im[..., ::-1]))
    w, h = pil.size
    s = CLIP_SHORT_SIDE / min(w, h)
    return pil.resize((max(CLIP_SHORT_SIDE, int(w * s)), max(CLIP_SHORT_SIDE, int(h * s))), Image.BICUBIC)


def clip_embeddings(items: list[SourceItem], model_name: str, pretrained: str, cache: Path,
                    batch: int = 64) -> dict[str, np.ndarray]:
    """Đặc trưng ảnh CLIP (chuẩn hoá L2), cache ``.npz`` theo khoá; chỉ tính ảnh chưa có."""
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["keys"].tolist(), z["emb"], strict=True))
    todo = [it for it in items if it.key not in have]
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import nullcontext

        import torch

        from xla.gpu_lock import gpu_lock

        # Giải mã + thu nhỏ TRƯỚC khi giữ khoá GPU (4 luồng; cv2 nhả GIL): JPEG 12 MP của HERIDAL ~1,7 s/ảnh, không
        # để lane khác chờ khoá trong lúc CPU giải mã.
        with ThreadPoolExecutor(CLIP_DECODE_THREADS) as ex:
            thumbs = list(ex.map(lambda it: _clip_thumb(it.image), todo))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with gpu_lock("degrade:clip-emb", timeout_s=GPU_LOCK_TIMEOUT_S) if device == "cuda" else nullcontext():
            scorer = ClipScorer(model_name, pretrained, device)
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                with torch.no_grad():
                    b = torch.stack([scorer.preprocess(t) for t in thumbs[i:i + batch]]).to(device)
                    f = scorer.model.encode_image(b).float()
                    f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
                have.update(zip((it.key for it in chunk), f, strict=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(have)
        np.savez(cache, keys=np.array(keys), emb=np.stack([have[k] for k in keys]))
    return have


def _probe_training(cfg: dict, cache: Path) -> tuple:
    """Dữ liệu huấn luyện probe góc nhìn (VisDrone): nhãn chuỗi không ``mixed`` + nhãn từng ảnh, dương =
    ``accept_views``. Trả (emb, accept, seq_tr, img_tr, xs, ys, xi, yi)."""
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
    return emb, accept, seq_tr, img_tr, xs, ys, xi, yi


def cmd_probe_external(cfg: dict, cache_vis: Path, cache_ext: Path) -> int:
    """Probe góc nhìn đã fit (cùng dữ liệu + siêu tham số ``fit-probe`` của VisDrone, không thêm nhãn ngoài) chấm ảnh
    nguồn ngoài → ``external.probe_scores_file``. Đặc trưng CLIP ảnh ngoài cache riêng (GPU trong ``gpu_lock``)."""
    from sklearn.linear_model import LogisticRegression

    ext = external_cfg(cfg)
    if ext is None:
        raise ValueError("configs/sources.yaml chưa có khối external.sets")
    nd = cfg["nadir"]
    items = collect_external(cfg)
    emb = clip_embeddings(items, nd["clip_model"], nd["clip_pretrained"], cache_ext)
    _, accept, _, _, xs, ys, xi, yi = _probe_training(cfg, cache_vis)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    clf.fit(np.vstack([xs, xi]), np.concatenate([ys, yi]))
    prob = clf.predict_proba(np.stack([emb[it.key] for it in items]))[:, 1]
    out = external_probe_path(ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it, pr in sorted(zip(items, prob, strict=True), key=lambda t: t[0].key):
            f.write(json.dumps({"key": it.key, "probe_nadir": round(float(pr), 5)}) + "\n")
    t = float(ext["min_probe_nadir"])
    per = {}
    for it, pr in zip(items, prob, strict=True):
        d = per.setdefault(it.source, {"images": 0, "pass": 0})
        d["images"] += 1
        d["pass"] += bool(pr >= t)
    print(json.dumps({"accept_views": sorted(accept), "threshold": t, "per_set": per, "out": str(out)},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_fit_probe(cfg: dict, cache: Path) -> int:
    """Probe tuyến tính (logistic, C=1, cân lớp) trên đặc trưng CLIP: nhãn = lớp soát tay (chuỗi nadir/near/oblique +
    nhãn từng ảnh), dương = ``accept_views``. Chấm mọi ảnh VisDrone → ``probe_scores_file``; in kiểm chéo 5 phần trên
    ảnh có nhãn riêng (ảnh chuỗi mixed) để chọn ``min_probe_nadir``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    nd = cfg["nadir"]
    emb, accept, seq_tr, img_tr, xs, ys, xi, yi = _probe_training(cfg, cache)
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


def _read_thumb(path: str, tw: int) -> np.ndarray:
    """Đọc ảnh đã giảm cỡ lúc giải mã JPEG (1/2, 1/4, 1/8) sao cho còn rộng ≥ ``tw``."""
    from PIL import Image

    with Image.open(path) as im:
        w = im.width
    flag = cv2.IMREAD_COLOR
    for f, k in ((cv2.IMREAD_REDUCED_COLOR_8, 8), (cv2.IMREAD_REDUCED_COLOR_4, 4), (cv2.IMREAD_REDUCED_COLOR_2, 2)):
        if w / k >= tw:
            flag = f
            break
    im = cv2.imread(path, flag)
    if im is None:
        raise RuntimeError(f"không đọc được {path}")
    return im


def cmd_external_sheet(cfg: dict, source: str, out_dir: Path, per_sheet: int, n: int, seed: int) -> int:
    """Lưới soát tay góc nhìn một bộ ngoài: ``n`` ảnh ngẫu nhiên (seed), ô 360×270 ghi số thứ tự, điểm probe, nhóm.
    ``review.csv`` ghi idx → khoá để chép nhãn soát vào ``configs/view_external.yaml`` (``images``)."""
    ext = external_cfg(cfg)
    if ext is None:
        raise ValueError("configs/sources.yaml chưa có khối external.sets")
    items = collect_external(cfg, only=source)
    attach_external_probe(items, ext)
    rng = np.random.default_rng(seed)
    pick = [items[i] for i in sorted(rng.choice(len(items), size=min(n, len(items)), replace=False))]
    out_dir.mkdir(parents=True, exist_ok=True)
    tw, th, cols = 360, 270, 5
    rows = ["idx,key,group,split,n_person,probe_nadir"]
    for s in range(0, len(pick), per_sheet):
        tiles = []
        for j, it in enumerate(pick[s:s + per_sheet]):
            idx = s + j
            t = cv2.resize(_read_thumb(it.image, tw), (tw, th), interpolation=cv2.INTER_AREA)
            pr = "-" if it.probe_nadir is None else f"{it.probe_nadir:.2f}"
            cv2.rectangle(t, (0, 0), (tw, 20), (0, 0, 0), -1)
            cv2.putText(t, f"#{idx} p{pr} {it.group}"[:40], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
            rows.append(f"{idx},{it.key},{it.group},{it.split},{len(it.boxes)},{pr}")
        while len(tiles) % cols:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
        cv2.imwrite(str(out_dir / f"{source}_{s // per_sheet:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out_dir / f"{source}_review.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"[sources] {source}: {len(pick)} ảnh → {out_dir}")
    return 0


def cmd_seq_sheet(cfg: dict, out_dir: Path, per_sheet: int, source: str = "visdrone", n: int = 100,
                  seed: int = 20261125) -> int:
    """Lưới soát nhãn góc nhìn theo chuỗi VisDrone: mỗi hàng một chuỗi, 3 khung (đầu / giữa / cuối) có ghi mã chuỗi.
    ``source`` là bộ ngoài → lưới ``n`` ảnh ngẫu nhiên của bộ đó (``cmd_external_sheet``)."""
    if source != "visdrone":
        return cmd_external_sheet(cfg, source, out_dir, per_sheet, n, seed)
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
    if external_cfg(cfg) is not None:
        st["external"] = external_stats(cfg)
    print(json.dumps(st, ensure_ascii=False, indent=None if as_json else 2))
    return 0


def _quant(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, np.float64)
    return {"n": len(vals), **{f"p{q}": round(float(np.percentile(a, q)), 1) for q in (5, 25, 50, 75, 95)},
            "lt_8px": int((a < 8).sum()), "lt_16px": int((a < 16).sum())}


def _short_sides(items: list[SourceItem]) -> list[float]:
    return [float(min(b[2] - b[0], b[3] - b[1])) for it in items for b in it.boxes]


def _counts(items: list[SourceItem]) -> dict:
    return {"images": len(items), "with_person": sum(bool(i.boxes) for i in items),
            "persons": sum(len(i.boxes) for i in items), "ignore_regions": sum(len(i.ignore) for i in items),
            "groups": len({i.group for i in items}),
            "splits": {sp: sum(i.split == sp for i in items) for sp in ("train", "val")}}


def review_precision(items: list[SourceItem], ext: dict, labels: dict) -> dict:
    """Độ chính xác của probe trên mẫu soát tay (``images`` của view_external.yaml): dương = lớp ``accept_views``,
    chọn = probe ≥ ``min_probe_nadir``."""
    rev = [it for it in items if it.key in labels["images"] and it.probe_nadir is not None]
    if not rev:
        return {"n": 0}
    acc, t = set(ext["accept_views"]), float(ext["min_probe_nadir"])
    pos = np.array([labels["images"][it.key] in acc for it in rev])
    sel = np.array([it.probe_nadir >= t for it in rev])
    tp = int((pos & sel).sum())
    lab: dict[str, int] = {}
    for it in rev:
        lab[labels["images"][it.key]] = lab.get(labels["images"][it.key], 0) + 1
    return {"n": len(rev), "labels": dict(sorted(lab.items())), "accepted_by_hand": int(pos.sum()),
            "selected_by_probe": int(sel.sum()), "true_pos": tp,
            "precision": None if not sel.any() else round(tp / int(sel.sum()), 3),
            "recall": None if not pos.any() else round(tp / int(pos.sum()), 3)}


def external_stats(cfg: dict) -> dict:
    """Khối ``external`` của ``stats``: số ảnh/hộp trước–sau lọc, cạnh ngắn hộp (px nguồn; sau cắt ô với bộ có
    ``tile``), precision probe trên mẫu soát tay. Bộ chưa giải nén / chưa chấm probe ghi rõ trạng thái."""
    ext = external_cfg(cfg)
    labels = external_view_labels(ext)
    out: dict = {}
    for name in external_sets(ext):
        scfg = ext["sets"][name]
        try:
            items = EXTERNAL_LOADERS[name](external_root(ext) / name, scfg)
        except FileNotFoundError as e:
            out[name] = {"status": "not_extracted", "error": str(e)}
            continue
        attach_external_probe(items, ext)
        d = {"use": scfg.get("use"), "in_d02x": bool(scfg.get("d02x", True)), "before_filter": _counts(items),
             "box_short_px": _quant(_short_sides(items)), "probe_scored": sum(i.probe_nadir is not None for i in items)}
        ready = all(i.probe_nadir is not None or external_view(i, labels) not in (None, "mixed") for i in items)
        if ready:
            kept = [i for i in items if is_nadir_external(i, ext, labels)]
            d["after_filter"] = _counts(kept)
            d["after_filter"]["box_short_px"] = _quant(_short_sides(kept))
            if scfg.get("tile"):
                tiles = [t for i in kept for t in tile_item(i, scfg["tile"])]
                d["tiles"] = _counts(tiles) | {"negative_tiles": sum(not t.boxes for t in tiles),
                                               "box_short_px": _quant(_short_sides(tiles))}
        else:
            d["after_filter"] = {"status": "probe_missing"}
        d["review"] = review_precision(items, ext, labels)
        out[name] = d
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "review-sheet", "seq-sheet", "fit-probe", "pick-val", "stats", "extract",
                                    "probe-external"])
    ap.add_argument("--config", default=str(paths.repo_root() / "configs" / "sources.yaml"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--per-sheet", type=int, default=25)
    ap.add_argument("--out", type=Path, default=paths.cache_dir() / "degrade" / "nadir_review")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-val", type=int, default=200, help="pick-val: số ảnh có người tối thiểu của val")
    ap.add_argument("--max-extra", type=int, default=180, help="pick-val: ảnh có người tối đa chuyển thêm")
    ap.add_argument("--set", default=None, choices=EXTERNAL_SETS, help="extract: chỉ một bộ ngoài")
    ap.add_argument("--force", action="store_true", help="extract: giải nén lại dù đã có")
    ap.add_argument("--verify", action="store_true", help="extract: chỉ băm lại file trên đĩa, so với manifest")
    ap.add_argument("--source", default="visdrone", choices=("visdrone", *EXTERNAL_SETS),
                    help="seq-sheet: bộ ngoài → lưới --n ảnh ngẫu nhiên (--seed) để soát tay")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "extract":
        return cmd_extract(cfg, a.set, a.force, a.verify)
    if a.cmd == "probe-external":
        emb_dir = paths.cache_dir() / "degrade"
        return cmd_probe_external(cfg, emb_dir / f"clip_emb_{cfg['nadir']['clip_model']}.npz",
                                  emb_dir / cfg["external"]["clip_cache"])
    if a.cmd == "score":
        return cmd_score(cfg, a.batch, a.device)
    if a.cmd == "review-sheet":
        return cmd_review_sheet(cfg, a.n, a.seed, a.out, a.per_sheet)
    if a.cmd == "seq-sheet":
        return cmd_seq_sheet(cfg, a.out, a.per_sheet, a.source, a.n, a.seed)
    if a.cmd == "pick-val":
        return cmd_pick_val(cfg, a.seed, a.min_val, a.max_extra, paths.repo_root() / "configs" / "visdrone_valsplit.yaml")
    if a.cmd == "fit-probe":
        return cmd_fit_probe(cfg, paths.cache_dir() / "degrade" / f"clip_emb_{cfg['nadir']['clip_model']}.npz")
    return cmd_stats(cfg, a.json)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SourceItem", "asdict", "collect", "collect_external", "is_nadir", "is_nadir_external",
           "parse_visdrone_annotation", "select", "select_external", "sky_fraction", "tile_item"]
