"""Nguồn ngoài có nhãn người (AI v2 phase 02, khối ``external`` của ``configs/sources.yaml``): giải nén có kiểm
sha256 + manifest, trình đọc HERIDAL (VOC XML), TinyPerson (COCO), WiSARD VIS (YOLO), cắt ô HERIDAL, lọc góc nhìn
(nhãn soát tay ``configs/view_external.yaml`` rồi probe CLIP cùng ngưỡng VisDrone). Chỉ vào tập trộn ``d02x``.

Tách khỏi ``xla.degrade.sources`` (26/11/2026, ``results/v2/data/README.md`` §5 mục 6), không đổi hành vi. Lệnh CLI và
mọi tên ở đây vẫn gọi được qua ``xla.degrade.sources`` (tên được nhập lại ở đó). ``SourceItem`` / ``_image_hw`` /
``VIEW_CLASSES`` nhập trong hàm vì ``sources`` nhập module này lúc nạp (tránh vòng nhập).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import tarfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import numpy as np
import yaml

from xla import paths

if TYPE_CHECKING:
    from xla.degrade.sources import SourceItem

# ----------------------------------------------------------------------------- nguồn ngoài: giải nén + manifest

EXTERNAL_SETS = ("heridal", "tinyperson", "wisard")
HERIDAL_STEM_RE = re.compile(r"^(train|test)_([A-Z]+)_\d+$")     # train_BRA_0001: mã nơi bay = cảnh
TINYPERSON_GROUP_RE = re.compile(r"^(.+?_V\d+)_I\d+$")            # bb_V0032_I0001640: khung của video V0032
TINYPERSON_PERSON = ("sea_person", "earth_person")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
EXTRACT_MANIFEST = "manifest.jsonl"
EXTRACT_INFO = "extract.json"
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
    from xla.degrade.sources import SourceItem, _image_hw

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
    from xla.degrade.sources import SourceItem

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
    from xla.degrade.sources import SourceItem, _image_hw

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
    from xla.degrade.sources import SourceItem

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
    from xla.degrade.sources import VIEW_CLASSES

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
