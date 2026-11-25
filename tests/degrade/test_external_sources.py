"""AI v2 phase 02 Accept: nguồn ngoài (HERIDAL, TinyPerson, WiSARD) + tập trộn ``d02x``.

Phần CI: trình đọc từng bộ trên nhãn mẫu ``tests/degrade/fixtures/external/`` (ảnh tổng hợp sinh vào thư mục tạm),
map lớp, vùng bỏ qua, cắt ô, giải nén an toàn + manifest sha256, lọc góc nhìn, dựng ``d02x`` nhỏ, không rò nhóm,
không ảnh ngoài nào trong git. Phần ``data``: dữ liệu thật trên ổ D (manifest khớp, ``d02x`` có nguồn ngoài).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from xla.degrade import build, sources

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent / "fixtures" / "external"
SEED = 20261125


def _img(path: Path, w: int, h: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(path.suffix, im)
    assert ok
    path.write_bytes(buf.tobytes())


@pytest.fixture
def ext_tree(tmp_path: Path) -> Path:
    """Cây nguồn ngoài giả: nhãn chép từ fixtures, ảnh tổng hợp sinh theo tên/cỡ trong nhãn."""
    root = tmp_path / "external"
    shutil.copytree(FIX, root, ignore=shutil.ignore_patterns("README.md"))
    jpg = root / "heridal" / "heridal_keras_retinanet_voc" / "JPEGImages"
    _img(jpg / "train_AAA_0001.jpg", 2400, 1800, 1)
    _img(jpg / "train_AAA_0002.jpg", 2400, 1800, 2)          # không XML = ảnh nền
    _img(jpg / "test_BBB_0001.jpg", 2400, 1800, 3)           # XML không có <size>
    for k in ("train", "test"):
        data = json.loads((root / "tinyperson" / f"tiny_set_{k}.json").read_text(encoding="utf-8"))
        for i, im in enumerate(data["images"]):
            _img(root / "tinyperson" / k / im["file_name"], im["width"], im["height"], 10 + i)
    for d in (root / "wisard").iterdir():
        for t in d.glob("*_0000000?.txt"):
            _img(t.with_suffix(".jpeg"), 1280, 720, 20)
    return root


def _cfg(root: Path, tmp_path: Path, labels: dict | None = None) -> dict:
    cfg = yaml.safe_load((REPO / "configs" / "sources.yaml").read_text(encoding="utf-8"))
    ext = cfg["external"]
    ext["root"] = str(root)
    ext["sets"]["heridal"]["val_groups"] = ["BBB"]
    lab = tmp_path / "view_external.yaml"
    lab.write_text(yaml.safe_dump(labels or {"groups": {}, "images": {}}), encoding="utf-8")
    ext["view_labels_file"] = str(lab)
    return cfg


def _write_probe(cfg: dict, scores: dict[str, float]) -> None:
    p = sources.external_probe_path(cfg["external"])
    p.write_text("".join(json.dumps({"key": k, "probe_nadir": v}) + "\n" for k, v in scores.items()), encoding="utf-8")


# ----------------------------------------------------------------------------- trình đọc từng bộ

def test_heridal_voc_person_groups_and_background(ext_tree: Path, tmp_path: Path) -> None:
    scfg = _cfg(ext_tree, tmp_path)["external"]["sets"]["heridal"]
    items = {it.key: it for it in sources.load_heridal(ext_tree / "heridal", scfg)}
    assert set(items) == {"heridal/train_AAA_0001", "heridal/train_AAA_0002", "heridal/test_BBB_0001"}
    a = items["heridal/train_AAA_0001"]
    assert a.boxes == [[100, 100, 160, 180], [1200, 500, 1260, 560], [1250, 100, 1330, 160]]  # "Person" → person
    assert (a.source, a.group, a.split, a.wh) == ("heridal", "AAA", "train", (2400, 1800))
    assert items["heridal/train_AAA_0002"].boxes == []                     # ảnh nền: lớp âm
    b = items["heridal/test_BBB_0001"]
    assert (b.group, b.split, len(b.boxes)) == ("BBB", "val", 1)            # nhóm val theo nơi bay
    with pytest.raises(ValueError, match="lớp lạ"):
        sources.parse_voc("<annotation><object><name>car</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
                          "<xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>", ("person",))


def test_heridal_size_mismatch_only_fails_when_boxes_leave_image(ext_tree: Path, tmp_path: Path) -> None:
    scfg = _cfg(ext_tree, tmp_path)["external"]["sets"]["heridal"]
    ann = ext_tree / "heridal" / "heridal_keras_retinanet_voc" / "Annotations" / "train_AAA_0001.xml"
    _img(ann.parent.parent / "JPEGImages" / "train_AAA_0001.jpg", 2400, 1500, 1)   # ảnh thấp hơn <size>, hộp vẫn trong
    assert len(sources.load_heridal(ext_tree / "heridal", scfg)) == 3
    ann.write_text(ann.read_text(encoding="utf-8").replace("<ymax>560</ymax>", "<ymax>1600</ymax>"), encoding="utf-8")
    with pytest.raises(ValueError, match="vượt ảnh"):
        sources.load_heridal(ext_tree / "heridal", scfg)


def test_tinyperson_classes_ignore_flags_and_video_groups(ext_tree: Path, tmp_path: Path) -> None:
    scfg = _cfg(ext_tree, tmp_path)["external"]["sets"]["tinyperson"]
    items = {it.key: it for it in sources.load_tinyperson(ext_tree / "tinyperson", scfg)}
    a = items["tinyperson/bb_V0001_I0000010"]
    assert a.boxes == [[100, 200, 110, 220]]                               # earth_person
    assert a.ignore == [[300, 210, 308, 218], [1000, 10, 1200, 60]]        # uncertain, ignore+logo → bỏ qua
    assert (a.group, a.split, a.wh) == ("bb_V0001", "train", (1280, 720))
    assert items["tinyperson/bb_V0001_I0000020"].boxes == [[500, 400, 512, 424]]   # sea_person; hộp rộng 0 bỏ
    bg = items["tinyperson/bing_P000_123_bg"]
    assert (bg.boxes, bg.group) == ([], "bing_P000_123_bg")               # ảnh rời là nhóm riêng
    t = items["tinyperson/youtube_V0002_I0000005"]
    assert (t.split, t.group, len(t.boxes)) == ("val", "youtube_V0002", 1)  # json test → val


def test_tinyperson_unknown_category_fails(ext_tree: Path, tmp_path: Path) -> None:
    scfg = _cfg(ext_tree, tmp_path)["external"]["sets"]["tinyperson"]
    jp = ext_tree / "tinyperson" / "tiny_set_test.json"
    data = json.loads(jp.read_text(encoding="utf-8"))
    data["categories"].append({"id": 3, "name": "boat"})
    jp.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="boat"):
        sources.load_tinyperson(ext_tree / "tinyperson", scfg)


def test_wisard_vis_only_yolo_any_class_is_person(ext_tree: Path, tmp_path: Path) -> None:
    scfg = _cfg(ext_tree, tmp_path)["external"]["sets"]["wisard"]
    items = sources.load_wisard(ext_tree / "wisard", scfg)
    assert [it.key for it in items] == ["wisard/210417_Fixture_Site_VIS_0001_00000001",
                                        "wisard/210417_Fixture_Site_VIS_0001_00000002"]   # IR + count.txt bỏ
    a = items[0]
    np.testing.assert_allclose(a.boxes, [[576, 288, 704, 432], [288, 504, 352, 576]])  # lớp 3 cũng là người
    assert (a.group, a.split) == ("210417_Fixture_Site_VIS_0001", "train")
    assert items[1].boxes == []
    with pytest.raises(ValueError, match="sai định dạng"):
        sources.parse_yolo("0 0.5 0.5 0.1\n", (100, 100))


def test_group_split_leak_is_an_error() -> None:
    a = sources.SourceItem("tinyperson/x1", "tinyperson", "x", "train", (10, 10), group="bb_V0001")
    b = sources.SourceItem("tinyperson/x2", "tinyperson", "x", "val", (10, 10), group="bb_V0001")
    c = sources.SourceItem("wisard/x3", "wisard", "x", "val", (10, 10), group="bb_V0001")   # bộ khác: không tính
    sources.check_group_split([a, c])
    with pytest.raises(ValueError, match="bb_V0001"):
        sources.check_group_split([a, b])


# ----------------------------------------------------------------------------- cắt ô HERIDAL

def test_tile_starts_cover_axis_with_overlap() -> None:
    assert sources.tile_starts(2400, 1280, 160) == [0, 1120]
    assert sources.tile_starts(1800, 960, 160) == [0, 800, 840]
    assert sources.tile_starts(4000, 1280, 160) == [0, 1120, 2240, 2720]
    assert sources.tile_starts(900, 960, 160) == [0]


def test_tiles_keep_half_area_boxes_cut_rest_and_own_each_box_once(ext_tree: Path, tmp_path: Path) -> None:
    scfg = _cfg(ext_tree, tmp_path)["external"]["sets"]["heridal"]
    item = next(i for i in sources.load_heridal(ext_tree / "heridal", scfg) if i.key.endswith("AAA_0001"))
    tiles = sources.tile_item(item, scfg["tile"])
    by = {t.key: t for t in tiles}
    t0, t1 = by["heridal/train_AAA_0001_t00"], by["heridal/train_AAA_0001_t01"]
    assert (t0.crop, t0.wh, t1.crop) == ([0, 0, 1280, 960], (1280, 960), [1120, 0, 2400, 960])
    assert t0.boxes == [[100, 100, 160, 180], [1200, 500, 1260, 560]]
    assert t0.ignore == [[1250, 100, 1280, 160]]          # còn 30/80 = 37,5 % < 50 % → vùng bỏ qua, không dương
    assert t1.boxes == [[80, 500, 140, 560], [130, 100, 210, 160]]
    pos = [t for t in tiles if t.boxes]
    neg = [t for t in tiles if not t.boxes]
    assert len(pos) == 2 and len(neg) == 1 and not neg[0].ignore     # 1 ô nền (neg_tiles_per_image)
    assert neg[0].crop[1] >= 800                                      # ô nền không chạm người nào
    assert all((t.split, t.group, t.image) == (item.split, item.group, item.image) for t in tiles)
    assert sources.tile_item(item, scfg["tile"]) == tiles             # tất định
    bg = next(i for i in sources.load_heridal(ext_tree / "heridal", scfg) if i.key.endswith("AAA_0002"))
    assert [len(t.boxes) for t in sources.tile_item(bg, scfg["tile"])] == [0]


def test_render_unit_reads_the_tile_crop(ext_tree: Path, tmp_path: Path) -> None:
    cfgs = _cfg(ext_tree, tmp_path)
    item = next(i for i in sources.load_heridal(ext_tree / "heridal", cfgs["external"]["sets"]["heridal"])
                if i.key.endswith("AAA_0001"))
    tile = sources.tile_item(item, cfgs["external"]["sets"]["heridal"]["tile"])[0]
    img = cv2.imread(item.image, cv2.IMREAD_COLOR)
    d = sources.asdict(tile)
    x0, y0, x1, y1 = tile.crop
    np.testing.assert_array_equal(build._load_source(d), img[y0:y1, x0:x1])
    d["crop"] = [2000, 0, 3280, 960]
    with pytest.raises(ValueError, match="vượt ảnh"):
        build._load_source(d)


# ----------------------------------------------------------------------------- giải nén + manifest

def _zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for n, b in members.items():
            z.writestr(n, b)


def _meta(*paths: Path) -> list[dict]:
    return [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def test_extract_filters_members_and_manifest_sha256_matches(tmp_path: Path) -> None:
    arc = tmp_path / "dl" / "s.zip"
    arc.parent.mkdir()
    _zip(arc, {"__MACOSX/s/JPEGImages/._a.jpg": b"mac", "s/JPEGImages/a.jpg": b"A" * 100,
               "s/JPEGImages/b.jpg": b"B" * 50, "s/Site_IR_0001/c.jpeg": b"ir", "s/readme.txt": b"r"})
    scfg = {"archives": ["s.zip"], "include": ["s/JPEGImages/*"], "exclude": ["__MACOSX/*"]}
    dest = tmp_path / "out"
    info = sources.extract_set("s", scfg, _meta(arc), arc.parent, dest)
    rows = sources.read_manifest(dest)
    assert [r["rel"] for r in rows] == ["s/JPEGImages/a.jpg", "s/JPEGImages/b.jpg"]
    assert rows[0]["sha256"] == hashlib.sha256(b"A" * 100).hexdigest() and rows[0]["bytes"] == 100
    assert not (dest / "__MACOSX").exists() and not (dest / "s" / "Site_IR_0001").exists()
    assert info["manifest_sha256"] == hashlib.sha256((dest / sources.EXTRACT_MANIFEST).read_bytes()).hexdigest()
    assert sources.verify_manifest(dest) == []
    assert sources.extract_set("s", scfg, _meta(arc), arc.parent, dest).get("reused") is True
    (dest / "s" / "JPEGImages" / "b.jpg").write_bytes(b"X" * 50)                 # cùng cỡ, khác nội dung
    assert sources.verify_manifest(dest) == ["lệch sha256 s/JPEGImages/b.jpg"]
    assert sources.verify_manifest(dest, full=False) == []
    (dest / "s" / "JPEGImages" / "a.jpg").unlink()
    assert sources.verify_manifest(dest, full=False) == ["thiếu s/JPEGImages/a.jpg"]


def test_extract_rejects_wrong_archive_sha_and_unsafe_members(tmp_path: Path) -> None:
    arc = tmp_path / "s.zip"
    _zip(arc, {"s/a.jpg": b"a"})
    scfg = {"archives": ["s.zip"], "include": ["*"]}
    with pytest.raises(ValueError, match="sha256"):
        sources.extract_set("s", scfg, [{"name": "s.zip", "sha256": "0" * 64}], tmp_path, tmp_path / "o")
    with pytest.raises(KeyError):
        sources.extract_set("s", scfg, [], tmp_path, tmp_path / "o")
    tgz = tmp_path / "evil.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        data = b"x"
        ti = tarfile.TarInfo("../evil.txt")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    with pytest.raises(ValueError, match="nguy hiểm"):
        sources.extract_set("e", {"archives": ["evil.tar.gz"], "include": ["*"]}, _meta(tgz), tmp_path, tmp_path / "e")
    assert not (tmp_path / "evil.txt").exists()
    for bad in ("/abs/x", "C:/x", "a/../b"):
        with pytest.raises(ValueError):
            sources.safe_member(bad)


def test_extract_copies_loose_json_and_tar_members(tmp_path: Path) -> None:
    j = tmp_path / "tiny_set_train.json"
    j.write_text("{}", encoding="utf-8")
    tgz = tmp_path / "train.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for n, b in {"train/labeled_images/a.jpg": b"a", "train/no_label_images/b.jpg": b"b"}.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    scfg = {"archives": ["train.tar.gz", "tiny_set_train.json"], "include": ["train/labeled_images/*", "tiny_set_*.json"]}
    sources.extract_set("t", scfg, _meta(tgz, j), tmp_path, tmp_path / "o")
    assert [r["rel"] for r in sources.read_manifest(tmp_path / "o")] == ["tiny_set_train.json",
                                                                         "train/labeled_images/a.jpg"]


# ----------------------------------------------------------------------------- lọc góc nhìn

def test_external_view_order_image_then_group_then_probe(ext_tree: Path, tmp_path: Path) -> None:
    labels = {"groups": {"wisard/210417_Fixture_Site_VIS_0001": "mixed", "tinyperson/bb_V0001": "oblique"},
              "images": {"tinyperson/bb_V0001_I0000020": "near"}}
    cfg = _cfg(ext_tree, tmp_path, labels)
    ext = cfg["external"]
    lab = sources.external_view_labels(ext)
    items = {it.key: it for it in sources.collect_external(cfg)}
    tp = items["tinyperson/bb_V0001_I0000010"]
    assert not sources.is_nadir_external(tp, ext, lab)                    # nhãn nhóm oblique, không cần probe
    assert sources.is_nadir_external(items["tinyperson/bb_V0001_I0000020"], ext, lab)   # nhãn ảnh thắng nhóm
    w = items["wisard/210417_Fixture_Site_VIS_0001_00000001"]
    with pytest.raises(ValueError, match="probe"):
        sources.is_nadir_external(w, ext, lab)                            # nhóm mixed mà chưa có điểm
    w.probe_nadir = 0.8
    assert sources.is_nadir_external(w, ext, lab)
    w.probe_nadir = 0.79
    assert not sources.is_nadir_external(w, ext, lab)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"images": {"heridal/x": "mixed"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="không hợp lệ"):
        sources.load_external_view_labels(bad)


def test_select_external_filters_then_tiles_and_respects_d02x_flag(ext_tree: Path, tmp_path: Path) -> None:
    cfg = _cfg(ext_tree, tmp_path, {"groups": {}, "images": {"heridal/train_AAA_0002": "oblique"}})
    keys = [it.key for it in sources.collect_external(cfg)]
    _write_probe(cfg, {k: (0.1 if "bb_V0001_I0000010" in k else 0.95) for k in keys})
    sel = sources.select_external(cfg)
    got = {it.key for it in sel}
    assert "tinyperson/bb_V0001_I0000010" not in got                      # probe 0,1 < 0,8
    assert not any(k.startswith("heridal/train_AAA_0002") for k in got)   # nhãn tay oblique
    assert {"heridal/train_AAA_0001_t00", "heridal/train_AAA_0001_t01"} <= got
    assert all(it.crop for it in sel if it.source == "heridal")
    assert not any(it.source == "heridal" and "_t" not in it.key for it in sel)   # HERIDAL chỉ vào dưới dạng ô
    cfg["external"]["sets"]["tinyperson"]["d02x"] = False
    assert not any(it.source == "tinyperson" for it in sources.select_external(cfg))
    st = sources.external_stats(cfg)
    assert st["heridal"]["before_filter"]["images"] == 3 and st["heridal"]["after_filter"]["images"] == 2
    assert st["heridal"]["tiles"]["negative_tiles"] >= 1
    assert st["heridal"]["review"] == {"n": 1, "labels": {"oblique": 1}, "accepted_by_hand": 0, "selected_by_probe": 1,
                                       "true_pos": 0, "precision": 0.0, "recall": None}


# ----------------------------------------------------------------------------- tập trộn d02x

def test_d02x_mixes_visdrone_and_external_with_both_variants(ext_tree: Path, tmp_path: Path) -> None:
    cfg = _cfg(ext_tree, tmp_path)
    _write_probe(cfg, {it.key: 0.95 for it in sources.collect_external(cfg)})
    vd_img = tmp_path / "vd" / "0000001_00001_d_0000001.jpg"
    _img(vd_img, 1360, 765, 5)
    vd = sources.SourceItem("visdrone/0000001_00001_d_0000001", "visdrone", str(vd_img), "train", (1360, 765),
                            [[600, 300, 640, 380]])
    items = [vd, *sources.select_external(cfg)]
    units = build.plan_units(items, 2, SEED, None, build.MIXED_SETS["d02x"], sources.windows_by_source(cfg))
    assert sum(u.endswith("@d0") for _, u in units) == sum(u.endswith("@d2") for _, u in units) == len(units) // 2
    assert sum(1 for d, u in units if d["source"] == "heridal") == 2 * sum(it.source == "heridal" for it in items)
    ctx = build.make_context(["d02x"], REPO / "configs" / "camera_profile.yaml", cfg,
                             REPO / "configs" / "degrade_d1.yaml", REPO / "configs" / "degrade_d2.yaml")
    summary = build.build_set("d02x", units, SEED, ctx, tmp_path / "sets", workers=1)
    d = tmp_path / "sets" / "d02x"
    recs = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["written"] == len(recs) > 0
    in_d02x = {n for n, s in cfg["external"]["sets"].items() if s.get("d02x", True)}
    assert {r["source"] for r in recs} == {"visdrone"} | in_d02x and "heridal" in in_d02x
    assert {r["variant"] for r in recs} == {"d0", "d2"}
    assert all(r["name"].startswith(f"{r['variant']}__") and r["set"] == "d02x" for r in recs)
    pairs: dict[str, set] = {}
    for r in recs:
        pairs.setdefault(r["unit"].rsplit("@", 1)[0], set()).add(r["variant"])
        lab = (d / r["split"] / "labels" / (Path(r["name"]).stem + ".txt")).read_text(encoding="utf-8")
        assert len(lab.splitlines()) == r["n_boxes"]
    assert all(v == {"d0", "d2"} for v in pairs.values())               # cùng cửa sổ cho cả hai biến thể
    her = [r for r in recs if r["source"] == "heridal"]
    assert all(r["crop"] and r["group"] in ("AAA", "BBB") for r in her)
    assert {r["split"] for r in her if r["group"] == "BBB"} == {"val"}
    split_of: dict[tuple, set] = {}
    for r in recs:
        if r.get("group"):
            split_of.setdefault((r["source"], r["group"]), set()).add(r["split"])
    assert all(len(s) == 1 for s in split_of.values())                  # không rò nhóm giữa train/val
    one = next(r for r in her if r["variant"] == "d2")
    item = next(dd for dd, u in units if u == one["unit"])
    same = build.render_unit("d2", item, one["unit"].rsplit("@", 1)[0], SEED, ctx)   # = ảnh của tập d2 đơn
    assert hashlib.sha256(same["data"]).hexdigest() == one["sha256"]
    assert yaml.safe_load((d / "dataset.yaml").read_text(encoding="utf-8"))["names"] == {0: "person"}
    key_d2 = build.build_key("d2", SEED, ctx)
    assert build.build_key("d02x", SEED, ctx) != key_d2


def test_mixed_set_registered_in_cli_and_v1_sets_unchanged() -> None:
    assert build.MIXED_SETS == {"d02x": ("d0", "d2")}
    assert build.split_unit("visdrone/a#0@d2") == ("visdrone/a#0", "d2")
    assert build.split_unit("visdrone/a#0") == ("visdrone/a#0", None)
    item = sources.SourceItem("visdrone/a", "visdrone", "x.jpg", "train", (1360, 765))
    assert sorted(u for _, u in build.plan_units([item], 2, 1, None)) == ["visdrone/a#0", "visdrone/a#1"]
    assert sources.SourceItem("x/a", "x", "x", "train", (1, 1)).crop is None   # tập đơn: không cắt ô, không nhóm


# ----------------------------------------------------------------------------- không ảnh ngoài trong git

def _tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def _looks_external(stem: str) -> bool:
    return bool(sources.HERIDAL_STEM_RE.match(stem) or sources.TINYPERSON_GROUP_RE.match(stem)
                or "_VIS_" in stem or "_IR_" in stem or stem.startswith("bing"))


def test_no_external_image_is_tracked_in_git() -> None:
    imgs = [p for p in _tracked() if Path(p).suffix.lower() in sources.IMAGE_EXTS]
    assert not [p for p in imgs if p.startswith("tests/degrade/fixtures/")]
    assert not [p for p in imgs if _looks_external(Path(p).stem)]


# ----------------------------------------------------------------------------- dữ liệu thật (ổ D)

@pytest.mark.data
def test_real_external_manifests_match_config_and_none_in_git(require_data) -> None:
    cfg = sources.load_config(REPO / "configs" / "sources.yaml")
    ext = cfg["external"]
    tracked = {hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in _tracked()
               if Path(p).suffix.lower() in sources.IMAGE_EXTS and (REPO / p).is_file()}
    for name in sources.external_sets(ext):
        d = require_data(sources.external_root(ext) / name, f"python -m xla.degrade.sources extract --set {name}")
        info = json.loads((d / sources.EXTRACT_INFO).read_text(encoding="utf-8"))
        want = {f["name"]: f for f in cfg["downloads"][name]["files"]}
        assert [(a["name"], a["sha256"], a["bytes"]) for a in info["archives"]] == [
            (a, want[a]["sha256"], want[a]["bytes"]) for a in ext["sets"][name]["archives"]]
        assert info["manifest_sha256"] == sources.file_sha256(d / sources.EXTRACT_MANIFEST)
        assert sources.verify_manifest(d, full=False) == []
        assert not tracked & {r["sha256"] for r in sources.read_manifest(d)}


@pytest.mark.data
def test_real_d02x_has_external_sources_and_no_group_leak(require_data) -> None:
    from xla import paths

    d = require_data(paths.data_root() / "d02x", "python -m xla.degrade.build --set d02x --workers 4 --json")
    recs = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    srcs = {r["source"] for r in recs}
    assert "visdrone" in srcs and srcs & set(sources.EXTERNAL_SETS)
    pairs: dict[str, set] = {}
    split_of: dict[tuple, set] = {}
    for r in recs:
        pairs.setdefault(r["unit"].rsplit("@", 1)[0], set()).add(r["variant"])
        if r.get("group"):
            split_of.setdefault((r["source"], r["group"]), set()).add(r["split"])
    assert all(v == {"d0", "d2"} for v in pairs.values())
    assert all(len(s) == 1 for s in split_of.values())
    assert not [r for r in recs if r["name"][4:].startswith(("2026", "real"))]   # không khung bay thật
    n_img = sum(1 for sp in ("train", "val") for _ in (d / sp / "images").iterdir())
    assert n_img == len(recs)
