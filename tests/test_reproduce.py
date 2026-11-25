"""Phase 18 — tái lập: `reproduce.py --check` đạt, báo cáo/slide/README khớp manifest và luật dự án.

Phần chạy lại cần dự đoán đã lưu + nhãn thật (marker `data`, CI loại); phần còn lại chỉ đọc file đã commit nên chạy ở CI.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "BAO_CAO_DO_AN.md"
SLIDE = ROOT / "docs" / "slide" / "index.html"
MANIFEST = json.loads((ROOT / "results" / "reproduce" / "manifest.json").read_text(encoding="utf-8"))
REF_RE = re.compile(r"results/[\w./-]+\.(?:md|svg|png|jpg|json)")
BLOCK_RE = re.compile(r"<!-- bang: (results/reproduce/tables/[\w.-]+\.md) -->\n(.*?)<!-- /bang -->", re.DOTALL)


@pytest.mark.data
def test_reproduce_check_dat():
    r = subprocess.run([sys.executable, "scripts/reproduce.py", "--check", "--json"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", check=False)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["pass"] is True


def test_moi_bang_hinh_bao_cao_nhac_toi_co_sha256():
    known = set(MANIFEST["outputs"]) | set(MANIFEST["assets"])
    refs = {r for r in REF_RE.findall(REPORT.read_text(encoding="utf-8"))
            if r.startswith("results/reproduce/") or r.endswith((".png", ".jpg", ".svg"))}
    refs.discard("results/reproduce/manifest.json")
    assert refs, "báo cáo không nhắc bảng/hình nào"
    assert refs <= known, sorted(refs - known)
    # chiều ngược: bảng/hình nào đã dựng cũng phải được báo cáo dùng, không dựng thừa
    assert set(MANIFEST["outputs"]) <= refs, sorted(set(MANIFEST["outputs"]) - refs)


def test_bang_trong_bao_cao_khop_file_da_dung():
    blocks = BLOCK_RE.findall(REPORT.read_text(encoding="utf-8"))
    assert {p for p, _ in blocks} == {k for k in MANIFEST["outputs"] if "/tables/" in k}
    for path, body in blocks:
        assert body == (ROOT / path).read_text(encoding="utf-8"), f"{path}: chạy scripts/reproduce.py --write"


V2_TABLES = ("results/reproduce/tables/t7_v2_nhan_dien.md", "results/reproduce/tables/t8_v2_ab.md",
             "results/reproduce/tables/t9_v2_ms.md")
V2_FIG = "results/reproduce/fig/f4_v2_bao_nham_recall.svg"
V2_CSV = tuple(f"results/v2/system/preds/system_{st}_test.csv" for st in ("detector", "verifier", "track"))


def test_v2_bang_hinh_co_trong_manifest():
    """Phase 17 AI v2: t7–t9 + f4 được dựng và có sha256; ba CSV cả hệ nằm trong đầu vào được `--check` chấm lại."""
    for rel in (*V2_TABLES, V2_FIG):
        assert rel in MANIFEST["outputs"], f"{rel}: thiếu trong manifest (chạy scripts/reproduce.py --write)"
        assert (ROOT / rel).is_file(), rel
    assert set(V2_CSV) <= set(MANIFEST["inputs"]), sorted(set(V2_CSV) - set(MANIFEST["inputs"]))


def test_v2_moi_bang_co_khoi_trong_bao_cao():
    blocks = dict(BLOCK_RE.findall(REPORT.read_text(encoding="utf-8")))
    for rel in V2_TABLES:
        assert rel in blocks, f"BAO_CAO_DO_AN.md thiếu khối <!-- bang: {rel} -->"
        assert blocks[rel] == (ROOT / rel).read_text(encoding="utf-8"), f"{rel}: chạy scripts/reproduce.py --write"
    assert V2_FIG in REPORT.read_text(encoding="utf-8")


NUM_RE = re.compile(r"[+−-]?\d+,\d+|\d+/\d+")


def test_slide_so_v2_lay_tu_bang():
    """Số trong phần v2 của slide (giữa `<!-- v2 -->` và `<!-- /v2 -->`) phải có nguyên văn trong t7–t9: không gõ tay."""
    html = SLIDE.read_text(encoding="utf-8")
    parts = re.findall(r"<!-- v2 -->(.*?)<!-- /v2 -->", html, flags=re.DOTALL)
    assert parts, "slide không có phần v2"
    tables = "".join((ROOT / rel).read_text(encoding="utf-8") for rel in V2_TABLES).replace("−", "-")
    nums = {n.replace("−", "-").lstrip("+") for p in parts for n in NUM_RE.findall(re.sub(r"<[^>]+>", " ", p))}
    assert nums, "phần v2 của slide không có số nào"
    missing = sorted(n for n in nums if n not in tables)
    assert not missing, f"số trên slide không có trong t7–t9: {missing}"


def test_slide_local_khong_tai_tu_mang():
    html = SLIDE.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "<head>" in html and "<body" in html
    body = re.sub(r"<!-- cite -->.*?<!-- /cite -->", "", html, flags=re.DOTALL)
    assert not re.search(r"https?://", body), "URL mạng ngoài phần trích dẫn"
    for src in re.findall(r'src="([^"]+)"', html):
        assert (SLIDE.parent / src).resolve().is_file(), src


def test_readme_khong_con_camera_cu():
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if re.search(r"OV2640|ESP32-CAM", line):
            assert "nhầm" in line, f"README còn camera cũ ngoài câu giải thích: {line}"


def test_nhan_moc_coco_khop_trong_so_da_ghi():
    """Tên + cỡ vào của mốc COCO trong bảng lấy theo trọng số ghi trong JSON kết quả (từng ghi nhầm YOLO11n, khung 640)."""
    meta = json.loads((ROOT / "results/baseline/baseline.json").read_text(encoding="utf-8"))["run_meta"]
    e4 = json.loads((ROOT / "results/e4/e4.json").read_text(encoding="utf-8"))["detectors"]["coco"]
    for weights, table, want in ((meta["weights"]["path"], "t1_nhan_dien.md", f"COCO @{meta['imgsz']}"),
                                 (e4["weights"], "t5_e4.md", "COCO")):
        name = "YOLO" + PureWindowsPath(weights).stem.removeprefix("yolo")  # JSON ghi cả đường Windows lẫn kiểu '/'
        row = (ROOT / "results/reproduce/tables" / table).read_text(encoding="utf-8").splitlines()[2]
        assert f"{name} {want}" in row, f"{table}: dòng mốc {row!r} không khớp trọng số {weights}"
