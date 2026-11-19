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
