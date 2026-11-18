"""Phase 17 — `plans/ai/` và `ml/` của IOT-CV trỏ về XLA, không file nào bị xoá khi chưa duyệt.

Đọc `results/iotcv_refs.json` (danh sách tham chiếu tới `ml/`, `plans/ai` trong IOT-CV đã phân loại).
Thiếu repo IOT-CV → FAIL (marker `data`, CI loại).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xla.paths import iotcv_root, repo_root

pytestmark = pytest.mark.data

REFS = repo_root() / "results" / "iotcv_refs.json"
MARK = "Đã thay thế"


@pytest.fixture
def root(require_data) -> Path:
    return require_data(Path(iotcv_root()), "repo IOT-CV (configs/paths.yaml: iotcv_root)")


@pytest.fixture
def refs() -> dict:
    return json.loads(REFS.read_text(encoding="utf-8"))


def test_moi_file_plans_ai_co_khoi_da_thay_the(root):
    files = sorted((root / "plans" / "ai").glob("ai-phase-0*.md"))
    assert len(files) == 4
    for f in files:
        head = "\n".join(f.read_text(encoding="utf-8").splitlines()[:15])
        assert MARK in head and "XLA" in head and "OV3660" in head, f.name


def test_ml_readme_tro_ve_xla(root):
    text = (root / "ml" / "README.md").read_text(encoding="utf-8")
    assert "XLA" in text and "lay_model.py" in text
    assert "ml/README.md" in (root / "ml" / "pyproject.toml").read_text(encoding="utf-8").splitlines()[0]


def test_tham_chieu_loai_a_nam_trong_file_co_con_tro(root, refs):
    assert refs["n_refs"] == len(refs["refs"]) and refs["n_type_a"] > 0
    assert {r["type"] for r in refs["refs"]} <= {"a", "b"}
    for f in sorted({r["file"] for r in refs["refs"] if r["type"] == "a"}):
        assert "XLA" in (root / f).read_text(encoding="utf-8"), f


def test_khung_suon_ml_scripts_chua_bi_xoa_khi_chua_duyet(root, refs):
    skel = refs["ml_scripts_skeleton"]
    now = sorted(p.name for p in (root / "ml" / "scripts").glob("*.py"))
    if skel["delete_approved"]:
        return  # chủ dự án đã duyệt xoá (ghi ngày vào manifest)
    assert now == skel["files"] and len(now) == 6
