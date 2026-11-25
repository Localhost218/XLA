"""``xla.verify.neg_audit``: soát tay âm d02x của verifier — tỉ lệ Wilson, kiểm review khớp mẫu, và ``audit.json`` đã
commit tính lại đúng từ ``sample.json`` + ``review.csv`` (không sửa tay một trong ba)."""

from __future__ import annotations

import json

import pytest

from xla.paths import repo_root
from xla.verify import neg_audit as NA

AUDIT_DIR = repo_root() / "results" / "v2" / "verifier" / "neg_audit"


def _rows(n: int) -> list[dict]:
    return [{"id": f"r{i}", "source": "visdrone" if i % 2 else "heridal", "degrade": "d0"} for i in range(n)]


def test_summarize_counts_sure_and_unsure_separately() -> None:
    rows = _rows(10)
    rev = {r["id"]: {"person": 1 if i < 2 else None if i < 5 else 0, "note": ""} for i, r in enumerate(rows)}
    s = NA.summarize(rows, rev)
    assert s["all"]["person"] == 2 and s["all"]["unsure"] == 3 and s["all"]["rate"] == pytest.approx(0.2)
    assert s["all"]["rate_incl_unsure"] == pytest.approx(0.5)
    lo, hi = s["all"]["wilson95"]
    assert 0 < lo < 0.2 < hi < 1
    assert s["material"] is (lo > NA.MATERIAL_LO)
    assert sum(v["n"] for v in s["by_source"].values()) == 10
    assert s["person_ids"] == ["r0", "r1"]


def test_read_review_rejects_missing_extra_or_bad_label(tmp_path) -> None:
    p = tmp_path / "review.csv"
    p.write_text("id,person,note\nr0,1,\nr1,x,\n", encoding="utf-8")
    with pytest.raises(NA.AuditError):
        NA.read_review(p, ["r0", "r1"])
    p.write_text("id,person,note\nr0,1,\n", encoding="utf-8")
    with pytest.raises(NA.AuditError):
        NA.read_review(p, ["r0", "r1"])
    with pytest.raises(NA.AuditError):
        NA.read_review(tmp_path / "missing.csv", ["r0"])


def test_committed_audit_recomputes_from_sample_and_review() -> None:
    sample = json.loads((AUDIT_DIR / NA.SAMPLE_FILE).read_text(encoding="utf-8"))
    rows = sample["rows"]
    assert len(rows) == sample["n"] == len({r["id"] for r in rows})
    review = NA.read_review(AUDIT_DIR / NA.REVIEW_FILE, [r["id"] for r in rows])
    got = {"seed": sample["seed"], "pool": sample["pool"], **NA.summarize(rows, review)}
    assert json.loads(json.dumps(got)) == json.loads((AUDIT_DIR / NA.AUDIT_FILE).read_text(encoding="utf-8"))
