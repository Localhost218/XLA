"""Accept phase 14 AI v2: trang A/B v2 (không lộ khoá, đủ cặp, lặp đổi bên, chỉ khung dev), ab_score D13, guard_fp,
decision.json."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from xla.eval import ab_page, ab_score, guard_fp
from xla.paths import repo_root

SEED = 20261125


def _cands(n: int = 90) -> list[dict]:
    out = []
    for i in range(n):
        sess, ab, person = f"S#{i % 3}", ("3-8", "8-14", ">14")[i % 3], i % 2 == 0
        boxes = np.array([[100.0 + i, 100.0, 140.0 + i, 160.0]]) if person else np.zeros((0, 4))
        out.append({"frame": f"f{i:03d}.jpg", "session": sess, "alt_m": 5.0, "alt_bin": ab, "has_person": person,
                    "block": i // 10, "boxes": boxes, "stratum": [sess, ab, person]})
    return out


def _fns() -> dict:
    import cv2

    return {"raw": lambda f: f, "classic": lambda f: 255 - f,
            "offline": lambda f: cv2.resize(f, (1280, 960), interpolation=cv2.INTER_NEAREST)}


def _load(name: str) -> np.ndarray:
    rng = np.random.default_rng(int(name[1:4]))
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


def _build(tmp: Path, zoom: int = 1) -> dict:
    return ab_page.build_v2(_cands(), _load, _fns(), tmp, comparisons=ab_page.COMPARISONS_V2, per_comparison=25,
                            n_repeats=5, seed=SEED, crop_left=16, variants_info={}, zoom=zoom)


@pytest.mark.parametrize("zoom", [1, 2])
def test_page_counts_no_leak_repeats_swapped(tmp_path: Path, zoom: int) -> None:
    res = _build(tmp_path, zoom)
    assert res["by_comparison"] == {c: 25 for c in ab_page.COMPARISONS_V2}
    assert res["n_pairs"] == 80 and all(v >= 1 for v in res["repeats_by_comparison"].values())
    chk = ab_page.check_page(tmp_path)
    assert chk["ok"], chk
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    pairs = {p["id"]: p for p in key["pairs"]}
    for p in pairs.values():
        if p["repeat_of"]:
            src = pairs[p["repeat_of"]]
            assert (p["left"], p["right"]) == (src["right"], src["left"]) and p["left_img"] == src["right_img"]
            assert p["id"] > src["id"]
    frames = [p["frame"] for p in pairs.values() if p["repeat_of"] is None]
    assert len(set(frames)) == 75
    html = (tmp_path / "ab-test.html").read_text(encoding="utf-8")
    for p in pairs.values():
        assert p["frame"] not in html
    import cv2

    for p in pairs.values():
        a, b = (cv2.imread(str(tmp_path / p[s])).shape for s in ("left_img", "right_img"))
        assert a[1] * b[0] == a[0] * b[1]  # cùng tỉ lệ khung: trình duyệt phóng về cùng cỡ
        sizes = {p["left"]: a, p["right"]: b}
        if "offline" in sizes:
            other = next(v for v in sizes if v != "offline")
            assert sizes["offline"][0] == 2 * sizes[other][0]  # offline ở độ phân giải gốc ×2, không hạ cỡ


def test_check_page_catches_leak(tmp_path: Path) -> None:
    _build(tmp_path)
    p = tmp_path / "ab-test.html"
    p.write_text(p.read_text(encoding="utf-8").replace("Ảnh trái", "offline"), encoding="utf-8")
    assert ab_page.check_page(tmp_path)["leaks"] == ["offline"]


def test_dev_candidates_only_dev() -> None:
    root = repo_root()
    split = root / ab_page.SPLIT_REL
    if not split.is_file() or not (root / ab_page.LABELS_REL / "devpool").is_dir():
        pytest.skip("không có dữ liệu thật")
    rows = {r["frame"]: r["split"] for r in json.loads(split.read_text(encoding="utf-8"))["frames"]}
    cands = ab_page.dev_candidates(root, split, root / ab_page.LABELS_REL)
    assert len(cands) >= 75 and {rows[c["frame"]] for c in cands} == {"dev"}


@pytest.mark.parametrize("out", ["results/ab-v2", "results/ab-v2-zoom"])
def test_built_pages_dev_only_and_complete(out: str) -> None:
    d = repo_root() / out
    if not (d / "ab_key.json").is_file():
        pytest.skip(f"chưa dựng {out}")
    assert ab_page.check_page(d)["ok"]
    key = json.loads((d / "ab_key.json").read_text(encoding="utf-8"))
    rows = {r["frame"]: r["split"] for r in
            json.loads((repo_root() / ab_page.SPLIT_REL).read_text(encoding="utf-8"))["frames"]}
    assert {rows[p["frame"]] for p in key["pairs"]} == {"dev"}
    orig = [p for p in key["pairs"] if p["repeat_of"] is None]
    assert {c: sum(p["comparison"] == c for p in orig) for c in key["comparisons"]} == \
        {c: 25 for c in ab_page.COMPARISONS_V2}
    assert len(key["pairs"]) - len(orig) == 5


def _fake_export(key: dict, wins: dict[str, int]) -> dict:
    left = {c: 0 for c in wins}
    choices = []
    for p in key["pairs"]:
        cand = key["comparisons"][p["comparison"]][0]
        if p["repeat_of"] is None:
            win = left[p["comparison"]] < wins[p["comparison"]]
            left[p["comparison"]] += 1
        else:
            win = True
        side = ("left" if p["left"] == cand else "right") if win else ("right" if p["left"] == cand else "left")
        choices.append({"id": p["id"], "choice": side})
    return {"format": ab_score.EXPORT_FORMAT, "key_id": key["key_id"], "choices": choices}


def test_score_d13_17_fails_18_passes(tmp_path: Path) -> None:
    _build(tmp_path)
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    res = ab_score.score(_fake_export(key, {"offline_vs_raw": 17, "offline_vs_classic": 18, "classic_vs_raw": 25}),
                         key)
    c = res["comparisons"]
    assert c["offline_vs_raw"]["wins"] == 17 and c["offline_vs_raw"]["passes"] is False
    assert c["offline_vs_raw"]["ci95_wilson"][0] == pytest.approx(0.484, abs=1e-3)
    assert c["offline_vs_classic"]["wins"] == 18 and c["offline_vs_classic"]["passes"] is True
    assert c["classic_vs_raw"]["passes"] is True


def _neg_doc() -> dict:
    frames = [{"frame": f"c{i:03d}.jpg", "session": f"s#{i // 20}", "split": "cal"} for i in range(40)]
    frames += [{"frame": f"h{i:03d}.jpg", "session": "s#9", "split": "holdout"} for i in range(10)]
    return {"frames": frames}


def test_guard_rejects_fp_increase_and_passes_decrease() -> None:
    cal = guard_fp.split_frames(_neg_doc(), "cal")
    worse = guard_fp.evaluate(cal, lambda n: 0, lambda n: 1 if int(n[1:4]) % 2 else 0, n_boot=2000)
    assert worse["diff"]["ci95"][0] > 0 and worse["decision"] == "reject"
    better = guard_fp.evaluate(cal, lambda n: 1 if int(n[1:4]) % 4 == 0 else 0, lambda n: 0, n_boot=2000)
    assert better["decision"] == "pass" and better["cal_offline"]["fp_per_frame"] == 0
    same = guard_fp.evaluate(cal, lambda n: int(n[1:4]) % 3 == 0, lambda n: int(n[1:4]) % 3 == 0, n_boot=2000)
    assert same["decision"] == "pass"


def test_guard_never_reads_holdout_for_decision() -> None:
    doc = _neg_doc()
    cal = guard_fp.split_frames(doc, "cal")
    assert all(f["split"] == "cal" for f in cal) and len(cal) == 40

    def no_holdout(n: str) -> int:
        if n.startswith("h"):
            raise AssertionError(f"đã đọc khung holdout {n}")
        return 0

    res = guard_fp.evaluate(cal, no_holdout, no_holdout, n_boot=500)
    assert res["decision"] == "pass"
    assert "holdout" not in inspect.signature(guard_fp.evaluate).parameters
    rep = guard_fp.holdout_report([])
    assert rep["n_frames"] == 0 and rep["status"] == guard_fp.HOLDOUT_EMPTY and rep["fp_holdout_raw"] is None


def test_block_ids_within_session() -> None:
    cal = guard_fp.split_frames(_neg_doc(), "cal")
    ids = guard_fp.block_ids(cal, 10)
    assert len(set(ids)) == 4 and all(i.split("/")[0] in {"s#0", "s#1"} for i in ids)


def test_decision_schema(tmp_path: Path) -> None:
    _build(tmp_path)
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    good = ab_score.score(_fake_export(key, {c: 20 for c in ab_page.COMPARISONS_V2}), key)
    bad = ab_score.score(_fake_export(key, {c: 12 for c in ab_page.COMPARISONS_V2}), key)
    gpass = {"decision": "pass", "diff": {"diff": -0.03, "ci95": [-0.1, 0.04]}}
    d = ab_score.decision({"full": bad, "zoom": good}, gpass)
    assert tuple(d) == ab_score.DECISION_KEYS and d["format"] == ab_score.DECISION_FORMAT
    assert d["offline_ship"] is True and d["hybrid_replaces_classic"] is False and d["classic_default_unchanged"]
    assert ab_score.decision({"full": bad}, gpass)["offline_ship"] is False
    assert ab_score.decision({"full": good}, {"decision": "reject", "diff": {}})["offline_ship"] is False
    json.dumps(d)
    real = repo_root() / "results/ab-v2/decision.json"
    if real.is_file():
        assert tuple(json.loads(real.read_text(encoding="utf-8"))) == ab_score.DECISION_KEYS


def test_guard_result_on_disk_matches_meta() -> None:
    root = repo_root()
    g = root / guard_fp.OUT_REL
    if not g.is_file():
        pytest.skip("chưa chạy guard_fp")
    res = json.loads(g.read_text(encoding="utf-8"))
    meta = json.loads((root / "models/release/enhancer-offline-v1.0.0/meta.json").read_text(encoding="utf-8"))
    assert meta["onnx_sha256"] == res["bundle"]["onnx_sha256"]
    assert meta["metrics"]["fp_cal_offline"] == res["cal_offline"]["fp_per_frame"]
    assert meta["metrics"]["fp_holdout_raw"] is None and res["holdout"]["n_frames"] == 0
