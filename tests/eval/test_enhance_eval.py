"""Phase 14 bước 11: đánh giá nhánh A, trang thử mù A/B, chấm A/B, E4.

Phần đơn vị (không marker) chạy trên ảnh tổng hợp. Phần ``data`` kiểm file thật ``results/ab/*``, ``results/e4/e4.json``
do Owner sinh bằng model thật (thiếu file → FAIL, không skip).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from xla.eval import ab_page, ab_score, enhance_eval
from xla.eval.temporal import flicker_ratio
from xla.paths import repo_root

VARIANT_WORDS = ("raw", "classic", "net", "enhanc", "goc", "co_dien")
AB_DIR = repo_root() / "results" / "ab"
E4_JSON = repo_root() / "results" / "e4" / "e4.json"


def _seq(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------- nhấp nháy


def test_flicker_identity_is_one() -> None:
    runs = [_seq(4, 1), _seq(6, 2)]
    pairs = [enhance_eval.run_flicker(r, r, 16) for r in runs]
    assert enhance_eval.pooled_flicker([p[0] for p in pairs], [p[1] for p in pairs]) == pytest.approx(1.0)


def test_pooled_flicker_equals_temporal_over_all_runs() -> None:
    raw = [_seq(4, 3), _seq(3, 4), _seq(7, 5)]
    proc = [[(x // 2 + 64).astype(np.uint8) for x in r] for r in raw]
    pairs = [enhance_eval.run_flicker(r, p, 16) for r, p in zip(raw, proc, strict=True)]
    pooled = enhance_eval.pooled_flicker([p[0] for p in pairs], [p[1] for p in pairs])
    assert pooled == pytest.approx(flicker_ratio(raw, proc, crop_left=16), rel=1e-9)
    assert pooled < 1.0


def test_test_region_rows_keep_test_spans_only() -> None:
    split = [{"frame": "a", "t_unix": 10.0, "split": "test", "block": 1},
             {"frame": "b", "t_unix": 20.0, "split": "test", "block": 1},
             {"frame": "c", "t_unix": 50.0, "split": "dev", "block": 2},
             {"frame": "d", "t_unix": 60.0, "split": "dev", "block": 2}]
    man = [{"frame": f"m{t}", "kind": "flight", "t_unix": str(t)} for t in (10.0, 15.0, 15.1, 30.0, 55.0)]
    man.append({"frame": "n", "kind": "neg", "t_unix": "12"})
    rows = enhance_eval.test_region_rows(split, man)
    assert [r["frame"] for r in rows] == ["m10.0", "m15.0", "m15.1"]
    assert {r["block"] for r in rows} == {1}


# ---------------------------------------------------------------- bootstrap, tỉ lệ đổi, D5


def test_block_samples_take_whole_blocks() -> None:
    blocks = [1, 1, 2, 3, 3, 3]
    for s in enhance_eval.block_samples(blocks, 20, 7):
        picked = [blocks[i] for i in s]
        for b in set(picked):
            assert picked.count(b) % blocks.count(b) == 0
    assert enhance_eval.block_samples([], 5, 0) == []


def test_change_summary_identity_and_halving() -> None:
    v = np.asarray([1.0, 2.0, 3.0, 4.0])
    samples = enhance_eval.block_samples([0, 0, 1, 1], 50, 1)
    s = enhance_eval.change_summary({"raw": v, "classic": v / 2, "net": v.copy()}, "mean", samples)
    assert s["net"]["change_vs_raw"] == 0.0 and s["net"]["change_ci95"] == [0.0, 0.0]
    assert s["classic"]["change_vs_raw"] == pytest.approx(-0.5)
    assert s["classic"]["change_ci95"] == [pytest.approx(-0.5), pytest.approx(-0.5)]


@pytest.mark.parametrize(("lo", "want"), [(0.01, "enhanced"), (0.0, "raw"), (-0.2, "raw"), (None, "raw")])
def test_d5_recommendation_only_when_ci_above_zero(lo: float | None, want: str) -> None:
    assert enhance_eval.d5_recommendation(lo) == want


# ---------------------------------------------------------------- nạp hàm cổ điển


def test_load_classic_file_spec(tmp_path: Path) -> None:
    mod = tmp_path / "cl.py"
    mod.write_text("def load_params(p=None):\n    return {'k': 3}\n\n"
                   "def enhance(f, params):\n    out = f.copy()\n    out[:] = params['k']\n    return out\n",
                   encoding="utf-8")
    fn, info = enhance_eval.load_classic(f"{mod}:enhance")
    out = fn(np.zeros((480, 640, 3), np.uint8))
    assert out.dtype == np.uint8 and int(out.max()) == 3 and info["params"] == {"k": 3}
    with pytest.raises(enhance_eval.EvalError):
        fn(np.zeros((10, 10, 3), np.uint8))  # đầu ra sai cỡ → lỗi rõ ràng, không im lặng


def test_load_classic_errors(tmp_path: Path) -> None:
    with pytest.raises(enhance_eval.UsageError):
        enhance_eval.load_classic("khong_co_dau_hai_cham")
    with pytest.raises(enhance_eval.EvalError):
        enhance_eval.load_classic("xla_khong_ton_tai_123:enhance")
    with pytest.raises(enhance_eval.EvalError):
        enhance_eval.load_classic(f"{tmp_path / 'thieu.py'}:enhance")


# ---------------------------------------------------------------- chấm A/B


def test_wilson_matches_hand_computation() -> None:
    # p̂ = 18/25 = 0,72; z² = 3,841459; tâm = (0,72 + 0,076829)/1,153658 = 0,690698;
    # nửa = 1,698913 · sqrt(0,72·0,28/25 + 3,841459/2500) = 0,166465 → [0,5242, 0,8572]
    lo, hi = ab_score.wilson(18, 25)
    assert lo == pytest.approx(0.5242, abs=1e-4) and hi == pytest.approx(0.8572, abs=1e-4)
    lo, hi = ab_score.wilson(14, 25)
    assert lo == pytest.approx(0.3707, abs=1e-4) and hi == pytest.approx(0.7333, abs=1e-4)
    assert ab_score.wilson(0, 10)[0] == 0.0 and ab_score.wilson(10, 10)[1] == pytest.approx(1.0)


def _sample_key_export(net_wins_raw: int = 18, net_wins_classic: int = 14,
                       agree: int = 4) -> tuple[dict, dict]:
    pairs, choices = [], []
    i = 0
    for comp, other, wins in (("net_vs_raw", "raw", net_wins_raw), ("net_vs_classic", "classic", net_wins_classic)):
        for j in range(25):
            i += 1
            left, right = ("net", other) if j % 2 == 0 else (other, "net")
            pairs.append({"id": f"{i:02d}", "comparison": comp, "left": left, "right": right, "repeat_of": None})
            net_side = "left" if left == "net" else "right"
            choices.append({"id": f"{i:02d}", "choice": net_side if j < wins else
                            ("right" if net_side == "left" else "left")})
    by_id = {p["id"]: p for p in pairs}
    chosen = {c["id"]: c["choice"] for c in choices}
    for r in range(5):
        i += 1
        src = by_id[f"{r * 10 + 1:02d}"]
        pairs.append(src | {"id": f"{i:02d}", "left": src["right"], "right": src["left"], "repeat_of": src["id"]})
        same = "right" if chosen[src["id"]] == "left" else "left"  # đổi bên → cùng biến thể = bên kia
        choices.append({"id": f"{i:02d}", "choice": same if r < agree else chosen[src["id"]]})
    key = {"format": ab_score.KEY_FORMAT, "key_id": "abc123", "pairs": pairs}
    export = {"format": ab_score.EXPORT_FORMAT, "key_id": "abc123", "choices": choices}
    return key, export


def test_ab_score_sample() -> None:
    key, export = _sample_key_export()
    r = ab_score.score(export, key)
    raw, cl = r["comparisons"]["net_vs_raw"], r["comparisons"]["net_vs_classic"]
    assert (raw["n"], raw["net_wins"], raw["win_rate"]) == (25, 18, 0.72)
    assert raw["ci95_wilson"] == [pytest.approx(0.5242, abs=1e-4), pytest.approx(0.8572, abs=1e-4)]
    assert raw["meets_criterion"] is True and raw["verdict_vs_50"] == "mang_thang"
    assert (cl["net_wins"], cl["win_rate"], cl["meets_criterion"]) == (14, 0.56, True)
    assert cl["verdict_vs_50"] == "chua_phan_dinh"  # CI chạm 50 %: không nói "thắng"
    assert r["consistency"] == r["consistency"] | {"n_repeats": 5, "agree": 4, "rate": 0.8}
    assert r["n_choices"] == 55


def _write(p: Path, doc: dict) -> Path:
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_ab_score_cli_exit_codes(tmp_path: Path) -> None:
    key, export = _sample_key_export()
    kp = _write(tmp_path / "key.json", key)
    assert ab_score.main([str(_write(tmp_path / "ok.json", export)), "--key", str(kp), "--json"]) == 0
    missing = export | {"choices": export["choices"][:-1]}
    assert ab_score.main([str(_write(tmp_path / "miss.json", missing)), "--key", str(kp)]) == 1
    wrong = export | {"key_id": "khac"}
    assert ab_score.main([str(_write(tmp_path / "wrong.json", wrong)), "--key", str(kp)]) == 1
    assert ab_score.main([str(tmp_path / "khong_co.json"), "--key", str(kp)]) == 2


# ---------------------------------------------------------------- trang thử mù


def _cands(n: int) -> list[dict]:
    bins = ("3-8", "8-14", ">14")
    return [{"frame": f"f{i:03d}.jpg", "alt_m": 5.0 + i, "alt_bin": bins[i % 3], "has_person": i % 2 == 0,
             "block": i // 10} for i in range(n)]


def _fns() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    def classic(f: np.ndarray) -> np.ndarray:
        return (f // 2).astype(np.uint8)

    def net(f: np.ndarray) -> np.ndarray:
        return np.full_like(f, 200)

    return {"raw": lambda f: f, "classic": classic, "net": net}


def _build(out: Path, **kw: object) -> dict:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    base[:, :16] = 5  # dải mép tối của khung gốc
    args = {"n_pairs": 55, "n_repeats": 5, "seed": 20261118, "crop_left": 16, "variants_info": {"x": 1}} | kw
    return ab_page.build(_cands(80), lambda _f: base.copy(), _fns(), out, **args)  # type: ignore[arg-type]


def test_ab_page_build_structure(tmp_path: Path) -> None:
    res = _build(tmp_path)
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    pairs = key["pairs"]
    assert res["n_pairs"] == len(pairs) == 55 and res["by_comparison"] == {"net_vs_raw": 25, "net_vs_classic": 25}
    orig = [p for p in pairs if p["repeat_of"] is None]
    reps = [p for p in pairs if p["repeat_of"] is not None]
    assert len(reps) == 5 and len({p["frame"] for p in orig}) == 50
    assert {p["comparison"] for p in reps} == {"net_vs_raw", "net_vs_classic"}
    per = Counter((p["alt_bin"], p["has_person"], p["comparison"]) for p in orig)
    for s in {(p["alt_bin"], p["has_person"]) for p in orig}:  # hai so sánh cân tầng (lệch ≤ 1)
        assert abs(per[(*s, "net_vs_raw")] - per[(*s, "net_vs_classic")]) <= 1, s
    pos = {p["id"]: i for i, p in enumerate(pairs)}
    by_id = {p["id"]: p for p in pairs}
    for r in reps:
        src = by_id[r["repeat_of"]]
        assert pos[r["id"]] - pos[src["id"]] > ab_page.MIN_REPEAT_GAP
        assert (r["left"], r["right"]) == (src["right"], src["left"])
        assert (r["left_img"], r["right_img"]) == (src["right_img"], src["left_img"])
    net_left = sum(p["left"] == "net" for p in orig)
    assert 10 <= net_left <= 40  # trái/phải ngẫu nhiên, không dồn một bên
    imgs = sorted((tmp_path / "img").glob("*.jpg"))
    assert len(imgs) == 100 and {f"img/{p.name}" for p in imgs} == set(key["images_sha256"])
    # hai ảnh của cùng khung đã cắt dải mép giống nhau: cỡ hiển thị 624×480 cho mọi biến thể
    import cv2

    assert {cv2.imread(str(p)).shape for p in imgs[:6]} == {(480, 624, 3)}
    assert key["key_id"] == ab_page.key_id_of(pairs)


def test_ab_page_html_is_blind_and_offline(tmp_path: Path) -> None:
    _build(tmp_path)
    html = (tmp_path / "ab-test.html").read_text(encoding="utf-8")
    _check_html(html)
    for p in (tmp_path / "img").iterdir():
        assert not any(w in p.name.lower() for w in VARIANT_WORDS), p.name
    data = json.loads(re.search(r"const PAIRS = (\[.*?\]);", html).group(1))  # type: ignore[union-attr]
    assert len(data) == 55 and all(set(d) == {"id", "l", "r"} for d in data)


def test_ab_page_refuses_overwrite_and_seed_is_reproducible(tmp_path: Path) -> None:
    a = _build(tmp_path / "a")
    with pytest.raises(ab_page.AbPageError):
        _build(tmp_path / "a")
    b = _build(tmp_path / "a", force=True)
    assert a["key_id"] == b["key_id"] and len(list((tmp_path / "a" / "img").glob("*.jpg"))) == 100
    with pytest.raises(enhance_eval.UsageError):
        _build(tmp_path / "c", n_pairs=54)


def test_ab_page_export_roundtrip_scores(tmp_path: Path) -> None:
    """Xuất giả lập đúng dạng trang tạo ra → ab_score đọc được."""
    _build(tmp_path)
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    export = {"format": ab_score.EXPORT_FORMAT, "key_id": key["key_id"],
              "choices": [{"id": p["id"], "choice": "left" if p["left"] == "net" else "right"} for p in key["pairs"]]}
    r = ab_score.score(export, key)
    assert r["comparisons"]["net_vs_raw"]["win_rate"] == 1.0 and r["consistency"]["rate"] == 1.0


def _check_html(html: str) -> None:
    low = html.lower()
    assert low.startswith("<!doctype html>") and "<head>" in low and "<body>" in low
    assert "http" not in low  # không URL mạng nào: mở thẳng file://
    assert "<script src" not in low and "<link" not in low  # không thư viện/tài nguyên ngoài
    for w in ("classic", "enhanc", "net_vs", '"raw"', "'raw'"):
        assert w not in low, w
    assert "localstorage" in low and "xuất kết quả" in low


# ---------------------------------------------------------------- file thật (Owner sinh bằng model thật)


@pytest.mark.data
def test_results_ab_html(require_data: Callable[..., Path]) -> None:
    html = require_data(AB_DIR / "ab-test.html", "python -m xla.eval.ab_page build --pairs 55 --seed 20261118")
    _check_html(html.read_text(encoding="utf-8"))


@pytest.mark.data
def test_results_ab_images_blind(require_data: Callable[..., Path]) -> None:
    key = json.loads(require_data(AB_DIR / "ab_key.json").read_text(encoding="utf-8"))
    names = sorted(p.name for p in require_data(AB_DIR / "img").glob("*.jpg"))
    assert names and {f"img/{n}" for n in names} == set(key["images_sha256"])
    for n in names:
        assert not any(w in n.lower() for w in VARIANT_WORDS), n
    assert key["n_pairs"] >= 55


@pytest.mark.data
def test_results_ab_result_matches_key(require_data: Callable[..., Path]) -> None:
    export = json.loads(require_data(AB_DIR / "ab_result.json", "chủ dự án chấm rồi bấm Xuất").read_text(
        encoding="utf-8"))
    key = json.loads(require_data(AB_DIR / "ab_key.json").read_text(encoding="utf-8"))
    assert len(export["choices"]) >= 55
    r = ab_score.score(export, key)  # lệch khoá / thiếu lượt → AbScoreError
    assert r["n_choices"] == len(key["pairs"])


@pytest.mark.data
def test_results_e4_has_six_cells(require_data: Callable[..., Path]) -> None:
    e4 = json.loads(require_data(E4_JSON, "python -m xla.eval.enhance_eval e4 --json").read_text(encoding="utf-8"))
    cells = e4["cells"]
    assert set(cells) == {"coco", "bundle"}
    for det in cells.values():
        assert set(det) == set(enhance_eval.VARIANTS)
        for c in det.values():
            assert c["mAP50"]["value"] is not None and len(c["mAP50"]["ci95"]) == 2
    assert e4["d5"]["recommended_input_variant"] in ("raw", "enhanced")


# ---------------------------------------------------------------- CLI đầu-cuối trên khung thật của repo, bundle giả


@pytest.fixture
def dummy_args(tmp_path: Path) -> list[str]:
    from xla.contract import make_dummy_bundle

    enh = make_dummy_bundle(tmp_path, "enhancer")  # graph = khung vào, cột mép lặp cột 16
    cl = tmp_path / "copy_classic.py"
    cl.write_text("def enhance(f):\n    return f.copy()\n", encoding="utf-8")
    return ["--enhancer", str(enh), "--classic", f"{cl}:enhance", "--providers", "cpu", "--boot", "20"]


def test_cli_identity_variants_change_nothing(dummy_args: list[str], tmp_path: Path) -> None:
    out = tmp_path / "phys.json"
    assert enhance_eval.main(["physical", *dummy_args, "--limit", "3", "--out", str(out), "--json"]) == 0
    m = json.loads(out.read_text(encoding="utf-8"))["physical"]["metrics"]
    for k in enhance_eval.PHYSICAL:
        assert m[k]["net"]["change_vs_raw"] == 0.0 and m[k]["classic"]["change_vs_raw"] == 0.0, k
    out = tmp_path / "fl.json"
    assert enhance_eval.main(["flicker", *dummy_args, "--limit", "10", "--out", str(out), "--json"]) == 0
    fl = json.loads(out.read_text(encoding="utf-8"))["flicker"]
    assert fl["n_runs"] >= 1
    assert fl["variants"]["net"]["flicker_ratio"] == 1.0 and fl["variants"]["classic"]["flicker_ratio"] == 1.0


def test_cli_exit_codes(dummy_args: list[str], tmp_path: Path) -> None:
    args = list(dummy_args)
    args[1] = str(tmp_path / "khong-co-bundle")
    assert enhance_eval.main(["physical", *args, "--limit", "1"]) == 2
    assert enhance_eval.main(["physical", *dummy_args, "--providers", "gpu-la", "--limit", "1"]) == 2
    bad = list(dummy_args)
    bad[3] = "xla_khong_ton_tai_123:enhance"
    assert enhance_eval.main(["physical", *bad, "--limit", "1"]) == 1
