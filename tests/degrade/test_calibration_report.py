"""Accept phase 07: báo cáo hiệu chỉnh D2 (`results/realism/calibration.json`), so D1/D2 (`d1_vs_d2.json`), config D2,
và các mảnh mã mới (ghi YAML, gộp số đo, Real-SR, chia val theo chuỗi)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from xla import paths
from xla.degrade import calibrate, d2_camera, realsr, sources
from xla.degrade.sources import SourceItem

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "results" / "realism" / "calibration.json"
CMP = ROOT / "results" / "realism" / "d1_vs_d2.json"
D2_CFG = ROOT / "configs" / "degrade_d2.yaml"
SPLIT = ROOT / "data" / "real" / "split.json"


def _split_names(name: str) -> set[str]:
    return {f["frame"] for f in json.loads(SPLIT.read_text(encoding="utf-8"))["frames"] if f.get("split") == name}


@pytest.fixture(scope="module")
def cal() -> dict:
    return json.loads(CAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cmp_() -> dict:
    return json.loads(CMP.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- báo cáo

def test_calibration_used_dev_frames_only(cal) -> None:
    used = set(cal["real"]["frames_used"])
    assert cal["real"]["split"] == "dev" and len(used) == cal["real"]["n_frames"] > 0
    assert not used & _split_names("test"), "khung hiệu chỉnh không được thuộc test"
    assert used <= _split_names("dev")


def test_calibration_distance_not_worse(cal) -> None:
    assert cal["after"]["total"] <= cal["before"]["total"]
    assert cal["improved"] is (cal["after"]["total"] < cal["before"]["total"])
    assert len(cal["history"]) == cal["optimizer"]["evals"] >= 2
    assert min(h["total"] for h in cal["history"]) == pytest.approx(cal["after"]["total"])


def test_d2_config_valid_and_params_version_bumped(cal) -> None:
    cfg = d2_camera.load_config(D2_CFG)
    assert int(cfg["params_version"]) == cal["after"]["params_version"] > cal["before"]["params_version"]
    for s in cal["space"]:
        v = calibrate.get_key(cfg, s["key"])
        want = cal["after"]["values"][s["name"]]
        got = (v[0] + v[1]) / 2 if s.get("mode") == "center" and v[0] > 0 else v
        if s.get("mode") == "center" and v[0] == 0:
            continue  # tâm sát 0: khoảng bị cắt ở 0, không suy lại được tâm
        assert got == pytest.approx(want, abs=2e-3), s["name"]
    prof = d2_camera.load_profile(ROOT / "configs" / "camera_profile.yaml")
    win = np.random.default_rng(0).integers(0, 255, (480, 624, 3), dtype=np.uint8)
    frame, q, _ = d2_camera.degrade(win, np.random.default_rng(1), cfg, prof)
    assert frame.shape == (480, 640, 3)
    data, _ = d2_camera.encode_final(frame, q, prof)
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape == (480, 640, 3)


def test_d1_vs_d2_report_fields(cmp_) -> None:
    rows = {r["row"]: r for r in cmp_["rows"]}
    assert {"d1", "d2", "real_dev"} <= set(rows)
    for name in ("d1", "d2", "real_dev"):
        r = rows[name]
        for tier in ("c2st", "kid"):
            for k in ("mean", "ci_low", "ci_high", "n_patches"):
                assert k in r[tier], (name, tier, k)
            assert r[tier]["ci_low"] <= r[tier]["mean"] <= r[tier]["ci_high"]
        assert r["kid"]["n_repeats"] == 10 and r["kid"]["seed"] == r["seed"]
        assert r["n_patches"] == rows["d2"]["n_patches"] and r["seed"] == rows["d2"]["seed"]
        assert np.isfinite(r["stats"]["total"])
    assert cmp_["real"]["split"] == "test" and cmp_["real"]["labels_used"] is False
    assert cmp_["smoke"]["gpu_seconds"] > 0 and cmp_["smoke"]["n_patches"] == 1000


@pytest.mark.data
def test_generated_sets_share_source_units(require_data) -> None:
    units = {}
    for s in ("d0", "d1", "d2"):
        man = require_data(paths.data_root() / s / "manifest.jsonl", "build --set all")
        recs = [json.loads(line) for line in man.read_text(encoding="utf-8").splitlines() if line.strip()]
        units[s] = {(r["unit"], r["split"]) for r in recs}
    assert units["d0"] == units["d1"] == units["d2"] and len(units["d0"]) > 1000


# ----------------------------------------------------------------------------- mã

def test_set_yaml_value_keeps_comments() -> None:
    text = "version: \"0.1.0\"\nparams_version: 1   # c\nhaze:\n  dark_scale: 1.0   # núm\n  t_range: [0.2, 1.0]\nlens:\n" \
           "  dark_scale: 9\n"
    out = calibrate.set_yaml_value(text, "haze.dark_scale", 0.8765)
    out = calibrate.set_yaml_value(out, "haze.t_range", [0.25, 0.9])
    out = calibrate.set_top_value(out, "params_version", "2")
    d = yaml.safe_load(out)
    assert d["haze"] == {"dark_scale": 0.8765, "t_range": [0.25, 0.9]} and d["lens"]["dark_scale"] == 9
    assert d["params_version"] == 2 and "# núm" in out and "# c" in out
    with pytest.raises(KeyError):
        calibrate.set_yaml_value(text, "haze.missing", 1.0)


def test_space_center_mode_and_profile_lower_bound() -> None:
    cfg = {"lens": {"s": [0.85, 1.15]}, "tone": {"g": 2.2},
           "calibration": {"params": [{"name": "s", "key": "lens.s", "mode": "center", "lo": 0.3, "hi": 1.6},
                                      {"name": "g", "key": "tone.g", "lo_from_profile": "color.w", "lo": 1.0,
                                       "hi": 2.6}]}}
    space = calibrate.resolve_space(cfg, {"color": {"w": 1.57}})
    assert space[0]["x0"] == pytest.approx(1.0) and space[0]["half_width"] == pytest.approx(0.15)
    assert space[1]["lo"] == pytest.approx(1.57) and space[1]["x0"] == pytest.approx(1.57)
    new = calibrate.apply_values(cfg, space, [0.5, 2.0])
    assert new["lens"]["s"] == [0.35, 0.65] and new["tone"]["g"] == 2.0 and cfg["tone"]["g"] == 2.2
    u = calibrate.to_unit(space, [0.5, 2.0])
    assert calibrate.from_unit(space, u) == pytest.approx([0.5, 2.0])


def test_cma_resumes_from_checkpoint_identically(tmp_path) -> None:
    space = [{"name": f"p{i}", "key": f"b.p{i}", "lo": 0.0, "hi": 1.0, "x0": 0.9} for i in range(3)]
    base = {"b": {f"p{i}": 0.9 for i in range(3)}}

    def ev(cfgs):
        return [{"total": float(sum((c["b"][f"p{i}"] - 0.3) ** 2 for i in range(3))), "items": {}} for c in cfgs]

    def quiet(_):
        return None

    full = calibrate.run_cma(ev, base, space, 22, 0.2, 5, quiet)
    ck, pr = tmp_path / "c.pkl", tmp_path / "c.progress.jsonl"
    part = calibrate.run_cma(ev, base, space, 8, 0.2, 5, quiet, ck, pr)   # "bị ngắt" sau 8 lượt
    resumed = calibrate.run_cma(ev, base, space, 22, 0.2, 5, quiet, ck, pr)
    assert len(part[0]) == 8 and [h["total"] for h in resumed[0]] == pytest.approx([h["total"] for h in full[0]])
    assert resumed[1]["total"] == pytest.approx(full[1]["total"])
    assert full[1]["total"] < full[2]["total"]
    assert len(pr.read_text(encoding="utf-8").splitlines()) == len(resumed[0])


def test_in_memory_measure_matches_lowlevel_measure_set(tmp_path) -> None:
    from xla.eval.lowlevel import measure_set

    rng = np.random.default_rng(3)
    files = []
    for i in range(2):
        img = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 3)
        p = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        files.append(p)
    a = measure_set(files, 16)
    b = calibrate.aggregate([calibrate.measure_bytes(p.read_bytes()) for p in files])
    assert a["scalars"].keys() == b["scalars"].keys()
    for k in a["scalars"]:
        np.testing.assert_allclose(a["scalars"][k], b["scalars"][k], equal_nan=True)
    assert a["nlf"] == b["nlf"] and a["psd"] == b["psd"] and a["headers"] == b["headers"]


def test_realsr_residual_has_no_block_dc_and_offset_breaks_grid() -> None:
    rng = np.random.default_rng(0)
    patch = rng.random((32, 32, 3)).astype(np.float32) + np.kron(rng.random((4, 4, 1)), np.ones((8, 8, 1)))
    r = realsr.block_mean_residual(patch)
    assert np.abs(r.reshape(4, 8, 4, 8, 3).mean(axis=(1, 3))).max() < 1e-5
    pool = {"residual": np.full((5, 32, 32, 3), 0.01, np.float32), "y_mean": np.full(5, 120.0, np.float32)}
    pool["by_bin"] = [np.flatnonzero(realsr.bin_of(pool["y_mean"]) == b) for b in range(realsr.N_BINS)]
    img = np.full((480, 624, 3), 0.5, np.float32)
    out, extra = realsr.inject(img, rng, pool, 2.0)
    assert np.allclose(out, 0.52) and all(o % 8 for o in extra["realsr_offset"])


def _item(seq: str, split: str, n: int, person: bool) -> SourceItem:
    return SourceItem(f"visdrone/{seq}_{n}", "visdrone", f"x/{seq}_00001_d_{n:07d}.jpg", split, (10, 10),
                      [[0, 0, 1, 1]] if person else [], [])


def test_pick_val_sequences_whole_sequences_deterministic() -> None:
    items = [_item(f"{s:07d}", "train", n, n % 3 != 0) for s in range(20) for n in range(10)]
    items += [_item("0000999", "val", n, True) for n in range(5)]
    a = sources.pick_val_sequences(items, 7, 30, 40)
    assert a == sources.pick_val_sequences(items, 7, 30, 40)
    assert all(s.startswith("train/") for s in a)
    extra = sum(bool(i.boxes) for i in items if sources.visdrone_seq(i) in set(a))
    assert 25 <= extra <= 40
    with pytest.raises(ValueError):
        sources.pick_val_sequences(items, 7, 10_000, 40)


def test_repo_val_split_moves_only_train_sequences() -> None:
    vs = yaml.safe_load((ROOT / "configs" / "visdrone_valsplit.yaml").read_text(encoding="utf-8"))
    assert vs["val_with_person_after"] >= 200 and len(vs["train_to_val"]) == vs["moved_sequences"]
    assert all(s.startswith("train/") for s in vs["train_to_val"])
    assert sources.load_config(ROOT / "configs" / "sources.yaml")["val_split_file"] == "configs/visdrone_valsplit.yaml"
