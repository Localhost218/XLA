"""Phase 11 AI v2: ảnh rõ offline ×2 (``xla.enhance.sr``, ``sr_pairs``, kiến trúc vendor, ``enhancer-offline-v1.0.0``)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from xla import contract
from xla.enhance import sr
from xla.enhance.arch.rrdbnet import RRDBNet
from xla.paths import repo_root

CFG = sr.load_config("configs/sr_x2.yaml")
BUNDLE = repo_root() / "models" / "release" / "enhancer-offline-v1.0.0"
RESULTS = repo_root() / CFG["results_dir"]


# ---- kiến trúc vendor + trọng số công bố --------------------------------------------------------------------------
@pytest.mark.data
@pytest.mark.parametrize("name", list(CFG["candidates"]))
def test_vendor_loads_published_weights_strict_with_checksum(name: str, require_data) -> None:
    spec = CFG["candidates"][name]
    require_data(Path(CFG["weights_dir"]) / spec["weights"], spec["url"])
    net = sr.build_candidate(CFG, name)  # sha256 khác config → ValueError; khoá lệch → strict load lỗi
    with torch.no_grad():
        y = net(torch.rand(1, 3, 16, 24))
    assert y.shape == (1, 3, 32, 48)


@pytest.mark.data
def test_discriminator_loads_published_weights(require_data) -> None:
    from xla.enhance.arch.rrdbnet import UNetDiscriminatorSN

    spec = CFG["discriminator"]
    require_data(Path(CFG["weights_dir"]) / spec["weights"], spec["url"])
    d = UNetDiscriminatorSN(3, 64, True)
    d.load_state_dict(sr.published_state(spec, Path(CFG["weights_dir"])), strict=True)


def test_published_state_rejects_wrong_checksum(tmp_path: Path) -> None:
    w = tmp_path / "w.pth"
    torch.save({"params_ema": {"x": torch.zeros(1)}}, w)
    spec = {"weights": "w.pth", "weights_key": "params_ema", "sha256": "0" * 64, "url": "-"}
    with pytest.raises(ValueError, match="sha256"):
        sr.published_state(spec, tmp_path)


# ---- graph §11 ---------------------------------------------------------------------------------------------------
def test_srgraph_output_shape_dtype_and_left_strip() -> None:
    net = RRDBNet(3, 3, scale=2, num_feat=8, num_block=1, num_grow_ch=4).eval()
    g = sr.SRGraph(net, crop_left=16, half=False).eval()
    frame = np.random.default_rng(0).integers(0, 256, contract.FRAME_SHAPE, dtype=np.uint8)
    out = sr.run_graph(g, [frame], device="cpu")[0]
    assert out.shape == contract.OFFLINE_SHAPE == (960, 1280, 3)
    assert out.dtype == np.uint8
    assert (out[:, :32] == out[:, 32:33]).all()  # cột [0, 2·crop_left) lặp cột 2·crop_left


def test_offline_meta_has_contract_required_keys() -> None:
    meta = sr.offline_meta(version="1.0.0", onnx_sha="a" * 64, crop_left=16, profile_version="1.2.0", train_run="t",
                           network={"candidate": "x"}, s_per_frame=0.2, peak_vram_mb=900.0, machine={"gpu": {}})
    for k in contract.META_KIND["enhancer-offline"]:
        assert k in meta, k
    for k in contract.META_COMMON:
        if k not in contract.META_COMMON_OPTIONAL["enhancer-offline"]:
            assert k in meta, k
    assert meta["output_size"] == [1280, 960] and meta["scale"] == 2 and meta["temporal_window"] == 1


# ---- select: luật chọn --------------------------------------------------------------------------------------------
def _row(name: str, fp_off: float, *, fp_raw: float = 0.5, speed: bool = True, flicker: bool = True,
         score: float = 0.0) -> dict:
    return {"candidate": name, "pass_speed": speed, "fp_cal_raw": fp_raw, "fp_cal_offline": fp_off,
            "pass_flicker": flicker, "score": score}


def test_choose_rejects_candidate_that_increases_false_positives() -> None:
    rows = [_row("tang_bao_nham", 0.51, score=9.0), _row("giu_nguyen", 0.50, score=1.0)]
    best, _ = sr.choose(rows)
    assert best is not None and best["candidate"] == "giu_nguyen"


def test_choose_rejects_slow_candidate_and_kills_when_none_left() -> None:
    best, how = sr.choose([_row("cham", 0.1, speed=False), _row("tang", 0.9)])
    assert best is None and "Kill (d)" in how


def test_choose_prefers_flicker_pass_then_score() -> None:
    rows = [_row("nhap_nhay", 0.4, flicker=False, score=5.0), _row("em", 0.4, score=1.0), _row("em2", 0.4, score=2.0)]
    best, _ = sr.choose(rows)
    assert best["candidate"] == "em2"


def test_choose_lowest_flicker_when_none_pass() -> None:
    rows = [{**_row("gan", 0.09, flicker=False, score=2.4), "flicker_ratio": 1.55},
            {**_row("khong_gan", 0.07, flicker=False, score=2.1), "flicker_ratio": 1.35}]
    best, how = sr.choose(rows)
    assert best["candidate"] == "khong_gan" and "thấp nhất" in how


def test_choose_never_picks_reference_row() -> None:
    ref = {**_row("ref_bicubic", 0.1, score=99.0), "kind": "reference"}
    best, _ = sr.choose([ref, _row("that", 0.4, score=0.5)])
    assert best["candidate"] == "that"


def test_hours_for_flight() -> None:
    assert sr.hours_for_flight(0.5, 6.0, 20) == pytest.approx(1.0)


# ---- cặp SR -------------------------------------------------------------------------------------------------------
def test_make_pair_geometry_and_determinism() -> None:
    from xla.degrade import d2_camera
    from xla.enhance.sr_pairs import make_pair

    pc = CFG["pairs"]
    d2_cfg = d2_camera.load_config(repo_root() / pc["d2_config"])
    assert int(d2_cfg["params_version"]) == int(pc["d2_params_version"])
    profile = d2_camera.load_profile(repo_root() / pc["profile"])
    img = np.random.default_rng(1).integers(0, 256, (1050, 1400, 3), dtype=np.uint8)
    lr1, hr1, info1 = make_pair(img, 7, d2_cfg, profile)
    lr2, hr2, info2 = make_pair(img, 7, d2_cfg, profile)
    assert lr1 == lr2 and (hr1 == hr2).all() and info1 == info2
    lr = cv2.imdecode(np.frombuffer(lr1, np.uint8), cv2.IMREAD_COLOR)
    assert lr.shape == contract.FRAME_SHAPE
    assert hr1.shape == (960, 1248, 3)
    with pytest.raises(ValueError):
        make_pair(img[:900], 7, d2_cfg, profile)


def _write_pair_set(root: Path, name: str, lr: np.ndarray, hr: np.ndarray) -> None:
    for sub, img in (("lr", lr), ("hr", hr)):
        d = root / sub / "train" / "images"
        d.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        (d / name).write_bytes(buf.tobytes())  # đuôi .jpg nhưng nội dung PNG: không mất mát cho phép so khớp


def test_pairpatches_scale2_maps_lr_columns_to_hr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xla.enhance.train_net import PairPatches

    monkeypatch.setenv("XLA_DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(2)
    lr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    hr = cv2.resize(lr[:, 16:], (1248, 960), interpolation=cv2.INTER_NEAREST)
    _write_pair_set(tmp_path, "a.jpg", lr, hr)
    cfg = {"data": {"input_set": "lr", "target_set": "hr", "patch": 32, "grid": 8, "flip": False}}
    ds = PairPatches(["a.jpg"], cfg, crop_left=16, length=8, scale=2)
    for i in range(8):
        a, b = ds[i]
        assert a.shape == (3, 32, 32) and b.shape == (3, 64, 64)
        up = torch.nn.functional.interpolate(a[None], scale_factor=2, mode="nearest")[0]
        assert torch.equal(up, b)


def test_pairpatches_scale1_unchanged_same_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xla.enhance.train_net import PairPatches

    monkeypatch.setenv("XLA_DATA_ROOT", str(tmp_path))
    img = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    _write_pair_set(tmp_path, "a.jpg", img, img)
    cfg = {"data": {"input_set": "lr", "target_set": "hr", "patch": 64, "grid": 8, "flip": True}}
    ds = PairPatches(["a.jpg"], cfg, crop_left=16, length=4)
    for i in range(4):
        a, b = ds[i]
        assert torch.equal(a, b)


# ---- bundle hoặc Kill (d) -----------------------------------------------------------------------------------------
@pytest.mark.data
def test_bundle_valid_with_parity_or_kill_recorded() -> None:
    status = RESULTS / "status.json"
    if not (BUNDLE / "meta.json").is_file():
        assert status.is_file(), "không có bundle thì results/v2/sr/status.json phải ghi Kill (d) kèm số đo"
        st = json.loads(status.read_text(encoding="utf-8"))
        assert str(st.get("killed", "")).startswith("d") and st.get("measurements")
        return
    assert contract.validate_bundle(BUNDLE) == []
    meta = json.loads((BUNDLE / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "enhancer-offline" and meta["output_size"] == [1280, 960] and meta["fp16"] is True
    assert meta["s_per_frame"] <= CFG["bench"]["max_s_per_frame"]
    exp = json.loads((RESULTS / "export.json").read_text(encoding="utf-8"))
    assert exp["onnx_sha256"] == meta["onnx_sha256"]
    assert exp["parity"]["ok"] and exp["parity"]["ort_cpu_vs_torch_mae_worst"] <= 1.0


@pytest.mark.data
def test_gpu_ledger_within_phase_cap() -> None:
    assert sr.gpu_hours_used(CFG) <= float(CFG["gpu_budget_hours"])
