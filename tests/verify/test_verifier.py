"""Phase 07 AI v2: verifier (``xla.verify``) — manifest sạch cal/holdout/test, hình học vùng cắt §9, graph N = 0,
parity ORT ↔ PyTorch, bundle ``verifier-v1.0.0`` (meta §9, ``xla.contract validate``).

Ca graph dùng backbone khởi tạo ngẫu nhiên (không cần trọng số, không cần GPU); ca bundle cần ``verifier.onnx`` (ngoài
git, Release ``models-v2.0.0``) thì mới chạy phần ORT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from xla import contract as C
from xla.paths import repo_root
from xla.verify import common as V
from xla.verify import mine as M

ROOT = repo_root()
RES = ROOT / "results" / "v2" / "verifier"
BUNDLE = ROOT / "models" / "release" / "verifier-v1.0.0"
MIN_SCORE_SPREAD = 0.05


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "verifier.yaml").read_text("utf-8"))


@pytest.fixture(scope="module")
def rows(cfg: dict) -> list[dict]:
    return M.load_manifest(cfg)


@pytest.fixture(scope="module")
def rs() -> dict:
    return V.real_sets()


def test_config_candidates_follow_deferred_10(cfg: dict) -> None:
    cands = cfg["train"]["candidates"].values()
    assert {c["backbone"] for c in cands} <= {"mobilenet_v3_small", "efficientnet_b0"}
    assert {c["crop_px"] for c in cands} <= set(C.VERIFIER_CROP_PX)
    assert cfg["train"]["workers"] <= 4
    assert cfg["select"]["max_recall_drop"] == 0.05 and cfg["select"]["kill_min_fp_reduction"] == 0.5
    assert cfg["select"]["cal_fp_target"] == 0.03


def test_manifest_has_no_cal_holdout_reserved_or_test_frame(rows: list[dict], rs: dict) -> None:
    bad = V.forbidden_frames(rs)
    assert len(bad) >= len(rs["cal"]) + len(rs["test"])
    real = [r for r in rows if r["src"] in ("mine", "devpool")]
    assert real, "manifest không có mẫu thật"
    assert not {r["image"] for r in real} & bad
    assert M.check_manifest(rows, rs) == []
    allowed = set(rs["mine"]) | set(rs["devpool"])
    assert {r["image"] for r in real} <= allowed


def test_check_manifest_flags_a_cal_frame(rows: list[dict], rs: dict) -> None:
    fake = dict(rows[0], src="mine", image=rs["cal"][0], id="min-fake")
    errs = M.check_manifest([fake], rs)
    assert len(errs) == 1 and "cal/holdout" in errs[0]


def test_manifest_folds_and_labels(rows: list[dict], rs: dict) -> None:
    for r in rows:
        if r["src"] in ("mine", "devpool"):
            assert r["fold"] == rs["fold_of"][r["image"]] and 0 <= r["fold"] < rs["n_folds"]
        else:
            assert r["fold"] == -1
    assert all(r["label"] == 0 for r in rows if r["src"] == "mine"), "khung mine không người: chỉ có âm"
    assert all(r["label"] == 1 for r in rows if r["src"] == "tinyperson")
    assert len({r["id"] for r in rows}) == len(rows)
    # âm mine chia theo phiên: một phiên -> một fold
    sess = {}
    for r in rows:
        if r["src"] == "mine":
            assert sess.setdefault(rs["session_of"][r["image"]], r["fold"]) == r["fold"]


def test_review_drops_are_absent(rows: list[dict], cfg: dict) -> None:
    p = RES / M.REVIEW_CSV
    ids = {ln.split(",")[0] for ln in p.read_text("utf-8").splitlines()[1:] if ",drop," in ln}
    assert ids and not ids & {r["id"] for r in rows}
    assert json.loads((RES / "mine.json").read_text("utf-8"))["n"] == len(rows)


def test_context_boxes_numpy_matches_torch_and_contract() -> None:
    import torch

    b = np.asarray([[100, 100, 140, 180], [0, 0, 4, 6], [630, 470, 650, 500], [300, 200, 301, 201]], np.float32)
    n = V.context_boxes(b, 1.5, 32.0)
    t = V.context_boxes(torch.from_numpy(b), 1.5, 32.0).numpy()
    np.testing.assert_allclose(n, t, atol=1e-5)
    np.testing.assert_allclose(n[0], [60, 80, 180, 200])  # cạnh 80 * 1,5 = 120 quanh tâm (120, 140)
    np.testing.assert_allclose(n[3], [284.5, 184.5, 316.5, 216.5])  # min_side 32
    assert n[:, 0].min() >= C.CROP_LEFT_V1 and n[:, 2].max() <= C.FRAME_W and n[:, 3].max() <= C.FRAME_H


def test_reference_crop_matches_contract_roialign() -> None:
    """``V.roi_crops`` (PyTorch ``roi_align(aligned=False)``) = graph RoiAlign §9 của ``xla.contract`` trên ORT CPU."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, C.FRAME_SHAPE, dtype=np.uint8)
    boxes = V.context_boxes(np.asarray([[100, 100, 140, 180], [20, 0, 70, 40], [500, 300, 530, 330]], np.float32),
                            1.5, 32.0)
    for cp in C.VERIFIER_CROP_PX:
        import onnxruntime as ort

        s = ort.InferenceSession(C.roi_align_probe_model(cp).SerializeToString(), providers=["CPUExecutionProvider"])
        got = s.run(None, {C.INPUT_NAME: frame, C.BOXES_NAME: boxes})[0]
        np.testing.assert_allclose(got, V.roi_crops(frame, boxes, cp), atol=1e-3)


@pytest.mark.parametrize("backbone,crop_px", [("mobilenet_v3_small", 64), ("efficientnet_b0", 96)])
def test_graph_exports_with_dynamic_n_and_parity(tmp_path: Path, backbone: str, crop_px: int) -> None:
    import onnx
    import torch

    from xla.verify import export_onnx as E
    from xla.verify import train as T

    torch.manual_seed(0)
    bb = T.build_backbone(backbone, pretrained=False)
    # Backbone ngẫu nhiên mà để thống kê BN tuỳ ý (mean ~0, var ~1) thì tín hiệu tắt dần qua các khối: effb0 ra đặc
    # trưng gần như không phụ thuộc ảnh (độ lệch giữa các vùng cắt ~1e-9), 4 điểm chỉ khác nhau 1 ulp float32 và CI
    # Linux ra hằng 0,64185 (26/11/2026). Hiệu chỉnh lại BN bằng vài lô ngẫu nhiên (train mode, trung bình cộng dồn)
    # cho mỗi tầng ra đúng thang đo, nên điểm khác nhau thật và phép so parity mới có nghĩa.
    for m in bb.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = None
    bb.train()
    with torch.no_grad():
        for _ in range(4):
            bb(torch.randn(16, 3, crop_px, crop_px))
        for m in bb.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1:  # đầu ra; gốc std 0,01 -> điểm gần hằng 0,5
                m.weight.normal_(0, 0.5)
    mod = E.graph_module(bb.eval(), crop_px, 1.5, 32.0)
    p = tmp_path / "verifier.onnx"
    E.export(mod, p)
    m = onnx.load(str(p))
    assert [i.name for i in m.graph.input] == [C.INPUT_NAME, C.BOXES_NAME]
    assert [o.name for o in m.graph.output] == ["scores"]
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, C.FRAME_SHAPE, dtype=np.uint8)
    feeds = {"probe": (np.zeros(C.FRAME_SHAPE, np.uint8), np.asarray([C.VERIFIER_PROBE_BOX], np.float32)),
             "many": (frame, np.asarray([[100, 100, 140, 180], [0, 0, 10, 10], [600, 450, 650, 500],
                                         [300, 200, 360, 330]], np.float32)),
             "empty": (frame, np.zeros((0, 4), np.float32))}
    par = E.parity(mod, p, feeds, cuda=False)
    assert par["cpu"]["pass"], par
    s = E.ort_session(p, cuda=False)
    out = s.run(None, {C.INPUT_NAME: frame, C.BOXES_NAME: np.zeros((0, 4), np.float32)})[0]
    assert out.shape == (0,) and out.dtype == np.float32
    many = s.run(None, {C.INPUT_NAME: frame, C.BOXES_NAME: feeds["many"][1]})[0]
    assert many.shape == (4,) and np.all((many >= 0) & (many <= 1))
    # Ngưỡng cỡ 1e6 lần nhiễu làm tròn float32 (1 ulp quanh 0,64 = 6e-8): điểm phải phụ thuộc nội dung vùng cắt thật.
    assert np.ptp(many) > MIN_SCORE_SPREAD, f"điểm gần hằng {many} (ptp {np.ptp(many):.3g}) — graph không nhìn ảnh"


def _select() -> dict:
    return json.loads((RES / "select.json").read_text("utf-8"))


def test_select_and_kill_c_consistent_with_bundle() -> None:
    sel = _select()
    assert sel["chosen"] in sel["candidates"]
    r = sel["candidates"][sel["chosen"]]
    a = r["at_recall_drop"]
    assert r["before"]["oof"]["recall"] - a["oof"]["recall"] <= 0.05 + 1e-9
    assert sel["kill_c"] == (a["fp_reduction_cal"] < 0.5)
    assert (BUNDLE / "meta.json").is_file() == (not sel["kill_c"]), "Kill (c) ↔ không có bundle"


@pytest.mark.skipif(not (BUNDLE / "meta.json").is_file(), reason="Kill (c): không có bundle")
def test_bundle_meta_section_9() -> None:
    meta = json.loads((BUNDLE / "meta.json").read_text("utf-8"))
    sel = _select()
    for k in ("crop_px", "context_scale", "min_side_px", "crop_left", "conf_threshold", "input_variant", "backbone",
              "train_run", "metrics"):
        assert k in meta, k
    assert meta["model"] == "verifier" and meta["version"] == "1.0.0" and meta["contract_version"] == "1.1"
    assert meta["input_variant"] == "raw" and meta["crop_px"] in C.VERIFIER_CROP_PX
    assert set(meta["metrics"]) >= set(C.METRICS_KEYS["verifier"])
    assert all(isinstance(meta["metrics"][k], float) for k in C.METRICS_KEYS["verifier"])
    cfg = yaml.safe_load((ROOT / "configs" / "verifier.yaml").read_text("utf-8"))
    chosen = cfg["train"]["candidates"][sel["chosen"]]
    assert meta["backbone"] == chosen["backbone"] and meta["crop_px"] == chosen["crop_px"]
    assert meta["train_run"].endswith(f"{sel['chosen']}_final")
    ms = meta["ms_per_frame"]
    assert set(ms["by_n"]) == {"1", "5", "20"} and "machine" in ms
    assert meta["export"]["parity"]["cpu"]["pass"] and meta["export"]["parity"]["cuda"]["pass"]
    boxes = json.loads((BUNDLE / "golden" / "boxes.json").read_text("utf-8"))
    assert any(len(v) == 0 for v in boxes.values()), "golden phải có một khung N = 0"


@pytest.mark.skipif(not (BUNDLE / "verifier.onnx").is_file(), reason="verifier.onnx ngoài git (Release)")
def test_bundle_validates_and_runs_n0() -> None:
    assert C.validate_bundle(BUNDLE) == []
    from xla.verify import export_onnx as E

    s = E.ort_session(BUNDLE / "verifier.onnx", cuda=False)
    z = s.run(None, {C.INPUT_NAME: np.zeros(C.FRAME_SHAPE, np.uint8), C.BOXES_NAME: np.zeros((0, 4), np.float32)})[0]
    assert z.shape == (0,)
