"""Phase 06 AI v2: chọn trò bằng OOF k-fold (``xla.detect.kfold``), chặn rò test/cal/holdout, bundle ``detector-v2.0.0``.

Ca đầu dùng dữ liệu thật đã commit + CSV giả; ca cuối đọc ``results/v2/detect/`` và bundle (phase 06 tạo).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from xla.detect import kfold as K
from xla.eval import detection as D
from xla.paths import repo_root

ROOT = repo_root()
RES = ROOT / "results" / "v2" / "detect"
BUNDLE = ROOT / "models" / "release" / "detector-v2.0.0"


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "student_v2.yaml").read_text("utf-8"))


def _neg(kind: str) -> str:
    doc = json.loads((ROOT / "data/real/neg_flight_split.json").read_text("utf-8"))
    return next(f["frame"] for f in doc["frames"] if f["split"] == kind)


def _csv(path: Path, frames: list[str]) -> Path:
    path.write_text("frame,x1,y1,x2,y2,conf\n" + "".join(f"{f},100,100,120,140,0.5\n" for f in frames), "utf-8")
    return path


def test_config_valid() -> None:
    c = _cfg()
    assert set(c["candidates"]) == {"C1", "C2", "C3"}
    assert c["candidates"]["C2"]["pseudo"] and not c["candidates"]["C1"]["pseudo"]
    assert float(c["gpu_cap_h"]) <= 2.0 and float(c["extension_h"]) <= 0.75
    assert int(c["train"]["imgsz"]) == 1280 and int(c["train"]["workers"]) <= 4
    assert c["export"]["version"] == "2.0.0" and c["export"]["out"].endswith("detector-v2.0.0")


@pytest.mark.parametrize("kind", ["test", "cal", "outside"])
def test_select_rejects_oof_with_test_cal_or_foreign(tmp_path, kind) -> None:
    dp = D.devpool_frames()
    split = {f["frame"]: f["split"] for f in json.loads((ROOT / "data/real/split.json").read_text("utf-8"))["frames"]}
    bad = {"test": next(f for f, s in split.items() if s == "test"), "cal": _neg("cal"),
           "outside": next(f for f, s in split.items() if s == "neg")}[kind]
    with pytest.raises(K.OOFLeakError):
        K.check_oof_frames(_csv(tmp_path / "p.csv", dp[:5] + [bad]), ROOT)


def test_oof_accepts_devpool_and_mine(tmp_path) -> None:
    got = K.check_oof_frames(_csv(tmp_path / "p.csv", D.devpool_frames()[:3] + [_neg("mine")]), ROOT)
    assert len(got) == 4


def test_f1_threshold_rejects_cal_frames() -> None:
    cal = _neg("cal")
    fe = D.FrameEval(frame=cal, n_gt=0)
    with pytest.raises(D.TestLeakError):
        D.select_f1_threshold([fe], {cal: "dev"}, D.reserved_frames())


def test_devpool_eval_has_no_reserved_negatives() -> None:
    assert not set(D.mine_frames()) & D.reserved_frames()
    assert not set(D.devpool_frames()) & D.reserved_frames()


def test_devpool_evaluate_runs_and_threshold_is_oof(tmp_path) -> None:
    from xla.eval.detection import load_labels

    gt, _, _ = load_labels(ROOT / "data/real/labels", "devpool", D.devpool_frames(), allow_missing=False)
    lines = ["frame,x1,y1,x2,y2,conf"]
    rng = np.random.default_rng(0)
    for f, b in gt.items():
        for x1, y1, x2, y2 in b:
            lines.append(f"{f},{x1},{y1},{x2},{y2},{rng.uniform(0.3, 0.9):.3f}")
    p = tmp_path / "gt.csv"
    p.write_text("\n".join(lines) + "\n", "utf-8")
    ev = D.evaluate(p, "devpool", boot_n=0)
    assert ev["metrics"]["mAP50"]["value"] > 0.99
    assert ev["f1_threshold_from_devval"]["source"] == "devpool OOF"


def test_pick_winner_prefers_simpler_on_overlap() -> None:
    def t(lo: float, hi: float) -> dict:
        return {"metrics": {"mAP50": {"value": (lo + hi) / 2, "ci95": [lo, hi]}}}

    assert K.pick_winner({"C1": t(0.3, 0.5), "C3": t(0.2, 0.35), "C2": t(0.31, 0.6)})[0] == "C3"
    assert K.pick_winner({"C1": t(0.3, 0.5), "C3": t(0.1, 0.2)})[0] == "C1"


def test_c2_fold_excludes_fold_sessions() -> None:
    ctx = K.real_context(ROOT)
    assert len(ctx["folds"]) == 2
    for i, fold in enumerate(ctx["folds"]):
        assert {ctx["session"][f] for f in fold} == ctx["fold_sessions"][i]


# ---------------------------------------------------------------- kết quả phase 06 (đã commit)


def test_select_json_and_readme() -> None:
    sel = json.loads((RES / "select.json").read_text("utf-8"))
    assert sel["winner"] in sel["candidates"]
    for c in sel["candidates"].values():
        assert c["metrics"]["mAP50"]["ci95"][0] is not None
    text = (RES / "README.md").read_text("utf-8").lower()
    for key in ("oof", "test", "ms", "sahi", "giờ gpu"):
        assert key in text, key


def test_bundle_validates_or_kill_b_documented() -> None:
    sel = json.loads((RES / "select.json").read_text("utf-8"))
    if sel["kill_b"]["triggered"]:
        assert not (BUNDLE / "meta.json").exists()
        assert "Kill (b)" in (RES / "README.md").read_text("utf-8")
        return
    from xla import contract as C

    if not (BUNDLE / "detector.onnx").is_file():
        pytest.skip("detector.onnx không vào git (.gitignore): chỉ kiểm được ở máy đã xuất / sau khi tải Release")
    assert C.validate_bundle(BUNDLE) == []
    meta = json.loads((BUNDLE / "meta.json").read_text("utf-8"))
    assert meta["version"] == "2.0.0" and "CUDA" in meta["ms_per_frame"]["runtime"]
    assert meta["ms_per_frame"]["p95"] <= 30.0
    assert abs(meta["conf_threshold"] - sel["oof_threshold"]) < 1e-6
