"""A/B lần 2 phóng to (``ab_page --zoom``): vùng cắt đúng cỡ, nằm trên người, giống nhau hai bên, cặp lặp cùng vùng."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xla.eval import ab_page

CROP = 8


def _cands(n: int) -> list[dict]:
    out = []
    for i in range(n):
        person = i % 2 == 0
        boxes = np.array([[500.0, 400.0, 520.0, 440.0]]) if person else np.zeros((0, 4))
        out.append({"frame": f"f{i:03d}.jpg", "alt_m": 5.0 + (i % 4) * 4, "alt_bin": ["<6", "6-10", "10-14", ">14"][i % 4],
                    "has_person": person, "block": i % 3, "boxes": boxes})
    return out


def _fns() -> dict:
    return {"raw": lambda x: x, "net": lambda x: 255 - x, "classic": lambda x: x // 2}


def test_pick_region_centres_on_largest_person_and_clamps() -> None:
    rng = np.random.default_rng(0)
    boxes = np.array([[10.0, 10.0, 12.0, 12.0], [600.0, 460.0, 638.0, 479.0]])
    x0, y0, w, h = ab_page.pick_region(640 - CROP, 480, 2, boxes, CROP, rng)
    assert (w, h) == (316, 240)
    assert x0 + w == 640 - CROP and y0 + h == 480  # hộp lớn ở góc dưới phải → bị kẹp vào mép
    x0, y0, w, h = ab_page.pick_region(640 - CROP, 480, 2, np.zeros((0, 4)), CROP, rng)
    assert 0 <= x0 <= 640 - CROP - w and 0 <= y0 <= 480 - h


def test_zoom_build_same_region_both_sides_and_repeats(tmp_path: Path) -> None:
    yy, xx = np.mgrid[0:480, 0:640]
    base = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    res = ab_page.build(_cands(80), lambda _f: base.copy(), _fns(), tmp_path, n_pairs=25, n_repeats=5, seed=3,
                        crop_left=CROP, variants_info={}, zoom=2)
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    assert res["n_pairs"] == 25 and key["zoom"] == 2
    by_id = {p["id"]: p for p in key["pairs"]}
    for p in key["pairs"]:
        x0, y0, w, h = p["region"]
        assert (w, h) == ((640 - CROP) // 2, 240)
        if p["has_person"]:  # tâm hộp người (510 − CROP, 420) nằm trong vùng
            assert x0 <= 510 - CROP < x0 + w and y0 <= 420 < y0 + h
        if p["repeat_of"]:
            assert p["region"] == by_id[p["repeat_of"]]["region"]
        for side in ("left", "right"):
            img = cv2.imread(str(tmp_path / p[f"{side}_img"]))
            assert img.shape == (h * 2, w * 2, 3)
        if p["repeat_of"] is None and "raw" in (p["left"], p["right"]):
            raw_side = "left" if p["left"] == "raw" else "right"
            img = cv2.imread(str(tmp_path / p[f"{raw_side}_img"]))
            want = cv2.resize(base[:, CROP:][y0:y0 + h, x0:x0 + w], (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
            assert np.abs(img.astype(int) - want.astype(int)).mean() < 3  # chỉ sai số JPEG q95


def test_zoom_one_keeps_full_frame(tmp_path: Path) -> None:
    base = np.full((480, 640, 3), 100, np.uint8)
    ab_page.build(_cands(80), lambda _f: base.copy(), _fns(), tmp_path, n_pairs=25, n_repeats=5, seed=3,
                  crop_left=CROP, variants_info={})
    key = json.loads((tmp_path / "ab_key.json").read_text(encoding="utf-8"))
    assert key["zoom"] == 1 and all("region" not in p for p in key["pairs"])
    assert cv2.imread(str(tmp_path / key["pairs"][0]["left_img"])).shape == (480, 640 - CROP, 3)
