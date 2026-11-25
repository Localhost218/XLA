"""Soát tay âm báo nhầm d02x của verifier (``ext_neg_det``: hộp detector-v1 có IoU < 0,1 với nhãn bộ ngoài): người
không nhãn có lọt vào tập âm không (README §8 mục 3).

AX::

    python -m xla.verify.neg_audit sheet --n 200 --seed 20261126 --out results/v2/verifier/neg_audit --json
    #   mẫu ngẫu nhiên có seed từ manifest.jsonl → sample.json + sheet_NN.jpg; mỗi ô ba khung: vùng cắt verifier thấy
    #   (ảnh d02x, ngữ cảnh 3×) | cùng hộp trên ảnh NGUỒN sạch độ phân giải thật (ngữ cảnh 3×) | nguồn phóng hộp 1,5×
    python -m xla.verify.neg_audit score --out results/v2/verifier/neg_audit --json
    #   đọc review.csv (id, person: 1/0/? , note) do người soát ghi → tỉ lệ + Wilson 95 % → audit.json
    # exit 0 xong; 1 thiếu / lệch dữ liệu (review thiếu id, nhãn lạ); 2 sai cách gọi
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from xla import contract as C
from xla.eval.ab_score import wilson
from xla.paths import repo_root
from xla.verify import common as V

NEG_TYPE = "ext_neg_det"
SAMPLE_FILE = "sample.json"
REVIEW_FILE = "review.csv"
AUDIT_FILE = "audit.json"
PANEL_PX = 200
COLS, ROWS = 2, 6
CTX_SCALE, ZOOM_SCALE, MIN_SIDE = 3.0, 1.5, 40.0
LABELS = {"1": 1, "0": 0, "?": None}
SHEET_JPEG_Q = 90  # PNG 17 trang ~30 MB, quá nặng cho git
MATERIAL_LO = 0.05  # cận dưới Wilson > 5 % → coi là đáng kể (brief 26/11)


class AuditError(Exception):
    """Dữ liệu thiếu / lệch → exit 1."""


def source_of(row: dict[str, Any]) -> str:
    """``d0__heridal__test_…`` → ``heridal``; ``d2__visdrone__…`` → ``visdrone``."""
    parts = row["image"].split("__")
    return parts[1] if len(parts) > 2 else "?"


def draw_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` âm ``ext_neg_det`` ngẫu nhiên có seed (không lặp), giữ thứ tự rút."""
    pool = sorted((r for r in rows if r["src"] == "d02x" and r["type"] == NEG_TYPE), key=lambda r: r["id"])
    if len(pool) < n:
        raise AuditError(f"chỉ có {len(pool)} âm {NEG_TYPE}, cần {n}")
    return random.Random(seed).sample(pool, n)


def _panel(im: np.ndarray, box: list[float], scale: float) -> np.ndarray:
    import cv2

    ctx = V.context_boxes(np.asarray([box], np.float32), scale, MIN_SIDE, 0, im.shape[1], im.shape[0])
    crop = np.ascontiguousarray(V.roi_crops(im, ctx, PANEL_PX)[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8))
    kx, ky = PANEL_PX / float(ctx[0, 2] - ctx[0, 0]), PANEL_PX / float(ctx[0, 3] - ctx[0, 1])  # mép ảnh: ô không vuông
    x0, y0 = (box[0] - ctx[0, 0]) * kx, (box[1] - ctx[0, 1]) * ky
    x1, y1 = (box[2] - ctx[0, 0]) * kx, (box[3] - ctx[0, 1]) * ky
    cv2.rectangle(crop, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
    return crop


def source_box(row: dict[str, Any], m: dict[str, Any]) -> list[float]:
    """Hộp trên ảnh 640×480 d02x → px ảnh nguồn gốc: ``crop`` (ô HERIDAL) + (x − ``x_lo`` + ``window.x0``) / ``scale``
    (cột ``x_lo`` bên trái là dải đen của khung; kiểm trên nhãn HERIDAL test_BLI_0001: khớp từng px)."""
    w, c = m["window"], m.get("crop") or [0, 0]
    xs = [c[0] + (row["box"][i] - row["x_lo"] + w["x0"]) / w["scale_x"] for i in (0, 2)]
    ys = [c[1] + (row["box"][i] + w["y0"]) / w["scale_y"] for i in (1, 3)]
    return [xs[0], ys[0], xs[1], ys[1]]


def _cut(im: np.ndarray, box: list[float], scale: float, min_side: float) -> np.ndarray:
    """Vùng vuông quanh hộp (cạnh = max(cạnh hộp × ``scale``, ``min_side``)) trên ảnh lớn, đổi cỡ ``PANEL_PX``."""
    import cv2

    ctx = V.context_boxes(np.asarray([box], np.float32), scale, min_side, 0, im.shape[1], im.shape[0])[0]
    x0, y0, x1, y1 = (round(float(v)) for v in ctx)
    crop = cv2.resize(im[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)], (PANEL_PX, PANEL_PX), interpolation=cv2.INTER_CUBIC)
    kx, ky = PANEL_PX / max(x1 - x0, 1), PANEL_PX / max(y1 - y0, 1)
    cv2.rectangle(crop, (int((box[0] - x0) * kx), int((box[1] - y0) * ky)), (int((box[2] - x0) * kx),
                  int((box[3] - y0) * ky)), (0, 255, 255), 1)
    return crop


def cmd_sheet(cfg: dict[str, Any], n: int, seed: int, out: Path) -> dict[str, Any]:
    import cv2

    from xla.verify.mine import _d02x_manifest, load_manifest

    pick = draw_sample(load_manifest(cfg), n, seed)
    man = {m["name"]: m for m in _d02x_manifest(cfg)}
    ddir = Path(cfg["d02x_dir"]) / "train" / "images"
    out.mkdir(parents=True, exist_ok=True)
    tw, th = 3 * PANEL_PX + 12, PANEL_PX + 18
    per = COLS * ROWS
    pages = []
    for p0 in range(0, len(pick), per):
        part = pick[p0 : p0 + per]
        sheet = np.full((th * ((len(part) + COLS - 1) // COLS), tw * COLS, 3), 40, np.uint8)
        for i, r in enumerate(part):
            m = man[r["image"]]
            im = C.decode_frame(ddir / r["image"])
            src = cv2.imdecode(np.fromfile(m["source_image"], np.uint8), cv2.IMREAD_COLOR)
            if src is None:
                raise AuditError(f"không đọc được ảnh nguồn {m['source_image']}")
            sb = source_box(r, m)
            y, x = divmod(i, COLS)
            oy, ox = y * th + 18, x * tw
            k = MIN_SIDE / m["window"]["scale_x"]
            panels = [_panel(im, r["box"], CTX_SCALE), _cut(src, sb, CTX_SCALE, k), _cut(src, sb, ZOOM_SCALE, k / 2)]
            for j, pan in enumerate(panels):
                sheet[oy : oy + PANEL_PX, ox + j * (PANEL_PX + 4) : ox + j * (PANEL_PX + 4) + PANEL_PX] = pan
            side = min(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1])
            cv2.putText(sheet, f"{p0 + i:03d} {source_of(r)} {r['image'][:2]} s{r['score']:.2f} {side:.0f}px "
                        f"(nguon x{1 / m['window']['scale_x']:.1f})", (ox + 2, oy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 0), 1)
        path = out / f"sheet_{p0 // per:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_Q]):
            raise AuditError(f"không ghi được {path}")
        pages.append(path.relative_to(repo_root()).as_posix() if path.is_relative_to(repo_root()) else str(path))
    sample = [{"idx": i, "id": r["id"], "image": r["image"], "box": r["box"], "score": r["score"],
               "source": source_of(r), "degrade": r["image"][:2]} for i, r in enumerate(pick)]
    V.write_json(out / SAMPLE_FILE, {"seed": seed, "n": n, "type": NEG_TYPE,
                                      "pool": sum(r["src"] == "d02x" and r["type"] == NEG_TYPE
                                                  for r in load_manifest(cfg)), "rows": sample})
    return {"pages": pages, "n": n, "by_source": dict(Counter(s["source"] for s in sample))}


def read_review(path: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"thiếu {path} (cột id,person,note; person 1 = có người, 0 = không, ? = không chắc)")
    with path.open(encoding="utf-8", newline="") as f:
        got = {r["id"]: r for r in csv.DictReader(f)}
    missing = [i for i in ids if i not in got]
    extra = [i for i in got if i not in set(ids)]
    bad = {i: r["person"] for i, r in got.items() if r["person"].strip() not in LABELS}
    if missing or extra or bad:
        raise AuditError(f"review lệch mẫu: thiếu {missing[:5]}, thừa {extra[:5]}, nhãn lạ {dict(list(bad.items())[:5])}")
    return {i: {"person": LABELS[r["person"].strip()], "note": r.get("note", "")} for i, r in got.items()}


def summarize(sample: list[dict[str, Any]], review: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Tỉ lệ có người: chính = ca chắc chắn (``1``) / mọi ô; cận trên = (``1`` + ``?``) / mọi ô. Wilson 95 % cả hai."""

    def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        k = sum(review[r["id"]]["person"] == 1 for r in rows)
        u = sum(review[r["id"]]["person"] is None for r in rows)
        n = len(rows)
        return {"n": n, "person": k, "unsure": u, "rate": k / n if n else None, "wilson95": list(wilson(k, n)),
                "rate_incl_unsure": (k + u) / n if n else None, "wilson95_incl_unsure": list(wilson(k + u, n))}

    out = {"all": rate(sample)}
    out["by_source"] = {s: rate([r for r in sample if r["source"] == s]) for s in sorted({r["source"] for r in sample})}
    out["by_degrade"] = {d: rate([r for r in sample if r["degrade"] == d]) for d in sorted({r["degrade"] for r in sample})}
    out["material"] = out["all"]["wilson95"][0] > MATERIAL_LO
    out["material_rule"] = f"cận dưới Wilson 95 % của tỉ lệ chắc chắn có người > {MATERIAL_LO}"
    out["person_ids"] = [r["id"] for r in sample if review[r["id"]]["person"] == 1]
    out["unsure_ids"] = [r["id"] for r in sample if review[r["id"]]["person"] is None]
    return out


def cmd_score(out: Path) -> dict[str, Any]:
    doc = json.loads((out / SAMPLE_FILE).read_text(encoding="utf-8"))
    rows = doc["rows"]
    res = {"seed": doc["seed"], "pool": doc["pool"], **summarize(rows, read_review(out / REVIEW_FILE, [r["id"] for r in rows]))}
    V.write_json(out / AUDIT_FILE, res)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.verify.neg_audit", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["sheet", "score"])
    ap.add_argument("--config", default=str(V.CONFIG_REL))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--out", default="results/v2/verifier/neg_audit")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) if Path(a.out).is_absolute() else repo_root() / a.out
    try:
        res = cmd_sheet(V.load_cfg(a.config), a.n, a.seed, out) if a.cmd == "sheet" else cmd_score(out)
    except (AuditError, OSError) as e:
        V.emit({"error": str(e)}, a.json)
        return 1
    V.emit(res, a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
