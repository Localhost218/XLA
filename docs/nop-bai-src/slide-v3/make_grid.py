"""Lưới cùng khung: nhãn thật / mốc / E1 D2 / bundle; và mảnh phóng to của ba cách làm rõ."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("D:/XLA")
sys.path.insert(0, str(REPO))
from xla.enhance import classic  # noqa: E402
import onnxruntime as ort  # noqa: E402

OUT = REPO / "docs/nop-bai-src/latex/fig"
FR = REPO / "data/real/frames"
LAB = REPO / "data/real/labels/test"

SRC = {
    "moc": (REPO / "results/baseline/pred/edge16.csv", 0.06760),
    "e1d2": (REPO / "results/e1/preds/d2_test.csv", 0.0885),
    "bundle": (REPO / "results/e2/preds/onnx_detector-v1.0.0_all.csv", 0.07669),
}
COL = {"gt": (60, 200, 40), "moc": (120, 120, 120), "e1d2": (230, 120, 20), "bundle": (20, 110, 240)}


def gt(stem):
    out = []
    for line in (LAB / f"{stem}.txt").read_text().splitlines():
        if line.strip():
            _, cx, cy, w, h = map(float, line.split())
            out.append(((cx - w / 2) * 640, (cy - h / 2) * 480, (cx + w / 2) * 640, (cy + h / 2) * 480))
    return out


def pred(path, thr, stem):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["frame"] == f"{stem}.jpg" and float(r["conf"]) >= thr:
                out.append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
    return out


# (khung, vùng cắt x0,y0,x1,y1 theo toạ độ khung gốc)
FRAMES = [("20261102-100941_005_00934", (16, 0, 216, 150)), ("20261102-100941_003_02520", (380, 110, 640, 380))]
S = 3
for fi, (stem, (x0, y0, x1, y1)) in enumerate(FRAMES):
    img = cv2.imread(str(FR / f"{stem}.jpg"))
    counts = {}
    for key in ("gt", "moc", "e1d2", "bundle"):
        boxes = gt(stem) if key == "gt" else pred(SRC[key][0], SRC[key][1], stem)
        counts[key] = len(boxes)
        big = cv2.resize(img[y0:y1, x0:x1], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
        for (a, b, c, d) in boxes:
            p1 = (int((a - x0) * S), int((b - y0) * S))
            p2 = (int((c - x0) * S), int((d - y0) * S))
            cv2.rectangle(big, p1, p2, COL[key], 3)
        cv2.imwrite(str(OUT / f"grid{fi}_{key}.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(stem, counts)

sess = ort.InferenceSession(str(REPO / "models/release/enhancer-v1.0.0/enhancer.onnx"), providers=["CPUExecutionProvider"])
params = classic.load_params()
ZOOM = [("20261102-100941_005_00934", (16, 0, 216, 150)), ("20261102-100941_004_01252", (330, 280, 530, 430))]
for zi, (stem, (x0, y0, x1, y1)) in enumerate(ZOOM):
    raw = cv2.imread(str(FR / f"{stem}.jpg"))
    vs = {"raw": raw, "classic": classic.enhance(raw, params), "net": sess.run(["frame_out"], {"frame": raw})[0]}
    for k, im in vs.items():
        z = cv2.resize(im[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"zoom{zi}_{k}.jpg"), z, [cv2.IMWRITE_JPEG_QUALITY, 94])
print("ok")
