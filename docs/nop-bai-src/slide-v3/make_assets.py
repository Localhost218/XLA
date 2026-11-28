"""Sinh ảnh và file số liệu cho báo cáo LaTeX (chỉ khung bay của nhóm + ảnh thử tổng hợp, không ảnh VisDrone).

Chạy: D:\\DevCache\\venvs\\xla\\Scripts\\python.exe make_assets.py   (cwd bất kỳ)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("D:/XLA")
sys.path.insert(0, str(REPO))

from xla.degrade import d2_camera as d2  # noqa: E402
from xla.enhance import classic  # noqa: E402

OUT = REPO / "docs/nop-bai-src/latex/fig"
OUT.mkdir(parents=True, exist_ok=True)
FR = REPO / "data/real/frames"
LAB = REPO / "data/real/labels/test"
CROP = 16


def imread(stem: str) -> np.ndarray:
    img = cv2.imread(str(FR / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(stem)
    return img


def save(name: str, img: np.ndarray, q: int = 92) -> None:
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    print("wrote", name, img.shape)


def pink_frac(img: np.ndarray) -> float:
    b, g, r = [img[:, CROP:, i].astype(np.int16) for i in range(3)]
    return float(((r - g > 25) & (b - g > 25)).mean())


def read_gt(stem: str) -> list[tuple[float, float, float, float]]:
    out = []
    p = LAB / f"{stem}.txt"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        out.append(((cx - w / 2) * 640, (cy - h / 2) * 480, (cx + w / 2) * 640, (cy + h / 2) * 480))
    return out


def read_pred(csv_path: Path, stem: str, thr: float) -> list[tuple[float, float, float, float, float]]:
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["frame"] == f"{stem}.jpg" and float(row["conf"]) >= thr:
                out.append((float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), float(row["conf"])))
    return out


# ------------------------------------------------------------------ chọn khung
test_stems = sorted(p.stem for p in LAB.glob("*.txt"))
stats = []
for s in test_stems:
    img = imread(s)
    n = len(read_gt(s))
    stats.append((s, n, pink_frac(img)))
by_pink = sorted(stats, key=lambda t: -t[2])
print("top pink test frames:", by_pink[:8])
people = [t for t in stats if t[1] >= 3]
people_pink = sorted(people, key=lambda t: -t[2])
print("people+pink:", people_pink[:6])

# ------------------------------------------------------------------ làm rõ: gốc / cổ điển / mạng
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(REPO / "models/release/enhancer-v1.0.0/enhancer.onnx"),
                            providers=["CPUExecutionProvider"])
params = classic.load_params()

ENH_STEMS = ["20261102-100941_005_00934", by_pink[0][0]]
if len(sys.argv) > 1:
    ENH_STEMS = sys.argv[1:]
for i, s in enumerate(ENH_STEMS):
    raw = imread(s)
    cl = classic.enhance(raw, params)
    net = sess.run(["frame_out"], {"frame": raw})[0]
    for tag, im in (("raw", raw), ("classic", cl), ("net", net)):
        save(f"enh{i}_{tag}.jpg", im[:, CROP:])
    print("enh", i, s, "pink", pink_frac(raw), pink_frac(cl), pink_frac(net))

# ------------------------------------------------------------------ nhãn thật vs hộp model
BUNDLE_CSV = REPO / "results/e2/preds/onnx_detector-v1.0.0_all.csv"
E1D2_CSV = REPO / "results/e1/preds/d2_test.csv"
DET_STEMS = ["20261102-100941_005_00934", "20261102-100941_003_02520"]
for i, s in enumerate(DET_STEMS):
    img = imread(s)
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gt = read_gt(s)
    bun = read_pred(BUNDLE_CSV, s, 0.07669)
    e1 = read_pred(E1D2_CSV, s, 0.0885)
    for (x1, y1, x2, y2) in gt:
        cv2.rectangle(big, (int(x1 * 2) - 3, int(y1 * 2) - 3), (int(x2 * 2) + 3, int(y2 * 2) + 3), (60, 200, 40), 3)
    for (x1, y1, x2, y2, c) in e1:
        cv2.rectangle(big, (int(x1 * 2), int(y1 * 2)), (int(x2 * 2), int(y2 * 2)), (230, 120, 20), 2)
    for (x1, y1, x2, y2, c) in bun:
        cv2.rectangle(big, (int(x1 * 2) + 3, int(y1 * 2) + 3), (int(x2 * 2) - 3, int(y2 * 2) - 3), (20, 110, 240), 2)
    save(f"det{i}.jpg", big[:, CROP * 2:])
    print("det", i, s, "gt", len(gt), "e1d2", len(e1), "bundle", len(bun))
    json.dump({"stem": s, "gt": len(gt), "e1d2": len(e1), "bundle": len(bun)},
              open(OUT / f"det{i}.json", "w", encoding="utf-8"))

# ------------------------------------------------------------------ lỗi camera: các mảnh phóng to
art = imread(by_pink[0][0])
H, W = art.shape[:2]
b, g, r = [art[:, :, k].astype(np.int16) for k in range(3)]
mask = ((r - g > 25) & (b - g > 25)).astype(np.uint8)
mask[:, :CROP] = 0
ys, xs = np.nonzero(mask)
cy, cx = int(np.median(ys)), int(np.median(xs))
x0, y0 = int(np.clip(cx - 40, CROP, W - 80)), int(np.clip(cy - 30, 0, H - 60))
crop_pink = art[y0:y0 + 60, x0:x0 + 80]
save("art_pink.png".replace(".png", ".jpg"), cv2.resize(crop_pink, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
# dải mép trái: 64 cột đầu, nửa dưới
strip = art[200:440, 0:64]
save("art_strip.jpg", cv2.resize(strip, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
# khối JPEG: vùng 40x30 ở giữa, phóng 8 lần
blk = art[220:250, 300:340]
save("art_block.jpg", cv2.resize(blk, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
save("art_full.jpg", art)
print("artifact frame", by_pink[0][0], "pink crop at", x0, y0)

# ------------------------------------------------------------------ 3D: dark channel của một khung thật
dc_src = imread("20261102-100941_005_00934")
dc = d2.dark_channel(dc_src[:, CROP:].astype(np.float32) / 255.0, 15) * 255.0
small = cv2.resize(dc, (36, 27), interpolation=cv2.INTER_AREA)
with open(OUT / "dark_channel.dat", "w", encoding="utf-8") as f:
    f.write("x y z\n")
    for j in range(small.shape[0]):
        for i in range(small.shape[1]):
            f.write(f"{i * (624 / 35):.1f} {j * (480 / 26):.1f} {small[j, i]:.2f}\n")
        f.write("\n")
print("dark channel median", float(np.median(dc)), "min", float(dc.min()), "max", float(dc.max()))

# ------------------------------------------------------------------ 3D: đường đi của màu xám qua khâu tone của D2
cfg = d2.load_config(REPO / "configs/degrade_d2.yaml")
prof = d2.load_profile(REPO / "configs/camera_profile.yaml")
tc = cfg["tone"]
gr, gb = float(tc["isp_wb_gain_r"]), float(tc["isp_wb_gain_b"])
cr = float(prof["color"]["r_over_g"]["p50"]) * tc["cast_r"]
cbb = float(prof["color"]["b_over_g"]["p50"]) * tc["cast_b"]
k = d2.highlight_gain(prof) * tc["highlight_gain_scale"]
bright = float(np.mean(tc["brightness"]))
e = bright / k
with open(OUT / "tone_path.dat", "w", encoding="utf-8") as f:
    f.write("r g b\n")
    for v in np.linspace(0.0, 1.0, 81):
        lin = d2.srgb_to_linear(np.array([v, v, v], np.float32))  # B,G,R như nhau
        raw = lin * np.array([1 / gb, 1.0, 1 / gr], np.float32) * e
        raw = np.minimum(raw, 1.0)
        out = np.clip(raw * np.array([gb * cbb, 1.0, gr * cr], np.float32) * k, 0, 1)
        s = d2.linear_to_srgb(out)
        f.write(f"{s[2]:.4f} {s[1]:.4f} {s[0]:.4f}\n")
print("tone k", k, "e", e, "gr", gr, "gb", gb)

# điểm sáng thật (Y > 170) trên khung nhiều hồng: tách điểm hồng và điểm không hồng
rng = np.random.default_rng(20261118)
yy = 0.299 * r + 0.587 * g + 0.114 * b
sel = (yy > 170)
sel[:, :CROP] = False
idx = np.flatnonzero(sel.ravel())
pink_idx = idx[mask.ravel()[idx] > 0]
other_idx = idx[mask.ravel()[idx] == 0]
pick_p = rng.choice(pink_idx, size=min(220, len(pink_idx)), replace=False)
pick_o = rng.choice(other_idx, size=min(220, len(other_idx)), replace=False)
for name, pick in (("pts_pink.dat", pick_p), ("pts_other.dat", pick_o)):
    with open(OUT / name, "w", encoding="utf-8") as f:
        f.write("r g b\n")
        for t in pick:
            f.write(f"{r.ravel()[t] / 255:.4f} {g.ravel()[t] / 255:.4f} {b.ravel()[t] / 255:.4f}\n")
print("highlight pts pink", len(pink_idx), "other", len(other_idx))

# ------------------------------------------------------------------ D2 từng khâu trên ảnh thử tổng hợp
rng = np.random.default_rng(7)
Hh, Ww = d2.WIN_H, d2.WIN_W
yy2, xx2 = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
noise = cv2.GaussianBlur(rng.standard_normal((Hh, Ww)).astype(np.float32), (0, 0), 3)
noise2 = cv2.GaussianBlur(rng.standard_normal((Hh, Ww)).astype(np.float32), (0, 0), 14)
grass = np.stack([60 + 18 * noise + 10 * noise2, 140 + 25 * noise + 20 * noise2, 70 + 15 * noise], 2)
scene = grass.copy()
road = (np.abs((xx2 - 0.55 * yy2) - 200) < 55)
scene[road] = (np.stack([150 + 6 * noise, 150 + 6 * noise, 148 + 6 * noise], 2))[road]
cv2.rectangle(scene, (420, 40), (560, 150), (245, 245, 245), -1)      # mái tôn trắng (vùng sáng)
cv2.rectangle(scene, (430, 50), (550, 140), (252, 252, 252), -1)
cv2.rectangle(scene, (60, 330), (170, 420), (40, 60, 180), -1)        # mái đỏ
cv2.circle(scene, (300, 380), 38, (235, 240, 240), -1)                  # bồn nước trắng
for (px, py, col) in ((250, 120, (40, 40, 200)), (330, 200, (200, 120, 30)), (380, 140, (30, 160, 230)),
                      (150, 230, (220, 220, 220))):
    cv2.ellipse(scene, (px, py), (9, 13), 0, 0, 360, col, -1)          # người nhìn từ trên
    cv2.circle(scene, (px, py - 2), 5, (40, 60, 90), -1)
    cv2.ellipse(scene, (px + 10, py + 8), (8, 4), 20, 0, 360, (30, 50, 40), -1)  # bóng
for kx in range(0, Ww, 48):                                              # vạch kẻ đường
    pass
scene = np.clip(scene, 0, 255).astype(np.uint8)
scene_f = scene.astype(np.float32) / 255.0
p: dict = {}
lin = d2.srgb_to_linear(scene_f)
st = [("d2_0_src.jpg", scene)]
lin_h = d2.stage_haze(lin, rng, cfg["haze"], prof, p.setdefault("haze", {}))
st.append(("d2_1_haze.jpg", lin_h))
lin_l = d2.stage_lens(lin_h, rng, cfg["lens"], prof, p.setdefault("lens", {}))
st.append(("d2_2_lens.jpg", lin_l))
lin_t = d2.stage_tone(lin_l, rng, cfg["tone"], prof, p.setdefault("tone", {}))
st.append(("d2_3_tone.jpg", lin_t))
srgb = d2.linear_to_srgb(lin_t)
srgb_i = d2.stage_isp(srgb, rng, cfg["isp"], prof, p.setdefault("isp", {}))
frame = d2.stage_strip(srgb_i, rng, cfg["strip"], prof, p.setdefault("strip", {}))
fr8 = np.round(frame * 255).clip(0, 255).astype(np.uint8)
jpg, _ = d2.encode_final(fr8, 12, prof)
final = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
for name, im in st:
    if im.dtype != np.uint8:
        im = np.round(d2.linear_to_srgb(im) * 255).clip(0, 255).astype(np.uint8)
    pad = np.zeros((Hh, Ww + CROP, 3), np.uint8)
    pad[:, CROP:] = im
    pad[:, :CROP] = im[:, CROP - 1::-1]
    save(name, pad)
save("d2_4_isp.jpg", np.concatenate([srgb_i[:, CROP - 1::-1], srgb_i], 1).__mul__(255).clip(0, 255).astype(np.uint8))
save("d2_5_final.jpg", final, q=97)
json.dump(p, open(OUT / "d2_params_used.json", "w", encoding="utf-8"), indent=1, default=float)
print("d2 params", json.dumps(p, default=float)[:600])
