"""Lưới 12 khung tiêu biểu có ghi chú + hộp YOLO."""
from pathlib import Path
import pandas as pd, numpy as np, cv2

OUT = Path(__file__).parent
f = pd.read_csv(OUT / "features.csv")
det = pd.read_csv(OUT / "det11s_labeled.csv")
fl = f[f.flying]
sel = []
def add(row, tag):
    if row.file not in [s[0] for s in sel]:
        sel.append((row.file, tag))
s = fl.sort_values("lap")
for i in range(1, 4): add(s.iloc[-i], "net nhat")
for i in range(3): add(s.iloc[i], "mo nhat")
pc = det[det.lab == "người"].groupby("file").conf.max().sort_values(ascending=False)
used = []
for fn in pc.index:
    n_ = int(fn.split("_")[1][:5])
    if all(abs(n_ - u) > 300 for u in used) and len(used) < 3:
        used.append(n_); add(f[f.file == fn].iloc[0], "co nguoi")
for _, r in fl.sort_values("mag_px").tail(1).iterrows(): add(r, "am hong")
for _, r in fl.sort_values("grad_aniso").tail(1).iterrows(): add(r, "nhoe chuyen dong")
for _, r in f[~f.flying & f.rec.str.startswith("20261101")].sort_values("mean_Y").head(1).iterrows(): add(r, "toi (01/11)")
tiles = []
for fn, tag in sel[:12]:
    r = f[f.file == fn].iloc[0]
    im = cv2.imread(str(OUT / "samples" / fn))
    for _, b in det[det.file == fn].iterrows():
        col = (0, 255, 0) if b.lab == "người" else (0, 0, 255)
        cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x1 + b.w), int(b.y1 + b.h)), col, 2)
        cv2.putText(im, f"{b.conf:.2f}", (int(b.x1), int(b.y1) - 4), 0, 0.6, (0, 0, 255), 2)
    im = cv2.resize(im, (480, 360))
    im = np.vstack([np.zeros((44, 480, 3), np.uint8), im])
    txt1 = f"{tag} | {fn[:15]} #{r.n}"
    txt2 = f"cao {r.alt:.1f}m lap {r.lap:.0f} hong {r.mag_px:.1f}% aniso {r.grad_aniso:.2f} {r['size']/1024:.0f}KB"
    cv2.putText(im, txt1.encode("ascii", "ignore").decode(), (4, 18), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(im, txt2, (4, 38), 0, 0.5, (255, 255, 255), 1)
    tiles.append(im)
while len(tiles) < 12: tiles.append(np.zeros((404, 480, 3), np.uint8))
g = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
cv2.imwrite(str(OUT / "luoi-12-khung.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 88])
for fn, tag in sel[:12]: print(tag, fn)
