"""Slide images: tier-1 vs tier-2 boxes on two test frames, and one A/B frame as raw / classic / offline."""
import csv, json, os, shutil
from PIL import Image, ImageDraw
R = "D:/XLA"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
os.makedirs(OUT, exist_ok=True)

def load(p, thr):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if float(r["conf"]) >= thr:
            d.setdefault(r["frame"], []).append([float(r[k]) for k in ("x1", "y1", "x2", "y2", "conf")])
    return d
det = load(R + "/results/v2/system/preds/system_detector_test.csv", 0.07669)
ver = load(R + "/results/v2/system/preds/system_verifier_test.csv", 0.35)

def gt(f):
    out = []
    for ln in open(f"{R}/data/real/labels/test/{f[:-4]}.txt"):
        c, x, y, w, h = map(float, ln.split())
        out.append([(x - w / 2) * 640, (y - h / 2) * 480, (x + w / 2) * 640, (y + h / 2) * 480])
    return out

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0

S = 2
def draw(frame, boxes, g, path):
    im = Image.open(f"{R}/data/real/frames/{frame}").convert("RGB").resize((640 * S, 480 * S), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for b in g:
        dr.rectangle([b[0] * S - 3, b[1] * S - 3, b[2] * S + 3, b[3] * S + 3], outline=(255, 255, 255), width=2)
    tp = fp = 0
    for b in boxes:
        ok = max([iou(b, x) for x in g] + [0]) >= 0.5
        tp += ok; fp += not ok
        col = (30, 200, 110) if ok else (235, 40, 110)
        dr.rectangle([b[0] * S, b[1] * S, b[2] * S, b[3] * S], outline=col, width=5)
    im.save(path, quality=92)
    return tp, fp

info = {}
for f in ["20261102-100941_005_00809.jpg", "20261102-100941_002_00005.jpg"]:
    g = gt(f)
    a = draw(f, det.get(f, []), g, os.path.join(OUT, f"box_{f[16:-4]}_t1.jpg"))
    b = draw(f, ver.get(f, []), g, os.path.join(OUT, f"box_{f[16:-4]}_t2.jpg"))
    info[f] = {"gt": len(g), "t1_tp_fp": a, "t2_tp_fp": b}

key = json.load(open(R + "/results/ab-v2/ab_key.json", encoding="utf-8"))
byf = {}
for p in key["pairs"]:
    if p["repeat_of"]:
        continue
    for side in ("left", "right"):
        byf.setdefault(p["frame"], {})[p[side]] = R + "/results/ab-v2/" + p[side + "_img"]
trip = [f for f, v in byf.items() if len(v) == 3]
info["ab_triples"] = trip
pick = trip[0] if trip else None
if pick:
    for k, v in byf[pick].items():
        shutil.copy(v, os.path.join(OUT, f"ab_{k}.jpg"))
    info["ab_pick"] = pick
    for k in ("raw", "classic", "offline"):
        info["size_" + k] = Image.open(os.path.join(OUT, f"ab_{k}.jpg")).size
print(json.dumps(info, indent=1, ensure_ascii=False))

# no frame has all three variants: take an offline_vs_classic pair and cut the raw frame the same way
pairs = [p for p in key["pairs"] if p["comparison"] == "offline_vs_classic" and not p["repeat_of"] and p["has_person"]]
p = pairs[0]
for side in ("left", "right"):
    shutil.copy(R + "/results/ab-v2/" + p[side + "_img"], os.path.join(OUT, f"ab_{p[side]}.jpg"))
raw = Image.open(f"{R}/data/real/frames/{p['frame']}").convert("RGB").crop((16, 0, 640, 480))
raw.save(os.path.join(OUT, "ab_raw.jpg"), quality=95)
print(p["frame"], p["alt_m"], {k: Image.open(os.path.join(OUT, f"ab_{k}.jpg")).size for k in ("raw", "classic", "offline")})
print([ (q["frame"], q["alt_m"]) for q in pairs[:8]])
for t in ("t1", "t2"):
    im = Image.open(os.path.join(OUT, f"box_005_00809_{t}.jpg"))
    im.crop((175 * S, 300 * S, 495 * S, 480 * S)).save(os.path.join(OUT, f"boxz_005_00809_{t}.jpg"), quality=92)
box = (220, 140, 580, 263)
for k in ("raw", "classic", "offline"):
    im = Image.open(os.path.join(OUT, f"ab_{k}.jpg"))
    f = im.size[0] / 624
    im.crop(tuple(int(v * f) for v in box)).resize((900, 309), Image.LANCZOS).save(os.path.join(OUT, f"ab_{k}_z.jpg"), quality=92)
