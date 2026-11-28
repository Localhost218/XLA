import csv, os, json
R = "D:/XLA"
lab = R + "/data/real/labels/test"
test = json.load(open(R + "/data/real/labels/manifest.json", encoding="utf-8"))["test_frames"]
def load(p, thr):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if float(r["conf"]) >= thr:
            d.setdefault(r["frame"], []).append([float(r[k]) for k in ("x1", "y1", "x2", "y2", "conf")])
    return d
det = load(R + "/results/v2/system/preds/system_detector_test.csv", 0.07669)
ver = load(R + "/results/v2/system/preds/system_verifier_test.csv", 0.35)
ver0 = load(R + "/results/v2/system/preds/system_verifier_test.csv", 0.0)
def gt(f):
    out = []
    for ln in open(os.path.join(lab, f[:-4] + ".txt")):
        c, x, y, w, h = map(float, ln.split())
        out.append([(x - w / 2) * 640, (y - h / 2) * 480, (x + w / 2) * 640, (y + h / 2) * 480])
    return out
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0
blank_fp1 = blank_fp2 = 0; nb = 0
cands = []
for f in test:
    g = gt(f)
    d = det.get(f, []); v = ver.get(f, [])
    if not g:
        nb += 1; blank_fp1 += len(d); blank_fp2 += len(v)
    fp1 = sum(1 for b in d if max([iou(b, x) for x in g] + [0]) < 0.5)
    fp2 = sum(1 for b in v if max([iou(b, x) for x in g] + [0]) < 0.5)
    tp2 = sum(1 for b in v if max([iou(b, x) for x in g] + [0]) >= 0.5)
    cands.append((fp1 - fp2, tp2, len(g), f, fp1, fp2))
print("blank frames", nb, "fp1", blank_fp1, blank_fp1 / nb, "fp2", blank_fp2, blank_fp2 / nb)
print("ver rows total", sum(len(x) for x in ver0.values()))
for c in sorted(cands, reverse=True)[:15]:
    print(c)
