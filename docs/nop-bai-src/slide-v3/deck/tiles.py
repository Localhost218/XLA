import csv, sys, os
from PIL import Image, ImageDraw
X = 'D:/XLA'
OUT = sys.argv[1]
SRC = {  # name: (csv, threshold from t1_nhan_dien.md)
 'base': (f'{X}/results/baseline/pred/edge16.csv', 0.068),
 'd2': (f'{X}/results/e1/preds/d2_test.csv', 0.089),
 'bundle': (f'{X}/results/e2/preds/onnx_detector-v1.0.0_all.csv', 0.077),
}
COL = {'gt': (22, 163, 74), 'base': (100, 116, 139), 'd2': (13, 148, 136), 'bundle': (124, 58, 237)}
def preds(path, thr):
    d = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if float(r['conf']) >= thr:
            d.setdefault(r['frame'], []).append(tuple(float(r[k]) for k in ('x1','y1','x2','y2')))
    return d
P = {k: preds(*v) for k, v in SRC.items()}
def gt(frame):
    out = []
    for ln in open(f'{X}/results/labels/agent_reviewed_test/{frame[:-4]}.txt'):
        c, cx, cy, w, h = map(float, ln.split())
        out.append(((cx-w/2)*640, (cy-h/2)*480, (cx+w/2)*640, (cy+h/2)*480))
    return out
frames = sys.argv[2:]
for fr in frames:
    img = Image.open(f'{X}/data/real/frames/{fr}').convert('RGB')
    for k in ['gt', 'base', 'd2', 'bundle']:
        boxes = gt(fr) if k == 'gt' else P[k].get(fr, [])
        im = img.copy(); dr = ImageDraw.Draw(im)
        for b in boxes:
            dr.rectangle(b, outline=COL[k], width=3)
        im.save(os.path.join(OUT, f'{fr[:-4]}_{k}.png'))
        print(fr, k, len(boxes))
