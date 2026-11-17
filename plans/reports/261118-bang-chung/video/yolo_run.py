"""YOLO11n (COCO person) trên mẫu khung: mọi khung lúc bay + mẫu mặt đất. Ghi detections.csv."""
import sys
from pathlib import Path
import pandas as pd, torch, cv2, numpy as np
from ultralytics import YOLO

OUT = Path(__file__).parent
W = Path(r"D:\IOT_Tools\downloads\yolo-weights")
f = pd.read_csv(OUT / "features.csv")
model = YOLO(str(W / sys.argv[1]) if len(sys.argv) > 1 else str(W / "yolo11n.pt"))
dev = 0 if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == 0 else "")
rows = []
for _, r in f.iterrows():
    im = cv2.imread(str(OUT / "samples" / r.file))
    for imgsz in (640, 1280):
        res = model.predict(im, imgsz=imgsz, conf=0.10, classes=[0], device=dev, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rows.append(dict(file=r.file, flying=r.flying, alt=r.alt, imgsz=imgsz, conf=float(b.conf), x1=x1, y1=y1,
                             w=x2 - x1, h=y2 - y1))
        if not len(res.boxes):
            rows.append(dict(file=r.file, flying=r.flying, alt=r.alt, imgsz=imgsz, conf=np.nan))
pd.DataFrame(rows).to_csv(OUT / f"detections_{Path(model.ckpt_path).stem}.csv", index=False)
print("xong", len(rows))
