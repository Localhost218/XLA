"""Chạy pipeline v2 của D:\\XLA trên khung thật của drone IOT-CV, đo thời gian, lưu ảnh trước/sau."""
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, r"D:\XLA")
from src import uav_pipeline as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR, OUT_DIR = os.path.join(HERE, "in"), os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def metrics(img):
    return dict(
        pci=P.compute_pink_cast_index(img), ent=P.compute_entropy(img), lap=P.compute_laplacian_var(img),
        sigma=P.estimate_noise_sigma(img), lmean=P.compute_l_mean(img), crush=P.compute_crush_ratio(img),
        clip=P.compute_clip_ratio(img),
    )


files = sorted(glob.glob(os.path.join(IN_DIR, "*.png")))
# warm-up (nạp ONNX, cấp phát) - không tính giờ
P.run_full_pipeline(cv2.imread(files[0]))

rows = []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    img = cv2.imread(f)
    t0 = time.perf_counter()
    r = P.run_full_pipeline(img, return_intermediates=True)
    wall = (time.perf_counter() - t0) * 1000.0
    out = r["output"]
    # Đường "video" của repo: hạ về cao 360, Stage1+2, upscale_to_hd 720p
    small = cv2.resize(img, P.fit_size(img.shape[1], img.shape[0], 10 ** 6, 360), interpolation=cv2.INTER_AREA)
    t1 = time.perf_counter()
    s1, a = P.run_stage1_preprocess(small)
    s2, _, _, _ = P.run_stage2_adaptive(s1, alpha_mask=a)
    P.upscale_to_hd(s2, target_height=720, target_width=1280)
    video_ms = (time.perf_counter() - t1) * 1000.0

    mi, mo = metrics(img), metrics(out)
    si = r["scene_info"]
    rows.append(dict(name=name, wall_ms=wall, s1=r["stage1_ms"], s2=r["stage2_ms"], s3=r["stage3_ms"], video_ms=video_ms,
                     alpha_mean=float(r["alpha_mask"].mean()), lift=si.get("lift_policy", 0), s_haze=si.get("s_haze", 0),
                     gain_mean=si.get("gain_mean", 1), **{f"in_{k}": v for k, v in mi.items()},
                     **{f"out_{k}": v for k, v in mo.items()}))
    sbs = np.hstack([label(img, "INPUT"), label(r["stage1"], "STAGE1"), label(out, "V2 OUTPUT")])
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_sbs.jpg"), sbs, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # crop 2x giữa ảnh để soi halo/nhiễu
    h, w = img.shape[:2]
    y0, x0 = h // 4, w // 4
    ci = cv2.resize(img[y0:y0 + h // 2, x0:x0 + w // 2], (w, h), interpolation=cv2.INTER_NEAREST)
    co = cv2.resize(out[y0:y0 + h // 2, x0:x0 + w // 2], (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_crop2x.jpg"), np.hstack([label(ci, "INPUT 2x"), label(co, "V2 2x")]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(name, f"{wall:.0f}ms", flush=True)

import csv  # noqa: E402

with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    for r_ in rows:
        wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r_.items()})

hdr = "name wall s1 s2 s3 video720 | PCI in->out | ent | lap | sigma | L | crush | alpha lift haze"
print(hdr)
for r_ in rows:
    print(f"{r_['name']:8s} {r_['wall_ms']:6.0f} {r_['s1']:5.0f} {r_['s2']:5.0f} {r_['s3']:5.0f} {r_['video_ms']:6.0f} | "
          f"{r_['in_pci']:.2f}->{r_['out_pci']:.2f} | {r_['in_ent']:.2f}->{r_['out_ent']:.2f} | "
          f"{r_['in_lap']:.0f}->{r_['out_lap']:.0f} | {r_['in_sigma']:.2f}->{r_['out_sigma']:.2f} | "
          f"{r_['in_lmean']:.1f}->{r_['out_lmean']:.1f} | {r_['in_crush']:.2f}->{r_['out_crush']:.2f} | "
          f"{r_['alpha_mean']:.3f} {r_['lift']:.2f} {r_['s_haze']:.2f}")
w = np.array([r_["wall_ms"] for r_ in rows])
v = np.array([r_["video_ms"] for r_ in rows])
print(f"native 640x480: mean {w.mean():.0f} ms, min {w.min():.0f}, max {w.max():.0f}, fps {1000 / w.mean():.2f}")
print(f"video path 360->720p: mean {v.mean():.0f} ms, fps {1000 / v.mean():.2f}")
