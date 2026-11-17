"""Đặc trưng suy giảm trên mẫu khung + kiểm toàn bộ khung (giải mã, EOI, quality từ bảng DQT)."""
import csv, json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd

REPO = Path(r"d:\Coding\IOT-CV")
sys.path.insert(0, str(REPO))
from backend.flights.video import Recording  # noqa: E402

OUT = Path(__file__).parent
(OUT / "samples").mkdir(exist_ok=True)
TZ = 7 * 3600
# Cửa sổ bay từ `python -m backend.flights --events` (giờ VN, chuyến có "bay: có" và có video)
FLIGHTS = [("101046", 125), ("101435", 153), ("102319", 138), ("102636", 159), ("103039", 162), ("104039", 214)]

STD_LUM = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99])
ZIGZAG = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,
    57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])


def ijg_quality(jpg: bytes):
    """Ước lượng quality IJG (1–100) từ bảng lượng tử độ sáng. ESP32 'quality' 12 ≈ ngược thang (thấp = đẹp)."""
    i = jpg.find(b"\xff\xdb")
    if i < 0:
        return None
    pq_tq = jpg[i + 4]
    if pq_tq >> 4:
        return None
    zz = np.frombuffer(jpg[i + 5:i + 69], np.uint8).astype(float)
    q = np.empty(64); q[ZIGZAG] = zz
    s = np.mean(q * 100 / STD_LUM)
    return round((200 - s) / 2 if s <= 100 else 5000 / s, 1)


def in_flight(t_local_sec_of_day, date_ok):
    if not date_ok:
        return False
    for hhmmss, dur in FLIGHTS:
        h, m, s = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:])
        st = h * 3600 + m * 60 + s
        if st <= t_local_sec_of_day <= st + dur:
            return True
    return False


def feats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    # Nhiễu: phương pháp Immerkaer (1996)
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    sigma = np.sqrt(np.pi / 2) / (6 * (W - 2) * (H - 2)) * np.abs(cv2.filter2D(g, -1, M)[1:-1, 1:-1]).sum()
    b, gg, r = [img[..., k].astype(np.float32).mean() for k in range(3)]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blocking: chênh lệch tuyệt đối ở biên khối 8 px so với bên trong
    dx = np.abs(np.diff(g, axis=1)); dy = np.abs(np.diff(g, axis=0))
    bx = dx[:, 7::8].mean() / max(np.delete(dx, np.s_[7::8], axis=1).mean(), 1e-3)
    by = dy[7::8, :].mean() / max(np.delete(dy, np.s_[7::8], axis=0).mean(), 1e-3)
    # Motion blur: bất đẳng hướng năng lượng gradient (1 = đẳng hướng)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0); sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ex, ey = float((sx ** 2).mean()), float((sy ** 2).mean())
    # Tenengrad
    ten = float(np.sqrt(sx ** 2 + sy ** 2).mean())
    # Xé hình: số dòng cuối liên tiếp gần như phẳng (std < 2) — khung cụt ESP32 thường tô xám/xanh đáy
    row_std = g.std(axis=1)
    tail = 0
    for v in row_std[::-1]:
        if v < 2: tail += 1
        else: break
    # Đột biến giữa 2 nửa khung theo dòng (khung ghép từ 2 lần chụp)
    rm = g.mean(axis=1); jump = float(np.abs(np.diff(rm)).max())
    R, G, B = img[..., 2].astype(int), img[..., 1].astype(int), img[..., 0].astype(int)
    mag_px = float(((R - G > 25) & (B - G > 25)).mean() * 100)
    left = float(g[:, :12].mean()); left_ref = float(g[:, 20:40].mean())
    dark = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    return dict(mag_px=mag_px, left_strip=left, left_ref=left_ref, dark_ch=float(dark.mean()), contrast=float(np.percentile(g, 99) - np.percentile(g, 1)), lap=float(lap), noise=float(sigma), tenengrad=ten, mean_Y=float(g.mean()), p1=float(np.percentile(g, 1)),
                p99=float(np.percentile(g, 99)), clip_hi=float((g >= 250).mean() * 100), clip_lo=float((g <= 5).mean() * 100),
                R_G=r / max(gg, 1), B_G=b / max(gg, 1), sat=float(hsv[..., 1].mean()),
                magenta=float(((r + b) / 2 - gg)), block=float((bx + by) / 2), grad_aniso=max(ex, ey) / max(min(ex, ey), 1e-3),
                flat_tail_rows=tail, row_jump=jump)


# ---- 1) Quét toàn bộ khung: giải mã, EOI, quality
allf = []
for d in sorted((REPO / "logs/recordings").iterdir()):
    rec = Recording(d)
    with (d / "frames.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for n, r in enumerate(rows):
        jpg = rec.frame(n)
        t = float(r["unix_time"]); tl = t + TZ
        sod = tl % 86400
        date_ok = d.name == "20261102-100941"
        alt = float(r["rel_alt_m"]) if r.get("rel_alt_m") else np.nan
        rec_ = dict(rec=d.name, n=n, t=t, alt=alt, flying=bool(in_flight(sod, date_ok) and alt > 3.0),
                    size=len(jpg) if jpg else 0, soi=jpg is not None, eoi=bool(jpg and jpg.rstrip(b"\x00")[-2:] == b"\xff\xd9"),
                    q=ijg_quality(jpg) if jpg else None, decoded=False)
        allf.append(rec_)
df = pd.DataFrame(allf)

# Giải mã thử toàn bộ (rẻ: 640x480)
for d, sub in df.groupby("rec"):
    rec = Recording(REPO / "logs/recordings" / d)
    for idx in sub.index:
        jpg = rec.frame(int(df.at[idx, "n"]))
        if jpg:
            im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            df.at[idx, "decoded"] = im is not None
df.to_csv(OUT / "all_frames_scan.csv", index=False)

# ---- 2) Mẫu: 300 khung lúc bay + 100 khung còn lại rải đều theo session
fly = df[df.flying]
rest = df[~df.flying & (df["size"] > 0)]
pick = list(fly.iloc[np.linspace(0, len(fly) - 1, min(300, len(fly))).astype(int)].index)
recs = sorted(rest.rec.unique())
per = max(1, 100 // len(recs))
for r_ in recs:
    s = rest[rest.rec == r_]
    pick += list(s.iloc[np.linspace(0, len(s) - 1, min(per, len(s))).astype(int)].index)
pick = sorted(set(pick))
out = []
cache = {}
for idx in pick:
    row = df.loc[idx]
    rec = cache.setdefault(row.rec, Recording(REPO / "logs/recordings" / row.rec))
    jpg = rec.frame(int(row.n))
    im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg else None
    if im is None:
        continue
    fn = f"{row.rec}_{int(row.n):05d}.jpg"
    (OUT / "samples" / fn).write_bytes(jpg)
    out.append(dict(file=fn, rec=row.rec, n=int(row.n), t=row.t, alt=row.alt, flying=bool(row.flying), size=int(row["size"]), q=row.q, **feats(im)))
pd.DataFrame(out).to_csv(OUT / "features.csv", index=False)
print("tổng khung", len(df), "bay", int(df.flying.sum()), "mẫu", len(out))
print("giải mã hỏng", int((~df.decoded).sum()), "thiếu EOI", int((~df.eoi).sum()))
print(df.groupby("rec").agg(q_min=("q", "min"), q_med=("q", "median"), q_max=("q", "max"), bad=("decoded", lambda s: int((~s).sum())), noeoi=("eoi", lambda s: int((~s).sum()))).to_string())
