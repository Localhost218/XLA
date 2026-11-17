"""Read-only scan: count JPEG frames in MJPEG AVIs and dump DQT/SOF of sampled frames.

Usage: python -I jpeg_dqt_scan.py <recordings_dir>
"""
import collections
import hashlib
import os
import struct
import sys

# zigzag -> natural order index
ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15,
    23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]


def frames_from_avi(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while True:
        j = data.find(b"00dc", i)
        if j < 0:
            return
        size = struct.unpack_from("<I", data, j + 4)[0]
        body = data[j + 8 : j + 8 + size]
        if body[:2] == b"\xff\xd8":
            yield body
        i = j + 8 + size


def parse_jpeg(buf):
    out = {"dqt": {}, "sof": None, "app": []}
    i = 2
    n = len(buf)
    while i < n - 4:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            break
        seglen = struct.unpack_from(">H", buf, i + 2)[0]
        seg = buf[i + 4 : i + 2 + seglen]
        if marker == 0xDB:
            k = 0
            while k < len(seg):
                pq_tq = seg[k]
                pq, tq = pq_tq >> 4, pq_tq & 15
                k += 1
                if pq == 0:
                    vals = list(seg[k : k + 64])
                    k += 64
                else:
                    vals = list(struct.unpack_from(">64H", seg, k))
                    k += 128
                nat = [0] * 64
                for z, v in enumerate(vals):
                    nat[ZIGZAG[z]] = v
                out["dqt"][tq] = nat
        elif marker in (0xC0, 0xC1, 0xC2):
            p, h, w, nc = struct.unpack_from(">BHHB", seg, 0)
            comps = []
            for c in range(nc):
                cid, hv, tq = struct.unpack_from(">BBB", seg, 6 + 3 * c)
                comps.append((cid, hv >> 4, hv & 15, tq))
            out["sof"] = (marker, w, h, comps)
        elif 0xE0 <= marker <= 0xEF:
            out["app"].append((marker, seg[:16]))
        elif marker == 0xDA:
            break
        i += 2 + seglen
    return out


def main(root):
    total = 0
    sizes = []
    table_hist = collections.Counter()
    example = {}
    sof_hist = collections.Counter()
    per_session = collections.Counter()
    files = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(".avi"):
                files.append(os.path.join(dp, f))
    files.sort()
    for idx, path in enumerate(files):
        sess = os.path.basename(os.path.dirname(path))
        for k, fr in enumerate(frames_from_avi(path)):
            total += 1
            per_session[sess] += 1
            sizes.append(len(fr))
            if k % 25 == 0:  # sample frames for DQT stats
                p = parse_jpeg(fr)
                key = tuple(tuple(p["dqt"].get(t, ())) for t in sorted(p["dqt"]))
                h = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
                table_hist[h] += 1
                if h not in example:
                    example[h] = (path, p)
                if p["sof"]:
                    m, w, hh, comps = p["sof"]
                    sof_hist[(hex(m), w, hh, tuple((c[1], c[2]) for c in comps))] += 1
    sizes.sort()
    print("avi_files", len(files), "frames_total", total)
    print("per_session", dict(per_session))
    if sizes:
        q = lambda r: sizes[int(r * (len(sizes) - 1))]
        print("frame_bytes p05/p50/p95", q(0.05), q(0.5), q(0.95))
    print("SOF (marker,w,h,sampling per comp):", dict(sof_hist))
    print("distinct DQT sets (sampled):", len(table_hist))
    for h, cnt in table_hist.most_common(8):
        path, p = example[h]
        luma = p["dqt"].get(0)
        chroma = p["dqt"].get(1)
        print(f"-- set {h} count={cnt} example={os.path.relpath(path, root)}")
        if luma:
            print("   luma row0:", luma[:8], " DC:", luma[0], " mean:", round(sum(luma) / 64, 2))
        if chroma:
            print("   chroma row0:", chroma[:8], " mean:", round(sum(chroma) / 64, 2))
        print("   app markers:", [(hex(m), s[:6]) for m, s in p["app"]])


if __name__ == "__main__":
    main(sys.argv[1])
