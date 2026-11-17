import os, struct, sys
def movi_count(path):
    d = open(path, "rb").read()
    assert d[:4] == b"RIFF"
    end = 8 + struct.unpack_from("<I", d, 4)[0]
    i = 12; n = 0; hdr_frames = None
    while i + 8 <= min(end, len(d)):
        cc = d[i:i+4]; sz = struct.unpack_from("<I", d, i+4)[0]
        if cc == b"LIST" and d[i+8:i+12] == b"movi":
            j = i + 12; mend = i + 8 + sz
            while j + 8 <= min(mend, len(d)):
                c2 = d[j:j+4]; s2 = struct.unpack_from("<I", d, j+4)[0]
                if c2 == b"00dc": n += 1
                j += 8 + s2 + (s2 & 1)
        if cc == b"LIST" and d[i+8:i+12] == b"hdrl":
            hdr_frames = struct.unpack_from("<I", d, i + 12 + 8 + 16)[0]  # avih dwTotalFrames
        i += 8 + sz + (sz & 1)
    return n, hdr_frames
root = sys.argv[1]
for s in sorted(os.listdir(root)):
    p = os.path.join(root, s); tot = 0; th = 0
    for f in sorted(os.listdir(p)):
        if f.endswith(".avi"):
            n, h = movi_count(os.path.join(p, f)); tot += n; th += (h or 0)
    print(s, "movi_00dc", tot, "avih_total", th)
