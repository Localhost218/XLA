import re, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
pat = re.compile(r"\bv1\b|\bv2\b|AI v2|phiên bản|ship|—|–|OV2640|YOLO11n|không đạt|Kill|v\d\.\d\.\d", re.I)
hits = 0
def key(n):
    m = re.search(r"(\d+)\.xml$", n); return (n.split("/")[1], int(m.group(1)) if m else 0)
for n in sorted(z.namelist(), key=lambda n: (n.split("/")[1] if "/" in n else n, int(re.search(r"(\d+)\.xml$", n).group(1)) if re.search(r"(\d+)\.xml$", n) else 0)):
    if not n.endswith(".xml") or not (n.startswith("ppt/slides/slide") or n.startswith("ppt/notesSlides/") or n.startswith("ppt/charts/chart")): continue
    parts = [a or b for a, b in re.findall(r"<a:t>([^<]*)</a:t>|<c:v>([^<]*)</c:v>", z.read(n).decode("utf-8"))]
    for t in parts:
        if pat.search(t):
            hits += 1; print(n.split("/")[-1], "|", t[:160])
print("hits", hits)
