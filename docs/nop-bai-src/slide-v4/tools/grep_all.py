import re, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
pat = re.compile(r"—|–|OV2640|ESP32-CAM|YOLO11n|không chỉ|đóng vai trò|Tóm lại|toàn diện|không đạt|Kill|nổi bật|vượt trội|đột phá|then chốt|khai phá|hành trình|Hãy cùng|Đáng chú ý|Có thể thấy rằng|thất bại|bản trước ghi nhầm", re.I)
hits = 0
for n in sorted(z.namelist()):
    if not n.endswith(".xml") or not ("slides/" in n or "notesSlides/" in n or "charts/" in n): continue
    if "slideLayout" in n or "slideMaster" in n: continue
    txt = "".join(re.findall(r"<a:t>([^<]*)</a:t>|<c:v>([^<]*)</c:v>", z.read(n).decode("utf-8")) and [a or b for a, b in re.findall(r"<a:t>([^<]*)</a:t>|<c:v>([^<]*)</c:v>", z.read(n).decode("utf-8"))] or [])
    for m in pat.finditer(txt):
        hits += 1; print(n, repr(txt[max(0, m.start()-40):m.end()+40]))
print("hits", hits)
