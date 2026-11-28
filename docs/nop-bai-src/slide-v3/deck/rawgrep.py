import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1])
pat = re.compile(r"OV2640|ESP32[- ]?CAM|bản trước|bản cũ|ghi nhầm|archive|v2-nhom|YOLO11n|phiên bản cũ|nhầm", re.I)
for n in z.namelist():
    if n.endswith(('.xml', '.rels')):
        d = z.read(n).decode('utf-8', 'replace')
        for m in pat.finditer(d):
            print(n, '::', d[max(0,m.start()-120):m.end()+80].replace('\n',' '))
