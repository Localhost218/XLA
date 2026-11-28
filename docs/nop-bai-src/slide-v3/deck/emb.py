import sys, zipfile, re, io
z = zipfile.ZipFile(sys.argv[1])
pat = re.compile(r"OV2640|ESP32[- ]?CAM|bản trước|bản cũ|ghi nhầm|archive|v2-nhom|YOLO11n", re.I)
for n in z.namelist():
    if n.startswith('ppt/embeddings/'):
        b = z.read(n)
        try:
            ez = zipfile.ZipFile(io.BytesIO(b))
            for m in ez.namelist():
                d = ez.read(m).decode('utf-8','replace')
                for x in pat.finditer(d): print(n, m, d[max(0,x.start()-60):x.end()+60])
        except Exception as e:
            print(n, 'not zip', e)
print('emb done', [n for n in z.namelist() if n.startswith('ppt/embeddings/')])
