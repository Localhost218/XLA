import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1])
print(z.read('docProps/core.xml').decode())
print(z.read('docProps/app.xml').decode()[:3000])
names = z.namelist()
print([n for n in names if re.match(r'ppt/slides/slide\d+\.xml$', n)])
pres = z.read('ppt/_rels/presentation.xml.rels').decode()
print(re.findall(r'Target="(slides/slide\d+\.xml)"', pres))
print([n for n in names if 'comment' in n.lower()])
for n in names:
    if n.endswith('.xml'):
        d = z.read(n).decode('utf-8','replace')
        for m in re.finditer(r'YOLO[^<"]{0,30}', d):
            print(n, m.group(0))
