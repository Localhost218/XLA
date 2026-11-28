import sys, zipfile, re, collections
z = zipfile.ZipFile(sys.argv[1])
c = collections.Counter(); where = collections.defaultdict(set)
for n in z.namelist():
    if n.endswith('.xml'):
        d = z.read(n).decode('utf-8','replace')
        for m in re.finditer(r'<a:(latin|ea|cs|sym|font)\b[^>]*typeface="([^"]*)"', d):
            c[(m.group(1), m.group(2))] += 1; where[(m.group(1), m.group(2))].add(n.split('/')[1] if '/' in n else n)
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(v, k, sorted(where[k])[:6])
