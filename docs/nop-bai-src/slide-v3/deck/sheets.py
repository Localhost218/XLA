import sys
from PIL import Image
d, ids, out = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
W, H = 1000, 562
sh = Image.new('RGB', (W * 2, H * ((len(ids) + 1) // 2)), 'white')
for k, i in enumerate(ids):
    im = Image.open(f'{d}/slide-{i:02d}.png').resize((W, H), Image.LANCZOS)
    sh.paste(im, ((k % 2) * W, (k // 2) * H))
sh.save(out)
