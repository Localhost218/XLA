import sys, glob, os
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, 'slide-*.png')))
os.makedirs(out, exist_ok=True)
for k in range(0, len(files), 4):
    sheet = Image.new('RGB', (1600, 900), 'white')
    for j, f in enumerate(files[k:k+4]):
        im = Image.open(f).convert('RGB').resize((796, 448), Image.LANCZOS)
        sheet.paste(im, ((j % 2) * 804, (j // 2) * 452))
    sheet.save(os.path.join(out, f'sheet-{k//4+1}.png'))
print(len(files))
