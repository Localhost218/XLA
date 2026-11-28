import sys, re
from pptx import Presentation
from pptx.util import Emu
PAT = re.compile(r"OV2640|ESP32[- ]?CAM|bản trước|bản cũ|ghi nhầm|archive|v2-nhom|YOLO11n|phiên bản cũ|OV3660|ESP32-S3|N16R8|q12|640", re.I)
path = sys.argv[1]
full = len(sys.argv) > 2
prs = Presentation(path)
def walk(shapes, depth=0):
    for sh in shapes:
        yield sh, depth
        if sh.shape_type == 6:  # group
            yield from walk(sh.shapes, depth+1)
for i, s in enumerate(prs.slides, 1):
    for sh, d in walk(s.shapes):
        texts = []
        if sh.has_text_frame:
            for pi, p in enumerate(sh.text_frame.paragraphs):
                t = "".join(r.text for r in p.runs)
                texts.append((f"p{pi}", t))
        if getattr(sh, "has_table", False) and sh.has_table:
            for ri, row in enumerate(sh.table.rows):
                for ci, c in enumerate(row.cells):
                    texts.append((f"r{ri}c{ci}", c.text))
        for k, t in texts:
            if full or PAT.search(t):
                print(f"S{i} [{sh.shape_id}:{sh.name}] d{d} {k}: {t!r}")
    if s.has_notes_slide:
        nt = s.notes_slide.notes_text_frame.text if s.notes_slide.notes_text_frame else ""
        if full or PAT.search(nt):
            print(f"S{i} NOTES: {nt!r}")
