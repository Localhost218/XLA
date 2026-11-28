"""Rewrite deck text + notes, drop agenda slide, renumber, switch every font to Be Vietnam Pro."""
import re
import sys

from lxml import etree
from pptx import Presentation

sys.path.insert(0, sys.argv[3])
import content_fix  # noqa: E402,F401
from content import BODY, DROP, NOTES  # noqa: E402

SRC, DST = sys.argv[1], sys.argv[2]
FONT = "Be Vietnam Pro"
FONT_SEMI = "Be Vietnam Pro SemiBold"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"

prs = Presentation(SRC)
slides = list(prs.slides)
n_old = len(slides)


def paragraphs(shapes):
    for sh in shapes:
        if sh.shape_type == 6:
            yield from paragraphs(sh.shapes)
        elif sh.has_text_frame:
            yield from sh.text_frame.paragraphs


# 1. body text
for no, mapping in BODY.items():
    pars = list(paragraphs(slides[no - 1].shapes))
    for old, new in mapping.items():
        hits = [p for p in pars if "".join(r.text for r in p.runs) == old]
        assert len(hits) == 1, (no, old, len(hits))
        p = hits[0]
        assert len(p.runs) == 1, (no, old)
        p.runs[0].text = new

# 2. notes paragraph 0
for no, text in NOTES.items():
    tf = slides[no - 1].notes_slide.notes_text_frame
    p0 = tf.paragraphs[0]
    assert len(p0.runs) == 1, no
    p0.runs[0].text = text

# 3. drop slides
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
for no in sorted(DROP, reverse=True):
    sid = ids[no - 1]
    prs.part.drop_rel(sid.rId)
    sldIdLst.remove(sid)
n_new = len(prs.slides)

# 4. renumber "NN / 23"
pat = re.compile(r"^(\d{2}) / %d$" % n_old)
for idx, s in enumerate(prs.slides, 1):
    for p in paragraphs(s.shapes):
        t = "".join(r.text for r in p.runs)
        if pat.match(t):
            p.runs[0].text = f"{idx:02d} / {n_new}"

# 5. fonts in every XML part
RPR = {"rPr", "defRPr", "endParaRPr"}


def fix_tree(root):
    changed = 0
    for el in root.iter("{%s}latin" % A, "{%s}ea" % A, "{%s}cs" % A, "{%s}buFont" % A):
        tf = el.get("typeface")
        if tf is None or tf.startswith("+"):
            continue
        new = FONT_SEMI if "semibold" in tf.lower() else FONT
        el.set("typeface", new)
        for att in ("panose", "pitchFamily", "charset"):
            if att in el.attrib:
                del el.attrib[att]
        changed += 1
    for el in list(root.iter("{%s}font" % A)):  # theme script fallbacks (Estrangelo Edessa, Ebrima, ...)
        el.getparent().remove(el)
        changed += 1
    return changed


total = 0
for part in prs.part.package.iter_parts():
    ct = part.content_type
    if not ct.endswith("+xml") and not ct.endswith("/xml"):
        continue
    if hasattr(part, "_element"):
        total += fix_tree(part._element)
    elif "theme" in ct:
        root = etree.fromstring(part.blob)
        c = fix_tree(root)
        if c:
            part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            total += c

prs.core_properties.title = "Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660"
prs.save(DST)
print("slides", n_old, "->", n_new, "font edits", total)

# 6. geometry fixes after the font change (new slide numbers)
E = 914400
def shape(n, sid):
    return next(sh for sh in prs.slides[n - 1].shapes if sh.shape_id == sid)
shape(3, 21).left = int(10.35 * E); shape(3, 21).width = int(2.3 * E)  # "100" overlapped "khung dev-val"
from pptx.util import Pt
for r in shape(18, 10).text_frame.paragraphs[0].runs:
    r.font.size = Pt(18)  # "Camera -> backend/vision" wrapped
prs.save(DST)
print("geometry fixed")
