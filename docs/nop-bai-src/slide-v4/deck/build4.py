"""v7 template -> AI v2 deck (<= 30 slides). Usage: py -3.13 -I build4.py v7.pptx out.pptx <deckdir>"""
import copy
import os
import re
import unicodedata
import sys

from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt

sys.path.insert(0, sys.argv[3])
import common as C  # noqa: E402
import kit  # noqa: E402
import slides1 as A  # noqa: E402
import slides1b as A2  # noqa: E402
import slides2 as B  # noqa: E402
import slides_v2 as V  # noqa: E402

SRC, DST = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
C.init(prs)
S = C.S
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
RPTF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tex", "fig")


# ---------------------------------------------------------------- cover: third tile becomes the boxed frame
def cover(s):
    sh = {x.shape_id: x for x in s.shapes}
    old = sh[22]
    pic = s.shapes.add_picture(os.path.join(IMG, "box_002_00005_t2.jpg"), old.left, old.top, old.width, old.height)
    old._element.addprevious(pic._element)
    old._element.getparent().remove(old._element)
    sh[21].text_frame.paragraphs[0].runs[0].text = "Ảnh rõ classic"
    sh[24].text_frame.paragraphs[0].runs[0].text = "Hộp sau tầng 2"
    sh[15].text_frame.paragraphs[0].runs[0].text = "GVHD: PGS.TS. Phạm Văn Cường  ·  26/11/2026"
    sh[15].width = Inches(6.2)
    C.notes(s, ["Đề tài: làm rõ ảnh và tìm người trên khung bay thật của camera ESP32-S3 + OV3660, JPEG 640×480 q12.",
                "Hệ ba tầng tìm người (nhận diện, xác minh, lọc track) và Ảnh rõ classic ONNX bật sẵn.",
                "Ba ảnh nhỏ: khung gốc, sau bộ lọc cổ điển, và một khung test có hộp sau tầng 2."],
            "results/v2/README.md; BAO_CAO_DO_AN.md mục 6.")
    return s


order = [cover(S[0]), A.s_problem(), V.s_system(), A.s_theory1(), A.s_theory2(), A.s_data(), V.s_ext_data(),
         A.s_d2(), A.s_realism(), A2.s_yolo(), A2.s_metrics(), A2.s_training(), B.s_a1(), V.s_detector_cmp(),
         V.s_verifier(), V.s_verifier_res(), V.s_calib(), V.s_system_res(), V.s_boxes(),
         B.s_classic(), B.s_dehaze(), V.s_classic_onnx(), B.s_nafnet(), V.s_enh_cmp(), V.s_enh_imgs(), V.s_ab(),
         V.s_iotcv(), V.s_refs()]
assert len(order) <= 30, len(order)
assert len({id(x) for x in order}) == len(order)

# ---------------------------------------------------------------- reorder and drop unused template slides
lst = prs.slides._sldIdLst
by_part = {prs.part.related_part(e.rId): e for e in lst}
keep = [by_part[sl.part] for sl in order]
for e in list(lst):
    lst.remove(e)
    if e not in keep:
        prs.part.drop_rel(e.rId)
for e in keep:
    lst.append(e)

# ---------------------------------------------------------------- page numbers + title as first notes line
n = len(prs.slides)
pat = re.compile(r"^\d{2} / \d{2}$")
for i, sl in enumerate(prs.slides, 1):
    title = None
    for sh in sl.shapes:
        if sh.has_text_frame and pat.match(sh.text_frame.text):
            sh.text_frame.paragraphs[0].runs[0].text = f"{i:02d} / {n}"
        if sh.shape_id == 5 and sh.has_text_frame and i > 1:
            title = sh.text_frame.text
            if len(title) > 64:  # keep long titles on one line
                sh.width = Inches(12.3)
                for rr in sh.text_frame.paragraphs[0].runs:
                    rr.font.size = Pt(min(rr.font.size.pt if rr.font.size else 28, int(28 * 58 / len(title))))
    if title:
        tf = sl.notes_slide.notes_text_frame
        first = tf.paragraphs[0]
        p = copy.deepcopy(first._p)
        first._p.addprevious(p)
        for r in list(p):
            if r.tag.endswith("}r"):
                p.remove(r)
        tf.paragraphs[0].text = "Tiêu đề: " + title

# ---------------------------------------------------------------- glyphs Be Vietnam Pro lacks -> Cambria Math runs
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def split_runs(root):
    changed = 0
    for r in list(root.iter("{%s}r" % A_NS)):
        t = r.find("{%s}t" % A_NS)
        if t is None or not t.text:
            continue
        miss = [ord(c) not in kit._CMAP and not c.isspace() for c in t.text]
        if not any(miss):
            continue
        rpr = r.find("{%s}rPr" % A_NS)
        if rpr is not None:
            lat = rpr.find("{%s}latin" % A_NS)
            if lat is not None and lat.get("typeface") == kit.MATHFONT and all(miss):
                continue
        segs, cur, cm = [], "", None
        for c, m in zip(t.text, miss):
            if cur and m != cm:
                segs.append((cur, cm)); cur = ""
            cur += c; cm = m
        segs.append((cur, cm))
        parent = r.getparent(); idx = parent.index(r)
        for k, (seg, m) in enumerate(segs):
            nr = copy.deepcopy(r)
            nr.find("{%s}t" % A_NS).text = seg
            if m:
                np_ = nr.find("{%s}rPr" % A_NS)
                if np_ is None:
                    np_ = etree.Element("{%s}rPr" % A_NS); nr.insert(0, np_)
                for tag in ("latin", "ea", "cs"):
                    el = np_.find("{%s}%s" % (A_NS, tag))
                    if el is None:
                        el = etree.SubElement(np_, "{%s}%s" % (A_NS, tag))
                    el.set("typeface", kit.MATHFONT)
            parent.insert(idx + k, nr)
        parent.remove(r)
        changed += 1
    return changed


def no_dash(root):
    # brief §3: no en/em dash anywhere; ranges use a plain hyphen
    k = 0
    for el in root.iter("{%s}t" % A_NS, "{http://schemas.openxmlformats.org/drawingml/2006/chart}v"):
        if el.text and ("–" in el.text or "—" in el.text):
            el.text = el.text.replace("–", "-").replace("—", "-"); k += 1
    return k


dash = 0
for sl in prs.slides:
    dash += no_dash(sl._element) + no_dash(sl.notes_slide._element)
    for rel in sl.part.rels.values():
        if rel.reltype.endswith("/chart"):
            dash += no_dash(rel.target_part._element)
print("dash fixes", dash)

# no version mentions anywhere: a notes source line that cites a versioned path is dropped
VER = re.compile(r"\bv1\b|\bv2\b|phiên bản|ship|v\d\.\d\.\d", re.I)
dropped = 0
for sl in prs.slides:
    tf = sl.notes_slide.notes_text_frame
    for p in list(tf.paragraphs):
        if unicodedata.normalize("NFC", p.text).startswith(unicodedata.normalize("NFC", "Nguồn:")) and VER.search(p.text):
            p._p.getparent().remove(p._p); dropped += 1
print("source lines dropped", dropped)

total = 0
for sl in prs.slides:
    total += split_runs(sl._element)
    for rel in sl.part.rels.values():
        if rel.reltype.endswith("/chart"):
            total += split_runs(rel.target_part._element)
prs.core_properties.title = "Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660"
prs.save(DST)
print("slides", n, "glyph splits", total)
