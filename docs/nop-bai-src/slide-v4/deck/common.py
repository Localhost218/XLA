"""Deck plumbing shared by the v8 build: clone a template slide, set badge/title/background, clear, notes."""
import copy

from pptx.dml.color import RGBColor
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches

import kit

PRS = None
S = None          # v7 slides, 0-based
PARTS = {}        # part key -> (template slide, badge colour)


def init(prs):
    global PRS, S
    PRS = prs
    S = list(prs.slides)
    kit.init_card(S[21], 7)  # card shape of the v7 "limits" slide
    PARTS.update({"blue": (S[1], "0284C7"), "teal": (S[4], "0D9488"), "orange": (S[6], "D97706"),
                  "purple": (S[11], "6D45D9"), "end": (S[20], "0284C7")})


def _hdr(slide):
    return {s.shape_id: s for s in slide.shapes}


def set_bg(slide, part):
    src = PARTS[part][0]
    if src is slide:
        return
    old = slide._element.cSld.find(qn("p:bg"))
    if old is not None:
        old.getparent().remove(old)
    nbg = copy.deepcopy(src._element.cSld.find(qn("p:bg")))
    blip = nbg.find(".//" + qn("a:blip"))
    img = src.part.related_part(blip.get(qn("r:embed")))
    blip.set(qn("r:embed"), slide.part.relate_to(img, RT.IMAGE))
    slide._element.cSld.insert(0, nbg)


def header(slide, part, badge, title):
    sh = _hdr(slide)
    colour = PARTS[part][1]
    w = 0.42 + 0.113 * len(badge)
    for k in (3, 4):
        sh[k].width = Inches(w)
    sh[3].fill.solid(); sh[3].fill.fore_color.rgb = RGBColor.from_string(colour)
    sh[4].text_frame.paragraphs[0].runs[0].text = badge
    sh[5].text_frame.paragraphs[0].runs[0].text = title
    set_bg(slide, part)


def clear(slide, keep_pictures=False):
    for s in list(slide.shapes):
        if s.shape_id in kit.HEADER_IDS:
            continue
        if keep_pictures and s.shape_type == 13:
            continue
        s._element.getparent().remove(s._element)
    xml = slide._element.xml
    for rid, rel in list(slide.part.rels.items()):
        if rel.reltype in (RT.CHART, RT.IMAGE) and f'"{rid}"' not in xml:
            slide.part.drop_rel(rid)


def new(part):
    src = PARTS[part][0]
    s = PRS.slides.add_slide(src.slide_layout)
    for e in list(s.shapes):
        e._element.getparent().remove(e._element)
    for e in src.shapes:
        if e.shape_id in kit.HEADER_IDS:
            s.shapes._spTree.append(copy.deepcopy(e._element))
    ns = s.notes_slide
    for e in list(ns.shapes):
        e._element.getparent().remove(e._element)
    for e in src.notes_slide.shapes:
        ns.shapes._spTree.append(copy.deepcopy(e._element))
    set_bg(s, part)
    return s


def notes(slide, points, source):
    """Brief pointers only; the transcript agent writes the full script."""
    tf = slide.notes_slide.notes_text_frame
    tf.text = points[0]
    for p in points[1:]:
        tf.add_paragraph().text = p
    tf.add_paragraph().text = "Nguồn: " + source
