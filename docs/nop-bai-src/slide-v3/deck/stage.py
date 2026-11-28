"""Native, editable stage diagrams (panels, nodes, arrows, objective bars, detail cards) + LaTeX formula pictures."""
import os

from lxml import etree
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

import kit
from kit import INK, MUTED, TEXT

EQDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tex", "eq")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tex", "fig_png")
RPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tex", "fig")
EQ_SCALE = 0.9  # every display formula: 18 pt source x 0.9 = 16,2 pt on the slide
RED = "B4232F"

# panel palette, the same everywhere in the deck: fill, border, objective-bar colour
PAL = {
    "yel": ("FFF5D1", "E5C15C", "8A6A12"),
    "lav": ("EEE8FA", "B4A2E6", "5B3FA0"),
    "blu": ("E3F0FB", "93C2EA", "1F5A96"),
    "mint": ("E2F4EC", "8ACDB0", "1E7A58"),
    "peach": ("FDECDF", "EFB58C", "B5531B"),
}
NODE = {"learn": ("EAF5E3", "7FAE63", False), "frozen": ("FFFFFF", "5F768D", True),
        "data": ("FFFFFF", "B8C6D3", False), "plain": ("F4F7FA", "B8C6D3", False), "dark": ("0E2238", "0E2238", False)}


def _shape(slide, x, y, w, h, fill, border, dashed=False, radius=0.12, lw=1.0):
    s = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if border is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = RGBColor.from_string(border); s.line.width = Pt(lw)
        if dashed:
            s.line.dash_style = MSO_LINE.DASH
    s.shadow.inherit = False
    tf = s.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(0.04))
    tf.word_wrap = True
    return s


def _label(shape, lines, size, color, bold=False, anchor="m", align="c"):
    tf = shape.text_frame
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[align]
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for t, st in runs:
            kit.add_runs(p, t, st.get("size", size), st.get("color", color), st.get("bold", bold))


def panel(slide, x, y, w, h, pal, title, dashed=False, title_size=14):
    fill, border, _ = PAL[pal] if pal else ("FFFFFF", "5F768D", True)
    s = _shape(slide, x, y, w, h, fill, border, dashed or pal is None, radius=0.06, lw=1.25)
    if title:
        kit.text(slide, x + 0.08, y + 0.08, w - 0.16, 0.34, title, title_size, INK, True, align="c")
    return s


def node(slide, x, y, w, h, label=None, kind="learn", size=12, color=INK, bold=False):
    fill, border, dashed = NODE[kind]
    s = _shape(slide, x, y, w, h, fill, border, dashed, radius=0.14)
    if label:
        _label(s, label, size, "FFFFFF" if kind == "dark" else color, bold)
    return s


def bar(slide, x, y, w, h, pal, label, size=13):
    s = _shape(slide, x, y, w, h, PAL[pal][2], None, radius=0.2)
    if label:
        _label(s, label, size, "FFFFFF", True)
    return s


def arrow(slide, x1, y1, x2, y2, color=INK, width=1.5, dash=False):
    c = kit.line(slide, x1, y1, x2, y2, color, width, dash)
    ln = c.line._get_or_add_ln()
    te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "med"); te.set("len", "med")
    return c


def detail(slide, x, y, w, h, label, lines, size=14):
    """White card with a red top rule, red caps label and 1-3 lines of concrete detail."""
    kit.card(slide, x, y, w, h)
    r = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.05))
    r.fill.solid(); r.fill.fore_color.rgb = RGBColor.from_string(RED); r.line.fill.background(); r.shadow.inherit = False
    kit.text(slide, x + 0.18, y + 0.14, w - 0.36, 0.26, label.upper(), 12, RED, True)
    kit.text(slide, x + 0.18, y + 0.42, w - 0.36, h - 0.5, lines, size, TEXT, spacing=1.05)


def eq_size(name, scale=EQ_SCALE):
    im = Image.open(os.path.join(EQDIR, name + ".png"))
    return im.width / 300 * scale, im.height / 300 * scale


def eq(slide, name, x, y, scale=EQ_SCALE, align="l"):
    w, h = eq_size(name, scale)
    if align == "c":
        x -= w / 2
    elif align == "r":
        x -= w
    slide.shapes.add_picture(os.path.join(EQDIR, name + ".png"), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def fig(slide, name, x, y, w=None, h=None):
    path = os.path.join(FIGDIR, name + ".png") if not name.endswith(".jpg") else os.path.join(RPT, name)
    return kit.picture(slide, path, x, y, w, h)


def framed(slide, path, x, y, w, h, border="B8C6D3"):
    p = kit.picture(slide, path, x, y, w, h)
    p.line.color.rgb = RGBColor.from_string(border); p.line.width = Pt(0.75)
    return p
