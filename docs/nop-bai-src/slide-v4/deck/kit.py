"""Helpers to add template-styled content (cards, text, native charts, booktabs table) to the XLA deck."""
import copy

from lxml import etree
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FONT, SEMI = "Be Vietnam Pro", "Be Vietnam Pro SemiBold"
INK, TEXT, MUTED, GRID = "0E2238", "33475C", "5F768D", "E1E9F0"
SLATE, GOLD, BLUE, TEAL, PURPLE, PINK, ORANGE = "8CA0B3", "C99A5B", "0284C7", "0D9488", "6D45D9", "D6246E", "D97706"
CARD_XML = None  # set by init_card(slide)
HEADER_IDS = {3, 4, 5, 6}


def vn(x, nd=3):
    s = f"{x:.{nd}f}".replace(".", ",")
    return s.replace("-", "−")


def init_card(slide, shape_id):
    global CARD_XML
    sh = next(s for s in slide.shapes if s.shape_id == shape_id)
    CARD_XML = copy.deepcopy(sh._element)


from fontTools.ttLib import TTFont as _TT
_CMAP = set(_TT("D:/IOT_Tools/downloads/fonts/BeVietnamPro-Regular.ttf").getBestCmap())
MATHFONT = "Cambria Math"


def add_runs(p, t, size, color, bold=False):
    """Add text to paragraph p; characters Be Vietnam Pro lacks (Greek, arrows, set symbols) go in Cambria Math runs."""
    import re as _re
    t = _re.sub(r"(?<=\d) (?=\d{3}(?!\d))", " ", t)  # 1 115: never break a number across lines
    segs, cur, cur_m = [], "", None
    for ch in t:
        m = ord(ch) not in _CMAP and not ch.isspace()
        if cur and m != cur_m:
            segs.append((cur, cur_m)); cur = ""
        cur += ch; cur_m = m
    if cur:
        segs.append((cur, cur_m))
    runs = []
    for seg, m in segs:
        r = p.add_run(); r.text = seg
        _font(r, size, color, bold, MATHFONT if m else None)
        runs.append(r)
    return runs


def _font(run, size, color, bold=False, face=None):
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    rPr = run._r.get_or_add_rPr()
    name = face or (SEMI if bold else FONT)
    if bold:
        rPr.set("b", "1")
    for tag in ("latin", "ea", "cs"):
        el = rPr.find(qn(f"a:{tag}"))
        if el is None:
            el = etree.SubElement(rPr, qn(f"a:{tag}"))
        el.set("typeface", name)


def card(slide, x, y, w, h, fill=None):
    el = copy.deepcopy(CARD_XML)
    off = el.find(".//" + qn("a:off"))
    ext = el.find(".//" + qn("a:ext"))
    off.set("x", str(int(Inches(x)))); off.set("y", str(int(Inches(y))))
    ext.set("cx", str(int(Inches(w)))); ext.set("cy", str(int(Inches(h))))
    if fill:
        el.find(".//" + qn("a:srgbClr")).set("val", fill)
    nv = el.find(".//" + qn("p:cNvPr"))
    nv.set("id", str(_next_id(slide)))
    slide.shapes._spTree.append(el)
    return el


def _next_id(slide):
    return max([int(e.get("id")) for e in slide.shapes._spTree.iter(qn("p:cNvPr"))] + [1]) + 1


def text(slide, x, y, w, h, lines, size=14, color=TEXT, bold=False, align="l", anchor="t", spacing=None):
    """lines: str or list of str / list of (str, dict) runs per paragraph."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if spacing:
            p.line_spacing = spacing
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for t, st in runs:
            add_runs(p, t, st.get("size", size), st.get("color", color), st.get("bold", bold))
    return tb


def chip(slide, x, y, w, h, label, color, size=11):
    s = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))  # rounded rectangle
    s.adjustments[0] = 0.5
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(color)
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    _font(r, size, "FFFFFF", True)
    return s


def line(slide, x1, y1, x2, y2, color, width=1.5, dash=False):
    c = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(color)
    c.line.width = Pt(width)
    if dash:
        ln = c.line._get_or_add_ln()
        d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    return c


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


# ---------- charts ----------

def _chart_fonts(chart, size=12, color=TEXT):
    chart.font.size = Pt(size)
    chart.font.color.rgb = RGBColor.from_string(color)
    chart.font.name = FONT


def _manual_layout(chart, x, y, w, h):
    """Fix the inner plot area as fractions of the chart frame, so overlays can be placed exactly."""
    plotArea = chart._chartSpace.find(".//" + qn("c:plotArea"))
    lay = plotArea.find(qn("c:layout"))
    if lay is None:
        lay = etree.Element(qn("c:layout")); plotArea.insert(0, lay)
    for ch in list(lay):
        lay.remove(ch)
    ml = etree.SubElement(lay, qn("c:manualLayout"))
    for tag, val in (("layoutTarget", "inner"), ("xMode", "edge"), ("yMode", "edge"), ("x", x), ("y", y), ("w", w), ("h", h)):
        etree.SubElement(ml, qn(f"c:{tag}")).set("val", str(val))


def _axis_style(ax, show_labels=True, grid=False, line_color=GRID):
    ax.has_major_gridlines = grid
    if grid:
        gl = ax.major_gridlines.format.line
        gl.color.rgb = RGBColor.from_string(GRID); gl.width = Pt(0.75)
    ax.format.line.color.rgb = RGBColor.from_string(line_color)
    if not show_labels:
        ax.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    ax.tick_labels.font.size = Pt(12)
    ax.tick_labels.font.color.rgb = RGBColor.from_string(TEXT)


def _label_points(series, labels, size=12, color=INK, bold=True, pos=None):
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        dl = series.points[i].data_label
        tf = dl.text_frame
        tf.text = lab
        for r in tf.paragraphs[0].runs:
            _font(r, size, color, bold)
        if pos is not None:
            dl.position = pos[i] if isinstance(pos, list) else pos


def _color_series(series, color, points=None):
    series.format.fill.solid(); series.format.fill.fore_color.rgb = RGBColor.from_string(color)
    series.format.line.fill.background()
    if points:
        for i, c in points.items():
            pt = series.points[i]
            pt.format.fill.solid(); pt.format.fill.fore_color.rgb = RGBColor.from_string(c)
            pt.format.line.fill.background()


def _err_bars(series, plus, minus, color=INK):
    ser = series._element
    eb = etree.Element(qn("c:errBars"))
    etree.SubElement(eb, qn("c:errBarType")).set("val", "both")
    etree.SubElement(eb, qn("c:errValType")).set("val", "cust")
    etree.SubElement(eb, qn("c:noEndCap")).set("val", "0")
    for tag, vals in (("c:plus", plus), ("c:minus", minus)):
        holder = etree.SubElement(eb, qn(tag))
        nl = etree.SubElement(holder, qn("c:numLit"))
        etree.SubElement(nl, qn("c:formatCode")).text = "General"
        etree.SubElement(nl, qn("c:ptCount")).set("val", str(len(vals)))
        for i, v in enumerate(vals):
            pt = etree.SubElement(nl, qn("c:pt")); pt.set("idx", str(i))
            etree.SubElement(pt, qn("c:v")).text = repr(float(v))
    spPr = etree.SubElement(eb, qn("c:spPr"))
    ln = etree.SubElement(spPr, qn("a:ln")); ln.set("w", "15875")
    sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    cat = ser.find(qn("c:cat"))
    cat.addprevious(eb)


def bar_chart(slide, x, y, w, h, cats, series, horizontal=False, gap=60, overlap=0, legend=False,
              vmin=None, vmax=None, major=None, axis_labels=False, grid=False, layout=None, number_format='0', legend_pos=None):
    """series: list of (name, values, color, labels or None, point_colors or None)."""
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, *_ in series:
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _chart_fonts(ch)
    ch.has_title = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = legend_pos or XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
    plot = ch.plots[0]; plot.gap_width = gap; plot.overlap = overlap
    va, ca = ch.value_axis, ch.category_axis
    if vmin is not None: va.minimum_scale = vmin
    if vmax is not None: va.maximum_scale = vmax
    if major is not None: va.major_unit = major
    va.tick_labels.number_format = number_format; va.tick_labels.number_format_is_linked = False
    _axis_style(va, show_labels=axis_labels, grid=grid)
    va.format.line.fill.background()
    _axis_style(ca)
    ca.major_tick_mark = 2  # none
    va.major_tick_mark = 2
    if horizontal:
        ca.reverse_order = True
        # reversed categories put the value axis on top; make it cross the category axis at its max (= bottom)
        cr = va._element.find(qn("c:crosses"))
        if cr is not None:
            cr.set("val", "max")
    for s, (name, vals, color, labels, pcol, *rest) in zip(plot.series, series):
        _color_series(s, color, pcol)
        if labels:
            _label_points(s, labels, pos=rest[0] if rest else None)
    if layout:
        _manual_layout(ch, *layout)
    return gf


def scatter(slide, x, y, w, h, points, xr, yr, layout):
    """points: list of (label, xv, yv, color, label_pos). xr/yr: (min, max, major)."""
    cd = XyChartData()
    for lab, xv, yv, *_ in points:
        s = cd.add_series(lab); s.add_data_point(xv, yv)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _chart_fonts(ch)
    ch.has_title = False
    ch.has_legend = False
    for s, (lab, xv, yv, color, pos) in zip(ch.plots[0].series, points):
        s.marker.style = 8  # circle
        s.marker.size = 13
        s.marker.format.fill.solid(); s.marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
        s.marker.format.line.color.rgb = RGBColor.from_string("FFFFFF")
        s.format.line.fill.background()
        _label_points(s, [lab], size=12, color=INK, bold=True, pos=pos)
    for ax, (lo, hi, mj) in ((ch.value_axis, yr), (ch.category_axis, xr)):
        ax.minimum_scale, ax.maximum_scale, ax.major_unit = lo, hi, mj
        _axis_style(ax, show_labels=False, grid=True)
        ax.major_tick_mark = 2
    _manual_layout(ch, *layout)
    return gf


def plot_xy(frame_x, frame_y, frame_w, frame_h, layout):
    """Inner plot box in slide inches."""
    lx, ly, lw, lh = layout
    return frame_x + lx * frame_w, frame_y + ly * frame_h, lw * frame_w, lh * frame_h


# ---------- booktabs table ----------

def _border(tcPr, side, w, color):
    ln = etree.SubElement(tcPr, qn(f"a:ln{side}"))
    ln.set("w", str(int(w))); ln.set("cmpd", "sng")
    if w == 0:
        etree.SubElement(ln, qn("a:noFill"))
    else:
        sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", color)


def table(slide, x, y, col_w, row_h, rows, size=13, header_rows=1, aligns=None):
    """rows: list of list of str (or (str, dict)). Booktabs: top + mid + bottom rule, no fills."""
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid" base; borders set explicitly below
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tbl.cell(i, j)
            val = rows[i][j]
            t, st = (val, {}) if isinstance(val, str) else val
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[(aligns or "l" * nc)[j]]
            hdr = i < header_rows
            add_runs(p, t, st.get("size", size - 1 if hdr else size), st.get("color", MUTED if hdr else TEXT), st.get("bold", hdr))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = 0; cell.margin_bottom = 0
            tcPr = cell._tc.get_or_add_tcPr()
            for ch in list(tcPr):
                tcPr.remove(ch)
            top = Emu(Pt(1.5)) if i == 0 else 0
            bottom = Emu(Pt(1.5)) if i == nr - 1 else (Emu(Pt(0.75)) if i == header_rows - 1 else Emu(Pt(0.25)))
            bcol = INK if (i == 0 or i == nr - 1 or i == header_rows - 1) else GRID
            _border(tcPr, "L", 0, INK); _border(tcPr, "R", 0, INK)
            _border(tcPr, "T", top, INK)
            _border(tcPr, "B", bottom, bcol)
            etree.SubElement(tcPr, qn("a:noFill"))
    return gf


def line_chart(slide, x, y, w, h, cats, series, vmin=None, vmax=None, major=None, layout=None, legend=False,
               number_format='0', axis_labels=True):
    """series: list of (name, values, color, dashed). Category x axis, straight lines with round markers."""
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, *_ in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _chart_fonts(ch)
    ch.has_title = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
    va, ca = ch.value_axis, ch.category_axis
    if vmin is not None: va.minimum_scale = vmin
    if vmax is not None: va.maximum_scale = vmax
    if major is not None: va.major_unit = major
    va.tick_labels.number_format = number_format; va.tick_labels.number_format_is_linked = False
    _axis_style(va, show_labels=axis_labels, grid=True)
    va.format.line.fill.background()
    _axis_style(ca)
    ca.major_tick_mark = 2; va.major_tick_mark = 2
    for s, (name, vals, color, dashed) in zip(ch.plots[0].series, series):
        s.smooth = False
        s.format.line.color.rgb = RGBColor.from_string(color); s.format.line.width = Pt(2.25)
        if dashed:
            ln = s.format.line._get_or_add_ln()
            d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
        s.marker.style = 8; s.marker.size = 6
        s.marker.format.fill.solid(); s.marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
        s.marker.format.line.color.rgb = RGBColor.from_string(color)
    if layout:
        _manual_layout(ch, *layout)
    return gf


def log_axis(gf, base=10):
    """Turn the value axis of a chart into a log axis."""
    va = gf.chart.value_axis._element
    sc = va.find(qn("c:scaling"))
    lb = etree.Element(qn("c:logBase")); lb.set("val", str(base))
    sc.insert(0, lb)
