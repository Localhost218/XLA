"""Dựng kịch bản thuyết trình AI v2 (.docx, đọc nguyên văn theo từng slide) từ transcript.yaml, rồi kiểm.

Chạy:  PYTHONUTF8=1 py -3.13 -I build_transcript.py [--yaml ...] [--outline ...] [--pptx ...] [--out ...]
                                                     [--text-out <txt>]
Cần python-docx, python-pptx, PyYAML. Chỉ đọc pptx và OUTLINE, không sửa.

transcript.yaml:
  project, subtitle          -- hai dòng đầu file
  title_overrides: {N: str}  -- tiêu đề slide N thay chuỗi của OUTLINE (đã chốt với người làm slide)
  numbers_derived: {số: lý do}  -- số tự tính (hiệu, tỉ lệ) không có nguyên văn trong nguồn
  slides: [{n, text: [str | {f: str}]}]  -- str = lời nói; {f: ...} = dòng công thức, đặt ngay trước đoạn đọc nó

Đề mục mỗi slide = "Slide N: <cột Tiêu đề của OUTLINE.md>" (style Heading 1, in đậm).
Kiểm: đủ slide 1..28 đúng thứ tự; đề mục đọc lại từ docx khớp từng chữ; regex brief §3 và cụm từ cấm trên toàn bộ chữ của
docx; không nhắc phiên bản (VERSION_RE); số từ lời nói mỗi slide (cảnh báo ngoài 250–600); mọi con số trong lời nói
và công thức có trong nguồn hoặc trong
``numbers_derived``. Mã thoát 0 = dựng xong, mọi kiểm đạt; 1 = có kiểm không đạt.
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import yaml
from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
XLA = Path(r"D:\XLA")
SRC = XLA / "docs/nop-bai-src"
DEFAULT_OUTLINE = SRC / "slide-v4/OUTLINE.md"
DEFAULT_PPTX = XLA / "docs/nop-bai/v2/XLA_UAV_Nhom10.pptx"
DEFAULT_OUT = XLA / "docs/nop-bai/v2/TranscriptThuyetTrinh_Nhom10.docx"
ULTRA = Path(r"D:\DevCache\venvs\xla\Lib\site-packages\ultralytics")
FONT = "Be Vietnam Pro"
MATH_FONT = "Cambria Math"
BODY_PT, HEAD_PT, MATH_PT = 13, 15, 12.5
N_SLIDES = 28
WORDS_MIN, WORDS_MAX = 250, 600
WORDS_EXEMPT = {1, 28}  # bìa và tài liệu tham khảo: không phải slide nội dung
BRIEF_RE = re.compile(r"—|–|OV2640|ESP32-CAM|YOLO11n|không chỉ|đóng vai trò|Tóm lại|toàn diện|không đạt|Kill")
VERSION_RE = re.compile(r"\bv1\b|\bv2\b|AI v2|phiên bản|bản trước|bản này|ship|nâng cấp|bundle|-v\d", re.I)
BANNED = ["nổi bật", "vượt trội", "đột phá", "then chốt", "khai phá", "hành trình", "Hãy cùng", "Đáng chú ý",
          "Có thể thấy rằng", "thất bại", "Nhìn chung", "Bên cạnh đó", "Ngoài ra", "Hơn nữa", "Như vậy", "bản cũ",
          "mạnh mẽ", "tối ưu", "mang lại", "giải pháp hiệu quả"]
SOURCE_GLOBS = [
    "docs/nop-bai-src/BRIEF-20261126.md", "docs/nop-bai-src/slide-v4/OUTLINE.md",
    "docs/nop-bai-src/slide-v4/tex/eqlist.py", "docs/nop-bai-src/latex-v2/sections/*.tex",
    "docs/nop-bai-src/latex/sections/*.tex", "docs/nop-bai-src/latex/fig/*.tex",
    "docs/nop-bai-src/transcript/transcript.yaml", "BAO_CAO_DO_AN.md", "configs/*.yaml", "xla/**/*.py",
    "scripts/*.py", "models/release/*/meta.json", "results/**/README.md", "results/reproduce/tables/*.md",
    "results/**/*.json",
]
MAX_SOURCE_BYTES = 3_000_000
ULTRA_FILES = ["cfg/models/11/yolo11.yaml", "cfg/default.yaml", "utils/loss.py", "utils/tal.py"]
NUM = re.compile(r"\d+(?:,\d+)?")
GROUP_SPACE = re.compile(r"(?<=\d)(?:\\,|\s|\u00a0|\u202f)(?=\d{3}\b)")


def outline_titles(path):
    """{N: tiêu đề} từ bảng ``| # | Badge | Tiêu đề | Ý ... |`` của OUTLINE.md."""
    titles = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|([^|]*)\|([^|]*)\|", line)
        if m:
            titles[int(m.group(1))] = m.group(3).strip()
    return titles


def display(text):
    """Chữ hiển thị: khoảng nhóm nghìn và khoảng trước % thành NBSP để không gãy dòng giữa số."""
    text = re.sub(r"(?<=\d) (?=\d{3}\b)", "\u00a0", text)
    return text.replace(" %", "\u00a0%")


def set_font(run, size, bold=False, italic=False, color=None, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), font)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def para(doc, text, size, *, style=None, bold=False, color=None, before=0, after=6, keep=False, indent=0.0,
         font=FONT, spacing=1.25, page_break=False):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing
    pf.keep_with_next = keep
    pf.page_break_before = page_break
    if indent:
        pf.left_indent = Cm(indent)
    set_font(p.add_run(display(text)), size, bold, False, color, font)
    return p


def plain_style_font(style, font):
    """Bỏ font theo theme của style (Heading 1 mặc định dùng theme) để font đặt tay có hiệu lực."""
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(k), None)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(k), font)


def build(data, titles, out):
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.8)
    s.top_margin = s.bottom_margin = Cm(1.6)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    plain_style_font(normal, FONT)
    h1 = doc.styles["Heading 1"]
    plain_style_font(h1, FONT)
    h1.font.size = Pt(HEAD_PT)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor.from_string("1F3A5F")
    doc.core_properties.title = data["project"]

    para(doc, data["project"], 16, bold=True, after=2)
    para(doc, data["subtitle"], BODY_PT, color="555555", after=12)
    for i, sl in enumerate(data["slides"]):
        para(doc, f"Slide {sl['n']}: {titles[sl['n']]}", HEAD_PT, style="Heading 1", bold=True, color="1F3A5F",
             before=0 if i else 6, after=8, keep=True, page_break=i > 0, spacing=1.15)
        for t in sl["text"]:
            if isinstance(t, dict):  # {f: ...}: dòng công thức, đặt ngay trước đoạn đọc nó
                para(doc, t["f"], MATH_PT, color="222222", indent=0.8, before=2, after=6, font=MATH_FONT, keep=True,
                     spacing=1.15)
            else:
                para(doc, t, BODY_PT)
    doc.save(out)


def deck_slides(pptx_path):
    from pptx import Presentation

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == 6:
                yield from walk(sh.shapes)
            else:
                yield sh

    per_slide, everything = [], []
    for sl in Presentation(str(pptx_path)).slides:
        texts = []
        for sh in walk(sl.shapes):
            if sh.has_text_frame:
                texts.append(re.sub(r"\s+", " ", sh.text_frame.text).strip())
            if getattr(sh, "has_table", False) and sh.has_table:
                texts += [c.text for r in sh.table.rows for c in r.cells]
            if getattr(sh, "has_chart", False) and sh.has_chart:
                for pl in sh.chart.plots:
                    for se in pl.series:
                        texts += [str(v) for v in se.values if v is not None]
        notes = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
        per_slide.append(texts)
        everything += texts + [notes]
    return per_slide, "\n".join(everything)


def source_corpus(deck_all):
    parts = [deck_all]
    for g in SOURCE_GLOBS:
        for f in glob.glob(str(XLA / g), recursive=True):
            p = Path(f)
            if p.stat().st_size <= MAX_SOURCE_BYTES:
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
    for f in ULTRA_FILES:
        p = ULTRA / f
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    s = "\n".join(parts).replace("{,}", ",")  # LaTeX viết dấu phẩy thập phân là {,}
    return GROUP_SPACE.sub("", s)


def check(data, titles, outline, pptx_path, out, text_out):
    ok = True
    nums = [sl["n"] for sl in data["slides"]]
    if nums != list(range(1, N_SLIDES + 1)):
        print(f"FAIL slides must be 1..{N_SLIDES} in order, got {nums}")
        ok = False

    doc = Document(out)
    heads = [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]
    expect = [f"Slide {n}: {titles[n]}" for n in range(1, N_SLIDES + 1)]
    bad = [(i + 1, h) for i, (h, e) in enumerate(zip(heads, expect)) if h.replace("\u00a0", " ") != e]
    if len(heads) != N_SLIDES or bad:
        print(f"FAIL headings: {len(heads)} found, mismatches {bad}")
        ok = False
    else:
        print(f"headings: {len(heads)}/{N_SLIDES}, match OUTLINE word for word"
              + (f" (override: slides {sorted(data.get('title_overrides') or {})})" if data.get("title_overrides")
                 else ""))
    for n, t in (data.get("title_overrides") or {}).items():
        print(f"  slide {n}: OUTLINE {outline.get(int(n))!r} -> {t!r}")

    per_slide, deck_all = deck_slides(pptx_path)
    print(f"pptx slides: {len(per_slide)}")
    for n in range(1, min(N_SLIDES, len(per_slide)) + 1):
        if titles[n].replace(" (bìa)", "") not in per_slide[n - 1] and titles[n] not in per_slide[n - 1]:
            print(f"  INFO slide {n}: heading title not found verbatim on pptx slide")

    doc_text = "\n".join(p.text for p in doc.paragraphs)
    if text_out:
        text_out.write_text(doc_text, encoding="utf-8")
        print(f"text dump: {text_out}")
    hits = [(i + 1, m.group(0)) for i, line in enumerate(doc_text.splitlines()) for m in BRIEF_RE.finditer(line)]
    if hits:
        print(f"FAIL brief §3 regex: {hits}")
        ok = False
    else:
        print("brief §3 regex: clean")
    vhits = [(i + 1, m.group(0)) for i, line in enumerate(doc_text.splitlines()) for m in VERSION_RE.finditer(line)]
    if vhits:
        print(f"FAIL version mentions: {vhits}")
        ok = False
    else:
        print("version mentions: none")
    low = doc_text.lower()
    found = [(b, low.count(b.lower())) for b in BANNED if b.lower() in low]
    if found:
        print(f"FAIL banned phrases: {found}")
        ok = False
    else:
        print("banned phrases: none")

    total = 0
    for sl in data["slides"]:
        w = len(" ".join(t for t in sl["text"] if isinstance(t, str)).split())
        total += w
        flag = "" if sl["n"] in WORDS_EXEMPT or WORDS_MIN <= w <= WORDS_MAX else f"  WARN outside {WORDS_MIN}-{WORDS_MAX}"
        print(f"  slide {sl['n']:>2}: {w} words{flag}")
    n_f = sum(isinstance(t, dict) for sl in data["slides"] for t in sl["text"])
    print(f"spoken words: {total}; formula lines: {n_f}")

    corpus = source_corpus(deck_all)
    derived = {str(k) for k in (data.get("numbers_derived") or {})}
    missing = []
    for sl in data["slides"]:
        blob = GROUP_SPACE.sub("", " ".join(t if isinstance(t, str) else t["f"] for t in sl["text"]))
        for n in sorted(set(NUM.findall(blob))):
            if n in derived or n in corpus or n.replace(",", ".") in corpus:
                continue
            missing.append((sl["n"], n))
    if missing:
        print(f"FAIL numbers not found in any source: {missing}")
        ok = False
    else:
        print("numbers: all found in sources or numbers_derived")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=HERE / "transcript.yaml")
    ap.add_argument("--outline", type=Path, default=DEFAULT_OUTLINE)
    ap.add_argument("--pptx", type=Path, default=DEFAULT_PPTX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--text-out", type=Path, default=None)
    a = ap.parse_args()
    data = yaml.safe_load(a.yaml.read_text(encoding="utf-8"))
    outline = outline_titles(a.outline)
    titles = dict(outline)
    for n, t in (data.get("title_overrides") or {}).items():
        titles[int(n)] = t
    missing = [n for n in range(1, N_SLIDES + 1) if n not in titles]
    if missing:
        sys.exit(f"OUTLINE thiếu tiêu đề slide {missing}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    build(data, titles, a.out)
    print(f"wrote {a.out}")
    sys.exit(0 if check(data, titles, outline, a.pptx, a.out, a.text_out) else 1)


if __name__ == "__main__":
    main()
