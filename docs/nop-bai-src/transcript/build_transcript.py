"""Dựng kịch bản thuyết trình .docx (bản đầy đủ, không giới hạn thời gian) từ transcript.yaml, rồi kiểm.

Chạy:  py -3.13 build_transcript.py [--pptx <deck>] [--out <docx>] [--yaml <yaml>]
Cần python-docx, python-pptx, PyYAML. Chỉ đọc pptx, không sửa.

transcript.yaml:
  sections: [{section, name, text, nhan}]  -- khoá theo số mục của dan-y-slide-v3.md;
            text: chuỗi = lời nói, {f: "..."} = dòng công thức (đặt ngay trước đoạn đọc nó)
  slides:   [{number, title, sections: [số mục]}]     -- thêm khi có deck cuối; thiếu = bản nháp theo mục

Kiểm: pptx ≤ 30 slide; có ``slides`` thì số slide, thứ tự, tiêu đề (đúng chữ trên slide) và việc mỗi mục được gán đúng
một lần; đủ mục 1–20; cụm từ cấm; mọi con số trong lời nói và công thức có trong nguồn (deck, bảng t1–t6, báo cáo,
LaTeX, configs, code, meta.json, cấu hình YOLO11 của ultralytics).
Mã thoát 0 = dựng xong, mọi kiểm đạt; 1 = có kiểm không đạt.
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
DEFAULT_PPTX = XLA / "docs/nop-bai/XLA_UAV_Nhom10.pptx"
DEFAULT_OUT = XLA / "docs/nop-bai/TranscriptThuyetTrinh_Nhom10.docx"
ULTRA = Path(r"D:\DevCache\venvs\xla\Lib\site-packages\ultralytics")
FONT = "Be Vietnam Pro"
MATH_FONT = "Cambria Math"
BODY_PT, HEAD_PT, SUB_PT, NOTE_PT, MATH_PT = 13, 14, 13, 11, 12.5
MAX_SLIDES = 30
N_SECTIONS = 20
BANNED = ["—", "không chỉ", "đóng vai trò", "Tóm lại", "Nhìn chung", "Bên cạnh đó", "Ngoài ra", "Hơn nữa",
          "Như vậy", "OV2640", "ESP32-CAM", "YOLO11n", "bản cũ", "đột phá", "toàn diện", "vượt trội", "mạnh mẽ",
          "tối ưu", "mang lại", "giải pháp hiệu quả"]
SOURCE_GLOBS = [
    "results/reproduce/tables/*.md", "BAO_CAO_DO_AN.md", "docs/nop-bai-src/latex/sections/*.tex",
    "docs/nop-bai-src/latex/fig/*.tex", "configs/*.yaml", "xla/**/*.py", "models/release/*/meta.json",
    "results/**/README.md",
]
ULTRA_FILES = ["cfg/models/11/yolo11.yaml", "cfg/default.yaml", "utils/loss.py", "utils/tal.py"]
NUM = re.compile(r"\d+(?:,\d+)?")


def set_font(run, size, bold=False, italic=False, color=None, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def para(doc, text, size, *, bold=False, italic=False, color=None, before=0, after=4, keep=False, indent=0.0,
         font=FONT, spacing=1.1):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing
    pf.keep_with_next = keep
    if indent:
        pf.left_indent = Cm(indent)
    set_font(p.add_run(text), size, bold, italic, color, font)
    return p


def section_body(doc, sec, with_subhead):
    if with_subhead:
        para(doc, f"Mục {sec['section']} · {sec['name']}", SUB_PT, bold=True, color="3A5F8F", before=6, after=3,
             keep=True)
    for t in sec["text"]:
        if isinstance(t, dict):  # {f: ...}: dòng công thức, đặt ngay trước đoạn đọc nó
            para(doc, t["f"], MATH_PT, color="222222", indent=0.8, before=2, after=4, font=MATH_FONT, keep=True)
        else:
            para(doc, t, BODY_PT, after=4)
    para(doc, f"Nhấn: {sec['nhan']}", NOTE_PT, italic=True, color="666666", after=4)


def build(data, out):
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.8)
    s.top_margin = s.bottom_margin = Cm(1.6)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    para(doc, data["project"], 15, bold=True, after=2)
    para(doc, data["subtitle"], BODY_PT, color="555555", after=10)
    by_no = {x["section"]: x for x in data["sections"]}
    if data.get("slides"):
        for sl in data["slides"]:
            para(doc, f"Slide {sl['number']} · {sl['title']}", HEAD_PT, bold=True, color="1F3A5F", before=12,
                 after=4, keep=True)
            many = len(sl["sections"]) > 1
            for k in sl["sections"]:
                section_body(doc, by_no[k], many)
    else:
        for sec in data["sections"]:
            para(doc, f"Mục {sec['section']} · {sec['name']}", HEAD_PT, bold=True, color="1F3A5F", before=12,
                 after=4, keep=True)
            section_body(doc, sec, False)
    doc.save(out)


def deck_text(pptx_path):
    from pptx import Presentation

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == 6:
                yield from walk(sh.shapes)
            else:
                yield sh

    deck = Presentation(str(pptx_path))
    per_slide, everything = [], []
    for sl in deck.slides:
        texts = []
        for sh in walk(sl.shapes):
            if sh.has_text_frame:
                texts.append(re.sub(r"\s+", " ", sh.text_frame.text).strip())
            if getattr(sh, "has_table", False) and sh.has_table:
                texts += [c.text for r in sh.table.rows for c in r.cells]
        notes = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
        per_slide.append(texts)
        everything += texts + [notes]
    return per_slide, "\n".join(everything)


def source_corpus(deck_all):
    parts = [deck_all]
    for g in SOURCE_GLOBS:
        for f in glob.glob(str(XLA / g), recursive=True):
            parts.append(Path(f).read_text(encoding="utf-8", errors="replace"))
    for f in ULTRA_FILES:
        p = ULTRA / f
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    s = "\n".join(parts).replace("{,}", ",")  # LaTeX viết dấu phẩy thập phân là {,}
    return re.sub(r"(?<=\d)(?:\\,|\s|\u00a0|\u202f)(?=\d{3}\b)", "", s)


def check(data, pptx_path, out):
    ok = True
    per_slide, deck_all = deck_text(pptx_path)
    n_deck = len(per_slide)
    print(f"pptx slides: {n_deck} (max {MAX_SLIDES})")
    if n_deck > MAX_SLIDES:
        print(f"FAIL pptx has {n_deck} slides > {MAX_SLIDES}")
        ok = False

    secs = [x["section"] for x in data["sections"]]
    if secs != list(range(1, N_SECTIONS + 1)):
        print(f"FAIL sections must be 1..{N_SECTIONS} in order, got {secs}")
        ok = False

    slides = data.get("slides")
    if slides:
        if len(slides) != n_deck:
            print(f"FAIL slide count: yaml {len(slides)} vs pptx {n_deck}")
            ok = False
        mapped = [k for sl in slides for k in sl["sections"]]
        if sorted(mapped) != list(range(1, N_SECTIONS + 1)):
            print(f"FAIL each section 1..{N_SECTIONS} must map to exactly one slide, got {sorted(mapped)}")
            ok = False
        if mapped != sorted(mapped):
            print("FAIL sections out of order across slides")
            ok = False
        for i, (sl, texts) in enumerate(zip(slides, per_slide), 1):
            if sl["number"] != i:
                print(f"FAIL numbering at {i}: {sl['number']}")
                ok = False
            if sl["title"] not in texts:
                print(f"FAIL slide {i} title not on slide: {sl['title']!r}")
                ok = False
    else:
        print("DRAFT: no slide mapping yet (headings by outline section)")

    doc_text = "\n".join(p.text for p in Document(out).paragraphs)
    for b in BANNED:
        n = doc_text.lower().count(b.lower())
        if n:
            print(f"FAIL banned {b!r}: {n}")
            ok = False
    print("banned phrases: " + ("none" if ok else "see above"))

    spoken = [t for x in data["sections"] for t in x["text"] if isinstance(t, str)]
    n_f = sum(isinstance(t, dict) for x in data["sections"] for t in x["text"])
    print(f"spoken words: {len(' '.join(spoken).split())}; formula lines: {n_f}")

    corpus = source_corpus(deck_all)
    missing = []
    for x in data["sections"]:
        blob = " ".join([t if isinstance(t, str) else t["f"] for t in x["text"]] + [x["nhan"]])
        for n in sorted(set(NUM.findall(blob))):
            if n not in corpus and n.replace(",", ".") not in corpus:
                missing.append((x["section"], n))
    if missing:
        print(f"FAIL numbers not found in any source: {missing}")
        ok = False
    else:
        print("numbers: all found in sources")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pptx", type=Path, default=DEFAULT_PPTX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--yaml", type=Path, default=HERE / "transcript.yaml")
    a = ap.parse_args()
    data = yaml.safe_load(a.yaml.read_text(encoding="utf-8"))
    build(data, a.out)
    print(f"wrote {a.out}")
    sys.exit(0 if check(data, a.pptx, a.out) else 1)


if __name__ == "__main__":
    main()
