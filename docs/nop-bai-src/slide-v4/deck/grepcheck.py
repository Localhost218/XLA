import sys
from pptx import Presentation
prs = Presentation(sys.argv[1])
def paras(shapes):
    for sh in shapes:
        if sh.shape_type == 6: yield from paras(sh.shapes)
        elif sh.has_text_frame:
            for p in sh.text_frame.paragraphs: yield p.text
        if getattr(sh, 'has_table', False) and sh.has_table:
            for row in sh.table.rows:
                for c in row.cells: yield c.text
        if getattr(sh, 'has_chart', False) and sh.has_chart:
            from lxml import etree
            yield ' '.join(t for t in sh.chart.part._element.itertext())
body, notes = [], []
for s in prs.slides:
    body += list(paras(s.shapes)); notes.append(s.notes_slide.notes_text_frame.text)
terms = ["—", "không chỉ", "đóng vai trò", "toàn diện", "đột phá", "vượt trội", "mạnh mẽ", "tối ưu", "giải pháp hiệu quả", "mang lại", "liền mạch", "OV2640", "ESP32-CAM", "ESP32 CAM", "YOLO11n", "bản trước", "bản cũ", "ghi nhầm", "archive", "v2-nhom", "bản làm lại", "%"]
for t in terms:
    b = sum(x.lower().count(t.lower()) for x in body); n = sum(x.lower().count(t.lower()) for x in notes)
    print(f"{t!r}: body={b} notes={n}")
print("core title:", prs.core_properties.title, "| slides:", len(prs.slides))
import re
for i, s in enumerate(prs.slides, 1):
    words = sum(len(re.findall(r'\w+', x)) for x in paras(s.shapes) if not re.match(r'^\d\d / \d\d$', x))
    print(i, words, end='; ')
