"""Regenerate OUTLINE.md from the shipped pptx (title + notes per slide)."""
import sys
from pptx import Presentation
SRC, DST = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
L = ["# OUTLINE slide v4", "",
     "Sinh tự động từ `XLA_UAV_Nhom10.pptx` đã nộp (tiêu đề + ghi chú từng slide). %d slide." % len(prs.slides),
     "Cột Ý chính lấy nguyên ghi chú; dòng Nguồn chỉ file kết quả trong XLA.", "",
     "| # | Badge | Tiêu đề (đúng chữ trên slide) | Ý chính, số, công thức |", "|---|---|---|---|"]
for i, s in enumerate(prs.slides, 1):
    sh = {x.shape_id: x for x in s.shapes}
    title = sh[5].text_frame.text.replace("\n", " ") if 5 in sh and sh[5].has_text_frame else ""
    badge = next((x.text_frame.text for x in s.shapes if x.has_text_frame and "·" in x.text_frame.text
                  and x.text_frame.text.isupper() and len(x.text_frame.text) < 40), "")
    ps = [p.text.strip() for p in s.notes_slide.notes_text_frame.paragraphs if p.text.strip()]
    ps = [p for p in ps if not p.startswith("Tiêu đề:")]
    if i == 1:
        title = "Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660 (bìa)"
    cell = "<br>".join(p.replace("|", "\|") for p in ps)
    L.append(f"| {i} | {badge} | {title} | {cell} |")
open(DST, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("rows", len(prs.slides))
