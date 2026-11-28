import sys
from pptx import Presentation
prs = Presentation(sys.argv[1])
for i, sl in enumerate(prs.slides, 1):
    out = []
    for sh in sl.shapes:
        if sh.has_text_frame:
            t = sh.text_frame.text.replace("\n", " / ")
            if t.strip(): out.append(t)
        if sh.has_table:
            for r in sh.table.rows:
                out.append(" | ".join(c.text for c in r.cells))
        if sh.has_chart:
            ch = sh.chart
            try:
                cats = list(ch.plots[0].categories)
                out.append("CHART cats: " + "; ".join(map(str, cats)))
            except Exception: pass
    print(f"=== {i}"); print("\n".join(out))
    n = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
    print("NOTES:", n.replace("\n", " // "))
