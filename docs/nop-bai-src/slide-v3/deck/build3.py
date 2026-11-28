"""v7 (25 slides) -> v8: outline v3 order, <= 30 slides, stage diagrams, LaTeX formulas, report figures.
Usage: py -3.13 -I build3.py v7.pptx v8.pptx <deckdir>"""
import copy
import re
import sys

from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn

sys.path.insert(0, sys.argv[3])
import common as C  # noqa: E402
import kit  # noqa: E402
import slides1 as A  # noqa: E402
import slides1b as A2  # noqa: E402
import slides2 as B  # noqa: E402

SRC, DST = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
C.init(prs)
S = C.S

order = [S[0], A.s_problem(), A.s_system(), A.s_theory1(), A.s_theory2(), A.s_data(), A.s_d2(), A.s_realism(),
         A2.s_yolo(), A2.s_metrics(), A2.s_training(), A2.s_train_results(), B.s_a1()]
order.append(B.s_keep_header(S[12], "orange", "09 · SO SÁNH NHẬN DIỆN", notes=(
    ["Mỗi chấm là một bộ nhận diện trên 198 khung test; góc trên trái là tốt (mAP cao, ít báo nhầm).",
     "E1 D2: 0,236 và 0,302 hộp mỗi khung; D1: 0,143 và 0,994; D0 mAP 0,297 nhưng báo nhầm 1,195."],
    "results/reproduce/tables/t1_nhan_dien.md.")))
order.append(B.s_keep_header(S[13], "orange", "09 · SO SÁNH NHẬN DIỆN", notes=(
    ["Hai khung test có nhãn tay; mỗi cột vẽ hộp ở đúng ngưỡng vận hành của bộ đó (mốc 0,068, E1 D2 0,089, bundle 0,077).",
     "Chỉ để minh hoạ; kết luận dựa trên số của 198 khung."],
    "results/baseline/pred/edge16.csv; results/e1/preds/d2_test.csv; results/e2/preds/onnx_detector-v1.0.0_all.csv.")))
order += [B.s_classic(), B.s_dehaze(), B.s_nafnet()]
order.append(B.s_keep_header(S[14], "purple", "12 · ĐÁNH GIÁ ẢNH RÕ", notes=(
    ["Cùng một khung: gốc, sau cổ điển, sau NAFNet-w16; hàng dưới phóng 2× góc trên trái.",
     "Trên 198 khung test: cổ điển giảm điểm hồng 89 %, dark channel 52 %; mạng 99,9 % và 37 %."],
    "results/enhance/enhance_eval.json; t3_tieu_chi_anh_ro.md.")))
order += [B.s_c2(), B.s_c3()]
order.append(B.s_keep_header(S[17], "purple", "13 · E4", notes=(
    ["Hai bộ nhận diện chạy trên ba bản của cùng 198 khung test, so mAP50 ghép cặp với khung gốc.",
     "Bộ đã tinh chỉnh học trên ảnh xấu kiểu D2 nên ảnh đã làm rõ là ảnh lạ: −0,085 (mạng), −0,112 (cổ điển), CI dưới 0.",
     "Luật chốt trước: chỉ chuyển sang ảnh rõ khi CI của hiệu nằm trọn trên 0."],
    "results/reproduce/tables/t5_e4.md; results/e4/e4.json.")))
order.append(B.s_contract())
order.append(B.s_keep_header(S[18], "end", "14 · TÍCH HỢP IOT-CV", notes=(
    ["ORT CUDA trong backend IOT-CV, phát lại JPEG của chuyến 02/11 đúng nhịp, bỏ 0,3 % khung.",
     "Nhịp thật có người xem Ảnh rõ: tổng 37,5 / 54,1 ms p50 / p95, vượt ngân sách 50 ms; hết tốc lực 44,9 ms, đạt.",
     "Lý do: GPU rảnh khoảng 100 ms giữa hai khung nên hạ xung (1 890 so với 3 105 MHz)."],
    "results/reproduce/tables/t6_iotcv_ms.md.")))
order.append(B.s_keep_header(S[20], "end", "15 · QUYẾT ĐỊNH", notes=(
    ["Bảng gom các phương án: chỉ bundle detector-v1.0.0 được ship và bật; NAFNet có trong IOT-CV nhưng tắt mặc định.",
     "Bộ lọc cổ điển thắng A/B nhưng chưa nằm trong hợp đồng ONNX."],
    "t1_nhan_dien.md; t3_tieu_chi_anh_ro.md; t4_ab.md; t6_iotcv_ms.md.")))
order += [B.s_discussion(), B.s_conclusion(), B.s_refs()]
assert len(order) <= 30, len(order)
assert len({id(x) for x in order}) == len(order)

# ---------------------------------------------------------------- reorder and drop unused v7 slides
lst = prs.slides._sldIdLst
by_part = {prs.part.related_part(e.rId): e for e in lst}
keep = [by_part[sl.part] for sl in order]
for e in list(lst):
    lst.remove(e)
    if e not in keep:
        prs.part.drop_rel(e.rId)
for e in keep:
    lst.append(e)

# ---------------------------------------------------------------- page numbers
n = len(prs.slides)
pat = re.compile(r"^\d{2} / \d{2}$")
for i, sl in enumerate(prs.slides, 1):
    for sh in sl.shapes:
        if sh.has_text_frame and pat.match(sh.text_frame.text):
            sh.text_frame.paragraphs[0].runs[0].text = f"{i:02d} / {n}"

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


total = 0
for sl in prs.slides:
    total += split_runs(sl._element)
    for rel in sl.part.rels.values():
        if rel.reltype.endswith("/chart"):
            total += split_runs(rel.target_part._element)
prs.core_properties.title = "Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660"
prs.save(DST)
print("slides", n, "glyph splits", total)
