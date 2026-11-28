"""v5 (22 slides) -> comparison deck (25 slides). Numbers come from D:/XLA/results/reproduce/tables/t1..t6 only."""
import copy
import re
import sys

from pptx import Presentation
from pptx.enum.chart import XL_LABEL_POSITION as LP
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn

sys.path.insert(0, sys.argv[3])
import kit  # noqa: E402
from kit import (BLUE, GOLD, INK, MUTED, ORANGE, PINK, PURPLE, SLATE, TEAL, TEXT, card, chip, line, picture,  # noqa: E402
                 text, vn)

SRC, DST, HERE = sys.argv[1], sys.argv[2], sys.argv[3]
IMG = HERE + "/img/"
prs = Presentation(SRC)
S = list(prs.slides)  # v5 order, 0-based
kit.init_card(S[18], 7)  # card shape of the "limits" slide


def header(slide, badge=None, title=None):
    sh = {s.shape_id: s for s in slide.shapes}
    if badge:
        sh[4].text_frame.paragraphs[0].runs[0].text = badge
    if title:
        sh[5].text_frame.paragraphs[0].runs[0].text = title


def clear(slide):
    for s in list(slide.shapes):
        if s.shape_id not in kit.HEADER_IDS:
            s._element.getparent().remove(s._element)
    xml = slide._element.xml
    for rid, rel in list(slide.part.rels.items()):
        if rel.reltype in (RT.CHART, RT.IMAGE) and f'"{rid}"' not in xml:
            slide.part.drop_rel(rid)


def clone(src):
    new = prs.slides.add_slide(src.slide_layout)
    for s in list(new.shapes):
        s._element.getparent().remove(s._element)
    bg = src._element.cSld.find(qn("p:bg"))
    nbg = copy.deepcopy(bg)
    blip = nbg.find(".//" + qn("a:blip"))
    part = src.part.related_part(blip.get(qn("r:embed")))
    blip.set(qn("r:embed"), new.part.relate_to(part, RT.IMAGE))
    new._element.cSld.insert(0, nbg)
    for s in src.shapes:
        if s.shape_id in kit.HEADER_IDS:
            new.shapes._spTree.append(copy.deepcopy(s._element))
    ns = new.notes_slide
    for e in list(ns.shapes):
        e._element.getparent().remove(e._element)
    for e in src.notes_slide.shapes:
        ns.shapes._spTree.append(copy.deepcopy(e._element))
    return new


def notes(slide, body, source):
    tf = slide.notes_slide.notes_text_frame
    tf.text = body
    p = tf.add_paragraph(); p = tf.add_paragraph(); p.text = "Nguồn: " + source


# ---------------------------------------------------------------- formula images -> text (decimal comma)
FORMULAS = {8: (22, "e = max(0, (R + B)/2 − G),        δ = 0,56·α·e", 30),
            9: (34, "R′ = R − δ,     B′ = B − δ,     G′ = G + 0,704·δ", 26),
            10: (17, "t(x) = max(0,4; 1 − ω·dc(x)/A),     J = (I − A)/t(x) + A", 22)}
for idx, (sid, ftxt, fsz) in FORMULAS.items():
    pic = next(x for x in S[idx].shapes if x.shape_id == sid)
    E_ = 914400
    text(S[idx], pic.left / E_, pic.top / E_, pic.width / E_, pic.height / E_, ftxt, fsz, INK, align="c", anchor="m")
    pic._element.getparent().remove(pic._element)

# ---------------------------------------------------------------- B. D1 vs D2 (v5 slide 5)
s = S[4]; clear(s)
header(s, "02 · D1 VÀ D2", "D2 giống camera hơn D1 và báo nhầm ít hơn 3 lần")
card(s, 0.62, 1.85, 6.35, 4.75); card(s, 7.17, 1.85, 5.55, 4.75)
text(s, 0.92, 2.08, 5.8, 0.4, "Độ giống thật, đo trên patch ảnh", 20, BLUE, True)
kit.table(s, 0.92, 2.7, [2.1, 1.15, 1.15, 1.4], 0.55, [
    ["Thước đo", "D1", "D2", "Thật với thật"],
    ["Lệch thống kê ↓", "5,928", ("0,796", {"color": TEAL, "bold": True}), "0,330"],
    ["C2ST (0,5 là tốt)", "0,936", "0,950", "0,542"],
    ["KID ×100 ↓", "4,443", "4,409", "0,371"],
], aligns="lrrr")
text(s, 0.92, 5.15, 5.8, 1.2, "D2 thắng ở lệch thống kê. C2ST và KID của hai bộ gần như bằng nhau, cả hai còn xa mức thật.", 14, TEXT)
text(s, 7.47, 2.08, 5.0, 0.4, "Nhận diện trên 198 khung test", 20, TEAL, True)
kit.table(s, 7.47, 2.7, [2.0, 1.5, 1.5], 0.55, [
    ["Kết quả", "D1", "D2"],
    ["mAP50", "0,143", "0,236"],
    ["CI 95 %", "[0,083; 0,185]", "[0,069; 0,370]"],
    ["Báo nhầm/khung ↓", "0,994", ("0,302", {"color": TEAL, "bold": True})],
], aligns="lrr")
text(s, 7.47, 5.15, 4.95, 1.2, "Báo nhầm: chênh −0,692, CI [−1,238; −0,246], D2 thắng rõ. mAP50: CI chênh còn chứa 0.", 14, TEXT)
notes(s, "Bên trái là độ giống thật của ảnh làm xấu so với khung dev thật. Lệch thống kê: D1 5,928, D2 0,796, hai tập thật so với nhau 0,330. C2ST là độ chính xác của bộ phân loại tách patch thật khỏi patch làm xấu, 0,5 nghĩa là không tách được: D1 0,936, D2 0,950. KID nhân 100: D1 4,443, D2 4,409. Bên phải là tác dụng lên bộ nhận diện YOLO11s tinh chỉnh trên ảnh của từng bộ. mAP50: D1 0,143, D2 0,236, chênh 0,093 nhưng CI [−0,073; +0,252] còn chứa 0. Báo nhầm trên khung bay không người: D1 0,994, D2 0,302 hộp mỗi khung. Nhóm nghĩ C2ST cao vì cảnh phố VisDrone khác bãi bay; giả thuyết này chưa kiểm riêng.",
      "results/reproduce/tables/t2_do_giong_that.md; t1_nhan_dien.md; BAO_CAO_DO_AN.md mục 5.")

# ---------------------------------------------------------------- A1. mAP50 + CI, 9 rows (v5 slide 12)
T1 = [  # name, mAP50, lo, hi, FP, color
    ("Mốc YOLO11s COCO @1280", 0.126, 0.030, 0.224, 0.597, SLATE),
    ("E1 D0 ảnh sạch", 0.297, 0.036, 0.408, 1.195, GOLD),
    ("E1 D1 ngẫu nhiên", 0.143, 0.083, 0.185, 0.994, BLUE),
    ("E1 D2 bản sao camera", 0.236, 0.069, 0.370, 0.302, TEAL),
    ("E1 D2 + siêu phân giải", 0.149, 0.011, 0.282, 1.956, PURPLE),
    ("E2 thầy YOLO11m", 0.218, 0.107, 0.349, 0.390, MUTED),
    ("E2 trò vòng 1", 0.156, 0.036, 0.235, 0.868, ORANGE),
    ("E2 trò vòng 2", 0.313, 0.035, 0.472, 0.522, ORANGE),
    ("Bundle detector-v1.0.0", 0.120, 0.032, 0.197, 0.881, PINK),
]
s = S[11]; clear(s)
header(s, "03 · NHẬN DIỆN", "Vòng 2 cao nhất (0,313), nhưng CI của 9 bộ chồng lên nhau")
card(s, 0.62, 1.85, 8.35, 5.1)
text(s, 0.9, 2.02, 7.8, 0.3, "MAP50 · 198 KHUNG TEST THẬT · THANH ĐEN LÀ CI 95 %", 12, PURPLE, True)
FX, FY, FW, FH = 0.8, 2.35, 8.0, 4.15
LAY = (0.34, 0.02, 0.62, 0.86)
gf = kit.bar_chart(s, FX, FY, FW, FH, [r[0] for r in T1],
                   [("mAP50", [r[1] for r in T1], SLATE, [vn(r[1]) for r in T1], {i: r[5] for i, r in enumerate(T1)})],
                   horizontal=True, gap=45, vmin=0, vmax=0.5, major=0.1, grid=True, layout=LAY)
ser = gf.chart.plots[0].series[0]
kit._err_bars(ser, [r[3] - r[1] for r in T1], [r[1] - r[2] for r in T1])
for p_ in range(len(T1)):
    ser.points[p_].data_label.position = LP.INSIDE_BASE
    for r in ser.points[p_].data_label.text_frame.paragraphs[0].runs:
        r.font.color.rgb = kit.RGBColor.from_string("FFFFFF")
px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
for k in range(6):
    v = k / 10
    text(s, px + v / 0.5 * pw - 0.3, py + ph + 0.05, 0.6, 0.25, vn(v, 1) if k else "0", 11, MUTED, align="c")
card(s, 9.17, 1.85, 3.55, 5.1)
text(s, 9.45, 2.08, 3.0, 0.4, "Vì sao ship vòng 1", 20, PINK, True)
text(s, 9.45, 2.65, 3.0, 4.0, ["Luật chốt trước: chọn theo dev-val 100 khung.",
                               "Dev-val cho vòng 1 0,379, vòng 2 chỉ 0,296.",
                               "Bundle là vòng 1 xuất ONNX, đo lại trên đường ONNX được 0,120."], 14, TEXT, spacing=1.15)
notes(s, "Chín dòng của bảng nhận diện, cùng 198 khung test thật, CI 95 % bằng bootstrap theo khối thời gian, 2000 mẫu. Mốc là YOLO11s COCO ở khung 1280, bỏ hộp ở 16 px mép trái. E1 tinh chỉnh YOLO11s trên VisDrone qua từng bộ làm xấu. E2 dùng YOLO11m làm thầy gán nhãn giả cho khung bay chưa nhãn; trò vòng 2 cao nhất trên test (0,313) nhưng CI [0,035; 0,472] rất rộng và trùm lên gần hết các bộ khác. Theo luật đã chốt, nhóm chọn theo dev-val: vòng 1 được 0,379, vòng 2 được 0,296, nên ship vòng 1. Bundle ONNX chấm bằng đúng đường ONNX đạt 0,120 [0,032; 0,197].",
      "results/reproduce/tables/t1_nhan_dien.md; results/e2/README.md.")

# ---------------------------------------------------------------- A2. scatter mAP50 vs FP (v5 slide 13)
s = S[12]; clear(s)
header(s, "03 · NHẬN DIỆN", "D2 cho ít báo nhầm nhất: 0,30 hộp/khung, D1 là 0,99")
card(s, 0.62, 1.85, 12.1, 5.1)
FX, FY, FW, FH = 0.85, 2.3, 8.75, 4.3
LAY = (0.09, 0.02, 0.88, 0.82)
short = ["Mốc", "D0", "D1", "D2", "D2+SR", "Thầy", "Vòng 1", "Vòng 2", "Bundle"]
pos = [LP.RIGHT, LP.RIGHT, LP.RIGHT, LP.LEFT, LP.LEFT, LP.RIGHT, LP.LEFT, LP.RIGHT, LP.BELOW]
kit.scatter(s, FX, FY, FW, FH, [(short[i], r[4], r[1], r[5], pos[i]) for i, r in enumerate(T1)],
            (0, 2.0, 0.5), (0, 0.4, 0.1), LAY)
px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
for k in range(5):
    v = k * 0.5
    text(s, px + v / 2.0 * pw - 0.3, py + ph + 0.05, 0.6, 0.25, vn(v, 1).replace(",0", "") if v else "0", 11, MUTED, align="c")
for k in range(5):
    v = k * 0.1
    text(s, px - 0.62, py + ph - v / 0.4 * ph - 0.12, 0.55, 0.25, vn(v, 1) if k else "0", 11, MUTED, align="r")
text(s, px, py + ph + 0.32, pw, 0.3, "Báo nhầm, hộp mỗi khung bay không người →  ít hơn là tốt", 12, TEXT, align="c")
text(s, 0.9, 2.0, 4.0, 0.3, "MAP50 ↑", 12, PURPLE, True)
text(s, px + 0.12, py + 0.08, 4.0, 0.3, "↖ góc tốt: mAP cao, ít báo nhầm", 12, TEAL, True)
text(s, 9.85, 2.1, 2.7, 4.6, [
    [("Mốc", {"bold": True, "color": SLATE}), (" YOLO11s COCO @1280, chưa tinh chỉnh", {})],
    [("E1", {"bold": True, "color": TEAL}), (" tinh chỉnh trên D0, D1, D2, D2+SR", {})],
    [("E2", {"bold": True, "color": ORANGE}), (" thầy YOLO11m, trò vòng 1 và 2", {})],
    [("Bundle", {"bold": True, "color": PINK}), (" là trò vòng 1 xuất ONNX", {})],
    [("D0 có mAP cao nhưng báo nhầm 1,20 hộp mỗi khung.", {"color": MUTED})],
], 14, TEXT, spacing=1.2)
notes(s, "Mỗi chấm là một bộ nhận diện trên cùng 198 khung test. Trục dọc là mAP50, trục ngang là số hộp báo nhầm trung bình trên mỗi khung bay không có người, nên góc trên bên trái là tốt nhất. D2 nằm gần góc đó nhất: 0,236 mAP50 và 0,302 hộp sai mỗi khung, trong khi D1 là 0,143 và 0,994. Thầy YOLO11m đứng gần D2 (0,218 và 0,390). D0 ảnh sạch có mAP cao (0,297) nhưng báo nhầm 1,195; thêm siêu phân giải vào D2 làm báo nhầm tăng lên 1,956. Ngưỡng của từng model chọn bằng F1 lớn nhất trên dev-val.",
      "results/reproduce/tables/t1_nhan_dien.md.")

# ---------------------------------------------------------------- A3. same-frame boxes (new, after A2)
s = clone(S[11])
header(s, "03 · NHẬN DIỆN", "Khung có 6 người: mốc vẽ 8 hộp, D2 vẽ 4, bundle vẽ 8")
cols = [("Nhãn tay", "gt", kit.RGBColor and "16A34A"), ("Mốc YOLO11s COCO", "base", SLATE),
        ("E1 D2", "d2", TEAL), ("Bundle detector-v1.0.0", "bundle", PURPLE)]
counts = {"20261102-100941_007_01031": {"gt": "6 người", "base": "8 hộp", "d2": "4 hộp", "bundle": "8 hộp"},
          "20261102-100941_001_01007": {"gt": "5 người", "base": "1 hộp", "d2": "7 hộp", "bundle": "4 hộp"}}
TW, TH, GAP = 2.88, 2.16, 0.19
for c, (lab, key, colr) in enumerate(cols):
    x = 0.62 + c * (TW + GAP)
    text(s, x, 1.85, TW, 0.3, lab, 14, colr, True)
    for r, fr in enumerate(counts):
        y = 2.22 + r * 2.52
        picture(s, f"{IMG}{fr}_{key}.png", x, y, TW, TH)
        text(s, x, y + TH + 0.03, TW, 0.25, counts[fr][key], 12, MUTED)
notes(s, "Hai khung thật trong tập test, chuyến 20261102-100941: khung 007_01031 có 6 người, khung 001_01007 có 5 người theo nhãn tay đã duyệt. Mỗi cột vẽ hộp của một bộ nhận diện ở đúng ngưỡng vận hành của nó (chọn bằng F1 trên dev-val): mốc 0,068, E1 D2 0,089, bundle 0,077. Ở khung trên, mốc và bundle vẽ 8 hộp, có hộp trùm cả bóng người; D2 vẽ 4 hộp. Ở khung dưới, mốc chỉ vẽ 1 hộp, D2 vẽ 7, bundle 4. Hai khung chỉ để minh hoạ, kết luận dựa vào số trên 198 khung.",
      "results/baseline/pred/edge16.csv; results/e1/preds/d2_test.csv; results/e2/preds/onnx_detector-v1.0.0_all.csv; results/labels/agent_reviewed_test/.")

# ---------------------------------------------------------------- C1. same frame raw / classic / net + zoom (v5 slide 14)
s = S[13]; clear(s)
header(s, "03 · LÀM RÕ", "Cùng khung: mạng xoá hồng sạch hơn, cổ điển khử mù tốt hơn")
TW, TH = 3.05, 2.23
heads = [("Khung gốc", "JPEG q12", "raw", SLATE), ("Cổ điển", "9,2 ms trên CPU", "classic", GOLD),
         ("NAFNet-w16", "20,7 ms trên GPU", "net", TEAL)]
for c, (h1, h2, key, colr) in enumerate(heads):
    x = 0.62 + c * (TW + 0.2)
    text(s, x, 1.82, TW, 0.32, [[(h1, {"bold": True, "color": colr, "size": 15}), ("  " + h2, {"color": MUTED, "size": 12})]])
    picture(s, f"{IMG}c1_{key}_full.png", x, 2.2, TW, TH)
    picture(s, f"{IMG}c1_{key}_zoom.png", x, 4.58, TW, TH)
card(s, 10.4, 1.85, 2.32, 4.96)
text(s, 10.6, 2.05, 1.95, 4.6, [
    [("Hàng trên", {"bold": True, "color": INK})], [("cả khung", {})], [("", {})],
    [("Hàng dưới", {"bold": True, "color": INK})], [("góc trên trái, phóng 2×", {})], [("", {})],
    [("Điểm hồng", {"bold": True, "color": INK})], [("−89 % và −99,9 %", {})], [("", {})],
    [("Dark channel", {"bold": True, "color": INK})], [("−52 % và −37 %", {})],
], 13, TEXT)
notes(s, "Cùng một khung bay, ba cột là khung gốc, sau bộ lọc cổ điển và sau mạng NAFNet bề rộng 16. Hàng dưới phóng 2 lần góc trên bên trái theo điểm ảnh gần nhất, giống cách trang A/B lần 2 phóng vùng ảnh. Trên 198 khung test, bộ cổ điển giảm điểm hồng 89 % và dark channel 52 %; mạng xoá 99,9 % điểm hồng nhưng dark channel chỉ giảm 37 %, màu ra nguội và nhạt. Cổ điển chạy 9,2 ms trên CPU, mạng 20,7 ms trên GPU.",
      "results/enhance/enhance_eval.json; t3_tieu_chi_anh_ro.md; BAO_CAO_DO_AN.md mục 4.3.")

# ---------------------------------------------------------------- C2. criteria + ms (new, after C1)
s = clone(S[13])
header(s, "03 · LÀM RÕ", "Mạng đạt 3/7 tiêu chí, cổ điển giảm mù 52 % so với 37 %")
tiles = [  # title, threshold label, vmax, thr, classic, net, labels
    ("Điểm hồng giảm", "ngưỡng 90 %", 100, 90, 89.2, 99.9, ("89 %", "99,9 %")),
    ("Dark channel giảm", "ngưỡng 50 %", 100, 50, 51.6, 36.8, ("52 %", "37 %")),
    ("Nhấp nháy giữa hai khung", "ngưỡng 1×, không tăng", 2.5, 1.0, 2.18, 1.81, ("2,18×", "1,81×")),
    ("Thời gian mỗi khung", "ngưỡng 30 ms", 35, 30, 9.2, 20.7, ("9,2 ms", "20,7 ms")),
]
LAY = (0.2, 0.06, 0.7, 0.88)
for i, (ttl, thl, vmax, thr, cv, nv, labs) in enumerate(tiles):
    x = 0.62 + (i % 2) * 6.15; y = 1.85 + (i // 2) * 2.6
    card(s, x, y, 5.95, 2.42)
    text(s, x + 0.28, y + 0.16, 3.6, 0.35, ttl, 17, INK, True)
    text(s, x + 3.6, y + 0.2, 2.1, 0.3, thl, 12, PINK, align="r")
    FX, FY, FW, FH = x + 0.15, y + 0.62, 5.65, 1.65
    kit.bar_chart(s, FX, FY, FW, FH, ["Cổ điển", "Mạng"],
                  [("v", [cv, nv], GOLD, list(labs), {1: TEAL}, LP.OUTSIDE_END)],
                  horizontal=True, gap=40, vmin=0, vmax=vmax, layout=LAY)
    px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
    xx = px + thr / vmax * pw
    line(s, xx, py - 0.02, xx, py + ph + 0.02, PINK, 1.5, dash=True)
notes(s, "Bảy tiêu chí đăng ký trước cho mạng. Mạng đạt ba: xoá 99,9 % điểm hồng, chạy 20,7 ms mỗi khung, và không làm giảm mAP ở nhánh nhận diện vì nhánh này luôn chạy trên khung gốc. Mạng trượt bốn: dark channel chỉ giảm 37 % (cần 50 %), nhấp nháy giữa hai khung liền nhau tăng 1,81 lần (cần không tăng), thắng ảnh gốc 11/25 lượt A/B (cần 70 %), thắng cổ điển 1/25 (cần 55 %). Bộ cổ điển cũng làm nhấp nháy tăng, 2,18 lần. Vạch hồng là ngưỡng của từng tiêu chí.",
      "results/reproduce/tables/t3_tieu_chi_anh_ro.md; results/enhance/enhance_eval.json (physical, flicker); BAO_CAO_DO_AN.md mục 4.3.")

# ---------------------------------------------------------------- D. E4 rebuilt (v5 slide 16)
s = S[15]; clear(s)
header(s, None, "Cho ảnh qua mạng trước, bộ đã tinh chỉnh mất 0,085 mAP50")
card(s, 0.62, 1.85, 8.0, 5.1)
text(s, 0.9, 2.02, 7.4, 0.3, "MAP50 · 198 KHUNG TEST · CÙNG BỘ NHẬN DIỆN, ĐỔI ẢNH ĐẦU VÀO", 12, PURPLE, True)
E4 = [("Khung gốc", [0.126, 0.119], SLATE), ("Sau cổ điển", [0.081, 0.007], GOLD), ("Sau mạng", [0.195, 0.035], TEAL)]
kit.bar_chart(s, 0.85, 2.4, 7.55, 4.4, ["YOLO11s COCO (có sẵn)", "detector-v1.0.0 (đã tinh chỉnh)"],
              [(n, v, c, [vn(a) for a in v], None, LP.OUTSIDE_END) for n, v, c in E4],
              gap=80, legend=True, vmin=0, vmax=0.22)
card(s, 8.82, 1.85, 3.9, 2.45)
text(s, 9.1, 2.05, 3.4, 0.7, "−0,085", 40, TEAL, True)
text(s, 9.1, 2.85, 3.4, 1.3, "Bộ đã tinh chỉnh, ảnh qua mạng. CI [−0,139; −0,011] nằm trọn dưới 0.", 14, TEXT)
card(s, 8.82, 4.5, 3.9, 2.45)
text(s, 9.1, 4.7, 3.4, 0.7, "+0,069", 40, MUTED, True)
text(s, 9.1, 5.5, 3.4, 1.3, "Bộ có sẵn, ảnh qua mạng. CI [−0,033; +0,231] còn chứa 0.", 14, TEXT)
notes(s, "Hai bộ nhận diện chạy trên ba bản của cùng 198 khung test, so mAP50 ghép cặp với khung gốc. Với YOLO11s COCO có sẵn, ảnh qua mạng tăng 0,069 nhưng CI [−0,033; +0,231] chứa 0; ảnh qua cổ điển giảm 0,046. Với bộ đã tinh chỉnh, ảnh qua mạng giảm 0,085, CI [−0,139; −0,011] nằm trọn dưới 0; ảnh qua cổ điển giảm 0,112. Bộ này học trên ảnh xấu kiểu D2 nên ảnh đã làm đẹp lại là ảnh lạ với nó. Luật chốt trước chỉ cho chuyển sang ảnh đã làm rõ khi CI nằm trọn trên 0, nên bộ nhận diện luôn chạy trên khung gốc.",
      "results/reproduce/tables/t5_e4.md; results/e4/e4.json.")

# ---------------------------------------------------------------- E. runtime vs budget (v5 slide 17)
s = S[16]; clear(s)
header(s, None, "Ở nhịp thật p95 tổng 54,1 ms, vượt ngân sách 50 ms")
card(s, 0.62, 1.85, 12.1, 4.75)
text(s, 0.9, 2.02, 8.0, 0.3, "MS MỖI KHUNG · ORT CUDA, RTX 4060 LAPTOP · PHÁT LẠI CHUYẾN 02/11", 12, PURPLE, True)
cats = ["Detector, nhịp thật", "Enhancer, nhịp thật", "Tổng, nhịp thật", "Tổng, chỉ detector", "Tổng, hết tốc lực"]
p50 = [23.6, 12.5, 37.5, 36.6, 42.5]; p95 = [33.4, 19.0, 54.1, 67.9, 44.9]
FX, FY, FW, FH = 0.85, 2.35, 11.65, 4.1
LAY = (0.2, 0.1, 0.76, 0.8)
kit.bar_chart(s, FX, FY, FW, FH, cats,
              [("p50", p50, "8CBFE0", [vn(v, 1) for v in p50], None, LP.OUTSIDE_END),
               ("p95", p95, BLUE, [vn(v, 1) for v in p95], None, LP.OUTSIDE_END)],
              horizontal=True, gap=55, legend=False, vmin=0, vmax=70, major=10, axis_labels=True, grid=True, layout=LAY)
text(s, 9.2, 2.02, 3.3, 0.3, [[("■ ", {"color": "8CBFE0"}), ("p50     ", {}), ("■ ", {"color": BLUE}), ("p95", {})]], 13, TEXT, align="r")
px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
for v, lab, col in ((50, "ngân sách tổng 50 ms", PINK), (30, "detector 30 ms", ORANGE)):
    xx = px + v / 70 * pw
    line(s, xx, py, xx, py + ph, col, 1.75, dash=True)
    text(s, xx + 0.06, py - 0.02, 2.2, 0.25, lab, 11, col, True)
text(s, 0.62, 6.7, 12.1, 0.3, "Giữa hai khung camera GPU rảnh khoảng 100 ms và hạ xung, nên chạy hết tốc lực lại đạt: p95 tổng 44,9 ms.", 13, MUTED)
notes(s, "Hai model chạy trong backend IOT-CV bằng ONNX Runtime CUDA trên JPEG ghi được của chuyến 02/11, phát lại đúng nhịp camera, bỏ 0,3 % khung. Khi có người xem Ảnh rõ: detector 23,6 / 33,4 ms p50 / p95, enhancer 12,5 / 19,0 ms, tổng 37,5 / 54,1 ms, 7,2 khung mỗi giây. Chỉ chạy detector thì tổng p95 còn lên 67,9 ms vì GPU rảnh lâu hơn. Chạy hết tốc lực được 23,6 khung/s, p95 tổng 44,9 ms và detector 28,1 ms, đều trong ngân sách. Ngân sách đặt trước là p95 tổng 50 ms và detector 30 ms. Nhóm chấp nhận và ghi vào giới hạn.",
      "results/reproduce/tables/t6_iotcv_ms.md; BAO_CAO_DO_AN.md mục 4.5.")

# ---------------------------------------------------------------- F. decision table (new, after contract slide)
s = clone(S[18])
header(s, "04 · QUYẾT ĐỊNH", "Chọn gì và vì sao")
rows = [["Phương án", "Chính xác", "Báo nhầm", "Tốc độ", "Quyết định", "Lý do"],
        ["Mốc YOLO11s COCO @1280", "0,126", "0,597", "–", "", "chỉ là mốc so sánh"],
        ["E1 D1, làm xấu ngẫu nhiên", "0,143", "0,994", "–", "", "báo nhầm gấp 3 lần D2"],
        ["E1 D2, bản sao camera", "0,236", "0,302", "–", "", "điểm xuất phát cho E2"],
        ["E2 trò vòng 2", "0,313", "0,522", "–", "", "dev-val thấp hơn vòng 1"],
        ["Bundle detector-v1.0.0", "0,120", "0,881", "23,6 ms", "", "dev-val chọn, đúng hợp đồng"],
        ["Bộ lọc cổ điển", "24/25 A/B", "–", "9,2 ms CPU", "", "đối chứng, nhấp nháy 2,18×"],
        ["NAFNet-w16", "11/25 với gốc", "–", "20,7 ms GPU", "", "thua cổ điển 24/25 lượt"]]
CW = [2.75, 1.5, 1.4, 1.5, 1.65, 3.3]
RH = 0.56
kit.table(s, 0.62, 1.95, CW, RH, rows, size=14, aligns="lrrrll")
chips = [("không ship", SLATE), ("không ship", SLATE), ("nền E2", BLUE), ("không ship", SLATE),
         ("ship", TEAL), ("không ship", SLATE), ("ship, tắt", ORANGE)]
cx = 0.62 + sum(CW[:4]) + 0.08
for i, (lab, col) in enumerate(chips):
    chip(s, cx, 1.95 + (i + 1) * RH + 0.13, 1.25, 0.3, lab, col, 11)
text(s, 0.62, 6.58, 12.1, 0.55, "Chính xác: mAP50 trên 198 khung test, hoặc lượt thắng A/B của ảnh rõ. Báo nhầm: hộp mỗi khung bay không người. Tốc độ chỉ đo cho model đã ship.", 12, MUTED)
notes(s, "Bảng gom các phương án đã thử. Mốc COCO chỉ để so sánh. D1 báo nhầm 0,994 hộp mỗi khung, gấp khoảng 3 lần D2. E1 D2 không ship trực tiếp nhưng là trọng số khởi đầu của trò E2. Vòng 2 có mAP50 cao nhất trên test nhưng dev-val thấp hơn vòng 1, và luật chốt trước là chọn theo dev-val, nên bundle detector-v1.0.0 là vòng 1; trong IOT-CV nó chạy 23,6 ms p50. Ở nhánh ảnh rõ, người chấm chọn bản cổ điển 24/25 lượt, nhưng bộ cổ điển cũng làm nhấp nháy tăng 2,18 lần và chỉ là đối chứng. NAFNet có trong IOT-CV dưới dạng enhancer-v1.0.0 nhưng không bật mặc định vì chỉ đạt 3/7 tiêu chí.",
      "t1_nhan_dien.md; t3_tieu_chi_anh_ro.md; t4_ab.md; t6_iotcv_ms.md; configs/selftrain_e2.yaml.")

# ---------------------------------------------------------------- order + numbering
lst = prs.slides._sldIdLst
ids = list(lst)
old = ids[:22]; new = ids[22:]  # A3, C2, F in creation order
order = old[:13] + [new[0]] + [old[13], new[1]] + old[14:18] + [new[2]] + old[18:]
for e in ids:
    lst.remove(e)
for e in order:
    lst.append(e)
n = len(prs.slides)
pat = re.compile(r"^\d{2} / \d{2}$")
for i, sl in enumerate(prs.slides, 1):
    for sh in sl.shapes:
        if sh.has_text_frame and pat.match(sh.text_frame.text):
            sh.text_frame.paragraphs[0].runs[0].text = f"{i:02d} / {n}"
prs.save(DST)
print("slides", n)
