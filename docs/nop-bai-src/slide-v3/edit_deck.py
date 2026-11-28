"""Sửa XLA_UAV_Nhom10.pptx cho khớp kết quả cuối (giữ template, nền, bố cục của nhóm)."""
import copy
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

SRC, DST, ASSETS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FIT = ASSETS / "fit"
FIT.mkdir(exist_ok=True)
EMU = 914400
TOTAL = 23
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

prs = Presentation(str(SRC))
S = {i + 1: s for i, s in enumerate(prs.slides)}  # số slide gốc → slide


# ----------------------------------------------------------------------------- helpers
def sh(slide, sid):
    for s in slide.shapes:
        if s.shape_id == sid:
            return s
    raise KeyError(f"shape {sid} not found")


def text(slide, sid, paras, size=None):
    s = sh(slide, sid)
    paras = [paras] if isinstance(paras, str) else paras
    tf = s.text_frame
    ps = list(tf.paragraphs)
    while len(ps) < len(paras):
        new = copy.deepcopy(ps[-1]._p)
        ps[-1]._p.addnext(new)
        ps = list(tf.paragraphs)
    for p, t in zip(ps, paras):
        runs = p.runs
        if not runs:
            r = p.add_run()
            runs = [r]
        runs[0].text = t
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
        if size:
            runs[0].font.size = Pt(size)
    for p in ps[len(paras):]:
        p._p.getparent().remove(p._p)


def delete(slide, *sids):
    for sid in sids:
        e = sh(slide, sid)._element
        e.getparent().remove(e)


def geom(slide, sid, x=None, y=None, w=None, h=None):
    s = sh(slide, sid)
    if x is not None: s.left = Inches(x)
    if y is not None: s.top = Inches(y)
    if w is not None: s.width = Inches(w)
    if h is not None: s.height = Inches(h)


def chip(slide, rect_id, text_id, label, header=True):
    text(slide, text_id, label)
    w = (0.49 + 0.108 * len(label)) if header else (0.35 + 0.105 * len(label))
    geom(slide, rect_id, w=w)
    geom(slide, text_id, w=w)


def header(slide, no, label, title):
    chip(slide, 3, 4, label)
    text(slide, 5, title)
    text(slide, 6, f"{no:02d} / {TOTAL}")


def picture(slide, sid, img, anchor="center"):
    s = sh(slide, sid)
    aspect = s.width / s.height
    im = Image.open(ASSETS / img).convert("RGB")
    w, h = im.size
    if w / h > aspect:  # cắt ngang
        nw = int(h * aspect)
        x0 = (w - nw) // 2
        im = im.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(w / aspect)
        y0 = 0 if anchor == "top" else (h - nh) // 2
        im = im.crop((0, y0, w, y0 + nh))
    out = FIT / f"{Path(img).stem}_{s.width}_{s.height}_{anchor}.jpg"
    im.save(out, quality=92)
    _, rid = slide.part.get_or_add_image_part(str(out))
    blip = s._element.find(".//" + qn("a:blip"))
    blip.set(qn("r:embed"), rid)
    src = s._element.find(".//" + qn("a:srcRect"))
    if src is not None:
        src.getparent().remove(src)


def picture_png(slide, sid, img):
    s = sh(slide, sid)
    _, rid = slide.part.get_or_add_image_part(str(ASSETS / img))
    s._element.find(".//" + qn("a:blip")).set(qn("r:embed"), rid)
    src = s._element.find(".//" + qn("a:srcRect"))
    if src is not None:
        src.getparent().remove(src)


def notes(slide, body):
    ns = slide.notes_slide
    if ns.notes_text_frame is None:  # slide mới: notes master không có body → chép khung ghi chú của slide 17
        src_tree = S[17].notes_slide.shapes._spTree
        tree = ns.shapes._spTree
        for e in list(tree)[2:]:
            tree.remove(e)
        for e in list(src_tree)[2:]:
            if e.find(".//" + qn("a:blip")) is None:
                tree.append(copy.deepcopy(e))
    ns.notes_text_frame.text = body


def clone_textbox(slide, sid, x, y, w, h, value, size=None):
    src = sh(slide, sid)._element
    new = copy.deepcopy(src)
    src.getparent().append(new)
    ids = [int(e.get("id")) for e in slide._element.iter(qn("p:cNvPr"))]
    new.find(".//" + qn("p:cNvPr")).set("id", str(max(ids) + 1))
    shape = slide.shapes[-1]
    shape.left, shape.top, shape.width, shape.height = Inches(x), Inches(y), Inches(w), Inches(h)
    text(slide, shape.shape_id, value, size)
    return shape


def set_chart(slide, sid, cats, series, fmt="0.000", vmax=None, colors=None, axfmt="0.0", major=None, labels=False):
    gf = sh(slide, sid)
    ch = gf.chart
    cd = CategoryChartData(number_format=fmt)
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    ch.replace_data(cd)
    for nf in ch._chartSpace.iter(qn("c:numFmt")):
        nf.set("formatCode", fmt)
    if vmax is not None:
        ch.value_axis.maximum_scale = vmax
        ch.value_axis.minimum_scale = 0
    va = ch.value_axis
    va.has_major_gridlines = True
    va.tick_labels.number_format = axfmt
    va.tick_labels.number_format_is_linked = False
    if major:
        va.major_unit = major
    if labels:
        pl = ch.plots[0]
        pl.has_data_labels = True
        dl = pl.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.show_value = True
        dl.font.size = Pt(14)
        from pptx.enum.chart import XL_LABEL_POSITION
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    if colors:
        pts = ch.plots[0].series[0].points
        for i, c in enumerate(colors):
            pts[i].format.fill.solid()
            pts[i].format.fill.fore_color.rgb = RGBColor.from_string(c)
    return ch


def clone_slide(src, chart_parts=None):
    """Slide mới, chép nền + mọi shape của `src`; quan hệ ảnh trỏ cùng part, chart trỏ part trong `chart_parts`."""
    new = prs.slides.add_slide(prs.slide_layouts[0])
    for e in list(new.shapes._spTree)[2:]:
        new.shapes._spTree.remove(e)
    rmap = {}
    chart_parts = list(chart_parts or [])

    def remap(el):
        for node in el.iter():
            for attr in list(node.attrib):
                if attr.startswith("{%s}" % R_NS):
                    old = node.get(attr)
                    if old not in rmap:
                        rel = src.part.rels[old]
                        if rel.reltype.endswith("/chart"):
                            rmap[old] = new.part.relate_to(chart_parts.pop(0), rel.reltype)
                        else:
                            rmap[old] = new.part.relate_to(rel.target_part, rel.reltype)
                    node.set(attr, rmap[old])

    bg = src._element.cSld.find(qn("p:bg"))
    if bg is not None:
        nbg = copy.deepcopy(bg)
        remap(nbg)
        new._element.cSld.insert(0, nbg)
    for e in list(src.shapes._spTree)[2:]:
        ne = copy.deepcopy(e)
        remap(ne)
        new.shapes._spTree.append(ne)
    return new


# ----------------------------------------------------------------------------- slide mới (cấu trúc trước)
def chart_part(slide, sid):
    rid = sh(slide, sid)._element.find(".//" + qn("c:chart")).get(qn("r:id"))
    return slide.part.related_part(rid)


chart1 = chart_part(S[14], 21)
chart2 = chart_part(S[17], 9)
chart1._element = copy.deepcopy(chart2._element)
E2 = clone_slide(S[17], [chart1])
LIM = clone_slide(S[25])
TEAM = clone_slide(S[25])

# ----------------------------------------------------------------------------- 1 bìa
s = S[1]
text(s, 4, "ĐỒ ÁN XỬ LÝ ẢNH  ·  PTIT  ·  NHÓM 10")
text(s, 5, ["Làm rõ ảnh UAV", "và tìm người"], 50)
text(s, 6, "từ camera ESP32-S3 + OV3660", 24)
for sid, t in ((8, "Ám hồng"), (10, "Mù sương"), (12, "Khối JPEG"), (14, "Tìm người")):
    text(s, sid, t)
text(s, 15, "GVHD: PGS.TS. Phạm Văn Cường")
for sid, img, lab, lid in ((16, "f1_raw.jpg", "Khung gốc", 18), (19, "f1_classic.jpg", "Cổ điển", 21),
                           (22, "f1_net.jpg", "Mạng NAFNet", 24)):
    picture(s, sid, img, "top")
    text(s, lid, lab)
notes(s, "Đồ án làm hai việc trên khung bay thật của camera ESP32-S3 + OV3660 (JPEG 640×480, chất lượng 12): làm rõ "
         "ảnh cho người vận hành và phát hiện người để cảnh báo. Ba ảnh nhỏ là cùng một khung bay ngày 02/11/2026: "
         "khung gốc, sau bộ lọc cổ điển và sau mạng NAFNet. Hình drone trên bìa chỉ là minh hoạ.\n\n"
         "Nguồn: BAO_CAO_DO_AN.md (Tóm tắt); khung 20261102-100941_005_00934.")

# ----------------------------------------------------------------------------- 2 nội dung
s = S[2]
text(s, 6, f"02 / {TOTAL}")
for sid, t in ((10, "Bài toán và dữ liệu"), (11, "Camera thật · 3 kiểu lỗi · 198 khung test"),
               (15, "Phương pháp"), (16, "D1/D2 · YOLO11s · cổ điển · NAFNet"),
               (20, "Thực nghiệm"), (21, "E1 · E2 · A/B mù · E4"),
               (25, "Tích hợp, kết luận"), (26, "IOT-CV · ms · giới hạn · phân công")):
    text(s, sid, t)
for sid, img in ((8, "f1_raw.jpg"), (13, "p_mask.jpg"), (18, "f1_classic.jpg"), (23, "f1_det.jpg")):
    picture(s, sid, img, "top")
notes(s, "Bốn phần: bài toán và dữ liệu thật; phương pháp (bộ làm xấu D1/D2, bộ nhận diện YOLO11s, bộ lọc cổ điển, "
         "mạng NAFNet); thực nghiệm E1, E2, chấm mù A/B, E4; cuối cùng là chạy thật trong phần mềm drone IOT-CV, giới "
         "hạn và phân công.\n\nNguồn: BAO_CAO_DO_AN.md.")

# ----------------------------------------------------------------------------- 3 lỗi camera
s = S[3]
header(s, 3, "01 · BÀI TOÁN", "Camera thật làm hỏng ảnh theo ba kiểu đo được")
for pid, img, tid, t, did, d in ((8, "fault_pink.jpg", 10, "Ám hồng", 11, "Kênh G sụt khi điểm ảnh gần bão hoà."),
                                 (13, "fault_haze.jpg", 15, "Mù + tối góc", 16, "Dark channel p50 144/255; góc 0,81× tâm."),
                                 (18, "fault_jpeg.jpg", 20, "JPEG + mép trái", 21, "Khối 8×8, 4:2:2; dải tối ~14 px.")):
    picture(s, pid, img)
    text(s, tid, t)
    text(s, did, d)
notes(s, "Hồ sơ camera đo trên 1115 khung dev: ám hồng ở vùng gần bão hoà (31 % khung có hơn 0,5 % điểm hồng), "
         "dark channel trung vị 143,7/255 (mù như sương), góc ảnh chỉ sáng 0,81 lần tâm, tỉ số khối 8×8 trung vị 2,68 "
         "do JPEG q12 lấy mẫu 4:2:2, và dải tối khoảng 14 px ở mép trái (hệ thống cắt 16 px). Ảnh phải là vùng mép trái "
         "phóng to 4 lần. Bản trước của nhóm viết cho ESP32-CAM OV2640 và chấm trên ảnh tự sinh; camera thật khác loại.\n\n"
         "Nguồn: results/camera/profile_report.md; configs/camera_profile.yaml.")

# ----------------------------------------------------------------------------- 4 dữ liệu
s = S[4]
header(s, 4, "01 · DỮ LIỆU", "Chỉ chấm điểm trên khung bay thật, chia theo khối thời gian")
text(s, 8, "198")
text(s, 9, "khung test, 39 khung có người")
picture(s, 10, "f1_gt.jpg", "top")
chip(s, 12, 13, "NHÃN NGƯỜI (GT)", header=False)
picture(s, 14, "negflight.jpg")
chip(s, 16, 17, "KHUNG KHÔNG NGƯỜI", header=False)
delete(s, 18)
text(s, 20, "100")
text(s, 21, "khung dev-val")
text(s, 22, "Chỉ để chọn model và ngưỡng.")
text(s, 24, "459")
text(s, 25, "khung âm")
text(s, 26, "300 trong nhà + 159 khung bay, để đếm báo nhầm.")
notes(s, "Chuyến bay 02/11/2026, phiên 20261102-100941. Khung chọn cách nhau ít nhất 1 giây và chia theo khối thời "
         "gian 30 giây, không chia ngẫu nhiên, để khung test không có khung gần giống trong tập huấn luyện. Tập test 198 "
         "khung (39 khung có người, 149 hộp người); dev-val 100 khung chỉ dùng chọn model và ngưỡng; thêm 300 khung âm "
         "trong nhà và 159 khung bay không người để đếm báo nhầm. Nhãn do nhiều model và SAHI gán trước, soát từng khung "
         "bằng mắt, chủ dự án duyệt lần cuối trên Label Studio.\n\nNguồn: BAO_CAO_DO_AN.md mục 2; data/real/split.json.")

# ----------------------------------------------------------------------------- 5 D1/D2
s = S[5]
header(s, 5, "02 · BỘ LÀM XẤU", "D2 sao chép camera để làm xấu ảnh huấn luyện")
picture(s, 7, "d2_src.jpg")
chip(s, 9, 10, "ẢNH NGUỒN", header=False)
picture(s, 11, "d2_out.jpg")
chip(s, 13, 14, "SAU D2", header=False)
for sid, t in ((18, "Mù theo mô hình khí quyển"), (22, "PSF + tối góc cos⁴"), (26, "Cắt kênh → ám hồng, nhiễu"),
               (30, "Mép trái + JPEG DQT q12")):
    text(s, sid, t)
text(s, 35, "D1: chuỗi Real-ESRGAN, tham số ngẫu nhiên   ·   D2: tham số hiệu chỉnh tự động cho khớp khung dev thật")
notes(s, "D1 là cách làm phổ biến khi không biết camera đích: chuỗi suy giảm kiểu Real-ESRGAN với tham số rút ngẫu nhiên. "
         "D2 là bản sao camera: màn mờ theo mô hình khí quyển (công thức), PSF và tối góc cos⁴ theo tiêu cự đo được, cắt "
         "từng kênh rồi nhân cân bằng trắng nên vùng sáng thành hồng, nhiễu theo NLF, dải mép trái, rồi nén JPEG một lần "
         "bằng đúng bảng lượng tử của camera ở q12, 4:2:2. Tham số được hiệu chỉnh tự động (xla.degrade.calibrate). "
         "Ảnh nguồn huấn luyện là VisDrone; NOMAD đã cân nhắc nhưng loại vì ảnh chụp xiên 45–60°. Ảnh VisDrone không "
         "được đăng lại, nên ảnh minh hoạ ở đây lấy nguồn là một khung bay của nhóm sau bộ lọc cổ điển rồi cho qua D2.\n\n"
         "Nguồn: xla/degrade/d2_camera.py; configs/degrade_d2.yaml; BAO_CAO_DO_AN.md mục 3.1.")

# ----------------------------------------------------------------------------- 6 độ giống thật (slide 7 gốc)
s = S[7]
header(s, 6, "02 · ĐỘ GIỐNG THẬT", "D2 khớp thống kê pixel, nhưng bộ phân loại vẫn tách được")
delete(s, *range(9, 18), *range(21, 30), *range(33, 42))
for hid, did, big, title, desc in (
        (8, 18, "5,93 → 0,80", "Lệch thống kê", "D1 → D2, tổng độ lệch 18 thống kê mức thấp. Thật–thật: 0,33."),
        (20, 30, "0,94 · 0,95", "C2ST", "D1 · D2. Mức 0,5 là không phân biệt được. Thật–thật: 0,54."),
        (32, 42, "4,44 · 4,41", "KID ×100", "D1 · D2, trên đặc trưng DINOv2. Thật–thật: 0,37.")):
    text(s, hid, title)
    x = sh(s, hid).left / EMU
    clone_textbox(s, hid, x, 2.8, 3.29, 0.8, big, 34)
    geom(s, did, y=3.85, h=1.9)
    text(s, did, desc)
text(s, 43, "Khớp thống kê là cần nhưng chưa đủ: nội dung ảnh VisDrone vẫn khác bãi bay của nhóm.")
notes(s, "Ba thước đo chạy trên patch ảnh. Lệch thống kê: D1 5,928, D2 0,796, D2 + Real-SR 0,663, mức sàn thật–thật "
         "0,330. C2ST là độ chính xác của bộ phân loại phân biệt patch thật với patch làm xấu: D1 0,936, D2 0,950, "
         "thật–thật 0,542. KID ×100: D1 4,443, D2 4,409, thật–thật 0,371. D2 sao chép camera đúng ở mức pixel, nhưng "
         "bộ phân loại vẫn tách được; nhiều khả năng do nội dung VisDrone (phố đô thị) khác bãi bay, giả thuyết này chưa "
         "kiểm riêng.\n\nNguồn: results/reproduce/tables/t2_do_giong_that.md.")

# ----------------------------------------------------------------------------- 7 hệ thống (slide 6 gốc)
s = S[6]
header(s, 7, "02 · HỆ THỐNG", "Nhận diện chạy trên khung gốc, ảnh rõ chỉ cho người xem")
text(s, 10, "Hồ sơ camera")
text(s, 11, ["DQT q12 · tối góc · hồng", "đo trên 1115 khung dev"])
text(s, 15, "Nhận diện")
text(s, 16, ["YOLO11s học trên ảnh D2", "tự huấn luyện · ONNX 1280"])
text(s, 20, "Làm rõ")
text(s, 21, ["Cổ điển 9,2 ms trên CPU", "NAFNet-w16 20,7 ms trên GPU"])
notes(s, "Hồ sơ camera là nền của mọi bước: bộ làm xấu D2, bộ lọc cổ điển và cả khung giả trong web dùng chung các số "
         "đo này. Bộ nhận diện YOLO11s học trên ảnh làm xấu D2 rồi tự huấn luyện trên khung bay chưa nhãn, xuất ONNX khung "
         "1280. Nhánh làm rõ có bộ lọc cổ điển và mạng NAFNet bề rộng 16. Quy tắc D5: nhận diện luôn chạy trên khung gốc, "
         "ảnh rõ chỉ để người xem. Hình lớp ảnh ở giữa là minh hoạ khái niệm.\n\n"
         "Nguồn: docs/hop-dong-model.md §2–3; BAO_CAO_DO_AN.md mục 3.")

# ----------------------------------------------------------------------------- 8 nhận diện (slide 15 gốc)
s = S[15]
header(s, 8, "02 · NHẬN DIỆN", "YOLO11s học ảnh D2, rồi tự huấn luyện trên khung bay")
picture(s, 7, "f1_gt.jpg", "top")
chip(s, 9, 10, "NHÃN TAY", header=False)
picture(s, 11, "f1_det.jpg", "top")
chip(s, 13, 14, "DETECTOR-V1.0.0", header=False)
text(s, 17, "YOLO11s · 1280 px")
text(s, 18, "E1: D0 · D1 · D2 · D2+SR")
text(s, 19, ["Chọn model: mAP50 dev-val", "Ngưỡng: F1 tối đa dev-val"])
delete(s, *range(23, 33), *range(36, 73))
geom(s, 21, w=3.3)
geom(s, 22, w=3.3)
text(s, 21, "E2: thầy YOLO11m")
text(s, 22, "gán nhãn giả khung bay")
text(s, 34, "Trò YOLO11s")
geom(s, 73, x=8.0, w=4.5)
geom(s, 74, x=8.0, w=4.5)
text(s, 73, "Xuất ONNX, resize trong graph")
text(s, 74, "ship trò vòng 1 theo dev-val")
notes(s, "Mốc là YOLO11s COCO có sẵn, khung 1280, bỏ 16 px mép trái. E1 tinh chỉnh YOLO11s trên VisDrone qua từng bộ "
         "làm xấu: D0 ảnh sạch, D1, D2, D2 + siêu phân giải; chọn model bằng mAP50 trên dev-val. E2: thầy YOLO11m gán nhãn "
         "giả cho khung bay chưa nhãn, trò YOLO11s học lại hai vòng; ship trò vòng 1 theo dev-val, xuất ONNX có cắt mép và "
         "resize trong graph, ngưỡng vận hành 0,077 chọn bằng F1 tối đa trên dev-val. Hình phải là hộp của chính bundle "
         "detector-v1.0.0 trên một khung test, hình trái là nhãn tay.\n\n"
         "Nguồn: configs/train_e1.yaml; configs/selftrain_e2.yaml; models/release/detector-v1.0.0/meta.json.")

# ----------------------------------------------------------------------------- 9 cổ điển (slide 12 gốc)
s = S[12]
header(s, 9, "02 · CỔ ĐIỂN", "Bộ lọc cổ điển chạy ngược chuỗi suy giảm của camera")
for pid, img, tid, t, did, d in ((8, "p_st2.jpg", 10, "Gỡ khối + hồng", 11, "Khử khối 8×8 ngay sau giải mã, rồi khử hồng."),
                                 (13, "p_st3.jpg", 15, "Bù tối góc", 16, "Chia cho bản đồ cos⁴, tiêu cự 1016 px."),
                                 (18, "p_st5.jpg", 20, "Khử mù + nét", 21, "Dark channel 15 px, unsharp trên Y.")):
    picture(s, pid, img)
    text(s, tid, t)
    text(s, did, d)
text(s, 22, "Bỏ siêu phân giải và giải chập của bản trước: trên khung thật chúng làm nổi khối JPEG.")
notes(s, "Thứ tự bước đảo chuỗi suy giảm của camera: JPEG, rồi cắt kênh và cân bằng trắng của ISP, rồi tối góc, rồi màn "
         "mờ. Khử khối đặt ngay sau giải mã giảm tỉ số khối còn 2,15; khử hồng trước khử mù giảm 93 % điểm hồng, đặt sau "
         "chỉ giảm 46–68 % vì khử mù nhân chênh lệch màu lên. Ba ảnh là kết quả cộng dồn sau từng bước trên một khung bay "
         "có ám hồng. Bộ lọc giữ ý tưởng mặt nạ hồng và cân bằng trắng có loại trừ của bản trước.\n\n"
         "Nguồn: xla/enhance/classic.py (docstring, STAGES); configs/enhance_classic.yaml.")

# ----------------------------------------------------------------------------- 10 mặt nạ hồng (slide 8)
s = S[8]
header(s, 10, "02 · KHỬ ÁM HỒNG", "Mặt nạ mềm chỉ chọn vùng hồng gần bão hoà")
picture(s, 7, "p_raw.jpg")
picture(s, 11, "p_mask.jpg")
picture(s, 15, "p_st2.jpg")
chip(s, 17, 18, "SAU KHỬ HỒNG", header=False)
notes(s, "Cổng gồm hai điều kiện: max(R, B) gần bão hoà (từ 230) và cả R lẫn B vượt G (từ 6 mức). Vật đỏ bão hoà có B "
         "thấp nên không bị đụng. Cổng được nở 21 px và làm mềm thành mặt nạ alpha; mỗi lượt sửa trừ delta khỏi R và B, "
         "cộng 0,704 delta vào G; chạy hai lượt. Ảnh giữa là alpha thật của lượt đầu trên khung này.\n\n"
         "Nguồn: demagenta trong xla/enhance/classic.py; configs/enhance_classic.yaml (magenta).")

# ----------------------------------------------------------------------------- 11 giữ độ chói (slide 9)
s = S[9]
header(s, 11, "02 · KHỬ ÁM HỒNG", "Sửa sắc độ gần bảo toàn độ chói")
text(s, 8, "MỘT ĐIỂM GẦN BÃO HOÀ · α = 1 · MỘT LƯỢT")
base, k = 5.0, 0.0075
for bar, lab, v in ((10, 11, 245), (13, 14, 200), (16, 17, 240), (20, 21, 221), (23, 24, 217), (26, 27, 216)):
    geom(s, bar, y=base - v * k, h=v * k)
    text(s, lab, str(v))
text(s, 32, "δ = 0,56·α·e; hệ số 0,704 bù vào G giữ Y gần như không đổi.")
notes(s, "Ví dụ một điểm gần bão hoà R, G, B = 245, 200, 240, alpha bằng 1: e = (245 + 240)/2 − 200 = 42,5; delta = "
         "0,56 × 42,5 ≈ 23,8. Sau một lượt: R ≈ 221, G ≈ 217, B ≈ 216. Độ chói Y trước 218,0, sau 218,1. Pipeline thật "
         "còn lượt thứ hai và bơm high-pass của G ở vùng bão hoà mạnh.\n\nNguồn: demagenta trong xla/enhance/classic.py.")

# ----------------------------------------------------------------------------- 12 khử mù (slide 13)
s = S[13]
header(s, 12, "02 · KHỬ MÙ", "Dark channel ước lượng độ truyền t(x) để khử mù")
picture(s, 7, "p_st3.jpg")
text(s, 9, "TRƯỚC KHỬ MÙ")
picture(s, 10, "p_tmap.jpg")
text(s, 12, "BẢN ĐỒ t(x) THÔ")
picture(s, 13, "p_st5.jpg")
text(s, 15, "SAU KHỬ MÙ + LÀM NÉT")
text(s, 19, "t thấp")
text(s, 20, "t cao")
picture_png(s, 17, "eq_dehaze.png")
text(s, 18, "patch 15 px   ·   ω = 0,85   ·   A: phân vị 99,9 của dark channel   ·   t tinh chỉnh bằng guided filter")
notes(s, "Dark channel là min của ba kênh rồi lấy min trong cửa sổ 15 px. Ánh sáng môi trường A lấy trung bình các điểm "
         "có dark channel từ phân vị 99,9, kẹp trong 180–250. Độ truyền t = 1 − 0,85·dc/A, sàn 0,4, tinh chỉnh bằng "
         "guided filter theo Y ở 1/4 độ phân giải. Bản đồ giữa là t thô trước guided filter; tím là t thấp (mù dày). Ảnh "
         "phải gồm cả bước làm nét unsharp trên Y. Trên tập test bộ cổ điển giảm dark channel trung vị 51,6 %.\n\n"
         "Nguồn: dehaze trong xla/enhance/classic.py; He, Sun & Tang (2011).")

# ----------------------------------------------------------------------------- 13 E1 (slide 17)
s = S[17]
header(s, 13, "03 · THỰC NGHIỆM", "E1: D2 hơn D1 về mAP50, nhưng CI còn chạm 0")
text(s, 8, "MAP50 · 198 KHUNG TEST THẬT")
set_chart(s, 9, ["Mốc COCO", "D0 sạch", "D1 ngẫu nhiên", "D2 camera", "D2 + SR"],
          [("mAP50", (0.126, 0.297, 0.143, 0.236, 0.149))], vmax=0.35, major=0.1)
text(s, 11, "Báo nhầm")
text(s, 12, "D2: 0,30 hộp/khung bay không người, D1: 0,99. Chênh có CI trọn dưới 0.")
text(s, 14, "Chọn theo dev-val")
text(s, 15, "Dev-val chọn D2 (0,329). D0 cao trên test nhưng báo nhầm 1,20.")
notes(s, "mAP50 trên 198 khung test: mốc YOLO11s COCO 0,126; D0 0,297; D1 0,143; D2 0,236; D2 + siêu phân giải 0,149. "
         "D2 hơn D1 0,093 mAP50 nhưng CI 95 % [−0,073; +0,252] chạm 0, nên chưa đủ để nói D2 hơn về mAP. Chỗ D2 hơn rõ là "
         "báo nhầm: 0,302 so với 0,994 hộp sai mỗi khung bay không người, chênh −0,692 [−1,238; −0,246]. D0 có CI rất rộng "
         "và báo nhầm cao. CI bằng bootstrap theo khối thời gian, 2000 mẫu.\n\n"
         "Nguồn: results/reproduce/tables/t1_nhan_dien.md; results/e1/README.md.")

# ----------------------------------------------------------------------------- 14 E2 (slide mới)
s = E2
header(s, 14, "03 · THỰC NGHIỆM", "E2: dev-val chọn trò vòng 1, test lại nghiêng về vòng 2")
text(s, 8, "MAP50 · 198 KHUNG TEST THẬT")
set_chart(s, 9, ["E1 D2", "Thầy YOLO11m", "Trò vòng 1", "Trò vòng 2", "Bundle ONNX"],
          [("mAP50", (0.236, 0.218, 0.156, 0.313, 0.120))], vmax=0.35, major=0.1)
text(s, 11, "Dev-val chọn sai")
text(s, 12, "Dev-val vòng 2 là 0,296, thấp hơn vòng 1 (0,379), nên theo luật ship vòng 1.")
text(s, 14, "Bundle đã ship")
text(s, 15, "detector-v1.0.0: mAP50 0,120, lệch bản .pt −0,003; 20,0 ms p50 ORT.")
notes(s, "Thầy YOLO11m 0,218; trò YOLO11s vòng 1 0,156; vòng 2 0,313 cao nhất trên test nhưng dev-val của nó thấp hơn "
         "vòng 1 (0,296 so với 0,379), nên theo luật đã chốt nhóm ship vòng 1. Bundle ONNX chấm bằng chính đường ONNX "
         "(cắt mép, letterbox 1280 trong graph) đạt 0,120 [0,032; 0,197], báo nhầm 0,881 hộp mỗi khung bay không người. "
         "ORT CUDA chạy riêng 20,0 / 20,2 ms p50 / p95. Bài học chính: 100 khung dev-val quá ít để xếp hạng các model "
         "sát nhau.\n\nNguồn: results/reproduce/tables/t1_nhan_dien.md; results/e2/README.md; "
         "models/release/detector-v1.0.0/meta.json.")

# ----------------------------------------------------------------------------- 15 ba đầu ra (slide 19)
s = S[19]
header(s, 15, "03 · LÀM RÕ", "Cổ điển và mạng trên cùng một khung bay")
for lid, title, pid, img, bid, big, did, d in (
        (8, "KHUNG GỐC", 9, "f1_raw.jpg", 11, "q12", 12, "JPEG 640×480 từ camera"),
        (14, "CỔ ĐIỂN", 15, "f1_classic.jpg", 17, "9,2 ms", 18, "CPU · hồng −89 % · mù −52 %"),
        (20, "MẠNG NAFNET-W16", 21, "f1_net.jpg", 23, "20,7 ms", 24, "GPU · hồng −99,9 % · mù −37 %")):
    text(s, lid, title)
    picture(s, pid, img, "top")
    text(s, bid, big)
    text(s, did, d)
notes(s, "Bộ cổ điển chạy 9,2 ms mỗi khung trên CPU (p50), giảm điểm hồng 89 % và dark channel 52 %. Mạng NAFNet bề rộng "
         "16 học từ cặp ảnh sạch và ảnh làm xấu D2, chạy 20,7 ms trên GPU, xoá 99,9 % điểm hồng nhưng màu ra nguội và nhạt, "
         "dark channel chỉ giảm 37 % (cần 50 %). Nhấp nháy giữa hai khung liền nhau tăng ở cả hai: mạng 1,81 lần, cổ điển "
         "2,18 lần. Mạng đạt 3/7 tiêu chí đăng ký trước.\n\n"
         "Nguồn: results/reproduce/tables/t3_tieu_chi_anh_ro.md; thuc-thi-phase-10-14.md §7.")

# ----------------------------------------------------------------------------- 16 A/B (slide 18)
s = S[18]
header(s, 16, "03 · THỰC NGHIỆM", "Chấm mù A/B: người chấm chọn bản cổ điển 24/25 lượt")
text(s, 8, "LƯỢT MẠNG THẮNG / 25 · L1 CẢ KHUNG · L2 PHÓNG TO 2×")
set_chart(s, 9, ["L2 vs cổ điển", "L2 vs gốc", "L1 vs cổ điển", "L1 vs gốc"],
          [("Lượt thắng", (5, 13, 1, 11))], fmt="0", vmax=25, axfmt="0", major=5,
          colors=("C99A5B", "0D9488", "C99A5B", "0D9488"))
for ml in sh(s, 9).chart._chartSpace.iter(qn("c:manualLayout")):
    ml.find(qn("c:x")).set("val", "0.30")
    ml.find(qn("c:w")).set("val", "0.62")
picture(s, 10, "f1_raw.jpg")
chip(s, 12, 13, "GỐC", header=False)
picture(s, 14, "f1_net.jpg")
chip(s, 16, 17, "MẠNG", header=False)
notes(s, "Người chấm thấy hai ảnh cạnh nhau, trái phải xáo trộn, chọn ảnh dễ nhìn hơn. Lần 1, cả khung: mạng thắng ảnh "
         "gốc 11/25 (0,44, Wilson [0,267; 0,629]), thắng cổ điển 1/25, tức cổ điển thắng 24/25. Lần 2 phóng to 2 lần: "
         "13/25 với ảnh gốc, 5/25 với cổ điển. Ngưỡng đạt là 70 % với gốc và 55 % với cổ điển, nên mạng không được bật làm "
         "mặc định. Nếu chỉ đếm các lượt có người, mạng thắng gốc 17/21; đó là quan sát hậu kiểm, không dùng để đổi kết "
         "luận.\n\nNguồn: results/reproduce/tables/t4_ab.md; results/ab/README.md.")

# ----------------------------------------------------------------------------- 17 E4 (slide 20)
s = S[20]
header(s, 17, "03 · E4", "E4: làm rõ trước khi nhận diện làm bộ đã tinh chỉnh kém đi")
text(s, 9, "Khung gốc")
text(s, 11, "Sau cổ điển")
text(s, 13, "Sau mạng")
text(s, 14, "mAP50 ↑")
set_chart(s, 15, ["YOLO11s COCO (có sẵn)", "detector-v1.0.0 (đã tinh chỉnh)"],
          [("Khung gốc", (0.126, 0.119)), ("Sau cổ điển", (0.081, 0.007)), ("Sau mạng", (0.195, 0.035))],
          vmax=0.25, axfmt="0.00", major=0.05, labels=True)
text(s, 17, "−0,085")
text(s, 18, "bộ đã tinh chỉnh + ảnh mạng, CI [−0,139; −0,011]")
text(s, 20, "Quy tắc D5: nhận diện luôn chạy trên khung gốc", 18)
notes(s, "Chạy hai bộ nhận diện trên ba biến thể khung test và so mAP50 ghép cặp với khung gốc. Với YOLO11s COCO có sẵn, "
         "ảnh qua mạng tăng 0,069 nhưng CI [−0,033; +0,231] chạm 0; qua cổ điển giảm 0,046. Với bộ đã tinh chỉnh, ảnh qua "
         "mạng giảm 0,085 với CI trọn dưới 0, qua cổ điển giảm 0,112. Bộ đã tinh chỉnh học trên ảnh xấu kiểu D2 nên ảnh đẹp "
         "lại là ảnh lạ với nó. Luật chốt trước: chỉ chuyển sang ảnh đã làm rõ khi CI nằm trọn trên 0, nên giữ khung gốc."
         "\n\nNguồn: results/reproduce/tables/t5_e4.md; results/e4/e4.json.")

# ----------------------------------------------------------------------------- 18 ms IOT-CV (slide 21)
s = S[21]
header(s, 18, "04 · TÍCH HỢP", "Trong IOT-CV: theo kịp camera, nhưng p95 vượt ngân sách")
text(s, 8, "23,6 ms")
text(s, 9, "detector · p50 ở nhịp camera thật")
text(s, 11, "37,5 ms")
text(s, 12, "tổng hai model · p50 · 7,2 khung/s")
text(s, 13, "PHÂN BỔ p50 Ở NHỊP THẬT, CÓ NGƯỜI XEM ẢNH RÕ")
total_w, gap = 12.09, 0.06
w_det = (total_w - gap) * 23.6 / 36.1
geom(s, 14, x=0.62, w=w_det)
geom(s, 15, x=0.82, w=w_det - 0.4)
text(s, 15, "Detector · 23,6 ms")
geom(s, 16, x=0.62 + w_det + gap, w=total_w - w_det - gap)
geom(s, 17, x=0.82 + w_det + gap, w=total_w - w_det - gap - 0.4)
text(s, 17, "Enhancer · 12,5 ms")
delete(s, 18, 19, 22)
text(s, 20, "YOLO11s, khung 1280")
geom(s, 21, x=0.62 + w_det + gap, w=4.0)
text(s, 21, "NAFNet-w16")
text(s, 23, "Phát lại chuyến 02/11 · ORT CUDA, RTX 4060 Laptop · bỏ 0,3 % khung · p95 tổng 54,1 ms > ngân sách 50 ms")
notes(s, "Hai model chạy trong backend IOT-CV qua ONNX Runtime CUDA, trên JPEG ghi được của chuyến 02/11, phát lại đúng "
         "nhịp. Có người xem Ảnh rõ: detector 23,6 / 33,4 ms p50 / p95, enhancer 12,5 / 19,0, tổng 37,5 / 54,1 ms, 7,2 khung "
         "mỗi giây, bỏ 0,3 % khung. Chạy hết tốc lực đạt 23,6 khung/s và đạt ngân sách (tổng p95 44,9 ms). Ở nhịp thật GPU "
         "rảnh khoảng 100 ms giữa hai khung và hạ xung, nên p95 vượt ngân sách 50 ms; nhóm chấp nhận và ghi là giới hạn.\n\n"
         "Nguồn: results/reproduce/tables/t6_iotcv_ms.md; BAO_CAO_DO_AN.md mục 4.5.")

# ----------------------------------------------------------------------------- 19 tích hợp (slide 22)
s = S[22]
header(s, 19, "04 · TÍCH HỢP", "Hai bundle ONNX nạp vào IOT-CV qua một hợp đồng cố định")
picture(s, 7, "f1_det.jpg", "top")
text(s, 10, "Camera  →  backend/vision")
text(s, 12, ["Detector trên khung gốc", "Hộp gửi qua WebSocket", "Enhancer cho luồng Ảnh rõ"])
text(s, 14, "Sai hợp đồng  →  từ chối", 22)
text(s, 17, "detector-v1.0.0 · enhancer-v1.0.0 · ORT CUDA")
notes(s, "Hợp đồng model (docs/hop-dong-model.md) cố định đầu vào là khung 640×480 BGR uint8 nguyên khung, cắt mép 16 px "
         "nằm trong graph; detector trả boxes theo toạ độ khung gốc và scores; enhancer trả khung cùng cỡ. Bundle gồm file "
         "ONNX, meta.json có sha256, và 3 khung golden để kiểm sai số. IOT-CV đặt bundle ở models/vision, từ chối nạp model "
         "sai hợp đồng và báo lý do lên web. Ảnh trái là hộp của detector-v1.0.0 trên một khung bay.\n\n"
         "Nguồn: docs/hop-dong-model.md; IOT-CV backend/vision.")

# ----------------------------------------------------------------------------- 20 giới hạn (slide mới)
s = LIM
header(s, 20, "04 · GIỚI HẠN", "Giới hạn của kết quả")
for hid, bid, h, b in ((8, 9, "Một bãi, một buổi", "Mọi số đo từ chuyến 02/11/2026; ánh sáng, độ cao, nền khác có thể đổi kết quả."),
                       (11, 12, "Tập test nhỏ", "39 khung có người, CI của mAP rộng khoảng ±0,1; dev-val 100 khung chọn sai ở E2."),
                       (14, 15, "Hồ sơ chỉ ở q12", "Bảng DQT đo ở q12; web hạ xuống q40 khi Wi-Fi yếu, mức đó chưa đo."),
                       (17, 18, "Ngân sách ms", "Ở nhịp camera, GPU hạ xung giữa hai khung: p95 tổng không đạt 50 ms.")):
    text(s, hid, h)
    text(s, bid, b)
notes(s, "Bốn giới hạn chính. Thứ nhất, mọi số đo đến từ một bãi, một buổi sáng 02/11/2026. Thứ hai, chỉ 39 khung test "
         "có người nên CI của mAP rộng; dev-val 100 khung đã không chọn đúng model ở E2. Thứ ba, bảng lượng tử chỉ đo ở "
         "q12 trong khi web hạ chất lượng xuống q40 khi Wi-Fi yếu. Thứ tư, ở nhịp camera thật trên laptop này p95 không đạt "
         "ngân sách 50 ms. Ngoài ra nhãn do máy gán rồi người duyệt, vẫn có thể sót người rất nhỏ; NR-IQA chỉ để mô tả.\n\n"
         "Nguồn: BAO_CAO_DO_AN.md mục 6.")

# ----------------------------------------------------------------------------- 21 phân công (slide mới)
s = TEAM
header(s, 21, "PHÂN CÔNG", "Phân công nhóm 10")
for hid, bid, h, b in (
        (8, 9, "Nguyễn Duy Nghĩa", ["B23DCCN600", "Làm rõ ảnh (cổ điển + NAFNet), đánh giá A/B, tích hợp IOT-CV."]),
        (11, 12, "Nguyễn Tự Kiên", ["B23DCCN465", "Bộ làm xấu D1/D2, đo độ giống thật (C2ST/KID), hiệu chỉnh D2."]),
        (14, 15, "Nguyễn Minh Toàn", ["B23DCCN832", "Nhận diện người: mốc, E1, E2 tự huấn luyện, xuất ONNX."]),
        (17, 18, "Nguyễn Thành Trung", ["B23DCCN861", "Thu khung bay thật, hồ sơ camera, chia tập, gán nhãn test."])):
    text(s, hid, h)
    text(s, bid, b)
notes(s, "Phân công theo phần việc của bản làm lại trên camera thật. Chủ dự án duyệt nhãn tập test và chấm A/B. Trợ lý AI "
         "(Claude Code) viết phần lớn mã và chạy thí nghiệm; nhóm đọc, duyệt và chịu trách nhiệm về kết luận.\n\n"
         "Nguồn: phân công nhóm; BAO_CAO_DO_AN.md mục 9–10.")

# ----------------------------------------------------------------------------- 22 tài liệu (slide 25)
s = S[25]
header(s, 22, "TÀI LIỆU THAM KHẢO", "Tài liệu phương pháp và nguồn thực nghiệm")
for hid, bid, h, b in (
        (8, 9, "Khử mù và làm xấu ảnh", ["He, Sun & Tang (2011). Dark Channel Prior. IEEE TPAMI 33(12).",
                                          "Wang et al. (2021). Real-ESRGAN. ICCVW."]),
        (11, 12, "Mạng và nhận diện", ["Chen et al. (2022). NAFNet. ECCV.",
                                       "Ultralytics YOLO11; RoyChowdhury et al. (2019). Self-training. CVPR."]),
        (14, 15, "Dữ liệu và độ giống thật", ["Zhu et al. (2022). VisDrone. IEEE TPAMI 44(11).",
                                              "Lopez-Paz (2017) C2ST; Bińkowski (2018) KID; Oquab (2024) DINOv2."]),
        (17, 18, "Mã và dữ liệu", ["github.com/Localhost218/XLA",
                                   "scripts/reproduce.py --check  ·  results/reproduce/"])):
    text(s, hid, h)
    text(s, bid, b)
notes(s, "Các phương pháp tham khảo từ công trình ghi trên slide rồi điều chỉnh theo camera của nhóm. Mọi bảng và hình "
         "sinh bằng scripts/reproduce.py từ kết quả đã lưu; lệnh --check chấm lại 15 tệp dự đoán trên CPU và so sha256 "
         "của 9 bảng, hình. Danh sách đầy đủ ở BAO_CAO_DO_AN.md.\n\nNguồn: BAO_CAO_DO_AN.md, Tài liệu tham khảo.")

# ----------------------------------------------------------------------------- 23 kết luận (slide 26)
s = S[26]
header(s, 23, "KẾT LUẬN", "Kết quả và hướng phát triển")
text(s, 9, "0,30")
text(s, 10, "báo nhầm/khung bay · D2 (D1: 0,99)")
text(s, 12, "24/25")
text(s, 13, "lượt A/B chọn bộ cổ điển")
text(s, 15, "Hai câu trả lời")
text(s, 16, "D2 hơn D1 ở báo nhầm, mAP chưa chắc. Làm rõ trước khi nhận diện làm hại bộ đã tinh chỉnh.")
text(s, 18, "Tiếp theo")
text(s, 19, "Sửa phơi sáng, WB ở cảm biến · thêm khung có người · ràng buộc màu, thời gian cho mạng · thử imgsz 960.")
notes(s, "Câu 1: làm xấu theo số đo camera (D2) thắng làm xấu ngẫu nhiên ở độ giống thật mức pixel và ở báo nhầm (0,30 so "
         "với 0,99 hộp mỗi khung bay không người), còn mAP chưa kết luận được. Câu 2: làm rõ trước khi nhận diện không giúp "
         "bộ đã tinh chỉnh (giảm rõ), với bộ có sẵn có xu hướng giúp nhưng CI chạm 0. Ảnh rõ: người chấm chọn bộ cổ điển "
         "24/25 lượt, mạng chưa bật mặc định. Việc tiếp: sửa tận gốc ở cảm biến (phơi sáng, cân bằng trắng, đo DQT q40), "
         "thu thêm khung có người nhiều buổi, thêm ràng buộc màu và thời gian khi huấn luyện mạng, thử detector imgsz 960 "
         "hoặc FP16.\n\nNguồn: BAO_CAO_DO_AN.md mục 5, 7.")

# ----------------------------------------------------------------------------- thứ tự + xoá
order_src = [1, 2, 3, 4, 5, 7, 6, 15, 12, 8, 9, 13, 17, "E2", 19, 18, 20, 21, 22, "LIM", "TEAM", 25, 26]
new_map = {"E2": E2, "LIM": LIM, "TEAM": TEAM}
lst = prs.slides._sldIdLst
by_slide = {}
for sid_el in list(lst):
    sl = prs.part.related_part(sid_el.get(qn("r:id"))).slide
    by_slide[id(sl)] = sid_el
keep = []
for k in order_src:
    sl = new_map[k] if isinstance(k, str) else S[k]
    keep.append(by_slide[id(sl)])
for el in list(lst):
    lst.remove(el)
    if el not in keep:
        prs.part.drop_rel(el.get(qn("r:id")))
for el in keep:
    lst.append(el)
assert len(keep) == TOTAL
prs.save(str(DST))
print("saved", DST, len(prs.slides))
