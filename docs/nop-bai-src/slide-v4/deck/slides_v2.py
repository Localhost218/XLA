"""AI v2 slides: three-tier system, external data, detector comparison, verifier, system results,
classic ONNX, enhancement comparison, blind A/B, IOT-CV integration, conclusion, references.
Every number is copied from D:/XLA/results/v2/**/README.md, results/reproduce/tables/t7-t9 and results/ab-v2."""
import json
import os

from pptx.enum.chart import XL_LABEL_POSITION as LP

import common as C
import kit
import stage as G
from kit import BLUE, GOLD, INK, MUTED, ORANGE, PINK, PURPLE, SLATE, TEAL, TEXT, card, line, text, vn
from slides1 import L, W, gap_arrows, stage_row

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "img") + "/"
RPT = G.RPT + "/"
GREEN = "1E9E5A"


def ci(lo, hi, nd=3):
    return f"[{vn(lo, nd)}; {vn(hi, nd)}]"


def sgn(x, nd=3):
    return ("+" if x >= 0 else "") + vn(x, nd)


def xticks(s, px, py, pw, ph, vmax, step, nd=1):
    k, v = 0, 0.0
    while v <= vmax + 1e-9:
        text(s, px + v / vmax * pw - 0.3, py + ph + 0.04, 0.6, 0.25, vn(v, nd) if v else "0", 11, MUTED, align="c")
        k += 1; v = k * step


# ------------------------------------------------------------------ system: three tiers + Ảnh rõ
def s_system():
    s = C.new("blue")
    C.header(s, "blue", "01 · BÀI TOÁN", "Hệ tìm người ba tầng, một luồng Ảnh rõ chạy song song")
    G.panel(s, L, 1.85, 1.62, 2.85, None, "Đầu vào")
    G.framed(s, RPT + "art_full.jpg", 0.74, 2.3, 1.38, 1.04)
    text(s, 0.72, 3.42, 1.42, 1.2, ["ESP32-S3 N16R8", "+ OV3660,", "nhìn xuống"], 12, TEXT, align="c")
    specs = [("yel", "Tầng 1: detector"), ("lav", "Tầng 2: verifier"), ("blu", "Tầng 3: lọc track"), ("mint", "Ảnh rõ: classic")]
    xs = stage_row(s, 2.39, 1.85, 2.85, [2.44] * 4, 0.15, specs)
    nodes = [[("gói ONNX nhận diện", "learn"), ("YOLO11s, letterbox 1 280", "data"), ("hộp có s ≥ 0,0767", "data")],
             [("cắt b′ quanh hộp, 96×96", "data"), ("EfficientNet-B0", "learn"), ("giữ hộp có v ≥ 0,35", "data")],
             [("ghép hộp theo IoU 0,3", "data"), ("k = 1, n = 1", "learn"), ("sự kiện theo từng track", "data")],
             [("gói ONNX làm rõ", "learn"), ("5 tầng lọc, ONNX opset 17", "data"), ("bật sẵn trên web", "data")]]
    bars = ["mAP50 0,119", "báo nhầm 0,874 → 0,289", "mAP50 cả hệ 0,210", "p95 19,3 ms trong IOT-CV"]
    for x, ns, b, pal in zip(xs, nodes, bars, ["yel", "lav", "blu", "mint"]):
        for k, (lab, kind) in enumerate(ns):
            y = 2.32 + k * 0.62
            G.node(s, x + 0.12, y, 2.2, 0.44, lab, kind, 11)
            if k < 2:
                G.arrow(s, x + 1.22, y + 0.44, x + 1.22, y + 0.62, INK, 1.25)
        G.bar(s, x + 0.1, 4.22, 2.24, 0.36, pal, b, 11)
    gap_arrows(s, [(L + 1.62, xs[0]), (xs[0] + 2.44, xs[1]), (xs[1] + 2.44, xs[2])], 2.54)
    cards = [("Đầu vào", ["JPEG q12,", "2–9 khung/s"]),
             ("Giai đoạn 1", ["ONNX của E2 vòng 1", "+ recommended_track"]),
             ("Giai đoạn 2", ["cal: 15 → 2 hộp sai,", "giữ 58/218 người"]),
             ("Giai đoạn 3", ["k = 1: hộp đã qua", "tầng 2 là đủ"]),
             ("Giai đoạn 4", ["A/B mù 25/25;", "đọc khung gốc"])]
    for i, (lab, lines) in enumerate(cards):
        x = L if i == 0 else xs[i - 1]
        w = 1.62 if i == 0 else 2.44
        G.detail(s, x, 4.85, w, 1.02, lab, lines, 12 if i == 0 else 13)
    G.detail(s, L, 6.0, 5.97, 0.95, "Câu hỏi 1",
             ["Thêm một tầng xác minh có bớt báo nhầm mà vẫn giữ đủ người trên khung bay thật?"], 14)
    G.detail(s, 6.75, 6.0, 5.97, 0.95, "Câu hỏi 2",
             ["Cách làm rõ nào vừa đủ nhanh cho luồng trực tiếp vừa được người xem mù chọn?"], 14)
    C.notes(s, ["Một khung camera vào ba tầng nhận diện nối tiếp; luồng Ảnh rõ chạy riêng trên cùng khung gốc, không đưa vào nhận diện.",
                "Tầng 1 là YOLO11s E2 (gói ONNX nhận diện); tầng 2 verifier EfficientNet-B0 mới; tầng 3 lọc track k = n = 1.",
                "Ảnh rõ là bộ lọc cổ điển xuất ONNX (gói ONNX làm rõ), bật sẵn trên web.",
                "Hai câu hỏi: tầng xác minh có bớt báo nhầm không, và cách làm rõ nào được chọn."],
            "results/v2/README.md (bundle); results/v2/system/README.md §1–4; results/v2/iotcv/README.md §1–2.")
    return s


# ------------------------------------------------------------------ external data
def s_ext_data():
    s = C.new("blue")
    C.header(s, "blue", "03 · DỮ LIỆU", "Dữ liệu ngoài: người HERIDAL cỡ 42 px, gần camera thật")
    G.panel(s, L, 1.85, 1.6, 1.95, None, "Ba bộ ngoài")
    for k, lab in enumerate(["HERIDAL", "TinyPerson", "WiSARD"]):
        G.node(s, 0.74, 2.28 + k * 0.5, 1.36, 0.4, lab, "data", 11)
    specs = [("yel", "1. Giấy phép, sha256"), ("lav", "2. Probe góc nhìn"), ("blu", "3. Soát tay"), ("mint", "4. Cắt ô, làm xấu D2")]
    xs = stage_row(s, 2.37, 1.85, 1.95, [2.47] * 4, 0.12, specs)
    sub = [["CC BY 3.0 · MIT;", "kiểm sha256 file nén"], ["CLIP ViT-B-32 + logistic,", "điểm ≥ 0,8: nadir/near"],
           ["100 ảnh mỗi bộ,", "4 tờ × 25 ô"], ["ô 1 280×960 chồng 160 px,", "cửa sổ 624×480 + dải mép"]]
    for x, t in zip(xs, sub):
        text(s, x + 0.1, 2.32, 2.27, 0.62, t, 12, TEXT, align="c")
    bars = ["SARD bỏ: cần thuê bao", "precision 1,00 ở 3 bộ", "HERIDAL: 16/17 nơi nadir", "d02x: 15 216 ảnh"]
    for x, pal, b in zip(xs, ["yel", "lav", "blu", "mint"], bars):
        G.bar(s, x + 0.12, 3.3, 2.23, 0.36, pal, b, 11)
    gap_arrows(s, [(L + 1.6, xs[0])] + [(xs[i] + 2.47, xs[i + 1]) for i in range(3)], 2.6)
    card(s, L, 3.95, 7.4, 3.0)
    text(s, 0.85, 4.07, 7.0, 0.3, "SAU LỌC GÓC NHÌN · PROBE TRÊN 100 ẢNH SOÁT TAY MỖI BỘ", 12, BLUE, True)
    rows = [["Bộ", "Trước lọc", "Sau lọc (ảnh / người)", "Probe P · R", "Dùng cho"],
            ["HERIDAL", "1 684", "1 684 / 3 229", "1,00 · 0,77", "d02x, verifier"],
            ["TinyPerson", "1 532", "173 / 1 682", "1,00 · 0,37", "dương verifier"],
            ["WiSARD", "264", "110 / 424", "1,00 · 0,40", "d02x, verifier"]]
    kit.table(s, 0.85, 4.45, [1.3, 0.95, 2.0, 1.15, 1.65], 0.44, rows, size=13, aligns="lrrrl")
    text(s, 0.85, 6.32, 7.0, 0.5, "d02x = VisDrone + ô HERIDAL + WiSARD, mỗi đơn vị một bản D0 và một bản D2: 13 008 train / 2 208 val, nguồn ngoài 54 %.", 12, MUTED)
    card(s, 8.17, 3.95, 4.55, 3.0)
    text(s, 8.4, 4.07, 4.1, 0.3, "CẠNH NGẮN HỘP NGƯỜI, P50 (PX)", 12, BLUE, True)
    cats = ["VisDrone", "HERIDAL", "WiSARD", "camera thật"]
    vals = [14.0, 42.1, 46.5, 53.5]
    kit.bar_chart(s, 8.25, 4.4, 4.4, 2.45, cats,
                  [("p50", vals, SLATE, [vn(v, 1) for v in vals], {0: SLATE, 1: TEAL, 2: BLUE, 3: PINK}, LP.OUTSIDE_END)],
                  horizontal=True, gap=40, vmin=0, vmax=65, layout=(0.3, 0.04, 0.6, 0.9))
    C.notes(s, ["Ba bộ ảnh nhìn từ trên xuống thêm cho hệ ba tầng: HERIDAL 1 684 ảnh, WiSARD 110, TinyPerson 173 sau lọc góc nhìn.",
                "Probe góc nhìn là logistic trên đặc trưng CLIP, như đã dùng cho VisDrone; trên 300 ảnh soát tay không nhận nhầm ảnh xiên nào.",
                "Người trong HERIDAL và WiSARD cỡ 42–47 px, gần cỡ người của camera thật (53,5 px) hơn VisDrone (14 px).",
                "TinyPerson còn ít ảnh nên chỉ làm mẫu dương cho verifier; SARD bỏ vì cần thuê bao trả phí."],
            "results/v2/data/README.md §1–5; configs/view_external.yaml.")
    return s


# ------------------------------------------------------------------ detector candidates vs v1
def s_detector_cmp():
    s = C.new("orange")
    C.header(s, "orange", "09 · SO SÁNH NHẬN DIỆN", "Thầy YOLO11m, trò C1 và C3 so với YOLO11s E2: giữ E2")
    card(s, L, 1.85, 6.2, 5.1)
    text(s, 0.9, 1.98, 5.8, 0.3, "MAP50 OOF · 160 KHUNG DEVPOOL, 218 NGƯỜI · CI 95 %", 12, ORANGE, True)
    rows = [("thầy YOLO11m (.pt)", 0.197, 0.156, 0.251, MUTED), ("trò C1: d02x (.pt)", 0.306, 0.208, 0.417, BLUE),
            ("trò C3: D0 + nền (.pt)", 0.321, 0.210, 0.446, TEAL), ("YOLO11s E2 r1 (.pt)", 0.340, None, None, ORANGE),
            ("YOLO11s E2 (ONNX)", 0.229, 0.144, 0.312, PINK)]
    FX, FY, FW, FH = 0.75, 2.35, 5.95, 3.05
    LAY = (0.42, 0.02, 0.54, 0.88)
    gf = kit.bar_chart(s, FX, FY, FW, FH, [r[0] for r in rows],
                       [("m", [r[1] for r in rows], SLATE, [vn(r[1]) for r in rows], {i: r[4] for i, r in enumerate(rows)}, LP.INSIDE_BASE)],
                       horizontal=True, gap=45, vmin=0, vmax=0.5, major=0.1, grid=True, layout=LAY)
    ser = gf.chart.plots[0].series[0]
    kit._err_bars(ser, [(r[3] - r[1]) if r[3] else 0 for r in rows], [(r[1] - r[2]) if r[2] else 0 for r in rows])
    for i in range(len(rows)):
        for r in ser.points[i].data_label.text_frame.paragraphs[0].runs:
            r.font.color.rgb = kit.RGBColor.from_string("FFFFFF")
    px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
    xticks(s, px, py, pw, ph, 0.5, 0.1)
    G.eq(s, "oof", 0.9, 5.82, 0.85)
    text(s, 2.4, 5.8, 4.3, 1.1, ["OOF: khung fold k chấm bằng model học không có fold k (fold 0 = phiên #5, fold 1 = #1–#4).",
                                 "E2 đã học nhãn giả khung dev nên OOF có lợi cho E2."], 12, TEXT)
    card(s, 7.0, 1.85, 5.72, 5.1)
    text(s, 7.25, 1.98, 5.3, 0.3, "HIỆU GHÉP CẶP B − A, CÙNG ĐƯỜNG DỰ ĐOÁN", 12, ORANGE, True)
    trs = [["B − A", "mAP50 OOF", "CI 95 %"],
           ["thầy − E2", "−0,031", "[−0,092; +0,052]"],
           ["C3 − E2 (.pt)", "−0,020", "[−0,059; +0,021]"],
           ["C3 − E2 (ONNX)", "−0,041", "[−0,078; +0,018]"],
           ["C1 − C3 (.pt)", "−0,015", "[−0,060; +0,018]"]]
    kit.table(s, 7.25, 2.38, [1.85, 1.35, 2.05], 0.42, trs, size=13, aligns="lrr")
    G.eq(s, "pdiff", 7.25, 4.6, 0.8)
    text(s, 7.25, 4.98, 5.3, 0.85, ["Thầy tìm thêm người nhỏ nhưng xếp điểm kém hơn: không dùng làm thầy nhãn giả.",
                                    "Graph ONNX bỏ hộp < 0,05, cắt điểm thấp của C3. Cắt ô SAHI 320: 0,293, 174 ms."], 11, TEXT, spacing=1.0)
    kit.table(s, 7.25, 5.85, [1.95, 1.55, 1.75], 0.26, [
        ["OOF devpool", "AR-small", "báo nhầm/khung trống"],
        ["thầy YOLO11m", "0,379", "0,28"],
        ["trò C3 (.pt)", "0,446", "0,54"],
        ["YOLO11s E2 (ONNX)", "0,172", "0,28"]], size=11, aligns="lrr")
    C.notes(s, ["Thử thay YOLO11s E2: thầy YOLO11m để làm nhãn giả, trò C1 học d02x, trò C3 học VisDrone sạch cộng khung nền.",
                "Luật chọn chỉ đọc OOF devpool; mọi hiệu so với E2 có CI chứa 0 nên hệ giữ E2, ONNX không đổi.",
                "Cùng đường ONNX, C3 kém E2 0,041 vì graph cắt hộp điểm dưới 0,05. C3 chạy 21,6 / 22,1 ms p50 / p95.",
                "Cắt ô SAHI 320 px kém hơn (0,293) và chậm 9 lần nên không tích hợp."],
            "results/v2/teacher/README.md §3; results/v2/detect/README.md §1–7.")
    return s


# ------------------------------------------------------------------ verifier stages
def s_verifier():
    s = C.new("orange")
    C.header(s, "orange", "10 · VERIFIER", "Verifier EfficientNet-B0 xem lại từng hộp trên vùng cắt 96×96")
    G.panel(s, L, 1.85, 1.6, 2.6, None, "Đầu vào")
    G.node(s, 0.74, 2.32, 1.36, 0.85, ["khung gốc", "uint8 BGR"], "data", 11)
    G.node(s, 0.74, 3.35, 1.36, 0.85, ["N hộp tầng 1", "s ≥ 0,0767"], "data", 11)
    specs = [("yel", "1. Vùng ngữ cảnh"), ("lav", "2. RoIAlign 96×96"), ("blu", "3. EfficientNet-B0"), ("mint", "4. Ngưỡng")]
    xs = stage_row(s, 2.37, 1.85, 2.6, [2.47] * 4, 0.12, specs)
    text(s, xs[0] + 0.1, 2.3, 2.27, 0.9, ["cạnh s = 1,5 × cạnh lớn,", "tối thiểu 32 px, giữ tâm,", "kẹp [16; 640] × [0; 480]"], 11, TEXT, align="c")
    G.node(s, xs[1] + 0.14, 2.32, 2.19, 0.42, "sampling_ratio 2", "data", 11)
    G.node(s, xs[1] + 0.14, 2.92, 2.19, 0.42, "BGR → RGB, chuẩn hoá", "data", 11)
    G.arrow(s, xs[1] + 1.235, 2.74, xs[1] + 1.235, 2.92, INK, 1.25)
    G.node(s, xs[2] + 0.14, 2.32, 2.19, 0.42, "trọng số ImageNet", "frozen", 11)
    G.node(s, xs[2] + 0.14, 2.92, 2.19, 0.42, "1 logit → sigmoid → v", "learn", 11)
    G.arrow(s, xs[2] + 1.235, 2.74, xs[2] + 1.235, 2.92, INK, 1.25)
    G.node(s, xs[3] + 0.14, 2.32, 2.19, 0.42, "v ≥ 0,35: giữ hộp", "learn", 11)
    G.node(s, xs[3] + 0.14, 2.92, 2.19, 0.42, "N = 0: trả [0]", "data", 11)
    G.arrow(s, xs[3] + 1.235, 2.74, xs[3] + 1.235, 2.92, INK, 1.25)
    bars = ["context_scale 1,5", "N hộp động, 1 graph", "4 ứng viên, chọn B0", "0,35 chọn trên cal"]
    for x, pal, b in zip(xs, ["yel", "lav", "blu", "mint"], bars):
        G.bar(s, x + 0.12, 3.92, 2.23, 0.36, pal, b, 11)
    gap_arrows(s, [(L + 1.6, xs[0])] + [(xs[i] + 2.47, xs[i + 1]) for i in range(3)], 2.75)
    cards = [(L, 1.6, "Đầu vào", ["hộp tầng 1,", "≤ 100 hộp"]),
             (xs[0], 2.47, "Giai đoạn 1", ["nhìn cả nền quanh hộp:", "túi nhựa, bụi cây, bóng"]),
             (xs[1], 2.47, "Giai đoạn 2", ["lệch so với PyTorch:", "CPU 1e-6, CUDA 3,4e-4"]),
             (xs[2], 2.47, "Giai đoạn 3", ["MobileNetV3-S và B0,", "vùng cắt 64 hoặc 96"]),
             (xs[3], 2.47, "Giai đoạn 4", ["N = 5: p50 5,96 ms", "p95 8,52 ms"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 4.42, w, 1.05, lab, lines, 12)
    card(s, L, 5.6, W, 1.35)
    G.eq(s, "vbce", 0.85, 5.68, 0.78)
    G.eq(s, "vroi", 0.85, 6.4, 0.68)
    text(s, 4.75, 5.68, 7.85, 1.25, [
        [("18 766 mẫu: ", {"bold": True, "color": INK}),
         ("âm khó 91 hộp thật (mine 32, devpool 59), âm nền 1 888, âm báo nhầm trên d02x 9 444; dương 218 + 80 thật, 3 320 + 3 000 d02x, 725 TinyPerson.", {})],
        [("Học: ", {"bold": True, "color": INK}),
         ("w = 4 cho âm khó thật, 3 cho dương thật; AdamW 1e-3 OneCycle, batch 256, ≤ 12 epoch; mỗi mẫu 3 bản rung tâm ±12 %, cỡ ±20 %, JPEG q 25–85.", {})]],
         12, TEXT, spacing=1.05)
    C.notes(s, ["Verifier nhận khung gốc và các hộp của tầng 1, cắt vùng vuông rộng gấp 1,5 lần cạnh lớn của hộp, tối thiểu 32 px.",
                "RoIAlign đưa mỗi vùng về 96×96, EfficientNet-B0 cho một logit, sigmoid ra điểm v; giữ hộp khi v ≥ 0,35.",
                "Mẫu học: 18 766 vùng cắt; âm khó thật chỉ 91 hộp nên thêm âm báo nhầm trên d02x và tăng trọng số mẫu thật.",
                "Graph một model, N động, opset 17; lệch PyTorch 1e-6 trên CPU."],
            "results/v2/verifier/README.md §2–6; xla/verify/common.py:122-140; configs/verifier.yaml.")
    return s


# ------------------------------------------------------------------ verifier results
def s_verifier_res():
    s = C.new("orange")
    C.header(s, "orange", "10 · VERIFIER", "EfficientNet-B0 bỏ 82 % báo nhầm OOF, cal 15 → 1 hộp")
    cands = ["chỉ detector", "MobileNetV3-S 64", "MobileNetV3-S 96", "EfficientNet-B0 64", "EfficientNet-B0 96"]
    cols = {0: SLATE, 1: BLUE, 2: BLUE, 3: TEAL, 4: ORANGE}
    for i, (ttl, vals, vmax) in enumerate([("BÁO NHẦM OOF · 245 KHUNG (HỘP)", [92, 33, 38, 40, 17], 110),
                                           ("BÁO NHẦM CAL · 150 KHUNG KHÔNG NGƯỜI (HỘP)", [15, 0, 3, 1, 1], 18)]):
        x = L + i * 3.75
        card(s, x, 1.85, 3.6, 3.15)
        text(s, x + 0.2, 1.97, 3.3, 0.5, ttl, 11, ORANGE, True)
        kit.bar_chart(s, x + 0.05, 2.45, 3.5, 2.5, cands if i == 0 else [""] * 5,
                      [("n", vals, SLATE, [str(v) for v in vals], cols, LP.OUTSIDE_END)],
                      horizontal=True, gap=35, vmin=0, vmax=vmax, layout=(0.5 if i == 0 else 0.05, 0.03, 0.4 if i == 0 else 0.8, 0.92))
    text(s, L, 5.1, 7.35, 0.62, "Mỗi ứng viên ở ngưỡng làm recall OOF giảm ≤ 0,05 (recall 0,266 → 0,220). Chọn: giảm báo nhầm OOF nhiều nhất.", 12, MUTED)
    card(s, 8.12, 1.85, 4.6, 3.15)
    text(s, 8.35, 1.97, 4.2, 0.3, "MS ORT CUDA THEO SỐ HỘP N · P50 / P95", 11, ORANGE, True)
    rows = [["", "N = 1", "N = 5", "N = 20"],
            [("B0 96 (chọn)", {"bold": True, "color": ORANGE}), "4,15/6,98", "5,96/8,52", "13,33/15,19"],
            ["MNv3-S 64", "2,72/5,43", "2,93/5,70", "3,85/6,50"],
            ["MNv3-S 96", "3,24/5,88", "3,79/6,19", "6,66/8,91"],
            ["B0 64", "3,72/6,80", "4,48/7,59", "7,72/10,37"]]
    kit.table(s, 8.3, 2.35, [1.3, 0.95, 0.95, 1.1], 0.44, rows, size=11, aligns="lrrr")
    text(s, 8.35, 4.6, 4.2, 0.35, "GPU 73 °C, CPU không hạ xung; 300 lần đo.", 11, MUTED)
    card(s, L, 5.75, W, 1.2)
    text(s, 0.85, 5.85, 3.0, 0.3, "Ngưỡng ghi vào bundle", 14, INK, True)
    text(s, 0.85, 6.2, 3.0, 0.7, "luật D7 trên cal + devpool", 12, MUTED)
    kpis = [("0,100 → 0,013", "hộp sai mỗi khung cal"), ("0,266 → 0,266", "recall OOF, 58/218 người"), ("92 → 58", "báo nhầm OOF (−37 %)")]
    for k, (big, small) in enumerate(kpis):
        x = 3.9 + k * 2.95
        text(s, x, 5.82, 2.85, 0.5, big, 22, ORANGE, True)
        text(s, x, 6.4, 2.85, 0.45, small, 12, TEXT)
    C.notes(s, ["Bốn ứng viên: hai backbone MobileNetV3-Small và EfficientNet-B0, hai cỡ vùng cắt 64 và 96.",
                "So ở cùng mức mất recall OOF ≤ 0,05: B0 96 còn 17 hộp sai OOF (giảm 82 %), cal 15 → 1 hộp.",
                "MobileNetV3-S 64 nhanh hơn (2,93 ms ở N = 5) nhưng lọc kém hơn; luật chọn ưu tiên giảm báo nhầm.",
                "Ở ngưỡng ghi bundle, cal còn 0,013 hộp mỗi khung và recall OOF giữ nguyên 0,266."],
            "results/v2/verifier/README.md §4–5 (select.json, ms_candidates.json).")
    return s


# ------------------------------------------------------------------ calibration grid
def s_calib():
    s = C.new("orange")
    C.header(s, "orange", "10 · VERIFIER", "Lưới 720 điểm chọn 0,35, k = n = 1: giữ 58/218 người")
    d = json.load(open("D:/XLA/results/v2/system/calibrate.json", encoding="utf-8"))
    R = {r["key"]: r for r in d["rows"]}
    ts = [i * 0.05 for i in range(20)]
    k1 = [R["t%.2f_k1_n1_g1" % t]["tp"] for t in ts]
    k2 = [R["t%.2f_k2_n3_g1" % t]["tp"] for t in ts]
    fp = [R["t%.2f_k1_n1_g1" % t]["devpool_fp"] for t in ts]
    card(s, L, 1.85, 7.35, 5.1)
    text(s, 0.9, 1.98, 6.9, 0.3, "DEVPOOL 160 KHUNG · THEO NGƯỠNG VERIFIER (BƯỚC 0,05)", 12, ORANGE, True)
    cats = [vn(t, 2) if i % 2 == 0 else "" for i, t in enumerate(ts)]
    FX, FY, FW, FH = 0.75, 2.6, 7.1, 3.6
    LAY = (0.08, 0.03, 0.9, 0.84)
    kit.line_chart(s, FX, FY, FW, FH, cats, [("người, k = n = 1", k1, ORANGE, False), ("người, k = 2, n = 3", k2, BLUE, True),
                                            ("hộp sai, k = n = 1", fp, SLATE, False)],
                   vmin=0, vmax=80, major=20, layout=LAY)
    px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
    xx = px + (7 + 0.5) / 20 * pw
    line(s, xx, py, xx, py + ph, PINK, 1.5, dash=True)
    text(s, xx + 0.06, py + 0.02, 1.4, 0.3, "chọn 0,35", 12, PINK, True)
    for lab, col, x in [("người, k = n = 1", ORANGE, 0.95), ("người, k = 2, n = 3", BLUE, 3.0), ("hộp sai, k = n = 1", SLATE, 5.25)]:
        kit.chip(s, x, 2.3, 1.95, 0.27, lab, col, 10)
    text(s, 0.9, 6.35, 6.9, 0.5, "Mọi ngưỡng 0,05–0,35 cho cùng 58 người và 2 hộp cal; luật lấy ngưỡng cao nhất trong số đó.", 12, MUTED)
    card(s, 8.12, 1.85, 4.6, 5.1)
    text(s, 8.35, 1.98, 4.2, 0.3, "BỐN ĐIỂM TRÊN LƯỚI", 12, ORANGE, True)
    rows = [["Điểm", "Cal", "Người", "Hộp sai"],
            ["chỉ detector", "15", "58", "79"],
            [("0,35 · k1 n1", {"bold": True, "color": ORANGE}), "2", "58", "54"],
            ["0,40 · k1 n1", "1", "57", "53"],
            ["0,05 · k2 n3", "0", "48", "33"]]
    kit.table(s, 8.3, 2.38, [1.55, 0.75, 0.95, 0.95], 0.42, rows, size=12, aligns="lrrr")
    G.eq(s, "d7", 8.3, 4.65, 0.72)
    text(s, 8.35, 5.05, 4.2, 1.85, ["720 điểm = 20 ngưỡng × (k, n) × max_gap; 708 điểm có cal ≤ 0,03.",
                                    "Lọc track k ≥ 2 bỏ nốt 2 hộp cal nhưng mất 10 người.",
                                    "3 điểm phát lại thật trùng CSV mô phỏng từng byte."], 12, TEXT, spacing=1.05)
    C.notes(s, ["Điểm vận hành chọn trên cal (150 khung không người) và devpool (218 người), không đụng test.",
                "Luật: báo nhầm cal ≤ 0,03 hộp mỗi khung, rồi lấy recall devpool lớn nhất; hoà thì lấy cấu hình đơn giản nhất.",
                "Chọn ngưỡng 0,35, k = n = 1: 2 hộp cal, giữ đủ 58 người như chỉ detector, bỏ 25 trong 79 hộp sai devpool.",
                "Lọc track k = 2, n = 3 sạch cal nhưng chỉ còn 48 người nên không chọn."],
            "results/v2/system/README.md §3; results/v2/system/calibrate.json.")
    return s


# ------------------------------------------------------------------ system results (t7)
def s_system_res():
    s = C.new("orange")
    C.header(s, "orange", "11 · KẾT QUẢ CẢ HỆ", "YOLO11s một tầng → ba tầng: mAP50 0,120 → 0,210, báo nhầm 0,881 → 0,289")
    sets = [("E2 ONNX", 0.120, 0.032, 0.197, 0.881, None, None, PINK),
            ("mốc COCO", 0.126, 0.030, 0.224, 0.597, None, None, SLATE),
            ("tầng 1", 0.119, 0.032, 0.197, 0.874, 0.333, 1.486, GOLD),
            ("tầng 2", 0.210, 0.122, 0.312, 0.289, 0.036, 0.631, ORANGE),
            ("tầng 3", 0.210, 0.122, 0.312, 0.289, 0.036, 0.631, TEAL)]
    for i, (ttl, vmax, step, getv, geterr) in enumerate([
            ("MAP50 · 198 KHUNG TEST · CI 95 %", 0.35, 0.1, lambda r: r[1], lambda r: (r[3] - r[1], r[1] - r[2])),
            ("BÁO NHẦM / KHUNG BAY TRỐNG · 159 KHUNG", 1.6, 0.4, lambda r: r[4], lambda r: ((r[6] - r[4], r[4] - r[5]) if r[5] is not None else (0, 0)))]):
        x = L + i * 3.62
        card(s, x, 1.85, 3.5, 3.6)
        text(s, x + 0.18, 1.97, 3.2, 0.5, ttl, 11, ORANGE, True)
        FX, FY, FW, FH = x + 0.05, 2.45, 3.4, 2.65
        LAY = (0.3 if i == 0 else 0.04, 0.02, 0.66 if i == 0 else 0.9, 0.9)
        vals = [getv(r) for r in sets]
        gf = kit.bar_chart(s, FX, FY, FW, FH, [r[0] for r in sets] if i == 0 else [""] * 5,
                           [("v", vals, SLATE, [vn(v) for v in vals], {k: r[7] for k, r in enumerate(sets)}, LP.INSIDE_BASE)],
                           horizontal=True, gap=40, vmin=0, vmax=vmax, major=step, grid=True, layout=LAY)
        ser = gf.chart.plots[0].series[0]
        kit._err_bars(ser, [geterr(r)[0] for r in sets], [geterr(r)[1] for r in sets])
        for k in range(5):
            for rr in ser.points[k].data_label.text_frame.paragraphs[0].runs:
                rr.font.color.rgb = kit.RGBColor.from_string("FFFFFF")
        px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
        xticks(s, px, py, pw, ph, vmax, step)
        if i == 1:  # short bars: move the label past the CI whisker so it is not crossed
            for k in (3, 4):
                for rr in ser.points[k].data_label.text_frame.paragraphs[0].runs:
                    rr.text = ""
                hi = sets[k][6]
                text(s, px + pw * hi / vmax + 0.05, py + ph * (k + 0.5) / 5 - 0.15, 0.7, 0.3, vn(sets[k][4]), 12, sets[k][7], True)
    card(s, 7.95, 1.85, 4.77, 3.6)
    text(s, 8.18, 1.97, 4.4, 0.3, "HIỆU GHÉP CẶP BA TẦNG − MỘT TẦNG, CÙNG MẪU KHỐI", 11, ORANGE, True)
    big = [("+0,090", "mAP50, CI [+0,026; +0,189]"), ("−0,591", "báo nhầm/khung, CI [−1,033; −0,273]"), ("0,396 = 0,396", "recall ở ngưỡng: 59/149 người")]
    for k, (b, t) in enumerate(big):
        y = 2.4 + k * 0.98
        text(s, 8.18, y, 4.4, 0.5, b, 24, ORANGE if k < 2 else TEAL, True)
        text(s, 8.18, y + 0.5, 4.4, 0.4, t, 12, TEXT)
    card(s, L, 5.6, W, 1.35)
    rows = [["Tầng (CSV của IOT-CV)", "mAP50 [CI 95 %]", "Recall @ ngưỡng", "Báo nhầm bay trống [CI]", "Trong nhà (300)", "Ngưỡng"],
            ["1 gói nhận diện", "0,119 [0,032; 0,197]", "0,396", "0,874 [0,333; 1,486]", "0,003", "0,077"],
            [("2 + gói xác minh", {"bold": True, "color": ORANGE}), "0,210 [0,122; 0,312]", "0,396", "0,289 [0,036; 0,631]", "0,003", "0,350"]]
    kit.table(s, 0.8, 5.68, [2.3, 2.15, 1.5, 2.35, 1.5, 1.05], 0.38, rows, size=12, aligns="lrrrrr")
    C.notes(s, ["Chấm một lần trên 198 khung test, cùng đường IOT-CV đọc JPEG của bản ghi, CI bootstrap theo khối 30 s.",
                "Tầng 1 trùng E2 ONNX một tầng. Verifier nâng mAP50 0,120 lên 0,210 và hạ báo nhầm 0,881 xuống 0,289 hộp mỗi khung bay trống.",
                "Hai hiệu ghép cặp có CI không chạm 0; recall ở ngưỡng giữ nguyên 0,396, không mất người nào.",
                "Tầng 3 trùng tầng 2 từng hộp vì k = 1; báo nhầm trong nhà 0,003 hộp mỗi khung."],
            "results/reproduce/tables/t7_v2_nhan_dien.md; results/v2/system/README.md §4 (score.json).")
    return s


# ------------------------------------------------------------------ real frames with boxes
def s_boxes():
    s = C.new("orange")
    C.header(s, "orange", "11 · KẾT QUẢ CẢ HỆ", "Trên khung test, tầng 2 bỏ hộp sai ở rác hồng, giữ đủ 4 người")
    card(s, L, 1.85, 6.0, 5.1)
    card(s, 6.72, 1.85, 6.0, 5.1)
    text(s, 0.85, 1.97, 5.6, 0.3, "KHUNG 002_00005 · KHÔNG CÓ NGƯỜI", 12, ORANGE, True)
    for k, (t, lab) in enumerate([("t1", "tầng 1: 8 hộp sai"), ("t2", "tầng 2: 1 hộp sai")]):
        x = 0.82 + k * 2.83
        G.framed(s, IMG + f"box_002_00005_{t}.jpg", x, 2.35, 2.72, 2.04)
        text(s, x, 4.45, 2.72, 0.3, lab, 13, INK, True, align="c")
    text(s, 0.85, 4.95, 5.6, 1.9, ["YOLO11s nhầm túi, tấm nhựa hồng và đá vụn trên đường đất là người.",
                                   "Verifier xem lại vùng cắt rộng 1,5 lần quanh từng hộp và bỏ 7 trong 8 hộp.",
                                   "Cả 159 khung bay trống: 139 hộp → 46 hộp."], 13, TEXT, spacing=1.05)
    text(s, 6.95, 1.97, 5.6, 0.3, "KHUNG 005_00809 · 4 NGƯỜI (PHÓNG 2×)", 12, ORANGE, True)
    for k, (t, lab) in enumerate([("t1", "tầng 1: 4 đúng, 5 sai"), ("t2", "tầng 2: 4 đúng, 2 sai")]):
        y = 2.35 + k * 1.95
        G.framed(s, IMG + f"boxz_005_00809_{t}.jpg", 6.92, y, 3.0, 1.69)
        text(s, 10.05, y + 0.6, 2.5, 0.5, lab, 13, INK, True)
    for k, (col, lab) in enumerate([(GREEN, "đúng: IoU ≥ 0,5 với nhãn"), (PINK, "sai"), ("8CA0B3", "khung trắng: nhãn tay")]):
        kit.chip(s, 6.95 + k * 1.88, 6.38, 1.78, 0.3, lab, col, 10)
    C.notes(s, ["Hai khung test minh hoạ, vẽ ở đúng ngưỡng vận hành: tầng 1 ngưỡng 0,0767, tầng 2 thêm verifier 0,35.",
                "Khung không người: 8 hộp sai trên rác hồng và đá còn 1.",
                "Khung có 4 người: giữ cả 4 hộp đúng, hộp sai 5 còn 2.",
                "Kết luận dựa trên số của 198 khung ở slide trước, ảnh chỉ để minh hoạ."],
            "results/v2/system/preds/system_detector_test.csv, system_verifier_test.csv; data/real/labels/test.")
    return s


# ------------------------------------------------------------------ classic -> ONNX
def s_classic_onnx():
    s = C.new("purple")
    C.header(s, "purple", "12 · LỌC CỔ ĐIỂN", "Classic thành ONNX 703 nút, lệch numpy MAE ≤ 0,018")
    card(s, L, 1.85, 6.4, 5.1)
    text(s, 0.85, 1.97, 6.0, 0.3, "PHÉP CỦA CLASSIC → TRONG GRAPH (OPSET 17, FP32)", 12, PURPLE, True)
    rows = [["numpy / cv2", "ONNX", "Vì sao"],
            ["cv2 uint8 có bão hoà", "float, Round, Clip", "làm tròn như cvRound"],
            ["np.percentile 99,9 / 0,5", "TopK + nội suy", "opset 17 không có quantile"],
            ["if mức đen < sàn", "Where", "không có if lúc trace"],
            ["erode / dilate", "MaxPool", "biên như cv2"],
            ["blur, resize ×1/4", "AveragePool + Gather", "đệm phản chiếu, shape tĩnh"],
            ["GaussianBlur", "lát dịch có trọng số", "tránh TF32 của Conv"]]
    kit.table(s, 0.85, 2.35, [2.15, 1.95, 1.95], 0.4, rows, size=12, aligns="lll")
    G.eq(s, "mae", 0.85, 5.3, 0.72)
    text(s, 4.15, 5.25, 2.75, 0.95, ["Không If, Loop, Conv, MatMul trong graph.", "fp16 không dùng: 5 865 chỉ có bước 4,0."], 12, TEXT, spacing=1.05)
    text(s, 0.85, 6.25, 6.0, 0.6, "50 khung dev, khác khung trace: cổng khử hồng mở ở 46, đóng ở 4; test CI dựng khung tổng hợp phủ cả hai phía của hai nhánh.", 12, MUTED)
    card(s, 7.17, 1.85, 5.55, 2.6)
    text(s, 7.4, 1.97, 5.1, 0.3, "ĐỦ GRAPH, KHUNG NGUYÊN → KHUNG NGUYÊN", 12, PURPLE, True)
    prow = [["So", "MAE TB", "MAE tệ nhất", "Lệch max"],
            ["ORT CPU ↔ numpy", "0,0020", "0,0179", "8"],
            ["ORT CUDA ↔ numpy", "0,0012", "0,0100", "8"],
            ["CUDA ↔ CPU", "0,0013", "0,0155", "2"]]
    kit.table(s, 7.4, 2.35, [1.95, 1.0, 1.2, 0.95], 0.42, prow, size=12, aligns="lrrr")
    card(s, 7.17, 4.6, 5.55, 2.35)
    text(s, 7.4, 4.72, 5.1, 0.3, "198 KHUNG TEST · GỐC → CLASSIC ONNX", 12, PURPLE, True)
    cats = ["% điểm hồng", "dark channel TB"]
    kit.bar_chart(s, 7.25, 5.05, 5.4, 1.85, cats,
                  [("gốc", [100, 100], SLATE, ["0,943 %", "140,2"], None, LP.INSIDE_END),
                   ("classic", [10.8, 49.8], PURPLE, ["0,102 %", "69,8"], None, LP.OUTSIDE_END)],
                  horizontal=True, gap=40, overlap=-10, vmin=0, vmax=118, layout=(0.3, 0.02, 0.66, 0.94))
    C.notes(s, ["Bộ lọc numpy vẫn là nguồn chuẩn; graph ONNX dựng lại đúng năm tầng theo cùng thứ tự và cùng tham số.",
                "Mỗi phép cv2 đổi sang phép ONNX tương đương để giữ từng lần làm tròn uint8.",
                "Trên 50 khung dev, lệch MAE ≤ 0,018 thang 0–255; golden CUDA và CPU đều đạt dung sai của IOT-CV.",
                "Trên 198 khung test: điểm hồng giảm 89,2 %, dark channel trung bình giảm 50,2 %; nhấp nháy 2,18 bằng dev."],
            "results/v2/classic/README.md §1–4.1 (export.json, meta.json metrics_test).")
    return s


# ------------------------------------------------------------------ enhancement method comparison
def s_enh_cmp():
    s = C.new("purple")
    C.header(s, "purple", "13 · SO SÁNH LÀM RÕ", "Sáu cách làm rõ: classic 19,3 ms, SwinIR 17,4 s mỗi khung")
    card(s, L, 1.85, W, 3.3)
    rows = [["Phương pháp", "Mô hình", "ms / khung (ORT CUDA)", "Đo trên khung thật", "Lý do chọn hay không"],
            [("classic ONNX", {"bold": True, "color": PURPLE}), "5 tầng lọc", "19,3 / 25,3", "hồng −89 %, dark −50 %", "chạy trực tiếp: thắng gốc 25/25"],
            ["NAFNet-w16", "1,14 triệu", "20,6 / 27,9", "dark −37 % (test)", "khử mù ít hơn, chậm hơn 1,3 ms"],
            ["lai classic → NAFNet", "1,14 triệu", "chưa đo", "dark 56,0 → 111,5", "học theo D0 nên trả lại màn mờ"],
            ["Real-ESRGAN ×2 tinh chỉnh", "16,7 triệu", "404 / 526", "nhấp nháy 1,35; cal 0,067", "offline; A/B 16/25 với classic"],
            ["SwinIR-M ×2", "11,7 triệu", "17 400", "chưa chấm", "32,6 giờ GPU cho chuyến 20 phút"],
            ["BasicVSR++ (video)", "DeformConv", "3 340 (PyTorch)", "chưa chấm", "ORT không có kernel DeformConv"]]
    kit.table(s, 0.85, 1.98, [2.6, 1.3, 2.15, 2.6, 3.0], 0.42, rows, size=12, aligns="llrll")
    card(s, L, 5.3, 6.6, 1.65)
    text(s, 0.85, 5.4, 6.2, 0.3, "MS MỖI KHUNG, TRỤC LOG", 11, PURPLE, True)
    cats = ["classic", "NAFNet", "Real-ESRGAN", "BasicVSR++", "SwinIR"]
    vals = [19.3, 20.6, 404, 3340, 17400]
    gf = kit.bar_chart(s, 0.7, 5.62, 6.45, 1.3, cats,
                       [("ms", vals, SLATE, ["19,3", "20,6", "404", "3 340", "17 400"],
                         {0: PURPLE, 1: TEAL, 2: BLUE, 3: GOLD, 4: PINK}, LP.OUTSIDE_END)],
                       horizontal=False, gap=30, vmin=1, vmax=100000, layout=(0.02, 0.24, 0.96, 0.5))
    kit.log_axis(gf)
    card(s, 7.37, 5.3, 5.35, 1.65)
    G.eq(s, "srloss", 7.55, 5.42, 0.62)
    text(s, 7.55, 5.95, 5.0, 0.95, "Real-ESRGAN tinh chỉnh L1 + perceptual VGG19, 2 481 bước. Bản thêm GAN chỉ chạy 206 bước vì tràn VRAM.", 12, TEXT)
    C.notes(s, ["Sáu cách làm rõ được đo trên cùng máy RTX 4060 Laptop; classic ONNX nhanh nhất và được chọn cho luồng trực tiếp.",
                "NAFNet chậm hơn classic 1,3 ms p50 khi đo xen kẽ và khử mù ít hơn.",
                "Mạng lai sau classic học theo đích D0 nên trả lại màn mờ: dark channel 56,0 lên 111,5.",
                "Real-ESRGAN ×2 đủ nhanh cho offline (0,76 giờ mỗi chuyến 20 phút); SwinIR quá chậm; BasicVSR++ không chạy được trên ORT."],
            "results/v2/classic/README.md §3, §5; results/v2/sr/README.md §1–4; results/v2/vsr/README.md; t9_v2_ms.md.")
    return s


# ------------------------------------------------------------------ images: raw / classic / offline
def s_enh_imgs():
    s = C.new("purple")
    C.header(s, "purple", "13 · SO SÁNH LÀM RÕ", "Cùng khung 003_02415: gốc, classic và Real-ESRGAN ×2")
    labs = [("ab_raw.jpg", "Gốc", "hồng 0,943 %, dark 140,2 (test)", SLATE),
            ("ab_classic.jpg", "Classic ONNX", "19,3 ms; hồng −89 %, dark −50 %", PURPLE),
            ("ab_offline.jpg", "Real-ESRGAN ×2 (offline)", "0,404 s; màu ngả lạnh, nhạt", BLUE)]
    for k, (f, t, sub, col) in enumerate(labs):
        x = L + k * 4.07
        card(s, x, 1.85, 3.93, 5.1)
        G.framed(s, IMG + f, x + 0.14, 1.99, 3.65, 2.81)
        G.framed(s, IMG + f.replace(".jpg", "_z.jpg"), x + 0.14, 4.9, 3.65, 1.25)
        text(s, x + 0.14, 6.22, 3.65, 0.3, t, 14, col, True)
        text(s, x + 0.14, 6.52, 3.65, 0.3, sub, 12, TEXT)
    C.notes(s, ["Một khung dev ở 15,5 m có người, lấy từ trang A/B mù: trái gốc, giữa classic, phải bản offline ×2.",
                "Hàng dưới phóng quanh người. Classic xoá phần lớn vệt hồng và khử mù; bản offline xoá hồng nhưng màu ngả lạnh.",
                "Số trên ảnh lấy từ 198 khung test (classic) và bench offline 20 khung dev.",
                "Ảnh chỉ minh hoạ; quyết định dựa trên A/B mù ở slide sau."],
            "results/ab-v2/ab_key.json (khung 20261102-100941_003_02415); results/v2/classic/README.md §4.1; results/v2/sr/README.md §3–4.")
    return s


# ------------------------------------------------------------------ blind A/B + guard
def s_ab():
    s = C.new("purple")
    C.header(s, "purple", "14 · A/B MÙ", "A/B mù: classic thắng gốc 25/25, offline ×2 và classic chưa tách được (16/25)")
    from pptx.util import Inches, Pt
    for x in s.shapes:
        if x.shape_id == 5:  # long title: one line at a smaller size
            for rr in x.text_frame.paragraphs[0].runs:
                rr.font.size = Pt(21)
            x.width = Inches(12.3)
    card(s, L, 1.85, 7.2, 5.1)
    text(s, 0.9, 1.98, 6.7, 0.3, "TỈ LỆ BÊN THỨ NHẤT THẮNG · 25 CẶP · THANH ĐEN: WILSON 95 %", 12, PURPLE, True)
    rows = [("classic vs gốc", 1.0, 0.867, 1.0, PURPLE), ("offline ×2 vs gốc", 1.0, 0.867, 1.0, BLUE),
            ("offline ×2 vs classic", 0.64, 0.445, 0.798, GOLD)]
    FX, FY, FW, FH = 0.75, 2.35, 6.95, 2.7
    LAY = (0.3, 0.02, 0.64, 0.86)
    gf = kit.bar_chart(s, FX, FY, FW, FH, [r[0] for r in rows],
                       [("p", [r[1] for r in rows], TEAL, ["25/25", "25/25", "16/25"], {i: r[4] for i, r in enumerate(rows)}, LP.INSIDE_BASE)],
                       horizontal=True, gap=55, vmin=0, vmax=1.0, layout=LAY, grid=True, major=0.25)
    ser = gf.chart.plots[0].series[0]
    kit._err_bars(ser, [r[3] - r[1] for r in rows], [r[1] - r[2] for r in rows])
    for i in range(3):
        for r in ser.points[i].data_label.text_frame.paragraphs[0].runs:
            r.font.color.rgb = kit.RGBColor.from_string("FFFFFF")
    px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
    xx = px + 0.5 * pw
    line(s, xx, py, xx, py + ph, PINK, 2.0, dash=True)
    for k in range(5):
        v = k * 0.25
        text(s, px + v * pw - 0.3, py + ph + 0.03, 0.6, 0.25, vn(v, 2) if 0 < v < 1 else str(int(v)), 11, MUTED, align="c")
    text(s, 0.9, 5.25, 2.2, 0.3, "Wilson, z = 1,96:", 12, MUTED, True)
    G.eq(s, "wilson", 0.9, 5.55, 0.72)
    text(s, 3.85, 5.3, 3.8, 1.6, ["Cận dưới > 0,5: bên thứ nhất thắng (vạch hồng).",
                                  "classic vs gốc: [0,867; 1,000]; offline vs classic: [0,445; 0,798].",
                                  "5 cặp lặp đổi bên: 5/5 chọn như lần đầu."], 12, TEXT, spacing=1.05)
    card(s, 8.0, 1.85, 4.72, 2.7)
    text(s, 8.22, 1.98, 4.3, 0.3, "HÀNG RÀO BÁO NHẦM · 150 KHUNG CAL", 12, PURPLE, True)
    grows = [["Khung vào detector", "Hộp", "/ khung"], ["gốc", "15", "0,100"], ["offline ×2", "10", "0,067"]]
    kit.table(s, 8.22, 2.38, [2.1, 0.9, 1.2], 0.4, grows, size=12, aligns="lrr")
    text(s, 8.22, 3.7, 4.3, 0.8, [[("hiệu −0,033 [−0,123; +0,039]: ", {"bold": True, "color": INK}), ("cận dưới ≤ 0, qua", {})],
                                  "16 khối, 2 phiên; YOLO11s ở ngưỡng 0,0767."], 12, TEXT, spacing=1.05)
    card(s, 8.0, 4.7, 4.72, 2.25)
    text(s, 8.22, 4.82, 4.3, 0.3, "CÁCH CHẤM", 12, PURPLE, True)
    text(s, 8.22, 5.15, 4.3, 1.75, ["80 cặp: 3 × 25 + 5 cặp lặp, trái phải xáo, tên file ngẫu nhiên.",
                                    "Cả khung, JPEG q95 4:4:4, cắt dải mép 16 cột.",
                                    "Kết luận: web dùng classic; offline ×2 không bật."], 12, TEXT, spacing=1.05)
    C.notes(s, ["Trang A/B offline, người chấm không biết bên nào là gì; mỗi so sánh 25 cặp lấy theo tầng độ cao và có người.",
                "Classic thắng gốc 25/25, Wilson [0,867; 1,000]; offline thắng gốc 25/25.",
                "Offline thắng classic 16/25, cận dưới Wilson 0,445 dưới 0,5 nên không phân biệt được: chọn classic vì nhanh hơn 20 lần.",
                "Hàng rào: ảnh offline không làm detector báo nhầm nhiều hơn trên cal (15 → 10 hộp)."],
            "results/reproduce/tables/t8_v2_ab.md; results/ab-v2/README.md, ab_score.json, guard_fp.json, decision.json.")
    return s


# ------------------------------------------------------------------ IOT-CV integration, ABBA ms
def s_iotcv():
    s = C.new("end")
    C.header(s, "end", "15 · TÍCH HỢP IOT-CV", "Đo xen kẽ ABBA trong IOT-CV: p95 có Ảnh rõ 99,1 → 89,5 ms")
    card(s, L, 1.85, 6.3, 5.1)
    text(s, 0.85, 1.97, 6.0, 0.3, "P95 MS · NHỊP THẬT, CÓ NGƯỜI XEM ẢNH RÕ · 600 KHUNG", 12, BLUE, True)
    cats = ["detector", "verifier", "enhancer", "tổng"]
    v1 = [63.3, 0, 34.4, 99.1]; v2 = [63.4, 8.9, 19.3, 89.5]
    kit.bar_chart(s, 0.7, 2.3, 6.15, 3.2, cats,
                  [("một tầng + NAFNet", v1, "9FB3C8", ["63,3", "", "34,4", "99,1"], None, LP.OUTSIDE_END),
                   ("ba tầng + bộ lọc cổ điển", v2, BLUE, ["63,4", "8,9", "19,3", "89,5"], None, LP.OUTSIDE_END)],
                  horizontal=True, gap=45, vmin=0, vmax=115, layout=(0.18, 0.02, 0.76, 0.94))
    kit.chip(s, 1.0, 5.55, 2.2, 0.28, "một tầng + NAFNet", "9FB3C8", 11)
    kit.chip(s, 3.35, 5.55, 3.0, 0.28, "ba tầng + bộ lọc cổ điển", BLUE, 11)
    text(s, 0.85, 5.95, 5.9, 0.95, ["4 lượt A-B-B-A trong một tiến trình, bản ghi 20261102-100941 khung 284–884.",
                                    "GPU hạ xung 780–802 MHz, 70–72 °C; khung bỏ 7,8 · 8,2 % → 4,2 · 4,7 %."], 12, MUTED, spacing=1.05)
    card(s, 7.07, 1.85, 5.65, 2.55)
    text(s, 7.3, 1.97, 5.2, 0.3, "BA GÓI ONNX · GOLDEN CUDA VÀ CPU", 12, BLUE, True)
    rows = [["Gói ONNX", "sha256", "Golden"],
            ["nhận diện", "23b1da69…aef", "IoU ≥ 0,9998"],
            ["xác minh", "d566343a…d58", "Δ điểm ≤ 0,0003"],
            ["làm rõ", "83d96e8b…77b", "lệch max ≤ 2"]]
    kit.table(s, 7.3, 2.35, [1.85, 1.75, 1.6], 0.42, rows, size=12, aligns="lll")
    card(s, 7.07, 4.55, 5.65, 2.4)
    text(s, 7.3, 4.68, 5.2, 0.3, "TRANG CAMERA & AI CỦA WEB", 11, BLUE, True)
    text(s, 7.3, 5.05, 5.2, 1.8, ["Cả ba model báo đang chạy trên CUDA, kèm ms p50/p95.",
                                  "Ảnh rõ classic bật sẵn; nút ảnh rõ offline tắt kèm lý do, API offline trả 503.",
                                  "Test tích hợp IOT-CV: 7 đạt, 1 bỏ qua (offline)."], 14, TEXT, spacing=1.1)
    C.notes(s, ["IOT-CV nạp ba gói ONNX; mỗi gói qua golden trên cả CUDA và CPU trước khi nạp.",
                "Bench ABBA: A = một tầng + NAFNet; B = ba tầng + bộ lọc cổ điển ONNX, xen kẽ để triệt lệch do xung GPU.",
                "Luồng có Ảnh rõ: p95 tổng 99,1 → 89,5 ms; classic nhẹ hơn NAFNet khoảng 15 ms, bù phần verifier 8,9 ms.",
                "Số đo lúc GPU hạ xung 780 MHz, ghi kèm điều kiện; lệch giữa hai lượt cùng cấu hình ≤ 3,4 ms."],
            "results/reproduce/tables/t9_v2_ms.md; results/v2/iotcv/README.md §1–3, §6.")
    return s


# ------------------------------------------------------------------ conclusion
def s_conclusion():
    s = C.new("end")
    C.header(s, "end", "KẾT LUẬN", "Ba tầng bớt 67 % báo nhầm, Ảnh rõ classic bật sẵn trên web")
    card(s, L, 1.85, 7.25, 5.1)
    text(s, 0.9, 2.0, 6.8, 0.3, "CÂU HỎI 1 · TẦNG XÁC MINH", 12, ORANGE, True)
    text(s, 0.9, 2.32, 3.3, 0.7, "0,881 → 0,289", 30, ORANGE, True)
    text(s, 4.3, 2.4, 3.4, 0.8, "hộp báo nhầm mỗi khung bay trống, CI hiệu [−1,033; −0,273]", 13, TEXT)
    text(s, 0.9, 3.15, 6.8, 0.6, "mAP50 0,120 → 0,210 (+0,090 [+0,026; +0,189]), recall giữ 0,396; verifier p95 ~9 ms.", 13, TEXT)
    text(s, 0.9, 3.95, 6.8, 0.3, "CÂU HỎI 2 · CÁCH LÀM RÕ", 12, PURPLE, True)
    text(s, 0.9, 4.27, 3.3, 0.7, "25/25", 30, PURPLE, True)
    text(s, 4.3, 4.35, 3.4, 0.8, "lượt A/B mù classic thắng gốc, Wilson [0,867; 1,000]", 13, TEXT)
    text(s, 0.9, 5.1, 6.8, 1.75, ["Classic ONNX 19,3 ms nhanh hơn Real-ESRGAN ×2 khoảng 20 lần; A/B cả khung không tách được hai bản.",
                                  "Trong IOT-CV, p95 tổng có Ảnh rõ 99,1 → 89,5 ms; golden CUDA và CPU đạt."], 13, TEXT, spacing=1.05)
    card(s, 8.07, 1.85, 4.65, 5.1)
    text(s, 8.35, 2.0, 4.1, 0.35, "Việc tiếp theo", 18, BLUE, True)
    steps = ["Bay một buổi bãi trống để có tập khung âm giữ kín, độc lập với test.",
             "Chấm lại trò C3 và v1 trên tập sạch với cả hai trước khi đổi detector.",
             "Bỏ âm d02x VisDrone có người không nhãn (soát tay: 6 % [3,5; 10,2]).",
             "MobileNetV3-S 64 (2,93 ms) nếu cần thêm ms cho luồng chỉ nhận diện."]
    for k, st in enumerate(steps):
        y = 2.55 + k * 1.08
        text(s, 8.35, y, 0.4, 0.4, str(k + 1), 20, BLUE, True)
        text(s, 8.8, y + 0.03, 3.75, 1.0, st, 13, TEXT)
    C.notes(s, ["Câu hỏi 1: verifier giảm báo nhầm trên khung bay trống 67 % và nâng mAP50 0,090, cả hai CI không chạm 0, không mất người.",
                "Câu hỏi 2: classic ONNX được người chấm mù chọn hơn gốc 25/25 và nhanh hơn bản offline khoảng 20 lần.",
                "IOT-CV chạy ba bundle; p95 tổng có Ảnh rõ giảm 9,6 ms so với v1 cùng lúc đo.",
                "Phạm vi: một buổi bay, 39 khung test có người. Việc tiếp theo là tập âm độc lập và tập chấm sạch để so detector."],
            "results/v2/README.md; t7_v2_nhan_dien.md; t8_v2_ab.md; t9_v2_ms.md; results/v2/verifier/README.md §8.")
    return s


# ------------------------------------------------------------------ references
REFS = [
    "Phạm Văn Cường, Bài giảng Xử lý ảnh, chương 1–9, Khoa Trí tuệ Nhân tạo, PTIT, 2026.",
    "R. C. Gonzalez, R. E. Woods, Digital Image Processing, 4th ed., Pearson, 2018.",
    "G. Jocher, J. Qiu, Ultralytics YOLO11, gói ultralytics 8.4, 2024.",
    "M. Tan, Q. Le, EfficientNet: Rethinking Model Scaling for CNNs, ICML 2019.",
    "K. He, G. Gkioxari, P. Dollár, R. Girshick, Mask R-CNN (RoIAlign), ICCV 2017.",
    "K. He, J. Sun, X. Tang, Single Image Haze Removal Using Dark Channel Prior, IEEE TPAMI 33(12), 2011.",
    "K. He, J. Sun, X. Tang, Guided Image Filtering, IEEE TPAMI 35(6), 2013.",
    "L. Chen, X. Chu, X. Zhang, J. Sun, Simple Baselines for Image Restoration, ECCV 2022.",
    "X. Wang, L. Xie, C. Dong, Y. Shan, Real-ESRGAN, ICCV Workshops 2021.",
    "J. Liang và cộng sự, SwinIR: Image Restoration Using Swin Transformer, ICCV Workshops 2021.",
    "K. C. K. Chan và cộng sự, BasicVSR++, CVPR 2022.",
    "J. Johnson, A. Alahi, L. Fei-Fei, Perceptual Losses for Real-Time Style Transfer, ECCV 2016.",
    "P. Zhu, L. Wen, D. Du và cộng sự, Detection and Tracking Meet Drones Challenge, IEEE TPAMI 44(11), 2022.",
    "D. Božić-Štulić, Ž. Marušić, S. Gotovac, Deep Learning Approach in Aerial Imagery for SAR (HERIDAL), IJCV 2019.",
    "X. Yu và cộng sự, Scale Match for Tiny Person Detection (TinyPerson), WACV 2020.",
    "D. Broyles và cộng sự, WiSARD: A Labeled Visual and Thermal Image Dataset for SAR, IROS 2022.",
    "A. Radford và cộng sự, Learning Transferable Visual Models From Natural Language Supervision (CLIP), ICML 2021.",
    "D. Lopez-Paz, M. Oquab, Revisiting Classifier Two-Sample Tests, ICLR 2017.",
    "M. Bińkowski và cộng sự, Demystifying MMD GANs, ICLR 2018; M. Oquab và cộng sự, DINOv2, TMLR 2024.",
    "Y. Zhang và cộng sự, ByteTrack, ECCV 2022.",
    "N. Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772, 2016.",
    "E. B. Wilson, Probable Inference, the Law of Succession, and Statistical Inference, JASA 22, 1927.",
    "ISO/IEC 10918-1, Digital compression and coding of continuous-tone still images, phụ lục K, 1994.",
    "ONNX Runtime developers, ONNX Runtime (onnxruntime-gpu 1.24.4), Microsoft.",
]


def s_refs():
    s = C.new("end")
    C.header(s, "end", "TÀI LIỆU THAM KHẢO", "Tài liệu tham khảo")
    half = (len(REFS) + 1) // 2
    for c in range(2):
        card(s, L + c * 6.15, 1.85, 5.95, 5.1)
        lines = [[(f"[{i + 1}] ", {"bold": True, "color": BLUE}), (REFS[i], {})] for i in range(c * half, min(len(REFS), (c + 1) * half))]
        text(s, L + c * 6.15 + 0.22, 1.98, 5.5, 4.9, lines, 12, TEXT, spacing=1.0)
    C.notes(s, ["Tài liệu cho lý thuyết, model và bộ dữ liệu dùng trong đồ án.",
                "Mã nguồn và số liệu tái lập: repo XLA, scripts/reproduce.py."],
            "docs/nop-bai-src/latex/sections/09-tai-lieu.tex; results/v2/data/README.md §1.")
    return s
