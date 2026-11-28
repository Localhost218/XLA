"""v8 slides for outline sections 11-20 (comparison, enhancement, E4, integration, decision, discussion, end)."""
from pptx.enum.chart import XL_LABEL_POSITION as LP

import common as C
import kit
import stage as G
from kit import BLUE, GOLD, INK, MUTED, ORANGE, PINK, PURPLE, SLATE, TEAL, TEXT, card, line, text, vn
from slides1 import L, W, gap_arrows, stage_row

RPT = G.RPT + "/"
T1 = [  # name, mAP50, lo, hi, mAP50-95, AR small, FP, thr, colour
    ("Mốc YOLO11s COCO @1280", 0.126, 0.030, 0.224, 0.063, 0.531, 0.597, 0.068, SLATE),
    ("E1 D0 ảnh sạch", 0.297, 0.036, 0.408, 0.100, 0.519, 1.195, 0.053, GOLD),
    ("E1 D1 ngẫu nhiên", 0.143, 0.083, 0.185, 0.047, 0.444, 0.994, 0.023, BLUE),
    ("E1 D2 bản sao camera", 0.236, 0.069, 0.370, 0.079, 0.426, 0.302, 0.089, TEAL),
    ("E1 D2 + siêu phân giải", 0.149, 0.011, 0.282, 0.048, 0.440, 1.956, 0.059, PURPLE),
    ("YOLO11m (thầy dự kiến)", 0.218, 0.107, 0.349, 0.060, 0.425, 0.390, 0.054, MUTED),
    ("E2 trò vòng 1", 0.156, 0.036, 0.235, 0.052, 0.419, 0.868, 0.075, ORANGE),
    ("E2 trò vòng 2", 0.313, 0.035, 0.472, 0.157, 0.446, 0.522, 0.018, ORANGE),
    ("E2 vòng 1, gói ONNX", 0.120, 0.032, 0.197, 0.042, 0.240, 0.881, 0.077, PINK),
]


# ------------------------------------------------------------------ §11 A1: chart + full table, rows aligned
def s_a1():
    s = C.S[11]
    C.clear(s)
    C.header(s, "orange", "09 · SO SÁNH NHẬN DIỆN", "Chín cấu hình huấn luyện YOLO11s: vòng 2 cao nhất (0,313), CI chồng nhau")
    card(s, L, 1.85, W, 5.1)
    text(s, 0.9, 1.98, 6.0, 0.3, "MAP50 · 198 KHUNG TEST · THANH ĐEN LÀ CI 95 %", 12, ORANGE, True)
    FX, FY, FW, FH = 0.75, 2.5, 6.75, 4.1
    LAY = (0.43, 0.0, 0.53, 0.93)
    gf = kit.bar_chart(s, FX, FY, FW, FH, [r[0] for r in T1],
                       [("mAP50", [r[1] for r in T1], SLATE, [vn(r[1]) for r in T1], {i: r[8] for i, r in enumerate(T1)})],
                       horizontal=True, gap=40, vmin=0, vmax=0.5, major=0.1, grid=True, layout=LAY)
    ser = gf.chart.plots[0].series[0]
    kit._err_bars(ser, [r[3] - r[1] for r in T1], [r[1] - r[2] for r in T1])
    for p_ in range(len(T1)):
        ser.points[p_].data_label.position = LP.INSIDE_BASE
        for r in ser.points[p_].data_label.text_frame.paragraphs[0].runs:
            r.font.color.rgb = kit.RGBColor.from_string("FFFFFF")
    px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
    for k in range(6):
        v = k / 10
        text(s, px + v / 0.5 * pw - 0.3, py + ph + 0.04, 0.6, 0.25, vn(v, 1) if k else "0", 11, MUTED, align="c")
    rh = ph / 9
    rows = [["CI 95 %", "mAP50-95", "AR nhỏ", "Báo nhầm", "Ngưỡng"]]
    for r in T1:
        fp = (vn(r[6]), {"bold": True, "color": TEAL}) if r[6] == 0.302 else vn(r[6])
        rows.append([f"[{vn(r[2])}; {vn(r[3])}]", vn(r[4]), vn(r[5]), fp, vn(r[7])])
    kit.table(s, 7.6, py - rh, [1.7, 0.9, 0.7, 0.8, 0.9], rh, rows, size=12, aligns="rrrrr")
    text(s, 7.6, py + ph + 0.04, 5.0, 0.3, "Báo nhầm: hộp mỗi khung bay không người. Ngưỡng: F1 lớn nhất trên dev-val.", 11, MUTED)
    C.notes(s, ["Đủ chín dòng của bảng t1; mốc là YOLO11s COCO ở 1 280 px.",
                "mAP50 cao nhất: E2 vòng 2 0,313, nhưng CI [0,035; 0,472] chồng lên gần hết các bộ khác.",
                "Báo nhầm thấp nhất: E1 D2 0,302 hộp mỗi khung; AR nhỏ là recall trên người nhỏ (diện tích < 32²).",
                "Gói ONNX nhận diện là E2 vòng 1, chấm bằng đúng đường ONNX: 0,120."],
            "results/reproduce/tables/t1_nhan_dien.md; results/baseline/pred/run_meta.json (yolo11s, imgsz 1280).")
    return s


def s_keep_header(s, part, badge, title=None, notes=None):
    sh = {x.shape_id: x for x in s.shapes}
    C.header(s, part, badge, title or sh[5].text_frame.text)
    if notes:
        C.notes(s, *notes)
    return s


# ------------------------------------------------------------------ §12 classic chain
def s_classic():
    s = C.new("purple")
    C.header(s, "purple", "12 · LỌC CỔ ĐIỂN", "Bộ lọc cổ điển đi ngược chuỗi hỏng qua năm tầng")
    G.panel(s, L, 1.85, 1.5, 2.3, None, "Đầu vào")
    G.node(s, 0.72, 2.3, 1.3, 0.72, ["khung gốc I", "BGR uint8"], "data", 11)
    G.node(s, 0.72, 3.18, 1.3, 0.72, ["f = 1 016,1 px", "từ hồ sơ"], "frozen", 11)
    specs = [("yel", "1. Khử khối"), ("peach", "2. Khử hồng"), ("lav", "3. Bù tối góc"), ("blu", "4. Khử mù, làm nét")]
    xs = stage_row(s, 2.27, 1.85, 2.3, [2.5] * 4, 0.12, specs)
    G.node(s, xs[0] + 0.15, 2.3, 2.2, 0.48, "lưới 8×8: biên dọc rồi ngang", "learn", 11)
    G.eq(s, "cl_deblock", xs[0] + 1.25, 2.95, 0.66, align="c")
    G.node(s, xs[1] + 0.15, 2.3, 2.2, 0.48, ["cổng: max(R,B) ≥ 230", "và min(R,B) − G ≥ 6"], "learn", 10)
    G.node(s, xs[1] + 0.15, 2.9, 2.2, 0.48, "trừ phần dư hồng δ, 2 lượt", "learn", 11)
    G.arrow(s, xs[1] + 1.25, 2.78, xs[1] + 1.25, 2.9, INK, 1.25)
    G.eq(s, "cl_vig", xs[2] + 1.25, 2.4, 0.6, align="c")
    text(s, xs[2] + 0.12, 2.92, 2.26, 0.5, "chỉ cần nhân, không chia", 11, TEXT, align="c")
    chain = [("dark", 0.56), ("A", 0.32), ("t̃", 0.32), ("GF", 0.42), ("J", 0.32)]
    nx = xs[3] + 0.12
    for k, (lab, cw) in enumerate(chain):
        G.node(s, nx, 2.35, cw, 0.42, lab, "learn", 11)
        if k < 4:
            G.arrow(s, nx + cw, 2.56, nx + cw + 0.08, 2.56, INK, 1.0)
        nx += cw + 0.08
    G.node(s, xs[3] + 0.15, 2.95, 2.2, 0.42, "làm nét Y, nén lõi nhiễu", "plain", 11)
    bars = ["biên khối 2,73 → 2,08", "điểm hồng −91,6 %", "góc sáng lên ×1,32", "dark 142,7 → 65,3"]
    for x, pal, b in zip(xs, ["yel", "peach", "lav", "blu"], bars):
        G.bar(s, x + 0.12, 3.62, 2.26, 0.38, pal, b, 12)
    gap_arrows(s, [(L + 1.5, xs[0])] + [(xs[i] + 2.5, xs[i + 1]) for i in range(3)], 2.54)
    card(s, L, 4.3, W, 1.3)
    G.eq(s, "cl_gate", 0.85, 4.38, 0.78)
    G.eq(s, "cl_rho", 0.85, 4.72, 0.78)
    G.eq(s, "cl_alpha", 0.85, 5.06, 0.78)
    G.eq(s, "cl_depink", 5.6, 4.38, 0.78)
    G.eq(s, "cl_depink2", 5.6, 4.86, 0.78)
    text(s, 5.6, 5.15, 4.0, 0.4, "0,704 = (0,299 + 0,114)/0,587: giữ độ chói BT.601", 11, MUTED)
    text(s, 9.65, 4.4, 2.9, 1.15, ["Cổng sửa g (bộ lọc): R hoặc B gần trần và cả hai vượt G. R−G > 25, B−G > 25 chỉ là luật đếm điểm hồng."], 12, TEXT)
    cards = [(L, 1.5, "Tổng p50", ["numpy 9,6 ms", "ONNX 19,3 ms"]),
             (xs[0], 2.5, "Khử khối · 2,20 ms", ["α 10, β 4, s 0,5, λ 0,5", "lưới còn nguyên"]),
             (xs[1], 2.5, "Khử hồng · 2,57 ms", ["trước khử mù xoá 93 %,", "đặt sau chỉ 46–68 %"]),
             (xs[2], 2.5, "Tối góc · 0,15 ms", ["trần hệ số 1,35;", "góc 624×480 cần 1,32"]),
             (xs[3], 2.5, "Mù 2,21 + nét 1,59 ms", ["ô 15 px, ω 0,85,", "chặn t ≥ 0,4"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 5.72, w, 1.23, lab, lines, 12 if w > 2 else 11)
    C.notes(s, ["Thứ tự trong code: khử khối, khử hồng, bù tối góc, khử mù, làm nét (cắt 16 cột trước, đệm lại sau).",
                "Khử khối cần lưới 8×8 còn nguyên nên chạy đầu; khử hồng phải trước khử mù vì khử mù nhân sắc độ với 1/t.",
                "Khử hồng: cổng g dùng max(R,B) ≥ 230; ngưỡng R−G > 25 chỉ là luật đo, không phải cổng sửa.",
                "Thời gian đo trên 200 khung dev, Ryzen 9 7945HX, cv2 4 luồng; số dưới mỗi tầng là trước và sau trên tập dev."],
            "xla/enhance/classic.py:183-401; configs/enhance_classic.yaml; results/enhance/classic_bench.json.")
    return s


# ------------------------------------------------------------------ §12 dehaze theory
def s_dehaze():
    s = C.new("purple")
    C.header(s, "purple", "12 · LỌC CỔ ĐIỂN", "Khử mù: ước A và t̃ từ dark channel, làm mịn bằng guided filter")
    card(s, L, 1.85, 6.75, 5.1); card(s, 7.52, 1.85, 5.2, 5.1)
    fx, ex = 0.85, 4.62
    rows = [("dc", 2.0, ["ô 15×15, kênh tối nhất;", "vùng trong gần 0"]),
            ("air", 2.66, ["trung bình 0,1 % điểm dark", "cao nhất, kẹp 180–250"]),
            ("trans", 3.22, ["ω = 0,85: giữ lại 15 %", "màn mờ cho có chiều sâu"]),
            ("gf", 3.98, ["Y làm ảnh dẫn: t bám biên,", "không quầng; chạy ở 1/4 cỡ"]),
            ("gfp", 5.05, ["t ≥ 0,4: không khuếch đại", "nhiễu ở vùng mù đặc"]),
            ("recov", 5.62, ["trừ màn A, chia cho t,", "cộng lại A"])]
    for name, y, expl in rows:
        G.eq(s, name, fx, y)
        text(s, ex, y + 0.02, 2.62, 0.55, expl, 12, TEXT)
    text(s, fx, 6.32, 6.3, 0.55, "Code thêm hai chốt: t ≥ 1 − ω·min(I)/Ā tại từng điểm; kéo giãn khi mức đen ra thấp hơn min(mức đen vào; 8).", 12, MUTED)
    G.fig(s, "dc3d", 7.75, 1.98, 3.4)
    text(s, 11.15, 2.15, 1.45, 2.0, ["Khung test", "005_00934:", "trung vị 138,", "nền đường", "tới 216"], 12, MUTED)
    text(s, 7.8, 4.45, 4.7, 0.3, "Sách (He và cộng sự 2011):", 13, MUTED, True)
    G.eq(s, "he_book", 7.8, 4.78, 0.8)
    text(s, 7.8, 5.38, 4.7, 0.62, "Code chia cho Ā vô hướng thay vì I/A từng kênh, và chặn t trước khi phóng lại cỡ gốc.", 12, TEXT)
    text(s, 7.8, 6.05, 4.7, 0.8, [[("Test 198 khung: ", {"color": INK, "bold": True}),
                                   ("classic ONNX, dark channel trung bình 140,2 → 69,8, giảm 50,2 %.", {})]], 13, TEXT)
    C.notes(s, ["Dark channel prior: ở vùng không mù, trong ô 15 px luôn có kênh gần 0; màn mờ nâng giá trị này lên.",
                "A ước từng kênh từ 0,1 % điểm có dark channel cao nhất; t̃ = 1 − ω·dark/Ā với ω = 0,85.",
                "Guided filter r = 6, ε = 10⁻³ ở 1/4 độ phân giải, ảnh dẫn là độ chói Y; chặn t ≥ 0,4.",
                "So với sách: code chia cho Ā vô hướng, t0 áp trước khi phóng; thêm chặn từng điểm và kéo mức đen."],
            "xla/enhance/classic.py:316-369; configs/enhance_classic.yaml:10-21; results/enhance/enhance_eval.json; báo cáo LaTeX hình 9b.")
    return s


# ------------------------------------------------------------------ §13 NAFNet
def s_nafnet():
    s = C.new("purple")
    C.header(s, "purple", "13 · SO SÁNH LÀM RÕ", "NAFNet-w16 không dùng hàm kích hoạt, 1,14 triệu tham số")
    G.panel(s, L, 1.85, 1.6, 2.95, None, "Cặp học")
    G.framed(s, RPT + "d2_5_final.jpg", 0.74, 2.3, 1.36, 1.02)
    text(s, 0.72, 3.33, 1.4, 0.25, "vào: D2", 11, TEXT, True, align="c")
    G.framed(s, RPT + "d2_0_src.jpg", 0.74, 3.58, 1.36, 0.9)
    text(s, 0.72, 4.5, 1.4, 0.25, "đích: D0", 11, TEXT, True, align="c")
    ws = [2.75, 4.45, 2.9]
    specs = [("yel", "Mạng U 4 tầng"), ("lav", "NAFBlock"), ("blu", "Huấn luyện")]
    xs = stage_row(s, 2.37, 1.85, 2.95, ws, 0.12, specs)
    for k, lab in enumerate(["conv 3×3: 3 → 16 kênh", "mã hoá 16 → 32 → 64 → 128", "giữa 256 kênh, 1 khối", "giải mã, PixelShuffle"]):
        y = 2.28 + k * 0.5
        G.node(s, xs[0] + 0.15, y, 2.45, 0.38, lab, "learn", 11)
        if k < 3:
            G.arrow(s, xs[0] + 1.375, y + 0.38, xs[0] + 1.375, y + 0.5, INK, 1.0)
    G.eq(s, "gres", xs[0] + 1.375, 4.3, 0.7, align="c")
    row1 = ["LN", "1×1", "DW 3×3", "SG", "SCA", "1×1", "+β"]
    row2 = ["LN", "1×1", "SG", "1×1", "+γ"]
    for r, labs in enumerate([row1, row2]):
        x = xs[1] + 0.15
        for k, lab in enumerate(labs):
            w = 0.66 if lab == "DW 3×3" else 0.5
            G.node(s, x, 2.32 + r * 0.72, w, 0.44, lab, "dark" if lab.startswith("+") else "learn", 11)
            if k < len(labs) - 1:
                G.arrow(s, x + w, 2.54 + r * 0.72, x + w + 0.1, 2.54 + r * 0.72, INK, 1.0)
            x += w + 0.1
    G.eq(s, "sg", xs[1] + 0.2, 3.85, 0.72)
    G.eq(s, "sca", xs[1] + 1.65, 3.85, 0.72)
    G.eq(s, "nres", xs[1] + 0.2, 4.3, 0.72)
    G.eq(s, "nloss", xs[2] + 1.45, 2.38, 0.62, align="c")
    for k, lab in enumerate(["AdamW, lr 0,001 cosine", "batch 16, 4 500 bước", "mảnh 256×256, lật ngang/dọc"]):
        G.node(s, xs[2] + 0.15, 2.85 + k * 0.5, 2.6, 0.38, lab, "plain", 11)
    G.bar(s, xs[2] + 0.12, 4.37, 2.66, 0.34, "blu", "chọn bước 4 500", 12)
    gap_arrows(s, [(L + 1.6, xs[0])] + [(xs[i] + ws[i], xs[i + 1]) for i in range(2)], 2.47)
    cards = [(L, 1.6, "Dữ liệu", ["3 006 cặp,", "lưới 8 px"]),
             (xs[0], ws[0], "Mạng U", ["1 khối mỗi tầng, cộng", "ảnh vào ở đầu ra"]),
             (xs[1], ws[1], "NAFBlock", ["SimpleGate thay GELU: chia đôi kênh rồi", "nhân; β, γ khởi tạo 0, không dropout"]),
             (xs[2], ws[2], "Kết quả tổng hợp", ["PSNR 22,67 dB, SSIM 0,805", "(ảnh vào 13,92 dB, 0,628)"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 4.95, w, 1.12, lab, lines, 13 if w > 2 else 12)
    card(s, L, 6.2, W, 0.75)
    text(s, 0.85, 6.27, 11.7, 0.62, "NAFNet-w16 xuất ONNX fp16, opset 17. Đo xen kẽ trong cùng tiến trình: NAFNet 20,6 / 27,9 ms p50 / p95, classic ONNX 19,3 / 25,3 ms. Bản rộng 32 (29,2 triệu tham số) cần 82,1 ms p95 nên chọn w16.", 13, TEXT)
    C.notes(s, ["NAFNet học ánh xạ D2 → D0 trên cặp mảnh 256×256 đặt theo lưới 8 px để giữ pha khối JPEG.",
                "NAFBlock: LayerNorm, conv 1×1 mở đôi kênh, depthwise 3×3, SimpleGate X1⊙X2, SCA, conv 1×1, cộng tắt có hệ số β; nửa sau là FFN với γ.",
                "Loss L1 + (1 − SSIM), cửa sổ Gauss 11 px; AdamW betas (0,9; 0,9), lr 1e-3, 4 500 bước, batch 16, khởi tạo ngẫu nhiên.",
                "Luồng Ảnh rõ dùng bộ lọc cổ điển ONNX thay NAFNet; bảng so sánh ở slide sau."],
            "xla/enhance/arch/nafnet.py:35-157; xla/enhance/net.py:86-88; xla/enhance/train_net.py:116-264; results/restore/train.json; models/release/enhancer-v1.0.0/meta.json.")
    return s


# ------------------------------------------------------------------ §14 C2: criteria tiles + formulas
def s_c2():
    s = C.S[15]
    C.clear(s)
    C.header(s, "purple", "12 · ĐÁNH GIÁ ẢNH RÕ", "Mạng đạt 3/7 tiêu chí, cổ điển giảm mù 52 % so với 37 %")
    tiles = [("Điểm hồng giảm", "ngưỡng 90 %", 100, 90, 89.2, 99.9, ("89 %", "99,9 %")),
             ("Dark channel giảm", "ngưỡng 50 %", 100, 50, 51.6, 36.8, ("52 %", "37 %")),
             ("Nhấp nháy giữa hai khung", "ngưỡng 1×, không tăng", 2.5, 1.0, 2.18, 1.81, ("2,18×", "1,81×")),
             ("Thời gian mỗi khung", "ngưỡng 30 ms", 35, 30, 9.2, 20.7, ("9,2 ms", "20,7 ms"))]
    LAY = (0.2, 0.06, 0.7, 0.88)
    for i, (ttl, thl, vmax, thr, cv, nv, labs) in enumerate(tiles):
        x = L + (i % 2) * 6.15; y = 1.85 + (i // 2) * 2.17
        card(s, x, y, 5.95, 2.05)
        text(s, x + 0.28, y + 0.13, 3.6, 0.35, ttl, 16, INK, True)
        text(s, x + 3.6, y + 0.17, 2.1, 0.3, thl, 12, PINK, align="r")
        FX, FY, FW, FH = x + 0.15, y + 0.52, 5.65, 1.45
        kit.bar_chart(s, FX, FY, FW, FH, ["Cổ điển", "Mạng"],
                      [("v", [cv, nv], GOLD, list(labs), {1: TEAL}, LP.OUTSIDE_END)],
                      horizontal=True, gap=40, vmin=0, vmax=vmax, layout=LAY)
        px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
        xx = px + thr / vmax * pw
        line(s, xx, py - 0.02, xx, py + ph + 0.02, PINK, 1.5, dash=True)
    card(s, L, 6.2, W, 0.75)
    G.eq(s, "pinkdrop", 0.85, 6.26, 0.7)
    G.eq(s, "flick", 3.0, 6.24, 0.62)
    text(s, 5.2, 6.27, 7.35, 0.62, "p: tỉ lệ điểm R−G > 25, B−G > 25 trên 198 khung test; P, R: khung sau xử lý và khung gốc kế nhau (cách < 0,25 s, 91 chuỗi).", 12, TEXT)
    C.notes(s, ["Bảy tiêu chí chốt trước cho mạng; mạng đạt 3: hồng −99,9 %, 20,7 ms, không hại mAP vì nhận diện chạy trên khung gốc.",
                "Mạng trượt: dark channel −37 % (cần 50 %), nhấp nháy 1,81× (cần ≤ 1), A/B 11/25 với gốc, 1/25 với cổ điển.",
                "Cổ điển: hồng −89 % (sát ngưỡng 90 %), dark channel −52 %, nhấp nháy 2,18×, 9,2 ms p50 trên CPU.",
                "Công thức: tỉ lệ giảm điểm hồng và tỉ số nhấp nháy giữa khung đã xử lý và khung gốc."],
            "results/reproduce/tables/t3_tieu_chi_anh_ro.md; results/enhance/enhance_eval.json; xla/eval/enhance_eval.py:75-125; xla/eval/temporal.py:47-83.")
    return s


# ------------------------------------------------------------------ §14 C3: A/B with Wilson intervals
def s_c3():
    s = C.S[16]
    C.clear(s, keep_pictures=True)
    C.header(s, "purple", "12 · ĐÁNH GIÁ ẢNH RÕ", "Người chấm mù chọn bản cổ điển 24/25 lượt")
    card(s, L, 1.85, 7.2, 5.1)
    text(s, 0.9, 1.98, 6.7, 0.3, "TỈ LỆ MẠNG THẮNG · 25 LƯỢT MỖI CẶP · THANH ĐEN: WILSON 95 %", 12, PURPLE, True)
    rows = [("L1 mạng vs gốc", 0.44, 0.267, 0.629, 0.70, TEAL), ("L1 mạng vs cổ điển", 0.04, 0.007, 0.195, 0.55, GOLD),
            ("L2 mạng vs gốc", 0.52, 0.335, 0.700, 0.70, TEAL), ("L2 mạng vs cổ điển", 0.20, 0.089, 0.391, 0.55, GOLD)]
    FX, FY, FW, FH = 0.75, 2.35, 6.95, 2.9
    LAY = (0.3, 0.02, 0.64, 0.86)
    gf = kit.bar_chart(s, FX, FY, FW, FH, [r[0] for r in rows],
                       [("p", [r[1] for r in rows], TEAL, [vn(r[1], 2) for r in rows], {i: r[5] for i, r in enumerate(rows)}, LP.INSIDE_BASE)],
                       horizontal=True, gap=55, vmin=0, vmax=1.0, layout=LAY, grid=True, major=0.25)
    kit._err_bars(gf.chart.plots[0].series[0], [r[3] - r[1] for r in rows], [r[1] - r[2] for r in rows])
    px, py, pw, ph = kit.plot_xy(FX, FY, FW, FH, LAY)
    rh = ph / 4
    for i, r in enumerate(rows):
        xx = px + r[4] * pw
        line(s, xx, py + i * rh + 0.06, xx, py + (i + 1) * rh - 0.06, PINK, 2.0, dash=True)
    for k in range(5):
        v = k * 0.25
        text(s, px + v * pw - 0.3, py + ph + 0.03, 0.6, 0.25, vn(v, 2) if 0 < v < 1 else str(int(v)), 11, MUTED, align="c")
    text(s, px, py + ph + 0.27, pw, 0.25, "vạch hồng: ngưỡng đạt 0,70 (với gốc) và 0,55 (với cổ điển)", 11, PINK, align="c")
    G.eq(s, "wilson", 0.9, 5.82, 0.8)
    text(s, 4.2, 5.85, 3.5, 1.05, ["p: tỉ lệ thắng trên n = 25 lượt. Cận dưới > 0,5: mạng thắng; cận trên < 0,5: mạng thua.",
                                   "Cổ điển thắng 24/25: [0,805; 0,993]."], 12, TEXT)
    C.notes(s, ["Trang HTML offline, trái phải xáo, tên file ngẫu nhiên; 55 lượt mỗi lần gồm 5 cặp lặp để đo độ nhất quán (3/5).",
                "Lần 1 cả khung: mạng thắng gốc 11/25, thắng cổ điển 1/25. Lần 2 phóng 2× quanh người: 13/25 và 5/25.",
                "CI Wilson 95 % với z = 1,96; mạng vs cổ điển có cận trên dưới 0,5 cả hai lần, tức mạng thua.",
                "Một người chấm, nên kết quả chỉ là bằng chứng sơ bộ."],
            "results/reproduce/tables/t4_ab.md; results/ab/ab_score.json; results/ab-zoom/ab_score.json; xla/eval/ab_score.py:27-53.")
    return s


# ------------------------------------------------------------------ §16 contract stages
def s_contract():
    s = C.new("end")
    C.header(s, "end", "14 · TÍCH HỢP IOT-CV", "IOT-CV chỉ nạp gói ONNX qua được ba phép kiểm")
    G.panel(s, L, 1.85, 1.75, 2.6, None, "Gói model")
    for k, lab in enumerate(["model.onnx", "meta.json", "3 khung golden"]):
        G.node(s, 0.75, 2.3 + k * 0.62, 1.49, 0.46, lab, "data", 11)
    specs = [("yel", "1. Kiểm sha256"), ("lav", "2. Kiểm tensor"), ("blu", "3. Chạy golden"), ("mint", "4. Nạp hoặc từ chối")]
    xs = stage_row(s, 2.52, 1.85, 2.6, [2.43] * 4, 0.12, specs)
    nodes = [[("băm file onnx", "learn"), ("so onnx_sha256", "plain")],
             [("frame uint8 480×640×3 BGR", "learn"), ("boxes [N,4] · scores [N]", "plain")],
             [("3 khung dev, ra hộp", "learn"), ("IoU ≥ 0,95, lệch điểm ≤ 0,02", "plain")],
             [("đúng cả ba: nạp", "learn"), ("lệch một: web báo lý do", "frozen")]]
    bars = ["23b1da69…c59aef", "opset 17", "ORT khớp PyTorch", "input_variant: raw"]
    for x, ns, b, pal in zip(xs, nodes, bars, ["yel", "lav", "blu", "mint"]):
        for k, (lab, kind) in enumerate(ns):
            y = 2.32 + k * 0.72
            G.node(s, x + 0.12, y, 2.19, 0.5, lab, kind, 11)
            if k == 0:
                G.arrow(s, x + 1.215, y + 0.5, x + 1.215, y + 0.72, INK, 1.25)
        G.bar(s, x + 0.12, 3.9, 2.19, 0.36, pal, b, 12)
    gap_arrows(s, [(L + 1.75, xs[0])] + [(xs[i] + 2.43, xs[i + 1]) for i in range(3)], 2.57)
    cards = [(L, 1.75, "Phiên bản", ["detector-v1.0.0", "enhancer-v1.0.0"]),
             (xs[0], 2.43, "meta.json", ["version, crop_left 16,", "hồ sơ camera 1.2.0"]),
             (xs[1], 2.43, "Trong graph", ["cắt 16 cột, letterbox", "1 280, NMS 0,5"]),
             (xs[2], 2.43, "Độ khớp", ["ORT CPU so PyTorch:", "lệch 0,0000 mAP50"]),
             (xs[3], 2.43, "Ngưỡng", ["điểm ≥ 0,0767,", "tối đa 100 hộp"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 4.6, w, 1.1, lab, lines, 13 if w > 2 else 12)
    card(s, L, 5.85, W, 1.1)
    text(s, 0.85, 5.95, 11.7, 0.95, [[("Enhancer: ", {"bold": True, "color": INK}),
                                      ("frame vào, frame_out cùng cỡ, fp16; chỉ chạy khi có người xem luồng Ảnh rõ, không bật mặc định. ", {}),
                                      ("Bộ lọc cổ điển ", {"bold": True, "color": INK}),
                                      ("là hàm Python, chưa nằm trong hợp đồng; đưa vào IOT-CV cần mở rộng hợp đồng.", {})]], 13, TEXT)
    C.notes(s, ["Hợp đồng model: đầu vào frame uint8 [480, 640, 3] BGR; đầu ra boxes float32 [N, 4] theo toạ độ khung gốc, scores [N].",
                "IOT-CV kiểm sha256, tensor và 3 khung golden (IoU ≥ 0,95, lệch điểm ≤ 0,02); lệch một điều kiện là không nạp.",
                "Tiền và hậu xử lý nằm trong graph: cắt 16 cột, letterbox 1 280, NMS IoU 0,5, ngưỡng 0,0767.",
                "Cờ input_variant cho phép đổi luồng vào sau này mà không sửa code."],
            "xla/contract.py:37-77; docs/hop-dong-model.md; models/release/detector-v1.0.0/meta.json; báo cáo LaTeX mục 3.3.")
    return s


# ------------------------------------------------------------------ §18 discussion
def s_discussion():
    s = C.new("end")
    C.header(s, "end", "16 · THẢO LUẬN", "Dev-val chọn sai, mạng thua cổ điển, mọi số từ một buổi bay")
    cols = [("Vì sao dev-val chọn sai", ORANGE, [
        "Dev-val cùng chuyến với khung đã gắn nhãn giả, đệm chỉ 0,3 s.",
        "Recall nhãn giả 8,4 % và 11,2 %: trò học cách bỏ qua người nhỏ.",
        "100 khung, 36 có người: quá ít; E1 D2 nhảy 0,327 → 0,137 giữa hai epoch liền nhau."]),
        ("Vì sao mạng thua cổ điển", PURPLE, [
            "Mạng học D2 → D0: hồng giảm 99,9 % nhưng dark channel chỉ 36,8 %, cổ điển 51,6 %.",
            "Có lẽ mù trên cặp huấn luyện nhẹ hơn khung thật (chưa kiểm riêng).",
            "Màu nguội đến từ đích D0; mạng tẩy cả vật vốn màu hồng."]),
        ("Giới hạn", BLUE, [
            "Một bãi, một buổi sáng; 39 khung có người nên CI mAP50 rộng khoảng ±0,1.",
            "Không model nào đạt báo nhầm ≤ 0,05; tốt nhất E1 D2 0,302.",
            "DQT chỉ đo ở q12; A/B một người chấm; nhịp thật p95 54,1 ms."])]
    for i, (ttl, col, lines) in enumerate(cols):
        x = L + i * 4.1
        card(s, x, 1.85, 3.9, 5.1)
        text(s, x + 0.28, 2.05, 3.4, 0.4, ttl, 18, col, True)
        for k, ln in enumerate(lines):
            y = 2.7 + k * 1.38
            text(s, x + 0.28, y, 0.4, 0.4, str(k + 1), 22, col, True)
            text(s, x + 0.72, y + 0.04, 2.95, 1.3, ln, 14, TEXT)
    C.notes(s, ["Dev-val không dự đoán được test: ba nguyên nhân khả dĩ, chưa tách riêng được.",
                "Mạng làm rõ thừa hưởng thiên lệch của cặp D2 → D0; bộ cổ điển khử mù theo chính khung đang xem.",
                "Nhận diện chạy trên khung gốc vì bộ đã tinh chỉnh học trên ảnh xấu kiểu D2.",
                "Giới hạn: một buổi bay, 39 khung có người, một người chấm A/B, ngân sách ms ở nhịp thật chưa đạt."],
            "báo cáo LaTeX mục 5; results/e2/README.md; results/enhance/enhance_eval.json.")
    return s


# ------------------------------------------------------------------ §19 conclusion
def s_conclusion():
    s = C.S[24]
    C.clear(s)
    C.header(s, "end", "KẾT LUẬN", "D2 bớt báo nhầm; nhận diện giữ trên khung gốc")
    card(s, L, 1.85, 7.25, 5.1)
    text(s, 0.9, 2.02, 6.8, 0.3, "CÂU HỎI 1 · D2 SO VỚI D1", 12, TEAL, True)
    text(s, 0.9, 2.35, 3.2, 0.8, "0,99 → 0,30", 34, TEAL, True)
    text(s, 4.2, 2.42, 3.5, 0.9, "hộp báo nhầm mỗi khung bay không người, CI hiệu [−1,238; −0,246]", 13, TEXT)
    text(s, 0.9, 3.25, 6.8, 0.6, "mAP50 tăng 0,093 nhưng CI chạm 0; E1 D2 0,236 so với mốc 0,126.", 14, TEXT)
    text(s, 0.9, 4.0, 6.8, 0.3, "CÂU HỎI 2 · LÀM RÕ TRƯỚC KHI NHẬN DIỆN", 12, PURPLE, True)
    text(s, 0.9, 4.33, 3.2, 0.8, "−0,085", 34, PURPLE, True)
    text(s, 4.2, 4.4, 3.5, 0.9, "mAP50 của bộ đã tinh chỉnh khi ảnh qua mạng; qua cổ điển −0,112", 13, TEXT)
    text(s, 0.9, 5.25, 6.8, 1.6, ["Gói detector-v1.0.0 chạy 23,6 ms trong IOT-CV nhưng mAP50 chỉ 0,120 vì dev-val chọn sai.",
                                  "Lọc cổ điển 9,2 ms thắng NAFNet 24/25 lượt, đề xuất làm luồng ảnh rõ."], 14, TEXT, spacing=1.1)
    card(s, 8.07, 1.85, 4.65, 5.1)
    text(s, 8.35, 2.02, 4.1, 0.35, "Việc tiếp theo", 18, BLUE, True)
    steps = ["Hạ phơi sáng, chỉnh cân bằng trắng của OV3660; đo bảng lượng tử ở q40.",
             "Bay thêm nhiều độ cao, tách dev-val thành chuyến riêng, chọn lại E1 D2, r1, r2.",
             "Thêm loss màu a*b* và loss nhất quán giữa khung cho mạng làm rõ.",
             "Thử detector 960 px hoặc FP16 để vào ngân sách 50 ms."]
    for k, st in enumerate(steps):
        y = 2.55 + k * 1.08
        text(s, 8.35, y, 0.4, 0.4, str(k + 1), 20, BLUE, True)
        text(s, 8.8, y + 0.03, 3.7, 1.0, st, 14, TEXT)
    C.notes(s, ["Câu hỏi 1: D2 không chắc chắn hơn D1 về mAP50, nhưng giảm báo nhầm rõ, CI nằm trọn một phía.",
                "Câu hỏi 2: làm rõ trước làm mAP của bộ đã tinh chỉnh giảm, nên nhận diện chạy trên khung gốc.",
                "Việc cần làm ngay: dev-val tách khỏi khối huấn luyện để chọn lại giữa E1 D2 và r2.",
                "Bộ lọc cổ điển cần mở rộng hợp đồng model vì nó là hàm Python, không phải ONNX."],
            "báo cáo LaTeX mục 6 và 5.4; t1_nhan_dien.md; t5_e4.md; t4_ab.md.")
    return s


# ------------------------------------------------------------------ §20 references
REFS = [
    "Phạm Văn Cường, Bài giảng Xử lý ảnh, chương 1–9, Khoa Trí tuệ Nhân tạo, PTIT, 2026.",
    "R. C. Gonzalez, R. E. Woods, Digital Image Processing, 4th ed., Pearson, 2018.",
    "G. Jocher, J. Qiu, Ultralytics YOLO11, gói ultralytics 8.4, 2024.",
    "L. Chen, X. Chu, X. Zhang, J. Sun, Simple Baselines for Image Restoration, ECCV 2022.",
    "P. Zhu, L. Wen, D. Du và cộng sự, Detection and Tracking Meet Drones Challenge, IEEE TPAMI 44(11), 2022.",
    "K. He, J. Sun, X. Tang, Single Image Haze Removal Using Dark Channel Prior, IEEE TPAMI 33(12), 2011.",
    "K. He, J. Sun, X. Tang, Guided Image Filtering, IEEE TPAMI 35(6), 2013.",
    "D. Lopez-Paz, M. Oquab, Revisiting Classifier Two-Sample Tests, ICLR 2017.",
    "M. Bińkowski, D. J. Sutherland, M. Arbel, A. Gretton, Demystifying MMD GANs, ICLR 2018.",
    "M. Oquab và cộng sự, DINOv2: Learning Robust Visual Features without Supervision, TMLR 2024.",
    "X. Wang, L. Xie, C. Dong, Y. Shan, Real-ESRGAN, ICCV Workshops 2021.",
    "A. RoyChowdhury và cộng sự, Automatic Adaptation of Object Detectors to New Domains Using Self-Training, CVPR 2019.",
    "Y. Zhang và cộng sự, ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV 2022.",
    "N. Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772, 2016.",
    "ONNX Runtime developers, ONNX Runtime (onnxruntime-gpu 1.24.4), Microsoft.",
    "ISO/IEC 10918-1, Digital compression and coding of continuous-tone still images, phụ lục K, 1994.",
]


def s_refs():
    s = C.S[23]
    C.clear(s)
    C.header(s, "end", "TÀI LIỆU THAM KHẢO", "Tài liệu tham khảo")
    for c in range(2):
        card(s, L + c * 6.15, 1.85, 5.95, 5.1)
        lines = [[(f"[{i + 1}] ", {"bold": True, "color": BLUE}), (REFS[i], {})] for i in range(c * 8, c * 8 + 8)]
        text(s, L + c * 6.15 + 0.25, 2.02, 5.45, 4.85, lines, 12, TEXT, spacing=1.05)
    C.notes(s, ["Danh mục giống báo cáo LaTeX mục 9, cùng thứ tự.", "Mã nguồn và số liệu tái lập: repo XLA, scripts/reproduce.py."],
            "docs/nop-bai-src/latex/sections/09-tai-lieu.tex.")
    return s
