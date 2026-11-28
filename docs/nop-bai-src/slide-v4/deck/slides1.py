"""v8 slides for outline sections 2-10 (problem, theory, data, D2, realism, YOLO11s, metrics, training)."""
from pptx.enum.chart import XL_LABEL_POSITION as LP

import common as C
import kit
import stage as G
from kit import BLUE, GOLD, INK, MUTED, ORANGE, PINK, PURPLE, SLATE, TEAL, TEXT, card, text, vn

L, W = 0.62, 12.1
RPT = G.RPT + "/"


def gap_arrows(s, xs, y):
    """Arrows across the gaps between consecutive panels: xs = [(right_edge, next_left), ...]."""
    for a, b in xs:
        G.arrow(s, a - 0.06, y, b + 0.12, y, INK, 1.75)


def stage_row(s, x0, y, h, widths, gap, specs):
    """specs: list of (pal, title). Returns panel x positions."""
    xs, x = [], x0
    for w, (pal, title) in zip(widths, specs):
        G.panel(s, x, y, w, h, pal, title)
        xs.append(x); x += w + gap
    return xs


# ------------------------------------------------------------------ §2 problem: five measured defects
def s_problem():
    s = C.new("blue")
    C.header(s, "blue", "01 · BÀI TOÁN", "Năm lỗi đo được trên khung thật, nặng nhất là ám hồng")
    card(s, L, 1.85, 5.2, 5.1)
    G.framed(s, RPT + "art_full.jpg", 0.85, 2.05, 4.74, 3.56)
    text(s, 0.85, 5.66, 4.74, 0.28, "Khung 004_01252 (02/11): hồng, mù, tối góc", 12, MUTED)
    G.framed(s, RPT + "art_pink.jpg", 0.85, 6.0, 1.2, 0.9)
    G.framed(s, RPT + "art_block.jpg", 2.15, 6.0, 1.2, 0.9)
    G.framed(s, RPT + "art_strip.jpg", 3.45, 6.0, 0.34, 0.9)
    text(s, 3.92, 6.0, 1.7, 0.9, ["hồng ×4", "biên khối ×8", "dải mép trái"], 12, MUTED)
    card(s, 6.02, 1.85, 6.7, 5.1)
    text(s, 6.3, 2.02, 6.2, 0.3, "HỒ SƠ CAMERA · 1 115 KHUNG DEV · TRUNG VỊ (P5–P95)", 12, BLUE, True)
    rows = [["Lỗi", "Đại lượng đo", "Giá trị"],
            [("Ám hồng", {"bold": True, "color": PINK}), "điểm có R−G > 25 và B−G > 25", "0,17 % điểm; 31 % số khung có > 0,5 %"],
            [("Mù", {"bold": True, "color": BLUE}), "dark channel, ô 15 px (0–255)", "143,7 (134,6–152,1)"],
            [("Vỡ khối JPEG", {"bold": True, "color": PURPLE}), "tỉ số biên khối 8×8", "2,68 (2,15–3,52)"],
            [("Tối góc", {"bold": True, "color": ORANGE}), "sáng góc chia tâm; f khớp cos⁴", "0,81 (0,62–0,93); f = 1 016 px"],
            [("Dải mép trái", {"bold": True, "color": SLATE}), "bề rộng; độ sáng Y trong dải", "14 px (13–16); Y 100,9 so với 152,3"],
            [("Nhoè", {"bold": True, "color": TEAL}), "PSF Gauss khớp từ MTF50", "0,76 px"]]
    kit.table(s, 6.3, 2.4, [1.55, 2.5, 2.15], 0.56, rows, size=13, aligns="lll")
    text(s, 6.3, 6.42, 6.2, 0.45, "Khung JPEG 640×480, q12, 4:2:2, 2–9 khung/s. Khung test không tham gia đo.", 13, MUTED)
    C.notes(s, ["Năm lỗi chính của camera, đo trên 1 115 khung dev (hồ sơ ov3660-esp32s3-q12, bản 1.2.0).",
                "Ám hồng nặng nhất: 31 % số khung có hơn 0,5 % điểm hồng.",
                "Dải mép trái rộng 14 px, sáng 100,9 chứ không đen; mọi model bỏ 16 cột đầu.",
                "Hồ sơ này là đầu vào của D2, của bộ lọc cổ điển và bộ nén khung giả của IOT-CV."],
            "configs/camera_profile.yaml; xla/camera/measure.py; báo cáo LaTeX mục 1.2, bảng 1.")
    return s


# ------------------------------------------------------------------ §2 system: stage diagram
def s_system():
    s = C.new("blue")
    C.header(s, "blue", "01 · BÀI TOÁN", "Một khung đi qua bốn chặng, nhận diện luôn đọc khung gốc")
    G.panel(s, L, 1.85, 1.62, 2.85, None, "Đầu vào")
    G.framed(s, RPT + "art_full.jpg", 0.74, 2.3, 1.38, 1.04)
    text(s, 0.72, 3.42, 1.42, 1.2, ["ESP32-S3 N16R8", "+ OV3660,", "nhìn xuống"], 12, TEXT, align="c")
    widths = [2.44] * 4
    specs = [("yel", "1. Nén và gửi"), ("lav", "2. Backend IOT-CV"),
             ("blu", "3. Hai model ONNX"), ("mint", "4. Web")]
    xs = stage_row(s, 2.39, 1.85, 2.85, widths, 0.15, specs)
    nodes = [[("cảm biến OV3660", "data"), ("JPEG 640×480 · q12", "learn"), ("MJPEG qua HTTP", "data")],
             [("giải mã BGR uint8", "data"), ("480×640×3, bỏ 16 cột", "learn"), ("ONNX Runtime CUDA", "data")],
             [("detector-v1.0.0", "learn"), ("hộp + điểm tin cậy", "data"), ("enhancer-v1.0.0", "frozen")],
             [("cảnh báo có người", "data"), ("luồng Ảnh rõ", "frozen"), ("người vận hành", "data")]]
    bars = ["2–9 khung/s", "RTX 4060 Laptop", "p95 ≤ 50 ms/khung", "2 đầu ra, 1 khung vào"]
    pals = ["yel", "lav", "blu", "mint"]
    for x, ns, b, pal in zip(xs, nodes, bars, pals):
        for k, (lab, kind) in enumerate(ns):
            y = 2.32 + k * 0.62
            G.node(s, x + 0.17, y, 2.1, 0.44, lab, kind, 12)
            if k < 2:
                G.arrow(s, x + 1.22, y + 0.44, x + 1.22, y + 0.62, INK, 1.25)
        G.bar(s, x + 0.12, 4.22, 2.2, 0.36, pal, b, 12)
    gap_arrows(s, [(L + 1.62, xs[0])] + [(xs[i] + 2.44, xs[i + 1]) for i in range(3)], 2.54)
    cards = [("Camera", ["drone S500,", "ArduPilot"]),
             ("Chặng 1", ["bo chỉ nén và gửi;", "AI chạy trên laptop"]),
             ("Chặng 2", ["model nhận đúng khung", "camera gửi, chưa sửa"]),
             ("Chặng 3 · p50", ["detector 23,6 ms", "enhancer 12,5 ms"]),
             ("Chặng 4", ["hộp người + ảnh rõ", "cùng cỡ 640×480"])]
    for i, (lab, lines) in enumerate(cards):
        x = L if i == 0 else xs[i - 1]
        w = 1.62 if i == 0 else 2.44
        G.detail(s, x, 4.85, w, 1.02, lab, lines, 13)
    G.detail(s, L, 6.0, 5.97, 0.95, "Câu hỏi 1",
             ["Làm xấu ảnh theo số đo của chính camera (D2) có thắng làm xấu ngẫu nhiên (D1) trên khung thật?"], 14)
    G.detail(s, 6.75, 6.0, 5.97, 0.95, "Câu hỏi 2",
             ["Làm rõ ảnh trước khi nhận diện có giúp bộ nhận diện không? Điểm chỉ chấm trên khung bay thật."], 14)
    C.notes(s, ["Đường đi của một khung: camera nén JPEG q12, gửi MJPEG qua Wi-Fi, backend IOT-CV giải mã, hai model ONNX, web.",
                "Nét đứt: enhancer và luồng Ảnh rõ chỉ chạy khi có người xem, không bật mặc định.",
                "Ràng buộc: p95 tổng ≤ 50 ms mỗi khung, detector ≤ 30 ms.",
                "Hai câu hỏi nghiên cứu ở hàng dưới; luật chấm: chỉ chấm trên khung bay thật."],
            "báo cáo LaTeX mục 1.1; docs/hop-dong-model.md; t6_iotcv_ms.md.")
    return s


# ------------------------------------------------------------------ §3 theory 1: haze + vignetting
def s_theory1():
    s = C.new("blue")
    C.header(s, "blue", "02 · MÔ HÌNH ẢNH HỎNG", "Mù và tối góc là hai phép biến đổi lên ảnh sạch J")
    card(s, L, 1.85, 5.95, 5.1); card(s, 6.77, 1.85, 5.95, 5.1)
    x = 0.9
    text(s, x, 2.0, 5.4, 0.34, "Mô hình khí quyển (He và cộng sự)", 17, BLUE, True)
    G.eq(s, "haze", x, 2.45)
    text(s, x, 2.86, 5.4, 0.62, ["J: ảnh sạch, A: ánh sáng khí quyển, t: phần ánh sáng tới thẳng cảm biến.",
                                 "t càng nhỏ, điểm ảnh càng trôi về màu A và mất tương phản."], 13, TEXT)
    text(s, x, 3.58, 1.0, 0.3, "Sách:", 13, MUTED, True)
    G.eq(s, "haze_book", x + 0.72, 3.56)
    text(s, x + 2.3, 3.58, 3.1, 0.3, "t giảm theo độ sâu d", 13, MUTED)
    text(s, x, 3.98, 5.4, 0.5, "Camera nhìn thẳng xuống nên độ sâu gần như đều. D2 dùng một t cho cả ảnh:", 13, TEXT)
    G.eq(s, "haze_d2", x, 4.5)
    text(s, x + 2.6, 4.52, 2.8, 0.62, "d*: dark channel đích rút từ phân bố đo; mẫu số dùng dark channel của ảnh nguồn", 12, MUTED)
    text(s, x, 5.2, 5.4, 0.3, "Đo độ mù bằng dark channel trung bình:", 13, TEXT)
    G.eq(s, "dark_meas", x, 5.55)
    text(s, x + 2.95, 5.62, 2.5, 0.8, ["Khung thật: 143,7/255.", "Màn mờ phủ cả khung, gần như xa."], 13, PINK, True)
    x = 7.05
    text(s, x, 2.0, 5.4, 0.34, "Tối góc theo luật cos⁴", 17, ORANGE, True)
    G.eq(s, "vig", x, 2.45)
    G.eq(s, "vig_obs", x, 3.12)
    text(s, x, 3.52, 5.4, 0.62, ["r: khoảng cách tới tâm ảnh, f: tiêu cự tính bằng điểm ảnh.",
                                 "Góc ảnh nhận ít sáng hơn tâm, ảnh thật bằng J nhân G."], 13, TEXT)
    G.eq(s, "vig_fit", x, 4.2)
    G.fig(s, "vig3d", x - 0.1, 4.86, 2.98)
    text(s, x + 3.0, 4.95, 2.45, 1.9, ["V = 0,81: tỉ số sáng góc trên tâm đo được; tử số là khoảng cách từ tâm tới ô góc.",
                                       "Đường hồng: mép dưới với f = 690 px mà D2 dùng sau hiệu chỉnh."], 12, MUTED)
    C.notes(s, ["Mô hình khí quyển: J ảnh sạch, A ánh sáng khí quyển, t độ truyền.",
                "Sách cho t = e^(−βd); code D2 không dùng độ sâu, chọn một t cho mỗi ảnh để dark channel ra đúng giá trị đích.",
                "Tối góc: G(r) = cos⁴(arctan r/f); đo V = 0,81 rồi giải f = 1 016 px.",
                "Hình 3D: hệ số sáng trên khung 640×480 theo f của hồ sơ."],
            "xla/degrade/d2_camera.py:108-138; xla/camera/measure.py; xla/camera/profile.py:200-204; báo cáo LaTeX hình 9.")
    return s


# ------------------------------------------------------------------ §3 theory 2: JPEG + pink cast
def s_theory2():
    s = C.new("blue")
    C.header(s, "blue", "02 · MÔ HÌNH ẢNH HỎNG", "Khối 8×8 lộ ra vì q12; vùng sáng hồng vì G bão hoà trước")
    card(s, L, 1.85, 5.95, 5.1); card(s, 6.77, 1.85, 5.95, 5.1)
    x = 0.9
    text(s, x, 2.0, 5.4, 0.34, "Nén JPEG: DCT 8×8 rồi lượng tử", 17, PURPLE, True)
    G.eq(s, "dct", x, 2.42)
    text(s, x, 3.88, 5.4, 0.3, "f: khối 8×8 điểm ảnh, F: 64 hệ số tần số của khối.", 13, TEXT)
    G.eq(s, "quant", x, 4.24)
    text(s, x, 4.84, 5.4, 0.62, ["Q_K: bảng Annex K; camera dùng Q_K × 0,75 (q12), 4:2:2.",
                                 "Hệ số < Q/2 thành 0; mỗi khối làm tròn riêng."], 13, TEXT)
    text(s, x, 5.48, 5.4, 0.3, [[("DQT độ sáng, hàng đầu: ", {"color": MUTED}), ("12  8  8  12  18  30  38  46", {"bold": True})]], 13)
    G.eq(s, "block", x, 5.85)
    text(s, x + 3.15, 6.0, 2.3, 0.7, ["B = 2,68 trên khung thật;", "B = 1 là không có khối."], 13, PINK, True)
    x = 7.05
    text(s, x, 2.0, 5.4, 0.34, "Ám hồng trong khối màu RGB", 17, PINK, True)
    G.fig(s, "rgbcube", x - 0.15, 2.38, 3.05)
    G.eq(s, "pinkrule", x + 2.0, 2.45, 0.62)
    text(s, x + 3.0, 2.82, 2.5, 1.0, ["luật đo: chỉ để đếm", "điểm hồng"], 12, MUTED)
    text(s, x + 3.0, 3.42, 2.5, 1.0, ["Y vùng hồng:", "trung vị 222/255"], 13, PINK, True)
    text(s, x + 3.0, 4.1, 2.5, 0.7, ["cổng sửa của bộ lọc:", "max(R,B) ≥ 230, min(R,B) − G ≥ 6"], 11, MUTED)
    text(s, x, 4.86, 5.4, 0.3, "Khâu tông màu của D2 tái tạo cơ chế này:", 13, TEXT)
    G.eq(s, "tone", x, 5.2)
    G.eq(s, "tone_v", x, 5.58)
    text(s, x, 5.92, 5.4, 0.95, ["ISP nhân R, B với hệ số cân bằng trắng lớn hơn G.",
                                 "Ở vùng sáng G chạm trần rồi dừng, R và B còn tăng:",
                                 "điểm rời trục xám về phía hồng."], 13, TEXT)
    C.notes(s, ["JPEG: DCT 8×8 từng khối, chia cho bảng Q rồi làm tròn; ở q12 bảng thô nên hệ số cao tần mất, biên khối lệch.",
                "Bảng DQT đọc từ header file: Annex K × 0,75, 4:2:2; Pillow tái tạo đúng 20/20 khung.",
                "Ám hồng: G bão hoà trước R, B vì hệ số cân bằng trắng 2,13 và 1,812; hình khối RGB với 440 điểm sáng thật.",
                "Luật đo hồng R−G > 25 và B−G > 25 chỉ để đếm điểm; cổng sửa của bộ lọc là max(R,B) ≥ 230 và min(R,B) − G ≥ 6."],
            "xla/camera/measure.py:25,83-96,192; xla/camera/profile.py:65-75; xla/degrade/d2_camera.py:141-179; báo cáo LaTeX hình 7.")
    return s


# ------------------------------------------------------------------ §4 data
def s_data():
    s = C.new("blue")
    C.header(s, "blue", "03 · DỮ LIỆU", "Chia theo khối 30 s để khung test không có bản sao trong dev")
    card(s, L, 1.85, W, 3.0)
    text(s, 0.9, 1.98, 11.5, 0.3, "2 915 KHUNG BAY · 6 CHUYẾN SÁNG 02/11/2026 · SEED 20261118", 12, BLUE, True)
    G.fig(s, "timeline", 1.45, 2.25, 10.2)
    card(s, L, 4.98, 7.25, 1.97)
    rows = [["Tập", "Khung", "Hộp", "Dùng để"],
            ["dev (gồm dev-val 100)", "1 115", "115", "hồ sơ, hiệu chỉnh D2, nhãn giả"],
            ["test, cách nhau ≥ 1 s", "198", "149", "chấm điểm, 39 khung có người"],
            ["khung bay không người", "159", "0", "báo nhầm trên khung bay"],
            ["khung âm trong nhà", "300", "0", "báo nhầm trong nhà"]]
    kit.table(s, 0.8, 5.08, [2.3, 0.85, 0.75, 3.0], 0.35, rows, size=13, aligns="lrrl")
    G.detail(s, 8.07, 4.98, 4.65, 1.97, "Gán nhãn và ảnh nguồn",
             ["Nhãn trước: RT-DETR-x + YOLO11x, SAHI, WBF 2:1; soát 652 khung; duyệt trên Label Studio, sửa 39/151 hộp.",
              "VisDrone: 1 841 ảnh → 3 493 cửa sổ 624×480."], 13)
    C.notes(s, ["Khung liền nhau gần như giống hệt, chia ngẫu nhiên sẽ rò rỉ; nên chia 24 khối 30 s, 13 khối test theo tầng độ cao và có người.",
                "Vùng đệm 5 s quanh khối test; trong khối test khung cách nhau ≥ 1 s.",
                "Dev-val 100 khung chỉ để chọn checkpoint và ngưỡng; test không dùng cho quyết định nào.",
                "Huấn luyện dùng VisDrone: 3 006 cửa sổ huấn luyện (19 792 người), 487 kiểm tra; NOMAD bị loại vì nhìn xiên."],
            "results/data/split_report.json; results/baseline/baseline.json; data/real/labels/README.md; báo cáo LaTeX mục 2.")
    return s


# ------------------------------------------------------------------ §5 D2 stage diagram
def s_d2():
    s = C.new("teal")
    C.header(s, "teal", "04 · LÀM XẤU D2", "D2 dựng lại sáu khâu hỏng, 12 tham số do CMA-ES khớp")
    G.panel(s, L, 1.85, 1.75, 3.02, None, "Đầu vào")
    G.framed(s, RPT + "d2_0_src.jpg", 0.75, 2.3, 1.49, 1.12)
    text(s, 0.72, 3.47, 1.55, 0.5, ["ảnh nguồn J", "624×480"], 12, TEXT, align="c")
    G.node(s, 0.75, 4.05, 1.49, 0.66, ["hồ sơ camera", "1 115 khung dev"], "frozen", 11)
    specs = [("yel", "Khâu (a)–(b): quang học"), ("lav", "Khâu (c)–(d): cảm biến, ISP"), ("blu", "Khâu (e)–(f): đầu ra")]
    xs = stage_row(s, 2.52, 1.85, 2.4, [3.3, 3.3, 3.3], 0.1, specs)
    steps = [[("d2_1_haze.jpg", "(a) mù", "d2a"), ("d2_2_lens.jpg", "(b) ống kính", "d2b")],
             [("d2_3_tone.jpg", "(c) tông: hồng", "d2c"), ("d2_4_isp.jpg", "(d) ISP", "d2d")],
             [(None, "(e) dải mép", None), ("d2_5_final.jpg", "(f) JPEG", "d2f")]]
    for x, pair in zip(xs, steps):
        for k, (img, lab, e) in enumerate(pair):
            nx = x + 0.12 + k * 1.6
            G.node(s, nx, 2.28, 1.46, 1.92, None, "learn")
            if img:
                G.framed(s, RPT + img, nx + 0.08, 2.36, 1.3, 0.98)
            else:
                G.framed(s, RPT + "art_strip.jpg", nx + 0.53, 2.36, 0.4, 0.98)
            text(s, nx + 0.05, 3.38, 1.36, 0.26, lab, 12, INK, True, align="c")
            if e:
                G.eq(s, e, nx + 0.73, 3.72, 0.62, align="c")
            else:
                text(s, nx + 0.05, 3.7, 1.36, 0.4, "14 ± 1 px, vạch đỏ 60 px", 11, TEXT, align="c")
            if k == 0:
                G.arrow(s, nx + 1.46, 3.25, nx + 1.6, 3.25, INK, 1.25)
    gap_arrows(s, [(L + 1.75, xs[0]), (xs[0] + 3.3, xs[1]), (xs[1] + 3.3, xs[2])], 3.25)
    # objective bar across the three stages: real CMA-ES objective from xla/eval/lowlevel.py
    G.bar(s, 2.52, 4.33, 10.1, 0.88, "lav", None)
    G.eq(s, "obj_w", 2.7, 4.41, 0.82)
    text(s, 5.25, 4.47, 7.25, 0.65, [[("CMA-ES chỉnh 12 tham số để thống kê D2 khớp 1 115 khung dev: ", {"color": "FFFFFF"}),
                                      ("0,967 → 0,736 (155 lượt) → 0,712 (80 lượt)", {"color": "FFFFFF", "bold": True})],
                                     [("K: 18 đại lượng mức thấp, mỗi số hạng chia IQR của tập thật; W₁: Wasserstein-1", {"color": "E9E3F7", "size": 12})]], 13)
    cards = [(L, 1.75, "D1 đối chứng", ["Real-ESRGAN", "bậc hai"]),
             (xs[0], 3.3, "Khâu (a)–(b)", ["A ∈ [0,794; 0,944], t ∈ [0,2; 1]", "f ∈ [0,629; 0,729] × 1 016 px"]),
             (xs[1], 3.3, "Khâu (c)–(d)", ["WB 2,130/1,812; mức đen 0,0017", "nhiễu NLF × [0,425; 0,825]"]),
             (xs[2], 3.3, "Khâu (e)–(f)", ["80 % ảnh: bảng q12 đo được", "20 % ảnh: Annex K, q ∈ [19; 40]"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 5.38, w, 1.12, lab, lines, 13)
    text(s, L, 6.6, W, 0.35, "D1: nhoè 0,2–3,0, nhiễu 1–30, JPEG q 30–95 (4:2:0), lặp hai vòng. Ảnh minh hoạ là ảnh thử tự vẽ, không phải VisDrone.", 12, MUTED)
    C.notes(s, ["D2 đi theo thứ tự vật lý: không khí, ống kính, cảm biến và ISP, dải mép, nén JPEG.",
                "Khâu (c) không phủ màu hồng; nó cắt từng kênh rồi nhân lại hệ số cân bằng trắng, nên G dừng ở vùng sáng.",
                "CMA-ES: mục tiêu là trung bình W1/IQR trên 18 thống kê; vòng 1 155 lượt, vòng 2 80 lượt, 0,967 → 0,712.",
                "D1 là đối chứng 'không đo': chuỗi bậc hai Real-ESRGAN."],
            "xla/degrade/d2_camera.py; xla/degrade/calibrate.py:320; xla/eval/lowlevel.py:52-140; results/realism/calibration.json; configs/degrade_d2.yaml.")
    return s


# ------------------------------------------------------------------ §6 realism: three rulers + charts
def s_realism():
    s = C.new("teal")
    C.header(s, "teal", "05 · ĐỘ GIỐNG THẬT", "D2 lệch thống kê ít hơn D1 7 lần, C2ST vẫn tách được cả hai")
    G.panel(s, L, 1.85, 1.75, 2.55, None, "Hai tập mảnh")
    G.node(s, 0.75, 2.35, 1.49, 0.8, ["thật: 198", "khung test"], "data", 12)
    G.node(s, 0.75, 3.35, 1.49, 0.8, ["giả: 600 ảnh", "D1, D2, D2+SR"], "data", 12)
    specs = [("yel", "Thước 1: thống kê mức thấp"), ("lav", "Thước 2: C2ST"), ("blu", "Thước 3: KID")]
    xs = stage_row(s, 2.52, 1.85, 2.55, [3.3, 3.3, 3.3], 0.1, specs)
    G.eq(s, "stat1", xs[0] + 1.65, 2.3, 0.75, align="c")
    G.eq(s, "w1", xs[0] + 1.65, 2.98, 0.62, align="c")
    text(s, xs[0] + 0.12, 3.42, 3.06, 0.3, "18 đại lượng: màu, dark, nhiễu, khối, góc", 11, TEXT, align="c")
    G.eq(s, "hp", xs[1] + 0.15, 2.45, 0.75)
    G.node(s, xs[1] + 1.7, 2.32, 1.45, 0.42, "CNN 4 lớp", "learn", 11)
    G.arrow(s, xs[1] + 1.48, 2.53, xs[1] + 1.7, 2.53, INK, 1.25)
    G.eq(s, "c2st", xs[1] + 1.65, 2.98, 0.75, align="c")
    G.node(s, xs[2] + 0.12, 2.32, 1.45, 0.42, "DINOv2 ViT-L/14", "frozen", 10)
    G.arrow(s, xs[2] + 1.57, 2.53, xs[2] + 1.7, 2.53, INK, 1.25)
    G.eq(s, "kernel", xs[2] + 1.72, 2.41, 0.4)
    G.eq(s, "kid2", xs[2] + 1.65, 2.9, 0.75, align="c")
    for x, lab in zip(xs, ["W₁ chia IQR của tập thật", "mảnh 64×64; CNN 60 929 tham số", "CLS 1024 chiều, 10 tập con"]):
        G.bar(s, x + 0.12, 3.96, 3.06, 0.34, "lav" if "64" in lab else ("yel" if "IQR" in lab else "blu"), lab, 12)
    gap_arrows(s, [(L + 1.75, xs[0])], 3.1)
    cats = ["D1", "D2", "D2+SR", "thật–thật"]
    data = [("Lệch thống kê ↓", [5.928, 0.796, 0.663, 0.330], 6.5, [vn(v) for v in [5.928, 0.796, 0.663, 0.330]], None),
            ("C2ST, 0,5 là tốt", [0.936, 0.950, 0.923, 0.542], 1.0, [vn(v) for v in [0.936, 0.950, 0.923, 0.542]], 0.5),
            ("KID ↓", [4.443, 4.409, 3.994, 0.371], 5.0, [vn(v) for v in [4.443, 4.409, 3.994, 0.371]], None)]
    cols = [BLUE, TEAL, PURPLE, SLATE]
    for i, (ttl, vals, vmax, labs, ref) in enumerate(data):
        x, w = xs[i], 3.3
        card(s, x, 4.52, w, 2.43)
        text(s, x + 0.2, 4.62, w - 0.4, 0.3, ttl, 14, INK, True)
        LAY = (0.26, 0.04, 0.6, 0.9)
        gf = kit.bar_chart(s, x + 0.08, 4.95, w - 0.16, 1.95, cats,
                           [("v", vals, cols[0], labs, {k: c for k, c in enumerate(cols)}, LP.OUTSIDE_END)],
                           horizontal=True, gap=35, vmin=0, vmax=vmax, layout=LAY)
        if ref is not None:
            px, py, pw, ph = kit.plot_xy(x + 0.08, 4.95, w - 0.16, 1.95, LAY)
            xx = px + ref / vmax * pw
            kit.line(s, xx, py, xx, py + ph, PINK, 1.5, dash=True)
    G.detail(s, L, 4.52, 1.75, 2.43, "Đọc bảng", ["Hàng thật–thật", "là mức sàn:", "600 khung dev", "so 198 khung", "test."], 12)
    C.notes(s, ["Ba thước, đo trên mảnh cắt từ cột 16, không đổi cỡ, so với 198 khung test.",
                "Thống kê mức thấp: D1 5,928, D2 0,796, D2+SR 0,663, sàn thật–thật 0,330.",
                "C2ST: CNN nhỏ học tách mảnh thật khỏi mảnh giả trên phần dư I − Gauss(σ=2); D1 0,936, D2 0,950, còn xa 0,5.",
                "KID: MMD² với nhân đa thức bậc 3 trên đặc trưng DINOv2; D1 4,443, D2 4,409."],
            "results/reproduce/tables/t2_do_giong_that.md; xla/eval/c2st.py; xla/eval/kid.py:39-56; xla/eval/realism.py.")
    return s
