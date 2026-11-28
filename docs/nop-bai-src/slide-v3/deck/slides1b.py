"""v8 slides for outline sections 7-10 (YOLO11s, metrics, training, dev-val vs test)."""
from pptx.enum.chart import XL_LABEL_POSITION as LP

import common as C
import kit
import stage as G
from kit import BLUE, INK, MUTED, ORANGE, PINK, PURPLE, TEXT, card, text, vn
from slides1 import L, W, gap_arrows, stage_row


# ------------------------------------------------------------------ §7 YOLO11s
def s_yolo():
    s = C.new("orange")
    C.header(s, "orange", "06 · YOLO11S", "YOLO11s đọc ba tầng P3–P5 và học hộp bằng CIoU cộng DFL")
    ws = [1.55, 3.7, 2.85, 3.4]
    specs = [(None, "Ảnh vào"), ("yel", "Backbone"), ("lav", "Neck FPN + PAN"), ("blu", "Head Detect, không anchor")]
    xs = stage_row(s, L, 1.85, 2.45, ws, 0.2, specs)
    G.node(s, xs[0] + 0.12, 2.35, 1.31, 0.62, ["khung", "640×480"], "data", 12)
    G.node(s, xs[0] + 0.12, 3.25, 1.31, 0.82, ["cắt 16 cột,", "letterbox", "1 280"], "data", 11)
    G.arrow(s, xs[0] + 0.78, 2.97, xs[0] + 0.78, 3.25, INK, 1.25)
    bb = [("Conv s2 ×2", 0.72), ("C3k2 + Conv s2, ×3", 1.12), ("SPPF", 0.62), ("C2PSA", 0.72)]
    x = xs[1] + 0.12
    for k, (lab, w) in enumerate(bb):
        G.node(s, x, 2.38, w, 0.72, lab, "learn", 11)
        if k < 3:
            G.arrow(s, x + w, 2.74, x + w + 0.12, 2.74, INK, 1.25)
        x += w + 0.12
    for k, (lab, c) in enumerate([("P3 / 8", BLUE), ("P4 / 16", PURPLE), ("P5 / 32", PINK)]):
        kit.chip(s, xs[1] + 0.25 + k * 1.15, 3.35, 1.05, 0.3, lab, c, 11)
    text(s, xs[1] + 0.15, 3.75, 3.4, 0.4, "kênh 128 · 256 · 512 sau hệ số rộng 0,50", 11, MUTED, align="c")
    G.node(s, xs[2] + 0.15, 2.38, 2.55, 0.72, ["xuống: Upsample ×2,", "Concat, C3k2"], "learn", 11)
    G.node(s, xs[2] + 0.15, 3.3, 2.55, 0.72, ["lên: Conv bước 2,", "Concat, C3k2"], "learn", 11)
    G.arrow(s, xs[2] + 1.42, 3.1, xs[2] + 1.42, 3.3, INK, 1.25)
    G.node(s, xs[3] + 0.15, 2.38, 3.1, 0.5, "mỗi tầng P3, P4, P5", "plain", 12)
    G.node(s, xs[3] + 0.15, 3.05, 1.5, 0.95, ["nhánh hộp", "4 × 16 bin"], "learn", 11)
    G.node(s, xs[3] + 1.75, 3.05, 1.5, 0.95, ["nhánh lớp", "1 kênh: người"], "learn", 11)
    G.arrow(s, xs[3] + 0.9, 2.88, xs[3] + 0.9, 3.05, INK, 1.25)
    G.arrow(s, xs[3] + 2.5, 2.88, xs[3] + 2.5, 3.05, INK, 1.25)
    gap_arrows(s, [(xs[i] + ws[i], xs[i + 1]) for i in range(3)], 2.74)
    cards = [("Đầu vào", ["nền 114,", "chia 255"]),
             ("Backbone", ["SPPF: 3 lần maxpool 5×5 nối lại;", "C2PSA: tự chú ý 4 đầu trên P5"]),
             ("Neck", ["trộn tầng nông và sâu,", "ra 3 bản đồ bước 8, 16, 32"]),
             ("Head", ["mỗi ô đoán khoảng cách tới 4 cạnh;", "điểm lớp qua sigmoid, NMS sau"])]
    for x, w, (lab, lines) in zip(xs, ws, cards):
        G.detail(s, x, 4.42, w, 1.0, lab, lines, 12 if lab == "Đầu vào" else 13)
    card(s, L, 5.55, 4.75, 1.4); card(s, 5.52, 5.55, 4.15, 1.4); card(s, 9.82, 5.55, 2.9, 1.4)
    G.eq(s, "yloss", 0.8, 5.63, 0.8)
    G.eq(s, "ciou", 0.8, 5.92, 0.8)
    G.eq(s, "dfl", 5.7, 5.65, 0.8)
    text(s, 5.7, 6.32, 3.85, 0.6, "S: softmax trên 16 bin; nhãn y nằm giữa hai bin l và r, cạnh hộp là kỳ vọng.", 12, TEXT)
    G.eq(s, "tal", 10.0, 5.66, 0.9)
    text(s, 11.0, 5.62, 1.6, 0.5, "s: điểm lớp, u: CIoU", 12, TEXT)
    text(s, 10.0, 5.98, 2.6, 0.3, "gán 10 ô có t lớn nhất", 12, TEXT)
    G.eq(s, "bce", 10.0, 6.38, 0.8)
    C.notes(s, ["YOLO11s, ultralytics 8.4.215: bản s có hệ số sâu 0,50, rộng 0,50; ba đầu ra P3/8, P4/16, P5/32.",
                "C3k2 là C2f với khối bottleneck; SPPF nối 3 lần maxpool 5×5; C2PSA là khối tự chú ý trên P5.",
                "Head tách nhánh hộp và nhánh lớp, không anchor; hộp học bằng DFL 16 bin.",
                "Loss = 7,5 CIoU + 0,5 BCE + 1,5 DFL; gán nhãn TaskAligned t = s^0,5 u^6, top 10. Học chuyển từ COCO 80 lớp sang 1 lớp người, ảnh vào 1 280."],
            "ultralytics 8.4.215: cfg/models/11/yolo11.yaml, nn/modules/block.py, head.py, utils/loss.py, utils/tal.py, utils/metrics.py:159-171, cfg/default.yaml.")
    return s


# ------------------------------------------------------------------ §8 metrics pipeline
def s_metrics():
    s = C.new("orange")
    C.header(s, "orange", "07 · CHỈ SỐ", "mAP50 chấm trên 198 khung thật, CI 95 % bootstrap theo khối")
    G.panel(s, L, 1.85, 1.6, 2.6, None, "Đầu vào")
    G.node(s, 0.74, 2.32, 1.36, 0.85, ["hộp dự đoán", "+ điểm s"], "data", 12)
    G.node(s, 0.74, 3.38, 1.36, 0.85, ["nhãn tay", "149 hộp"], "data", 12)
    specs = [("yel", "1. Ghép theo IoU"), ("lav", "2. Đường P–R"), ("blu", "3. AP và mAP"), ("mint", "4. Bootstrap khối")]
    xs = stage_row(s, 2.37, 1.85, 2.6, [2.47] * 4, 0.12, specs)
    G.eq(s, "iou", xs[0] + 1.235, 2.42, 0.72, align="c")
    G.eq(s, "pr", xs[1] + 1.235, 2.45, 0.66, align="c")
    G.eq(s, "ap", xs[2] + 1.235, 2.36, 0.66, align="c")
    G.eq(s, "boot", xs[3] + 1.235, 2.48, 0.57, align="c")
    sub = [["TP khi IoU ≥ 0,5,", "mỗi nhãn ghép 1 lần"], ["quét ngưỡng điểm", "từ cao xuống thấp"],
           ["101 điểm recall,", "đường bao giảm dần"], ["13 khối bay + 213 khối", "âm, rút riêng từng loại"]]
    for x, t in zip(xs, sub):
        text(s, x + 0.1, 3.1, 2.27, 0.6, t, 12, TEXT, align="c")
    for x, pal, lab in zip(xs, ["yel", "lav", "blu", "mint"], ["TP, FP, FN", "P(r) trên 198 khung", "mAP50, mAP50-95", "CI 95 %, 2 000 lần"]):
        G.bar(s, x + 0.12, 3.95, 2.23, 0.36, pal, lab, 12)
    gap_arrows(s, [(L + 1.6, xs[0])] + [(xs[i] + 2.47, xs[i + 1]) for i in range(3)], 2.75)
    cards = [(L, 1.6, "Đầu ra", ["≤ 100 hộp,", "s ≥ 0,001"]),
             (xs[0], 2.47, "Cách ghép", ["pycocotools,", "một lớp: người"]),
             (xs[1], 2.47, "Ngưỡng vận hành", ["F1 lớn nhất trên dev-val;", "đụng test là báo lỗi"]),
             (xs[2], 2.47, "Hai mức IoU", ["mAP50: IoU 0,5;", "mAP50-95: 10 mức"]),
             (xs[3], 2.47, "Vì sao theo khối", ["khung trong khối giống", "nhau, rút lẻ cho CI hẹp giả"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 4.6, w, 1.05, lab, lines, 13 if w > 2 else 12)
    card(s, L, 5.8, W, 1.15)
    G.eq(s, "fp", 0.85, 5.9, 0.82)
    G.eq(s, "map", 3.75, 5.85, 0.72)
    text(s, 5.95, 5.93, 6.6, 0.95, [[("Tiêu chí chốt trước khi chạy: ", {"bold": True, "color": INK}),
                                     ("mAP50 ≥ 0,227 và cận dưới CI của hiệu so với mốc > 0; báo nhầm ≤ 0,05 hộp mỗi khung bay không người; ≤ 30 ms p95 trên ORT CUDA.", {})]], 13, TEXT)
    C.notes(s, ["Hộp dự đoán ghép với nhãn tay theo IoU ≥ 0,5; quét ngưỡng để có đường precision–recall.",
                "AP lấy trung bình precision bao trên 101 mức recall; mAP50-95 lấy trung bình 10 mức IoU 0,50 đến 0,95.",
                "Báo nhầm: số hộp có điểm ≥ ngưỡng F1 dev-val trên mỗi khung bay không người (159 khung).",
                "CI 95 %: bootstrap 2 000 lần theo khối 30 s, phân vị 2,5 và 97,5; hai model so trên cùng mẫu."],
            "xla/eval/detection.py:50-58,216-326,351-355; ultralytics/utils/metrics.py:775-796; báo cáo LaTeX mục 3.5.")
    return s


# ------------------------------------------------------------------ §9-10 training stages
def s_training():
    s = C.new("orange")
    C.header(s, "orange", "08 · HUẤN LUYỆN", "Bộ nhận diện qua bốn giai đoạn, từ COCO tới gói ONNX")
    G.panel(s, L, 1.85, 1.6, 2.95, None, "Đầu vào")
    for k, (lab, kind) in enumerate([("YOLO11s COCO", "frozen"), ("YOLO11m COCO", "frozen"),
                                     ("VisDrone D0–D2+SR", "data"), ("1 115 khung dev", "data")]):
        G.node(s, 0.72, 2.3 + k * 0.6, 1.4, 0.48, lab, kind, 11)
    specs = [("yel", "GĐ 1: mốc COCO"), ("lav", "GĐ 2: E1 tinh chỉnh"), ("blu", "GĐ 3: E2 tự huấn luyện"), ("mint", "GĐ 4: xuất ONNX")]
    xs = stage_row(s, 2.37, 1.85, 2.95, [2.47] * 4, 0.12, specs)
    nodes = [[("YOLO11s COCO", "frozen"), ("chỉ giữ lớp người", "plain"), ("bỏ hộp có x1 < 16", "plain")],
             [("đầu 80 lớp thành 1 lớp", "learn"), ("D0 · D1 · D2 · D2+SR", "learn"), ("giữ epoch dev-val tốt nhất", "plain")],
             [("thầy vòng 1: E1 D2", "learn"), ("nhãn giả: TTA + ByteTrack", "plain"), ("trò YOLO11s từ E1 D2", "learn")],
             [("opset 17, NMS trong graph", "learn"), ("meta.json + sha256", "plain"), ("3 khung golden", "plain")]]
    bars = ["test mAP50 0,126", "chọn D2: dev-val 0,329", "ship r1: dev-val 0,379", "IOT-CV kiểm rồi nạp"]
    for x, ns, b, pal in zip(xs, nodes, bars, ["yel", "lav", "blu", "mint"]):
        for k, (lab, kind) in enumerate(ns):
            y = 2.3 + k * 0.66
            G.node(s, x + 0.14, y, 2.19, 0.48, lab, kind, 11)
            if k < 2:
                G.arrow(s, x + 1.235, y + 0.48, x + 1.235, y + 0.66, INK, 1.25)
        G.bar(s, x + 0.12, 4.33, 2.23, 0.36, pal, b, 12)
    x3 = xs[2]  # E2 loop: the student of round k teaches round k+1
    kit.line(s, x3 + 2.33, 3.86, x3 + 2.42, 3.86, INK, 1.25)
    kit.line(s, x3 + 2.42, 3.86, x3 + 2.42, 2.54, INK, 1.25)
    G.arrow(s, x3 + 2.42, 2.54, x3 + 2.33, 2.54, INK, 1.25)
    text(s, x3 + 1.0, 4.02, 1.38, 0.26, "vòng 2: thầy = r1", 10, "5B3FA0", True, align="r")
    gap_arrows(s, [(L + 1.6, xs[0])] + [(xs[i] + 2.47, xs[i + 1]) for i in range(3)], 2.54)
    cards = [(L, 1.6, "Dữ liệu", ["3 006 + 487 cửa sổ,", "không khung thật"]),
             (xs[0], 2.47, "Giai đoạn 1", ["1 280 px; bỏ dải mép", "nâng 0,067 lên 0,126"]),
             (xs[1], 2.47, "Giai đoạn 2", ["6 epoch, 1 280 px, batch 4,", "AdamW lr 0,002, HSV tắt"]),
             (xs[2], 2.47, "Giai đoạn 3", ["τ cao 0,55 rồi 0,75; vệt ≥ 3", "khung; 4 epoch mỗi vòng"]),
             (xs[3], 2.47, "Giai đoạn 4", ["ngưỡng 0,0767, NMS 0,5,", "tối đa 100 hộp"])]
    for x, w, lab, lines in cards:
        G.detail(s, x, 4.92, w, 1.08, lab, lines, 13 if w > 2 else 12)
    card(s, L, 6.12, W, 0.83)
    text(s, 0.85, 6.2, 11.7, 0.7, "Thầy YOLO11m tinh chỉnh trên D2 cho nhãn giả kém E1 D2 (recall 0,075 so với 0,084, cùng precision 1,0) nên không làm thầy vòng 1. Trò luôn khởi tạo từ E1 D2 và học D2 + khung dev có nhãn giả (lặp 2 lần) + 300 khung âm.", 13, TEXT)
    C.notes(s, ["GĐ1 mốc: YOLO11s COCO ở 1 280 px, chỉ giữ lớp người, bỏ hộp nằm trên dải mép.",
                "GĐ2 E1: bốn lần chạy chỉ khác tập làm xấu; chọn checkpoint theo mAP50 dev-val; D2 thắng (0,329).",
                "GĐ3 E2: nhãn giả vòng 1 lấy từ E1 D2 (không phải YOLO11m); ByteTrack cứu hộp điểm thấp thuộc vệt ≥ 3 khung; r1 dev-val 0,379 nên ship r1.",
                "GĐ4: ONNX opset 17 với letterbox, NMS trong graph; meta.json, sha256, 3 khung golden."],
            "configs/train_e1.yaml; configs/selftrain_e2.yaml; xla/detect/selftrain.py; results/e1/e1.json; results/e2/pseudo_r1.json, pseudo_r2.json, e2.json; models/release/detector-v1.0.0/meta.json.")
    return s


# ------------------------------------------------------------------ §9-10 results: dev-val vs test
def s_train_results():
    s = C.new("orange")
    C.header(s, "orange", "08 · HUẤN LUYỆN", "Dev-val xếp vòng 1 đầu, test lại xếp vòng 2 đầu")
    card(s, L, 1.85, 7.35, 5.1)
    text(s, 0.9, 2.0, 6.9, 0.3, "MAP50 · DEV-VAL 100 KHUNG SO VỚI TEST 198 KHUNG", 12, ORANGE, True)
    cats = ["E1 D0", "E1 D1", "E1 D2", "E1 D2+SR", "E2 thầy YOLO11m", "E2 trò vòng 1", "E2 trò vòng 2"]
    dev = [0.295, 0.176, 0.329, 0.299, 0.267, 0.379, 0.296]
    tst = [0.297, 0.143, 0.236, 0.149, 0.218, 0.156, 0.313]
    LAY = (0.3, 0.02, 0.62, 0.9)
    kit.bar_chart(s, 0.75, 2.35, 7.1, 4.25, cats,
                  [("dev-val", dev, "F2C18D", [vn(v) for v in dev], None, LP.OUTSIDE_END),
                   ("test", tst, ORANGE, [vn(v) for v in tst], None, LP.OUTSIDE_END)],
                  horizontal=True, gap=45, vmin=0, vmax=0.45, layout=LAY)
    kit.chip(s, 5.55, 2.0, 1.05, 0.28, "dev-val", "E8A65C", 11)
    kit.chip(s, 6.7, 2.0, 0.9, 0.28, "test", ORANGE, 11)
    card(s, 8.17, 1.85, 4.55, 5.1)
    text(s, 8.42, 2.0, 4.1, 0.34, "Nhãn giả của E2 trên dev-val", 17, INK, True)
    kit.table(s, 8.42, 2.45, [1.3, 1.35, 1.35], 0.4, [
        ["", "Vòng 1", "Vòng 2"],
        ["thầy", "E1 D2", "trò r1"],
        ["τ cao", "0,55", "0,75"],
        ["precision", "1,000", "0,857"],
        ["recall", "0,084", "0,112"],
        ["ảnh học", "4 462", "4 452"]], size=13, aligns="lrr")
    text(s, 8.42, 5.0, 4.1, 1.9, ["Recall 8–11 %: gần như mọi người thật trên khung dev bị coi là nền.",
                                  "Dev-val cùng chuyến với khung nhãn giả, đệm chỉ 0,3 s, nên ưu ái các vòng E2."], 13, TEXT, spacing=1.1)
    C.notes(s, ["Luật chọn: checkpoint và model theo mAP50 dev-val; test không dùng để chọn.",
                "Dev-val: E1 D2 0,329, r1 0,379, r2 0,296, thầy 0,267. Test: E1 D2 0,236, r1 0,156, r2 0,313.",
                "Dev-val chọn r1 nhưng trên test r1 chỉ 0,156; ba nguyên nhân khả dĩ ở slide thảo luận.",
                "Nhãn giả có precision cao nhưng recall rất thấp (9/107 và 12/107 người)."],
            "results/e1/e1.json; results/e2/e2.json, pseudo_r1.json, pseudo_r2.json; t1_nhan_dien.md.")
    return s
