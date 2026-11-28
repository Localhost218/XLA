"""Round 3: drop every version mention (v1/v2, bundle version suffixes, 'ship'); compare by method name."""
import io
import sys


def rep(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for a, b in pairs:
        if a not in s:
            if b in s:
                continue
            sys.exit(f"MISSING in {path}: {a[:90]}")
        s = s.replace(a, b)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


rep("slides1b.py", [
    ("v1 và v2 chấm cùng đường, so ghép cặp trên cùng mẫu khối.", "Một tầng và ba tầng chấm cùng đường, so ghép cặp trên cùng mẫu khối."),
    ('"Detector v1 qua bốn giai đoạn, từ COCO tới gói ONNX"', '"YOLO11s qua bốn giai đoạn, từ COCO tới gói ONNX"'),
    ('"ship r1: dev-val 0,379"', '"chọn r1: dev-val 0,379"'),
    ("r1 dev-val 0,379 nên ship r1.", "r1 dev-val 0,379 nên chọn r1."),
])
rep("slides2.py", [
    ('("Bundle detector-v1.0.0", 0.120', '("E2 vòng 1, gói ONNX", 0.120'),
    ('"Chín bộ nhận diện v1: vòng 2 cao nhất (0,313), CI chồng nhau"', '"Chín cấu hình huấn luyện YOLO11s: vòng 2 cao nhất (0,313), CI chồng nhau"'),
    ('"Bundle ship là E2 vòng 1 xuất ONNX, chấm bằng đúng đường ONNX: 0,120."', '"Gói ONNX nhận diện là E2 vòng 1, chấm bằng đúng đường ONNX: 0,120."'),
    ("NAFNet-w16 là enhancer-v1.0.0 (ONNX fp16, opset 17).", "NAFNet-w16 xuất ONNX fp16, opset 17."),
    ('"v2 thay NAFNet bằng classic ONNX enhancer-v2.0.0; bảng so sánh ở slide sau."', '"Luồng Ảnh rõ dùng bộ lọc cổ điển ONNX thay NAFNet; bảng so sánh ở slide sau."'),
])
rep("slides_v2.py", [
    # slide 3 system
    ('"AI v2: ba tầng tìm người, một luồng Ảnh rõ chạy song song"', '"Hệ tìm người ba tầng, một luồng Ảnh rõ chạy song song"'),
    ('("detector-v1.0.1", "learn")', '("gói ONNX nhận diện", "learn")'),
    ('("enhancer-v2.0.0", "learn")', '("gói ONNX làm rõ", "learn")'),
    ('["ONNX của v1.0.0", "+ recommended_track"]', '["ONNX của E2 vòng 1", "+ recommended_track"]'),
    ('"Tầng 1 là detector v1 (YOLO11s, ONNX giữ nguyên)', '"Tầng 1 là YOLO11s E2 (gói ONNX nhận diện)'),
    ('"Ảnh rõ là bộ lọc cổ điển xuất ONNX (enhancer-v2.0.0), bật sẵn trên web."', '"Ảnh rõ là bộ lọc cổ điển xuất ONNX (gói ONNX làm rõ), bật sẵn trên web."'),
    ('"Hai câu hỏi của v2: tầng xác minh', '"Hai câu hỏi: tầng xác minh'),
    # slide 7
    ("Ba bộ ảnh nhìn từ trên xuống thêm cho v2:", "Ba bộ ảnh nhìn từ trên xuống thêm cho hệ ba tầng:"),
    # slide 14
    ('"Thầy YOLO11m, trò C1 và C3 so với v1: giữ detector v1"', '"Thầy YOLO11m, trò C1 và C3 so với YOLO11s E2: giữ E2"'),
    ('("v1 E2 r1 (.pt)", 0.340', '("YOLO11s E2 r1 (.pt)", 0.340'),
    ('("v1 bundle (ONNX)", 0.229', '("YOLO11s E2 (ONNX)", 0.229'),
    ('"v1 đã học nhãn giả khung dev nên OOF có lợi cho v1."', '"E2 đã học nhãn giả khung dev nên OOF có lợi cho E2."'),
    ('["thầy − v1",', '["thầy − E2",'), ('["C3 − v1 (.pt)",', '["C3 − E2 (.pt)",'), ('["C3 − v1 (ONNX)",', '["C3 − E2 (ONNX)",'),
    ('["v1 bundle (ONNX)", "0,172"', '["YOLO11s E2 (ONNX)", "0,172"'),
    ('"v2 thử thay detector: thầy YOLO11m', '"Thử thay YOLO11s E2: thầy YOLO11m'),
    ("mọi hiệu so với v1 có CI chứa 0 nên hệ giữ detector v1, ONNX không đổi.", "mọi hiệu so với E2 có CI chứa 0 nên hệ giữ E2, ONNX không đổi."),
    ("C3 kém v1 0,041", "C3 kém E2 0,041"),
    # slide 18
    ('"Cả hệ: mAP50 0,120 → 0,210, báo nhầm 0,881 → 0,289"', '"YOLO11s một tầng → ba tầng: mAP50 0,120 → 0,210, báo nhầm 0,881 → 0,289"'),
    ('sets = [("v1", 0.120', 'sets = [("E2 ONNX", 0.120'),
    ('"HIỆU GHÉP CẶP CẢ HỆ − V1, CÙNG MẪU KHỐI"', '"HIỆU GHÉP CẶP BA TẦNG − MỘT TẦNG, CÙNG MẪU KHỐI"'),
    ('["1 detector-v1.0.1",', '["1 gói nhận diện",'),
    ('("2 + verifier-v1.0.0",', '("2 + gói xác minh",'),
    ("Chấm một lần trên 198 khung test v1, cùng đường", "Chấm một lần trên 198 khung test, cùng đường"),
    ('"Tầng 1 trùng v1 (cùng ONNX).', '"Tầng 1 trùng E2 ONNX một tầng.'),
    # slide 19
    ('"Detector v1 nhầm túi,', '"YOLO11s nhầm túi,'),
    # slide 24
    ('("classic ONNX (v2.0.0)",', '("classic ONNX",'),
    ('["NAFNet-w16 (v1.0.0)",', '["NAFNet-w16",'),
    ('"dark −37 % (test v1)"', '"dark −37 % (test)"'),
    ('"NAFNet v1 chậm hơn classic', '"NAFNet chậm hơn classic'),
    # slide 26
    ('"16 khối, 2 phiên; detector v1 ở ngưỡng 0,0767."', '"16 khối, 2 phiên; YOLO11s ở ngưỡng 0,0767."'),
    ('"Kết luận: ship classic; offline không ship."', '"Kết luận: web dùng classic; offline ×2 không bật."'),
    # slide 27
    ('("v1", v1, "9FB3C8"', '("một tầng + NAFNet", v1, "9FB3C8"'),
    ('("v2", v2, BLUE', '("ba tầng + bộ lọc cổ điển", v2, BLUE'),
    ('kit.chip(s, 4.95, 5.55, 0.8, 0.28, "v1", "9FB3C8", 11)', 'kit.chip(s, 1.0, 5.55, 2.2, 0.28, "một tầng + NAFNet", "9FB3C8", 11)'),
    ('kit.chip(s, 5.85, 5.55, 0.8, 0.28, "v2", BLUE, 11)', 'kit.chip(s, 3.35, 5.55, 3.0, 0.28, "ba tầng + bộ lọc cổ điển", BLUE, 11)'),
    ('"4 lượt v1-v2-v2-v1 trong một tiến trình,', '"4 lượt A-B-B-A trong một tiến trình,'),
    ('"BA BUNDLE SHIP · GOLDEN CUDA VÀ CPU"', '"BA GÓI ONNX · GOLDEN CUDA VÀ CPU"'),
    ('[["Bundle", "sha256", "Golden"],', '[["Gói ONNX", "sha256", "Golden"],'),
    ('["detector-v1.0.1", "23b1da69', '["nhận diện", "23b1da69'),
    ('["verifier-v1.0.0", "d566343a', '["xác minh", "d566343a'),
    ('["enhancer-v2.0.0", "83d96e8b', '["làm rõ", "83d96e8b'),
    ('"Cả ba model báo đang chạy trên CUDA, kèm phiên bản và ms p50/p95."', '"Cả ba model báo đang chạy trên CUDA, kèm ms p50/p95."'),
    ('"IOT-CV nạp ba bundle ship; mỗi bundle qua golden', '"IOT-CV nạp ba gói ONNX; mỗi gói qua golden'),
    ('"Bench ABBA: v1 = detector + NAFNet; v2 = detector + verifier + classic ONNX,', '"Bench ABBA: A = một tầng + NAFNet; B = ba tầng + bộ lọc cổ điển ONNX,'),
    # slide 29 notes
    ('"Tài liệu cho lý thuyết, model và bộ dữ liệu dùng trong v1 và v2."', '"Tài liệu cho lý thuyết, model và bộ dữ liệu dùng trong đồ án."'),
    ('"Mã nguồn và số liệu tái lập: repo XLA, scripts/reproduce.py; kết quả v2 ở results/v2/."', '"Mã nguồn và số liệu tái lập: repo XLA, scripts/reproduce.py."'),
])
rep("build4.py", [
    ('"GVHD: PGS.TS. Phạm Văn Cường  ·  AI v2, 26/11/2026"', '"GVHD: PGS.TS. Phạm Văn Cường  ·  26/11/2026"'),
    ('"Bản AI v2: ba tầng tìm người (detector, verifier, lọc track)', '"Hệ ba tầng tìm người (nhận diện, xác minh, lọc track)'),
    ('V.s_iotcv(), V.s_conclusion(), V.s_refs()]', 'V.s_iotcv(), V.s_refs()]'),
    ('"Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660 (AI v2)"', '"Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660"'),
])
print("ok")
