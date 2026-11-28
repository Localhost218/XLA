"""One-time text fixes applied to the copied v3 slide modules (run from slide-v4/deck)."""
import io
import sys


def rep(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for a, b in pairs:
        if a not in s:
            if b in s:
                continue
            sys.exit(f"MISSING in {path}: {a[:80]}")
        s = s.replace(a, b)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


rep("slides1.py", [
    ('("K: 16 thống kê mức thấp + đường NLF + phổ nhiễu; W₁ là khoảng cách Wasserstein-1"',
     '("K: 18 đại lượng (màu, dark channel, nhiễu, độ nét, khối, tối góc, dải mép, phổ nhiễu); W₁: Wasserstein-1"'),
    ('(L, 1.75, "D1 đối chứng", ["Real-ESRGAN bậc hai,", "không đọc hồ sơ"])',
     '(L, 1.75, "D1 đối chứng", ["Real-ESRGAN", "bậc hai"])'),
    ("; giá trị là MMD² thô, nhãn '×100' trong bảng t2 không khớp code.\"", ".\""),
    ('"Luật đo hồng R−G > 25 và B−G > 25 không có ngưỡng độ sáng; ngưỡng 230 chỉ có trong bộ lọc."',
     '"Luật đo hồng R−G > 25 và B−G > 25 chỉ để đếm điểm; cổng sửa của bộ lọc là max(R,B) ≥ 230 và min(R,B) − G ≥ 6."'),
])
rep("slides1b.py", [
    ('("Tiêu chí chốt trước khi chạy: ", {"bold": True, "color": INK}),\n'
     '                                     ("mAP50 ≥ 0,227 và cận dưới CI của hiệu so với mốc > 0; báo nhầm ≤ 0,05 hộp mỗi khung bay không người; ≤ 30 ms p95 trên ORT CUDA.", {})',
     '("Tập chấm: ", {"bold": True, "color": INK}),\n'
     '                                     ("198 khung test (39 có người, 149 người), 159 khung bay trống, 300 khung trong nhà. v1 và v2 chấm cùng đường, so ghép cặp trên cùng mẫu khối.", {})'),
    ('C.header(s, "orange", "08 · HUẤN LUYỆN", "Bộ nhận diện qua bốn giai đoạn, từ COCO tới gói ONNX")',
     'C.header(s, "orange", "08 · HUẤN LUYỆN", "Detector v1 qua bốn giai đoạn, từ COCO tới gói ONNX")'),
])
rep("slides2.py", [
    ('("E2 thầy YOLO11m", 0.218', '("YOLO11m (thầy dự kiến)", 0.218'),
    ('"Vòng 2 cao nhất (0,313), nhưng CI của 9 bộ chồng lên nhau"',
     '"Chín bộ nhận diện v1: vòng 2 cao nhất (0,313), CI chồng nhau"'),
    ("C.notes(s, [\"Đủ chín dòng của bảng t1; mốc ghi đúng là YOLO11s COCO ở 1 280 px (bảng t1 gốc ghi nhầm 'YOLO11n').\",",
     'C.notes(s, ["Đủ chín dòng của bảng t1; mốc là YOLO11s COCO ở 1 280 px.",'),
    ('"10 · LỌC CỔ ĐIỂN", "Bộ lọc cổ điển đi ngược chuỗi hỏng: 9,2 ms trên CPU"',
     '"12 · LỌC CỔ ĐIỂN", "Bộ lọc cổ điển đi ngược chuỗi hỏng qua năm tầng"'),
    ('"10 · LỌC CỔ ĐIỂN", "Khử mù: ước A và t̃ từ dark channel, làm mịn bằng guided filter"',
     '"12 · LỌC CỔ ĐIỂN", "Khử mù: ước A và t̃ từ dark channel, làm mịn bằng guided filter"'),
    ('G.node(s, xs[1] + 0.15, 2.3, 2.2, 0.48, "mặt nạ α từ max(R,B) ≥ 230", "learn", 11)',
     'G.node(s, xs[1] + 0.15, 2.3, 2.2, 0.48, ["cổng: max(R,B) ≥ 230", "và min(R,B) − G ≥ 6"], "learn", 10)'),
    ('["g bật khi R hoặc B gần trần và vượt G; e là phần R, B dư so với G."]',
     '["Cổng sửa g (bộ lọc): R hoặc B gần trần và cả hai vượt G. R−G > 25, B−G > 25 chỉ là luật đếm điểm hồng."]'),
    ('(L, 1.5, "Tổng", ["p50 9,18 ms", "p95 10,21 ms"])', '(L, 1.5, "Tổng p50", ["numpy 9,6 ms", "ONNX 19,3 ms"])'),
    ('"Code thêm hai chốt: t ≥ 1 − ω·min(I)/Ā tại từng điểm, rồi kéo mức đen về ≤ 8 sau khôi phục."',
     '"Code thêm hai chốt: t ≥ 1 − ω·min(I)/Ā tại từng điểm; kéo giãn khi mức đen ra thấp hơn min(mức đen vào; 8)."'),
    ('"11 · NAFNET", "NAFNet-w16 không dùng hàm kích hoạt, 1,14 triệu tham số"',
     '"13 · SO SÁNH LÀM RÕ", "NAFNet-w16 không dùng hàm kích hoạt, 1,14 triệu tham số"'),
    ('"Bản rộng 32 (29,2 triệu tham số) chạy 82,1 ms p95, vượt trần 30 ms nên đổi sang w16. Checkpoint chọn theo nhấp nháy ít nhất (1,779) vì không bước nào giữ dưới 1,15. ONNX fp16, opset 17: 15,4 / 21,0 ms p50 / p95."',
     '"NAFNet-w16 là enhancer-v1.0.0 (ONNX fp16, opset 17). Đo xen kẽ trong cùng tiến trình: NAFNet 20,6 / 27,9 ms p50 / p95, classic ONNX 19,3 / 25,3 ms. Bản rộng 32 (29,2 triệu tham số) cần 82,1 ms p95 nên chọn w16."'),
    ('"Bundle enhancer-v1.0.0 không bật mặc định trong IOT-CV."]',
     '"v2 thay NAFNet bằng classic ONNX enhancer-v2.0.0; bảng so sánh ở slide sau."]'),
])
print("ok")
