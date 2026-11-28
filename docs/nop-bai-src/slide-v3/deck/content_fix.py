from content import BODY
def put(no, old, new):
    BODY.setdefault(no, {})[old] = new
put(3, "Dark channel p50 144/255; góc 0,81× tâm.", "Dark channel 144/255. Góc chỉ sáng 0,81 tâm.")
put(5, "Mù theo mô hình khí quyển", "Phủ màn mù")
put(5, "Cắt kênh → ám hồng, nhiễu", "Cắt kênh, thêm nhiễu")
put(6, "Khớp thống kê là cần nhưng chưa đủ: nội dung ảnh VisDrone vẫn khác bãi bay của nhóm.", "Có thể bộ phân biệt nhận ra cảnh phố VisDrone, khác bãi bay. Chưa kiểm riêng.")
put(7, "NAFNet-w16 20,7 ms trên GPU", "NAFNet 20,7 ms trên GPU")
put(13, "D2: 0,30 hộp/khung bay không người, D1: 0,99. Chênh có CI trọn dưới 0.", "D2 hơn D1 0,093 nhưng CI còn chứa 0.")
put(18, "Trong IOT-CV: theo kịp camera, nhưng p95 vượt ngân sách", "Trong IOT-CV p95 tổng lên 54,1 ms, vượt 50 ms")
put(18, "detector · p50 ở nhịp camera thật", "detector · p50, nhịp thật")
put(18, "tổng hai model · p50 · 7,2 khung/s", "hai model · p50 · 7,2 khung/s")
put(18, "Phát lại chuyến 02/11 · ORT CUDA, RTX 4060 Laptop · bỏ 0,3 % khung · p95 tổng 54,1 ms > ngân sách 50 ms", "Phát lại chuyến 02/11 trên RTX 4060 Laptop, ORT CUDA. Bỏ 0,3 % khung.")
put(19, "detector-v1.0.0 · enhancer-v1.0.0 · ORT CUDA", "detector-v1.0.0 · enhancer-v1.0.0")
put(20, "Ở nhịp camera, GPU hạ xung giữa hai khung: p95 tổng không đạt 50 ms.", "GPU hạ xung giữa hai khung; p95 tổng 54,1 ms.")
put(22, "scripts/reproduce.py --check  ·  results/reproduce/", "scripts/reproduce.py --check, kết quả ở results/reproduce/")
put(23, "Kết quả và hướng phát triển", "D2 bớt báo nhầm; ảnh rõ chỉ để người xem")
