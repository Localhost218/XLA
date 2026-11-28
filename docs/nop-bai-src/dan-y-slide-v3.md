# Dàn ý chung (đã sửa theo code 19/11: cổng hồng, t̃, trung bình W1/IQR, nhãn giả E2)

# slide v3 + transcript (chốt 19/11/2026)

Hợp đồng giữa agent slide và agent transcript. Slide theo đúng thứ tự mục này; transcript khớp từng slide.
TỐI ĐA 30 slide tính cả bìa và tài liệu tham khảo (người dùng chốt); mục gần nhau gộp chung slide. KHÔNG có slide phân công. Số liệu chỉ lấy từ `D:\XLA\results\reproduce\tables\t1..t6`,
`D:\XLA\BAO_CAO_DO_AN.md`, `D:\XLA\results\**` (JSON/README), `D:\XLA\configs\`, `D:\XLA\xla\` (code là chuẩn cho công thức).
Mốc là **YOLO11s COCO @1280** (thầy E2: YOLO11m). Dấu phẩy thập phân. Không nhắc ESP32-CAM / OV2640 / bản cũ.

1. Bìa
2. Bài toán: camera ESP32-S3 N16R8 + OV3660, JPEG 640×480 q12 qua Wi-Fi; 5 lỗi đo được (ám hồng, mù, vỡ khối, tối góc, dải đen 16 px); mục tiêu kép; ràng buộc thời gian thực (ngân sách 50 ms); hai câu hỏi nghiên cứu.
3. Lý thuyết mô hình ảnh suy giảm:
   - Mô hình mù khí quyển I(x) = J(x)·t(x) + A·(1 − t(x)), t(x) = e^(−β d(x)).
   - Tối góc: G(r) ≈ cos⁴θ (hoặc đúng dạng trong code/camera_profile), ảnh = J·G.
   - Nén JPEG: DCT 8×8 F(u,v) = ¼ C(u)C(v) ΣΣ f(x,y) cos((2x+1)uπ/16) cos((2y+1)vπ/16), lượng tử F_q = round(F/Q), vì sao vỡ khối; bảng DQT đo ở q12.
   - Ám hồng: biểu diễn trong RGB/không gian màu dùng trong code (luật ĐO điểm hồng: R−G > 25 và B−G > 25; cổng LỌC trong code: min(R,B) − G ≥ 6 và max(R,B) ≥ 230, có dốc).
4. Dữ liệu: chuyến bay 02/11/2026, số khung, chia theo khối 30 giây 1115/198/300 + dev-val 100 + 459 khung âm, vì sao chia theo khối (rò rỉ thời gian), 39 khung test có người, gán nhãn.
5. Làm xấu D1 vs D2: từng bước D2 với công thức/tham số; hiệu chỉnh D2 bằng CMA-ES (mục tiêu = TRUNG BÌNH 18 số hạng W1/IQR, không phải tổng); D1 ngẫu nhiên.
6. Đo độ giống thật: C2ST (bộ phân loại hai mẫu, độ chính xác → 0,5 là không phân biệt được), KID = MMD² với nhân đa thức k(x,y) = (xᵀy/d + 1)³ trên đặc trưng (đúng đặc trưng code dùng, vd. DINOv2/Inception), khoảng cách thống kê; kết quả D1 vs D2.
7. Lý thuyết bộ nhận diện: YOLO11s (backbone, neck, head anchor-free tách nhánh; khối C3k2, SPPF, C2PSA, kiểm qua tài liệu Ultralytics/context7, không bịa), hàm mất mát (box CIoU + DFL + BCE phân lớp), NMS, học chuyển giao từ COCO sang 1 lớp người; cỡ vào 1280.
8. Chỉ số đánh giá: IoU, precision/recall, AP = ∫₀¹ p(r) dr (nội suy), mAP50 vs mAP50-95, FP/khung bay không người, bootstrap theo khối cho CI 95 %.
9. E1: fine-tune trên D0/D1/D2/D2+SR, siêu tham số chính; kết quả.
10. E2 tự huấn luyện: nhãn giả, ngưỡng, thầy YOLO11m → trò YOLO11s, vòng 1/2 (nhãn giả vòng 1 lấy từ E1 D2, không phải YOLO11m: recall 0,084 so với 0,075); chọn theo dev-val.
11. So sánh 9 bộ nhận diện: bảng đủ chỉ số, mAP50 ± CI, scatter mAP50 với FP, cùng khung 4 bộ.
12. Làm rõ cổ điển, lý thuyết từng tầng theo đúng thứ tự code (deblock → khử hồng → bù tối góc → khử mù): công thức mặt nạ hồng; bù tối góc chia cho G; dark channel prior J^dark(x) = min_c min_{y∈Ω(x)} J^c(y), ước lượng A, t̃ = 1 − ω·dark/mean(A) theo code (ω 0,85, t ≥ 0,4), guided filter, khôi phục J = (I − A)/max(t, t₀) + A; ms mỗi tầng.
13. NAFNet: khối NAFBlock (LayerNorm, conv, SimpleGate X₁⊙X₂, Simplified Channel Attention), không hàm kích hoạt phi tuyến, cấu hình w16 thật, hàm mất mát và dữ liệu cặp huấn luyện thật trong code.
14. Đánh giá ảnh rõ: các tiêu chí t3 (dark channel, tỉ lệ hồng, nhấp nháy…, đúng định nghĩa), A/B mù với khoảng Wilson p̂ ± …, kết quả 24/25, 11/25, 13/25; so sánh cùng khung có phóng to.
15. E4: làm rõ trước khi nhận diện, kết quả hai bộ, vì sao giảm (lệch phân phối với ảnh huấn luyện).
16. Xuất ONNX + hợp đồng model (I/O, sha256, phiên bản) + tích hợp IOT-CV + thời gian chạy từng thành phần so ngân sách 50 ms (p50/p95).
17. Bảng "Chọn gì và vì sao".
18. Thảo luận + giới hạn (một buổi bay, 39 khung có người, CI rộng, dev-val không dự đoán test, ngân sách ms).
19. Kết luận + hướng tiếp.
20. Tài liệu tham khảo.
