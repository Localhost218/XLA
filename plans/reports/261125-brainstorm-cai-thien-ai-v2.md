# Brainstorm 25/11/2026 — cải thiện AI v2 (nhận diện người + ảnh rõ)

Người hỏi: chủ dự án. Nguồn trạng thái: `plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-10-14.md` §7, §10;
`results/e2/README.md`; `results/ab/README.md` §4.6.

## 1. Kiểm kê: đã xong / còn nợ

Plan 261118 (phase 01–18) **xong hết**, XLA PR #2–#4 đã merge, Release `models-v1.0.0`, IOT-CV PR #57.
Xong theo plan, nhưng chất lượng chưa đạt:

| Hạng mục | Số đo hiện tại | Tiêu chí | Kết luận |
|---|---|---|---|
| Detector `detector-v1.0.0` (trò r1) | test mAP50 0,120 [0,032–0,197]; mốc COCO 0,126; E1-D0 0,297; E1-D2 0,236 | B1 ≥ 0,227 | không đạt, còn kém E1 |
| Báo nhầm khung bay không người (`neg_flight`) | 0,881 hộp/khung | B2 ≤ 0,05 | không đạt: đây là lỗi túi rác thành người |
| Enhancer `enhancer-v1.0.0` (NAFNet-w16) | A/B vs gốc 11/25 (lần 2 13/25), vs cổ điển 1/25 | §8 nhánh A | không đạt, không làm mặc định |
| Cổ điển `xla.enhance.classic` | thắng mạng 24/25, 9 ms CPU | | đang tốt nhất, nhưng chưa làm mặc định |

Nợ nhỏ: `IOT-CV/ml/scripts/*.py` (6 file) chưa xoá; web chưa có ô trạng thái detector (chỉ có enhancer ở
`VideoPanel.tsx:80`); `test_ngan_sach_ms` đỏ có chủ đích (Owner đã chấp nhận 19/11, không động vào).
`docs/nop-bai*` chưa track: việc nộp bài riêng, không thuộc phạm vi này.

## 2. Nguyên nhân gốc

- **Detector:** dữ liệu ít (test có 74 khung có người, dev 1 115), người rất nhỏ ở độ cao lớn, JPEG nén nặng. Chọn
  model theo dev-val (100 khung) quá nhiễu: dev-val không dự đoán được test. Không có lớp phủ định cứng: 300 khung
  `neg` đều trong nhà, không chứa túi rác, bụi cây, bóng đổ ngoài bãi.
- **Enhancer:** trần 30 ms buộc dùng mạng 16 kênh, chỉ đủ khử khối JPEG nhẹ. Học từ cặp tổng hợp D2 nên màu bị
  nguội và nhạt, khử mù yếu, khung nối khung nhấp nháy.

## 3. Phương án đã chọn (chủ dự án chốt 25/11)

### Nhánh B — nhận diện: lọc nhiều tầng + dữ liệu ngoài

1. **Dữ liệu ngoài:** ảnh người nhìn từ drone có giấy phép dùng được (ứng viên TinyPerson, HERIDAL, SARD; VisDrone đã
   có). Kiểm giấy phép trước khi tải. Làm xấu bằng bản sao số camera D2 (`params_version` 3), lọc góc nhìn gần nadir
   như đã làm với VisDrone.
2. **Model tầng 1:** thầy to hơn (YOLO11m/l @1280) train trên D2 + dữ liệu ngoài, nhãn giả cho khung dev; trò
   YOLO11s giữ ngân sách ≤ 30 ms. Có tuỳ chọn cắt ô (kiểu SAHI) cho trang lịch sử bay offline.
3. **Tầng 2, xác minh từng hộp:** bộ phân loại nhỏ (MobileNetV3/EfficientNet-B0, 64–96 px) nhận vùng cắt quanh
   hộp, trả về người hay không phải người. Lớp âm lấy từ **hộp báo nhầm thật** trên khung bay không người (khai thác
   hard negative) + vật dễ nhầm trong dữ liệu ngoài.
4. **Tầng 3, theo thời gian:** ByteTrack đã có ở phase 11. Một track phải có ≥ k/n khung mới báo sự kiện
   `vision.person_detected`.
5. **Chấm điểm trung thực:** chia `neg_flight` theo **phiên bay**. Một phần đem khai thác hard negative, phần còn lại
   giữ kín để chấm B2. Chọn model bằng dev gộp lớn hơn (k-fold theo phiên), không dùng 100 khung dev-val. Ngưỡng
   hiệu chỉnh trên phần khai thác, không trên phần chấm.

### Nhánh A — ảnh rõ: hai tầng

1. **Trực tiếp (≤ 30 ms):** dùng `classic` làm "Ảnh rõ" mặc định trên luồng trực tiếp, vì đã thắng A/B. Nếu còn ngân
   sách thì thử thêm mạng nhẹ phía sau (classic → NAFNet sửa màu), chỉ giữ lại khi A/B thắng classic.
2. **Offline (lịch sử bay, không giới hạn ms):** model lớn, xử lý theo lô cho cả chuyến rồi lưu cache:
   - ứng viên 1: Real-ESRGAN ×2 (hoặc SwinIR-real), fine-tune với cách làm xấu D2;
   - ứng viên 2: mạng video nhiều khung (BasicVSR++/RVRT), dùng khung kề để hồi chi tiết và bớt nhấp nháy;
   - điều kiện vừa 8 GB VRAM (RTX 4060 Laptop), đo s/khung, chấp nhận nếu một chuyến 20 phút xử lý xong trong
     khoảng một giờ.
3. **Chấm:** A/B mù như phase 14 (cả khung + phóng to), so với gốc và với classic. Thêm hàng rào: mạng sinh không
   được bịa ra người hay chi tiết giả, nên detector chạy trên ảnh offline phải không tăng báo nhầm.

### Nợ nhỏ (làm luôn)

- Xoá `IOT-CV/ml/scripts/` sau khi grep tham chiếu.
- Ô trạng thái model trên web (detector + enhancer: loaded/rejected, phiên bản, provider, ms) dùng `KeyValue` của kit.
- Cập nhật `BAO_CAO_DO_AN.md` qua `reproduce.py --write` và slide trong XLA khi số đổi. Không đụng Drive.

### Quyền

XLA: toàn quyền như cũ. IOT-CV: chủ dự án cho toàn quyền 25/11 (nhánh, commit, PR, merge).

## 4. Rủi ro

- Giấy phép dữ liệu ngoài: một số bộ chỉ cho nghiên cứu. Đồ án là nghiên cứu, ghi rõ trong báo cáo.
- Test 74 khung cho CI rộng, nên cải thiện nhỏ sẽ không chứng minh được. Báo CI, không tuyên bố quá số đo.
- Super-resolution có thể sinh chi tiết giả: luôn để nút xem gốc, ghi rõ "ảnh do AI dựng lại".
- Máy: job nặng tối đa 4 worker; VRAM 8 GB giới hạn model video. Không hoãn việc vì nhiệt.
- Tầng 2 cộng thêm ms: phân loại ~96 px × vài hộp, ước dưới 3 ms, phải đo lại.

## 5. Chỉ số thành công

| | Hiện tại | Đích |
|---|---|---|
| Detector test mAP50 (đường ONNX, cả hệ) | 0,120 | ≥ 0,30 (đạt B1 0,227 là tối thiểu) |
| Báo nhầm/khung `neg_flight` phần giữ kín | 0,881 | ≤ 0,05 sau tầng 2+3 |
| Recall ở ngưỡng vận hành | 0,396 | ≥ 0,6 |
| Ảnh rõ trực tiếp A/B vs gốc | 11/25 (mạng) | classic hoặc lai thắng, cận dưới CI > 0,5 |
| Ảnh rõ offline A/B vs classic | — | thắng, cận dưới CI > 0,5 |
| ms trực tiếp | p95 ~54 ms nhịp thật | không tăng quá +5 ms |

## 6. Bước tiếp

`/t1k:plan` → plan `plans/261125-ai-v2/`, sau đó cook. Bundle mới đặt tên `detector-v2.0.0`, `verifier-v1.0.0`,
`enhancer-offline-v1.0.0`; hợp đồng `docs/hop-dong-model.md` cần thêm mục cho verifier và enhancer offline.
