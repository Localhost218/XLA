# Phase 14: Đánh giá nhánh A trên khung thật, E4 khôi phục-trước-nhận-diện, thử mù A/B

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 09, 11, 12, 13 | M (~1,5 ngày agent + ~10 phút chủ dự án), ~1 giờ GPU (chưa xác minh) | Không |

Lane: serial
Depends-on: 09, 11, 12, 13
Gate: human
Effort: M
Creates: `xla/eval/enhance_eval.py`, `xla/eval/ab_page.py`, `xla/eval/ab_score.py`, `results/enhance/enhance_eval.json`, `results/e4/e4.json`, `results/ab/ab-test.html`, `results/ab/ab_result.json`, `results/ab/ab_key.json`, `results/ab/README.md`, `tests/eval/test_enhance_eval.py`
AX: `python -m xla.eval.ab_page build --pairs 55 --seed 20261118 --json` dựng trang thử mù; `python -m xla.eval.ab_score <export.json> --json` in tỉ lệ thắng + CI Wilson, exit 1 khi thiếu lượt
Accept: `pytest tests/eval/test_enhance_eval.py`

## Mục tiêu

Chấm nhánh A **trên khung thật** (không có ảnh chuẩn): chỉ số vật lý, nhấp nháy, ms, NR-IQA chỉ để mô tả, thử mù A/B do chủ
dự án chấm; và trả lời câu hỏi nghiên cứu 2 bằng E4 (khôi phục trước nhận diện có giúp không). Gộp các lane thành PR #2.

## Đầu vào cần có

- Bundle `models/release/detector-v1.0.0` (phase 11), `models/release/enhancer-v1.0.0` (phase 13), `xla/enhance/classic.py`
  (phase 12), `xla/eval/detection.py` + nhãn (phase 08–09), `xla/camera/measure.py` (phase 04).
- Tiêu chí §8 nhánh A trong thiết kế; `results/baseline/tieu-chi-sau-m2.md`.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §3 (TaskGuard: cổng chọn ảnh khôi phục/gốc), §5
  (NR-IQA chỉ mô tả; `pyiqa` giấy phép PolyForm Noncommercial).
- `plan.md` D5, D12; Deferred 5, 8, 16; Human steps hàng 2.

## Việc theo thứ tự

1. **Gộp lane:** merge `lane/data`, `lane/degrade`, `lane/realism`, `lane/labels`, `lane/detect`, `lane/classic`,
   `lane/restore` vào `feat/ai-anh-chat-luong-thap`. File `.onnx` không theo merge (gitignore, Deferred 21): chép
   `detector.onnx`, `enhancer.onnx` từ worktree `lane/detect`, `lane/restore` vào đúng thư mục bundle của nhánh tích hợp, rồi
   `python -m xla.contract validate` từng bundle → `[]` (sha256 khớp `meta.onnx_sha256` đã commit). `pytest -m "not gpu"`
   toàn bộ xanh trước khi làm tiếp.
2. **Chỉ số vật lý** (`enhance_eval.py`) trên mọi khung test, ba biến thể gốc / cổ điển / mạng: % điểm hồng, dark channel
   trung vị, Y p1, độ bão hoà, blocking, Laplacian; tỉ lệ thay đổi so với gốc + CI bootstrap theo khối.
3. **Nhấp nháy:** trên chuỗi khung test liên tiếp cách nhau < 250 ms: trung bình |Δ| giữa khung liền kề sau xử lý chia cho
   của gốc (≤ 1 = không tăng).
4. **ms/khung:** cổ điển (CPU) và mạng (ORT CUDA EP) đọc từ bench/meta, đo lại một lần trên máy này.
5. **NR-IQA** (mô tả): MUSIQ + NIQE qua `pyiqa`, cùng crop, ghi giấy phép; **không** dùng để chọn gì.
6. **E4** (`results/e4/e4.json`): mAP50 + CI trên test cho (mốc COCO, detector bundle) × (khung gốc, cổ điển, mạng); biến
   thể có cổng (kiểu TaskGuard: chọn ảnh khôi phục khi điểm tự tin cao hơn) là tuỳ chọn. Đổi D5 sang `enhanced` **chỉ khi**
   CI của chênh lệch ghép cặp nằm trọn trên 0; khi đó báo Owner để tăng `contract_version` 1.1 (không tự sửa hợp đồng).
7. **Trang thử mù** (`ab_page.py`, Deferred 16): `results/ab/ab-test.html` là file HTML local đầy đủ `<!doctype html>`,
   `<head>`, `<body>`, mở thẳng bằng trình duyệt, không server, không mạng ngoài; 25 cặp mạng vs gốc + 25 cặp mạng vs cổ điển
   + 5 cặp lặp lại, trái/phải ngẫu nhiên theo seed, nhãn biến thể ẩn (chỉ có trong file khoá `results/ab/ab_key.json`);
   khung chọn từ test có đủ độ cao và có/không người; phím tắt ←/→; tiến độ + lựa chọn lưu `localStorage`; nút **Xuất kết quả**
   tải về JSON. Ảnh đặt cạnh file HTML (thư mục `results/ab/img/`), tên file là mã ngẫu nhiên không lộ biến thể; ảnh được
   commit cùng trang để chấm lại được (Deferred 14).
8. **Bước người (Gate: human):** agent mở sẵn trang trong trình duyệt của chủ dự án; chủ dự án chấm (~10 phút), bấm Xuất;
   agent chờ trong lượt, chép file vừa tải về vào `results/ab/ab_result.json`.
9. **Chấm A/B** (`ab_score.py`): tỉ lệ thắng mạng vs gốc, mạng vs cổ điển, kèm CI Wilson 95 %; độ nhất quán 5 cặp lặp.
10. `results/ab/README.md` + `enhance_eval.json`: bảng đối chiếu từng tiêu chí §8 nhánh A (đạt / không đạt / số đo), kết
    luận câu hỏi 2. Chọn enhancer ship (hoặc không ship theo Forecast "Kill (c)").
11. **Test** `tests/eval/test_enhance_eval.py`: hàm nhấp nháy = 1 khi ảnh ra = ảnh vào; `ab_score` trên JSON mẫu cho đúng tỉ
    lệ + CI Wilson tính tay; `ab_result.json` có ≥ 55 lượt và khớp khoá; HTML có `<!doctype html>`, không có URL `http`;
    tên file trong `results/ab/img/` không chứa tên biến thể; `e4.json` có đủ 6 ô.
12. Commit, đẩy, mở **PR #2** vào `main`, CI xanh rồi merge.

## Cổng pass

- Accept exit 0; `ab_result.json` có đủ lượt; PR #2 merge.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 25 lượt mỗi so sánh → CI rộng | high | medium | Báo CI Wilson; không nói "thắng" khi CI chạm 50 % |
| Chủ dự án nhận ra biến thể (vd. dải mép) | medium | medium | Cả hai ảnh trong cặp đều đã xử lý cột trái giống nhau |
| E4 cho kết quả trái dự đoán | medium | low | Đây là câu hỏi nghiên cứu; báo đúng như số |

## Rollback

Revert merge commit của PR #2; nhánh lane còn nguyên.
