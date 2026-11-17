# Phase 10: Thí nghiệm E1 — fine-tune YOLO trên D0/D1/D2

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 07, 09 | L (~2 ngày agent), 12–30 giờ GPU (chưa xác minh — smoke đo trước) | Không |

Lane: detect
Depends-on: 07, 09
Effort: L
Creates: `xla/detect/train.py`, `configs/train_e1.yaml`, `results/e1/smoke.json`, `results/e1/e1.json`, `results/e1/README.md`, `tests/detect/test_finetune_runs.py`
AX: `python -m xla.detect.train --config configs/train_e1.yaml --set d2 --smoke --json` chạy thử 1 epoch trên 10 % dữ liệu và in giờ/epoch, VRAM đỉnh, giờ ước cho lần đủ
Accept: `pytest tests/detect/test_finetune_runs.py`

## Mục tiêu

Trả lời câu hỏi nghiên cứu 1 (phần nhận diện): fine-tune cùng một detector trên D0 (sạch), D1 (ngẫu nhiên) và D2 (bản sao
camera), cùng ảnh nguồn, cùng siêu tham số, rồi chấm trên **tập test thật**. Model tốt nhất làm điểm xuất phát cho
self-training (phase 11).

## Đầu vào cần có

- D0/D1/D2 ở `D:\DevCache\datasets\xla\` + `dataset.yaml` (phase 07).
- Bộ chấm `xla/eval/detection.py`, mốc `results/baseline/baseline.json`, tiêu chí `results/baseline/tieu-chi-sau-m2.md`
  (phase 09); nhãn dev-val + test (phase 08).
- `d:\Coding\IOT-CV\plans\reports\260921-research-ai-vision-pipeline.md` §1.3 (cấu hình 8 GB VRAM).
- `plan.md` D9, D10; Deferred 10, 19.

## Việc theo thứ tự

1. **`xla/detect/train.py`:** bọc `ultralytics` train; đọc config; luôn chạy trong `xla.gpu_lock`; ghi run vào
   `D:\DevCache\runs\xla\e1\<set>\` (không vào repo); chọn checkpoint bằng mAP50 trên **dev-val thật** (callback chấm bằng
   `xla.eval.detection`, không dùng val tổng hợp để chọn); seed cố định; ghi lệnh, commit, sha256 manifest dữ liệu vào run.
2. **`configs/train_e1.yaml`:** YOLO11s COCO-pretrained, imgsz 1280, một lớp, cùng epoch / batch / augment cho ba tập;
   tắt augment màu HSV mạnh (màu là thứ đang học); mosaic giữ mặc định.
3. **Smoke (bắt buộc trước khi chạy đủ):** 1 epoch trên 10 % D2 → giờ/epoch, VRAM đỉnh, ước giờ đủ. VRAM tràn → hạ batch,
   rồi imgsz 960 (ghi Deferred mới vào `results/e1/README.md`, báo lead). Ước > 12 giờ GPU / lần đủ → báo lead trước (Forecast
   "Kill (b)"). Ghi `results/e1/smoke.json`.
4. **Chạy đủ** ba lần D0, D1, D2 (nối tiếp, chung khoá GPU). Mỗi lần: xuất CSV dự đoán trên test + neg (conf 0,001, bỏ hộp
   trong 16 px mép trái, đúng như mốc), chấm bằng `xla.eval.detection`, ngưỡng vận hành chọn trên dev-val.
5. `results/e1/e1.json` + `README.md`: bảng mốc / D0 / D1 / D2 × (mAP50, mAP50-95, AP-small, P/R, báo nhầm/khung âm,
   ms) kèm CI; chênh D2−D1 kèm CI bootstrap ghép cặp; giờ GPU thật từng lần. Viết kết luận đúng như số.
6. **Test** `tests/detect/test_finetune_runs.py`: `e1.json` có đủ ba set + mốc, mỗi set có `smoke_hours_per_epoch`,
   `gpu_hours`, `selected_by: "devval"`; manifest train của ba set không chứa khung thật nào; ngưỡng vận hành ghi nguồn
   `devval`; CI có `ci_low ≤ mean ≤ ci_high`.
7. Commit (config, kết quả, không trọng số) trên `lane/detect`, đẩy, báo lead kèm giờ GPU đo được.

## Cổng pass

- Accept exit 0; `e1.json` đủ ba lần chạy.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Giờ GPU vượt ước | medium | medium | Smoke trước; hạ imgsz/epoch có ghi lại |
| Chọn checkpoint trên dev-val ít người → nhiễu | medium | medium | Báo CI; dùng mAP50 trung bình 3 epoch cuối làm phụ |
| Laptop ngủ giữa lần huấn luyện | medium | low | Ultralytics `resume=True`; cắm sạc khi chạy dài |

## Rollback

Xoá thư mục run; kết quả JSON giữ lịch sử trong git.
