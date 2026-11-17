# Phase 09: Bộ đánh giá nhận diện + mốc YOLO11s có sẵn, sửa tiêu chí §8

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 08 | M (~1 ngày agent), ~0,3 giờ GPU (chưa xác minh) | Không |

Lane: labels
Depends-on: 08
Effort: M
Creates: `xla/eval/detection.py`, `xla/detect/baseline.py`, `results/baseline/baseline.json`, `results/baseline/tieu-chi-sau-m2.md`, `tests/eval/test_detection_eval.py`
AX: `python -m xla.eval.detection --pred <csv> --split test --json` in mAP50, mAP50-95, AP-small, P/R ở ngưỡng, báo nhầm/khung âm, CI bootstrap; exit 1 khi CSV có khung ngoài tập
Accept: `pytest tests/eval/test_detection_eval.py`

## Mục tiêu

Một bộ chấm duy nhất cho mọi model của dự án (mốc, fine-tune, ONNX, có/không khôi phục) trên định dạng CSV của hợp đồng §7,
rồi đo mốc YOLO11s COCO có sẵn trên tập test thật. Với số mốc, viết lại tiêu chí nhánh B của §8 nếu cần (thiết kế cho phép
sửa sau M2).

## Đầu vào cần có

- `data/real/labels/` (phase 08), `data/real/split.json`; hợp đồng §7.
- `plans/reports/261118-bang-chung/video/phan-tich-video.md` (mốc cũ: precision 41 % trên 300 khung, chưa có recall/mAP).
- Prior art: IOT-CV `ml/scripts/evaluate.py` (khung sườn; giao diện A1.6 in AR-small + độ trễ,
  `plans/ai/ai-phase-01-chuan-bi.md:562`); `pycocotools`.
- `plan.md` D11; Forecast "Kill (a)".

## Việc theo thứ tự

1. **`xla/eval/detection.py`:** đọc nhãn YOLO + CSV dự đoán → COCO (px khung gốc 640×480) → `pycocotools` cho mAP50,
   mAP50-95, AP-small (area < 32²) và AR-small; P/R/F1 ở ngưỡng vận hành truyền vào; ngưỡng tối ưu F1 (chỉ được chọn trên
   **dev-val**, áp lên test); báo nhầm trung bình mỗi khung **âm** (khung test không người + `split = "neg"`); ms/khung đọc từ
   cột tuỳ chọn; **bootstrap theo khối thời gian** (lấy lại khối 30 s, 2 000 lần, seed) cho CI 95 % mọi chỉ số; tách số liệu
   theo bin độ cao và cờ `hard`.
2. **`xla/detect/baseline.py`:** YOLO11s COCO, imgsz 1280, conf 0,001, lớp người, chạy trên test + neg + dev-val → CSV; ba
   biến thể: khung nguyên; bỏ hộp trong 16 px mép; + SAHI. Ghi ms/khung (RTX 4060, trong `xla.gpu_lock`).
3. Ghi `results/baseline/baseline.json` (mọi chỉ số + CI, ba biến thể, ngưỡng chọn trên dev-val).
4. **Sửa tiêu chí:** `results/baseline/tieu-chi-sau-m2.md` ghi lại tiêu chí nhánh B với số mốc thật: mAP50 mục tiêu =
   max(1,5 × mốc, mốc + 0,10); nếu mốc ≥ 0,7 thì theo Forecast "Kill (a)" đề xuất mục tiêu mới và báo Owner trước khi phase 10
   chạy. Không sửa file brainstorm (bản ghi đã duyệt); plan dẫn tới file này.
5. **Test** `tests/eval/test_detection_eval.py` (CPU, dữ liệu đồ chơi): dự đoán trùng nhãn → mAP50 = 1; không dự đoán →
   0; một nửa hộp đúng → giá trị tính tay; báo nhầm/khung âm đúng; bootstrap cùng seed ra cùng CI; CSV có khung lạ → exit 1;
   ngưỡng tối ưu chỉ lấy từ dev-val (truyền tập test vào chọn ngưỡng → lỗi).
6. Commit trên `lane/labels`, đẩy, báo lead (kèm số mốc để cập nhật Forecast).

## Cổng pass

- Accept exit 0; `baseline.json` có mAP50 + CI cho ba biến thể.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CI quá rộng vì ít người | high | medium | Báo CI, không nói quá; tách theo độ cao |
| Chọn ngưỡng trên test (rò rỉ) | low | high | Hàm chọn ngưỡng từ chối tập test; test bước 5 canh |

## Rollback

Revert commit; số mốc tính lại được từ CSV.
