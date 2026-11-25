# Thầy v2 + nhãn giả khung dev (AI v2 phase 05, lane detect)

Ngày 25/11/2026. Nhánh `lane/detect` (từ `feat/ai-v2` @ 69f8e6d). Mã: `xla/detect/selftrain.py` (khối AI v2 cuối file),
danh sách cho phép khung thật `xla/detect/train.py` (`check_real_frames`), chấm devpool `xla/eval/detection.py`
(split `devpool`). Cấu hình `configs/teacher_v2.yaml`. Test `tests/detect/test_teacher_pseudo.py`.

```bash
$PY -m xla.detect.selftrain teacher --config configs/teacher_v2.yaml --smoke --json   # -> smoke.json
$PY -m xla.detect.selftrain teacher --config configs/teacher_v2.yaml --json           # GPU, gpu_lock, train tiếp từ last.pt
$PY -m xla.detect.selftrain pseudo --config configs/teacher_v2.yaml --json            # nhãn giả + chấm devpool -> teacher.json
```

**Trần GPU cắt theo quyết định chủ dự án (25/11/2026, vì tốc độ): 1 giờ cho cả phase 05** thay cho 6 giờ của plan.
Đã dùng **0,834 giờ** (bảng giờ dưới). Vì trần này thầy fine-tune trên tập con cố định 3 600 ảnh d02x (+ 85 khung nền
`mine`) mỗi epoch, không học hết 13 008 ảnh.

## 1. Smoke và cỡ thầy (Deferred 8)

`smoke.json`: 60 bước batch 2 @1280 AMP trên 120 ảnh d02x, val 60 ảnh, mỗi cỡ một lần (không phải một epoch 10 %).
Máy: RTX 4060 Laptop 8 GB, driver 610.88, lúc đo 64 °C (khối `machine`).

| Cỡ | Khởi từ | VRAM đỉnh | giây/ảnh train | giờ/epoch (3 685 ảnh + val 800) ước | Vừa 8 GB |
|---|---|---|---|---|---|
| YOLO11l | COCO `yolo11l.pt` | 5,47 GB | 0,136 | 0,174 | có |
| YOLO11m | thầy E2 `last.pt` (YOLO11m @1280 học D2 4 epoch) | 4,28 GB | 0,106 | 0,136 | có |

Luật smoke (l vừa batch ≥ 2 và học được ≥ 6 000 ảnh trong `time_h`) chọn **l**. Chạy thật l: epoch 1 mAP50 val
**0,0052** sau 755 s (đầu một lớp học từ đầu trên ảnh khó; `optimizer: auto` của ultralytics chọn SGD lr 0,01 vì
20 epoch × 1 843 bước > 10 000, khác công thức AdamW lr 0,002 của E1). Trong ~0,6 giờ còn lại l không thể hội tụ, nên
**dừng l ở epoch 1** (run giữ ở `D:\DevCache\runs\xla\v2\detect\teacher_l_aborted\`, `aborted.json`) và đổi sang
**m khởi từ thầy E2** với AdamW lr 0,002 cố định (config ghi lý do). Thầy E2 không thấy khung thật nào (học D2) và
`last.pt` là epoch cuối, không phải checkpoint chọn bằng dev-val, nên không mang thông tin devpool.

## 2. Huấn luyện thầy

Tập `D:\DevCache\datasets\xla\v2_teacher\`: 3 600 ảnh d02x train lấy đều theo nguồn (seed 20261125; VisDrone, HERIDAL,
WiSARD theo tỉ lệ) + **85 khung `mine`** nhãn rỗng (2,3 % ≤ 10 %), val = 800 ảnh d02x val cùng cách lấy. Danh sách cho
phép kiểm trước khi ghi: `{mine: 85, pseudo: 0}`. Công thức E1: imgsz 1280, batch 2, AMP, HSV tắt, mosaic 1,0,
close_mosaic 1, cos LR, warmup 0,5, seed 20261118, deterministic, `workers` 4. Dừng theo `time` 0,55 giờ của ultralytics;
dừng sớm theo mAP50 val (patience 5) không kích hoạt. Checkpoint = mAP50 val d02x cao nhất (`best_val50.pt`), không khung
thật có nhãn nào dự phần.

| Epoch | mAP50 val | mAP50-95 val | train s | VRAM đỉnh |
|---|---|---|---|---|
| 1 | 0,134 | 0,051 | 505 | 4,41 GB |
| 2 | 0,140 | 0,055 | 502 | 4,59 GB |
| **3** (chọn) | **0,203** | 0,083 | 503 | 4,59 GB |
| 4 (dừng giữa epoch vì `time`) | 0,193 | 0,083 | 351 | 4,59 GB |

Trọng số: `D:\DevCache\runs\xla\v2\detect\teacher\weights\best.pt` (sha256 `7a6a0a81…`).

## 3. Thầy trên devpool (OOF: thầy không thấy khung devpool nào)

160 khung devpool (218 hộp) + 85 khung `mine` làm khung âm; CI 95 % bootstrap 2 000 lần theo khối 30 s; ngưỡng vận hành
= F1 tối đa trên chính devpool. v1 = `D:\XLA\models\release\detector-v1.0.0\detector.onnx` (ORT CPU, chỉ đọc).

| | mAP50 [CI] | mAP50-95 | AP-small | AR-small | P / R @ ngưỡng | báo nhầm/khung trống devpool | báo nhầm/khung mine |
|---|---|---|---|---|---|---|---|
| thầy v2 (m) | 0,197 [0,156; 0,251] | 0,081 | 0,104 | **0,379** | 0,31 / 0,28 @ 0,042 | 0,28 | 0,34 |
| detector-v1.0.0 | 0,229 [0,144; 0,312] | 0,104 | 0,074 | 0,172 | 0,37 / 0,32 @ 0,058 | 0,28 | 0,28 |

mAP50 theo fold (fold 0 = phiên #5, fold 1 = phiên #1–#4): thầy 0,251 / 0,171; v1 0,292 / 0,173.
Hiệu ghép cặp thầy − v1: mAP50 **−0,031 [−0,092; +0,052]**, AR-small +0,208 [+0,151; +0,285], recall@op −0,037
[−0,120; +0,057]. Thầy tìm được nhiều người nhỏ hơn (AR-small gấp đôi) nhưng xếp hạng điểm kém hơn nên mAP50 không hơn.

Lưu ý công bằng: v1 đã học nhãn giả khung dev ở E2 (trừ dev-val v1 + đệm 0,3 s), nên trên 74 khung devpool mới v1
**không** phải OOF — số của v1 ở đây có lợi thế. Khung `mine` nằm trong tập train của thầy (ảnh nền) nên báo nhầm trên
mine của thầy là trong mẫu.

**Hệ quả cho phase 06 (quy tắc chủ dự án 25/11):** thầy không rõ ràng hơn v1 trên devpool (cận dưới CI hiệu mAP50
−0,092 ≤ 0) → **bỏ C2** (không chạy fold), giờ GPU dồn cho C1/C3.

## 4. Nhãn giả (`D:\DevCache\datasets\xla\pseudo_v2\`)

Khung đích = mọi khung `dev` v1 trừ khung `cal` (khối 13, 21), `holdout` (0) và `mine`: **880 khung**, 42 chuỗi liên
tiếp (khoảng cách < 1 s). Thầy + TTA, conf 0,01, ByteTrack của ultralytics như E2 (chỉ để làm sạch nhãn giả, D4), luật
E2 (track ≥ 3 khung, τ_thấp 0,05, bỏ phát hiện lẻ, bỏ hộp chạm mép trái, bỏ khung có hộp ≥ τ_cao bị loại).
τ_cao của fold j chọn trên nhãn devpool của chính fold j (model fold i ≠ j mới huấn luyện trên nhãn giả fold j, nên
không rò vào OOF của nó):

| Fold | Phiên | Khung | Dùng được | Hộp | τ_cao | Chất lượng trên devpool fold đó (P / R) |
|---|---|---|---|---|---|---|
| 0 | #5 | 441 | 432 | 66 | 0,60 (nhỏ nhất đạt P ≥ 0,9) | 1,00 / 0,07 |
| 1 | #1–#4 | 439 | 426 | 151 | 0,35 (không τ nào đạt 0,9, lấy P cao nhất) | 0,60 / 0,12 |

Tổng 858 khung dùng được, 119 khung có hộp, 217 hộp; bỏ 20 khung (chuỗi ngắn hơn 3) + 2 khung (hộp ≥ τ_cao bị loại).
Nhãn giả recall rất thấp (7–12 % người thật) — khớp với việc thầy không hơn v1. `pseudo_manifest.json` (bản trong repo)
ghi từng khung: `frame, session, fold, use, why, boxes`; test kiểm không khung nào là test/cal/holdout/mine và phiên
của mỗi khung nằm đúng fold.

## 5. Giờ GPU (trần 1 giờ, chủ dự án 25/11)

| Phần | Giờ |
|---|---|
| Smoke l + m | 0,020 |
| Thầy l (dừng sau epoch 1) | 0,210 |
| Thầy m (4 epoch, `time` 0,55 h + val + khởi động) | 0,569 |
| Nhãn giả TTA 880 khung + dự đoán devpool/mine | 0,036 |
| **Tổng** | **0,834** |

## 6. Tệp

`smoke.json` (VRAM, giây/ảnh, `machine`, lựa chọn và lần đổi l → m), `teacher.json` (lịch sử val, nhãn giả, devpool
thầy/v1 + hiệu ghép cặp, giờ GPU), `pseudo_manifest.json`, `preds/teacher_devpool.csv`, `preds/v1_devpool.csv`
(CSV D11 trên devpool + mine, conf 0,001).
