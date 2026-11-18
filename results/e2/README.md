# E2: tự gán nhãn trên khung bay dev, xuất `detector-v1.0.0` (phase 11)

Lane `detect`, 18/11/2026. Mã: `xla/detect/selftrain.py`, cấu hình `configs/selftrain_e2.yaml`, xuất `xla/detect/export_onnx.py`.
Số máy đọc được nằm ở `results/e2/e2.json` (khoá `models`, `pseudo`, `compare`, `ship`, `bundle`), parity ở
`results/e2/parity.json`, metrics đường ONNX ở `results/e2/metrics_bundle.json`.

Cách làm, đúng plan bước 1–8:

1. **Thầy**: YOLO11m COCO @1280 fine-tune trên D2 (tập thắng E1), batch 2 (smoke batch 4 tràn VRAM 8,37 GB), 4 epoch,
   1,196 giờ GPU (71,8 phút, trong trần ~75 phút).
2. **Nhãn giả**: thầy chạy TTA trên 1 115 khung dev, ByteTrack (`persist`) theo từng chuỗi khung dev liên tiếp (cách < 1 s),
   `fuse_score: false`. τ_cao = giá trị nhỏ nhất trên lưới 0,10–0,80 đạt precision nhãn giả ≥ 0,9 trên dev-val; hộp
   ≥ τ_thấp 0,05 nằm trong track dài ≥ 3 khung có ít nhất một hộp ≥ τ_cao được cứu lại; bỏ hộp sát mép trái (x1 < 16),
   bỏ cả khung nếu còn hộp ≥ τ_cao không giữ được. Khung `neg` có nhãn rỗng. Khung dev-val và khung dev cách khung
   dev-val < 0,3 s bị loại khỏi tập train. **Không dùng khung test hay khối test nào (D10)**; `test_pseudo_datasets_have_no_test_or_devval` kiểm điều này trên tập thật.
3. **Trò**: YOLO11s @1280 khởi từ E1 D2, train trên D2 + khung dev có nhãn giả (×2) + 300 khung `neg`, mosaic, 4 epoch,
   batch 4. Mỗi vòng ≤ ~45 phút. Vòng sau chỉ chạy khi dev-val tăng theo **cả hai** thước đo (epoch tốt nhất và trung bình
   3 epoch cuối).
4. **Ship** model có dev-val cao nhất trong {E1 D2, các vòng E2}, không nhìn test.

## 1. Kết quả trên tập test thật

Ngưỡng vận hành = ngưỡng F1 chọn trên dev-val của chính model đó. CI 95 % là bootstrap khối (2 000 mẫu, seed 20261118).

| Model | dev-val mAP50 (epoch tốt nhất / TB 3 epoch cuối) | Ngưỡng | test mAP50 [CI 95 %] | mAP50-95 | AP nhỏ | P / R / F1 | báo nhầm/khung `neg_flight` [CI] / `neg_indoor` / gộp |
|---|---|---|---|---|---|---|---|
| mốc `edge16` (COCO, không fine-tune) | — | 0,0676 | 0,126 [0,030–0,224] | — | — | 0,193 / 0,329 / 0,243 | 0,597 / 0,117 / 0,283 |
| E1 D2 (`.pt`) | 0,329 / 0,229 | 0,0885 | **0,236** [0,069–0,370] | 0,079 | 0,090 | 0,312 / 0,456 / 0,371 | 0,302 [0,218–0,376] / 0,020 / 0,118 |
| Thầy YOLO11m (`.pt`, không dự chọn ship) | 0,267 / 0,172 | 0,0543 | 0,218 [0,107–0,349] | 0,060 | 0,081 | 0,267 / 0,450 / 0,335 | 0,390 / 0,683 / 0,582 |
| E2 vòng 1 `r1` (`.pt`) | **0,379** / **0,328** | 0,0749 | 0,156 [0,036–0,235] | 0,052 | 0,061 | 0,181 / 0,389 / 0,247 | 0,868 [0,386–1,411] / 0,000 / 0,301 |
| E2 vòng 2 `r2` (`.pt`) | 0,296 / 0,273 | 0,0178 | 0,313 [0,035–0,472] | 0,157 | 0,160 | 0,279 / 0,403 / 0,330 | 0,522 / 0,263 / 0,353 |
| **Bundle `detector-v1.0.0` = r1, đường ONNX** (ORT CPU, letterbox vuông §2) | 0,263 (ORT) | **0,07669** | **0,120** [0,032–0,197] | 0,042 | 0,050 | 0,188 / 0,396 / 0,255 | 0,881 [0,333–1,495] / 0,003 / 0,307 |

Dev-val theo epoch: thầy 0,025 / 0,062 / 0,188 / 0,267; r1 0,250 / 0,265 / 0,379 / 0,339; r2 0,124 / 0,263 / 0,296 / 0,262.
r2 thấp hơn r1 ở cả hai thước đo (0,296 < 0,379; 0,273 < 0,328) nên **dừng sau vòng 2, không chạy vòng 3**.

### Hiệu ghép cặp test mAP50 (B − A)

| A → B | Hiệu [CI 95 %] | P(B tốt hơn) |
|---|---|---|
| `edge16` → E1 D2 | +0,109 [−0,017…0,236] | — |
| `edge16` → thầy | +0,092 [−0,046…0,223] | — |
| `edge16` → r1 (`.pt`) | +0,029 [−0,075…0,081] | — |
| `edge16` → r2 (`.pt`) | +0,187 [−0,009…0,318] | 0,962 |
| `edge16` → bundle (ONNX) | −0,007 [−0,138…0,067] | — |
| E1 D2 → r1 | −0,080 [−0,187…0,000] | 0,026; báo nhầm `neg_flight` +0,566 [0,099…1,082] |
| E1 D2 → r2 | +0,077 [−0,037…0,136] | — |
| E1 D2 → thầy | −0,018 | — |
| r1 `.pt` → r1 ONNX | −0,036 [−0,071…0,016] | — |

## 2. Tiêu chí nhánh B (`results/baseline/tieu-chi-sau-m2.md`), chấm trên bundle (đường ONNX, §6b)

| Tiêu chí | Kết quả | Kết luận |
|---|---|---|
| **B1**: mAP50 test ≥ 0,227 và cận dưới CI hiệu so `edge16` > 0 | 0,120; hiệu −0,007 [−0,138…0,067] | **Không đạt** (không model nào của E1/E2 đạt cả hai vế) |
| **B2**: báo nhầm `neg_flight` ≤ 0,05 ở ngưỡng vận hành, cận trên CI ≤ 0,1 | 0,881 [0,333–1,495]; recall 0,396 [0,271–0,500] | **Không đạt** |
| **B3**: ≤ 30 ms/khung p95, ORT CUDA EP, RTX 4060 | p50 19,985 ms, p95 20,223 ms (300 khung, 30 khởi động) | **Đạt** |

Điều kiện đo B3: `onnxruntime-gpu 1.24.4` CUDA EP, RTX 4060 Laptop; `\Processor Information(_Total)\% Processor Performance`
min 159,8 / TB 168,1 trong 60 s (không hạ xung); GPU ~65 °C ngay sau khi đo. Đây là số chính thức (`meta.ms_per_frame`).

## 3. Kết luận (đúng như số)

- **Ship r1 theo dev-val**, đúng luật: r1 có dev-val cao nhất ở cả hai thước đo (0,379 / 0,328 so E1 D2 0,329 / 0,229,
  r2 0,296 / 0,273). Thứ tự theo TB 3 epoch cuối cũng chọn r1 (`ship.agree_with_last3_mean: true`).
- **Nhưng trên test, r1 kém E1 D2**: 0,156 so 0,236 (`.pt`), hiệu −0,080 với P(r1 tốt hơn) = 0,026, và báo nhầm trên khung
  bay không người tăng gần gấp ba. Bundle đường ONNX còn 0,120, ngang mốc `edge16`. Dev-val không dự đoán được test lần
  này. Nguyên nhân khả dĩ, chưa tách được:
  1. dev-val gồm 100 khung dev, nằm cùng chuyến bay với khung dev đã gắn nhãn giả. Vùng đệm 0,3 s chỉ loại khung sát
     nhất; các khung cùng cảnh vẫn vào train, nên dev-val bị lạc quan cho các vòng E2;
  2. recall nhãn giả rất thấp (r1 8,4 %, r2 11,2 %): gần như mọi người thật trong khung dev bị gắn là nền, trò học
     cách bỏ người nhỏ (thiên lệch xác nhận);
  3. 100 khung dev-val là ít: chênh letterbox vuông − chữ nhật đổi dấu giữa các model (mục 5), tức nhiễu ~±0,03.
- **r2 là model test tốt nhất** (0,313, mAP50-95 0,157) dù dev-val thấp hơn r1. Không ship r2 vì luật chọn theo dev-val
  (không chọn theo test); ghi lại để phase sau xét đổi cách chia dev-val.
- **Báo nhầm `neg_indoor` của E2 lạc quan**: 300 khung `neg` (trong nhà) được đưa vào train làm nền, nên r1/r2 và số gộp
  `fp_per_neg_frame` trong `meta.metrics` nhìn tốt hơn thật. B2 dùng `neg_flight` (159 khung bay, không vào train),
  không bị ảnh hưởng.

## 4. Giờ GPU thật

| Bước | Giây | Giờ | Worker DataLoader |
|---|---|---|---|
| Smoke thầy batch 4 (bỏ, tràn VRAM) | 167 | 0,046 | 2 |
| Smoke thầy batch 2 | 173 | 0,048 | 2 |
| **Thầy YOLO11m, 4 epoch** | 4 305 | 1,196 | 2 |
| Nhãn giả vòng 1 (YOLO11m 405 s + E1 D2 161 s) | 566 | 0,157 | — |
| Smoke trò r1 | 198 | 0,055 | 2 |
| **Trò r1, 4 epoch** | 2 167 | 0,602 | 2 |
| Nhãn giả vòng 2 (r1) | 111 | 0,031 | — |
| **Trò r2, 4 epoch** | 1 263 | 0,351 | **4** (lệnh Owner 23:08; không gặp lỗi bộ nhớ) |
| Tổng trong `e2.json` (`gpu_hours_total`) | | **2,486** | |
| Ngoài tổng: xuất bundle + parity + đo ms (wall 135 s), dò letterbox 4 model, 2 test `gpu` | ~ 5 phút | | |

Worker từng lần train ghi ở `e2.json` `models.<tên>.workers` (thầy [2], r1 [2], r2 [4]). 4 worker làm epoch r2 còn
~5 phút so ~9 phút của r1 với 2 worker.

## 5. Bundle `models/release/detector-v1.0.0/`

- `detector.onnx` (không commit): `D:\XLA-wt\detect\models\release\detector-v1.0.0\detector.onnx`, sha256
  `23b1da69165ffe05760bda23b4a41c858930e8a8fd8b0aebc039f1e0c5c59aef`, opset 17, onnxsim, NMS trong graph. Từ
  `D:/DevCache/runs/xla/e2/r1/weights/best_devval.pt` (sha256 `7e374765…ade5`).
- `meta.json`: `conf_threshold` 0,07669 (ngưỡng F1 dev-val của **đường ONNX**), `metrics` chấm trên đường ONNX (§6b),
  `ms_per_frame` đo trên CUDA. `validate` trả `[]`. sha256 trong meta = sha256 file = sha256 lần chấm ONNX.
- `golden/`: 3 khung dev-val (`20261102-100941_004_01748`, `_007_00497`, `_007_00820`) + `expected.json`.
- **Parity** (`results/e2/parity.json`), mốc cổng `pt_crop_square` (`.pt` với `rect=False`, letterbox vuông 1280 như
  graph §2) dev-val mAP50 0,26342: ORT CPU 0,26342 (Δ 0,0000), ORT CUDA 0,263284 (Δ −0,000136). Cả hai trong
  |Δ| ≤ 0,005 → `parity_pass: true`. Số tham khảo, không phải cổng: `pt_crop_rect` 0,29712 (Δ −0,0337),
  `pt_full_edge16` 0,31762 (Δ −0,0542).

## 6. Deferred / quyết định

**Chủ dự án chốt (18/11):**

- **Cổng parity đổi mốc sang `.pt rect=False`** (phương án A). Lý do: ORT ăn ảnh letterbox vuông 1280×1280 (hợp đồng §2),
  còn `.pt rect=True` dùng letterbox chữ nhật; so hai cách tiền xử lý khác nhau thì mọi model fine-tune đều trượt cổng
  dù graph đúng (ORT khớp `.pt rect=False` tới 0,0000). `.pt rect=True` và `full_edge16` giữ trong `parity.json` làm số
  tham khảo, trường `gate_reference` ghi rõ mốc cổng. Sửa ở `xla/detect/export_onnx.py` + `tests/detect/test_detector_export.py`.
- **Chênh letterbox vuông − chữ nhật trên dev-val theo từng model**: r1 −0,034, E1 D2 +0,021, r2 −0,016, thầy −0,005.
  Chênh **đổi dấu**, nên coi là nhiễu của 100 khung dev-val, không phải cái giá có hệ thống của hợp đồng §2.
  **Không đổi hợp đồng.** `meta.metrics` vẫn chấm bằng đường ONNX (§6b).
- Trò train với **4 worker** từ 23:08 (bỏ điều kiện bộ nhớ ảo ≥ 10 GB); chỉ lỗi bộ nhớ thật mới train tiếp từ `last.pt`
  với 2 worker. r2 không gặp lỗi bộ nhớ.

**Lane tự chốt (theo khuyến nghị đầu tiên của plan):**

- Thầy batch 2: smoke batch 4 dùng 8,37 GB VRAM, tràn RAM chung, 1,15 s/batch.
- Thầy vòng 1 chọn giữa YOLO11m và E1 D2 theo chất lượng nhãn giả trên dev-val (precision trước, rồi recall): E1 D2
  thắng (τ 0,55, P 1,0, R 0,084 so YOLO11m τ 0,45, P 1,0, R 0,075). Thầy vòng 2 = r1.
- Vòng 2: không τ nào trên lưới đạt precision ≥ 0,9; lấy τ có precision cao nhất (0,75, P 0,857, R 0,112) thay vì
  bỏ vòng. Ghi trong `pseudo_r2.json`.
- ByteTrack gọi thẳng trên phát hiện đã lưu (phát hiện GPU, track CPU), `fuse_score: false` (bật thì điểm ~0,15 không
  bao giờ khớp track), `lap` thay bằng bản thay thế trong mã + `use_lap=False` (không cài gói mới; ultralytics từng tự cài
  `lap` vào venv chung, đã gỡ).
- Tập nhãn giả tự chứa (`D:\DevCache\datasets\xla\e2_r<K>`, hard link ảnh D2) để không ghi đè `labels.cache` của D2.
- Khung có nhãn giả lặp ×2 trong `train.txt`; khung dev-val + vùng đệm 0,3 s bị loại.

**Deferred:**

- Dev-val cùng chuyến với khung dev gắn nhãn giả làm chọn model lệch (mục 3). Phase sau nên giữ dev-val theo khối thời
  gian xa hơn (vd. cả một đoạn bay), rồi mới quyết E2 có thắng E1 không.
- Recall nhãn giả ~10 % là nút thắt; τ_cao theo precision ≥ 0,9 trên 100 khung dev-val quá chặt khi người nhỏ.
- Bundle không đạt B1/B2; B3 đạt với dư 9,8 ms.

## 7. File

- `results/e2/e2.json`: mọi model, nhãn giả, smoke, so sánh, ship, bundle (đường `.pt` và đường ONNX).
- `results/e2/metrics_bundle.json`, `results/e2/parity.json`, `results/e2/pseudo_r{1,2}.json`, `results/e2/smoke_*.json`.
- `results/e2/eval/*.json` (cache chấm), `results/e2/preds/*.csv` (dự đoán test/neg/devval từng model, ONNX gộp).
- Lệnh: `python -m xla.detect.selftrain --teacher | --round K | --score | --onnx-score <bundle> [--smoke] [--json]`;
  xuất `python -m xla.detect.export_onnx --weights D:/DevCache/runs/xla/e2/r1/weights/best_devval.pt --out models/release/detector-v1.0.0
  --version 1.0.0 --conf 0.07669 --metrics results/e2/metrics_bundle.json --train-run e2/r1 --parity-out results/e2/parity.json`.
