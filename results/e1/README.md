# E1 — fine-tune YOLO11s trên D0 / D1 / D2 / D2+RealSR (phase 10)

Ngày 18/11/2026. Lane detect, nhánh `lane/detect`. Số trong file này lấy từ `e1.json` (máy đọc) và `eval/*.json`
(đầu ra đủ của `xla.eval.detection`). Dựng lại:

```powershell
. D:\XLA\scripts\env.ps1
$py = 'D:\DevCache\venvs\xla\Scripts\python.exe'
& $py -m xla.detect.train --config configs/train_e1.yaml --set d2 --smoke --json   # smoke.json
& $py -m xla.detect.train --config configs/train_e1.yaml --set d0 --json           # lặp cho d1, d2, d2sr (GPU)
& $py -m xla.detect.train --config configs/train_e1.yaml --time --json             # ms/khung (GPU)
& $py -m xla.detect.train --config configs/train_e1.yaml --score --json            # e1.json (CPU, ~40 phút)
```

Cách làm: YOLO11s COCO, đầu 80 lớp đổi thành 1 lớp, cùng cấu hình `configs/train_e1.yaml` cho cả bốn tập, chỉ khác
`data`. Tập sinh ở `D:\DevCache\datasets\xla\<set>` (3 006 ảnh train, 487 val, toàn VisDrone, không khung thật nào; sha256
manifest ghi trong `e1.json`). Mỗi epoch, callback dự đoán 100 khung **dev-val thật** bằng `last.pt`, chấm mAP50 bằng
`xla.eval.detection`, giữ checkpoint cao nhất (`best_devval.pt`). Val tổng hợp của ultralytics chỉ ghi lại. Ngưỡng vận
hành = ngưỡng F1 tối đa trên dev-val. Tập test không dùng để chọn gì (D10). CSV dự đoán (D11: conf 0,001, bỏ hộp
`x1 < 16`, đúng như mốc `edge16`) ở `preds/<set>_{test,neg,devval}.csv`. Trọng số ở
`D:\DevCache\runs\xla\e1\<set>\weights\best_devval.pt` (ngoài repo).

## 1. Kết quả trên tập test thật

198 khung test (39 có người, 149 hộp), báo nhầm đo trên 159 khung bay không người (`neg_flight`) và 300 khung `neg`
trong nhà. CI 95 % bootstrap 2 000 lần theo khối 30 s, seed 20261118.

| Model | Epoch chọn (dev-val mAP50) | Ngưỡng | mAP50 [CI] | mAP50-95 | AP-small | P / R / F1 @ ngưỡng |
|---|---|---|---|---|---|---|
| mốc `edge16` (COCO, không fine-tune) | — | 0,068 | 0,126 [0,030–0,224] | 0,063 | 0,054 | 0,193 / 0,329 / 0,243 |
| D0 (sạch) | 5 (0,295) | 0,053 | **0,297** [0,036–0,408] | 0,100 | 0,122 | 0,227 / 0,651 / 0,337 |
| D1 (ngẫu nhiên) | 5 (0,176) | 0,023 | 0,143 [0,083–0,185] | 0,047 | 0,063 | 0,131 / 0,356 / 0,192 |
| **D2 (bản sao camera)** | 6 (**0,329**) | 0,089 | 0,236 [0,069–0,370] | 0,079 | 0,090 | **0,312** / 0,456 / **0,371** |
| D2+RealSR | 3 (0,299) | 0,059 | 0,149 [0,011–0,282] | 0,048 | 0,060 | 0,147 / 0,510 / 0,228 |

Báo nhầm (hộp/khung âm) ở ngưỡng dev-val của từng model:

| Model | `neg_flight` (159) | `neg_indoor` (300) | gộp (459) |
|---|---|---|---|
| mốc `edge16` | 0,597 [0,397–0,884] | 0,117 [0,077–0,160] | 0,283 |
| D0 | 1,195 [0,767–1,751] | 0,053 [0,023–0,092] | 0,449 |
| D1 | 0,994 [0,503–1,596] | 0,327 [0,262–0,391] | 0,558 |
| **D2** | **0,302** [0,218–0,376] | **0,020** [0,003–0,040] | **0,118** |
| D2+RealSR | 1,956 [0,680–3,614] | 0,383 [0,320–0,451] | 0,928 |

### Hiệu ghép cặp (B − A, cùng mẫu khối cho hai model)

| So sánh | mAP50 [CI] | P(B tốt hơn) | recall @ ngưỡng | báo nhầm `neg_flight` [CI] | báo nhầm `neg_indoor` [CI] |
|---|---|---|---|---|---|
| D2 − D1 | +0,093 [−0,073 … +0,252] | 0,84 | +0,101 [−0,054 … +0,317] | **−0,692 [−1,238 … −0,246]** | **−0,307 [−0,373 … −0,243]** |
| D2+RealSR − D1 | +0,006 [−0,124 … +0,118] | 0,55 | +0,154 [−0,107 … +0,283] | +0,962 [−0,289 … +2,738] | +0,057 [−0,036 … +0,158] |
| D2 (tốt nhất) − `edge16` | +0,109 [−0,017 … +0,236] | 0,95 | +0,128 [−0,141 … +0,346] | **−0,296 [−0,584 … −0,090]** | **−0,097 [−0,144 … −0,051]** |
| D0 − `edge16` | **+0,170 [+0,016 … +0,224]** | 0,98 | **+0,322 [+0,091 … +0,405]** | +0,597 [+0,285 … +0,946] | −0,063 [−0,119 … −0,007] |
| D1 − `edge16` | +0,017 [−0,079 … +0,111] | 0,61 | +0,027 [−0,116 … +0,140] | +0,396 [0,000 … +0,861] | +0,210 [+0,124 … +0,293] |
| D2+RealSR − `edge16` | +0,023 [−0,137 … +0,141] | 0,64 | +0,181 [−0,224 … +0,328] | +1,358 [+0,146 … +3,064] | +0,267 [+0,190 … +0,345] |

Đậm = CI không vắt qua 0. Báo nhầm thấp hơn là tốt hơn.

## 2. Tiêu chí nhánh B (`results/baseline/tieu-chi-sau-m2.md`)

| | D0 | D1 | **D2 (best)** | D2+RealSR |
|---|---|---|---|---|
| **B1** mAP50 ≥ 0,227 và cận dưới CI hiệu với `edge16` > 0 | **đạt** (0,297; +0,016) | không (0,143) | **không**: 0,236 ≥ 0,227 nhưng cận dưới −0,017 | không (0,149) |
| **B2** `neg_flight` ≤ 0,05, cận trên CI ≤ 0,1 | không (1,195) | không (0,994) | **không** (0,302 [0,218–0,376]; recall 0,456) | không (1,956) |
| **B3** ≤ 30 ms p95 (ở đây PyTorch) | không (p95 66,8) | không (122,8) | **không** (p95 106,8; trung vị 26,3) | không (66,3) |

ms đo bằng `--time` (tiến trình riêng, mỗi model 498 khung test + neg sau 5 khung khởi động, PyTorch FP32, conf 0,001,
gồm tiền/hậu xử lý, không gồm đọc JPEG). Điều kiện đo: 18/11 16:49–16:50, GPU 81–86 °C (máy đang quá nhiệt, chủ dự án chọn
để nguyên), CPU toàn máy 9–20 %, lane khác chạy CPU song song. Trung vị 23,6 / 27,9 / 26,3 / 34,2 ms (D0 / D1 / D2 /
D2+RealSR); mốc `edge16` đo lúc máy rảnh 20,9 ms trung vị, 23,3 ms p95. Đây là số chính thức của lần chạy này. Số đo ngay
sau train trong cùng tiến trình còn cao hơn (p95 90–146 ms, `ms_per_frame_after_train`). B3 theo định nghĩa đo trên ORT CUDA
EP ở phase 11/16; số PyTorch ở đây chưa đạt.

## 3. Kết luận (đúng như số)

1. **D2 hơn D1, nhưng chỉ chắc chắn ở báo nhầm.** mAP50 D2 − D1 = +0,093, CI vắt qua 0 (P = 0,84): tập test này chưa đủ
   để nói D2 xếp hạng tốt hơn D1. Báo nhầm thì khác hẳn, CI không chạm 0: D2 giảm 0,69 hộp/khung bay không người và 0,31
   hộp/khung trong nhà so với D1. D1 (làm xấu ngẫu nhiên) còn báo nhầm nhiều hơn cả mốc COCO.
2. **D2+RealSR không hơn D1** (+0,006 mAP50, P = 0,55) và báo nhầm nhiều nhất trong bốn tập. KID tốt hơn ở phase 07 không
   chuyển thành nhận diện tốt hơn ở lần chạy này.
3. **Best theo dev-val là D2** (0,329 so với D0 0,295, D2+RealSR 0,299, D1 0,176). Đây là set phase 11 dùng
   (`e1.json` khoá `best`). D2 có precision, F1 và báo nhầm tốt nhất trên test, nhưng **chưa đạt B1**: điểm 0,236 vượt
   0,227 song cận dưới CI hiệu với mốc là −0,017.
4. **D0 là model duy nhất đạt B1 trên test** (0,297, CI hiệu [+0,016; +0,224]). Không chọn được D0 vì luật D10 chọn trên
   dev-val, và trên dev-val D0 thua D2. D0 tìm được nhiều người hơn (recall 0,65) nhưng báo nhầm ở khung bay gấp đôi mốc.
   Ghi nhận như một kết quả, không đổi lựa chọn sau khi đã thấy test.
5. **B2 không model nào đạt.** Tốt nhất là D2: 0,302 hộp/khung bay không người, giảm một nửa so với mốc nhưng còn gấp 6
   lần mục tiêu 0,05.
6. **Chọn checkpoint trên dev-val rất nhiễu.** mAP50 dev-val nhảy giữa các epoch cạnh nhau (D2: 0,075 / 0,122 / 0,327 /
   0,137 / 0,221 / 0,329; D0: 0,148 / 0,070 / 0,151 / 0,075 / 0,295 / 0,212). Trung bình 3 epoch cuối (chỉ số phụ):
   D0 0,194, D1 0,134, D2 0,229, D2+RealSR 0,217, cùng thứ tự với D2 đứng đầu. Với 6 epoch và 100 khung dev-val (36 khung có
   người), khoảng cách D2/D2+RealSR/D0 trên dev-val nhỏ hơn mức nhiễu giữa hai epoch.

## 4. Giờ GPU thật

| Lần | Giờ GPU (giữ khoá: train + dự đoán CSV) | Giờ/epoch đo | Ghi chú |
|---|---|---|---|
| smoke batch 8 | 0,049 | — | tràn VRAM, bỏ |
| smoke batch 4 | 0,028 | ước 0,119 | `smoke.json` |
| D0 | 0,676 (40,6 phút) | 0,108 | |
| D1 | 0,660 (39,6 phút) | 0,106 | |
| D2 | 0,759 (45,6 phút) | 0,120 | máy quá nhiệt (GPU 81–87 °C) |
| D2+RealSR | 0,729 (43,7 phút) | 0,116 | như trên |
| đo ms (`--time`, 4 model) | ~0,03 | | |
| **Tổng tính trong `e1.json`** | **2,85** | | smoke b4 + bốn lần đủ |

Thêm ~0,5 giờ GPU mất vào các lần chạy hỏng (ước theo giờ log, không có trong `e1.json`): lần đầu với 4 worker d0/d1 chết
vì hết commit memory (~23 phút), d2 bị dừng tay để chạy lại cả bốn cùng cấu hình (~1 phút), d0 mất 6 phút lúc laptop hết pin
tắt máy 13:24. Log ở `D:\DevCache\runs\xla\e1\aborted-w4\` và `logs\poweroff-1324\`. Smoke ước 0,119 giờ/epoch, đo thật
0,106–0,120, nên ước smoke dùng được.

## 5. Deferred / quyết định đã tự chốt (Owner vắng, theo khuyến nghị đầu tiên của plan)

| # | Quyết định | Số đo / lý do |
|---|---|---|
| E1-1 | **imgsz 1280, batch 4** | Smoke batch 8 @1280: VRAM đỉnh 8,52 GB > 8 GB, tràn sang RAM chung, 2,50 s/batch (`smoke_batch8.json`). Batch 4: 4,45 GB, 0,34 s/batch. Giữ 1280 vì batch 4 vừa VRAM; ultralytics tích luỹ gradient tới nbs 64 nên batch danh nghĩa cho optimizer không đổi. Không hạ 960. |
| E1-2 | **6 epoch, giống nhau cho cả bốn tập** | Smoke: 0,119 giờ/epoch (ước toàn epoch) hoặc 0,083 (ước theo trung vị batch); trần 40 phút/lần cho 5 hoặc 7 epoch, chọn 6. Đo thật 40–46 phút/lần, hai lần sau vượt trần vài phút vì máy quá nhiệt. Tổng ~2,85 giờ, trong ngân sách ~3 giờ. Ước 12–30 giờ GPU của plan (Deferred 19) thay bằng số này. |
| E1-3 | warmup 1 epoch, close_mosaic 2, mosaic 1,0, HSV h/s/v = 0, patience 1000, seed 20261118, deterministic | Mặc định warmup 3 / close_mosaic 10 dài hơn cả lần chạy. Tắt HSV vì màu/ám là thứ D1/D2 dạy. Optimizer `auto` chọn AdamW lr 0,002 (cùng cho bốn tập). |
| E1-4 | **workers 2** (không phải 4) | Lần đầu với 4 worker, d0 rồi d1 chết vì `_ArrayMemoryError` trong worker dataloader: Windows hết commit memory (RAM 40 GB + pagefile 2 GB) khi các lane restore/abeval/detect-export cùng chạy. Xoá run dở, chạy lại cả bốn với 2 worker để cùng cấu hình; không lần nào chết nữa. |
| E1-5 | Chọn checkpoint bằng mAP50 dev-val mỗi epoch, ngưỡng F1 dev-val | Đúng D10 và lệnh Owner. Trung bình 3 epoch cuối ghi kèm làm chỉ số phụ (rủi ro nhiễu của phase file). |
| E1-6 | Best = set có dev-val mAP50 cao nhất (D2), kể cả khi D0 đạt B1 trên test | Không chọn trên test (D10). |
| E1-7 | ms/B3 đo ở tiến trình riêng (`--time`) lúc GPU 81–86 °C | Lệnh chủ dự án: không chờ máy nguội, ghi số kèm điều kiện. Số đo ngay sau train trong cùng tiến trình cao hơn nhiều và không dùng cho B3. |
| E1-8 | YOLO26s không thử (Deferred 10) | Ngân sách GPU đêm nay chỉ đủ cho bốn tập với một kiến trúc. |

## 6. File

- `e1.json`: mọi số trên. Có `best` (hợp đồng §5.5 cho phase 11), `sets.<set>` (lịch sử dev-val từng epoch, giờ GPU,
  dữ liệu train, ms, tiêu chí), `compare` (sáu so sánh ghép cặp), `smoke`.
- `eval/<model>_test.json`: đầu ra đủ của `xla.eval.detection` (theo bin độ cao, cờ `hard`).
- `preds/<set>_{test,neg,devval}.csv`: dự đoán D11 (mỗi file < 1 MB).
- `smoke.json`, `smoke_batch8.json`: hai lần smoke.
- Test: `tests/detect/test_finetune_runs.py`.
