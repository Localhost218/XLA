# `results/ab/` — thử mù A/B ảnh rõ và đánh giá nhánh A (phase 14)

Trạng thái: **có số thật, A/B lần 1 đã chấm** (19/11/2026). Code và chạy thử bằng bundle giả do lane `abeval` làm
(18/11/2026); phase 14 chạy lại mọi số bằng model thật, chủ dự án chấm 55/55 lượt. Kết quả và kết luận ở §4.

A/B lần 2 (khung thật, phóng to 2× quanh người, 19/11/2026) ở `results/ab-zoom/README.md`: mạng vs gốc 13/25 = 0,52
[0,335; 0,700], vs cổ điển 5/25 = 0,20 [0,089; 0,391] — vẫn không đạt §8, quyết định §4.6 giữ nguyên.

| File | Nội dung | Sinh bằng |
|---|---|---|
| `ab-test.html` | trang thử mù, mở thẳng bằng trình duyệt (file://), không server, không mạng | `ab_page build` |
| `img/<mã>.jpg` | ảnh của các cặp; tên là mã hex ngẫu nhiên, không lộ biến thể | `ab_page build` |
| `ab_key.json` | khoá: mỗi cặp là so sánh nào, khung nào, bên trái/phải là biến thể nào | `ab_page build` |
| `ab_result.json` | file chủ dự án tải về từ nút **Xuất kết quả** (chép vào đây) | trang, bước người |
| `ab_score.json` | tỉ lệ thắng + CI Wilson, độ nhất quán, lệch bên | `ab_score --out` |
| `ab_analysis.json` | A/B theo tầng (độ cao, có/không người), thời gian chọn, màu từng biến thể trên ảnh của trang (kèm mã) | phân tích thêm, §4.4 |
| `../enhance/ms_remeasure.json` | ms/khung đo riêng từng model (detector, enhancer; ORT CUDA EP), kèm mã | đo lại bước 4 |
| `../enhance/classic_bench.json` | bench hàm cổ điển (CPU) đo lại lúc máy khoẻ | `xla.enhance.classic --bench` |
| `../e4/e4.json` | E4: 2 detector × 3 biến thể, mAP50 + CI, chênh ghép cặp so với khung gốc | `enhance_eval e4` |
| `../enhance/enhance_eval.json` | chỉ số vật lý, nhấp nháy, ms, NR-IQA, E4 tóm tắt, A/B, bảng tiêu chí §8 | `enhance_eval all` |

## 1. Lệnh (chạy từ gốc repo, môi trường như `thuc-thi-phase-10-14.md` §2)

Bundle thật mặc định: `models/release/enhancer-v1.0.0`, `models/release/detector-v1.0.0`; hàm cổ điển mặc định
`xla.enhance.classic:enhance` (tham số `configs/enhance_classic.yaml` qua `load_params`). GPU chạy trong `xla.gpu_lock`.

```bash
python -m xla.contract validate models/release/enhancer-v1.0.0        # [] trước khi làm gì
python -m xla.contract validate models/release/detector-v1.0.0
python -m xla.eval.enhance_eval all --json                             # → results/enhance/enhance_eval.json + results/e4/e4.json
python -m xla.eval.ab_page build --pairs 55 --seed 20261118 --json     # → ab-test.html + img/ + ab_key.json
# chủ dự án chấm, bấm Xuất kết quả; chép file tải về thành results/ab/ab_result.json
python -m xla.eval.ab_score results/ab/ab_result.json --out results/ab/ab_score.json --json  # CI Wilson; exit 1 khi thiếu lượt/lệch khoá
python -m xla.eval.enhance_eval all --score-only --json                # chạy lại (E4 chấm từ CSV có sẵn) để bảng tiêu chí có hai dòng A/B
python -m xla.enhance.classic --split dev --limit 200 --bench --threads 4 --bench-out results/enhance/classic_bench.json
python -m pytest tests/eval/test_enhance_eval.py                       # Accept (gồm test marker data trên file thật)
```

Cờ đổi model / hàm (giống nhau ở `enhance_eval` và `ab_page`): `--enhancer <bundle>`, `--classic <module:func | file.py:func>`,
`--classic-params <yaml>`, `--providers cuda,cpu`. Riêng `enhance_eval e4`/`all`: `--detector <bundle>`,
`--coco-weights <pt>`, `--coco-device 0|cpu`. Dựng lại trang khi đã có: thêm `--force` (ảnh cũ trong `img/` bị xoá; file
chấm cũ lệch khoá mới thì `ab_score` báo exit 1).

## 2. Cách dựng trang (Deferred 16)

- 55 lượt = 25 cặp mạng vs gốc + 25 cặp mạng vs cổ điển + 5 cặp lặp lại. Trái/phải ngẫu nhiên theo seed 20261118.
- Khung: tập test của `data/real/split.json`, mỗi khung dùng một lần, rải đều theo tầng (bin độ cao 3–8 / 8–14 / > 14 m ×
  có / không người theo nhãn test đã duyệt); hai so sánh cân tầng (lệch ≤ 1 cặp mỗi tầng). Số cặp mỗi tầng ghi ở
  `ab_key.json` → `strata`.
- Cặp lặp: cùng khung, cùng hai biến thể, **đổi bên**, cách lần đầu ≥ 10 lượt. Độ nhất quán = tỉ lệ cặp lặp chọn cùng
  biến thể như lần đầu.
- Mọi ảnh (gốc, cổ điển, mạng) đều cắt bỏ cột `[0, 16)` (dải tối mép trái) và lưu JPEG q95 4:4:4 cùng một cách, để không
  nhận ra biến thể qua dải mép hay kiểu nén.
- Lựa chọn và tiến độ lưu trong `localStorage` của trình duyệt (khoá `xla-ab-<key_id>`), đóng tab mở lại vẫn chấm tiếp.

## 3. Hướng dẫn chấm (cho chủ dự án, ~10 phút)

1. Mở `results/ab/ab-test.html` bằng Chrome/Edge (kéo file vào trình duyệt). Cửa sổ cỡ bình thường, không cần toàn màn hình.
2. Mỗi lượt có hai ảnh của **cùng một khung**. Chọn ảnh **rõ hơn, dễ nhìn hơn** (thấy người, vật, mặt đất rõ hơn):
   phím **←** nếu ảnh trái rõ hơn, **→** nếu ảnh phải rõ hơn, hoặc bấm vào ảnh.
3. Không có lựa chọn hoà; phân vân thì chọn ảnh dễ nhìn hơn. Đừng nghĩ lâu, ấn tượng đầu là đủ.
4. Bấm nhầm: **Backspace** (hoặc nút Quay lại) rồi chọn lại.
5. Xong 55/55: bấm **Xuất kết quả**, trình duyệt tải về `ab_result-<mã>.json`. Đưa file đó cho agent (agent chép thành
   `results/ab/ab_result.json`).

## 4. Kết quả (phase 14, model thật, 19/11/2026)

Model: enhancer `enhancer-v1.0.0` = NAFNet-w16 (`train_run` `nafnet_w16-train`, sha256 `1728de3e976112a2…`), cổ điển
`xla.enhance.classic:enhance` (`configs/enhance_classic.yaml` 1.0.0), detector `detector-v1.0.0` = E2 trò r1 (sha256
`23b1da69165ffe05…`). ORT 1.24.4 CUDA EP trên RTX 4060 Laptop; luồng CPU cv2/torch/ORT đặt trần 4 (`cpu_threads` trong
JSON). Khung: test 198 khung / 13 khối (vật lý, NR-IQA, E4), 91 chuỗi < 250 ms / 1 307 khung (nhấp nháy), neg 459 khung (E4).
Ngày chấm A/B: 19/11/2026 (`exported_at` 18/11 18:14 UTC), key `68eb0546fa303711`, 55/55 lượt.

`enhance_eval all` chạy hai lần (lần 2 `--score-only`): vật lý, nhấp nháy, NR-IQA, E4 ra trùng từng số (tất định).

### 4.1 Thử mù A/B (`ab_score.json`)

| So sánh | Lượt | Mạng thắng | Tỉ lệ | CI Wilson 95 % | Tiêu chí §8 | Đạt? |
|---|---|---|---|---|---|---|
| mạng vs gốc | 25 | 11 | 0,44 | [0,267; 0,629] | ≥ 70 % | **không** (CI chạm 50 %: chưa phân định hơn/kém gốc) |
| mạng vs cổ điển | 25 | 1 | 0,04 | [0,007; 0,195] | ≥ 55 % | **không** (CI trọn dưới 50 %: mạng thua rõ) |
| độ nhất quán (5 cặp lặp) | 5 | — | 3/5 = 0,60 | — | — | — |

Lệch bên: chọn ảnh trái 54 % (50 lượt không lặp). Thời gian chọn trung vị 3,2 s (vs gốc), 2,9 s (vs cổ điển). Kiểm tay
CI Wilson 11/25: tâm 0,448, nửa rộng 0,181 → [0,267; 0,629], khớp.

### 4.2 Bảng tiêu chí §8 nhánh A (`enhance_eval.json` → `criteria_branch_a`)

| Tiêu chí | Mạng | CI 95 % | Đạt? | Cổ điển (đối chiếu) |
|---|---|---|---|---|
| % điểm hồng giảm ≥ 90 % | −99,9 % (0,94 % → 0,0012 %) | [−99,98; −99,7] | **đạt** | −89,2 % [−95,4; −79,0], không đạt |
| dark channel trung vị giảm ≥ 50 % | −36,8 % (140,8 → 89,1) | [−45,9; −31,0] | **không** | −51,6 % (→ 68,1) [−54,9; −49,0], đạt |
| nhấp nháy không tăng (≤ 1) | 1,81 | [1,74; 1,92] | **không** | 2,18 [2,10; 2,31], không đạt |
| thắng ≥ 70 % A/B so với gốc | 0,44 | [0,267; 0,629] | **không** | — |
| thắng ≥ 55 % A/B so với cổ điển | 0,04 | [0,007; 0,195] | **không** | — |
| ≤ 30 ms/khung (ORT CUDA EP, p50) | 20,7 ms trong `all` (p95 25,3); 11,1 ms đo riêng (p95 11,3) | — | **đạt** | 8,3–9,7 ms CPU 4 luồng (bench 9,18 / p95 10,21) |
| không làm giảm mAP nhánh B | −0,085 nếu detector nhận ảnh mạng | [−0,139; −0,011] | **đạt theo cấu trúc** (D5 giữ khung gốc) | −0,112 [−0,182; −0,028] |

Mạng đạt 3/7 (hồng, tốc độ, mAP nhánh B theo cấu trúc), không đạt 4/7 (dark channel, nhấp nháy, cả hai A/B).

ms: số chính thức của `criteria` là p50 trong lệnh `all` (20,7 ms lần 2, 21,0 ms lần 1; đo sau ~4 phút GPU chạy liên tục
trong cùng tiến trình). Đo riêng từng model một session (`../enhance/ms_remeasure.json`, ba lần khớp nhau): enhancer 11,1 ms,
detector p50 20,0 / p95 20,3 ms (khớp `meta.ms_per_frame` 19,985 / 20,223). Máy lúc đo: cắm sạc, Legion Performance Mode,
`% Processor Performance` 167–179 %. Bench cổ điển đo lại thay số 95 ms lúc máy hạ xung 18/11: cv2 4 luồng nhanh hơn 32
luồng mặc định (p50 9,1 so với 10,9 ms cùng 200 khung). Cột ms trong CSV E4 (~37 ms COCO `.pt`, ~47 ms bundle) là thời gian
trong vòng xen kẽ 3 biến thể × 2 detector, không dùng làm số ms.

### 4.3 E4 — khôi phục trước nhận diện (câu hỏi nghiên cứu 2)

Test 198 khung (39 có người) + 459 khung âm; mAP50 [CI 95 % bootstrap theo khối]; ngưỡng vận hành COCO 0,25, bundle 0,0767.

| mAP50 (CI 95 %) | khung gốc | cổ điển | mạng |
|---|---|---|---|
| mốc YOLO11s COCO | 0,126 [0,030; 0,224] | 0,081 [0,020; 0,147] | 0,195 [0,034; 0,434] |
| detector bundle (r1) | 0,119 [0,032; 0,197] | 0,007 [0,000; 0,022] | 0,035 [0,012; 0,080] |

Chênh ghép cặp so với khung gốc (mAP50, CI 95 %):

| | cổ điển − gốc | mạng − gốc | báo nhầm/khung âm, mạng − gốc |
|---|---|---|---|
| mốc COCO | −0,046 [−0,088; −0,006] | +0,069 [−0,033; +0,231], P(mạng hơn) 0,86 | 0,037 → 0,026, −0,011 [−0,022; 0,000] |
| detector bundle | −0,112 [−0,182; −0,028] | −0,085 [−0,139; −0,011] | 0,307 → 0,107, −0,200 [−0,399; −0,036] |

Kiểm chéo: bundle × gốc 0,1195 = số đường ONNX của phase 11 (0,120); COCO × gốc 0,1265 = mốc `edge16` phase 09.

**Khuyến nghị D5: giữ `raw`.** CI của chênh bundle × mạng nằm trọn dưới 0, không đổi sang `enhanced`, không đổi hợp đồng.

### 4.4 Mô tả thêm

NR-IQA (MUSIQ, NIQE qua `pyiqa` 0.1.16, CUDA, cột [16, 640); giấy phép PolyForm Noncommercial 1.0.0 — chỉ mô tả, không dùng
để chọn): MUSIQ gốc 32,9 / cổ điển 33,0 (+0,4 % [−3,1; +4,7]) / mạng 38,0 (+15,5 % [+11,3; +19,6]); NIQE (thấp tốt) 6,42 /
5,53 (−13,9 %) / 5,47 (−14,9 %). MUSIQ chấm mạng cao hơn cổ điển trong khi người chấm chọn cổ điển 24/25 lượt: thêm một bằng
chứng NR-IQA không thay được A/B (Deferred 5).

Chỉ số vật lý khác (mạng / cổ điển, so với gốc): độ bão hoà +47 % / +209 %; blocking −37 % / −17 %; Laplacian +188 % / +584 %;
Y p1 116 → 44 / 53.

A/B theo tầng (`ab_analysis.json`, chỉ mô tả, n nhỏ): mạng vs gốc — có người 7/10 = 0,70 [0,40; 0,89], không người 4/15 = 0,27
[0,11; 0,52]; theo độ cao 3–8 m 3/7, 8–14 m 5/9, > 14 m 3/9. Mạng vs cổ điển — 0/10 ở 8–14 m, 0/9 ở > 14 m, 1/6 ở 3–8 m.

Màu trên chính ảnh của trang (`ab_analysis.json`, Lab OpenCV, 25 ảnh gốc / 25 cổ điển / 50 mạng):

| | L | a* | b* (> 0 ấm, < 0 nguội) | B − R | bão hoà HSV | tương phản L (độ lệch chuẩn) | dark channel trung vị |
|---|---|---|---|---|---|---|---|
| gốc | 68,2 | −3,0 | 3,0 | −2,1 | 16,5 | 9,2 | 139,2 |
| cổ điển | 51,6 | −5,9 | 6,5 | −4,5 | 49,0 | 16,0 | 64,1 |
| mạng | 54,5 | −3,1 | 1,4 | +2,6 | 23,5 | 14,6 | 93,7 |

### 4.5 Kết luận câu hỏi 2

Dự đoán trong thiết kế: "khôi phục giúp bộ nhận diện có sẵn, không giúp bộ đã fine-tune". Theo số:

- **Bộ đã fine-tune (bundle r1): khôi phục làm hại, có ý nghĩa thống kê.** Mạng −0,085 [−0,139; −0,011], cổ điển −0,112
  [−0,182; −0,028]. Vế "không giúp" đúng, và mạnh hơn dự đoán: detector học trên phân bố khung gốc (D2 mô phỏng lỗi camera)
  nên ảnh đã khôi phục là lệch phân bố. Mạng có giảm báo nhầm trên khung âm (0,307 → 0,107) nhưng mất nhiều hộp đúng hơn.
- **Bộ có sẵn (YOLO11s COCO): chưa chứng minh được là giúp.** Mạng +0,069 nhưng CI [−0,033; +0,231] chạm 0 (P 0,86, 39 khung
  có người); cổ điển lại làm hại có ý nghĩa (−0,046 [−0,088; −0,006]). Vế "giúp" chỉ có xu hướng với mạng, không với cổ điển.
- Hệ quả: D5 giữ khung gốc cho nhận diện. Bộ fine-tune chỉ ngang mốc COCO trên test (0,119 so 0,126), nên "mạng khôi phục rồi
  COCO" (0,195) là hướng đáng thử lại với nhiều khung có người hơn, chưa phải kết luận.

### 4.6 Quyết định ship enhancer (A/B lần 1, lần 2 xác nhận)

- **Forecast Kill (c) không kích hoạt**: Kill (c) chỉ xét tốc độ, và mạng ≤ 30 ms (20,7 ms trong `all`, 11,1 ms đo riêng).
  Bundle `enhancer-v1.0.0` hợp lệ hợp đồng (`validate` → `[]`), giữ trong `models/release/`.
- **Chất lượng không đạt §8 nhánh A** (4/7 không đạt, ghi đúng như số ở §4.2): không hơn ảnh gốc (0,44, CI chạm 50 %) và thua
  rõ cổ điển (1/25). **Quyết định: không ship `enhancer-v1.0.0` làm ảnh rõ khuyến nghị cho người xem.** IOT-CV vẫn tích hợp
  được theo hợp đồng nếu Owner muốn bật thử, nhưng mặc định coi như chưa có enhancer đạt (luồng `enhanced` trả 503 như hợp đồng
  quy định khi không có bundle) tới khi có bản sửa đạt A/B. **A/B lần 2** (phóng to 2×, `results/ab-zoom/`) cũng không
  đạt (0,52 vs gốc, 0,20 vs cổ điển) nên quyết định giữ nguyên. Lần 2 thấy thêm: quanh người mạng thắng gốc ổn định (gộp hai
  lần 17/21 = 0,81 [0,60; 0,92]), trên nền không người thua (7/29) — mô tả hậu kiểm, cần trang riêng khung có người để chốt.
- Cổ điển thắng mạng 24/25 và đạt dark channel, nhưng chính nó cũng không đạt nhấp nháy (2,18) và hồng (−89,2 %, sát ngưỡng),
  và là hàm Python, không phải bundle ONNX của hợp đồng: dùng cổ điển làm ảnh rõ ở IOT-CV là quyết định của Owner (ngoài hợp
  đồng hiện tại), không tự chốt ở đây.

Hướng sửa, kèm bằng chứng:

1. **Mạng ra màu nguội và nhạt.** b* mạng 1,4 so với gốc 3,0 và cổ điển 6,5; B − R của mạng +2,6 trong khi gốc −2,1 và cổ
   điển −4,5 (mạng đẩy ảnh sang xanh dương); bão hoà 23,5 so với 49,0 của cổ điển. Kiểm ảnh đích "sạch" của cặp huấn luyện D2
   có ám xanh/nhạt không; thêm loss màu (vd. L1 trên a*b* hoặc trên trung bình kênh), hoặc bước bù cân bằng trắng + bão hoà
   sau mạng, rồi chấm lại A/B.
2. **Mạng khử mù yếu hơn cổ điển.** Dark channel trung vị còn 89,1 (cổ điển 68,1; tiêu chí cần ≤ 70,4); tương phản L 14,6 so
   16,0; Laplacian +188 % so +584 %. Người chấm chọn cổ điển hết ở 8–14 m và > 14 m (0/19), nơi mù rõ nhất. Hướng: tăng mức mù
   trong làm xấu D2 cho khớp khung thật, hoặc dùng đầu ra cổ điển (đạt dark channel) làm đích/thầy cho mạng, hoặc ghép cổ điển
   khử mù trước rồi mạng khử khối/hồng sau (mạng giỏi hồng −99,9 % và blocking −37 %).
3. **Nhấp nháy tăng ở cả hai** (mạng 1,81, cổ điển 2,18). Cổ điển: làm mượt theo thời gian các đại lượng tính theo khung
   (airlight, mức đen) bằng EMA; mạng: thêm loss nhất quán thời gian hoặc chọn checkpoint theo `flicker_ratio` (đã có ở
   `xla.eval.temporal`).
4. A/B lần 1 chỉ 25 lượt mỗi so sánh, độ nhất quán 3/5: CI rộng. Bản sửa nên chấm lại bằng trang mới (seed khác) trước khi
   kết luận đạt 70 %.

## 5. Quyết định lane `abeval` tự chốt (18/11/2026, chủ dự án ngủ)

1. **Nhấp nháy trên khung không có trong `split.json`.** Khung test cách nhau ≥ 1 s nên `consecutive_runs` trên tập test
   ra 0 chuỗi. Lấy mọi khung bay của `data/real/manifest.csv` nằm trong khoảng thời gian của một khối test (khung bị bỏ khi
   chia test, không thuộc khối dev nào): 1 397 khung → 91 chuỗi < 250 ms, 1 307 khung. CI bootstrap theo khối test.
2. **Tỉ lệ nhấp nháy gộp** = trung bình có trọng số của `flicker_ratio` từng chuỗi, trọng số Σ|Δ gốc| — đúng bằng
   `flicker_ratio` trên mọi chuỗi cùng lúc (có test), nhưng không phải giữ hơn một chuỗi trong RAM.
3. **Thống kê gộp chỉ số vật lý:** trung bình cho % hồng, độ bão hoà, blocking; trung vị cho dark channel (đúng chữ tiêu chí
   §8), Y p1, Laplacian (đuôi dài). Tỉ lệ đổi = thống kê(biến thể)/thống kê(gốc) − 1, CI ghép cặp cùng mẫu khối.
4. **E4 chấm trên test + neg** (không chạy dev-val): ngưỡng vận hành cố định — detector bundle dùng `conf_threshold` của
   `meta.json`, mốc COCO dùng 0,25; mAP50 không phụ thuộc ngưỡng. Mốc COCO chạy lại bằng cùng cấu hình `xla.detect.baseline`
   (imgsz 1280, conf 0,001, lớp người, bỏ hộp `x1 < 16`) trên từng biến thể; luật bỏ hộp mép áp cho cả detector bundle.
   Biến thể có cổng kiểu TaskGuard (tuỳ chọn) chưa làm.
5. **Tiêu chí ms** xét p50 của mạng (ORT, provider ghi ở `providers_active`); p95 ghi kèm.
6. **"Không làm giảm mAP nhánh B"** đạt theo cấu trúc khi D5 giữ khung gốc (detector không nhận ảnh enhancer); số chênh
   E4 ghi kèm để đối chiếu.
7. **Ảnh thử mù** cắt bỏ dải mép (thay vì lặp cột) và hiển thị co giãn vừa cột (~810 px CSS ở 1707×900), JPEG q95 4:4:4 —
   giống nhau cho mọi biến thể.
8. **Ô E4 chấm toàn tập test** bằng `prepare`/`bootstrap` của `xla.eval.detection` (không tách bin độ cao/hard như
   `evaluate`, nhanh hơn ~6 lần; vẫn đối chiếu pycocotools); chênh ghép cặp dùng `compare` nguyên vẹn. Chấm lại không suy
   luận: `enhance_eval e4 --score-only` (đọc CSV ở `--pred-dir`, mặc định `results/e4/preds/`).

Kiểm chéo lần chạy giả (18/11): ô mốc COCO × khung gốc ra mAP50 0,1265 [0,0296; 0,2242], khớp `results/baseline/baseline.json`
biến thể `edge16` (0,1265 [0,0296; 0,2243]) — đường suy luận mốc của E4 đúng như phase 09.
