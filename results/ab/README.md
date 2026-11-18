# `results/ab/` — thử mù A/B ảnh rõ và đánh giá nhánh A (phase 14)

Trạng thái: **khung, chưa có số**. Code và chạy thử đầu-cuối bằng bundle giả do lane `abeval` làm (18/11/2026); Owner dựng
lại trang và mọi số đo bằng model thật khi gộp, rồi điền các ô `…` bên dưới.

| File | Nội dung | Sinh bằng |
|---|---|---|
| `ab-test.html` | trang thử mù, mở thẳng bằng trình duyệt (file://), không server, không mạng | `ab_page build` |
| `img/<mã>.jpg` | ảnh của các cặp; tên là mã hex ngẫu nhiên, không lộ biến thể | `ab_page build` |
| `ab_key.json` | khoá: mỗi cặp là so sánh nào, khung nào, bên trái/phải là biến thể nào | `ab_page build` |
| `ab_result.json` | file chủ dự án tải về từ nút **Xuất kết quả** (chép vào đây) | trang, bước người |
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
python -m xla.eval.ab_score results/ab/ab_result.json --json           # tỉ lệ thắng + CI Wilson, exit 1 khi thiếu lượt/lệch khoá
python -m xla.eval.enhance_eval all --json                             # chạy lại để bảng tiêu chí có cả hai dòng A/B
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

## 4. Kết quả (Owner điền sau khi chạy bằng model thật)

Model: enhancer `…` (sha256 `…`), cổ điển `…`, detector `…`. Ngày chấm: `…`.

### 4.1 Thử mù A/B (`ab_score`)

| So sánh | Lượt | Mạng thắng | Tỉ lệ | CI Wilson 95 % | Tiêu chí §8 | Đạt? |
|---|---|---|---|---|---|---|
| mạng vs gốc | 25 | … | … | … | ≥ 70 % | … |
| mạng vs cổ điển | 25 | … | … | … | ≥ 55 % | … |
| độ nhất quán (5 cặp lặp) | 5 | — | … | — | — | — |

Không nói "thắng" khi CI chạm 50 % (`verdict_vs_50` = `chua_phan_dinh`).

### 4.2 Bảng tiêu chí §8 nhánh A (`enhance_eval.json` → `criteria_branch_a`)

| Tiêu chí | Số đo (mạng) | CI 95 % | Đạt? |
|---|---|---|---|
| % điểm hồng giảm ≥ 90 % | … | … | … |
| dark channel trung vị giảm ≥ 50 % | … | … | … |
| nhấp nháy không tăng (≤ 1) | … | … | … |
| thắng ≥ 70 % A/B so với gốc | … | … | … |
| thắng ≥ 55 % A/B so với cổ điển | … | … | … |
| ≤ 30 ms/khung (ORT CUDA EP, p50) | … | — | … |
| không làm giảm mAP nhánh B | … | … | … |

### 4.3 E4 — khôi phục trước nhận diện (câu hỏi nghiên cứu 2)

| mAP50 (CI 95 %) | khung gốc | cổ điển | mạng |
|---|---|---|---|
| mốc YOLO11s COCO | … | … | … |
| detector bundle | … | … | … |

Chênh ghép cặp so với khung gốc (CI 95 %): … Khuyến nghị D5: … (đổi sang `enhanced` chỉ khi CI của chênh nằm trọn trên 0;
khi đó Owner tăng `contract_version` 1.1, lane không tự sửa hợp đồng).

### 4.4 Mô tả thêm

NR-IQA (MUSIQ, NIQE qua `pyiqa`, giấy phép PolyForm Noncommercial 1.0.0 — chỉ mô tả, không dùng để chọn): …

Kết luận câu hỏi 2 và quyết định ship enhancer (hoặc không ship theo Forecast "Kill (c)"): …

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
