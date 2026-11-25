# Trò YOLO11s chọn bằng OOF k-fold theo phiên (AI v2 phase 06, lane detect) — **Kill (b): không xuất `detector-v2.0.0`**

Ngày 25/11/2026. Nhánh `lane/detect`. Mã `xla/detect/kfold.py`, chấm `xla/eval/detection.py` (split `devpool`, ngưỡng
F1 OOF, khung cal/holdout không bao giờ là khung âm hay khung chọn ngưỡng). Cấu hình `configs/student_v2.yaml`. Test
`tests/detect/test_student_v2.py`.

```bash
$PY -m xla.detect.kfold smoke --config configs/student_v2.yaml --json       # VRAM + giây/ảnh -> smoke.json
$PY -m xla.detect.kfold run --config configs/student_v2.yaml --json         # C1, C3 (C2 theo cổng C2) -> run + dự đoán
$PY -m xla.detect.kfold select --config configs/student_v2.yaml --json      # OOF, chọn, Kill (b) -> select.json
$PY -m xla.detect.kfold export --config configs/student_v2.yaml --json      # bundle thử + parity + ms -> export.json
$PY -m xla.detect.kfold finalize --config configs/student_v2.yaml --json    # đường ONNX: Kill (b) chính thức -> finalize.json
$PY -m xla.detect.kfold score-test --config configs/student_v2.yaml --json  # test chỉ chấm -> test.json
$PY -m xla.detect.kfold tiling --config configs/student_v2.yaml --json      # SAHI -> tiling.json
```

**Trần GPU cắt theo quyết định chủ dự án (25/11/2026, vì tốc độ): 2 giờ cho cả phase 06** thay cho 6 giờ; được nới
45 phút một lần khi đường val của ứng viên thắng còn lên (không dùng, mục 6). Đã dùng **1,314 giờ** (mục 8). Tổng
phase 05 + 06: 0,834 + 1,314 = **2,15 giờ** (trần 3 h 45).

## 1. Kết luận

- Trò thắng trong OOF: **C3** (D0 + nền mine), mAP50 OOF 0,321 [0,210; 0,446]; C1 0,306 [0,208; 0,417]; C2 không
  chạy (cổng C2).
- **Kill (b) kích hoạt**: so với v1 trên **cùng đường** dự đoán, C3 không hơn v1 —
  đường `.pt`: −0,020 [−0,059; +0,021]; đường ONNX của bundle (cái thật sự ship): **−0,041 [−0,078; +0,018]**.
  Không bundle `detector-v2.0.0` nào được giữ (bundle thử đã xuất, qua `validate`, rồi bị `finalize` xoá); IOT-CV giữ v1.
- Test (chỉ chấm, không chọn gì) lại nói ngược: C3 `.pt` 0,363 so với v1 `.pt` 0,156, hiệu +0,207 [+0,019; +0,308].
  OOF devpool **thiên vị v1** (mục 4), nên Kill (b) ở đây có thể là một âm tính giả — ghi để chủ dự án quyết, không
  lấy test để đảo quyết định.

## 2. Ứng viên và công thức (Deferred 9)

Mọi ứng viên: YOLO11s khởi từ `D:\DevCache\runs\xla\e1\d0\weights\last.pt` (E1-D0 epoch cuối: không khung thật, không
chọn bằng dev-val), imgsz 1280, batch 4, **11 epoch**, AdamW lr 0,002 cos, warmup 1, close_mosaic 2, HSV tắt, mosaic 1,0,
seed 20261118, deterministic, workers 4; mỗi epoch 3 000 ảnh nguồn lấy đều theo `source` (seed 20261125) + 85 khung
`mine` nhãn rỗng (2,8 %); checkpoint = mAP50 val tập nguồn (500 ảnh; d0 có 487), dừng sớm patience 5 (không kích hoạt).
Smoke (`smoke.json`): VRAM 4,27 GB, 0,046 s/ảnh, ước 0,051 h/epoch; đo thật ~3,4 phút/epoch.

| | Tập nguồn | Giờ GPU | mAP50 val theo epoch (1 → 11) |
|---|---|---|---|
| C1 | d02x (VisDrone D0+D2 + HERIDAL + WiSARD) | 0,602 | 0,207 → 0,277 (5) → 0,414 (9) → **0,431** |
| C3 | d0 (VisDrone sạch) | 0,596 | 0,376 → 0,508 (5) → 0,612 (9) → 0,612 → **0,618** |
| C2 | C1 + nhãn giả dev, k-fold | — | **không chạy**: thầy phase 05 không rõ hơn v1 trên devpool (thầy − v1 mAP50 −0,031 [−0,092; +0,052], `results/v2/teacher/README.md`); quy tắc chủ dự án 25/11 |

(Val của C1 và C3 là hai tập khác nhau, không so giữa chúng.)

## 3. OOF devpool (đường `.pt`, conf 0,001) — bảng chọn

160 khung devpool (218 hộp, fold 0 = phiên #5, fold 1 = #1–#4) + 85 khung mine làm khung âm. C1/C3 không thấy khung
devpool nào → dự đoán toàn devpool là OOF. CI 95 % bootstrap 2 000 lần theo khối 30 s (khối nằm gọn trong một phiên =
một fold; chỉ 2 fold nên phương sai fold báo bằng mAP50 từng fold, không CI).

| | mAP50 [CI] | mAP50 fold 0 / fold 1 | mAP50-95 | AP-small | AR-small | P / R @ ngưỡng F1 OOF | báo nhầm/khung trống devpool | /khung mine |
|---|---|---|---|---|---|---|---|---|
| **C3** | **0,321 [0,210; 0,446]** | 0,449 / 0,210 | 0,154 | 0,144 | 0,446 | 0,35 / 0,40 @ 0,027 | 0,54 | 0,41 |
| C1 | 0,306 [0,208; 0,417] | 0,382 / 0,235 | 0,131 | 0,195 | 0,349 | 0,37 / 0,36 @ 0,050 | 0,34 | 0,21 |
| v1 `.pt` (E2 r1) | 0,340 | 0,434 / 0,248 | 0,147 | 0,122 | 0,387 | 0,47 / 0,39 | 0,26 | 0,34 |
| v1 ONNX (bundle) | 0,229 [0,144; 0,312] | 0,292 / 0,173 | 0,104 | 0,074 | 0,172 | 0,37 / 0,32 @ 0,058 | 0,28 | 0,28 |

Chọn: cận dưới CI mAP50 OOF cao nhất → C3 (0,210 so với 0,208); CI hai ứng viên chồng nhau, và C3 cũng là ứng viên đơn
giản hơn (C3 < C1 < C2), nên kết quả không đổi. Ghép cặp C1 − C3: mAP50 −0,015 [−0,060; +0,018], AR-small −0,097
[−0,173; −0,041]. Fold 1 (4 phiên, người nhỏ hơn) khó hơn fold 0 với mọi model; độ lệch giữa hai fold (0,2+) lớn hơn
mọi hiệu giữa ứng viên.

Ngưỡng OOF: F1 tối đa trên OOF trong miền ≥ `score_floor` 0,05 của graph (§2): **0,051** (đường `.pt`; F1 0,351,
P 0,418, R 0,303); tự do thì 0,027 — thấp hơn mức graph cho ra, nên không dùng được.

## 4. Kill (b) — so với `detector-v1.0.0`

| So ghép cặp mAP50 OOF (C3 − v1) | Hiệu [CI 95 %] | Kết luận |
|---|---|---|
| `.pt` − `.pt` (cùng hàm `predict`, conf 0,001) | −0,020 [−0,059; +0,021] | không hơn |
| ONNX − ONNX (bundle thử v2 vs bundle v1, cùng graph, `score_floor` 0,05) | −0,041 [−0,078; +0,018] | không hơn |
| *tham khảo, khác đường:* C3 `.pt` − v1 ONNX | +0,092 [+0,024; +0,166] | không công bằng, không dùng |

Lần `select` đầu so C3 `.pt` với v1 ONNX và cho "qua" — sai vì khác đường: graph bỏ mọi hộp < 0,05, mà C3 cho người
thật điểm thấp (ngưỡng F1 tự do 0,027), nên đường ONNX cắt mất phần lớn recall của C3 (C3 `.pt` lọc ở 0,05 còn mAP50
0,210; ONNX 0,188). `select` đã sửa: v1 đối chứng bằng chính `.pt` của nó (`configs/student_v2.yaml` khối `v1`), và
`finalize` kiểm lại trên đường ONNX trước khi giữ bundle.

**Vì sao OOF thiên vị v1:** v1 (E2 r1) học nhãn giả của khung dev v1 (mọi khung dev trừ dev-val + đệm 0,3 s, gồm phần lớn
74 khung devpool mới và hàng xóm của chúng) và được chọn checkpoint + ngưỡng trên 100 khung dev-val v1 (86 trong số đó nằm
trong devpool). Không có tập con devpool nào sạch với v1. Trên test (v1 chưa từng thấy) thứ tự đảo ngược (mục 5).

## 5. Test v1 (chỉ chấm, không chọn gì)

198 khung test + 300 khung `neg` trong nhà; ngưỡng vận hành C3 = 0,056 (F1 OOF đường ONNX của `finalize`); v1 = 0,077
(meta v1); edge16 = F1 dev-val của nó.

| | mAP50 [CI] | AR-small | recall @ ngưỡng | báo nhầm/khung bay trống | Hiệu mAP50 vs edge16 | Hiệu mAP50 vs v1 ONNX |
|---|---|---|---|---|---|---|
| C3 `.pt` | **0,363 [0,063; 0,507]** | 0,447 | 0,450 | 0,25 | +0,236 [+0,036; +0,339] | +0,243 [+0,005; +0,362] |
| C3 ONNX (bundle thử, đã xoá) | 0,280 [0,035; 0,413] | 0,215 | 0,396 | 0,23 | +0,153 [−0,018; +0,274] | +0,160 [−0,007; +0,267] |
| C1 `.pt` | 0,354 | — | — | — | — | — |
| v1 ONNX (bundle) | 0,120 [0,032; 0,197] | 0,240 | 0,396 | 0,88 | — | — |
| v1 `.pt` (E2 r1) | 0,156 [0,036; 0,235] | 0,419 | — | — | — | C3 `.pt` − v1 `.pt`: +0,207 [+0,019; +0,308] |
| edge16 (mốc) | 0,126 [0,030; 0,224] | — | — | — | — | — |

Tương quan OOF ↔ test giữa hai ứng viên đúng chiều (C3 > C1 ở cả hai: 0,321/0,363 và 0,306/0,354), nhưng chỉ 2 điểm.

## 6. Nới giờ (không dùng)

Đường val của C3 ở 3 epoch cuối: 0,612 → 0,612 → 0,618 (+0,006 cả ba epoch, LR cos đã về gần 0). Bằng chứng "còn lên"
quá yếu để tiêu 45 phút, và với Kill (b) trên cả hai đường thì nới thêm khó lật được cận dưới CI −0,059. `kfold extend`
có sẵn (học tiếp từ `last.pt`, quyết chỉ theo val tập nguồn) nếu chủ dự án muốn thử.

## 7. ms, cắt ô

- **ms** (bundle thử C3, `export.json`): ORT CUDA EP 1.24.4, RTX 4060 Laptop, **p50 21,6 / p95 22,1 ms** trên 300 khung
  dev, warmup 30; CPU 178 % hiệu năng, không hạ xung; GPU lúc đo 69 °C, driver 610.88. Đạt ≤ 30 ms. Parity ONNX ↔ `.pt`
  qua. `xla.contract validate` của bundle thử: `[]` (trước khi `finalize` xoá).
- **SAHI** (`tiling.json`, sahi 0.14.0 có sẵn trong venv, PyTorch, C3, test + neg, 170 s GPU): ô 640 chồng 0,2 trên khung
  640×480 là **một ô** = không cắt: mAP50 0,363, 19,5 / 21,0 ms (p50/p95). Ô 320 chồng 0,2 (cắt thật): mAP50 **0,293**,
  AR-small 0,431, 174 / 186 ms. Cắt ô **kém hơn** và chậm 9× → không tích hợp (Deferred 15).

## 8. Giờ GPU phase 06 (trần 2 giờ)

| Phần | Giờ |
|---|---|
| Smoke trò | 0,008 |
| C1 (11 epoch + khởi động) | 0,602 |
| C3 | 0,596 |
| Dự đoán OOF + test (C1, C3, v1 `.pt`) | 0,015 |
| Xuất bundle thử (parity + ms) + chấm test ONNX + OOF/test ONNX | 0,045 |
| SAHI | 0,047 |
| **Tổng** (`kfold gpu`) | **1,314** |

## 9. Việc mở cho chủ dự án

1. **Ngưỡng điểm của graph.** `score_floor` 0,05 (§2, mặc định) cắt recall của trò v2 vì điểm người thật thấp. Hạ
   `score_floor` (khoá có trong `meta.json`, nhưng `export_onnx.py` cố định 0,05 và `conf_threshold ≥ score_floor`) hoặc
   hiệu chỉnh điểm cần quyết định hợp đồng — không thuộc lane này.
2. **Đối chứng Kill (b).** OOF devpool không sạch với v1 (mục 4); test nói C3 hơn v1. Cần một tập chấm sạch với cả hai
   (holdout phiên mới, Kill (a) phase 03) trước khi kết luận v2 không hơn v1.

## 10. Tệp

`smoke.json`, `select.json` (bảng OOF, ghép cặp, Kill (b), ngưỡng), `export.json` (bundle thử, ms, `superseded`),
`finalize.json` (OOF + Kill (b) đường ONNX), `test.json`, `tiling.json`, `parity.json`, `preds/` (CSV D11:
`<C>_oof.csv`, `<C>_test_pt.csv`, `v1_pt_devpool.csv`, `detector-v2.0.0_{oof,test}_onnx.csv`, `tiling_<ô>.csv`).
Run + trọng số: `D:\DevCache\runs\xla\v2\detect\student\{C1,C3}\weights\best.pt`. Không ONNX từng fold (C2 không chạy).
