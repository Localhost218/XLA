# Verifier tầng 2 — `verifier-v1.0.0` (AI v2 phase 07, lane verifier)

Ngày 25/11/2026. Nhánh `lane/verifier` (từ `feat/ai-v2` @ 6666b9f). Mã `xla/verify/{common,mine,train,export_onnx}.py`,
cấu hình `configs/verifier.yaml`, test `tests/verify/test_verifier.py`. Hợp đồng `docs/hop-dong-model.md` §9.
Detector khai thác và chấm = **`detector-v1.0.0`** (phase 06 Kill (b)), đường ONNX, `score_floor` 0,05, ngưỡng vận hành
0,07669 của meta v1.

```bash
$PY -m xla.verify.mine detect --json        # GPU (gpu_lock): detector v1 trên mine + devpool + cal + 2 400 ảnh d02x
$PY -m xla.verify.mine sheet --json         # trang soát 200 âm thật -> D:\DevCache\runs\xla\v2\verifier\review\
$PY -m xla.verify.mine build --json         # manifest.jsonl + mine.json (đọc neg_review.csv)
$PY -m xla.verify.train prep --json         # CPU: vùng cắt rung + JPEG (D2), 4 worker
$PY -m xla.verify.train run --json          # GPU: 4 ứng viên x (2 fold + bản cuối), checkpoint mỗi epoch
$PY -m xla.verify.train select --json       # OOF, chọn, ngưỡng D7, Kill (c) -> select.json (mã 1 nếu Kill (c))
$PY -m xla.verify.export_onnx --json        # graph §9, parity, ms, bundle -> export.json
$PY -m xla.contract validate models/release/verifier-v1.0.0   # []
```

**Trần GPU: 30 phút cả phase** (quyết định chủ dự án 25/11, plan ghi 1 giờ). Đã dùng **629,5 s = 10,5 phút
(0,175 giờ)**, nên chạy đủ 2 backbone × 2 cỡ cắt, không phải cắt ứng viên (mục 7).

## 1. Kết luận

- Chọn **`efficientnet_b0`, crop 96** (`effb0_96`): giảm báo nhầm OOF nhiều nhất ở recall OOF giảm ≤ 0,05.
- **Kill (c) không kích hoạt**: ở ngưỡng 0,78 (recall OOF giảm 0,046), báo nhầm trên `cal` giảm **93 %** (15 → 1 hộp).
- Ngưỡng sơ bộ ghi vào meta (D7): **`conf_threshold` 0,375** — cal 0,100 → **0,013 hộp/khung** (≤ 0,03), recall OOF
  **không giảm** (0,266 → 0,266), báo nhầm OOF 92 → 58 (−37 %). Phase 09 chốt lại cùng k/n của track.
- **Không đạt đích ms**: ORT CUDA N = 5 p50 **5,96 ms** (đích < 3 ms). `mnv3s_64` đạt 2,93 ms nhưng lọc kém hơn (mục 5).
- Cỡ mẫu nhỏ: cal chỉ có **15** hộp báo nhầm của v1 trên 150 khung, OOF 92 báo nhầm và 218 người. Một hộp cal = 0,0067
  hộp/khung; một người = 0,0046 recall. Đọc các tỉ lệ ở mục 4 kèm số đếm.

## 2. Mẫu (`manifest.jsonl`, `mine.json`)

18 766 dòng `{id, src, image, box, label, type, fold, score, wh, x_lo}`; `id` = `<nguồn>-sha1(ảnh|hộp)[:10]` (ổn định
khi luật lọc đổi). **Không khung cal / holdout / khối giữ lại của cal / test v1 nào** (`check_manifest`, test kiểm);
detector chạy trên cal chỉ để chấm: hộp cal nằm ở `D:\DevCache\runs\xla\v2\verifier\det_cal.json`, ngoài repo (test
phase 03 coi mọi file trong `results/` có khung cal là manifest), không vào manifest.

| Loại | Nguồn | Số | Cách lấy |
|---|---|---|---|
| âm khó (hộp detector) | mine | 32 | mọi hộp v1 ≥ `score_floor` trên 85 khung mine (không người) |
| âm khó (hộp detector) | devpool | 59 | IoU lớn nhất với GT < 0,1 **và** phần hộp nằm trong một GT < 0,3; bỏ 9 khung `hard` |
| âm nền ngẫu nhiên | mine / devpool | 680 / 1 208 | 8 vùng/khung, cạnh 12–60 px, không chạm GT nới ×2 |
| âm báo nhầm ngoài | d02x | 9 444 | hộp v1 trên 2 400 ảnh train d02x (lấy đều HERIDAL / VisDrone / WiSARD), cùng luật IoU/IoA |
| dương GT | devpool | 218 | mọi hộp nhãn |
| dương hộp detector | devpool | 80 | IoU ≥ 0,5 với GT |
| dương hộp detector | d02x | 3 320 | IoU ≥ 0,5 với GT |
| dương GT | d02x | 3 000 | lấy đều theo nguồn, cạnh nhỏ ≥ 6 px |
| dương GT | TinyPerson | 725 | `select_external(cfg, for_d02x=False)`, split train, cạnh nhỏ ≥ 8 px (lọc góc còn ít ảnh) |

Âm khó thật **chỉ 91 hộp** (Risk "âm mine quá ít" xảy ra): v1 chỉ báo nhầm ~0,4 hộp/khung mine ở `score_floor`. Bù
bằng âm báo nhầm của d02x và âm nền; loss cân theo loại (`train.weights`: âm khó thật ×4, dương thật ×3).

Kiểu âm khó trên trang soát: túi / tấm nhựa hồng (nhiều nhất), bụi cây và lá, đá, **bóng người**, vệt sáng trên đường đất.

**Soát (bước 1).** Trang 10×10 của 200 âm thật (93 âm khó lúc đó + 107 âm nền, seed 20261126), agent xem từng ô và phóng
to 8 ô nghi. Bỏ 2 hộp nằm **trên người có nhãn** (đầu/tay và thân người áo đen; IoU < 0,1 chỉ vì hộp nhỏ hơn người) —
`neg_review.csv`. Hai ca này cùng một kiểu nên thêm luật `neg_max_ioa` 0,3; luật đó bỏ chúng trước bước soát, nên `mine.json` ghi
`dropped_by_review` 0 (91 âm khó còn lại).
Bóng người giữ làm âm (đúng là không phải người). Không thấy người sót trong âm nền.

## 3. Huấn luyện

`mobilenet_v3_small` và `efficientnet_b0` (torchvision, ImageNet), lớp cuối 1 logit, BCE có trọng số, AdamW lr 1e-3
OneCycle, batch 256, AMP, tối đa 12 epoch, dừng sớm patience 3 theo loss của **10 % ảnh ngoài giữ lại** (không theo fold
OOF). Vùng cắt chuẩn bị trước trên CPU bằng đúng phép cắt §9 (`context_scale` 1,5, `min_side_px` 32, kẹp
`[16, 640] × [0, 480]`, `roi_align(aligned=False, sampling_ratio=2)`): 3 bản mỗi mẫu (bản 0 = hộp gốc, 2 bản rung tâm
±12 %, cỡ ±20 %), mỗi bản nén lại JPEG q 25–85 với xác suất 0,5 (D2). Trên GPU: lật ngang, sáng ±20, tương phản ±20 %.

Fold: dương/âm devpool theo `folds.json` (fold 0 = phiên #5, fold 1 = #1–#4), âm mine theo phiên của nó (fold 0: 39 khung,
fold 1: 46 khung); mẫu ngoài luôn vào train. OOF của khung fold k = model huấn luyện không có fold k. Cal chấm bằng bản
cuối (mọi mẫu). Epoch tốt nhất / số epoch đã chạy (`train_runs.json`):

| | fold 0 | fold 1 | cuối | giây GPU (3 lần) |
|---|---|---|---|---|
| mnv3s_64 | 5 / 9 | 6 / 10 | 4 / 8 | 70 |
| mnv3s_96 | 2 / 6 | 4 / 8 | 2 / 6 | 60 |
| effb0_64 | 2 / 6 | 3 / 7 | 5 / 9 | 118 |
| effb0_96 | 1 / 5 | 1 / 5 | 3 / 7 | 183 |

Model học nhanh rồi quá khớp (loss giữ lại thấp nhất ở epoch 1–6), nên thêm epoch không có ích.

## 4. OOF, cal, chọn, ngưỡng (`select.json`)

Chấm mọi hộp v1 ≥ 0,07669: OOF = 245 khung (160 devpool + 85 mine), 218 người; cal = 150 khung không người. Báo nhầm =
hộp có IoU < 0,5 với mọi GT; recall = người có ít nhất một hộp giữ lại IoU ≥ 0,5. **Trước verifier**: OOF 92 báo nhầm
(0,376/khung), recall 0,266 (58/218); cal **15** báo nhầm (0,100/khung).

| | ngưỡng (recall giảm ≤ 0,05) | báo nhầm OOF | giảm OOF | recall OOF | báo nhầm cal | giảm cal | ngưỡng D7 | D7: báo nhầm OOF / recall / cal |
|---|---|---|---|---|---|---|---|---|
| mnv3s_64 | 0,805 | 33 | 64 % | 0,220 | 0 | 100 % | 0,085 | 79 / 0,266 / 4 |
| mnv3s_96 | 0,235 | 38 | 59 % | 0,220 | 3 | 80 % | 0,090 | 53 / 0,248 / 3 |
| effb0_64 | 0,445 | 40 | 57 % | 0,220 | 1 | 93 % | 0,070 | 74 / 0,266 / 3 |
| **effb0_96** | **0,780** | **17** | **82 %** | 0,220 | 1 | 93 % | **0,375** | **58 / 0,266 / 2** |

- **Chọn** (bước 2): giảm báo nhầm OOF lớn nhất ở recall OOF giảm ≤ 0,05 → `effb0_96` (82 % so với 64 % của á quân).
- **Kill (c)** (bước 4): cùng điểm đó, cal giảm 93 % ≥ 50 % → **không kích hoạt**, xuất bundle.
- **Ngưỡng D7** (bước 3): trong các ngưỡng có cal ≤ 0,03 hộp/khung, lấy recall OOF lớn nhất; hoà thì ít báo nhầm OOF hơn,
  rồi ngưỡng cao hơn → **0,375**: cal 2 hộp (0,013/khung), recall OOF giữ 0,266, báo nhầm OOF 58 (0,237/khung).
  `meta.metrics` ghi ở điểm này: `fp_per_cal_frame` 0,100 → 0,01333, `recall_oof` 0,26606 → 0,26606.
- Recall theo cỡ người (cạnh lớn, OOF, effb0_96): < 16 px 0/1 (detector đã sót); 16–32 px 20/76 trước, 20/76 ở D7, 16/76
  ở 0,78; ≥ 32 px 38/141, 38/141, 32/141. Verifier ở 0,78 làm rơi người nhỏ và lớn gần như cùng tỉ lệ.
- Recall < 0,6 là do **detector v1** (0,266 trước verifier), không phải verifier; D7 ghi "không đạt" phần recall.

## 5. Graph, parity, ms (`export.json`, `ms_candidates.json`)

Graph một model, opset 17, N động: nới hộp → kẹp → `RoiAlign` (`output_half_pixel`, `sampling_ratio` 2, kiểm thuộc tính
lúc xuất) → BGR→RGB, /255, ImageNet → backbone → sigmoid → `scores [N]`. N = 0 trả `[0]` (test + golden `002.jpg`).

- **Parity** golden (3 khung, 33 hộp gồm hộp v1, GT, hộp thử [100,100,140,180], 20 hộp ngẫu nhiên tới mép) so với PyTorch
  eager fp32 (`roi_align(aligned=False)`): ORT CPU |Δ| lớn nhất **1e-6**, ORT CUDA **3,4e-4** (≤ 0,02). Test còn so phép cắt
  tham chiếu với `xla.contract.roi_align_probe_model` (≤ 1e-3).
- `xla.contract validate models/release/verifier-v1.0.0` → `[]`, mã 0.
- **ms** ORT CUDA EP 1.24.4, RTX 4060 Laptop, warmup 30, 300 lần, khung golden, hộp ngẫu nhiên 4–80 px. CPU 168–173 %
  hiệu năng (không hạ xung), GPU 73 °C, driver 610.88; khối `machine` đầy đủ trong `meta.json`.

| | N = 1 p50 / p95 | N = 5 | N = 20 |
|---|---|---|---|
| **effb0_96 (ship)** | 4,15 / 6,98 | **5,96 / 8,52** | 13,33 / 15,19 |
| mnv3s_64 | 2,72 / 5,43 | 2,93 / 5,70 | 3,85 / 6,50 |
| mnv3s_96 | 3,24 / 5,88 | 3,79 / 6,19 | 6,66 / 8,91 |
| effb0_64 | 3,72 / 6,80 | 4,48 / 7,59 | 7,72 / 10,37 |

Đích < 3 ms ở N = 5 **không đạt** với model đã chọn; chỉ `mnv3s_64` đạt (2,93 ms). Luật chọn của phase là giảm báo nhầm
OOF, nên giữ `effb0_96` và báo ở đây. Hệ quả: dòng Forecast "ms trực tiếp không tăng quá +5 ms" có thể trượt ở khung có
≥ 5 hộp (p95 8,5 ms). Đổi sang `mnv3s_64` chỉ cần sửa `train.candidates` của bước chọn (trọng số đã có ở
`D:\DevCache\runs\xla\v2\verifier\runs\mnv3s_64_final\best.pt`): ở D7 nó giữ recall 0,266, cal 4 hộp (0,027), báo nhầm
OOF 79 (−14 % thay vì −37 %). Việc cho chủ dự án (mục 8).

## 6. Bundle `models/release/verifier-v1.0.0/`

`meta.json` (§9: `crop_px` 96, `context_scale` 1,5, `min_side_px` 32, `crop_left` 16, `conf_threshold` 0,375,
`input_variant` raw, `backbone` efficientnet_b0, `train_run` v2/verifier/effb0_96_final, `metrics`, `ms_per_frame` p50/p95
ở N = 5 + `by_n` + `machine`), `golden/` (khung devpool phiên #5 có người, devpool phiên #1–#4 có người, một khung mine
với N = 0; `boxes.json`, `expected.json` từ ORT CPU). `verifier.onnx` ngoài git (`.gitignore`), đính Release
`models-v2.0.0` ở phase 17.

## 7. Giờ GPU (trần 30 phút)

| Phần | giây |
|---|---|
| detector v1 trên 395 khung thật + 2 400 ảnh d02x (`mine detect`) | 158,6 |
| `mnv3s_64` (2 fold + cuối, kèm chấm) | 74,5 |
| `mnv3s_96`, `effb0_64`, `effb0_96` | 371,7 |
| xuất `effb0_96`: parity CUDA + ms | 9,7 |
| ms ba ứng viên còn lại | 15,0 |
| **Tổng** | **629,5 (10,5 phút)** |

Đủ 4 ứng viên vì mỗi lần chạy ≤ 80 s (luật tốc độ: chỉ cắt khi một lần > ~5 phút). Mọi lần chạy giữ `gpu_lock` và nhả ngay.

## 8. Việc mở

1. **ms vs lọc**: giữ `effb0_96` (5,96 ms ở N = 5) hay đổi `mnv3s_64` (2,93 ms, lọc kém hơn)? Quyết của chủ dự án; phase 09
   đo cả hệ cùng k/n.
2. Cal chỉ có 15 báo nhầm v1: mọi tỉ lệ cal ở đây rộng. Holdout = 0 (phase 03 Kill (a)), cần phiên bay trống mới để chấm sạch.
3. Âm báo nhầm d02x là hộp có IoU < 0,1 với nhãn của bộ ngoài; người không nhãn trong VisDrone/HERIDAL có thể lọt làm âm
   (không soát tay phần ngoài).

## 9. Tệp

`manifest.jsonl`, `mine.json` (đếm), `neg_review.csv`, `det_real.json` (hộp v1 ở `score_floor` trên mine + devpool,
để khai thác và chấm OOF; hộp cal ở `det_cal.json` của thư mục run), `select.json`, `train_runs.json` (epoch + giây GPU), `export.json`, `ms_candidates.json`. Run, vùng cắt, điểm,
trang soát: `D:\DevCache\runs\xla\v2\verifier\` (`runs/<ứng viên>_{f0,f1,final}/{best,last}.pt`, `scores/`,
`review/`, `logs/`).
