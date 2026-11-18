# Phase 13 — mạng khôi phục riêng camera (lane restore, 18/11/2026)

Bundle hiện tại (vòng 2): `models/release/enhancer-v1.0.0/` = **nafnet_w16** bước 4 500 (`meta.json` + `golden/` trong
git; `enhancer.onnx` không commit, sha256 `1728de3e976112a208ffb43f57113ba3d31f81aa9e6e9ee43cbcc1cf2fff382e`, nằm ở
cùng thư mục trên máy). `python -m xla.contract validate models/release/enhancer-v1.0.0` → `[]`; Accept
`tests/enhance/test_enhancer_export.py` 7/7 PASS, exit 0 (p95 21,0 ms ≤ 30 ms, trần không nới). So sánh với width32:
mục 8.

Mục 1–7 là vòng 1 (width32), giữ làm bản ghi thí nghiệm; bundle width32 sao ở `D:\DevCache\tmp\enhancer-width32-v1.0.0\`
(sha256 `2ad08ebf6fd7adfbd0b08bcb9e6a5064a8611e7fcc587a243dfb34e9d0f79669`, kèm `export_width32.json`).

Số liệu gốc: `speed.json` + `speed_round2.json` (đo tốc độ), `train.json` (smoke, train, select; `runs.<run_id>` giữ
mọi vòng), `export.json` (parity, ms, điều kiện đo của bundle hiện tại).

## 1. Tốc độ ứng viên (480×624, fp16, ORT 1.24.4 CUDA EP mặc định, warmup 30, đo 300, RTX 4060 Laptop)

Đo lúc máy bị hãm (`speed.json`: `throttled: true`; CPU "% Processor Performance" 22 % lúc chờ, 135 % lúc vào khoá,
153 % sau khi đo; GPU 79–81 °C). Cả năm dòng đo liền nhau trong một lần giữ khoá, cùng điều kiện.

| Ứng viên | Trọng số công bố | Tham số | ORT p50 | ORT p95 | PyTorch fp16 p95 | Giấy phép mã | Giấy phép trọng số |
|---|---|---|---|---|---|---|---|
| **nafnet** | NAFNet-SIDD-width32 | 29,2 M | 122,6 ms | **151,7 ms** | 61,2 ms | MIT | chưa xác minh riêng (repo MIT) |
| fbcnn | fbcnn_color | 71,9 M | 157,5 ms | 162,2 ms | 158,5 ms | Apache-2.0 | chưa xác minh riêng (repo Apache-2.0) |
| scunet | scunet_color_real_psnr | 17,9 M | 381,0 ms | 445,0 ms | 281,9 ms | Apache-2.0 | chưa xác minh riêng (repo Apache-2.0) |
| nafnet_lite (biến thể) | k khối đầu của SIDD-w32 | 6,8 M | 45,7 ms | 69,3 ms | 119,9 ms | MIT | như trên |
| nafnet_tiny (biến thể) | k khối đầu của SIDD-w32 | 4,5 M | 37,3 ms | 39,8 ms | 59,7 ms | MIT | như trên |

Giấy phép đọc từ file `LICENSE` của từng repo đã clone (`D:\IOT_Tools\downloads\restore\`); không repo nào ghi giấy phép
riêng cho trọng số, nên cột trọng số ghi "chưa xác minh riêng". Lần đo đầu (13:22, laptop chạy pin rồi sập nguồn) ở
`D:\DevCache\runs\xla\restore\speed\speed_battery_1322.json`, không dùng.

**Chọn: nafnet** — nhanh nhất theo p95 trong ba bản công bố đo cùng điều kiện. Không ứng viên nào đạt trần 20 ms; theo
lệnh chủ dự án 18/11 (máy bị hãm thì vẫn chọn ứng viên nhanh nhất, ghi lý do, xuất bundle, không dừng Kill (c) chỉ vì
hạ xung). Hai biến thể thu nhỏ đo kèm để có số, không chọn vì luật chỉ dùng biến thể khi bản công bố bị loại theo trần.

## 2. Huấn luyện (`configs/restore.yaml`, run `nafnet-train`)

- Cặp D2 (vào) → D0 (đích), 3 006 ảnh train; patch 256×256 cùng cửa sổ, gốc trên lưới 8 px từ `crop_left` = 16 (bỏ dải
  mép), lật ngang/dọc; loss L1 + (1 − SSIM), không GAN; khởi từ NAFNet-SIDD-width32; AdamW lr 1e-4 cosine, warmup 100,
  clip 0,01, AMP bf16, batch 4, workers 4.
- Smoke đầu batch 8: VRAM đỉnh 8,42 GB (vượt 8 GB, tràn bộ nhớ chia sẻ, 2,4 s/bước) → dừng ở bước 50, đổi batch 4.
- Smoke batch 4: 1,196 s/bước (0,33 giờ/1 000 bước), VRAM đỉnh 4,46 GB. GPU dùng chỉ ~20 % (máy bị hãm, nghẽn ở CPU).
- Số bước từ ngân sách: 4 500 s − đã dùng 1 758 s − dự phòng val 600 s → **1 500 bước** (ckpt mỗi 500).
- Val **số tổng hợp** (cặp D2→D0 val tự sinh, không dùng kết luận): PSNR 13,92 → 22,83 dB, SSIM 0,628 → 0,813 (487 cặp).

## 3. Chọn checkpoint trên khung dev thật (60 chuỗi liền kề, 1 086 khung; thống kê trên 362 khung)

| Checkpoint | flicker_ratio | % hồng giảm | dark channel giảm | blocking dư giảm | điểm |
|---|---|---|---|---|---|
| pretrained (SIDD) | 0,904 | 0,5 % | −1,7 % | 73,0 % | 0,718 |
| step_000500 | 2,007 | 69,3 % | 38,6 % | 69,9 % | 1,779 |
| step_001000 | 1,676 | 92,7 % | 28,5 % | 72,1 % | 1,933 |
| **step_001500** | **1,647** | **93,3 %** | **27,3 %** | **74,3 %** | **1,949** |

Khung gốc dev: pink_frac 0,0071, dark channel 143,0, blocking 2,73. Không checkpoint huấn luyện nào qua
`flicker_max` 1,15: khử mù kéo giãn tương phản nên |Δ| giữa khung liền kề tăng theo (giả thuyết, chưa tách đo). Luật
dự phòng (`train_net.SELECT_RULE`): chọn flicker thấp nhất; step_001500 vừa flicker thấp nhất vừa điểm cao nhất.

## 4. Xuất ONNX và bundle (`export.json`)

- Graph §3: uint8 BGR [480,640,3] → cắt 16 cột → RGB → /255 → fp16 → NAFNet → kẹp, ×255, làm tròn → uint8 → BGR → đệm
  cột [0,16) bằng lặp cột 16. opset 17. LayerNorm tính ở fp32 trong graph fp16: bản đầu chuẩn hoá ở fp16 cho ORT CUDA
  lệch ORT CPU tới 182/255 (MAE 13,4), vượt dung sai golden §4 của IOT-CV; sau sửa lệch lớn nhất 1.
- Parity 20 khung dev: ORT CPU vs PyTorch fp32 MAE 0,029 (≤ 1,0), lệch lớn nhất 1; ORT CUDA vs ORT CPU MAE 0,100,
  lớn nhất 1.
- ms ORT CUDA EP (khung uint8 đủ graph): **p50 75,7 ms, p95 82,1 ms**; CPU 159 % / 158 % trước/sau khi đo (không hãm),
  GPU 76–78 °C. Đây là số chính thức của lần chạy.
- Golden: 3 khung dev `20261102-100941_002_00413`, `_004_01932`, `_007_00750` (không khung test).
- `meta.metrics` (tập test thật) = null: lane không đụng khung test; số dev ở khoá thêm `metrics_dev`.

## 5. Tự xem 3 khung dev trước/sau

Mù và ám hồng vùng sáng giảm rõ (vệt hồng trên đường thành trắng, còn mờ ở khung 000), lưới khối 8×8 mờ đi, người và
vật giữ nguyên chỗ, không thấy chi tiết bịa hay viền lạ. Màu ra nguội và xám hơn (cỏ xanh thành xám xanh), vùng tối
đậm hơn. Cột [0,16) là cột 16 lặp lại theo hợp đồng nên thành vệt dọc.

## 6. Giờ GPU (giữ khoá)

speed lần pin 386 s + speed 417 s + smoke batch 8 bỏ dở ~300 s (ước) + smoke 656 s + train 1 957 s = 3 716 s
(1,03 giờ, trần 1,25 giờ). Ngoài trần đó: select 643 s, export 2 × 74 s, một lần thử xuất ONNX để dò lỗi (vài phút, không ghi giờ).

## 7. Quyết định tự chốt (chủ dự án ngủ)

1. NAFNet "bản nhỏ" = NAFNet-SIDD-width32 (bản công bố nhỏ nhất); FBCNN = fbcnn_color; SCUNet = scunet_color_real_psnr.
2. Chọn nafnet dù p95 > 20 ms (lệnh chủ dự án 18/11, máy bị hãm); bundle p95 82 ms > 30 ms nên
   `test_bundle_ms_p95` FAIL, giữ nguyên test.
3. batch 4 (batch 8 tràn VRAM); 1 500 bước theo ngân sách.
4. Luật chọn checkpoint dự phòng khi không ai qua flicker_max (mục 3).
5. LayerNorm fp32 trong graph fp16 (mục 4); xuất SCUNet không gấp hằng số (lỗi thiết bị của torch 2.6, chỉ ảnh hưởng đo).
6. Viết lại einops/timm của SCUNet bằng reshape/permute (không thêm gói); so với mã gốc: lệch 0,0 (SCUNet), 4,8e-7 (FBCNN).

## 8. Vòng 2 — nafnet_w16 thay width32 (18/11/2026)

Lý do: bundle width32 đo p95 82,1 ms lúc máy **không** bị hãm → `test_bundle_ms_p95` FAIL, phase 13 chưa xong. Trần
30 ms giữ nguyên.

**Đo biến thể** (`speed_round2.json`, 480×624 fp16 ORT CUDA EP, cùng một lần giữ khoá): nafnet_tiny (width 32, 1 khối/tầng,
4,5 M) 34,9 / 36,9 ms; nafnet_lite (6,8 M) 54,8 / 73,9 ms; **nafnet_w16** (width 16, 1 khối/tầng, 1,14 M) 19,0 / 32,9 ms
(p50 / p95). Chi phí nằm ở tầng độ phân giải đủ và tỉ lệ với width, nên mọi bản width 32 đều > 30 ms riêng phần mạng ở
máy này → bỏ tiny và lite, chọn w16. Bản w16 nhiều khối hơn (enc 1,1,2,4, mid 4) đo p95 64,5 ms → bỏ.

**Huấn luyện** (run `nafnet_w16-train`): khởi tạo ngẫu nhiên (width khác 32 nên không nạp được trọng số công bố), cùng
cặp D2→D0, cùng patch/lưới/dải mép, cùng loss L1 + (1 − SSIM), AMP bf16; batch 16, lr 1e-3 cosine (công thức gốc NAFNet),
4 500 bước theo smoke (0,454 s/bước, 257 s GPU), ckpt mỗi 1 000. Hai lần chủ dự án tắt máy (19:39 ở bước 2 200, 22:15 ở
bước 3 000) đều chạy tiếp từ `last.pt`, lịch lr giữ nguyên. **Workers**: bước 0–2 200 là 4 (`configs/restore.yaml`);
2 200–4 500 là 2 (máy ảo VMware chiếm RAM, rồi theo lệnh chủ dự án; lệnh đổi về 4 tới lúc train đã xong nên không áp).
Ghi ở `train.json` khoá `workers_by_steps`. Đoạn cuối 0,323 s/bước, VRAM đỉnh 3,46 GB, cổng CPU lúc vào 166,5 % (không hãm).

**Chọn checkpoint** (cùng luật, 60 chuỗi dev, 1 086 khung flicker, 362 khung thống kê):

| Checkpoint | flicker_ratio | % hồng giảm | dark channel giảm | blocking dư giảm | điểm |
|---|---|---|---|---|---|
| step_001000 | 2,097 | 99,9 % | 47,8 % | 55,0 % | 2,027 |
| step_002000 | 1,908 | 99,9 % | 38,5 % | 60,1 % | 1,985 |
| step_003000 | 1,854 | 99,9 % | 36,3 % | 64,6 % | 2,008 |
| step_004000 | 1,815 | 99,9 % | 34,7 % | 64,1 % | 1,988 |
| **step_004500** | **1,779** | **99,9 %** | **33,6 %** | **63,8 %** | **1,973** |

Không checkpoint nào qua `flicker_max` 1,15 → luật dự phòng chọn flicker thấp nhất: step_004500 (điểm cao nhất là
step_001000, không chọn vì flicker 2,10). Khởi tạo ngẫu nhiên nên không có dòng `pretrained`.

**So sánh width32 (vòng 1) với w16 (bundle hiện tại)**

| | width32 (vòng 1) | **nafnet_w16** (vòng 2) |
|---|---|---|
| Tham số / file ONNX | 29,2 M / 58,9 MB | 1,14 M / 2,4 MB |
| Khởi tạo, bước | NAFNet-SIDD-width32, 1 500 | ngẫu nhiên, 4 500 |
| Checkpoint chọn | step_001500 | step_004500 |
| flicker_ratio (dev thật) | 1,647 | 1,779 |
| % hồng giảm | 93,3 % | 99,9 % |
| dark channel giảm | 27,3 % | 33,6 % |
| blocking dư giảm | 74,3 % | 63,8 % |
| Điểm chọn | 1,949 | 1,973 |
| Val **số tổng hợp** PSNR / SSIM | 22,83 / 0,813 | 22,67 / 0,805 |
| Parity ORT CPU vs PyTorch, MAE / lệch lớn nhất | 0,029 / 1 | 0,029 / 1 |
| ORT CUDA vs ORT CPU, MAE / lệch lớn nhất | 0,100 / 1 | 0,072 / 1 |
| Bundle ORT CUDA EP p50 / p95 | 75,7 / 82,1 ms | **15,4 / 21,0 ms** |
| CPU % Processor Performance trước/sau đo | 159 / 158 % | 165 / 169 % |
| Accept (`test_bundle_ms_p95`, trần 30 ms) | FAIL | **PASS** |
| GPU huấn luyện | 1 957 s | 1 872 s |

**Flicker là kết quả thật.** flicker_ratio 1,65 của width32 và 1,78 của w16 đều > 1 (khung ra rung hơn khung vào) và
> `flicker_max` 1,15; định nghĩa `xla.eval.temporal.flicker_ratio` không đổi, ngưỡng không nới. Flicker của w16 giảm
dần theo bước (2,10 → 1,78) trong khi độ khử mù giảm theo, khớp giả thuyết ở mục 3 (kéo giãn tương phản làm |Δ| giữa
khung liền kề tăng); chưa tách đo nên vẫn là giả thuyết. Lớp gọi trong IOT-CV cần biết điều này trước khi bật mặc định.

**Tự xem 3 khung golden** (vào | width32 | w16, `D:\DevCache\tmp\restore-preview\w16_000.png`…`w16_002.png`): w16 xoá
vệt hồng trên đường gần hết (width32 còn vệt mờ ở khung 000), mù giảm, cỏ giữ màu xanh hơn width32 ở khung 001; người,
bóng và vật giữ nguyên chỗ, không thấy chi tiết bịa. Mất: vân chi tiết mịn mềm hơn width32 một chút, vùng tối đậm hơn;
vật **thật sự** màu hồng (áo hồng ở khung 000, mẩu giấy hồng ở khung 002) cũng bị tẩy thành trắng, vì mạng học xoá ám
hồng của camera mà không phân biệt được vật hồng. Cột [0,16) là cột 16 lặp lại theo hợp đồng.

**Giờ GPU vòng 2** (giữ khoá): speed 92 + 82 s, dò LN ~120 s (ước), smoke 257 s, train 1 872 s = ~2 423 s (40 phút,
trần 45 phút); ngoài trần: select 136 s, export 35 s.

**Quyết định tự chốt ở vòng 2**

1. Chọn nafnet_w16 (width 16, 1 khối/tầng), bỏ tiny/lite vì p95 > 30 ms ngay phần mạng.
2. Khởi tạo ngẫu nhiên, batch 16, lr 1e-3, 4 500 bước; giữ nguyên loss và luật chọn checkpoint của vòng 1.
3. Bundle width32 thay bằng w16 cùng phiên bản `enhancer-v1.0.0` (width32 chưa từng đạt Accept nên không phát hành).
