# Phase 13 — mạng khôi phục riêng camera (lane restore, 18/11/2026)

Bundle: `models/release/enhancer-v1.0.0/` (`meta.json` + `golden/` trong git; `enhancer.onnx` không commit, sha256
`2ad08ebf6fd7adfbd0b08bcb9e6a5064a8611e7fcc587a243dfb34e9d0f79669`, nằm ở cùng thư mục trên máy).
`python -m xla.contract validate models/release/enhancer-v1.0.0` → `[]`.

Số liệu gốc: `speed.json` (đo tốc độ), `train.json` (smoke, train, select), `export.json` (parity, ms, điều kiện đo).

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
