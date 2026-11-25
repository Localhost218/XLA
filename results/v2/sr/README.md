# Phase 11 AI v2: ảnh rõ offline ×2 (`enhancer-offline-v1.0.0`)

Lane `sr`, nhánh `lane/sr`. Hợp đồng `docs/hop-dong-model.md` §11 (ảnh đơn, `temporal_window` 1). Lệnh:
`python -m xla.enhance.sr bench|train|select|export --config configs/sr_x2.yaml --json`; cặp huấn luyện
`python -m xla.enhance.sr_pairs --config configs/sr_x2.yaml --json`. Trần **1,5 giờ GPU** cả phase (quyết định chủ dự án
25/11, plan ghi 4,5 h); sổ giờ `gpu_ledger.jsonl` (giờ giữ khoá GPU của từng lệnh, không tính lúc chờ khoá).

File số liệu: `bench.json`, `train_l1_percep.json`, `train_l1_percep_gan.json`, `select.json`, `export.json`, ảnh mẫu
`samples/`. Mọi số đo trên RTX 4060 Laptop 8 GB, khối `machine` ghi kèm trong từng file.

## 1. Bench zero-shot (bước 1)

20 khung dev 640×480 (`xla.enhance.export_onnx.dev_frames`), graph §11 đủ (cắt dải mép → mạng ×2 → uint8
[960,1280,3]), fp16. s/khung là trung bình `session.run` gồm chép host↔GPU. fps thật của bản ghi `20261102-100941` =
11 908 khung / 2 117,1 s trong 8 đoạn = **5,62 khung/s**, nên chuyến 20 phút = 6 750 khung.

| Ứng viên | Tham số | PyTorch fp16 s/khung | ORT CUDA fp16 s/khung (p95) | VRAM đỉnh PyTorch / ORT | Giờ cho chuyến 20 phút | Qua ≤ 0,5 s |
|---|---|---|---|---|---|---|
| Real-ESRGAN x2plus (RRDBNet 23 khối) | 16,7 M | 0,330 | **0,406** (0,537) | 665 MB / 2 064 MB | **0,76 h** | có |
| SwinIR-M x2 real (GAN) | 11,7 M | 2,70 | 17,4 (17,9) | 2 161 MB / 7 052 MB | 32,6 h | không |

SwinIR trên ORT chậm hơn PyTorch 6,4 lần: graph 716 MB vì mặt nạ cửa sổ dịch thành hằng số cho từng khối, VRAM ORT
7 GB/8 GB nên tràn sang bộ nhớ chia sẻ. Dù PyTorch 2,7 s/khung cũng đã quá trần 0,5 s, nên không thử tile (tile giảm VRAM,
không giảm phép tính) và SwinIR bị loại ở bước này. Real-ESRGAN qua trần nên cũng không cần tile.
Điều kiện đo: GPU 66 °C lúc vào, 81 °C lúc ra; CPU không hạ xung (2,5 GHz).

## 2. Cặp huấn luyện và fine-tune (bước 2–3)

Kiến trúc vendor (D8, không cài basicsr): `xla/enhance/arch/rrdbnet.py` (RRDBNet Apache-2.0 BasicSR + bộ phân biệt
UNetDiscriminatorSN BSD-3 Real-ESRGAN), `xla/enhance/arch/swinir.py` (Apache-2.0). Cả ba file trọng số nạp `strict`,
sha256 kiểm theo `configs/sr_x2.yaml` (nguồn không công bố checksum; số ghi lúc tải 25/11).

Cặp: 1 600 train + 19 val ảnh VisDrone đủ lớn (val chỉ có 19 ảnh cao ≥ 960 px), cửa sổ HR 1248×960 → hạ ×½
`INTER_AREA` → D2 `params_version` 3 → JPEG đúng camera (640×480 có dải mép). 4 worker, 64,8 s.

| Biến thể | Loss | Bước | s/bước (batch 8, patch LR 128) | VRAM đỉnh | PSNR val tổng hợp (đầu → cuối; bicubic 14,03) | Giờ GPU |
|---|---|---|---|---|---|---|
| `ft_l1_percep` | L1 + perceptual VGG19 | 2 481 | 0,58 | 4,56 GB | 13,90 → **19,92** | 0,41 h |
| `ft_l1_percep_gan` | L1 + perceptual + GAN 0,1 | 206 | 6,44 | 5,84 GB | 13,90 → 15,98 | 0,38 h |

PSNR val là **số tổng hợp** (đích là ảnh sạch, nên đo cả khử mù), không dùng để chọn. Biến thể GAN chậm 11 lần mỗi
bước: bộ phân biệt U-Net ở 256×256 đẩy tổng VRAM thiết bị lên 7,8 GB và tràn bộ nhớ chia sẻ, nên trong 22 phút chỉ chạy
206 bước. Giảm batch xuống 4 có lẽ hết tràn, nhưng không còn trần để chạy lại.

## 3. Chọn trên khung thật (bước 4)

163 khung dev trong 9 chuỗi liền kề (nhấp nháy + thống kê mức thấp trên mỗi khung thứ 2) và 150 khung `cal` không người
(báo nhầm detector v1, `conf_threshold` 0,0767). Ảnh ra hạ ×½ `INTER_AREA` về 640×480 trước khi đo. `ref_bicubic`
(phóng bicubic rồi hạ lại) là mốc cho phần thay đổi do chính phép đổi cỡ, không phải ứng viên.

| Ứng viên | flicker_ratio | pink_frac_drop | dark_channel_drop | blocking_excess_drop | Điểm | Báo nhầm cal / khung (gốc 0,100) | Kết quả |
|---|---|---|---|---|---|---|---|
| ref_bicubic (mốc) | 0,993 | 0,002 | −0,003 | 0,166 | 0,165 | 0,100 | mốc |
| Real-ESRGAN x2plus zero-shot | 0,831 | −0,047 | −0,039 | 0,858 | 0,772 | **0,147** (22 hộp) | loại: tăng báo nhầm |
| SwinIR-M x2 real zero-shot | – | – | – | – | – | – | loại: quá chậm (bench) |
| `ft_l1_percep` | **1,348** | 1,000 | 0,331 | 0,758 | 2,089 | **0,067** (10 hộp) | **chọn** |
| `ft_l1_percep_gan` | 1,553 | 1,000 | 0,478 | 0,917 | 2,395 | 0,093 (14 hộp) | qua, không chọn |

Luật (`xla.enhance.sr.SELECT_RULE`): loại ứng viên quá chậm hoặc tăng báo nhầm cal; không ai qua `flicker_max` 1,15 thì lấy
nhấp nháy thấp nhất, như luật chọn checkpoint phase 13. Lần chạy `select` đầu dùng luật "điểm cao nhất" và chọn biến thể
GAN; luật được sửa theo tiền lệ phase 13 rồi `select --rescore` (không chạy GPU lại) chọn `ft_l1_percep`, ghi lựa chọn cũ
ở `select.json` → `rescored_from`. Lý do: bản không GAN ít nhấp nháy hơn (1,35 so với 1,55), ít báo nhầm hơn
(0,067 so với 0,093), không có loss GAN nên ít rủi ro bịa chi tiết, và bản GAN mới chạy 206 bước.

Ảnh mẫu (`samples/<ứng viên>.jpg`, trái gốc phóng ×2 nearest, phải ảnh ra): bản fine-tune xoá vệt hồng và khử mù, nhưng
màu ngả lạnh và nhạt, đúng rủi ro "màu nguội, nhạt" của v1 ghi trong plan. Không ứng viên nào qua `flicker_max`: nhấp
nháy tăng 35 % so với gốc. Hai điểm này để hàng rào + A/B phase 14 quyết, plan không có dự phòng (Kill (d) nếu bị loại).

## 4. Xuất `enhancer-offline-v1.0.0` (bước 5)

`models/release/enhancer-offline-v1.0.0/` (meta.json + golden/; file `.onnx` không vào git, đính Release ở phase 17):
graph §11 opset 17, fp16, ảnh đơn, `network.candidate` = `ft_l1_percep` (RRDBNet khởi từ RealESRGAN_x2plus, 2 481 bước),
`onnx_sha256` `e93b3eaa714a003df0c6a77c11dacefc7b22666143dc8c490957dc1a5b53afc8`. `python -m xla.contract validate`:
`ok`, 0 lỗi. Checkpoint gốc: `D:\DevCache\runs\xla\v2\sr\l1_percep\final.pt` (không vào git).

| Đo | Giá trị |
|---|---|
| Parity ORT CPU vs PyTorch fp32 (5 khung dev, gồm 3 golden) | MAE lớn nhất 0,118, lệch lớn nhất 3/255 (trần MAE 1,0) |
| ORT CUDA vs ORT CPU (cùng 5 khung) | MAE lớn nhất 0,131, lệch lớn nhất 6 (dung sai golden §11: MAE ≤ 1, lớn nhất ≤ 8) |
| `s_per_frame` (ORT CUDA fp16, 20 khung dev) | 0,404 s (p50 0,378, p95 0,526), chuyến 20 phút ≈ 0,76 h |
| `peak_vram_mb` (ORT, cả thiết bị trừ nền) | 2 064 MB |

Lần xuất đầu (chuỗi tự chạy ngay sau `select` lần đầu) dựng bản GAN trước khi đổi luật chọn: parity CPU đạt
(MAE 0,198) nhưng CUDA so với CPU lệch lớn nhất 11 > 8 nên `ok: false`. Bản đó đã bị ghi đè bằng lần xuất thứ hai ở trên;
báo cáo cũ giữ ở `D:\DevCache\runs\xla\v2\sr\export_gan_superseded.json`.

`meta.metrics` (báo nhầm cal/holdout trên ảnh offline) để trống cho phase 14 ghi; số dev của `select` ở `metrics_dev`.

## 5. Giờ GPU

| Bước | Giờ GPU |
|---|---|
| bench (2 ứng viên, gồm xuất ONNX SwinIR 716 MB) | 0,139 |
| fine-tune `l1_percep` | 0,407 |
| fine-tune `l1_percep_gan` | 0,376 |
| select (3 ứng viên + mốc) | 0,113 |
| export ×2 | 0,012 |
| **Tổng** | **1,047 / 1,5** |

Thời gian chờ khoá GPU (lane khác đang chạy) không tính vào sổ: 1,79 giờ đồng hồ (cộng `wait_lock_seconds` các bước).

## 6. Kết luận

Ship ứng viên `ft_l1_percep` cho hàng rào + A/B phase 14: đủ nhanh (0,40 s/khung, 0,76 h cho chuyến 20 phút, đích
≤ 1 h), không tăng báo nhầm detector v1 trên `cal` (0,067 so với 0,100 của khung gốc; mẫu nhỏ: 10 so với 15 hộp trên 150
khung). Chưa đạt: nhấp nháy 1,35 > `flicker_max` 1,15, và màu ngả lạnh. Không có ứng viên dự phòng (plan): phase 14 loại
thì Kill (d), web ẩn nút offline.
