# Phase 11 — Offline SR ×2: Real-ESRGAN x2plus, SwinIR-M x2 real; `enhancer-offline-v1.0.0` (lane sr)

Lane: sr
Depends-on: 01, 03
Effort: L
Repo: XLA, worktree `D:\XLA-wt\sr`, nhánh `lane/sr` từ `feat/ai-v2`
Owns: `xla/enhance/sr_pairs.py`, `xla/enhance/arch/rrdbnet.py`, `xla/enhance/arch/swinir.py`, `xla/enhance/sr.py`, `configs/sr_x2.yaml`, `xla/enhance/train_net.py` (chỉ: đích ×2, loss perceptual + GAN tuỳ chọn), `models/release/enhancer-offline-v1.0.0/**`, `results/v2/sr/**`, `tests/enhance/test_offline_sr.py`
Creates: `xla/enhance/sr_pairs.py`, `xla/enhance/arch/rrdbnet.py`, `xla/enhance/arch/swinir.py`, `xla/enhance/sr.py`, `configs/sr_x2.yaml`, `tests/enhance/test_offline_sr.py`
AX: AX-01 `python -m xla.enhance.sr bench|train|select|export --config configs/sr_x2.yaml --json`
Accept: `pytest tests/enhance/test_offline_sr.py`

## Mục tiêu

Ảnh rõ offline cho trang Lịch sử bay: model lớn không giới hạn ms, ×2, chạy theo lô cả chuyến. Đích vận hành: chuyến 20 phút
xong ≤ 60 phút trên RTX 4060 Laptop 8 GB; không bịa người (lọc ở bước 4, hàng rào ở phase 14). Trần **4,5 giờ GPU**.

## Đầu vào

`contract.md` §C; trọng số `RealESRGAN_x2plus.pth` (release v0.2.1, BSD-3) và `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth`
(Apache-2.0) tải về `D:\IOT_Tools\downloads\weights\` rồi `D:\DevCache\weights\`; ảnh sạch VisDrone qua `load_visdrone` (chỉ
đọc; thêm ảnh ngoài phase 02 nếu đã có); `xla.degrade.d2_camera.degrade` (REUSE, `params_version` 3); `detector-v1.0.0` (để lọc
báo nhầm, không chờ phase 06); `data/real/neg_flight_split.json` (chỉ `cal`); hàm ghi bundle/golden của `xla/enhance/export_onnx.py`
(chỉ đọc, gọi lại); `xla/eval/{temporal,enhance_eval}.py`.

## Bước

1. `bench` (trước mọi huấn luyện, 30 phút): hai trọng số zero-shot trên 20 khung dev 640×480, PyTorch fp16 và ORT CUDA fp16:
   s/khung, VRAM đỉnh; ước giờ cho 20 phút video ở fps thật của bản ghi (đếm khung `logs/recordings/20261102-100941`).
   > 0,5 s/khung cả hai → thử tile trong graph (Slice + Concat chồng 16 px); vẫn quá → Kill (d). Số vào README.
2. Vendor `rrdbnet.py`, `swinir.py` (giấy phép đầu file, D8), không cài `basicsr`.
3. `sr_pairs`: ảnh sạch HR (cửa sổ 1248×960) → hạ ×½ → D2 → cặp (LR 624×480, HR 1248×960), `--workers 4`, seed theo ảnh, ghi
   `D:\DevCache\datasets\xla\sr_pairs_x2\` + manifest. Fine-tune Real-ESRGAN x2plus: (i) L1 + perceptual + GAN 0,1, (ii) L1 +
   perceptual (không GAN); bf16, `last.pt` mỗi 200 bước + resume, `workers ≤ 4`, khoá GPU D6.
4. `select` trên khung dev thật: thống kê mức thấp (dark channel, blocking, % hồng), nhấp nháy, và **báo nhầm detector v1 trên
   khung `cal` sau SR** (hạ ×½ về 640×480) — ứng viên tăng báo nhầm bị loại. Ứng viên: 2 zero-shot + 2 fine-tune; chọn **một**
   cho A/B. Không có dự phòng: nếu ứng viên bị hàng rào phase 14 loại thì không ship (Kill (d)).
5. `export` trong `sr.py`: graph §C (dải mép ×2), fp16, parity ORT vs PyTorch MAE ≤ 1,0 trên golden; meta bắt buộc §C
   (`peak_vram_mb`, `s_per_frame`, `machine`); bundle `models/release/enhancer-offline-v1.0.0`, `xla.contract validate`.
6. `results/v2/sr/README.md`: bảng ứng viên, s/khung, VRAM, chỉ số dev, báo nhầm cal, ảnh mẫu, giờ GPU.

## Tiêu chí xong

Accept xanh: kiến trúc vendor nạp đúng trọng số công bố (checksum); graph ra [960,1280,3]; parity đạt; meta đủ trường bắt buộc
§C; `select` từ chối ứng viên tăng báo nhầm (test với số giả).

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GAN bịa chi tiết / người giả | medium | high | Biến thể không GAN; lọc báo nhầm cal ở bước 4; hàng rào phase 14 |
| Quá chậm / tràn VRAM | medium | medium | Bước 1 đo trước; tile trong graph; Kill (d) |
| Cặp tổng hợp lệch camera thật (lỗi v1: màu nguội, nhạt) | medium | medium | Chọn trên khung thật; so zero-shot |

Rollback: không xuất bundle; web ẩn nút offline (API 503 theo §D).
