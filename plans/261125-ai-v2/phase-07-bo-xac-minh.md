# Phase 07 — Verifier: hard negative từ mine, phân loại vùng cắt, `verifier-v1.0.0` (lane verifier)

Lane: verifier
Depends-on: 01, 03, 06
Effort: L
Repo: XLA, worktree `D:\XLA-wt\verifier`, nhánh `lane/verifier` từ `feat/ai-v2` (sau khi gộp phase 06)
Owns: `xla/verify/**`, `configs/verifier.yaml`, `models/release/verifier-v1.0.0/**`, `results/v2/verifier/**`, `tests/verify/test_verifier.py`
Creates: `xla/verify/__init__.py`, `xla/verify/mine.py`, `xla/verify/train.py`, `xla/verify/export_onnx.py`, `tests/verify/test_verifier.py`
AX: AX-01 `python -m xla.verify.mine|train|export_onnx --config configs/verifier.yaml --json`, mã thoát 0/1/2
Accept: `pytest tests/verify/test_verifier.py`

## Mục tiêu

Tầng 2: bộ phân loại nhỏ nhận vùng cắt quanh từng hộp của detector v2, trả P(người). Lớp âm là **hộp báo nhầm thật** trên
khung `mine` (túi rác, bụi cây, bóng) + vật dễ nhầm trong dữ liệu ngoài. Ngưỡng sơ bộ trên `cal` (không trên mine đã huấn
luyện, không trên holdout). Trần **1 giờ GPU**.

## Đầu vào

`contract.md` §A (IO, meta, golden), detector v2 (phase 06; Kill (b) thì detector v1), `data/real/neg_flight_split.json`,
`data/real/labels/{devpool,folds.json}`, `d02x` (phase 02), `xla/detect/review.py` (`cmd_crop` :173), torchvision.

## Bước

1. `mine`: detector ở `score_floor` trên mọi khung `mine` → hộp trên khung không người = âm; hộp khớp nhãn devpool (IoU ≥ 0,5)
   = dương; thêm dương từ GT `d02x`, âm từ hộp báo nhầm trên ảnh nền `d02x`. Trang soát 200 âm ngẫu nhiên để agent bỏ người
   sót. Manifest (khung, hộp, nguồn, fold) — không khung cal/holdout/test (test kiểm).
2. Huấn luyện: `mobilenet_v3_small` và `efficientnet_b0` (torchvision, ImageNet), crop 64 và 96, `context_scale` 1,5;
   augmentation nhẹ (lật, sáng, JPEG theo D2); `workers ≤ 4`, checkpoint mỗi epoch. Chọn bằng k-fold: dương theo `folds.json`,
   âm theo phiên của `mine`; tiêu chí: giảm báo nhầm OOF ở recall OOF giảm ≤ 0,05.
3. Ngưỡng `conf_threshold` sơ bộ trên `cal` + OOF devpool theo D7 (phase 09 chốt cùng k/n).
4. Kill (c): giảm báo nhầm trên `cal` < 50 % ở recall OOF giảm ≤ 0,05 → không xuất, ghi lý do.
5. `export_onnx`: graph §A (nới hộp → kẹp → `RoiAlign output_half_pixel` → chuẩn hoá → backbone → sigmoid), N động; parity
   ORT (CPU và CUDA) vs PyTorch `roi_align(aligned=False)` trên golden `|Δ| ≤ 0,02`; ms ORT CUDA với N = 1, 5, 20 (đích < 3 ms ở N = 5).
6. Bundle `models/release/verifier-v1.0.0` (meta §A, golden), `xla.contract validate` ra 0. README `results/v2/verifier/` (có giờ GPU).

## Tiêu chí xong

Accept xanh: manifest không có khung cal/holdout/test; graph nhận N = 0; parity đạt; meta đủ trường §A; ms ghi `machine`.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Âm mine quá ít / một kiểu | medium | high | Thêm âm từ ảnh nền `d02x`; báo phân bố loại âm; human step tuỳ chọn ghi phiên trống |
| Vùng cắt quá nhỏ (người vài px) | high | medium | `min_side_px` + crop 64/96 chọn bằng k-fold; báo recall theo cỡ hộp |
| RoiAlign ORT lệch | low | medium | Parity bước 5 + test bundle giả phase 01 |

Rollback: không ship verifier; IOT-CV chạy detector + track (§A cho phép thiếu verifier).
