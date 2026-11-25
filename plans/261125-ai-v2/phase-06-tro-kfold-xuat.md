# Phase 06 — Trò YOLO11s, chọn bằng k-fold theo phiên, xuất `detector-v2.0.0` (lane detect)

Lane: detect
Depends-on: 01, 04, 05
Effort: L
Repo: XLA, worktree `D:\XLA-wt\detect` (tiếp sau phase 05)
Owns: `configs/student_v2.yaml`, `xla/detect/kfold.py`, `xla/eval/detection.py` (chỉ: nhận OOF devpool + loại `cal`, `holdout` trong `NEG_KINDS`), `models/release/detector-v2.0.0/**`, `results/v2/detect/**`, `tests/detect/test_student_v2.py`
Creates: `xla/detect/kfold.py`, `configs/student_v2.yaml`, `tests/detect/test_student_v2.py`
AX: AX-01 `python -m xla.detect.kfold run|select|tiling --config configs/student_v2.yaml --json`
Accept: `pytest tests/detect/test_student_v2.py`

## Mục tiêu

Chọn trò YOLO11s (≤ 30 ms) bằng OOF k-fold trên `devpool` thay vì dev-val 100 khung, đặt ngưỡng detector bằng F1 tối đa
trên OOF, xuất `detector-v2.0.0` (hợp đồng §2 không đổi; `recommended_track` thêm ở phase 09). Đo thêm cắt ô SAHI cho
offline. Trần **6 giờ GPU** tổng (mọi lần huấn luyện + xuất + đo).

## Đầu vào

`configs/student_v2.yaml` (C1, C2, C3 — Deferred 9), `data/real/labels/{devpool,folds.json}`, nhãn giả phase 05,
`xla/detect/train.py`, `xla/eval/detection.py` (`select_f1_threshold` :254-271, `NEG_KINDS` :277), `xla/detect/export_onnx.py`,
`repo: obss/sahi` trong venv (không có → dừng, báo, D8).

## Bước

1. C1 (D0+D2 trộn + nền mine) và C3 (D0 + nền mine): mỗi cái **một** lần huấn luyện (không thấy devpool) → dự đoán trên
   toàn devpool là OOF.
2. C2 (C1 + nhãn giả dev): mỗi fold i huấn luyện loại nhãn giả của khung thuộc fold i, dự đoán fold i → OOF ghép. Cùng
   epoch, imgsz 1280, batch, seed cho mọi ứng viên; mỗi lần một khoá GPU (D6). Giữ checkpoint từng fold.
3. `select`: mAP50 OOF (bootstrap theo fold), báo nhầm/khung trên khung không người của devpool và trên `mine`; chọn mAP50
   OOF cao nhất, hoà (CI chồng) thì chọn đơn giản hơn (C3 < C1 < C2). Bảng vào `results/v2/detect/README.md`.
4. Bản cuối: C1/C3 dùng luôn run bước 1; C2 huấn luyện lại trên mọi fold. Ngưỡng = F1 tối đa trên OOF (`select_f1_threshold`
   mở rộng, chặn test + cal + holdout). Nếu C2 thắng: xuất thêm ONNX **từng fold** (`D:\DevCache\runs\xla\v2\detect\folds\
   detector-fold<i>.onnx`) để phase 09 đo recall OOF thật bằng `replay_pred.py --detector-onnx`.
5. Kill (b): không ứng viên nào hơn `detector-v1.0.0` trên OOF (cận dưới CI hiệu ghép cặp > 0) → không xuất v2, ghi lý do.
6. Xuất `models/release/detector-v2.0.0` bằng `export_onnx.py` (`--version 2.0.0 --conf <thr> --imgsz 1280`), `xla.contract
   validate`, đo p50/p95 ms ORT CUDA (≤ 30 ms, ghi `machine`).
7. Chấm test (chỉ chấm, không chọn): mAP50, AR-small, recall ở ngưỡng, `compare` với `edge16` và v1.
8. Cắt ô: `tiling` chạy SAHI (ô 640, chồng 0,2) với trò thắng trên test, báo mAP50 + ms; chỉ ghi kết quả (Deferred 15).

## Tiêu chí xong

Accept xanh: `select` ném lỗi nếu nhận khung test/cal/holdout; bundle v2 qua `validate`; README có bảng OOF + test + ms +
tiling + giờ GPU đã dùng; trò ≤ 30 ms p95.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Giờ GPU k-fold vượt trần | medium | medium | Chỉ C2 huấn luyện theo fold; epoch từ smoke phase 05; trần 6 giờ ghi trong config, dừng ứng viên khi chạm |
| OOF vẫn không dự đoán được test | medium | high | Báo tương quan OOF ↔ test mọi ứng viên; không tuyên bố quá CI |
| Ngưỡng F1 OOF cho báo nhầm cao | high | medium | Báo nhầm xử lý ở tầng 2 + 3 (phase 07, 09); detector giữ recall |

Rollback: không xuất / xoá `detector-v2.0.0`; IOT-CV giữ v1 qua `.env`.
