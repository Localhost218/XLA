# Phase 05 — Thầy YOLO11l/m @1280 + nhãn giả dev (lane detect)

Lane: detect
Depends-on: 01, 02, 03, 04
Effort: L
Repo: XLA, worktree `D:\XLA-wt\detect`, nhánh `lane/detect` từ `feat/ai-v2`
Owns: `configs/teacher_v2.yaml`, `xla/detect/selftrain.py` (chỉ thêm nguồn + trọng số thầy cấu hình được), `xla/detect/train.py` (danh sách cho phép khung thật + ảnh nền), `results/v2/teacher/**`, `tests/detect/test_teacher_pseudo.py`
Creates: `configs/teacher_v2.yaml`, `tests/detect/test_teacher_pseudo.py`
AX: AX-01 `python -m xla.detect.selftrain teacher|pseudo --config configs/teacher_v2.yaml --smoke|--json`
Accept: `pytest tests/detect/test_teacher_pseudo.py`

## Mục tiêu

Một thầy lớn hơn huấn luyện trên `d02x` (VisDrone + ngoài, D0 + D2) cộng khung `mine` làm ảnh nền (nhãn rỗng), rồi gán
nhãn giả cho khung dev v1 (làm sạch bằng ByteTrack như E2) để phase 06 có ứng viên C2. Trần **6 giờ GPU**.

## Đầu vào

`xla/detect/selftrain.py` (`_bytetrack_without_lap` :253, `track_sequences` :296, `choose_workers` :123), `configs/selftrain_e2.yaml`,
`xla/detect/train.py` (`EpochTracker` :128-186, resume :214-224, chặn khung thật :84-86), tập `d02x` (phase 02),
`data/real/neg_flight_split.json` (chỉ `mine`), `data/real/labels/folds.json` (phase 04).

## Bước

1. Smoke VRAM + giờ: YOLO11l @1280 AMP batch 2 và 4, 1 epoch trên 10 %; vừa batch ≥ 2 → l, không thì m (Deferred 8). Ghi
   `results/v2/teacher/smoke.json` (VRAM đỉnh, giờ/epoch, `machine`).
2. `train.py`: thay chặn kiểu danh sách cấm (:84-86) bằng **danh sách cho phép** theo `contract.md` §F — khung thật chỉ
   được vào huấn luyện nếu là `mine` (ảnh nền, ≤ 10 % tập) hoặc `dev` của `split.json` mang nhãn giả và không thuộc fold
   đang loại; mọi khung thật khác (test, cal, holdout, chưa gán) → `ValueError`.
3. Huấn luyện thầy (epoch từ smoke), `workers ≤ 4`, checkpoint mỗi epoch, resume; chọn checkpoint theo mAP50 trên `val`
   của `d02x` (không dùng khung thật có nhãn nào).
4. Nhãn giả: thầy + TTA trên mọi khung dev v1 (không test/cal/holdout), làm sạch ByteTrack (`pseudo.tracker` như E2 — chỉ
   để làm sạch nhãn giả, không phải bộ lọc sự kiện, D4), ghi `D:\DevCache\datasets\xla\pseudo_v2\` + manifest (khung,
   phiên, fold theo `folds.json`).
5. `results/v2/teacher/README.md`: VRAM, giờ GPU, mAP50 val, số nhãn giả, chấm thầy trên `devpool` (thầy không thấy devpool,
   nên mọi khung là OOF).

## Tiêu chí xong

Accept xanh: config hợp lệ; `train.py` từ chối khung test/cal/holdout và khung chưa gán (test với manifest giả); manifest
nhãn giả không chứa khung test/cal/holdout; mỗi khung nhãn giả mang `session` và `fold`; smoke JSON có VRAM và giờ.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| YOLO11l không vừa 8 GB @1280 | medium | medium | Bước 1 → YOLO11m (v1 đã chạy m làm thầy) |
| Ảnh nền làm tụt recall | medium | medium | Tỉ lệ ≤ 10 %, ứng viên C1/C3 ở phase 06 để so |
| Máy sập giữa huấn luyện | medium | medium | Checkpoint mỗi epoch, resume từ `last.pt` |

Rollback: xoá run trong `D:\DevCache\runs\xla\v2\detect`; revert nhánh lane.
