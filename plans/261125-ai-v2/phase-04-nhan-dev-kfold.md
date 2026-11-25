# Phase 04 — Nhãn dev mở rộng + fold theo phiên (lane realdata)

Lane: realdata
Depends-on: 03
Effort: L
Repo: XLA, worktree `D:\XLA-wt\realdata` (tiếp sau phase 03)
Owns: `xla/labels/devpool.py`, `data/real/labels/devpool/**`, `data/real/labels/folds.json`, `tests/labels/test_dev_folds.py`
Creates: `xla/labels/devpool.py`, `tests/labels/test_dev_folds.py`
AX: AX-01 `python -m xla.labels.devpool select|prelabel|apply|folds --json`
Accept: `pytest tests/labels/test_dev_folds.py`

## Mục tiêu

Thay luật chọn model bằng 100 khung dev-val (quá nhiễu: E1-D0 tốt nhất trên test nhưng không được chọn, `results/e1/README.md`)
bằng một tập dev có nhãn lớn hơn, chia fold theo phiên, để phase 06/07/09 chọn model và ngưỡng trên dự đoán ngoài fold (OOF).

## Đầu vào

`data/real/sessions.json`, `data/real/split.json` (chỉ đọc), `xla/detect/prelabel.py` (`run` :169, `fuse` :80),
`xla/detect/review.py` (`cmd_render` :152, `cmd_apply` :245), `xla/detect/devval.py` (`DEVVAL_N` :35, manifest).

## Bước

1. Chọn `devpool`: mọi khung dev-val 100 (giữ nhãn cũ) + mẫu phân tầng (phiên × bin độ cao × có/không ứng viên người)
   từ khung dev v1 và khung phiên `mine`, cách nhau ≥ 1 s, đích 500–700 khung, ≥ 150 khung có người. Không khung test/cal/holdout.
2. Gán nhãn trước (D6 khoá GPU) như phase 08 v1; agent soát toàn bộ bằng ảnh ghép (`cmd_render`), sửa bằng `cmd_apply`;
   hộp chỉ một model thấy được soát kỹ. Đây là tập chọn model, không phải tập chấm: không cần chủ dự án duyệt.
3. `folds`: GroupKFold theo phiên khi ≥ 3 phiên có khung `devpool` có người; không đủ thì gom khối 30 s liền nhau thành 5 fold
   (Deferred 6). Ghi `folds.json` + số khung/người mỗi fold.
4. README `data/real/labels/devpool/README.md`: cách chọn, số hộp, chênh với dev-val cũ.

## Tiêu chí xong

Accept xanh: không giao với test/cal/holdout; mọi fold có ≥ 20 khung có người; nhãn YOLO hợp lệ 640×480; dev-val 100 cũ là
tập con với nhãn không đổi; seed ghi lại, chạy lại ra cùng `folds.json`.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Nhãn agent soát thiên về model gán trước | medium | medium | Model gán khác họ trò (RT-DETR + YOLO11x); báo chênh dev-val cũ (nhãn đã duyệt) như kiểm chéo |
| Một phiên chiếm hầu hết người | high | medium | Fold khối 5 khi < 3 phiên; báo phương sai giữa fold |

Rollback: xoá `devpool/` và `folds.json`; phase 06 lùi về dev-val (ghi rõ trong kết quả).
