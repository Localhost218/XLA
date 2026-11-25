# Phase 09 — Hiệu chỉnh k/n trên cal, chấm cả hệ trên test + holdout (serial)

Lane: serial
Depends-on: 03, 06, 07, 08
Effort: M
Repo: XLA (`feat/ai-v2`, sau khi gộp lane realdata, detect, verifier) + IOT-CV `lane/iotcv` (chỉ chạy, không sửa)
Owns: `xla/eval/system.py`, `models/release/detector-v2.0.0/meta.json` (chỉ thêm `recommended_track`), `models/release/verifier-v1.0.0/meta.json` (chỉ `conf_threshold`), `results/v2/system/**`, `plans/261125-ai-v2/plan.md` (chỉ Forecast re-score), `tests/integration/test_system_v2.py`
Creates: `xla/eval/system.py`, `tests/integration/test_system_v2.py`
AX: AX-01 `python -m xla.eval.system calibrate|score --iotcv <worktree> --json`, gọi `replay_pred.py` (§E) với env con sạch
Accept: `pytest tests/integration/test_system_v2.py`

## Mục tiêu

Chốt điểm vận hành cả hệ (ngưỡng verifier + k/n) **chỉ trên `cal` + OOF devpool** (D7), rồi chấm **một lần** trên test (B1,
recall) và holdout đã duyệt (B2), bằng đúng code runtime IOT-CV. Mở PR XLA đầu tiên.

## Đầu vào

`contract.md` §B, §E, §F; bundle `detector-v2.0.0`, `verifier-v1.0.0` (chép vào `D:\IOTCV-wt\iotcv\models\vision\` bằng
`lay_model.py`); ONNX từng fold nếu C2 thắng (phase 06 bước 4); `data/real/{sessions,neg_flight_split}.json`; `xla/eval/detection.py`.

## Bước

1. Gộp `lane/realdata`, `lane/detect`, `lane/verifier` vào `feat/ai-v2` (XLA); chạy toàn bộ pytest CPU.
2. `calibrate`: lưới k ∈ {1..4}, n ∈ {k..6}, `max_gap` ∈ {1, 2}, ngưỡng verifier bước 0,05 → `replay_pred.py --stage track`
   trên phiên `cal` (báo nhầm/khung) và trên các bản ghi chứa khung devpool (recall). **Luôn** truyền `--blank` = mọi khung
   test + holdout (+ mine khi `cal_mode: block`, để chỉ khung cal và devpool tạo ngữ cảnh track). Nếu C2 thắng, recall của
   khung fold i đo bằng `--detector-onnx detector-fold<i>.onnx` (OOF thật). Chọn theo D7, ưu tiên điểm đơn giản nhất trong
   vùng hoà. Ghi `results/v2/system/calibrate.json` + đường cong báo nhầm–recall.
3. Ghi `recommended_track` vào meta detector v2 và `conf_threshold` vào meta verifier; `xla.contract validate` cả hai.
4. `score` (một lần, không quay lại bước 2): test v1 — mAP50, AR-small, recall ở ngưỡng, CI bootstrap khối, `compare` với
   `edge16` và v1 ở ba tầng (detector / +verifier / +track; tầng track phát lại với `--blank` = holdout + cal); holdout —
   báo nhầm/khung theo tầng, theo phiên, theo ngày, CI (blank = test + cal + mine).
5. ms: `replay_bench.py` trên `logs/recordings/20261102-100941` với v2 → p95 từng bước, so v1 (+5 ms trần); ghi khối `machine`
   kèm số (không hoãn vì nhiệt). `test_ngan_sach_ms` không sửa.
6. `results/v2/system/README.md`: bảng chỉ số §5 brainstorm (hiện tại → v2, đạt/không đạt), CI, giới hạn (vật dễ nhầm có thể
   lặp giữa phiên cùng ngày). Owner chép số đo của các README lane (02, 03, 05, 06, 07) vào Forecast plan.md (Re-score).
7. PR XLA `feat/ai-v2` → `main` (nhánh B), mô tả có bảng + đường dẫn README; merge khi CI xanh.

## Tiêu chí xong

Accept xanh: test khẳng định `calibrate` luôn blank đủ test + holdout (bắt lệnh subprocess), không đọc nhãn test/holdout,
`score` không ghi meta, CSV cả hệ đúng cột §E; README đủ năm dòng chỉ số nhánh B.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ngữ cảnh track kéo phát hiện từ khung test sang khung dev kề | medium | high | `--blank` + xoá track ở biên đoạn (§E); test bước "Tiêu chí xong" |
| Lưới k/n quá khớp cal | medium | medium | Lưới nhỏ, điểm đơn giản nhất trong vùng hoà; holdout chấm một lần |
| Subprocess uv xoá venv repo kia | medium | high | Env con bỏ `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` (test kiểm env con) |
| B2 không đạt | medium | high | Báo đúng số + CI; không đổi holdout, không chấm lại sau khi xem |

Rollback: bỏ `recommended_track` khỏi meta (IOT-CV lùi về hành vi mặc định); revert PR.
