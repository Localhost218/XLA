# Rà soát plan AI v2 (261125-ai-v2), phase 01–18 — 26/11/2026

Lượt rà chỉ đọc (agent investigator, phiên cook phase 13–18). Phase 15, 17 lúc rà đang chạy. Mục §5 ghi ai xử lý tiếp.

## 0. Phạm vi
- XLA `D:\XLA` `feat/ai-v2` @ `3fadacc`, env theo plan.md "Luật chung", `-p no:cacheprovider -m "not gpu"`.
- IOT-CV: chỉ chạy ở `D:\IOTCV-wt\iotcv-debt` (`f2c562d`), venv `d:\Coding\IOT-CV\.venv` (pyproject + uv.lock trùng byte).
  Accept 08/13/18 không chạy lúc rà (lane/iotcv đang gộp; phase 15 chạy lại toàn bộ).
- Không chạy GPU, không huấn luyện.

## 1. Accept đã chạy
| Phase | Lệnh | Kết quả |
|---|---|---|
| 01 | tests/contract/test_model_contract.py | 52 passed; `make-dummy --kind verifier` → validate 0; 6 bundle validate 0 |
| 02 | tests/degrade/test_external_sources.py | 19 passed |
| 03 | tests/camera/test_sessions_neg_split.py | 32 passed |
| 04 | tests/labels/test_dev_folds.py | 11 passed |
| 05 | tests/detect/test_teacher_pseudo.py | 16 passed |
| 06 | tests/detect/test_student_v2.py | 12 passed |
| 07 | tests/verify/test_verifier.py | 12 passed cục bộ — **CI đỏ** (§4) |
| 09 | tests/integration/test_system_v2.py | 12 passed |
| 10 | tests/enhance/test_classic_onnx.py | 15 passed |
| 11 | tests/enhance/test_offline_sr.py | 17 passed |
| 12 | tests/enhance/test_video_sr.py | 12 passed |
| 14 | tests/eval/test_ab_v2.py | 12 passed |
| 16 | backend/tests/test_repo_hygiene.py (iotcv-debt) | 2 passed; backend/tests 423 passed |

## 2. Kết luận từng phase
| Phase | Kết luận |
|---|---|
| 01 | Đạt. `tests/contract/fixtures/` không có (bundle giả sinh trong tmp) — lệch có lý do. |
| 02 | Đạt, có lệch (TinyPerson ngoài d02x, chỉ làm dương verifier; manifest ngoài git). |
| 03 | Đạt; Kill (a) holdout 0; trang duyệt holdout không cần chạy (0 khung). |
| 04 | Lệch: devpool 160 khung / 70 có người (đích 500–700 / ≥ 150) — chỉ bay thêm mới đạt; 2 fold đủ ≥ 20 khung có người. |
| 05 | Đạt. Thầy m (l dừng epoch 1), trần 1 h, C2 bỏ. |
| 06 | Kill (b): không `detector-v2.0.0`; p95 22,1 ms ≤ 30. |
| 07 | Đạt cục bộ; ms N=5 5,96 > 3 (không đạt, đã ghi); CI đỏ một test. |
| 08 | File đủ trên lane/iotcv. |
| 09 | Đạt (số đo B1/recall/ms không đạt đích, B2 không chứng minh); PR #7 merged. |
| 10 | Đạt; lai không qua, không v2.1.0; `meta.metrics` của enhancer-v2.0.0 còn null. |
| 11 | Đạt; nhấp nháy 1,35 > 1,15 (đã ghi). |
| 12 | Đạt (dừng `killed: onnx` có số). |
| 13 | File đủ; bước 5 `.gitignore` chưa làm (bị chặn quyền) → thay bằng cache mặc định `logs/.cache/flights-ai` (đã ignore). |
| 14 | Đạt; lượt cả khung chấm, `offline_ship=false`; lượt phóng to không chấm (không quyết định). |
| 16 | Đạt. |

## 3. Plan
- Forecast :37 ("chưa gộp lane sr") cũ; dòng :28-31 chưa chấm lại bằng số đo (A/B 25/25, 16/25; guard pass; 0,76 h/chuyến).
- Deferred thiếu: TinyPerson chỉ verifier; trần GPU chủ dự án (05 1 h, 06 2 h, 07 30 ph, 11 1,5 h, 12 1 h, lai 30 ph);
  detector-v1.0.1 thay Owns detector-v2.0.0; holdout yếu (0 khung); luật A/B chỉ lượt cả khung (b545b64).
- Trạng thái cũ: `phase-02:6` `Status: READY`; Human steps :79-80 "later"; :38 "Not checked" còn ORT DeformConv.

## 4. Git và CI
- 7 worktree lane XLA: 0 commit chưa gộp, 0 file bẩn. `feat/ai-v2` hơn origin/main 14 commit (PR cuối phase 17).
- CI `feat/ai-v2` 3fadacc đỏ: `tests/verify/test_verifier.py::test_graph_exports_with_dynamic_n_and_parity[efficientnet_b0-96]`
  (`np.ptp(many)` = 0, 4 điểm 0,64185), 1 failed / 470 passed; b545b64 đỏ cùng test; 8b2fcae xanh; giữa hai mốc không đổi
  uv.lock/pyproject/xla/verify/tests/verify/.github. Cục bộ xanh.

## 5. Việc mở và ai xử lý
(c) làm được ngay (giao trong phiên 26/11):
1. CI đỏ test verifier → lane dọn nợ (tìm gốc, sửa).
2. `.gitignore` flights-ai → phase 15 (cache mặc định `logs/.cache/flights-ai`).
3. `results/ab-v2/README.md:3-4` ghi bước 4–5 chưa làm → lane dọn nợ.
4. `enhancer-v2.0.0/meta.json` metrics null → lane dọn nợ (đo trên test, CPU).
5. Forecast / Deferred / trạng thái plan → phase 17 bước 6.
6. Tách `xla/degrade/sources.py` (1 330 dòng) → lane dọn nợ (tách `external.py`, test giữ nguyên).
7. Âm d02x của verifier chưa soát tay → lane dọn nợ (soát mẫu, đo tỉ lệ người lọt vào âm, ghi README).

(d) cần chủ dự án / phần cứng: holdout B2 (bay bãi trống hoặc nới luật 30 s); devpool thiếu (bay); Kill (b) có thể âm tính giả
(cần tập chấm sạch); `score_floor` 0,05 (hợp đồng); verifier effb0 vs mnv3s_64; `kfold extend`; manifest ngoài vào git;
GAN SR thêm giờ GPU; lượt A/B phóng to (không quyết định).

## 6. Tác dụng phụ
`pytest backend/tests` nối dòng vào `logs/deadman.jsonl` (file trong git) — đã khôi phục ở iotcv-debt.
