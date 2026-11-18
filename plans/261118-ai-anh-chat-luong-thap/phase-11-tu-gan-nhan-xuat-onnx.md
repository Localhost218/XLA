# Phase 11: Thí nghiệm E2 — self-training trên khung thật, xuất `detector.onnx`

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 19/11/2026 | phase 02, 10 | L (~2,5 ngày agent), 10–25 giờ GPU (chưa xác minh — smoke đo trước) | Không |

Kết quả (19/11/2026, `b0eec03`, gộp `5a4a05f`): dừng sau vòng 2 (dev-val r2 0,296 < r1 0,379), **ship trò r1** theo dev-val.
Bundle `detector-v1.0.0` (đường ONNX): test mAP50 0,120 [0,032–0,197], B1 không đạt; báo nhầm khung bay 0,881, B2 không đạt;
ORT CUDA EP p50 19,985 / p95 20,223 ms, **B3 đạt**; parity Δ 0,0000 (CPU) / −0,0001 (CUDA) so `.pt rect=False`. Giờ GPU 2,49.
Chi tiết: `results/e2/README.md`, `results/e2/e2.json`, `models/release/detector-v1.0.0/meta.json`.

Lane: detect
Depends-on: 02, 10
Effort: L
Creates: `xla/detect/selftrain.py`, `xla/detect/export_onnx.py`, `configs/selftrain_e2.yaml`, `results/e2/e2.json`, `results/e2/README.md`, `results/e2/parity.json`, `tests/detect/test_detector_export.py`
AX: `python -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round K --smoke --json`; `python -m xla.detect.export_onnx --weights <pt> --out models/release/detector-v1.0.0 --json` dựng bundle, chạy kiểm hợp đồng, exit 1 khi lệch
Accept: `pytest tests/detect/test_detector_export.py`

## Mục tiêu

Dùng khung bay dev **không nhãn** để thích nghi detector với camera thật (cách thường tăng điểm mạnh nhất trên miền thật),
rồi xuất model tốt nhất thành bundle `detector-v1.0.0` đúng hợp đồng §2, §4 cho IOT-CV.

## Đầu vào cần có

- `results/e1/e1.json` + checkpoint tốt nhất E1 (phase 10); D0/D1/D2; khung dev (mọi khung liên tiếp) + `split = "neg"`.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §4 (nhãn giả + tracker kiểu RoyChowdhury, SimROD rút
  gọn, 2–3 vòng, chọn checkpoint bằng val thật; không Mean-Teacher nặng; rủi ro thiên lệch xác nhận).
- Hợp đồng `docs/hop-dong-model.md` §2, §4; `xla/contract.py` (phase 02).
- `d:\Coding\IOT-CV\plans\ai\ai-phase-02-huan-luyen-thi-nghiem.md` A2.4 (opset 17, kiểm tương đương độ chính xác sau xuất).
- `plan.md` D1, D5, D9, D10; Deferred 6.

## Việc theo thứ tự

1. **Thầy:** YOLO11m (l nếu VRAM cho phép) imgsz 1280, fine-tune trên tập nguồn thắng ở E1, chọn bằng dev-val. Smoke 1 epoch
   trước, ghi giờ.
2. **Nhãn giả** (`selftrain.py`): thầy + TTA chạy `track` (botsort hoặc bytetrack, `persist=True`) trên **từng chuỗi khung
   dev liên tiếp**; giữ hộp conf ≥ τ_cao thuộc track dài ≥ L khung; cứu hộp conf ≥ τ_thấp nằm trong track có hộp cao; bỏ
   phát hiện lẻ; khung `neg` = nhãn rỗng. τ_cao chọn trên dev-val để precision ≥ 0,9; L, τ_thấp trong config.
3. **Trò:** YOLO11s imgsz 1280, dữ liệu = tập nguồn thắng E1 + khung dev nhãn giả, trộn bằng mosaic nguồn/đích (SimROD rút
   gọn); chọn checkpoint bằng dev-val. Smoke 1 epoch trước. Lặp 2–3 vòng (trò vòng k làm thầy vòng k+1) chỉ khi mAP50
   dev-val vòng sau tăng; dừng khi không tăng.
4. **Chấm** trên test + neg bằng `xla.eval.detection` (ngưỡng từ dev-val); `results/e2/e2.json` + `README.md`: so mốc, E1
   tốt nhất, E2 từng vòng, kèm CI và giờ GPU thật; so với tiêu chí `results/baseline/tieu-chi-sau-m2.md`.
5. **Xuất ONNX** (`export_onnx.py`, D1): `ultralytics export` opset 17 → bọc graph theo §2 (nhận `frame` uint8 BGR
   [480,640,3]; cắt `crop_left`, letterbox, chuẩn hoá, NMS ONNX, quy toạ độ về khung gốc; ra `boxes`, `scores`); `onnxsim`.
   **Tương đương:** chấm bản ONNX (ORT CPU và ORT CUDA) và bản `.pt` trên dev-val: |ΔmAP50| ≤ 0,5 điểm, ghi `parity.json`;
   lệch hơn → tìm nguyên nhân (thường là letterbox/nội suy), không lặng lẽ dùng tiếp.
6. **Bundle** `models/release/detector-v1.0.0/`: `detector.onnx`, `meta.json` (metrics test, `conf_threshold` từ dev-val,
   `ms_per_frame` p50/p95 đo ORT CUDA EP trên RTX 4060, warmup 30 + đo 300 khung), `golden/` 3 khung **dev** + `expected.json`
   từ ORT CPU. Chạy `python -m xla.contract validate` → `[]`. `meta.json` + `golden/` được commit; riêng `detector.onnx`
   không vào git (Deferred 21): phase 14 chép nó sang nhánh tích hợp, phase 16 chép sang IOT-CV, phase 18 đưa lên GitHub Release.
7. **Test** `tests/detect/test_detector_export.py` (marker `data` + `gpu` cho phần đo): `validate_bundle` rỗng lỗi;
   `parity.json` |ΔmAP50| ≤ 0,5; `meta.metrics` có đủ khoá và khớp `e2.json`; `golden` không chứa khung test.
8. Commit (code, config, JSON, `meta.json` + `golden/` của bundle, không `.onnx`) trên `lane/detect`, đẩy, báo lead.

## Cổng pass

- Accept exit 0; bundle qua `xla.contract validate`.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Thiên lệch xác nhận: người nhỏ bị sót được củng cố | medium | high | Cứu hộp trong track; chọn checkpoint bằng dev-val thật; dừng khi không tăng |
| NMS/letterbox trong graph lệch ultralytics | medium | high | Kiểm tương đương bước 5 trước khi ra bundle |
| Self-training không hơn E1 | medium | medium | Vẫn xuất model tốt nhất (E1 hoặc E2) — bundle ghi `train_run` |

## Rollback

Xoá `models/release/detector-*`; xuất lại từ checkpoint E1.
