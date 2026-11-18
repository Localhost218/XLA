# Phase 16: IOT-CV — chạy model thật trên bản ghi 02/11, đo ms, kiểm web 1707×900

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 19/11/2026, ngân sách p95 nhịp thật chưa đạt (GPU hạ xung khi rảnh, thuc-thi-phase-10-14.md §10) | phase 11, 13, 14, 15 | M (~1 ngày agent), ~0,5 giờ GPU (chưa xác minh) | Không (phát lại bản ghi, không bay) |

Lane: serial
Depends-on: 11, 13, 14, 15
Effort: M
Creates: `d:/Coding/IOT-CV/scripts/vision/replay_bench.py`, `tests/integration/test_iotcv_replay.py`
AX: `uv run python scripts/vision/replay_bench.py --recording logs/recordings/20261102-100941 --json` (IOT-CV) phát lại JPEG gốc qua pipeline, in ms p50/p95 từng bước, khung/s, kết quả golden; exit 1 khi vượt ngân sách
Accept: `pytest tests/integration/test_iotcv_replay.py`

## Mục tiêu

Chứng minh hai model thật chạy trong IOT-CV đủ nhanh (tổng ≤ 50 ms/khung, giữ kịp 6–9 khung/s) trên RTX 4060, ra đúng như
XLA (golden trên GPU trong dung sai), và web hiện box + nút ảnh rõ đúng ở cỡ người dùng xem. Không bay, không motor quay:
phát lại bản ghi 02/11.

## Đầu vào cần có

- Bundle `D:\XLA\models\release\detector-v1.0.0`, `enhancer-v1.0.0` (phase 11, 13) và quyết định ship enhancer + D5 (phase 14,
  `results/ab/README.md`, `results/e4/e4.json`).
- Code IOT-CV phase 15 (chưa commit, trong working tree).
- `d:\Coding\IOT-CV\AGENTS.md`: `chay-tat-ca.bat` chỉ cho drone ảo; web kiểm ở 1707×900 nền ngày, trang Bay không cuộn;
  build bằng PowerShell; E2E trang trắng có thể là WinError 10013 (đọc ảnh lỗi trước khi kết luận); backend bật bằng
  `Start-Process` (memory `iot-cv-gh-ngoai-sandbox`).
- Hợp đồng §4 (dung sai golden), §5, §8.

## Việc theo thứ tự

1. Gộp `lane/iotcv` (phần XLA) vào nhánh tích hợp. Phối hợp phiên IOT-CV như D13 trước khi chạy backend (cổng 8000/5173 có
   thể đang bị phiên khác dùng).
2. `uv run python scripts/vision/lay_model.py --from D:\XLA\models\release\detector-v1.0.0` (và enhancer nếu ship); đặt
   `DETECTOR_SHA256`/`ENHANCER_SHA256` qua biến môi trường của lệnh chạy (không sửa `.env`).
3. `kiem_bundle.py` trên cả hai bundle: golden trên **GPU** trong dung sai §4.
4. **`replay_bench.py`:** đọc JPEG gốc từ bản ghi bằng `backend.flights.video.Recording` (không nén lại), đẩy qua pipeline ở
   nhịp thật của bản ghi; warmup 30 khung; đo detector, enhancer, giải mã, nén luồng enhanced, tổng (p50/p95), khung/s giữ
   được, số khung bị bỏ; ghi `logs/vision-bench/<thời điểm>.json` (thư mục log đã gitignore). Ngân sách: tổng p95 ≤ 50 ms,
   detector ≤ 30 ms, enhancer ≤ 30 ms.
5. **Kiểm web** (backend `CAMERA_FAKE=1`, `CAMERA_FAKE_CLIP` = mp4 dựng từ bản ghi bằng ffmpeg vào `D:\DevCache\tmp\`; fake stream
   nén bằng DQT thật nhờ phase 15): Playwright MCP mở `http://localhost:5173` ở 1707×900 nền ngày; chụp: luồng raw có box,
   bật "Ảnh rõ" → luồng enhanced, tắt → raw; trang Bay không cuộn; ảnh lưu `D:\DevCache\tmp\vision-web\`. Box giả bị thay
   bằng box thật (đếm `canvas.dataset.boxes`).
6. **Test XLA** `tests/integration/test_iotcv_replay.py` (marker `data` + `gpu`): đọc file bench mới nhất của IOT-CV (gốc từ
   `configs/paths.yaml`): ngân sách bước 4 đạt; golden đạt; khung/s giữ ≥ 6; sha256 bundle trong bench = `meta.onnx_sha256`
   của bundle XLA. Thiếu file bench → fail.
7. Báo lead: danh sách file IOT-CV đã đổi (phase 15 + 16), số đo, ảnh web — đầu vào cho human step "Bảo commit". Không commit IOT-CV.

## Cổng pass

- Accept exit 0; ảnh web đủ ba trạng thái.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tổng vượt 50 ms | medium | high | Detector trước, enhancer chỉ chạy khi có người xem; hạ chất lượng nén luồng enhanced; báo số thật |
| Backend thật của chủ dự án đang chạy (đang bay) | low | high | Không bật backend khi phiên khác báo đang bay (memory "lúc bay ưu tiên vận hành"); hỏi qua SendMessage |
| Golden lệch trên GPU | medium | medium | Dung sai §4; lệch hơn → xem lại provider/fp16, không nới dung sai |

## Rollback

Xoá `models/vision/` (gitignore) và bỏ biến môi trường; code phase 15 vẫn chạy ở chế độ không model (vision rejected, backend bay bình thường).
