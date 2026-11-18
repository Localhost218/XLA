# Phase 15: IOT-CV — nạp bundle ONNX, pipeline, luồng ảnh rõ, nút web, camera giả đúng DQT

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 19/11/2026 (IOT-CV, nhánh feat/ai-vision-phase-15-17) | phase 02 | L (~2,5 ngày agent), 0 giờ GPU huấn luyện | Không (chỉ laptop) |

Lane: iotcv
Depends-on: 02
Effort: L
Owns: `d:/Coding/IOT-CV/backend/vision/detector.py`, `d:/Coding/IOT-CV/backend/vision/events.py`, `d:/Coding/IOT-CV/backend/vision/fake_stream.py`, `d:/Coding/IOT-CV/backend/config.py`, `d:/Coding/IOT-CV/backend/app.py`, `d:/Coding/IOT-CV/backend/ws.py`, `d:/Coding/IOT-CV/backend/events.py`, `d:/Coding/IOT-CV/.env.example`, `d:/Coding/IOT-CV/pyproject.toml`, `d:/Coding/IOT-CV/uv.lock`, `d:/Coding/IOT-CV/backend/tests/test_fake_stream.py`, `d:/Coding/IOT-CV/frontend/src/components/VideoPanel/VideoPanel.tsx`
Creates: `d:/Coding/IOT-CV/backend/vision/model_bundle.py`, `d:/Coding/IOT-CV/backend/vision/enhancer.py`, `d:/Coding/IOT-CV/backend/vision/pipeline.py`, `d:/Coding/IOT-CV/backend/vision/assets/camera_dqt.json`, `d:/Coding/IOT-CV/scripts/vision/lay_model.py`, `d:/Coding/IOT-CV/scripts/vision/trich_dqt.py`, `d:/Coding/IOT-CV/scripts/vision/kiem_bundle.py`, `d:/Coding/IOT-CV/backend/tests/test_vision_ai.py`, `tests/integration/test_iotcv_vision.py`
AX: `uv run python scripts/vision/kiem_bundle.py <bundle-dir> --json` (IOT-CV) kiểm bundle bằng chính bộ nạp của backend, exit 0/1/2; `lay_model.py --from <bundle> --json` chép bundle + in sha256 cần đặt vào env
Accept: `pytest tests/integration/test_iotcv_vision.py`

## Mục tiêu

IOT-CV chạy được **bất kỳ** bundle đúng hợp đồng (D1–D6): nạp có kiểm sha256 + phiên bản, detector chạy trên khung gốc
gửi box qua message `detection` có sẵn, enhancer phục vụ luồng `variant=enhanced`, web có nút bật/tắt ảnh rõ. Làm với
**bundle giả** (phase 02) nên không chờ model thật. Thêm việc nhỏ: camera giả nén đúng bảng DQT thật (4:2:2) thay vì
OpenCV Q81. **IOT-CV chỉ ghi file, không commit/push/PR** (D13).

## Đầu vào cần có

- Hợp đồng `D:\XLA\docs\hop-dong-model.md` (đặc biệt §5, §8); `python -m xla.contract make-dummy` (phase 02).
- IOT-CV hiện trạng (đọc trước khi sửa): `backend/vision/detector.py` (khung sườn `PersonDetector`), `events.py`
  (`EventThrottle.should_emit` khung sườn), `stream.py` (`MjpegLatestFrameReader.subscribe`, `mjpeg_multipart`),
  `fake_stream.py` (`cv2_quality()` coi q12 là Q81), `backend/schemas.py:227–245` (`DetectionBox`, `DetectionPayload`),
  `backend/ws.py:587` (`detection_loop` gửi box giả), `backend/app.py` (`/api/video/stream`), `backend/config.py:171–226`,
  `frontend/src/components/VideoPanel/` (`DetectionOverlay.tsx` đã vẽ box theo `width/height` của message),
  `frontend/src/lib/detection.ts`, `pyproject.toml` (nhóm optional `vision`, `dev`).
- `d:\Coding\IOT-CV\AGENTS.md` (luật 2: chỉ ghi file; mục web: build bằng PowerShell, màu/chip lấy từ `lib/tone.ts`,
  `components/kit`, kiểm ở 1707×900 nền ngày; nhiều phiên song song); `SAFETY.md` mục 9.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §0 (DQT thật, Pillow `qtables` + 4:2:2).
- `plan.md` D3, D5, D6, D13; Deferred 12.

## Việc theo thứ tự

1. **Phối hợp trước khi sửa** (D13): `git -C d:\Coding\IOT-CV status --short`; file nào trong Owns đang bị sửa dở → nhắn phiên
   đang sống (ListAgents/SendMessage) để chia, không ghi đè. `AGENTS.md` đang có thay đổi chưa commit của phiên khác: không đụng.
2. **Runtime trước** (Deferred 12): thêm `onnxruntime-gpu` + `pillow` vào nhóm `vision`, `onnx` + `onnxruntime` + `pillow` vào
   `dev`; `uv sync --extra vision --extra dev` (cache ở `D:\DevCache\uv`); kiểm `CUDAExecutionProvider` nạp được trên Python
   3.13 với bundle giả, đo ms. Không được: thử `onnxruntime-directml`; vẫn không: CPU + ghi rõ, báo lead (rủi ro Forecast #2).
   **Bẫy đã gặp ở XLA (phase 01, 18/11/2026):** `onnxruntime-gpu` 1.32 đòi CUDA 13 (`cublasLt64_13.dll`), không tạo được
   CUDA EP trên máy này; 1.24.4 tạo được **chỉ khi đã `import torch` trước** (mượn DLL CUDA 12.4 + cuDNN 9 của torch), không
   import thì session lặng lẽ lùi về CPU. Backend IOT-CV không có torch: phải có CUDA 12 + cuDNN 9 riêng (gói pip
   `nvidia-cudnn-cu12`, `nvidia-cublas-cu12` … + `onnxruntime.preload_dlls()` nếu bản ORT có) hoặc rơi sang directml. Kiểm
   `session.get_providers()` sau khi tạo session, đừng tin `get_available_providers()`.
3. **`model_bundle.py`:** `load_bundle(onnx_path, sha256, kind)` thực hiện bốn kiểm của hợp đồng §5; lỗi → ném
   `BundleRejected(reason)`; tạo `InferenceSession` theo `VISION_ORT_PROVIDERS`; trả về đối tượng giữ session + meta.
4. **`detector.py`:** hiện thực `PersonDetector` trên `model_bundle` (bỏ ultralytics, bỏ `YOLO_WEIGHTS`/`DETECTION_IMGSZ`
   khỏi config, giữ `Detection` dataclass); `detect(frame_bgr)` kiểm cỡ 640×480 (khác → trả rỗng + cảnh báo theo nhịp) rồi lọc
   `conf_threshold` (ghi đè bởi `DETECTION_CONFIDENCE` nếu đặt).
5. **`enhancer.py`:** `Enhancer.enhance(frame_bgr) -> frame_bgr` cùng cỡ.
6. **`events.py`:** hiện thực `EventThrottle.should_emit(confidence, now)` đúng docstring (đủ `min_frames` khung liên tiếp
   trên ngưỡng → phát, rồi `cooldown_s`).
7. **`pipeline.py`:** đăng ký `subscribe` vào reader hiện có; một thread xử lý **khung mới nhất** (bỏ khung cũ, không hàng đợi);
   giải mã JPEG; detector mỗi khung → `DetectionPayload` (`frame_id`, `frame_ts` từ khung, `width` 640, `height` 480, `label`
   "person") qua callback; enhancer chỉ chạy khi có client `enhanced` (đếm tham chiếu), kết quả nén JPEG (chất lượng trong
   config) vào một holder khung-mới-nhất cho `mjpeg_multipart`; ghi ms từng bước để `/api/status` trả p50. Lỗi trong thread
   không giết backend (SAFETY 9): log + đếm.
8. **`config.py` / `.env.example`:** khoá §5 (`DETECTOR_ONNX`, `DETECTOR_SHA256`, `ENHANCER_ONNX`, `ENHANCER_SHA256`,
   `VISION_ORT_PROVIDERS`), chất lượng JPEG luồng enhanced; bỏ `YOLO_WEIGHTS`, `YOLO_DEVICE`, `DETECTION_IMGSZ` (tìm mọi chỗ
   dùng bằng grep trước khi bỏ).
9. **`app.py` / `ws.py` / `events.py` (backend):** `/api/video/stream?variant=raw|enhanced` (503 + lý do khi không có
   enhancer); khởi pipeline khi detector nạp được; `detection_loop` gửi box thật khi có pipeline, box giả chỉ khi
   `CAMERA_FAKE` và không có detector; sự kiện `vision.model_rejected` `{model, reason}`; `/api/status` có trạng thái vision
   (`loaded`, `rejected: reason`, `cpu_only`).
10. **Camera giả đúng DQT:** `scripts/vision/trich_dqt.py` đọc một khung thật trong `logs/recordings/20261102-100941` → ghi
    `backend/vision/assets/camera_dqt.json` (`qtable_luma`, `qtable_chroma` thứ tự tự nhiên, `subsampling "4:2:2"`, nguồn,
    sha256 khung nguồn); `fake_stream.py` khi quality = 12 nén bằng Pillow `qtables` + `subsampling="4:2:2"`; quality khác giữ
    hành vi hiện tại nhưng ghi chú "chưa đo"; thêm test vào `test_fake_stream.py`: DQT + SOF của khung giả = khung thật.
11. **Công cụ:** `scripts/vision/lay_model.py --from <bundle XLA>` chép vào `models/vision/<bundle>/`, in sha256 để đặt env (không
    sửa `.env`); `scripts/vision/kiem_bundle.py` chạy bộ nạp backend + golden với dung sai §4. **Không sửa `.gitignore`** của
    IOT-CV (AGENTS.md luật 2): dòng `*.onnx` có sẵn (`.gitignore:51`) đã chặn file model; `meta.json` + `golden/` nhỏ, chủ dự
    án quyết có commit cùng các file khác hay không (Human steps hàng 3).
12. **Web:** `VideoPanel.tsx` thêm nút bật/tắt "Ảnh rõ" (dùng `components/kit` + `lib/tone.ts`, không tô màu tại chỗ), chỉ đổi
    `src` giữa `variant=raw` và `variant=enhanced`, nhớ lựa chọn trong `localStorage`; 503 → nút tắt + chip lý do. Box giữ nguyên
    `DetectionOverlay`. `pnpm build` bằng **PowerShell**.
13. **Test IOT-CV** `backend/tests/test_vision_ai.py` (tự dựng bundle giả bằng `onnx.helper` đúng hợp đồng, không phụ thuộc XLA,
    chạy được trên CI IOT-CV): nạp đúng; sai sha256 / sai major / sai tên input → `BundleRejected` + backend vẫn trả `/api/health`;
    pipeline gửi `DetectionPayload` đúng schema; `variant=enhanced` không enhancer → 503; `should_emit` đủ ca; khung 320×240 → không
    box, không ngoại lệ.
14. **Test XLA** `tests/integration/test_iotcv_vision.py`: gọi subprocess `uv run --project <IOT-CV> --extra dev --extra vision pytest backend/tests/test_vision_ai.py backend/tests/test_fake_stream.py`
    (gốc IOT-CV từ `configs/paths.yaml`) phải exit 0; rồi dựng bundle giả bằng `xla.contract.make_dummy_bundle` và chạy
    `scripts/vision/kiem_bundle.py` của IOT-CV trên nó phải exit 0 — hai bản cài hợp đồng độc lập phải đồng ý. Thiếu IOT-CV → **fail**.
15. Không commit IOT-CV. Ghi danh sách file đã sửa/tạo + kết quả test vào báo cáo cho lead (human step "Bảo commit"). Phần XLA
    (test integration) commit trên `lane/iotcv`.

## Cổng pass

- Accept exit 0; `uv run pytest` toàn bộ IOT-CV (backend) xanh; `pnpm build` (PowerShell) xanh; ruff/oxlint sạch.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ORT GPU không chạy Py 3.13 | medium | high | Bước 2 làm đầu tiên, dự phòng directml/CPU có cảnh báo; hợp đồng không đổi |
| Phiên khác đang sửa `app.py`/`ws.py`/`config.py` | medium | high | Bước 1; chỉ sửa sau khi chia file |
| Pipeline làm chậm luồng raw hoặc telemetry | low | high | Thread riêng, khung-mới-nhất, không hàng đợi; test `/api/health` khi vision lỗi |
| Bỏ `YOLO_WEIGHTS` làm gãy chỗ khác | low | medium | Grep mọi tham chiếu trước khi bỏ (pre-delete check) |

## Rollback

`git -C d:\Coding\IOT-CV checkout -- <file>` cho file sửa, xoá file tạo mới — chỉ sau khi chủ dự án đồng ý (chưa commit nên
không có gì để revert trong lịch sử).
