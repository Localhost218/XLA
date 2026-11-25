# Phase 08 — IOT-CV: nạp verifier, lọc track k/n, sự kiện, `replay_pred.py` (lane iotcv)

Lane: iotcv
Depends-on: 01
Effort: L
Repo: IOT-CV, worktree `D:\IOTCV-wt\iotcv`, nhánh `lane/iotcv` từ `feat/ai-v2` IOT-CV (phase 01 đã tạo)
Owns: `backend/vision/model_bundle.py`, `backend/vision/verifier.py`, `backend/vision/track_filter.py`, `backend/vision/events.py`, `backend/vision/pipeline.py`, `backend/vision/runtime.py`, `backend/config.py`, `backend/schemas.py`, `scripts/vision/replay_pred.py`, `scripts/vision/replay_bench.py` (chỉ tách vòng phát lại dùng chung), `scripts/vision/lay_model.py`, `scripts/vision/kiem_bundle.py`, `backend/tests/test_vision_ai.py` (chỉ test throttle :227-235), `backend/tests/test_vision_v2.py`
Creates: `backend/vision/verifier.py`, `backend/vision/track_filter.py`, `scripts/vision/replay_pred.py`, `backend/tests/test_vision_v2.py`
AX: AX-01 `uv run --no-sync python scripts/vision/replay_pred.py ... --json` theo `contract.md` §E, mã thoát 0/1/2
Accept: `pytest backend/tests/test_vision_v2.py`

## Mục tiêu

Runtime cho tầng 2 + 3 theo `contract.md` §A, §B, §E, chạy được với bundle giả của phase 01 (không cần GPU): verifier sau
detector, lọc track k/n (cài đặt duy nhất) trước khi phát `vision.person_detected`, công cụ phát lại xuất CSV cả hệ.

## Đầu vào

`contract.md` (§A, §B, §C tên env, §D phần schema + env, §E), `backend/vision/model_bundle.py` (`load_bundle` :175-260,
`IO_SPEC` :38-50, env key :202, chạy thử :247), `backend/vision/runtime.py` (`_try` :53-66, sự kiện :88-94, `status` :96-105),
`backend/vision/events.py` (`EventThrottle` :33, `should_emit` :62-72), `backend/vision/pipeline.py` (:34-35, :80-82,
:199-202, :245-260), `backend/vision/detector.py` (:43-45, :67-76), `backend/config.py` (:177-207), `backend/schemas.py`
(:200-218), `scripts/vision/{replay_bench,lay_model,kiem_bundle}.py`, `backend/tests/test_vision_ai.py` (:227-235).

## Bước

1. `git status --short` + hỏi phiên đang sống trước khi sửa file dùng chung (D10).
2. `model_bundle.py`: `IO_SPEC` cho `verifier` (vào `frame` + `boxes`, ra `scores`) và `enhancer-offline` (§C); chạy thử theo
   kind (verifier: `boxes` [[100,100,140,180]]; detector/enhancer như cũ); kind offline: kiểm sha + meta + **mở session CPU
   không chạy** rồi đóng → `ready` (§D); tên env theo §C (`-` → `_`).
3. `verifier.py`: `BoxVerifier.from_config` (`VERIFIER_ONNX`, `VERIFIER_SHA256`, `VERIFIER_CONFIDENCE`), `verify(frame,
   boxes) -> scores`; N = 0 không gọi session.
4. `track_filter.py`: ghép IoU tham lam hộp ↔ track, cửa sổ n khung, xác nhận ≥ k lần trúng, xoá sau `max_gap`; cooldown
   toàn cục khi không có `recommended_track`, theo track khi có (§B). `EventThrottle` thành lớp tương thích gọi `TrackFilter`
   (không giữ bộ đếm vô hướng). Viết lại test throttle `test_vision_ai.py:227-235` cho nhận hộp; thêm test cho ba khác biệt
   (a)–(c) của §B.
5. `pipeline.py` / `runtime.py`: detector → verifier → track; hộp WS = hộp sau verifier; bước ms `verifier`; sự kiện thêm
   `track_id, hits, window, verifier_score`; `status()` thêm `verifier`, `enhancer_offline`, `enhancer_default_on`.
6. `config.py` + `schemas.py`: mọi env §A, §B, §C, §D (`VERIFIER_*`, `DETECTION_TRACK_*`, `ENHANCER_OFFLINE_*`,
   `FLIGHT_AI_CACHE_DIR`, `FLIGHT_AI_JPEG_QUALITY`, `FLIGHT_AI_LINK_GRACE_S`, `FLIGHT_AI_MIN_FREE_MB`, `ENHANCER_DEFAULT_ON`
   mặc định false) và schema §D — phase 13/18 dùng, không sửa lại (D10).
7. `replay_pred.py` theo §E (`--blank`, xoá track khi vào/ra đoạn blank, `--detector-onnx`, `--verifier-onnx`), dùng chung
   vòng đọc bản ghi tách từ `replay_bench.py` (không chép).
8. `lay_model.py`, `kiem_bundle.py`: golden verifier (`boxes.json`) và offline (1280×960).
9. Test `test_vision_v2.py`: bundle giả verifier nạp / bị từ chối đúng lý do; N = 0; track k/n (nhấp nháy không phát, ổn định
   phát một lần, cooldown); không `recommended_track` → cooldown toàn cục; `replay_pred.py` trên bản ghi giả: cột §E, khung
   blank không có dòng, `track_resets` đúng, `--frames` giao `--blank` → mã 2; schema status mới. Chạy cả `backend/tests/`
   + `ruff check backend`.

## Tiêu chí xong

Accept xanh và toàn bộ `backend/tests` xanh; nhánh `lane/iotcv` đẩy lên, báo Owner. `DetectionPayload` không đổi shape.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hành vi sự kiện khác cũ | high | medium | §B ghi rõ "gần đúng" + ba khác biệt có test; cooldown toàn cục khi không có `recommended_track` |
| Phiên khác đang sửa `config.py`/`schemas.py` | medium | high | Bước 1; D10 |
| Track ghép sai khi FPS thấp (1,8–3 khung/s ở xa) | medium | medium | `max_gap` + IoU cấu hình; phase 09 hiệu chỉnh trên bản ghi thật |

Rollback: revert nhánh; env `VERIFIER_ONNX` trống = tắt verifier.
