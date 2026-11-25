# Phase 13 — IOT-CV backend: job offline + cache + API, hộp sau verifier ở Lịch sử bay (lane iotcv)

Lane: iotcv
Depends-on: 01, 08
Effort: L
Repo: IOT-CV, worktree `D:\IOTCV-wt\iotcv` (tiếp sau phase 08, cùng nhánh `lane/iotcv`)
Owns: `backend/flights/offline.py`, `backend/flights/ai.py`, `backend/api.py`, `backend/app.py`, `backend/ws.py` (chỉ nếu status cần), `.gitignore` (một dòng `logs/flights-ai/`), `backend/tests/test_flights_offline.py`
Creates: `backend/flights/offline.py`, `backend/tests/test_flights_offline.py`
AX: AX-07 job có trạng thái đọc được qua `GET /api/flights/enhance-offline` (JSON §D), mã HTTP xác định
Accept: `pytest backend/tests/test_flights_offline.py`

## Mục tiêu

Phía backend của nhánh A theo `contract.md` §D: tính trước ảnh offline cho cả chuyến, cache trên đĩa, phục vụ `variant=offline`,
an toàn với chuyến bay (không chạy khi armed hoặc vừa mất liên kết, nhả VRAM khi dừng). Trang Lịch sử bay trả hộp sau verifier
(cùng điểm vận hành luồng trực tiếp, §B). Phần web ở phase 18. Chạy được với bundle giả phase 01.

## Đầu vào

`contract.md` §B (hộp Lịch sử bay), §C, §D; `backend/flights/ai.py` (`FlightFrameAIService` :82-95, `_enhance_now` :149-172,
detector-only :195-198, `_ENHANCE_POOL` :30), `backend/api.py` (:266-327), `backend/config.py` + `backend/schemas.py` (env +
schema đã có từ phase 08, không sửa — D10), `backend/telemetry.py` (`STATE.armed` :199, trạng thái liên kết), XLA
`xla/gpu_lock.py` (`read_holder` :66 — đọc file khoá `D:\DevCache\locks\gpu.lock`, không import gói `xla`).

## Bước

1. `git status --short` + hỏi phiên đang sống (D10).
2. `offline.py`: `OfflineEnhanceJob` một job một lúc (thread nền). Trước **mỗi** khung kiểm: `STATE.armed`, liên kết FC mất
   chưa quá `FLIGHT_AI_LINK_GRACE_S`, file khoá GPU XLA đang có người giữ → `paused` + `reason`, **đóng** session ORT; hết điều
   kiện → mở lại, chạy tiếp từ `done`. Ghi cache + `manifest.json` (§D, có `wall_s`); sha bundle khác → làm lại; bản video
   (`temporal_window > 1`) đọc cửa sổ trượt. Chỗ trống < `FLIGHT_AI_MIN_FREE_MB` → 507.
3. `api.py`: `POST`/`GET /api/flights/enhance-offline`, `variant=offline` cho `/api/flights/frame` (404 `not_cached`), mã lỗi
   §D; mở session lỗi → 503 kèm `reason`. `app.py` dựng job khi khởi động (bundle offline đã ở `ready` từ phase 08), không tự chạy.
4. `ai.py`: `/api/flights/ai` chạy detector → verifier (nếu `loaded`) với cùng ngưỡng luồng trực tiếp; trường trạng thái verifier
   trong `FlightFrameAI` nếu schema phase 08 đã có, không thì dừng, báo (D10).
5. `.gitignore`: thêm `logs/flights-ai/` (Owner cho toàn quyền IOT-CV 25/11; Deferred 14).
6. Test `test_flights_offline.py` (bundle giả ×2, `STATE` giả): chạy hết; tạm dừng khi armed và khi liên kết vừa mất, session
   đóng lúc dừng; chạy tiếp từ `done`; 409 busy/armed/link_recent; 503 khi thiếu bundle hoặc session lỗi; 507 khi đĩa đầy (giả);
   404 `not_cached`; sha đổi → làm lại; `/api/flights/ai` trả hộp sau verifier. Chạy toàn bộ `backend/tests`, `ruff check backend`.
7. Đẩy `lane/iotcv`, báo Owner.

## Tiêu chí xong

Accept xanh + toàn bộ `backend/tests` xanh.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Job chạy lúc đang bay mà backend mất liên kết FC | medium | high | Chặn khi liên kết mất < 300 s; kiểm trước mỗi khung; một job; không tự chạy khi khởi động |
| Phát hiện armed trễ vài giây (HEARTBEAT 1 Hz, mất gói UDP) | medium | medium | Kiểm mỗi khung + đóng session khi dừng; job không điều khiển bay, chỉ chiếm GPU |
| Đĩa đầy (≈ 7 000 khung × ~200 KB) | medium | medium | Kiểm chỗ trống trước và trong job → 507; xoá cache theo chuyến |

Rollback: revert nhánh; bỏ env `ENHANCER_OFFLINE_ONNX` → API offline 503.
