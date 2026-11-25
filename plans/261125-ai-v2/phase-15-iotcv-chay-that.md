# Phase 15 — IOT-CV chạy bundle thật, một chuyến offline thật, PR IOT-CV (serial)

Lane: serial
Depends-on: 09, 13, 14, 16, 18
Effort: M
Repo: IOT-CV `feat/ai-v2` (gộp `lane/iotcv`, `lane/iotcv-debt`) + XLA (test tích hợp, chạy từ `D:\XLA`)
Owns: `d:/Coding/IOT-CV/backend/config.py` (chỉ giá trị mặc định đường dẫn bundle + `ENHANCER_DEFAULT_ON`), `d:/Coding/IOT-CV/models/vision/**`, `d:/Coding/IOT-CV/AGENTS.md` (hàng bảng nguồn sự thật + bẫy mới), `tests/integration/test_iotcv_v2.py`, `results/v2/iotcv/**`
Creates: `tests/integration/test_iotcv_v2.py`
AX: n/a because phase only runs existing tools and edits defaults
Accept: `pytest tests/integration/test_iotcv_v2.py`

## Mục tiêu

Đưa bundle đã qua cổng vào IOT-CV và chứng minh trên dữ liệu thật: golden, ms (≤ +5 ms), tầng 2 + 3 trên bản ghi thật, một chuyến
offline thật, web 1707×900; "Ảnh rõ" classic **mặc định bật** theo quyết định Owner (brainstorm §3.A1, Deferred 17); PR IOT-CV.

## Đầu vào

`results/ab-v2/decision.json`, `results/ab-v2/README.md`, `results/v2/system/README.md`; `scripts/vision/{lay_model,kiem_bundle,replay_bench}.py`;
`tests/integration/test_iotcv_replay.py` (mẫu; `test_ngan_sach_ms` đỏ có chủ đích, không sửa).

## Bước

1. Gộp `lane/iotcv` và `lane/iotcv-debt` vào `feat/ai-v2` IOT-CV; toàn bộ `backend/tests`, `ruff`, `pnpm build/lint/test/typecheck`.
2. `lay_model.py --from` mỗi bundle ship (theo `decision.json` và Kill (b)–(d)) → `models/vision/`; `kiem_bundle.py` (CUDA rồi
   `--cpu`); đặt mặc định đường dẫn + sha trong `backend/config.py`; `ENHANCER_DEFAULT_ON = true` (classic, quyết định Owner).
3. `replay_bench.py` trên `logs/recordings/20261102-100941`: p95 từng bước + tổng, so bench v1 gần nhất (+5 ms trần); ghi khối
   `machine` kèm số, so sánh nêu rõ điều kiện (không hoãn vì nhiệt).
4. Chạy job offline thật trên chuyến dài nhất (POST rồi poll), ghi phút thực tế / phút bay (`manifest.wall_s`) — chỉ số Forecast,
   không phải điều kiện ship; mở trang Lịch sử bay so gốc ↔ offline.
5. Backend drone ảo (`chay-tat-ca.bat` / SITL) + `pnpm dev`: Chrome headless 1707×900 nền ngày — trang Bay (không cuộn, "Ảnh rõ"
   bật mặc định khi chưa chọn), Vision (bốn model `loaded`/`ready`), Lịch sử bay (offline). Đọc ảnh lỗi trước khi kết luận.
6. `tests/integration/test_iotcv_v2.py` (XLA, marker `data`+`gpu`): sha bundle IOT-CV = meta release XLA; golden CUDA; bundle trong
   `models/vision/` khớp `decision.json` (offline có ↔ `ship_offline`); `ENHANCER_DEFAULT_ON` mặc định true trong `config.py`;
   p95 tổng v2 ≤ p95 v1 + 5 ms; `realtime.errors == 0`; manifest offline của chuyến bước 4 có `done == total`.
7. `AGENTS.md` IOT-CV: cập nhật hàng "Phần AI" (bundle v2, verifier, offline), bẫy mới gặp; memory agent.
8. PR IOT-CV `feat/ai-v2` → `main` (base `staging` nếu remote có), mô tả có bảng ms + ảnh chụp + ghi chú: deploy workflow lọc
   `ml/**` sẽ chạy khi merge (phase 16 xoá `ml/scripts`); merge khi CI xanh.

## Tiêu chí xong

Accept xanh (riêng `test_ngan_sach_ms` của file cũ vẫn đỏ có chủ đích, nêu trong PR); PR IOT-CV merged.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ms vượt +5 ms | medium | medium | Verifier chỉ chạy khi có hộp; báo số; không khoá xung GPU |
| Chuyến offline > 60 phút | medium | low | Ghi số thật vào Forecast; job vẫn dùng được (chạy nền) |
| IOT-CV `main` cũ nạp `detector-v2.0.0` mà không có verifier/track | low | medium | Bundle v2 chỉ vào `models/vision/` cùng PR có verifier + track |
| Phiên khác đang chạy backend thật | medium | medium | `git status`, ListAgents trước khi tắt backend; dùng drone ảo |

Rollback: đổi đường dẫn bundle mặc định về v1; revert PR IOT-CV.
