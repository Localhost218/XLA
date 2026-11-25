# Phase 18 — IOT-CV web: nguồn ảnh offline, ô trạng thái model, mặc định "Ảnh rõ" (lane iotcv)

Lane: iotcv
Depends-on: 13
Effort: M
Repo: IOT-CV, worktree `D:\IOTCV-wt\iotcv` (tiếp sau phase 13, cùng nhánh `lane/iotcv`); lệnh chạy từ `frontend/` (Accept dùng vitest cục bộ của `frontend/`, cùng bộ test với `pnpm test`)
Owns: `frontend/src/components/flights/AiCompare.tsx`, `frontend/src/components/flights/AiViewer.tsx`, `frontend/src/components/VideoPanel/VideoPanel.tsx`, `frontend/src/components/vision/ModelStatus.tsx`, `frontend/src/pages/VisionPage.tsx`, `frontend/src/lib/flights.ts`, `frontend/tests/unit/modelStatus.test.tsx`, `frontend/tests/unit/flightsOffline.test.tsx`
Creates: `frontend/src/components/vision/ModelStatus.tsx`, `tests/unit/modelStatus.test.tsx`, `tests/unit/flightsOffline.test.tsx`
AX: n/a because phase changes web components only
Accept: `bunx vitest run tests/unit/modelStatus.test.tsx tests/unit/flightsOffline.test.tsx`

## Mục tiêu

Trên web (theo `contract.md` §D): Lịch sử bay chọn nguồn "Ảnh rõ" / "Ảnh rõ offline (AI dựng lại)", nút dựng offline cho
chuyến + tiến độ, nút xem gốc luôn có; ô trạng thái model (detector, verifier, enhancer, enhancer-offline: trạng thái, phiên bản,
provider, ms) bằng `KeyValue` của kit ở trang Vision; "Ảnh rõ" trực tiếp mặc định theo `enhancer_default_on` khi người dùng
chưa chọn.

## Đầu vào

`contract.md` §D; `AiCompare.tsx` (`loadCompare` :62-92, khoá :33-34, toggle :331-345), `AiViewer.tsx` (chỉ hiển thị),
`VideoPanel.tsx` (:41-57, :80-96, :199-219), `components/kit/index.tsx` (`KeyValue` :181-243, `KeyValueList` :249), `lib/tone.ts`,
`lib/flights.ts` (`frameUrl` :30-33), `frontend/src/pages/VisionPage.tsx`; API phase 13.

## Bước

1. `git status --short` + hỏi phiên đang sống (D10).
2. `lib/flights.ts`: `frameUrl(..., "offline")`, `startOffline(rec)`, `getOffline(rec)` theo §D.
3. `AiCompare.tsx`: chọn nguồn (`iot-cv.flight-ai-source`), nút "Dựng ảnh rõ offline cho chuyến này" + thanh tiến độ (poll mỗi
   2 s khi `running`; hiện `reason` khi `paused`/lỗi 409/503/507), nhãn "ảnh do AI dựng lại", ảnh offline 1280×960 cạnh gốc
   (`AiViewer` nhận URL); khung chưa cache → ảnh rõ trực tiếp + chip "chưa dựng".
4. `ModelStatus.tsx` trên `VisionPage.tsx`: `KeyValueList` bốn dòng; `tone` từ `lib/tone.ts` (`loaded` xanh, `ready` vàng "chưa
   chạy thật", `rejected` đỏ + `hint` = lý do, `disabled` xám), phiên bản, provider (chip cảnh báo khi CPU-only), ms p50/p95 từ
   `ms` (offline: `s_per_frame`). Không tô màu tại chỗ.
5. `VideoPanel.tsx`: chưa có khoá `iot-cv.video-enhanced` → theo `status.vision.enhancer_default_on`; đã có → giữ lựa chọn.
6. Test vitest: `modelStatus.test.tsx` (bốn trạng thái, lý do rejected, CPU-only), `flightsOffline.test.tsx` (chọn nguồn lưu
   localStorage, tiến độ, 404 `not_cached` → chip, mặc định "Ảnh rõ" theo `enhancer_default_on`). Chạy `pnpm build` (PowerShell),
   `pnpm lint`, `pnpm test`, `pnpm typecheck`.
7. Ảnh chụp Chrome headless 1707×900 nền ngày: Vision (ô trạng thái), Lịch sử bay (nguồn offline), trang Bay không cuộn; lưu ở
   `plans/reports/` của IOT-CV. Đẩy `lane/iotcv`, báo Owner.

## Tiêu chí xong

Accept xanh + mọi lệnh bước 6 xanh; ba ảnh chụp đúng 1707×900, trang Bay không có thanh cuộn cột.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Trang Bay bị cuộn | low | medium | Ô trạng thái ở trang Vision (Deferred 18); ảnh chụp bước 7 |
| Build từ Git Bash làm hỏng URL font | medium | low | Build bằng PowerShell (AGENTS.md) |
| E2E trang trắng do WinError 10013 | medium | low | Đọc ảnh lỗi trước khi kết luận |

Rollback: revert các commit web; backend phase 13 không phụ thuộc web.
