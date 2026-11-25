# Hợp đồng v1.1 (phần thêm) — XLA ↔ IOT-CV, plan `261125-ai-v2`

Owner: tác giả plan. Lane **không** sửa file này. Cần đổi: dừng, báo Owner; Owner sửa, tăng `contract_version`, báo lại
mọi lane. Phase 01 chép nội dung dưới đây vào `docs/hop-dong-model.md` (mục mới §9–§14), đặt `contract_version` "1.1"
và để lại một dòng con trỏ ở đây. Kiểm bằng `tests/contract/test_model_contract.py` (XLA) và
`backend/tests/test_vision_v2.py` (IOT-CV). Phần §1–§8 của `docs/hop-dong-model.md` giữ nguyên.

## §A. `verifier.onnx` (kind `verifier`)

| Mục | Giá trị |
|---|---|
| opset | 17 |
| Input 1 | `frame`, uint8 `[480, 640, 3]` BGR 0–255 (đúng §1) |
| Input 2 | `boxes`, float32 `[N, 4]` xyxy px khung gốc 640×480, N động, N = 0 hợp lệ (ra `scores` shape `[0]`) |
| Trong graph | với mỗi hộp: tâm giữ nguyên, cạnh = `max(w, h) × context_scale`, tối thiểu `min_side_px` → kẹp vào `[crop_left, 640) × [0, 480)` → `RoiAlign` (`coordinate_transformation_mode = "output_half_pixel"`, tương ứng `torchvision.ops.roi_align(..., aligned=False)`; `sampling_ratio` 2) về `crop_px × crop_px` → BGR→RGB → /255 → chuẩn hoá ImageNet → backbone → sigmoid. Parity mọi nơi dùng đúng chế độ này |
| Output | `scores`, float32 `[N]`, 0–1 = P(người), cùng thứ tự `boxes` |
| meta.json thêm | `model: "verifier"`, `crop_px` (64 hoặc 96), `context_scale`, `min_side_px`, `crop_left`, `conf_threshold`, `input_variant: "raw"`, `backbone`, `train_run`, `metrics` {`fp_per_cal_frame_before`, `fp_per_cal_frame_after`, `recall_oof_before`, `recall_oof_after`} |
| golden | 3 khung + `golden/boxes.json` (hộp vào) + `expected.json` (điểm ra); dung sai `|Δscore| ≤ 0,02` |
| IOT-CV dùng | hộp detector (`≥ conf_threshold` detector) → verifier → giữ hộp có `score ≥ verifier.conf_threshold` (env `VERIFIER_CONFIDENCE` ghi đè). Chạy thử lúc nạp: `{frame: zeros, boxes: [[100,100,140,180]]}`. Verifier thiếu / bị từ chối → chạy chỉ detector, trạng thái `rejected`/`disabled`, sự kiện `vision.model_rejected` như §5 |

## §B. Lọc theo thời gian + sự kiện

- meta detector thêm trường **tuỳ chọn** `recommended_track`: `{"k": int, "n": int, "iou": float, "max_gap": int}` (k ≤ n,
  `max_gap` = số khung một track được mất trước khi xoá).
- IOT-CV `backend/vision/track_filter.py` là cài đặt **duy nhất** của lọc sự kiện (D4): ghép IoU tham lam giữa hộp đã qua
  verifier và track đang sống, cửa sổ n khung, xác nhận khi ≥ k lần trúng. `EventThrottle` trở thành lớp tương thích gọi
  `TrackFilter`, không còn bộ đếm vô hướng riêng. Env `DETECTION_TRACK_K`, `DETECTION_TRACK_N`, `DETECTION_TRACK_IOU`,
  `DETECTION_TRACK_MAX_GAP` ghi đè.
- Không có `recommended_track` và env không đặt → `k = n = DETECTION_MIN_FRAMES`, cooldown **toàn cục** `DETECTION_COOLDOWN_S`
  (như hiện nay). Đây là **gần đúng**, không trùng khít hành vi cũ: khác ở (a) hộp nhảy chỗ IoU < `iou` giữa hai khung,
  (b) hai người cùng lúc, (c) track mất rồi tạo lại; ba trường hợp này có test ghi rõ hành vi mới. Có `recommended_track`
  → cooldown theo track.
- Sự kiện `vision.person_detected`: level, source, code, message **không đổi**; `detail` giữ `confidence, uav_lat,
  uav_lon, uav_alt` (`confidence` giờ là điểm detector cao nhất của track được xác nhận) và thêm `track_id` (int),
  `hits` (int), `window` (int = n), `verifier_score` (float | null).
- Message WS `detection` (`DetectionPayload`) **không đổi** shape; hộp gửi đi là hộp sau verifier. Trang Lịch sử bay
  (`/api/flights/ai`) cũng trả hộp sau verifier (cùng điểm vận hành với luồng trực tiếp).

## §C. `enhancer-offline.onnx` (kind `enhancer-offline`)

| Mục | Giá trị |
|---|---|
| Thư mục / file | `enhancer-offline-v<semver>/enhancer-offline.onnx`; `meta.model: "enhancer-offline"`; env IOT-CV `ENHANCER_OFFLINE_ONNX`, `ENHANCER_OFFLINE_SHA256` (tên env = `kind.replace("-", "_").upper()`, áp cho `lay_model.py` và `model_bundle.py`) |
| Ảnh đơn (`temporal_window: 1`) | Input `frame` uint8 `[480, 640, 3]` BGR; output `frame_out` uint8 `[960, 1280, 3]` BGR |
| Video (`temporal_window: T`, T lẻ) | Input `frames` uint8 `[T, 480, 640, 3]` BGR (khung giữa là khung đích; đầu/cuối chuyến lặp khung biên); output `frame_out` uint8 `[960, 1280, 3]` của khung giữa |
| Dải mép | cột `[0, 2·crop_left)` của output lấp bằng lặp cột `2·crop_left` (như §3, nhân 2) |
| meta.json thêm | bắt buộc: `scale: 2`, `output_size: [1280, 960]`, `temporal_window`, `network`, `fp16`, `peak_vram_mb`, `s_per_frame` (RTX 4060 Laptop, kèm `machine`). Tuỳ chọn (phase 14 ghi): `metrics` {`fp_cal_raw`, `fp_cal_offline`, `fp_holdout_raw`, `fp_holdout_offline`} |
| golden | 3 khung + `*.out.png` (cỡ 1280×960); dung sai MAE ≤ 1,0, max ≤ 8 |
| Không chạy trực tiếp | IOT-CV chỉ dùng trong job offline §D; không có ngân sách ms |

`enhancer-v2.0.0` (classic ONNX) dùng **đúng** hợp đồng §3 hiện có, `meta.network: "classic"`, thêm `classic_params_sha256`
(sha256 của `configs/enhance_classic.yaml` lúc xuất). Bản lai (nếu có) là `enhancer-v2.1.0`, `meta.network: "hybrid"`.

## §D. IOT-CV: job offline, API, trạng thái

- `POST /api/flights/enhance-offline?rec=<rec>` → 202 `{rec, state, done, total}` (`state` "queued" | "running"); 409
  `{error: "armed"}`; 409 `{error: "link_recent"}` (mất liên kết FC chưa quá `FLIGHT_AI_LINK_GRACE_S`, mặc định 300 s —
  có thể đang bay); 409 `{error: "busy", rec}`; 503 `{error: "enhancer_offline_unavailable", reason}` (kể cả khi mở
  session lỗi); 507 `{error: "disk_full", free_mb}` (chỗ trống < `FLIGHT_AI_MIN_FREE_MB`, mặc định 3000).
- `GET /api/flights/enhance-offline?rec=<rec>` → 200 `{rec, state: "none"|"queued"|"running"|"paused"|"done"|"failed",
  done: int, total: int, bundle_version: str|null, s_per_frame: float|null, reason: str|null}`. `paused` = armed, liên kết
  vừa mất, hoặc khoá GPU của XLA đang bị giữ (`reason` ghi lý do). Khi `paused` job **đóng** session ORT (trả VRAM),
  mở lại khi chạy tiếp; điều kiện được kiểm trước **mỗi** khung.
- `GET /api/flights/frame?rec&n&variant=offline` → `image/jpeg` 1280×960 từ cache; 404 `{error: "not_cached"}`.
  `variant` giờ là `raw | enhanced | offline`.
- Cache: `<FLIGHT_AI_CACHE_DIR>/<rec>/enhancer-offline-v<ver>/<n:06d>.jpg` + `manifest.json` `{bundle_version,
  onnx_sha256, total, done, jpeg_quality, started, finished, wall_s}`; sha khác → coi như chưa có. Mặc định
  `FLIGHT_AI_CACHE_DIR=logs/flights-ai` (gitignored), `FLIGHT_AI_JPEG_QUALITY=92`. Job chạy được tiếp từ `done`.
- `VisionModelStatus.state`: `"loaded" | "ready" | "rejected" | "disabled"`. `ready` = sha + meta đúng **và** mở được
  session CPU (không chạy) lúc khởi động, rồi đóng; web tô `ready` màu vàng (chưa chạy thật), `loaded` xanh.
- `VisionStatus` thêm `verifier: VisionModelStatus`, `enhancer_offline: VisionModelStatus`, `enhancer_default_on: bool`
  (env `ENHANCER_DEFAULT_ON`); `ms` thêm bước `"verifier"`.
- Khoá localStorage web: `iot-cv.flight-ai-source` (`"enhanced"` | `"offline"`); `iot-cv.video-enhanced` giữ nguyên
  (chưa có giá trị → lấy `enhancer_default_on`).

## §E. CSV cả hệ từ IOT-CV (XLA chấm)

`uv run --no-sync python scripts/vision/replay_pred.py --recording logs/recordings/<rec> --frames <list.txt>
--stage detector|verifier|track --out <csv> [--blank <list.txt>] [--detector-onnx <path>] [--verifier-onnx <path>]
[--track-k K --track-n N --track-iou F --track-max-gap G] [--verifier-conf F] --json` (gốc IOT-CV; env con **không** có
`UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` của XLA). Phát lại mọi khung theo thứ tự (để track có ngữ cảnh), **trừ** khung trong
`--blank`: khung đó không đưa vào detector, và trạng thái track bị **xoá** khi vào và khi ra một đoạn khung blank (không
cho phát hiện trên khung bị cấm lọt vào cửa sổ k/n của khung khác). Ghi ra chỉ các khung trong `--frames` (không được giao
với `--blank`, giao → mã 2). CSV `frame,x1,y1,x2,y2,conf` (D11; `frame` = `<rec>_<segment:03d>_<video_frame:05d>.jpg`),
`conf` = điểm detector (stage detector) hoặc điểm verifier (stage verifier/track). Mã thoát 0 / 1 (dữ liệu, bundle) / 2
(sai cách gọi). `--json` in `{frames_written, frames_replayed, frames_blank, track_resets, stage, track:{k,n,iou,max_gap},
bundles:{detector,verifier}}`.

## §F. Dữ liệu thật v2 (XLA)

- `data/real/sessions.json`: `{sessions: [{id: "<rec>#<i>", rec, t_start, t_end, n_frames, source: "flights-events"}]}`.
- `data/real/neg_flight_split.json`: `{seed: 20261125, sessions: {<id>: "mine"|"cal"|"holdout"}, frames: [{frame, session,
  split: "mine"|"cal"|"holdout", approved: bool}]}`.
  - `mine`: được làm ảnh nền huấn luyện detector và lớp âm verifier.
  - `cal`: chỉ dùng hiệu chỉnh ngưỡng verifier, k/n, hàng rào báo nhầm của ảnh offline; không huấn luyện.
  - `holdout`: chỉ chấm B2 một lần ở phase 09 và báo cáo; khung chỉ tính khi chủ dự án duyệt (`approved: true`). Không
    phase nào chọn model, ngưỡng hay quyết định ship dựa trên holdout.
- `data/real/labels/devpool/*.txt` (YOLO, 640×480) + `data/real/labels/folds.json` `{groups: "session"|"block5",
  folds: [[frame...], ...]}`. Khung test, cal, holdout không bao giờ ở đây.
- `split.json` v1 không đổi (sha256 `c1b0172840972b3951ffcd2060b482e87670015172f337830f8883815991192c`, 25/11/2026).
- Khung thật vào huấn luyện theo **danh sách cho phép**: chỉ khung `mine` (ảnh nền) và khung `dev` của `split.json` không
  thuộc `devpool` fold đang loại (nhãn giả). Mọi khung khác (test, cal, holdout, chưa gán) → lỗi.
