# Hợp đồng model XLA ↔ IOT-CV — `contract_version` 1.1

Owner: tác giả plan `plans/261118-ai-anh-chat-luong-thap/` (§1–§8) và plan `plans/261125-ai-v2/` (§9–§14). Lane
**không** sửa file này. Cần đổi: dừng, báo Owner; Owner sửa, tăng `contract_version` (major khi IOT-CV phải sửa code,
minor khi chỉ thêm trường tuỳ chọn), báo lại mọi lane. Phase 02 `git mv` file này thành `docs/hop-dong-model.md` (SSOT từ
đó) và để lại một dòng con trỏ ở đây. Kiểm bằng `tests/contract/test_model_contract.py` (XLA) và
`backend/tests/test_vision_ai.py`, `backend/tests/test_vision_v2.py` (IOT-CV).

Lịch sử đổi:

| Phiên bản | Ngày | Đổi gì |
|---|---|---|
| 1.0 | 18/11/2026 | §1–§8 (plan 261118, phase 02) |
| 1.1 | 25/11/2026 | Thêm §9–§14 (plan 261125-ai-v2, phase 01, chép từ `plans/261125-ai-v2/contract.md` §A–§F): kind `verifier` và `enhancer-offline`, `recommended_track` tuỳ chọn của detector, `enhancer-v2` classic/lai, job offline + API IOT-CV, CSV cả hệ, dữ liệu thật v2. §1–§8 giữ nguyên; bundle 1.0 vẫn hợp lệ (cùng major). Bundle `verifier`/`enhancer-offline` phải ghi `contract_version` ≥ 1.1. Tên thư mục bundle: `<detector\|enhancer\|verifier\|enhancer-offline>-v<MAJOR.MINOR.PATCH>`, bố cục §4 áp cho mọi kind (cộng phần thêm ở §9, §11) |

## 1. Khung camera (đầu vào chung)

| Trường | Giá trị |
|---|---|
| Nguồn | MJPEG của ESP32-S3 + OV3660, khung giải mã bằng `cv2.imdecode(buf, cv2.IMREAD_COLOR)` |
| Cỡ | 640 × 480 (VGA). Khung cỡ khác: model **không** chạy, IOT-CV log cảnh báo một lần mỗi phút + bỏ khung, không tự resize |
| Kiểu | `numpy.ndarray` uint8, shape `[480, 640, 3]`, thứ tự kênh **BGR**, thang 0–255, C-contiguous |
| Dải mép trái | Người gọi **không** cắt. Cắt nằm trong graph, độ rộng `crop_left` lấy từ `meta.json` (v1 = 16 px, đo dải ~13 px) |

## 2. `detector.onnx`

| Mục | Giá trị |
|---|---|
| opset | 17 |
| Input | tên `frame`, uint8, shape `[480, 640, 3]`, BGR, 0–255 (đúng §1) |
| Trong graph (theo thứ tự) | cắt cột `[crop_left, 640)` → BGR→RGB → letterbox tới `imgsz`×`imgsz` (tỉ lệ `r = min(imgsz/480, imgsz/(640−crop_left))`, nội suy bilinear, lề chia đều hai phía, giá trị lề 114) → float32 / 255 → NCHW → mạng → lọc `score ≥ score_floor` → NMS lớp người (`nms_iou`, tối đa `max_det`) → quy toạ độ: `x = (x_lb − pad_x)/r + crop_left`, `y = (y_lb − pad_y)/r` → kẹp vào `[0,640]×[0,480]` |
| Output 1 | tên `boxes`, float32, shape `[N, 4]`, `x1, y1, x2, y2` theo **pixel khung gốc 640×480** (gốc toạ độ góc trên trái), `x1 < x2`, `y1 < y2` |
| Output 2 | tên `scores`, float32, shape `[N]`, 0–1, giảm dần |
| N | 0 … `max_det`; N = 0 là hợp lệ (hai tensor rỗng đúng shape) |
| Ngưỡng vận hành | `conf_threshold` trong `meta.json`; IOT-CV lọc `scores ≥ conf_threshold` (env `DETECTION_CONFIDENCE` ghi đè nếu đặt). Graph chỉ lọc ở `score_floor` (mặc định 0,05) để giữ được đường PR khi chấm |
| Nhãn | một lớp; IOT-CV gắn `label = "person"` vào `DetectionBox` |

Đường nhận diện chạy trên **khung gốc**, không qua enhancer (D5), trừ khi `meta.json` của detector ghi `input_variant: "enhanced"` (chỉ có khi thí nghiệm E4 thắng; khi đó `contract_version` 1.1).

## 3. `enhancer.onnx`

| Mục | Giá trị |
|---|---|
| opset | 17 |
| Input | tên `frame`, như §2 |
| Trong graph | cắt `[crop_left, 640)`, (bù tối góc nếu model học), mạng, đệm cạnh tới bội số cần thiết rồi bỏ đệm, ép về uint8 có làm tròn |
| Output | tên `frame_out`, uint8, shape `[480, 640, 3]`, BGR, 0–255; cột `[0, crop_left)` = lặp lại cột `crop_left` của ảnh ra (để hộp §2 khớp toạ độ trên cả hai luồng) |

## 4. Bundle

```
<model>-v<MAJOR.MINOR.PATCH>/      ví dụ detector-v1.0.0/, enhancer-v1.0.0/
  <model>.onnx                      detector.onnx hoặc enhancer.onnx
  meta.json
  golden/
    000.jpg 001.jpg 002.jpg         3 khung THẬT từ tập dev (không phải test), JPEG gốc từ camera
    expected.json                   kết quả của chính file .onnx chạy ORT CPU trong XLA
```

`meta.json` (mọi khoá bắt buộc trừ khi ghi tuỳ chọn):

| Khoá | Kiểu | Ý nghĩa |
|---|---|---|
| `contract_version` | string | `"1.0"` |
| `model` | `"detector"` \| `"enhancer"` | |
| `version` | string semver | trùng tên thư mục |
| `onnx_sha256` | string hex 64 | sha256 của file `.onnx` |
| `frame_size` | `[640, 480]` | `[width, height]` |
| `crop_left` | int px | |
| `camera_profile_version` | string semver | `profile_version` của `configs/camera_profile.yaml` lúc huấn luyện |
| `imgsz` | int px | chỉ detector |
| `score_floor`, `nms_iou`, `max_det`, `conf_threshold` | float, float, int, float | chỉ detector |
| `input_variant` | `"raw"` \| `"enhanced"` | chỉ detector, mặc định `"raw"` |
| `train_run` | string | id lần huấn luyện trong `results/` của XLA |
| `metrics` | object | chỉ số trên tập test thật (detector: `map50`, `map50_95`, `precision`, `recall`, `fp_per_neg_frame`; enhancer: `pink_frac_drop`, `dark_channel_drop`, `flicker_ratio`) |
| `ms_per_frame` | object | `{"device": "RTX 4060 Laptop", "runtime": "onnxruntime-gpu x.y CUDA EP", "p50": float, "p95": float}` |
| `created` | string ISO 8601 | |

`expected.json`: detector `{"000.jpg": {"boxes": [[x1,y1,x2,y2],...], "scores": [...]}, ...}` (sau `score_floor`, trước `conf_threshold`);
enhancer `{"000.jpg": {"sha256_out": "<hex>", "mean": [b,g,r]}, ...}` cộng file `000.out.png` (PNG, không mất mát).
Dung sai khi IOT-CV kiểm golden trên GPU: detector — cùng số hộp có `score ≥ conf_threshold`, mỗi hộp IoU ≥ 0,95 với hộp
tương ứng, |Δscore| ≤ 0,02; enhancer — sai khác tuyệt đối trung bình ≤ 1,0 và lớn nhất ≤ 8 (thang 0–255).

## 5. IOT-CV nạp bundle

| Mục | Giá trị |
|---|---|
| Nơi đặt | `d:\Coding\IOT-CV\models\vision\<bundle>\` (file `.onnx` bị dòng `*.onnx` có sẵn trong `.gitignore` của IOT-CV chặn; không sửa `.gitignore`), chép bằng `scripts/vision/lay_model.py --from <đường dẫn bundle XLA>`; không tải qua mạng, không publish |
| Config (`backend/config.py`, env cùng tên) | `DETECTOR_ONNX` (mặc định `models/vision/detector-v1.0.0/detector.onnx`, tương đối `PROJECT_ROOT`), `DETECTOR_SHA256` (mặc định rỗng), `ENHANCER_ONNX` (mặc định `models/vision/enhancer-v1.0.0/enhancer.onnx`), `ENHANCER_SHA256` (mặc định rỗng), `VISION_ORT_PROVIDERS` (mặc định `CUDAExecutionProvider,CPUExecutionProvider`) |
| Kiểm lúc nạp | (1) file tồn tại; (2) sha256 file = env `*_SHA256` = `meta.onnx_sha256`; (3) major của `meta.contract_version` = major mà backend hỗ trợ; (4) tên/shape/dtype input-output khớp §2/§3 |
| Kiểm không đạt | Model đó **không nạp**; log ERROR một lần nêu lý do; phát sự kiện `vision.model_rejected` `{model, reason}`; phần còn lại của backend (MAVLink, WebSocket, luồng `raw`) chạy bình thường (SAFETY.md mục 9). Không có đường lùi im lặng |
| Provider | Nếu chỉ còn `CPUExecutionProvider` sau khi nạp: log WARNING to + trạng thái `vision.cpu_only` trong `/api/status` |
| Gỡ cũ | `YOLO_WEIGHTS`, `YOLO_DEVICE`, `DETECTION_IMGSZ` bỏ khỏi config (backend không còn ultralytics); `DETECTION_CONFIDENCE` thành tuỳ chọn ghi đè `conf_threshold` |

## 6. `configs/camera_profile.yaml` (XLA)

```yaml
schema_version: "1.0"          # đổi khi đổi tên/kiểu trường
profile_version: "1.0.0"        # đổi khi số đo đổi (camera, firmware, cài đặt, dữ liệu đo)
profile_id: ov3660-esp32s3-q12-20261102
measured_on:                    # chỉ khung dev (data/real/split.json)
  recording: "20261102-100941"
  split_file: data/real/split.json
  n_frames: <int>
  measured: <YYYY-MM-DD>
sensor: {model: OV3660, board: "ESP32-S3 N16R8", xclk_mhz: 10, mount: nadir}
frame: {width_px: 640, height_px: 480, format: MJPEG}
jpeg:
  esp32_quality: 12
  baseline: true
  subsampling: "4:2:2"          # Y 2x1, Cb/Cr 1x1
  qtable_luma: [64 int]         # thứ tự tự nhiên (hàng-cột), KHÔNG zigzag
  qtable_chroma: [64 int]
  annex_k_scale: 0.75           # q12; ijg_equivalent ≈ 62.5
  other_qualities:              # auto-quality 19..40
    rule: "annex_k * q / 16"
    verified: false
left_strip: {width_px: <p50>, crop_left_px: 16, mean_y_strip: <p50>, mean_y_ref: <p50>}
# Mọi đại lượng phân bố ghi {p5, p50, p95}; đơn vị ghi trong tên trường.
vignette: {corner_center_ratio: {p5, p50, p95}, model: cos4, focal_px_fit: <float>}
haze: {dark_channel_15px_0_255: {...}, y_p1_0_255: {...}, saturation_0_255: {...}}
sharpness: {laplacian_var: {...}, mtf50_cyc_per_px: {...}, psf_sigma_px_fit: <float>}
noise: {sigma_immerkaer_0_255: {...}, nlf: [[y_level_0_255, sigma_0_255], ...], psd_radial: [[f_cyc_per_px, power], ...]}
color: {r_over_g: {...}, b_over_g: {...}, wb_gain_r_fit: <float>, wb_gain_b_fit: <float>}
magenta: {rule: "R-G>25 and B-G>25", frac_pixels: {...}, frac_frames_gt_0p5pct: <float>, highlight_y_0_255: {...}}
blocking: {ratio_8x8: {...}}
motion_blur: {sobel_x2_over_y2: {...}, frac_frames_gt_1p5: <float>}
person_box_px: {short_side: {p5, p50, p95, min, max}, source: "nhãn dev"}
altitude_m: {flight_frames: {p5, p50, p95}, filter: "rel_alt_m > 3, bỏ điểm vọt"}
```

`camera_profile.yaml` là **số đo của camera**; tham số chuỗi làm xấu D2 nằm riêng ở `configs/degrade_d2.yaml` (phase 05/07),
mỗi tham số trỏ tới trường hồ sơ nó khớp.

Nghĩa của `color` (làm rõ 18/11/2026, không đổi tên/kiểu trường nên `contract_version` giữ 1.0): `r_over_g`, `b_over_g`
là tỉ số trung bình kênh của cả khung (ám màu toàn ảnh); `wb_gain_r_fit`, `wb_gain_b_fit` là **cận dưới** hệ số cân bằng
trắng của ISP tương đối kênh G, ước từ điểm hồng ở vùng sáng có R (B) ≥ 250: trung vị `R_lin/G_lin` (`B_lin/G_lin`) trong
sRGB tuyến tính; hệ số thật ≥ số này vì R/B đã bị cắt ở 255 (`xla.camera.measure.clipped_highlight_wb_ratios`).

## 7. Định dạng chấm điểm

- Nhãn: một `.txt` YOLO mỗi khung, `0 cx cy w h` chuẩn hoá theo **640×480 khung gốc**; khung không người = file rỗng.
- Dự đoán: CSV `frame,x1,y1,x2,y2,conf` (px khung gốc), một file một model, không lọc ngưỡng dưới `score_floor`.
- Manifest tập: `data/real/split.json` `{"seed": int, "block_s": 30, "frames": [{"frame": "<rec>_<segment>_<video_frame>.jpg", "t_unix": float, "alt_m": float, "block": int, "split": "dev"|"test"|"neg"}]}`.

## 8. Web (IOT-CV)

- `GET /api/video/stream?variant=raw|enhanced` (thiếu tham số = `raw`, hành vi cũ). `enhanced` khi enhancer chưa nạp →
  HTTP 503 `{"error": "enhancer_unavailable", "reason": "<lý do §5>"}`.
- Message WebSocket `detection` giữ nguyên `DetectionPayload` (`backend/schemas.py:239`): `frame_id`, `frame_ts`, `width`
  = 640, `height` = 480, `boxes[]` với `x1,y1,x2,y2,confidence,label`. Không thêm trường.
- Enhancer chỉ chạy khi có ít nhất một client xem `variant=enhanced`.

## 9. `verifier.onnx` (kind `verifier`, từ 1.1)

| Mục | Giá trị |
|---|---|
| Thư mục / file | `verifier-v<semver>/verifier.onnx`; `meta.model: "verifier"`; bố cục §4 |
| opset | 17 |
| Input 1 | `frame`, uint8 `[480, 640, 3]` BGR 0–255 (đúng §1) |
| Input 2 | `boxes`, float32 `[N, 4]` xyxy px khung gốc 640×480, N động, N = 0 hợp lệ (ra `scores` shape `[0]`) |
| Trong graph | với mỗi hộp: tâm giữ nguyên, cạnh = `max(w, h) × context_scale`, tối thiểu `min_side_px` → kẹp vào `[crop_left, 640) × [0, 480)` → `RoiAlign` (`coordinate_transformation_mode = "output_half_pixel"`, tương ứng `torchvision.ops.roi_align(..., aligned=False)`; `sampling_ratio` 2) về `crop_px × crop_px` → BGR→RGB → /255 → chuẩn hoá ImageNet → backbone → sigmoid. Parity mọi nơi dùng đúng chế độ này |
| Output | `scores`, float32 `[N]`, 0–1 = P(người), cùng thứ tự `boxes` |
| meta.json thêm | `model: "verifier"`, `crop_px` (64 hoặc 96), `context_scale`, `min_side_px`, `crop_left`, `conf_threshold`, `input_variant: "raw"`, `backbone`, `train_run`, `metrics` {`fp_per_cal_frame_before`, `fp_per_cal_frame_after`, `recall_oof_before`, `recall_oof_after`} |
| golden | 3 khung + `golden/boxes.json` (hộp vào) + `expected.json` (điểm ra); dung sai `|Δscore| ≤ 0,02` |
| IOT-CV dùng | hộp detector (`≥ conf_threshold` detector) → verifier → giữ hộp có `score ≥ verifier.conf_threshold` (env `VERIFIER_CONFIDENCE` ghi đè). Chạy thử lúc nạp: `{frame: zeros, boxes: [[100,100,140,180]]}`. Verifier thiếu / bị từ chối → chạy chỉ detector, trạng thái `rejected`/`disabled`, sự kiện `vision.model_rejected` như §5 |

Chi tiết hoá của `xla.contract` (không đổi nghĩa): `golden/boxes.json` = `{"000.jpg": [[x1, y1, x2, y2], ...], ...}`
(danh sách rỗng = N = 0), `expected.json` = `{"000.jpg": {"scores": [...]}, ...}` cùng thứ tự hộp.

## 10. Lọc theo thời gian + sự kiện (từ 1.1)

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

Kiểm của `xla.contract`: k, n nguyên dương, k ≤ n; `iou` trong (0, 1]; `max_gap` nguyên ≥ 0.

## 11. `enhancer-offline.onnx` (kind `enhancer-offline`, từ 1.1)

| Mục | Giá trị |
|---|---|
| Thư mục / file | `enhancer-offline-v<semver>/enhancer-offline.onnx`; `meta.model: "enhancer-offline"`; env IOT-CV `ENHANCER_OFFLINE_ONNX`, `ENHANCER_OFFLINE_SHA256` (tên env = `kind.replace("-", "_").upper()`, áp cho `lay_model.py` và `model_bundle.py`) |
| Ảnh đơn (`temporal_window: 1`) | Input `frame` uint8 `[480, 640, 3]` BGR; output `frame_out` uint8 `[960, 1280, 3]` BGR |
| Video (`temporal_window: T`, T lẻ) | Input `frames` uint8 `[T, 480, 640, 3]` BGR (khung giữa là khung đích; đầu/cuối chuyến lặp khung biên); output `frame_out` uint8 `[960, 1280, 3]` của khung giữa |
| Dải mép | cột `[0, 2·crop_left)` của output lấp bằng lặp cột `2·crop_left` (như §3, nhân 2) |
| meta.json thêm | bắt buộc: `scale: 2`, `output_size: [1280, 960]`, `temporal_window`, `network`, `fp16`, `peak_vram_mb`, `s_per_frame` (RTX 4060 Laptop, kèm `machine`). Tuỳ chọn (phase 14 ghi): `metrics` {`fp_cal_raw`, `fp_cal_offline`, `fp_holdout_raw`, `fp_holdout_offline`} |
| golden | 3 khung + `*.out.png` (cỡ 1280×960); dung sai MAE ≤ 1,0, max ≤ 8 |
| Không chạy trực tiếp | IOT-CV chỉ dùng trong job offline §13; không có ngân sách ms |

Chi tiết hoá của `xla.contract` (không đổi nghĩa): `metrics` và `ms_per_frame` của §4 là tuỳ chọn với kind này; `machine`
là string hoặc object (như khối `machine` của file bench); golden với T > 1 coi 3 khung golden là một chuỗi liên tiếp và
cửa sổ của khung i là `clip(i − T//2 … i + T//2)` (lặp khung biên).

## 12. `enhancer-v2`: classic / lai ONNX (từ 1.1)

`enhancer-v2.0.0` (classic ONNX) dùng **đúng** hợp đồng §3 hiện có, `meta.network: "classic"`, thêm `classic_params_sha256`
(sha256 của `configs/enhance_classic.yaml` lúc xuất). Bản lai (nếu có) là `enhancer-v2.1.0`, `meta.network: "hybrid"`.

Tên thư mục vẫn đúng dạng §4 (`enhancer-v2.1.0/`, không phải `enhancer-v2.1.0-hybrid/`). `meta.network` của enhancer là
tuỳ chọn (bundle 1.0 ghi object mô tả mạng); khi là `"classic"` thì `classic_params_sha256` (64 hex thường) bắt buộc.

## 13. IOT-CV: job offline, API, trạng thái (từ 1.1)

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

## 14. CSV cả hệ từ IOT-CV + dữ liệu thật v2 (từ 1.1)

### 14.1 CSV cả hệ (XLA chấm)

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

### 14.2 Dữ liệu thật v2 (XLA)

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
