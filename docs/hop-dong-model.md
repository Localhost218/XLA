# Hợp đồng model XLA ↔ IOT-CV — `contract_version` 1.0

Owner: tác giả plan `plans/261118-ai-anh-chat-luong-thap/`. Lane **không** sửa file này. Cần đổi: dừng, báo Owner;
Owner sửa, tăng `contract_version` (major khi IOT-CV phải sửa code, minor khi chỉ thêm trường tuỳ chọn), báo lại mọi lane.
Phase 02 `git mv` file này thành `docs/hop-dong-model.md` (SSOT từ đó) và để lại một dòng con trỏ ở đây.
Kiểm bằng `tests/contract/test_model_contract.py` (XLA) và `backend/tests/test_vision_ai.py` (IOT-CV).

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
