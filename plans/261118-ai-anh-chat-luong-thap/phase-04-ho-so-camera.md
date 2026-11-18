# Phase 04: Hồ sơ camera `camera_profile.yaml` v1 đo trên khung dev

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | phase 02, 03 | M (~1 ngày agent), 0 giờ GPU | Không |

Lane: data
Depends-on: 02, 03
Effort: M
Creates: `xla/camera/measure.py`, `xla/camera/profile.py`, `configs/camera_profile.yaml`, `results/camera/profile_report.md`, `tests/camera/test_camera_profile.py`
AX: `python -m xla.camera.profile build --split dev --json` đo và ghi hồ sơ; `python -m xla.camera.profile check configs/camera_profile.yaml` exit 1 khi lệch schema
Accept: `pytest tests/camera/test_camera_profile.py`

## Mục tiêu

Biến các số đo rời rạc của camera thành **một hồ sơ có phiên bản** (`profile_version` 1.0.0; → 18/11/2026: 1.1.0 khi đo lại trên dev mới sau chia lại, 1.2.0 khi sửa `wb_gain_*_fit` thành cận dưới hệ số WB ISP, xem `results/camera/profile_report.md` dòng 28 và 70) đúng schema của phase 02 —
nguồn sự thật cho bộ làm xấu D2 (phase 05/07), đối chứng cổ điển (phase 12), camera giả của IOT-CV (phase 15). Đo **chỉ
trên khung dev** (D10). Hồ sơ mô tả camera, không chứa tham số làm xấu.

## Đầu vào cần có

- Hợp đồng `docs/hop-dong-model.md` §6 + `configs/camera_profile.schema.json` (phase 02).
- `data/real/split.json`, `data/real/frames/` (phase 03).
- `plans/reports/261118-bang-chung/video/features.py` (đặc trưng đã viết: Laplacian, Immerkaer, dark channel, % hồng,
  blocking, dải mép, ijg_quality, vignetting), `video/phan-tich-video.md` (giá trị mẫu 300 khung để đối chiếu).
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §0 (q=12 = Annex K × 75 %, 4:2:2, một bộ bảng
  duy nhất; Pillow tái tạo 100 %), `nghien-cuu/jpeg_dqt_scan.py`, `nghien-cuu/annexk_check.py`; §2 (đo nhiễu **sau ISP**
  bằng NLF + PSD, không mô hình RAW).

## Việc theo thứ tự

1. `xla/camera/measure.py`: chuyển các hàm của `features.py` thành hàm thuần có docstring + đơn vị (dark channel 15×15,
   Y p1/p99, độ bão hoà, % hồng theo luật `R−G>25 và B−G>25`, Y vùng hồng, blocking 8×8, Laplacian var, Immerkaer σ, Sobel
   x²/y², tối góc góc/giữa, dải mép); thêm: đọc DQT + lấy mẫu màu từ header (theo `jpeg_dqt_scan.py`), NLF (σ theo mức Y
   trên vùng phẳng), PSD bán kính của phần dư cao tần, MTF50 ước từ cạnh mạnh (ghi rõ là ước lượng từ cảnh tự nhiên).
   Mọi phép đo bỏ cột `[0, crop_left)` trừ phép đo dải mép.
2. `xla/camera/profile.py`: `build(split="dev")` đo mọi khung dev, gom p5/p50/p95; kiểm **mọi** khung dev có cùng một bộ
   DQT + 4:2:2 (khác → exit 1, báo khung khác); ghi `configs/camera_profile.yaml` (schema 1.0, profile 1.0.0,
   `profile_id ov3660-esp32s3-q12-20261102`), `jpeg.other_qualities` theo Deferred 1 với `verified: false`;
   `person_box_px` đo từ hộp đếm sơ bộ của phase 03 (YOLO11s @1280, conf ≥ 0,5, ngoài dải mép, **chỉ khung dev**),
   ghi `source: "yolo11s-so-bo"` — không phase nào khác sửa file này (D10, sở hữu file); `check()` kiểm schema. Dải mép: đo bề rộng thực (kỳ vọng ~13 px), giữ `crop_left_px: 16`.
3. Kiểm tái tạo JPEG: lấy 20 khung dev, giải mã rồi nén lại bằng Pillow `qtables=<bảng hồ sơ>`, `subsampling="4:2:2"`,
   **không** truyền `quality`; DQT + SOF của file ra phải trùng file gốc.
4. `results/camera/profile_report.md`: bảng từng đại lượng (p5/p50/p95) cạnh số mẫu 300 khung của `phan-tich-video.md`,
   chênh lệch nào > 20 % thì giải thích (khác tập đo: dev so với mẫu chung).
5. **Test** `tests/camera/test_camera_profile.py`: hồ sơ qua schema; `measured_on.n_frames` = số khung dev trong
   `split.json`; không khung test nào nằm trong danh sách đo (đọc manifest đo); bảng DQT trong hồ sơ = bảng đọc từ một khung
   dev bất kỳ; bước 3 khớp 100 %; mỗi hàm đo có một ca tổng hợp biết trước đáp số (ảnh xám đều → % hồng 0, dải đen 13 px
   → bề rộng 13 ± 1). Phần cần dữ liệu thật gắn marker `data`.
6. Commit trên `lane/data`, đẩy, báo lead.

## Cổng pass

- Accept exit 0; `python -m xla.camera.profile check configs/camera_profile.yaml` exit 0.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MTF từ cảnh tự nhiên nhiễu | medium | low | Ghi là ước lượng; D2 hiệu chỉnh theo Laplacian + PSD cùng lúc (phase 07) |
| Hồ sơ khớp riêng buổi 02/11 | high | medium | `profile_id` ghi rõ buổi; hồ sơ mới = version mới, không ghi đè |

## Rollback

Revert commit; hồ sơ có version nên phase sau ghi rõ đã dùng bản nào.
