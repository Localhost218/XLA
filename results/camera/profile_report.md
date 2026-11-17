# Hồ sơ camera v1 — báo cáo đo (phase 04)

Hồ sơ: `configs/camera_profile.yaml` `profile_version` 1.1.0, `profile_id` `ov3660-esp32s3-q12-20261102`. Đo trên **1115 khung dev** của `data/real/split.json` (D10; 0 khung test — đếm giao với test: 0). Số đo từng khung: `results/camera/profile_measurements.csv`. Lệnh: `python -m xla.camera.profile build --split dev --json`.

Cột "mẫu 18/11" tính lại từ `plans/reports/261118-bang-chung/video/features.csv` (300 khung bay rải đều, gồm cả khung nay thuộc test, đo trên khung **nguyên** không cắt dải mép 16 px). Chênh lệch p50 > 20 % có giải thích ở cột ghi chú.

| Đại lượng (trường hồ sơ) | dev p5 | dev p50 | dev p95 | mẫu 18/11 p5 | p50 | p95 | Δp50 | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| `haze.dark_channel_15px_0_255` | 134.61 | 143.73 | 152.1 | 125.54 | 141.66 | 150.32 | +1 % |  |
| `haze.y_p1_0_255` | 89.0 | 113.0 | 129.0 | 63.0 | 80.0 | 107.0 | +41 % **>20 %** | mẫu cũ tính cả dải tối mép trái → p1 thấp hơn |
| `y_p99 (không vào hồ sơ riêng)` | 188.0 | 217.0 | 229.0 | 193.0 | 215.0 | 231.05 | +1 % |  |
| `haze.saturation_0_255` | 7.4521 | 16.159 | 22.679 | 8.1963 | 15.865 | 22.569 | +2 % |  |
| `magenta.frac_pixels` | 0.0 | 0.0017495 | 0.032649 | 0.0 | 0.0023014 | 0.026155 | -24 % **>20 %** | mẫu cũ ghi %, đã đổi ra tỉ lệ. Δ do tập đo khác: dev là vài khối 30 s liền (độ cao: `altitude_m` của hồ sơ), mẫu rải đều 6 chuyến; % hồng tăng theo độ cao (phan-tich-video.md); giá trị nhỏ, đuôi nặng nên p50 dao động mạnh |
| `blocking.ratio_8x8` | 2.1512 | 2.6792 | 3.5234 | 1.972 | 2.6334 | 3.1981 | +2 % | mẫu cũ có cả dải mép, lưới lệch pha |
| `sharpness.laplacian_var` | 13.447 | 34.101 | 67.412 | 16.011 | 33.354 | 106.06 | +2 % |  |
| `noise.sigma_immerkaer_0_255` | 0.24783 | 0.35324 | 0.47416 | 0.28238 | 0.35986 | 0.60743 | -2 % |  |
| `motion_blur.sobel_x2_over_y2` | 0.45984 | 0.85424 | 1.1757 | 1.0167 | 1.1752 | 1.8548 | -27 % **>20 %** | khác định nghĩa: mẫu cũ đo max/min (≥ 1), hồ sơ đo đúng tên Gx²/Gy² (< 1 khi nhoè dọc); so cùng định nghĩa ở dòng dưới |
| `max(Gx²/Gy², Gy²/Gx²) (dùng cho frac_frames_gt_1p5)` | 1.022 | 1.1903 | 2.1747 | 1.0167 | 1.1752 | 1.8548 | +1 % |  |
| `color.r_over_g` | 0.96684 | 0.9753 | 0.98435 | 0.96571 | 0.9752 | 0.98246 | +0 % |  |
| `color.b_over_g` | 0.94183 | 0.96423 | 0.98314 | 0.9443 | 0.96561 | 0.9826 | -0 % |  |
| `left_strip.mean_y_strip` | 87.64 | 100.89 | 123.42 | 81.266 | 98.774 | 120.73 | +2 % | mẫu cũ lấy cố định cột 0–11 |
| `vignette.corner_center_ratio` | 0.62455 | 0.81105 | 0.92974 | – | 0.77 | – | +5 % | mẫu cũ chỉ có số 0,77 trong văn bản |
| `sharpness.mtf50_cyc_per_px` | 0.17016 | 0.24702 | 0.33332 | – | – | – | – | mới, ước lượng từ cạnh cảnh tự nhiên |
| `left_strip.width_px` | 13.0 | 14.0 | 16.0 | – | – | – | – | mới; kỳ vọng ~13 px |

Hồ sơ trước để so: `results/camera/history/camera_profile-v1.0.0.yaml`.

## So v1.0.0 → v1.1.0

| Trường | v1.0.0 | v1.1.0 | Δ |
|---|---|---|---|
| `measured_on.n_frames` | 470.0 | 1115.0 | +137 % |
| `left_strip.width_px` | 14.0 | 14.0 | +0 % |
| `left_strip.mean_y_strip` | 98.36 | 100.89 | +3 % |
| `vignette.corner_center_ratio.p50` | 0.80699 | 0.81105 | +1 % |
| `vignette.focal_px_fit` | 1003.5 | 1016.1 | +1 % |
| `haze.dark_channel_15px_0_255.p50` | 144.78 | 143.73 | -1 % |
| `haze.y_p1_0_255.p50` | 112.0 | 113.0 | +1 % |
| `haze.saturation_0_255.p50` | 16.725 | 16.159 | -3 % |
| `sharpness.laplacian_var.p50` | 34.977 | 34.101 | -3 % |
| `sharpness.mtf50_cyc_per_px.p50` | 0.25061 | 0.24702 | -1 % |
| `sharpness.psf_sigma_px_fit` | 0.74774 | 0.75861 | +1 % |
| `noise.sigma_immerkaer_0_255.p50` | 0.35741 | 0.35324 | -1 % |
| `color.r_over_g.p50` | 0.97646 | 0.9753 | -0 % |
| `color.b_over_g.p50` | 0.96485 | 0.96423 | -0 % |
| `magenta.frac_pixels.p50` | 0.0035089 | 0.0017495 | -50 % |
| `magenta.frac_frames_gt_0p5pct` | 0.37234 | 0.31211 | -16 % |
| `magenta.highlight_y_0_255.p50` | 223.0 | 222.23 | -0 % |
| `blocking.ratio_8x8.p50` | 2.6564 | 2.6792 | +1 % |
| `motion_blur.sobel_x2_over_y2.p50` | 0.82461 | 0.85424 | +4 % |
| `motion_blur.frac_frames_gt_1p5` | 0.38936 | 0.31928 | -18 % |
| `person_box_px.short_side.p50` | 53.24 | 53.545 | +1 % |
| `altitude_m.flight_frames.p50` | 11.785 | 12.291 | +4 % |

## JPEG

- Mọi khung dev cùng **1** bộ (DQT luma, DQT chroma, lấy mẫu, baseline): `4:2:2`, baseline True, Annex K × 0.75 (ESP32 quality 12).
- Tái tạo bằng Pillow `qtables=[luma, chroma]`, `subsampling="4:2:2"`, không truyền `quality`: **20/20** khung dev khớp cả DQT lẫn SOF.
- Bảng cho auto-quality 19–40: giả thuyết `annex_k * q / 16`, `verified: false` (Deferred 1).

## Số suy ra

- Dải mép: rộng p50 14.0 px (định nghĩa: số cột đầu có Y dưới điểm giữa mức dải và mức nền; dải là dốc mềm ~8 px chứ không phải cạnh sắc), giữ `crop_left_px` 16.
- Tối góc cos⁴: `focal_px_fit` 1016.1 px suy từ p50 tỉ số góc/giữa (ô cạnh 1/6).
- PSF Gauss từ MTF50 p50: σ = 0.75861 px (MTF50 là **ước lượng từ cảnh tự nhiên**, nhiễu; phase 07 hiệu chỉnh cùng Laplacian + PSD).
- `motion_blur.frac_frames_gt_1p5` dùng max(Gx²/Gy², Gy²/Gx²) > 1,5 (bất đẳng hướng, như mẫu 18/11).
- `person_box_px`: 62 hộp YOLO11s @1280 conf ≥ 0.5 trên khung dev, ngoài dải mép; nguồn `yolo11s-so-bo` (chưa phải nhãn, còn hộp sai do đốm hồng).
- NLF: 6/8 bin Y đủ điểm phẳng (≥ 10% khung) — ảnh sáng, mức đen nâng, nên bin tối không có dữ liệu.
