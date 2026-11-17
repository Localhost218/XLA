# Kiểm repo D:\XLA (Localhost218/XLA, commit 3e9a764) — 18/11/2026

Nguồn: agent kiểm XLA (chỉ đọc + chạy thử; repo không bị sửa). File cùng thư mục: run_audit.py,
out\metrics.csv, in\A_t*.png, in\B_t*.png, out\*_sbs.jpg (vào | Stage1 | v2), out\*_crop2x.jpg,
out\gt001_clean_vs_v2.jpg, prov\manifest_check.jpg, prov\real_grid*.jpg, prov\synth_grid.jpg, prov\vid1.jpg,
prov\vid3.jpg. venv D:\DevCache\venvs\xla (Python 3.12).

## Tóm tắt
1. Không có ảnh nào từ camera drone này. 60 "real_uav" + 30 GT là ảnh drone phố Trung Quốc (biển chữ Hán),
   bị cộng hồng nhân tạo; manifest ghi sai nguồn (ảnh 001 ghi là khung 25 video FPV 1, thực là ngã tư).
   source_seq của real 021–060 (0000056…0000118) nối tiếp dãy GT → cùng một kho; tên kiểu VisDrone-DET
   (chưa đối chiếu). Mã sinh ba thư mục này không có trong repo.
2. raw_videos: HEVC 800×456 / H.264 1280×720 30 fps, encoder Lavf58 (đã mã hoá lại), cánh đồng/rừng/giông —
   không phải video drone này.
3. PSNR/SSIM/ΔE/PCI đo trên dữ liệu do chính mô hình suy giảm của nhóm sinh ra, thuật toán viết để đảo đúng
   mô hình đó → vòng tròn. Ngay trong vòng tròn v2 vẫn thua "không làm gì" vài chỗ (SSIM 0,7015→0,6996;
   motion_jpeg PSNR 28,67→24,50, ΔE 7,21→15,16).
4. Entropy/Laplacian thưởng nhiễu: Global HE có Laplacian 1108/1760, entropy 7,46/7,89 (cao nhất) dù PSNR 12,5.
5. Không có đánh giá phát hiện người (grep yolo/detect/person chỉ ra 1 docstring).
6. Cháy hồng CÓ thật trên OV3660 (sỏi/bê tông dưới nắng, 02/11). Stage 1 khử tốt, nhưng mặt nạ
   `(R+B)/2−G>6` và `sáng>110` làm bạc vật đỏ/hồng/tím thật: chạy trên GT sạch uav_gt_001 → PSNR 24,75 dB,
   ΔE 12,2 (mái ngói đỏ → nâu cam, dù hồng → xám).
7. Tốc độ trên 10 khung thật (CPU laptop): trung bình 448 ms/khung (≈2,2 khung/s; S1 ~74–219, S2 ~83–129,
   S3 ~185–248 ms); đường video 720p 376 ms (≈2,7 khung/s). Không đạt luồng 6–9 khung/s.
8. Trên khung thật: hết hồng, trong hơn; nhưng vignetting bị khuếch đại thành viền đen, vùng khử hồng thành
   trắng phẳng, cỏ ngả xanh bão hoà, ~0,5–1 % điểm ảnh bị đẩy về đen, lift_policy = 1,0 cả khung trong nhà
   mà L* vẫn giảm 2–5. Stage 3 (ESPCN + Richardson-Lucy + 720p/1080p) không thêm thông tin, tốn ~40 % thời gian.
9. Phần cứng mô tả sai (OV2640/ESP32-CAM). Web server bind 0.0.0.0.

## Tái dùng (sau khi sửa)
- recover_pink_highlights: chỉ kích hoạt khi gần bão hoà (max(R,B) ≳ 240), bỏ ngưỡng sáng 110.
- estimate_noise_sigma + khử nhiễu theo sigma.
- Trừ veil theo percentile — chỉ sau khi bù vignette (flat-field).
- Khung sinh dữ liệu có seed + manifest của data_augmentation.py (thay mô hình hồng bằng bão hoà kênh).

## Bỏ
Bảng 1–7 và kết luận định lượng; gọi real_uav/raw_videos là "camera ESP32 thật"; Stage 3; CLAHE + bù bão hoà
sắc độ; mô tả OV2640.
