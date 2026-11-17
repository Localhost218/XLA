# Phân tích video thật từ camera drone IOT-CV (ESP32-S3 + OV3660) — 18/11/2026

Nguồn: agent phân tích video (chỉ đọc `logs/recordings/`). File số liệu cùng thư mục: inventory.json,
all_frames_scan.csv, features.csv, samples/ (392 JPEG), detections_yolo11n.csv, detections_yolo11s.csv,
det11s_labeled.csv, crops_yolo11s.jpg, luoi-12-khung.jpg. Script: inventory.py, features.py, yolo_run.py,
grid.py. venv D:\DevCache\venvs\vidstats; trọng số D:\IOT_Tools\downloads\yolo-weights\.
Tác dụng phụ: `uv run python -m backend.flights --events` đã tạo lại d:\Coding\IOT-CV\.venv (Python 3.13).

## Tóm tắt
- 40 969 khung, tất cả 640×480 MJPEG. CHỈ `20261102-100941` (10:09–10:45, 6 chuyến) có cảnh bay:
  2 944 khung bay (trong cửa sổ chuyến của backend.flights và rel_alt_m > 3 m). Các lần ghi 01/11 không
  trùng chuyến nào (laptop ngủ 19:45), tối đen ám xanh lá. Bốn lần ghi 02/11 03:37–05:35 quay trong nhà.
- Camera chúc thẳng xuống (nadir). Người nhìn từ đỉnh đầu, cao 12–20 m, hộp 21–139 px (trung vị 50 px).
- 0 khung giải mã hỏng, 0 khung thiếu EOI, không xé. Mất sóng = mất trắng khung. Lúc bay 49 % thời gian
  nằm trong khoảng hở > 1 s (max 14,8 s).
- JPEG cố định IJG ≈ 62 (ESP32 quality 12), ước từ bảng DQT; chưa có cột jpeg_quality (mới có từ 15/11).
- Cột armed/mode trước 15/11 không tin được (HEARTBEAT DroneBridge); rel_alt_m trôi 1–4 m khi nằm đất,
  có điểm vọt 57,8 m.

## Từng chuyến (20261102-100941)
| Chuyến | Dài s | Khung | fps | Hở>1s | Tổng hở s | Hở max s | KB TB | Cao trung vị m |
|---|---|---|---|---|---|---|---|---|
| 10:10:46 | 125 | 678 | 5.42 | 27 | 43 | 4.0 | 23.2 | 4.7 |
| 10:14:35 | 153 | 917 | 5.99 | 20 | 45 | 14.8 | 25.2 | 5.2 |
| 10:23:19 | 138 | 735 | 5.33 | 28 | 60 | 9.2 | 24.1 | 0.2 (CSV trễ khi mất link) |
| 10:26:36 | 159 | 622 | 3.91 | 27 | 82 | 13.6 | 23.4 | 4.2 |
| 10:30:39 | 162 | 825 | 5.09 | 31 | 55 | 5.0 | 24.2 | 5.2 |
| 10:40:39 | 214 | 992 | 4.64 | 42 | 93 | 8.5 | 24.5 | 13.0 |

## Đặc trưng suy giảm (mẫu 300 khung bay + 92 khung đất)
| Đặc trưng | Bay trung vị [p5–p95] | Ghi chú |
|---|---|---|
| Laplacian var | 33 [16–106] | 79 % < 50; gần như không đổi theo độ cao (33/29/35) |
| Nhiễu σ (Immerkaer) | 0.36 [0.28–0.61] | rất thấp: ISP + JPEG xoá nhiễu cùng chi tiết |
| Y trung bình / p1 / p99 | 163 / 80 / 215 | mức đen bị nâng lên 80 |
| Tương phản p99−p1 | 135 [102–159] | |
| Dark channel 15×15 | 142 [126–150] | mờ đục / màn sương (ngoài trời rõ ≈ 0) |
| Cháy sáng ≥250 | 0.003 % | gần như không |
| R/G, B/G | 0.975 / 0.966 | ban ngày không ám cả khung; tối 01/11 ám xanh 0.62/0.61 |
| Bão hoà màu S | 16 | gần như xám |
| Điểm hồng/tím (R−G>25 và B−G>25) | 0.23 % [0–2.6] | 31 % khung bay > 0,5 %, 12 % > 2 %; Y vùng hồng 222; tăng theo độ cao 0.05→0.28→0.48 % |
| Blocking JPEG | 2.63 [1.97–3.20] | ranh khối 8×8 lộ rõ |
| Nhoè chuyển động Sobel x²/y² | 1.18 [1.02–1.86] | 19 % > 1,5 |
| Vignetting góc/giữa | 0.77 | |
| Dải mép trái (cột 0–11) | 97 vs 130 | MỌI khung, cả trong nhà: dải tối ~13 px, vạch đỏ đều |

## YOLO pretrained (COCO person), RTX 4060
| Model | imgsz | Khung bay có hộp ≥0.25 /300 | Hộp ≥0.25 | Khung hộp ≥0.5 |
|---|---|---|---|---|
| yolo11n | 640 | 33 | 46 | 5 |
| yolo11n | 1280 | 69 | 87 | 9 |
| yolo11s | 640 | 63 | 87 | 15 |
| yolo11s | 1280 | 95 | 128 | 20 |

Duyệt tay 128 hộp yolo11s@1280: 52 người thật (38 khung, cạnh trung vị 50 px, conf trung vị 0.37), 43 sai do
dải mép trái, 28 sai do đốm hồng/trắng, 5 chưa chắc. Precision 41 % (61 % nếu bỏ dải mép). Recall chưa đo;
đếm tay 4 khung: ~8 người, model bắt 4; người đứng nhóm hay sót. Có người ở 12,7 % khung bay mẫu
(ước ~370 khung có người trong 2 944).

## Xếp hạng vấn đề theo ảnh hưởng tới nhận diện
1. Người nhỏ, nhìn từ đỉnh đầu (COCO gần như không có góc này). imgsz 1280 tăng khung có hộp đúng 17→38.
2. Ám hồng vùng sáng (lỗi ISP/AWB khi bão hoà) — 28 hộp sai, 31 % khung bay.
3. Dải đen ~13 px mép trái — 43/128 hộp sai; cắt 16 px là bỏ được.
4. Mờ đục, tương phản thấp (dark channel 142, p1 80, S 16).
5. Ảnh mềm + blocking JPEG.
6. Mất khung theo cụm (đứt tracking).
7. Nhoè chuyển động vừa, vignetting 0.77.
8. Tối 01/11 (không có khung bay).

## Dữ liệu đủ làm gì
Đánh giá: đủ một bộ test nhỏ (gán ~300–500 khung cách nhau ≥ 1 s; một bãi, một buổi sáng, nadir).
Huấn luyện từ đầu: không đủ. Fine-tune: được nếu trộn VisDrone/Okutama. Đủ để ước tham số mô hình suy
giảm (mức đen, ám hồng, Q62, mềm). Chưa đo: recall/mAP, độ nét có tham chiếu, nguyên nhân dải mép và
màn mờ, tốc độ suy luận liên tục, ảnh hưởng của quality khác 12.
