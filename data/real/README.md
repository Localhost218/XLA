# `data/real/` — khung bay thật của camera drone IOT-CV (phase 03)

Khung JPEG **nguyên byte** camera ESP32-S3 + OV3660 đã gửi (MJPEG 640×480, ESP32 quality 12), chép thẳng từ chunk
`00dc` trong AVI của IOT-CV, không giải mã, không nén lại. Chủ dự án đã đồng ý commit khung vào repo, kể cả khung có
người và toạ độ GPS (18/11/2026, plan Deferred 14).

| File | Nội dung |
|---|---|
| `frames/<rec>_<segment>_<video_frame>.jpg` | 2 915 khung bay + 300 khung âm (trong nhà), tổng 67,2 MB |
| `manifest.csv` | mỗi khung: `kind` (flight/neg), lần ghi, segment, video_frame, giờ Unix, `alt_m`, chuyến, cỡ, sha256 |
| `split.json` | chia dev/test/neg (hợp đồng §7); **chỉ đọc** sau phase 03, chỉ được chia lại trước phase 08 |

Báo cáo và số đếm: `results/data/extract_report.json`, `results/data/split_report.json`,
`results/data/prelim_person_boxes.csv`, `results/data/flights_events.txt` (đầu ra nguyên văn của
`uv run python -m backend.flights --events` trong IOT-CV, 18/11/2026).

## Nguồn

- Bay: `d:\Coding\IOT-CV\logs\recordings\20261102-100941` (sáng 02/11/2026, 6 chuyến 10:10–10:44), lần ghi duy nhất có
  cảnh bay.
- Âm: bốn lần ghi trong nhà `20261102-033719`, `-040317`, `-042928`, `-043054` (03:37–05:35): 1 khung / 2 s mỗi lần ghi
  (2 719 ứng viên), rải đều xuống 300 khung. Không có độ cao → `alt_m = null` trong `split.json`.
- Đọc bằng `Recording` / `avi_chunk_index` của IOT-CV (`backend/flights/video.py`, import qua `xla.paths.iotcv_root()`,
  không chép mã). Ghép khung ↔ dòng `frames.csv` theo khoá (`segment`, `video_frame`), không theo thứ tự dòng.

## Bộ lọc khung bay và số đếm (`python -m xla.camera.frames extract --json`)

| Bước | Còn lại |
|---|---|
| Chunk `00dc` trong 8 file AVI | 17 775 |
| Có dòng `frames.csv` (chunk không có dòng = khung lấp chỗ mất hình, không độ cao: loại 5 867) | 11 908 |
| Trong cửa sổ 6 chuyến "bay: có" + có video (`flights_events.txt`, giờ VN) | 4 769 |
| `rel_alt_m > 3` | 2 944 |
| Không phải điểm vọt (lệch trung vị trượt căn giữa 5 s quá 5 m; loại 29) | **2 915** |
| Giải mã được 640×480 | 2 915 |

2 944 của plan đúng là số sau bước `rel_alt_m > 3` (`features.py` 18/11 không có bước lọc điểm vọt); 2 915 nằm trong dải
±10 %. 29 điểm vọt nằm quanh các lần mất link (độ cao CSV đứng yên hoặc vọt tới 57,8 m lúc 10:43:27). Cột `armed`/`mode`
trước 15/11 không tin được nên không dùng.

Khung theo chuyến: 10:10:46 → 371, 10:14:35 → 616, 10:23:19 → 241, 10:26:36 → 323, 10:30:39 → 530, 10:40:39 → 834.

## Đếm người sơ bộ (`python -m xla.camera.frames prelim --json`)

YOLO11s COCO (`D:\IOT_Tools\downloads\yolo-weights\yolo11s.pt`) @1280, conf ≥ 0,25, lớp person, chạy trong
`xla.gpu_lock`; bỏ hộp có `x1 < 16` (chạm dải tối mép trái — 521 hộp bị bỏ). Còn 769 hộp; 572/2 915 khung bay có ít nhất
một hộp; 0/300 khung âm. **Không phải nhãn**: theo duyệt tay 18/11, khoảng 40 % hộp ở ngưỡng này là đốm hồng/trắng.

## Chia dev/test (`python -m xla.camera.split build --seed 20261118 --json`)

Khối 30 s từ giờ bắt đầu chuyến (24 khối bay); tầng = (bin độ cao theo trung vị khối 3–8 / 8–14 / > 14 m, khối có người
sơ bộ); seed 20261118; khung test chọn tham lam cách nhau ≥ 1 s; khối dev giữ mọi khung, trừ khung cách khối test < 5 s.

| | Khung | Có người sơ bộ (3–8 / 8–14 / > 14 m) |
|---|---|---|
| dev | 470 | 16 / 18 / 40 |
| test | 283 | 3 / 13 / 46 (= 62) |
| neg | 300 | 0 |
| không thuộc tập nào | 1 861 khung khối test không được chọn (cách < 1 s) + 301 khung dev trong đệm 5 s | |

**Lệnh thoát 1: hai kiểm của plan không đạt — test 283 < 300 và test có người 62 < 150.** Nguyên nhân là kỳ vọng
trong plan, không phải lỗi chia: nếu đưa **toàn bộ** 2 915 khung bay vào test, chọn cách nhau ≥ 1 s chỉ được **362**
khung, trong đó **74** khung có người sơ bộ (`split_report.json` → `feasibility`). Ước "~370 khung có người" của báo cáo
18/11 đếm khung liên tiếp ~5 khung/s, không phải khung cách nhau 1 s. Camera chỉ phủ ~420 s trong 6 chuyến (49 % thời
gian bay nằm trong khoảng mất hình > 1 s).

Vì mục tiêu 300 khung test gần bằng trần 362, thuật toán (đúng quy tắc plan: lấy khối tới khi đủ 300, mỗi tầng chừa ít
nhất một khối cho dev) đưa 19/24 khối vào test và dev chỉ còn 470 khung. Phase 08 cần dev-val 100 khung cách nhau ≥ 2 s
— dev này không đủ. Đây là quyết định của Owner; chia lại bằng `--test-min/--test-max` (chỉ trước phase 08), các mốc
đã chạy thử cùng seed:

| `--test-min` | test | test có người | dev | dev có người | khối test |
|---|---|---|---|---|---|
| 100 | 103 | 23 | 1 877 | 367 | 7 |
| 150 | 154 | 32 | 1 410 | 317 | 11 |
| 200 | 198 | 45 | 1 115 | 224 | 13 |
| 300 (plan, đang dùng) | 283 | 62 | 470 | 74 | 19 |

Chia lại thì phải dựng lại hồ sơ camera (`python -m xla.camera.profile build --split dev`, tăng `profile_version`).
