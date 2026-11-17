# Phase 03: Nhập khung bay thật, ghép độ cao, chia dev/test theo khối thời gian

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 01 | M (~1 ngày agent), ~0,3 giờ GPU (đếm người sơ bộ, chưa xác minh) | Không |

Lane: data
Depends-on: 01
Effort: M
Creates: `xla/camera/frames.py`, `xla/camera/split.py`, `data/real/split.json`, `data/real/README.md`, `results/data/split_report.json`, `results/data/prelim_person_boxes.csv`, `tests/camera/test_frames_split.py`
AX: `python -m xla.camera.frames extract --json` và `python -m xla.camera.split build --seed 20261118 --json` in số khung sau từng bước lọc, exit 1 khi số đếm lệch dải mong đợi
Accept: `pytest tests/camera/test_frames_split.py`

## Mục tiêu

Lấy đúng các khung bay thật từ bản ghi `20261102-100941` của IOT-CV, giữ **nguyên byte JPEG camera đã gửi** (không nén
lại), gắn độ cao đúng khung, rồi chia dev/test theo khối thời gian **một lần** cho cả dự án (D10). Thêm tập mẫu âm (trong
nhà, không người). Sau phase này `data/real/split.json` là chỉ đọc.

## Đầu vào cần có

- Ràng buộc đã chốt: chưa thu thêm dữ liệu; chỉ 2 944 khung bay của `d:\Coding\IOT-CV\logs\recordings\20261102-100941`
  (6 chuyến) + khung trong nhà làm mẫu âm. Cột `armed`/`mode` của `frames.csv` trước 15/11 **không tin được**.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §0: AVI có 49 027 khung (chunk `00dc`) còn
  `frames.csv` chỉ 40 969 dòng → **ghép khung ↔ dòng theo `segment` + `video_frame`, không theo thứ tự dòng**; `rel_alt_m`
  có điểm vọt 57,8 m. Script tham khảo: `nghien-cuu/riff_count.py`.
- `plans/reports/261118-bang-chung/video/features.py` (cách đọc `Recording`, cửa sổ chuyến), `video/phan-tich-video.md`.
- IOT-CV (chỉ đọc, không sửa, không commit): `backend/flights/video.py` (`Recording`, `avi_chunk_index`);
  `uv run python -m backend.flights --events` chạy ở `d:\Coding\IOT-CV` (tác dụng phụ đã biết: tạo `.venv` Python 3.13 của IOT-CV nếu chưa có).
- `plan.md` Deferred 14, 15.

## Việc theo thứ tự

1. **Cửa sổ chuyến:** chạy `uv run python -m backend.flights --events` trong IOT-CV; lấy giờ bắt đầu + thời lượng các
   chuyến "bay: có" và có video. Lưu nguyên văn đầu ra vào `results/data/flights_events.txt`.
2. **`xla/camera/frames.py`:** import `Recording` từ IOT-CV theo `configs/paths.yaml` (không chép mã); duyệt từng `segment`,
   lấy byte JPEG theo `video_frame`, ghép dòng `frames.csv` theo khoá (`segment`, `video_frame`). Khung không có dòng CSV →
   không có độ cao → loại và đếm. Lọc: trong cửa sổ chuyến **và** `rel_alt_m > 3` **và** không phải điểm vọt (lệch trung vị
   trượt 5 s quá 5 m). Ghi `data/real/frames/<rec>_<segment>_<video_frame>.jpg` bằng **chép byte**, kèm sha256 vào manifest.
   In số khung sau từng bước lọc; số cuối kỳ vọng quanh 2 944 — lệch quá ±10 % thì exit 1 và báo, không tự nới lọc.
3. **Mẫu âm:** các bản ghi 02/11 03:37–05:35 (trong nhà): 1 khung / 2 s, tối đa 300, `split = "neg"`.
4. **Đếm người sơ bộ** (chỉ để phân tầng, không phải nhãn): YOLO11s COCO @1280, conf 0,25, bỏ hộp nằm trong 16 px mép trái,
   chạy trong `xla.gpu_lock`. Trọng số `D:\IOT_Tools\downloads\yolo-weights\yolo11s.pt`. Ghi mọi hộp vào
   `results/data/prelim_person_boxes.csv` (`frame,x1,y1,x2,y2,conf`) — phase 04 đọc file này để đo `person_box_px`.
5. **`xla/camera/split.py`** (Deferred 15): khối 30 s; phân tầng theo (bin độ cao 3–8 / 8–14 / > 14 m, có/không người sơ
   bộ); gán khối dev/test bằng seed 20261118 sao cho cả hai có người ở mọi bin độ cao có người; bỏ 5 s đệm ở biên khối dev
   kề khối test. Trong khối test, chọn khung cách nhau ≥ 1 s, mục tiêu 300–400. Khối dev giữ **mọi** khung (self-training
   phase 11 cần khung liên tiếp cho tracker). Ghi `data/real/split.json` đúng §7 hợp đồng.
6. `results/data/split_report.json`: số khung theo chuyến × split × bin độ cao × có người sơ bộ; `data/real/README.md` ghi
   nguồn, bộ lọc, seed, số đếm, và rằng `frames/` được commit vào repo (chủ dự án đã đồng ý 18/11/2026, Deferred 14).
7. Nếu test có < 150 khung có người sơ bộ: dừng, báo Owner (rủi ro hàng đầu của Forecast), không tự đổi cách chia.
8. **Test** `tests/camera/test_frames_split.py` (marker `data`): mỗi khung đã ghi có sha256 trùng byte trong AVI; mọi khung
   bay có `alt_m > 3`; dev ∩ test = ∅; không khung test nào cách khung dev < 5 s; khung test cách nhau ≥ 1 s; tập test có
   người ở ít nhất 2 bin độ cao; chạy lại `split build` cùng seed ra file giống hệt byte.
9. Commit trên nhánh `lane/data`, gồm cả `data/real/frames/` (file mới: `git add` đúng các đường dẫn rồi
   `git commit -- <đường dẫn>`, không `git add -A`). Trước khi commit in tổng dung lượng khung: ước 30–70 MB (cả buổi 02/11
   trung bình 9,3 KB/khung, đo 18/11/2026; khung bay ngoài trời có thể to hơn); quá 500 MB thì dừng, báo lead. Đẩy, báo lead.

## Cổng pass

- Accept exit 0.
- `split_report.json` có số khung test trong 300–400 hoặc lý do ghi rõ khi thấp hơn.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Quá ít người trong test | high | high | Bước 7 dừng sớm; báo cáo khai rõ, bootstrap theo khối |
| Ghép nhầm khung ↔ độ cao do lệch dòng CSV | medium | high | Ghép theo khoá, test so sha256 với AVI, đếm khung loại |
| Chạy lệnh trong IOT-CV làm bẩn repo đó | low | medium | Chỉ đọc; `git -C d:\Coding\IOT-CV status --short` trước/sau phải giống nhau (`.venv` đã gitignore) |

## Rollback

Xoá `data/real/` và nhánh `lane/data`; chỉ được chia lại trước khi phase 08 bắt đầu gán nhãn.
