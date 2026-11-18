# Phase 12: Đối chứng cổ điển viết lại, vector hoá, ≤ 30 ms

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | phase 04 | M (~1,5 ngày agent), 0 giờ GPU | Không |

Kết quả (18/11/2026, `e859e22`, gộp `dcc04d4`): trên 200 khung dev, điểm hồng −91,6 %, dark channel trung vị 142,7 → 65,3,
blocking 2,73 → 2,08. Bench đo lại ở phase 14 lúc máy khoẻ (cv2 4 luồng): p50 9,18 / p95 10,21 ms, ≤ 30 ms.
Chi tiết: `results/enhance/classic_bench.json`.

Lane: classic
Depends-on: 04
Effort: M
Creates: `xla/enhance/classic.py`, `configs/enhance_classic.yaml`, `results/enhance/classic_bench.json`, `tests/enhance/test_classic.py`
AX: `python -m xla.enhance.classic --in <jpg|dir> --out <dir> --bench --json` xử lý ảnh, in ms/khung p50/p95
Accept: `pytest tests/enhance/test_classic.py`

## Mục tiêu

Giữ phần cổ điển của nhóm làm **phương pháp đối chứng** cho nhánh A, sau khi sửa đúng các lỗi đã đo: cắt dải mép → bù tối
góc → khử màn mờ (dark channel, percentile) → khử hồng chỉ khi gần bão hoà → khử khối nhẹ → làm nét nhẹ. Vector hoá cho
≤ 30 ms/khung 640×480. Đây cũng là phần "xử lý ảnh cổ điển" của báo cáo môn.

## Đầu vào cần có

- `configs/camera_profile.yaml` (phase 04: `crop_left_px`, `vignette`, `haze`, `magenta`).
- `plans/reports/261118-bang-chung/kiem-xla/kiem-xla.md` §6–8 (mặt nạ hồng làm bạc vật đỏ thật; tối góc bị khuếch đại thành
  viền đen; ~0,5–1 % điểm ảnh bị đẩy về đen; Stage 3 bỏ).
- Prior art (đọc, không import — `src/` là bản v2 cũ đóng băng): `src/uav_pipeline.py:98` `recover_pink_highlights`,
  `:84` `estimate_noise_sigma`, `:180` `bilateral_denoise`, `:341` `_dehaze_dark_channel`, `:72` `_guided_filter`.
- Thiết kế §3 "Giữ lại/Bỏ" (bỏ CLAHE, bù bão hoà sắc độ, ESPCN/Richardson-Lucy, nâng 720p).

## Việc theo thứ tự

1. Viết lại từng bước thành hàm thuần numpy/cv2, không vòng Python theo điểm ảnh, tham số trong `configs/enhance_classic.yaml`:
   (a) cắt `[crop_left, 640)`; (b) chia cho bản đồ tối góc cos⁴ của hồ sơ (flat-field); (c) màn mờ: ước A từ percentile 99,9
   của dark channel, `t` có sàn, guided filter; trừ màn mờ theo percentile, giữ mức đen không tụt dưới p0,5 gốc;
   (d) khử hồng: kích hoạt khi `max(R,B) ≥ ~240` **và** `(R+B)/2 − G` vượt ngưỡng, bỏ cổng sáng 110, chỉ dịch sắc độ giữ Y
   (công thức gốc dòng 98–137); (e) khử khối: làm mượt có trọng số chỉ trên đường biên 8×8; (f) unsharp nhẹ theo σ nhiễu;
   (g) đệm lại cột trái bằng lặp cột `crop_left` (giống hợp đồng §3) → 640×480.
2. Bench: 200 khung dev, warmup 20, ms/khung p50/p95 trên CPU laptop, ghi `classic_bench.json` (CPU, phiên bản numpy/cv2).
   > 30 ms → profile từng bước, tối ưu bước chậm nhất trước khi bỏ bớt.
3. **Test** `tests/enhance/test_classic.py` (CPU, ảnh tổng hợp): mảng đỏ bão hoà không hồng (mái ngói) có ΔE2000 < 3 sau xử lý;
   mảng hồng gần bão hoà giảm ≥ 90 % điểm hồng theo luật hồ sơ; ảnh có màn mờ tổng hợp giảm dark channel trung vị ≥ 50 %;
   tỉ lệ điểm bị đẩy về ≤ 3 không tăng quá 0,1 điểm %; ảnh ra 640×480 uint8, cột trái đúng quy tắc lặp; `classic_bench.json`
   (nếu có) có p50 ≤ 30 (marker `data`).
4. Commit trên `lane/classic`, đẩy, báo lead.

## Cổng pass

- Accept exit 0; bench p50 ≤ 30 ms (hoặc lý do + số đo trong `classic_bench.json`).

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Không kịp 30 ms trên CPU | medium | low | Đây là đối chứng, không ship vào IOT-CV; ghi số thật |
| Bù tối góc khuếch đại nhiễu góc | medium | low | Giới hạn hệ số bù trong config, test bước 3 canh mức đen |

## Rollback

Revert commit; `src/` cũ không bị đụng.
