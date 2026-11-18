# Phase 07: Hiệu chỉnh D2 trên dev, báo KID/C2ST của D1 vs D2

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | phase 05, 06 | M (~1,5 ngày agent), 2–4 giờ GPU (chưa xác minh) | Không |

Lane: degrade
Depends-on: 05, 06
Effort: M
Owns: `configs/degrade_d2.yaml`
Creates: `xla/degrade/calibrate.py`, `results/realism/calibration.json`, `results/realism/d1_vs_d2.json`, `results/realism/README.md`, `tests/degrade/test_calibration_report.py`
AX: `python -m xla.degrade.calibrate --split dev --budget N --json` tìm tham số D2, in khoảng cách trước/sau, exit 1 khi không giảm
Accept: `pytest tests/degrade/test_calibration_report.py`

## Mục tiêu

Chỉnh vài tham số của chuỗi D2 cho thống kê ảnh tổng hợp khớp khung **dev** thật; sinh đủ D0/D1/D2 cho huấn luyện; trả lời
câu hỏi dữ liệu của §8: KID(D2, thật) < KID(D1, thật) với CI không chồng nhau, cộng C2ST và thống kê mức thấp, đo trên
khung thật **không dùng để hiệu chỉnh** (khung test, chỉ dùng ảnh, không dùng nhãn).

## Đầu vào cần có

- `xla/degrade/*` + `configs/degrade_d2.yaml` (phase 05); `xla/eval/realism.py`, `xla/eval/c2st.py` (phase 06).
- `data/real/split.json`; `configs/camera_profile.yaml`.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §2 (Real-SR là ablation tuỳ chọn; CycleGAN có bằng
  chứng hại → không làm), §5.
- `plan.md` D9, D10; Deferred 4, 7, 19.

## Việc theo thứ tự

1. **Tham số được chỉnh** (ít, có nghĩa vật lý; khoảng tìm ghi trong config): độ truyền `t` + ánh sáng nền `A` của màn mờ,
   σ PSF, hệ số cân bằng trắng R/B + nhân độ sáng trước cắt, mức khử nhiễu ISP, hệ số NLF. Không chỉnh DQT (đã đo đúng).
2. **Hàm mục tiêu** = tổng khoảng cách thống kê mức thấp (phase 06 bước 2) giữa D2 sinh trên 500 ảnh nguồn và khung **dev**.
   Tối ưu không đạo hàm (Nelder-Mead hoặc CMA-ES nhỏ), CPU, ngân sách `--budget` lượt; ghi đường hội tụ vào
   `results/realism/calibration.json`. Ghi tham số tốt nhất vào `configs/degrade_d2.yaml` (tăng `params_version`).
3. **Sinh đủ tập**: `python -m xla.degrade.build` cho D0, D1, D2 (cùng ảnh nguồn, cùng cửa sổ, seed 20261118); ghi số ảnh,
   cỡ ổ đĩa, sha256 manifest.
4. **So D1 vs D2 trên khung thật giữ riêng** (khung test, chỉ ảnh): thống kê mức thấp, C2ST, KID-DINOv2 (10 lần, CI 95 %),
   trong `xla.gpu_lock`. Smoke trước: 1 lần KID 1 000 patch, đo giờ GPU thật, ghi; rồi chạy đủ. Ghi `d1_vs_d2.json`.
   Kết luận viết đúng như số ra, kể cả khi D2 **không** thắng (§8 "Dữ liệu" là câu hỏi nghiên cứu, không phải điều kiện pass).
5. **Tuỳ chọn E3 (Deferred 7):** nếu còn giờ, biến thể D2+Real-SR (gom phần dư cao tần từ khung dev, không dán patch dính khối
   JPEG, JPEG đúng một lần ở cuối) đo cùng bước 4. Không làm thì ghi "không làm, lý do".
6. `results/realism/README.md`: bảng D1/D2 (/D2+RealSR) × (thống kê, C2ST, KID ± CI), tham số đã chỉnh, giờ GPU đo được.
7. **Test** `tests/degrade/test_calibration_report.py`: `calibration.json` có danh sách khung đã dùng và **không** khung nào
   thuộc test; khoảng cách sau ≤ trước; `d1_vs_d2.json` có đủ trường (mean, ci_low, ci_high, n_patches, n_repeats, seed, giờ
   GPU smoke) cho D1 và D2; `configs/degrade_d2.yaml` hợp lệ và `params_version` tăng; manifest D0/D1/D2 cùng tập ảnh nguồn.
8. Commit trên `lane/degrade`, đẩy, báo lead (kèm số giờ GPU đo được để lead cập nhật Forecast "Re-score").

## Cổng pass

- Accept exit 0; ba tập D0/D1/D2 sinh xong với số ảnh bằng nhau.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Khớp quá mức buổi 02/11 | high | medium | Báo cáo khai phạm vi; hồ sơ + tham số có version |
| D2 không thắng D1 | medium | medium | Vẫn báo; phase 10 vẫn so D0/D1/D2 trên nhận diện — kết quả nào cũng là đáp án câu hỏi 1 |

## Rollback

Trả `configs/degrade_d2.yaml` về `params_version` trước; tập sinh tái tạo bằng seed.
