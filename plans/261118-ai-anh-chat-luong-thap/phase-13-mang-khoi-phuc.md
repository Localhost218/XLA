# Phase 13: Mạng khôi phục riêng camera trên cặp D2, xuất `enhancer.onnx`

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | phase 02, 07 | L (~2,5 ngày agent), 6–20 giờ GPU (chưa xác minh — smoke đo trước) | Không |

Kết quả (18/11/2026, `868d881`, gộp `9650338`): vòng 1 NAFNet-SIDD-width32 p95 82,1 ms > 30 ms (Accept FAIL) → vòng 2
**NAFNet-w16** (1,14 M tham số, khởi tạo ngẫu nhiên, 4 500 bước, chọn `step_004500`): bundle `enhancer-v1.0.0` ORT CUDA EP
p50 15,4 / p95 21,0 ms, Accept 7/7 PASS; dev thật: hồng −99,9 %, dark channel −33,6 %, flicker 1,78 (> 1,15). Giờ GPU trong
khoá: vòng 1 1,03 + vòng 2 ~0,67. Chi tiết: `results/restore/README.md`, `models/release/enhancer-v1.0.0/meta.json`.

Lane: restore
Depends-on: 02, 07
Effort: L
Creates: `xla/enhance/arch/`, `xla/enhance/net.py`, `xla/enhance/train_net.py`, `xla/enhance/export_onnx.py`, `configs/restore.yaml`, `results/restore/speed.json`, `results/restore/train.json`, `tests/enhance/test_enhancer_export.py`
AX: `python -m xla.enhance.net speed --candidates nafnet,fbcnn,scunet --json` đo ms/khung từng ứng viên; `python -m xla.enhance.export_onnx --ckpt <pt> --out models/release/enhancer-v1.0.0 --json` dựng bundle và kiểm hợp đồng
Accept: `pytest tests/enhance/test_enhancer_export.py`

## Mục tiêu

Fine-tune một mạng khôi phục nhẹ trên cặp (D2 xấu → D0 sạch) để học đảo đúng chuỗi lỗi của camera này, chạy ≤ 30 ms/khung
trên RTX 4060, rồi xuất bundle `enhancer-v1.0.0` đúng hợp đồng §3, §4. Mạng phục vụ **người xem** (nhánh A); đường nhận diện
vẫn chạy khung gốc trừ khi E4 (phase 14) chứng minh khác (D5).

## Đầu vào cần có

- D0/D2 cùng ảnh nguồn (phase 07), `configs/degrade_d2.yaml`, `configs/camera_profile.yaml`.
- Hợp đồng `docs/hop-dong-model.md` §3, §4; `xla/contract.py`.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §3 (khôi phục tối ưu PSNR thường không giúp nhận diện),
  §6 hàng 4 (FBCNN/NAFNet).
- `plan.md` D9; Deferred 8, 11, 19; Forecast "Kill (c)".

## Việc theo thứ tự

1. **Đo tốc độ trước khi chọn** (`net.py speed`): NAFNet (bản nhỏ), FBCNN, SCUNet — trọng số công bố, ảnh vào 480×624,
   fp16, ORT CUDA EP sau khi xuất thử ONNX; warmup 30, đo 300. Ghi giấy phép từng mã/trọng số (đọc file LICENSE của repo gốc,
   ghi "chưa xác minh" nếu không đọc được) vào `results/restore/speed.json`. Chọn ứng viên nhanh nhất có p95 ≤ 20 ms (để dành
   ngân sách cho detector, tổng ≤ 50 ms). Không ứng viên nào đạt → dừng, báo lead (Forecast "Kill (c)").
2. **Mã mạng** đặt ở `xla/enhance/arch/` (chỉ file kiến trúc cần thiết, giữ header giấy phép gốc); `net.py` dựng/nạp model.
3. **Huấn luyện** (`train_net.py`, `configs/restore.yaml`, Deferred 11): cặp patch 256×256 từ D2 (vào) / D0 (đích) **cùng
   cửa sổ**, bỏ cột dải mép; loss L1 + (1 − SSIM), không GAN; khởi từ trọng số công bố; trong `xla.gpu_lock`. **Smoke** 500
   bước → giờ/1 000 bước, VRAM; ước tổng; > 10 giờ → báo lead. Chạy đủ; val PSNR/SSIM trên cặp D2 val (ghi là **số tổng hợp**,
   báo riêng, không dùng làm kết luận). Chọn checkpoint bằng thống kê mức thấp trên khung **dev** thật sau khôi phục
   (% hồng, dark channel, blocking — `xla/camera/measure.py`) cộng độ nhấp nháy trên chuỗi dev, **không** dùng NR-IQA (Deferred 5).
4. **Xuất ONNX** (`export_onnx.py`): bọc theo §3 (nhận `frame` uint8 BGR [480,640,3], cắt, chuẩn hoá, mạng, ép uint8 có làm
   tròn, đệm cột trái bằng lặp cột `crop_left`, ra `frame_out`); opset 17; so ORT CPU với PyTorch trên 20 khung dev: sai khác
   tuyệt đối trung bình ≤ 1,0; đo ms/khung ORT CUDA EP p50/p95.
5. **Bundle** `models/release/enhancer-v1.0.0/` (`enhancer.onnx`, `meta.json`, `golden/` 3 khung dev + `expected.json` +
   `*.out.png`); `python -m xla.contract validate` → `[]`.
6. **Test** `tests/enhance/test_enhancer_export.py` (marker `data`/`gpu` cho phần đo): `validate_bundle` rỗng lỗi;
   `meta.ms_per_frame.p95 ≤ 30`; cột `[0, crop_left)` của ảnh ra bằng cột `crop_left`; `speed.json` có đủ ba ứng viên và giấy
   phép; golden không chứa khung test.
7. Commit (code, config, JSON, `meta.json` + `golden/` của bundle; không `.onnx`/trọng số, Deferred 21) trên `lane/restore`,
   đẩy, báo lead kèm giờ GPU đo được.

## Cổng pass

- Accept exit 0; bundle qua `xla.contract validate`.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Mạng bịa chi tiết / làm mất người nhỏ | medium | high | Không GAN; E4 đo tác động lên nhận diện; đường nhận diện chạy khung gốc (D5) |
| Nhấp nháy giữa khung | medium | medium | Chọn checkpoint có tính nhấp nháy; phase 14 đo lại |
| Giấy phép trọng số không rõ | low | medium | Ghi trong `speed.json`; chọn ứng viên giấy phép rõ khi tốc độ ngang nhau |

## Rollback

Xoá `models/release/enhancer-*`; IOT-CV khi đó trả 503 cho luồng `enhanced` đúng hợp đồng.
