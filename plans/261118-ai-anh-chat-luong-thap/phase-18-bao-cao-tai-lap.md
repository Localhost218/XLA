# Phase 18: Báo cáo, slide, README, một lệnh tái lập, PR cuối

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 19/11/2026 | phase 07, 10, 11, 14, 16, 17 | M (~1,5 ngày agent) | Không |

Lane: serial
Depends-on: 07, 10, 11, 14, 16, 17
Gate: human
Effort: M
Owns: `README.md`, `BAO_CAO_DO_AN.md`
Creates: `scripts/reproduce.py`, `docs/slide/index.html`, `results/reproduce/manifest.json`, `tests/test_reproduce.py`
AX: `python scripts/reproduce.py --check --json` dựng lại mọi bảng + hình từ dữ liệu + seed và so sha256 với manifest, exit 1 khi lệch; `--full` chạy lại cả bước GPU
Accept: `pytest tests/test_reproduce.py`

## Mục tiêu

Khép dự án theo đề đồ án (`docs/DoAn_XuLyAnh.pdf`: pipeline nhiều bước, có đối chứng, ≥ 50 ảnh đánh giá, độ đo chuẩn) và tiêu
chí "Tái lập" của §8: một lệnh dựng lại toàn bộ bảng và hình. README và báo cáo viết lại đúng sự thật; slide là file HTML local.
Gộp mọi lane, mở PR #3, đưa bundle model lên GitHub Release (Deferred 21). Bước người: bảo commit IOT-CV.

## Đầu vào cần có

- Mọi `results/**` của phase 03–16; `archive/README.md` (phase 01); thiết kế §5.5 (hai câu hỏi nghiên cứu), §7, §8, §10.
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §8 (URL đã kiểm, dùng làm tài liệu trích dẫn).
- Giấy phép cần ghi: VisDrone (dùng nghiên cứu, không đăng lại ảnh), NOMAD CC BY 4.0 (ghi công), `pyiqa` PolyForm
  Noncommercial, mã/trọng số mạng khôi phục (`results/restore/speed.json`).
- Luật chủ máy: không publish Artifact/Drive; tài liệu là file trong repo; HTML có đủ `<!doctype html>`/`<head>`/`<body>`.
- `gh` đã đăng nhập, có quyền push `Localhost218/XLA` (lệnh mạng chạy ngoài sandbox).
- `plan.md` D12; Deferred 14, 21; Human steps hàng 3.

## Việc theo thứ tự

1. Gộp `lane/iotcv`, `lane/iotcv-docs` và mọi lane còn lại vào nhánh tích hợp; `pytest -m "not gpu"` xanh.
2. **`scripts/reproduce.py`:** đọc `results/reproduce/manifest.json` (danh sách bước: lệnh `python -m xla...`, đầu vào, seed,
   sha256 đầu ra); `--check` chạy lại các bước CPU từ dữ liệu cache + kết quả dự đoán đã lưu và so sha256 mọi bảng/hình;
   `--full` chạy cả bước GPU (huấn luyện) — ghi rõ giờ ước. Mọi bảng + hình trong báo cáo sinh từ script này, không dán tay.
3. **`BAO_CAO_DO_AN.md`** viết lại (dùng skill `t1k-human-writing` khi viết văn xuôi): bài toán + camera thật; hiện trạng đo
   được; phần cổ điển của nhóm (giữ, sửa gì, vì sao); bản sao số camera; độ giống thật; nhận diện E1/E2; ảnh rõ + A/B; E4; trả
   lời hai câu hỏi nghiên cứu kể cả khi kết quả trái dự đoán; giới hạn (1 bãi, 1 buổi, nhãn máy gán + duyệt, hồ sơ khớp 02/11,
   DQT q ≠ 12 chưa đo); §10 (sửa tận gốc ở lần bay tới) làm "việc tiếp"; trích dẫn. Phần cũ sai đã ở `archive/`.
4. **`README.md`** viết lại: mục tiêu, cấu trúc `xla/`, cài đặt (ổ D, `scripts/env.ps1`), lệnh tái lập, kết quả chính kèm CI,
   hợp đồng với IOT-CV, giấy phép dữ liệu.
5. **Slide** `docs/slide/index.html`: file local, đầy đủ khung HTML, ảnh nhúng từ `results/` bằng đường dẫn tương đối, không
   tải gì từ mạng; điều hướng bằng phím; ~12–15 trang.
6. **Bước người (Gate: human):** đưa danh sách file IOT-CV (phase 15–17) kèm kết quả test để chủ dự án bảo commit/PR; agent
   không tự commit IOT-CV. Khung bay thật đã nằm trong repo từ phase 03 (Deferred 14), không hỏi lại.
7. **Test** `tests/test_reproduce.py`: `reproduce.py --check` exit 0 (marker `data`); manifest có sha256 cho mọi bảng/hình
   mà báo cáo nhắc tới (quét tên file trong `BAO_CAO_DO_AN.md`); `docs/slide/index.html` có `<!doctype html>` và không có URL
   `http` ngoài phần trích dẫn; README không còn chữ "OV2640"/"ESP32-CAM" ngoài câu giải thích.
8. Commit, đẩy, mở **PR #3** vào `main`; CI xanh rồi merge; báo lead đường dẫn PR.
9. **GitHub Release cho bundle model** (Deferred 21): nén từng bundle đã ship (`detector-v<ver>`; `enhancer-v<ver>` nếu phase
   14 chọn ship) thành zip, `gh release create models-v<ver> <các zip> --repo Localhost218/XLA --target main` với ghi chú
   liệt kê `onnx_sha256` từng bundle. Tải lại từ Release vào thư mục trống, giải nén, `python -m xla.contract validate` →
   `[]`, sha256 khớp `meta.json` đã commit. README (bước 4) ghi tên release và cách tải.

## Cổng pass

- Accept exit 0; PR #3 merge; Release `models-v<ver>` có đủ bundle đã ship, bản tải về qua `xla.contract validate`; danh
  sách file IOT-CV ở bước 6 được ghi trong PR.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bảng trong báo cáo lệch kết quả thật | medium | high | Mọi bảng sinh bằng `reproduce.py`; test so sha256 |
| Công bố nhầm ảnh POG/VisDrone | low | high | Chỉ khung của camera này được commit; VisDrone/POG không bao giờ vào repo |
| Bundle trên Release lệch bản đã kiểm | low | medium | Bước 9 tải lại từ Release, validate và so `onnx_sha256` với `meta.json` trong git |

## Rollback

Revert merge PR #3; báo cáo cũ còn ở lịch sử git và `archive/`. Bundle trên Release sai: `gh release delete models-v<ver>
--cleanup-tag` rồi tạo lại.
