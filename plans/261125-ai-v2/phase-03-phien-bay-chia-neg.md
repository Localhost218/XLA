# Phase 03 — Mọi bản ghi, chỉ mục phiên bay, chia `neg_flight` theo phiên (lane realdata)

Lane: realdata
Depends-on: 01
Gate: human
Effort: M
Repo: XLA, worktree `D:\XLA-wt\realdata`, nhánh `lane/realdata` từ `feat/ai-v2`
Owns: `xla/camera/frames.py` (thêm trường `session`, nhập mọi bản ghi), `xla/camera/sessions.py`, `data/real/sessions.json`, `data/real/neg_flight_split.json`, `results/v2/neg-holdout/**`, `tests/camera/test_sessions_neg_split.py`
Creates: `xla/camera/sessions.py`, `tests/camera/test_sessions_neg_split.py`
AX: AX-01 `python -m xla.camera.sessions index|split|review|apply-review --json`, mã thoát 0/1/2
Accept: `pytest tests/camera/test_sessions_neg_split.py`

Cổng người: bước 6 chờ chủ dự án duyệt holdout (human step "Duyệt tập neg_flight giữ kín", plan.md).

## Mục tiêu

Làm cho "chia theo phiên bay" có nghĩa: dev/test v1 đều từ **một** bản ghi `20261102-100941` (đếm từ `data/real/split.json`),
nên cần chỉ mục phiên trên mọi bản ghi (`logs/recordings/20261101-*`, `20261102-*`, bản ghi mới nếu có) rồi chia khung bay
không người thành ba phần theo `contract.md` §F: `mine` (huấn luyện: ảnh nền, lớp âm), `cal` (chỉ hiệu chỉnh ngưỡng, k/n,
hàng rào offline) và `holdout` (chỉ chấm B2).

## Đầu vào

`xla/camera/frames.py` (`FlightWindow` :69, `frame_name` :77, `parse_flight_events` :89, `extract_flight` :188, `extract_neg`
:229), IOT-CV `python -m backend.flights --events` (subprocess, bỏ `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` khỏi env con),
`data/real/split.json` (chỉ đọc), `xla/detect/prelabel.py` (`run` :169), `xla/detect/review.py` (`cmd_crop` :173).

## Bước

1. `sessions index`: chạy `backend.flights --events` trên mọi bản ghi, ghi `data/real/sessions.json` (id `<rec>#<i>`).
   Đếm phiên, khung, thời lượng, điều kiện (giờ, độ sáng trung vị); ghi `results/v2/neg-holdout/README.md` (Owner chép số
   vào Forecast ở phase 09; lane không sửa `plan.md`).
2. Nhập khung của mọi phiên (giữ JPEG gốc, tên D11), bỏ khung trong cửa sổ cất/hạ cánh như v1.
3. Gán phiên (seed 20261125): phiên chứa khung dev/test v1 → không bao giờ `holdout`/`cal`; phiên còn lại chia `holdout`
   (ưu tiên ≥ 2 phiên, ≥ 300 khung), `cal` (≥ 1 phiên), phần còn lại `mine`; khung không người của các khối dev v1 vào `mine`.
   Không đủ phiên cho `cal` → `cal` = các khối 30 s xen kẽ trong phiên `mine` (khối nguyên, ghi rõ `cal_mode: "block"`).
4. Khung "không người": gán nhãn trước (RT-DETR-x + YOLO11x + SAHI, ngưỡng thấp 0,05, khoá GPU D6) trên khung mine/cal/holdout;
   khung có ứng viên → agent soát bằng ảnh cắt (`cmd_crop`); khung có người thật bị loại khỏi neg.
5. `sessions review`: trang `results/v2/neg-holdout/review/index.html` (ảnh ghép 4×4, ô "có người", nút Xuất JSON, trạng thái
   `localStorage`) cho **mọi** khung holdout; mở sẵn cho chủ dự án.
6. **Cổng người:** chủ dự án duyệt; `sessions apply-review <json>` đặt `approved`, loại khung bị đánh dấu.
7. Kill (a): holdout đã duyệt < 150 khung → ghi "không đủ chứng minh" vào README, vẫn tiếp tục.

## Tiêu chí xong

Accept xanh: (i) khung holdout/cal thuộc phiên (hoặc khối, khi `cal_mode: block`) không có khung dev/test; (ii) không khung
holdout/cal nào trong bất kỳ manifest huấn luyện / khai thác nào ở `data/`, `results/`, `D:\DevCache\datasets\xla` (quét
manifest); (iii) `split.json` sha256 bằng giá trị ghi trong `contract.md` §F; (iv) bản dữ liệu giả `tests/camera/fixtures/`
để chạy trong CI CPU.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bản ghi 01/11 tối/khác phân bố | medium | medium | Ghi điều kiện từng phiên; báo B2 theo phiên |
| Cùng một vật dễ nhầm (túi rác) có mặt ở phiên mine và holdout cùng ngày | medium | medium | Báo B2 theo phiên và theo ngày; ghi giới hạn trong báo cáo |
| Nhãn trước bỏ sót người nhỏ → neg bẩn | medium | high | Ngưỡng thấp + SAHI + agent soát + chủ dự án duyệt holdout |
| Rò holdout | low | high | Test (ii) trong CI; D5; danh sách cho phép ở `train.py` (phase 05) |

Rollback: revert nhánh lane; `split.json` v1 không đổi nên kết quả v1 tái lập được.
