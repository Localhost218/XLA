# Phase 14 — Hàng rào báo nhầm trên ảnh offline + thử mù A/B v2 (serial)

Lane: serial
Depends-on: 06, 10, 11, 12
Gate: human
Effort: M
Repo: XLA (`feat/ai-v2`, sau khi gộp lane classic, sr)
Owns: `xla/eval/ab_page.py` (nhiều so sánh, ảnh 1280×960), `xla/eval/ab_score.py` (tiêu chí D13), `xla/eval/guard_fp.py`, `models/release/enhancer-offline-v1.*/meta.json` (chỉ `metrics`), `results/ab-v2/**`, `results/ab-v2-zoom/**`, `tests/eval/test_ab_v2.py`
Creates: `xla/eval/guard_fp.py`, `tests/eval/test_ab_v2.py`
AX: AX-01 `python -m xla.eval.guard_fp --bundle <dir> --json`; `ab_page build --comparisons ... --json`; `ab_score ... --json`
Accept: `pytest tests/eval/test_ab_v2.py`

Cổng người: bước 4 chờ chủ dự án chấm hai lượt (human step "Chấm thử mù A/B v2", plan.md).

## Mục tiêu

Chấm nhánh A đúng thiết kế: hàng rào "mạng sinh không bịa người" trước (quyết định trên `cal`, holdout chỉ báo cáo), rồi A/B
mù (cả khung + phóng to) so với gốc và với classic, bằng đúng bundle sẽ ship, trên khung **dev** (không dùng khung test).

## Đầu vào

`enhancer-v2.0.0` (+ `enhancer-v2.1.0` lai nếu qua lọc phase 10), ứng viên offline duy nhất (`enhancer-offline-v1.0.0`, hoặc
v1.1.0 nếu phase 12 thay), detector v2 (hoặc v1 nếu Kill (b)), `data/real/neg_flight_split.json` (`cal`, `holdout` đã duyệt),
`xla/eval/ab_page.py` (`build` :175, `plan_pairs` :84, `pick_region` :135), `xla/eval/ab_score.py` (`wilson` :37, `score` :60), D13.

## Bước

1. `guard_fp`: detector trên `cal` gốc và `cal` sau offline (hạ ×½ về 640×480, cùng ngưỡng): báo nhầm/khung + CI bootstrap theo
   phiên/khối. Offline tăng báo nhầm (cận dưới CI hiệu > 0) → không vào A/B, không ship (Kill (d)); không có ứng viên dự phòng.
   Cùng phép đo trên holdout chỉ **ghi** vào meta + README (`fp_holdout_*`), không quyết định gì.
2. `ab_page` v2: một trang, các so sánh D13 xen kẽ; 25 cặp không lặp mỗi so sánh + 5 cặp lặp; khung **dev** phân tầng
   (phiên × bin độ cao × có người); hiển thị 1280×960 theo đúng cách web hiển thị (ảnh 640×480 phóng bằng trình duyệt như trang
   Lịch sử bay, không Lanczos riêng); trái/phải ngẫu nhiên seed 20261125; tên file hex, khoá ở `ab_key.json`. Lượt hai `--zoom 2`.
   Kiểm trang bằng Chrome headless (đủ cặp, không lộ tên biến thể trong HTML/URL).
3. Mở sẵn `results/ab-v2/ab-test.html` cho chủ dự án.
4. **Cổng người:** chủ dự án chấm hai lượt, bấm Xuất; agent chép JSON vào `results/ab-v2*/ab_result.json`.
5. `ab_score`: tỉ lệ thắng + Wilson 95 % từng so sánh, độ nhất quán cặp lặp; đạt = cận dưới > 0,5 (≥ 18/25). Ghi
   `results/ab-v2/README.md` + `results/ab-v2/decision.json` (máy đọc được): offline ship? lai thay classic? (theo thiết kế:
   lai chỉ giữ khi thắng classic); kết quả classic vs gốc là **chỉ số §5**, không đổi quyết định mặc định của Owner (Deferred 17).
   Giới hạn ghi rõ: ảnh SR ×2 không có lưới khối JPEG nên có thể nhận ra được.

## Tiêu chí xong

Accept xanh: trang không lộ khoá, đủ cặp mỗi so sánh, cặp lặp đổi bên, chỉ dùng khung dev; `ab_score` đúng trên kết quả giả
(17/25 không đạt, 18/25 đạt); `guard_fp` từ chối bundle tăng báo nhầm trên cal (số giả) và không đọc holdout khi quyết định;
`decision.json` đúng schema.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 25 cặp không đủ lực thống kê | high | medium | Báo CI, không nói quá; tiêu chí cố định trước khi chấm |
| Người chấm nhận ra ảnh SR (không có lưới khối) | medium | medium | Cùng cỡ, cùng cách phóng như web; ghi giới hạn; không tuyên bố quá |
| Hàng rào loại offline | medium | medium | Kill (d); web hiện 503 "không có" |

Rollback: không đổi gì trong IOT-CV từ phase này; quyết định ship áp ở phase 15.
