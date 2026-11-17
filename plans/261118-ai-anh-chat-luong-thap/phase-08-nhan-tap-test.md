# Phase 08: Nhãn tập test thật — máy gán trước, agent soát, chủ dự án duyệt

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 03 | L (~2 ngày agent + 20–30 phút chủ dự án), ~1–2 giờ GPU gán trước (chưa xác minh) | Không |

Lane: labels
Depends-on: 03
Gate: human
Effort: L
Creates: `xla/detect/prelabel.py`, `xla/detect/labelstudio.py`, `configs/labelstudio.xml`, `data/real/labels/manifest.json`, `results/labels/review_log.csv`, `tests/labels/test_testset_labels.py`
AX: `python -m xla.detect.prelabel --split test --json` gán trước + xuất JSON Label Studio; `python -m xla.detect.labelstudio export --json` chuyển nhãn đã duyệt sang YOLO txt, exit 1 khi còn khung chưa duyệt
Accept: `pytest tests/labels/test_testset_labels.py`

## Mục tiêu

Có nhãn người đáng tin cho **toàn bộ khung test** (300–400) và một tập **dev-val** ~100 khung (để chọn checkpoint, chọn
ngưỡng; D10). Nhãn test qua ba lớp: nhiều model khác họ YOLO gán trước → agent soát từng khung → chủ dự án duyệt lần cuối
trên Label Studio chạy local. Nhãn dev-val chỉ qua hai lớp đầu.

## Đầu vào cần có

- `data/real/split.json`, `data/real/frames/` (phase 03); hợp đồng §7 (định dạng nhãn).
- Thiết kế §5.6, §7 (nhãn máy gán thiên về model gán → dùng model khác họ).
- Prior art: `d:\Coding\IOT-CV\plans\ai\ai-phase-01-chuan-bi.md` A1.7 (Label Studio tự host, `label_config.xml`, nạp
  ảnh từ ổ đĩa), A2.3 SAHI (`plans\ai\ai-phase-02-huan-luyen-thi-nghiem.md:213`).
- `plan.md` Deferred 9; Human steps hàng 1.

## Việc theo thứ tự

1. **Chọn dev-val:** 100 khung dev, cách nhau ≥ 2 s, phân tầng như test; ghi danh sách vào `data/real/labels/manifest.json`
   (không sửa `split.json` — file đó chỉ đọc).
2. **Gán trước** (`prelabel.py`, trong `xla.gpu_lock`): RT-DETR-x COCO + SAHI (lát 640, chồng 0,2) làm chính; YOLO11x @1280
   + SAHI làm phiếu thứ hai; hợp bằng WBF (`ensemble-boxes`); chỉ lớp người; bỏ hộp nằm trọn trong 16 px mép trái; ghi
   nguồn từng hộp (cả hai / chỉ một model). Trọng số tải về `D:\IOT_Tools\downloads\yolo-weights\`.
3. **Agent soát từng khung:** dựng ảnh khung + hộp (phóng 2×, đánh số) và tự xem; sửa: bỏ hộp sai (đốm hồng/trắng, dải mép,
   bóng), thêm người sót, siết hộp; đặc biệt soát hộp "chỉ một model". Ghi mỗi sửa vào `results/labels/review_log.csv`
   (khung, hành động, lý do). Khung nhóm người / mờ: thêm cờ `hard` để báo cáo tách riêng.
4. **Label Studio local:** `uv tool install label-studio` (với `UV_TOOL_DIR=D:\DevCache\uv-tools`), dữ liệu
   `D:\DevCache\label-studio`, phục vụ ảnh từ `D:\XLA\data\real\frames` (local files serving), chỉ nghe localhost (kiểm bằng
   `netstat -ano` không có `0.0.0.0:8080`). Tạo project `xla-test-v1` với `configs/labelstudio.xml` (một nhãn `person`,
   RectangleLabels), nạp nhãn đã soát làm predictions. Mở sẵn đúng trang project cho chủ dự án.
5. **Bước người (Gate: human):** chủ dự án duyệt hết khung test (~20–30 phút), bấm Submit; agent chờ trong lượt, không làm
   phase khác của lane này. Xong: `labelstudio export` → `data/real/labels/test/*.txt`, dev-val → `data/real/labels/devval/*.txt`,
   manifest ghi `approved_by_owner: true`, ngày, số khung, số hộp, phiên bản model gán, số hộp chủ dự án sửa.
6. **Đối chiếu:** số hộp chủ dự án sửa / tổng; > 15 % thì ghi vào báo cáo là chất lượng soát của agent chưa đủ.
7. **Test** `tests/labels/test_testset_labels.py` (marker `data`): mọi khung test có file nhãn (rỗng = không người); manifest
   `approved_by_owner` true; toạ độ trong (0,1]; không hộp nào nằm trọn trong dải mép; dev-val ∩ test = ∅; dev-val ⊂ dev.
8. Commit nhãn + manifest + log trên `lane/labels` (ảnh khung đã commit ở phase 03, Deferred 14), đẩy, báo lead.

## Cổng pass

- Accept exit 0; manifest có `approved_by_owner: true`.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Nhãn thiên về model gán | medium | high | Model gán khác họ model được chấm; chủ dự án duyệt; báo cáo khai |
| Label Studio mở ra mạng ngoài | low | high | Chỉ localhost, kiểm `netstat` trước khi mở cho chủ dự án |
| Chủ dự án chưa rảnh duyệt | medium | medium | Phase 08 không nằm trên đường găng (chỉ phụ thuộc 03); phase 09–10 chờ |

## Rollback

Xoá `data/real/labels/` và project Label Studio; dữ liệu `split.json` không đổi.
