# Phase 17: IOT-CV — `plans/ai`, `ml/` trỏ về XLA (không xoá)

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 02 | S (~0,5 ngày agent) | Không |

Lane: iotcv-docs
Depends-on: 02
Effort: S
Owns: `d:/Coding/IOT-CV/plans/ai/ai-phase-01-chuan-bi.md`, `d:/Coding/IOT-CV/plans/ai/ai-phase-02-huan-luyen-thi-nghiem.md`, `d:/Coding/IOT-CV/plans/ai/ai-phase-03-du-lieu-that.md`, `d:/Coding/IOT-CV/plans/ai/ai-phase-04-bao-cao-xla.md`, `d:/Coding/IOT-CV/plans/_phase-index.md`, `d:/Coding/IOT-CV/plans/QUYET-DINH-CHUA-CHOT.md`, `d:/Coding/IOT-CV/ml/pyproject.toml`, `d:/Coding/IOT-CV/AGENTS.md`
Creates: `d:/Coding/IOT-CV/ml/README.md`, `results/iotcv_refs.json`, `tests/integration/test_iotcv_refs.py`
AX: n/a because this phase edits Markdown pointers and adds one test, no command-line tool
Accept: `pytest tests/integration/test_iotcv_refs.py`

## Mục tiêu

Ai mở `plans/ai/` hay `ml/` trong IOT-CV đều được dẫn tới XLA (nơi nghiên cứu) và hợp đồng model, không làm theo thiết kế
cũ viết cho OV2640. Không xoá file nào (xoá là human step tuỳ chọn, Deferred 17). IOT-CV chỉ ghi file, không commit (D13).

## Đầu vào cần có

- Thiết kế §6 ("`plans/ai` và `ml/` cũ đổi sang trỏ về XLA; kiểm tham chiếu trước khi xoá; khoảng 40 file đang nhắc tới chúng").
  Lúc lập plan: 57 file trong IOT-CV khớp `ml/` hoặc `plans/ai` (grep `*.md,*.py,*.ts,*.tsx,*.yaml,*.toml,*.json`, bỏ
  `node_modules`, `.venv`) — con số gồm cả tham chiếu hợp lệ.
- `plan.md` cạnh file này (đường dẫn plan XLA), `D:\XLA\docs\hop-dong-model.md`.
- `d:\Coding\IOT-CV\AGENTS.md`: lúc lập plan có thay đổi chưa commit của chính phiên brainstorm 18/11/2026 (luật 5 cho spawn
  sub-agent, luật 10 phần AI ở XLA, hàng "Nguồn sự thật" trỏ `D:\XLA\plans\`), chờ chủ dự án bảo commit. Giữ nguyên các thay
  đổi đó; phiên khác đang sửa thì phối hợp trước (D13).

## Việc theo thứ tự

1. `git -C d:\Coding\IOT-CV status --short`; file trong Owns đang sửa dở → nhắn phiên đang sống, chờ chia.
2. Liệt kê mọi tham chiếu tới `ml/` và `plans/ai` trong IOT-CV, phân loại: (a) trỏ vào thiết kế/script AI cũ → cần con trỏ;
   (b) tham chiếu hợp lệ còn đúng (vd. `ml/datasets` trong `ruff extend-exclude`) → giữ; ghi `results/iotcv_refs.json`
   (file, dòng, loại, việc đã làm).
3. Đầu mỗi `plans/ai/ai-phase-0*.md`: khối "Đã thay thế (18/11/2026)": phần AI nay ở repo XLA theo plan
   `plans/261118-ai-anh-chat-luong-thap/`; camera thật là OV3660 không phải OV2640; hợp đồng model ở `docs/hop-dong-model.md`
   của XLA; phần còn đúng của file này (VisDrone §2.1–2.3, Label Studio A1.7, SAHI A2.3, ONNX A2.4) đã được dẫn trong plan XLA.
   Không sửa thân file.
4. `ml/README.md` mới: `ml/` là khung sườn cũ, không dùng; script và cấu hình thật nằm ở XLA; IOT-CV chỉ nhận bundle ONNX qua
   `scripts/vision/lay_model.py` (phase 15). `ml/pyproject.toml`: một dòng chú thích đầu file trỏ README.
5. `plans/_phase-index.md` (các dòng AI Phase), `plans/QUYET-DINH-CHUA-CHOT.md` (mục đề tài Xử lý ảnh → đã chốt theo thiết kế
   18/11/2026, trỏ file thiết kế), `AGENTS.md` bảng "Nguồn sự thật": hàng "Phần AI xử lý ảnh chất lượng thấp" đã có từ
   18/11/2026, **sửa đúng hàng đó** (thêm hợp đồng `D:\XLA\docs\hop-dong-model.md` và chỗ đặt bundle `models/vision/`), không
   thêm hàng thứ hai cùng chủ đề. Giữ nguyên mọi khối phiên khác viết.
6. Danh sách các file khung sườn trong `ml/scripts/` (đều `NotImplementedError`) kèm số tham chiếu còn lại → báo lead cho human
   step "Duyệt xoá" (tuỳ chọn). Không xoá.
7. **Test XLA** `tests/integration/test_iotcv_refs.py`: mỗi file `plans/ai/ai-phase-0*.md` có khối "Đã thay thế" ở 15 dòng đầu;
   `ml/README.md` tồn tại và nhắc XLA; mọi tham chiếu loại (a) trong `iotcv_refs.json` nằm trong file đã có con trỏ; `ml/scripts/`
   vẫn còn đủ 6 file (chưa ai xoá khi chưa được duyệt) hoặc manifest ghi ngày chủ dự án duyệt xoá. Thiếu IOT-CV → fail.
8. Không commit IOT-CV; commit phần XLA trên `lane/iotcv-docs`, đẩy, báo lead kèm danh sách file IOT-CV đã sửa.

## Cổng pass

- Accept exit 0; `git -C d:\Coding\IOT-CV diff --stat` chỉ gồm file trong Owns/Creates.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Đè thay đổi chưa commit của phiên khác trong `AGENTS.md` | medium | high | Bước 1; sửa đúng một hàng bảng, giữ nguyên phần còn lại |
| Con trỏ làm hỏng liên kết cũ | low | low | Chỉ thêm khối, không đổi tên/đường dẫn file |

## Rollback

`git -C d:\Coding\IOT-CV checkout -- <file>` (sau khi chủ dự án đồng ý), xoá `ml/README.md`.
