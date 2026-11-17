# Phase 02: Hợp đồng model ONNX + hồ sơ camera — file, hằng số, test, bundle giả

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| chưa bắt đầu | phase 01 | S (~0,5 ngày agent), 0 giờ GPU | Không |

Lane: serial
Depends-on: 01
Effort: S
Owns: `plans/261118-ai-anh-chat-luong-thap/contract.md`
Creates: `docs/hop-dong-model.md`, `xla/contract.py`, `configs/camera_profile.schema.json`, `tests/contract/test_model_contract.py`
AX: `python -m xla.contract validate <bundle-dir> --json` in lỗi theo từng mục hợp đồng, exit 0/1/2; `make-dummy <out-dir>` sinh bundle giả cho lane IOT-CV
Accept: `pytest tests/contract/test_model_contract.py`

## Mục tiêu

Chốt ranh giới XLA ↔ IOT-CV **trước** khi lane nào chạy: hợp đồng thành file SSOT trong repo, có hằng số trong code, có
bộ kiểm bundle chạy được, có bundle giả đúng hợp đồng để lane IOT-CV (phase 15) làm song song mà không chờ model thật.
Mở PR số 1 (D12) để hợp đồng vào `main`.

## Đầu vào cần có

- `contract.md` cạnh plan (bản đã chốt, `contract_version` 1.0) — đọc hết.
- `plan.md` Decision register D1–D6, D11.
- Phase 01 xong (package `xla/`, môi trường có `onnx`, `onnxruntime`, `jsonschema`).

## Việc theo thứ tự

1. `git mv plans/261118-ai-anh-chat-luong-thap/contract.md docs/hop-dong-model.md`; tạo lại `contract.md` ở chỗ cũ chỉ một
   dòng: "Đã chuyển: `docs/hop-dong-model.md` là SSOT." Không sửa nội dung hợp đồng (muốn sửa: dừng, báo Owner).
2. `xla/contract.py`: hằng số từ hợp đồng (`CONTRACT_VERSION = "1.0"`, `FRAME_W = 640`, `FRAME_H = 480`, tên/dtype/shape
   input-output §2–3, khoá `meta.json` §4, dung sai golden §4); hàm `validate_bundle(dir) -> list[str]` (rỗng = đạt) kiểm:
   tên file, sha256 file = `meta.onnx_sha256`, khoá meta bắt buộc + kiểu, `contract_version` major, chữ ký I/O của graph
   (đọc bằng `onnx.load`, không chạy), golden có đủ 3 khung + `expected.json`, và chạy ORT CPU trên golden khớp
   `expected.json` trong dung sai; `make_dummy_bundle(out_dir, model)` dựng graph nhỏ bằng `onnx.helper`: detector giả
   nhận `frame` uint8 [480,640,3] và trả 1 hộp cố định `[100,100,150,160]` điểm 0,9; enhancer giả trả lại chính khung với
   cột `[0,16)` lặp cột 16; golden giả là 3 khung xám có hoạ tiết. CLI theo D7: `validate`, `make-dummy`, `--json`, exit 0/1/2.
3. `configs/camera_profile.schema.json` (JSON Schema draft 2020-12) đúng §6 hợp đồng: trường bắt buộc, kiểu, đơn vị theo
   tên trường, mỗi phân bố có `p5 ≤ p50 ≤ p95`, `qtable_*` đúng 64 số nguyên 1–255, `subsampling` ∈ {"4:2:0","4:2:2","4:4:4"}.
4. `tests/contract/test_model_contract.py`: bundle giả qua `validate_bundle` rỗng lỗi; năm bundle hỏng cố ý (sai sha256, sai
   tên input, sai dtype output, thiếu khoá meta, major `contract_version` = 2) mỗi cái báo đúng một lỗi đúng mục; schema
   nhận một hồ sơ mẫu tối thiểu hợp lệ và từ chối hồ sơ thiếu `jpeg.qtable_luma`; test chạy CPU, không cần dữ liệu.
5. Commit pathspec, đẩy nhánh, mở **PR #1** `feat/ai-anh-chat-luong-thap` → `main` (phase 01 + 02). CI xanh rồi merge
   (repo của chủ dự án, được toàn quyền). Sau merge, các lane cắt worktree từ nhánh tích hợp (D12).

## Cổng pass

- Accept exit 0.
- `python -m xla.contract make-dummy D:\DevCache\tmp\dummy && python -m xla.contract validate D:\DevCache\tmp\dummy\detector-v0.0.0`
  in `[]` (chạy hai lệnh riêng).
- PR #1 merge, CI xanh.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Graph giả không phản ánh được NMS/toạ độ thật | medium | low | Bundle giả chỉ để nối dây; tính đúng thật kiểm ở phase 11/13 bằng golden thật |
| Lane sau muốn đổi hợp đồng giữa chừng | medium | high | Lane dừng và báo; chỉ Owner sửa, tăng version, báo lại mọi lane |

## Rollback

Revert commit của phase; hợp đồng cũ nằm nguyên trong lịch sử git.
