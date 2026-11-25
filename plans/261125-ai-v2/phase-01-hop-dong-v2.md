# Phase 01 — Hợp đồng v1.1 + nhánh tích hợp hai repo (serial)

Lane: serial
Depends-on: none
Effort: M
Repo: XLA (nhánh `feat/ai-v2` cắt từ `main`) + IOT-CV (chỉ tạo nhánh `feat/ai-v2` từ `origin/main`)
Owns: `docs/hop-dong-model.md`, `xla/contract.py`, `tests/contract/test_model_contract.py`, `plans/261125-ai-v2/contract.md` (chỉ để lại con trỏ)
Creates: `tests/contract/fixtures/` (bundle giả verifier, enhancer-offline nếu cần)
AX: AX-01 `python -m xla.contract validate|make-dummy --kind verifier|enhancer-offline --json`, mã thoát 0/1/2 như D7 v1
Accept: `pytest tests/contract/test_model_contract.py`

## Mục tiêu

Chốt phần thêm của hợp đồng **trước** mọi lane: chép `contract.md` §A–§F vào `docs/hop-dong-model.md` (mục mới), đặt
`CONTRACT_VERSION = "1.1"`, mở rộng validator và bundle giả để lane IOT-CV (08, 13, 18) code được ngay với bundle giả;
tạo nhánh tích hợp `feat/ai-v2` ở **cả hai** repo để mọi lane tách worktree từ đó.

## Đầu vào

`contract.md` (cạnh plan), `docs/hop-dong-model.md`, `xla/contract.py` (`CONTRACT_VERSION` :35, `META_*`/`METRICS_KEYS`
:54-72, `BUNDLE_DIR_RE` :82, `validate_bundle` :394), `tests/contract/test_model_contract.py` (:49 hiện ghim "1.0").

## Bước

1. XLA: `git fetch origin && git switch -c feat/ai-v2 origin/main`; IOT-CV: `git -C d:/Coding/IOT-CV fetch origin` rồi
   tạo `feat/ai-v2` từ `origin/main` **không** đổi HEAD của clone chính (`git branch feat/ai-v2 origin/main`), đẩy cả hai.
2. `docs/hop-dong-model.md`: thêm §9 verifier, §10 lọc track + sự kiện, §11 enhancer-offline, §12 classic/lai ONNX, §13 API
   offline IOT-CV, §14 CSV cả hệ + dữ liệu v2 (chép đúng `contract.md`); đầu file ghi `contract_version` 1.1 + lịch sử đổi.
3. `xla/contract.py`: `CONTRACT_VERSION = "1.1"`; `BUNDLE_DIR_RE` nhận `detector|enhancer|verifier|enhancer-offline`; khoá meta
   + IO theo kind (metrics offline tuỳ chọn); `recommended_track` tuỳ chọn (k ≤ n, nguyên dương); `make-dummy --kind verifier`
   (RoiAlign `output_half_pixel` + hằng số) và `--kind enhancer-offline` (`Resize` nearest ×2); `validate` kiểm golden theo §A/§C.
4. Test (sửa `:49` thành "1.1" và thêm): docs khai `contract_version` 1.1 và có đủ tiêu đề §9–§14; bundle giả mỗi kind qua
   `validate`; bundle v1.0 (`models/release/*-v1.0.0`) vẫn hợp lệ dưới 1.1; meta sai (`k > n`, `output_size` sai, thiếu
   `crop_px`) bị từ chối có lý do; tên thư mục sai (vd. `enhancer-v2.1.0-hybrid`) bị từ chối; parity RoiAlign ORT vs
   `torchvision.ops.roi_align(aligned=False)` trên 20 hộp ngẫu nhiên + N = 0, dung sai 1e-3.
5. Để lại một dòng con trỏ trong `plans/261125-ai-v2/contract.md`. Commit pathspec, đẩy `feat/ai-v2`, báo Owner để mở lane.

## Tiêu chí xong

Accept xanh (test đỏ nếu docs chưa chép hoặc `CONTRACT_VERSION` còn 1.0); `make-dummy --kind verifier` rồi `validate` ra 0;
bundle v1 thật vẫn `validate` ra 0; `feat/ai-v2` có trên remote của cả hai repo.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RoiAlign lệch chế độ giữa graph và PyTorch | low | medium | Chế độ ghi cứng trong §A; test parity bước 4 (red-team 25/11 đo max Δ 0,0 với RoiAlign thuần) |
| Đổi hợp đồng giữa chừng | low | high | Lane không sửa; Owner tăng phiên bản và re-brief (Contract) |

Rollback: revert commit trên `feat/ai-v2`; bundle v1 không bị ảnh hưởng (test bước 4).
