# Phase 17 — Báo cáo, tái lập, slide, Release `models-v2.0.0`, PR cuối (serial)

Lane: serial
Depends-on: 09, 14, 15, 16
Effort: M
Repo: XLA (`feat/ai-v2`)
Owns: `scripts/reproduce.py`, `tests/test_reproduce.py`, `BAO_CAO_DO_AN.md` (chỉ khối `<!-- bang: -->` + mục v2), `README.md`, `docs/slide/index.html`, `results/v2/README.md`, `plans/261125-ai-v2/plan.md` (Forecast re-score, trạng thái)
AX: AX-01 `python scripts/reproduce.py --write|--check --json` thêm bảng t7–t9
Accept: `pytest tests/test_reproduce.py`

## Mục tiêu

Một lệnh dựng lại mọi bảng/hình v2, báo cáo và slide khớp số mới, bundle đính Release, PR cuối vào `main`. Không đụng Google
Drive, không đụng `docs/nop-bai*` (việc nộp bài riêng, ngoài phạm vi).

## Đầu vào

`results/v2/{data,teacher,detect,verifier,system,classic,sr,vsr,iotcv}/`, `results/ab-v2*/`, `scripts/reproduce.py` (`build` :302,
`sync_report` :323, `check` :354), `tests/test_reproduce.py`, `BAO_CAO_DO_AN.md`, `docs/slide/index.html`, `README.md`.

## Bước

1. `reproduce.py`: bảng t7 (nhánh B ba tầng: test mAP50, recall, báo nhầm holdout, CI), t8 (A/B v2), t9 (ms + giờ offline/chuyến);
   hình f4 (đường cong báo nhầm–recall); `--check` chấm lại CSV đã lưu, so sha256.
2. `tests/test_reproduce.py` thêm: t7–t9 và f4 có trong `results/reproduce/manifest.json`; mỗi bảng có khối `<!-- bang: t7… -->`
   tương ứng trong `BAO_CAO_DO_AN.md` và nội dung khối bằng file bảng (test đỏ nếu v2 chưa được thêm).
3. `--write` → `results/reproduce/` + `sync_report` vào `BAO_CAO_DO_AN.md`; thêm mục v2 (dữ liệu ngoài + giấy phép "chỉ nghiên
   cứu", luật chọn k-fold, chia mine/cal/holdout theo phiên, kết quả đạt/không đạt kèm CI, giới hạn: 74 khung có người, vật dễ
   nhầm có thể lặp giữa phiên, ảnh offline "do AI dựng lại").
4. `docs/slide/index.html`: cập nhật số v2 (chỉ khi số đổi), mở bằng trình duyệt kiểm hiển thị.
5. Release `models-v2.0.0` trên GitHub XLA (D12, Deferred 19): đính các `.onnx` ship, ghi sha256; `meta.json` + `golden/` trong git.
6. Forecast plan.md: chấm lại từng dòng bằng số đo; `results/v2/README.md` tổng hợp.
7. PR XLA `feat/ai-v2` → `main` (nhánh A + báo cáo), merge khi CI xanh. Về `main`, `git pull --ff-only`, xoá nhánh lane đã gộp.

## Tiêu chí xong

Accept xanh (gồm các test mới bước 2); `reproduce.py --check` ra 0; PR merged; Release có đủ sha256 khớp meta.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Báo cáo nói quá số đo | medium | high | Mọi số kèm CI; `--check` tái lập; dòng không đạt ghi rõ |
| Slide lệch số báo cáo | medium | low | Số lấy từ `results/reproduce/tables`, không gõ tay |

Rollback: revert PR; xoá Release `models-v2.0.0` (bundle v1 vẫn ở `models-v1.0.0`).
