# Phase 02 — Dữ liệu ngoài có giấy phép, lọc nadir, làm xấu D2 v3 (lane data-ext)

Lane: data-ext
Depends-on: 01
Gate: human
Status: BLOCKED
Effort: L
Repo: XLA, worktree `D:\XLA-wt\data-ext`, nhánh `lane/data-ext` từ `feat/ai-v2`
Owns: `xla/degrade/sources.py`, `configs/sources.yaml`, `configs/view_external.yaml`, `xla/degrade/build.py` (chỉ thêm nguồn `external`), `tests/degrade/test_external_sources.py`
Creates: `tests/degrade/test_external_sources.py`, `results/v2/data/README.md`
AX: AX-01 `python -m xla.degrade.sources stats --json` có thêm khối `external`; `build --set d02x --workers 4 --json`
Accept: `pytest tests/degrade/test_external_sources.py`

BLOCKED tới khi xong human step "Tải SARD" và "Tải TinyPerson" (plan.md). Phần VisDrone + HERIDAL (nếu tải được) làm trước.

## Mục tiêu

Thêm ảnh người nhìn từ drone có giấy phép rõ vào nguồn sạch, lọc gần nadir như VisDrone, làm xấu bằng D2
`params_version` 3, ra tập trộn `d02x` (D0 + D2, VisDrone + ngoài) cho phase 05/06/07. Cặp SR ×2 không thuộc phase này (phase 11 tự dựng).

## Đầu vào

`xla/degrade/sources.py` (`load_visdrone` :99, `is_nadir` :243, `cmd_fit_probe` :508, `cmd_seq_sheet` :554),
`configs/sources.yaml`, `configs/visdrone_view.yaml`, `xla/degrade/build.py` (:112-114, :273-274), `configs/degrade_d2.yaml`
(`params_version: 3`). Giấy phép: plan.md mục "Nguồn sự thật bên ngoài".

## Bước

1. Giấy phép trước khi tải: ghi từng bộ vào `configs/sources.yaml` (`license`, `url`, `retrieved`, `use: research`).
   SARD CC BY 4.0; TinyPerson "không ghi giấy phép dataset, mã MIT"; HERIDAL: thử trang chính và một bản sao công khai
   (Kaggle/GitHub), không tải được thì ghi `unavailable` và bỏ (Deferred 20). Không đẩy ảnh ngoài lên repo: chỉ manifest
   (đường dẫn tương đối + sha256) và mã tải.
2. Giải nén vào `D:\DevCache\datasets\xla\external\<bộ>\` (zip ở `D:\IOT_Tools\downloads\datasets\`).
3. `load_sard` (LabelImg/VOC, mọi tư thế → `person`), `load_tinyperson` (COCO; `sea person` + `earth person` → `person`,
   `ignore`/`uncertain` → vùng bỏ qua như category 0 VisDrone), `load_heridal` nếu có. Đếm ảnh, hộp, cỡ hộp.
4. Lọc góc nhìn: nhãn góc theo chuỗi/thư mục khi có (`configs/view_external.yaml`), còn lại probe CLIP đã fit (`cmd_fit_probe`)
   ngưỡng 0,8 như VisDrone; soát tay 100 ảnh mỗi bộ bằng `cmd_seq_sheet`, ghi precision của bộ lọc.
5. Dựng `d02x` bằng `build.py` (`--workers 4`), cùng hình học 624×480 và D2 v3; chia train/val theo chuỗi/cảnh, không theo ảnh.
6. `results/v2/data/README.md`: bảng số ảnh/hộp trước–sau lọc từng bộ, giấy phép, precision bộ lọc (Owner chép số vào Forecast ở phase 09; lane không sửa `plan.md`).

## Tiêu chí xong

Accept xanh (test: parser từng bộ trên mẫu nhỏ trong `tests/degrade/fixtures/`, map lớp đúng, vùng bỏ qua, không ảnh
ngoài nào trong repo, manifest sha256 khớp, `d02x` có cả nguồn ngoài, chia theo chuỗi không rò). Tập `d02x` có ≥ 1 bộ
ngoài sau lọc.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ảnh ngoài hầu hết nhìn xiên (TinyPerson bãi biển) | high | medium | Bước 4 đếm trước; bộ nào < 200 ảnh nadir thì chỉ dùng cho verifier (lớp dương), ghi Deferred |
| Tải SARD trễ (cần đăng nhập) | medium | medium | Human step "now"; VisDrone + HERIDAL chạy trước; phase 05 có thể bắt đầu với bộ có sẵn khi Owner đồng ý |
| CPU quá tải khi làm xấu hàng loạt | medium | medium | `--workers 4`, ghi theo lô, chạy tiếp được |

Rollback: xoá nhánh `lane/data-ext`; dữ liệu ở `D:\DevCache` không vào git.
