# Phase 02 — Dữ liệu ngoài có giấy phép, lọc nadir, làm xấu D2 v3 (lane data-ext)

Lane: data-ext
Depends-on: 01
Gate: none (dữ liệu đã tải đủ 25/11)
Status: DONE 25/11/2026 (lane data-ext đã gộp vào `feat/ai-v2`; kết quả `results/v2/data/README.md`)
Effort: L
Repo: XLA, worktree `D:\XLA-wt\data-ext`, nhánh `lane/data-ext` từ `feat/ai-v2`
Owns: `xla/degrade/sources.py`, `configs/sources.yaml`, `configs/view_external.yaml`, `xla/degrade/build.py` (chỉ thêm nguồn `external`), `tests/degrade/test_external_sources.py`
Creates: `tests/degrade/test_external_sources.py`, `results/v2/data/README.md`
AX: AX-01 `python -m xla.degrade.sources stats --json` có thêm khối `external`; `build --set d02x --workers 4 --json`
Accept: `pytest tests/degrade/test_external_sources.py`

Dữ liệu ngoài đã nằm sẵn trên đĩa (25/11/2026, Deferred 20–21 + 31 trong plan.md). SARD **bỏ**: IEEE DataPort đòi gói
thuê bao trả phí kể cả khi đã đăng nhập, không có bản sao công khai. Ba bộ dùng:

| Bộ | File (trong `D:\IOT_Tools\downloads\datasets\`) | Cỡ | Giấy phép | Nội dung đã kiểm |
|---|---|---|---|---|
| HERIDAL | `heridal\heridal_keras_retinanet_voc.zip` (Zenodo 5662351, md5 khớp) | 7,74 GB | CC BY 3.0 | đồi đá/bụi cây Croatia–BiH, 40–65 m; đếm 25/11: 1 684 ảnh JPG 4000×3000 (tên `train_*` 1 583, `test_*` 101), 1 650 file VOC XML, 3 235 hộp đều lớp `person` (ảnh không có XML = nền không người, dùng làm lớp âm); `ImageSets/Main/{train,val,trainval,test}.txt`; bỏ thư mục rác `__MACOSX/` |
| TinyPerson | `tinyperson\train.tar.gz`, `test.tar.gz`, `tiny_set_train.json`, `tiny_set_test.json` | 0,96 GB | dataset không ghi, mã MIT | json: 746 ảnh train, 786 ảnh test (COCO) |
| WiSARD Multi-Modal Sample | `wisard\WiSARD_Multi_Modal_Sample.zip` | 0,97 GB | MIT | thư mục `*_VIS_0003`: 264 ảnh RGB + nhãn YOLO txt; bỏ `*_IR_*` (nhiệt) |

Không có `params/`, `coco/`, `annotations/corner/` của TinyPerson (cố ý không tải).

## Mục tiêu

Thêm ảnh người nhìn từ drone có giấy phép rõ vào nguồn sạch, lọc gần nadir như VisDrone, làm xấu bằng D2
`params_version` 3, ra tập trộn `d02x` (D0 + D2, VisDrone + ngoài) cho phase 05/06/07. Cặp SR ×2 không thuộc phase này (phase 11 tự dựng).

## Đầu vào

`xla/degrade/sources.py` (`load_visdrone` :99, `is_nadir` :243, `cmd_fit_probe` :508, `cmd_seq_sheet` :554),
`configs/sources.yaml`, `configs/visdrone_view.yaml`, `xla/degrade/build.py` (:112-114, :273-274), `configs/degrade_d2.yaml`
(`params_version: 3`). Giấy phép: plan.md mục "Nguồn sự thật bên ngoài".

## Bước

1. Giấy phép trước khi tải: ghi từng bộ vào `configs/sources.yaml` (`license`, `url`, `retrieved`, `use: research`).
   HERIDAL CC BY 3.0 (Zenodo 5662351, bản keras-retinanet của nhóm AIR; ghi nguồn gốc Božić-Štulić et al., IJCV 2019);
   TinyPerson "không ghi giấy phép dataset, mã MIT" (Google Drive trong README `ucas-vg/PointTinyBenchmark`); WiSARD MIT
   (https://sites.google.com/uw.edu/wisard, Broyles et al., IROS 2022); SARD `unavailable: paid subscription` (Deferred 31).
   Không đẩy ảnh ngoài lên repo: chỉ manifest (đường dẫn tương đối + sha256) và mã đọc.
2. Giải nén vào `D:\DevCache\datasets\xla\external\<bộ>\` (zip ở `D:\IOT_Tools\downloads\datasets\`). WiSARD chỉ giải nén
   thư mục `*_VIS_*`. Kiểm sha256 từng file nén, ghi vào manifest.
3. `load_heridal` (VOC XML, lớp `person`; ảnh cỡ lớn → cắt ô 1280 có chồng mép trước khi làm xấu, giữ hộp bị cắt ≥ 50 %
   diện tích), `load_tinyperson` (COCO; `sea person` + `earth person` → `person`, `ignore`/`uncertain` → vùng bỏ qua như
   category 0 VisDrone; ảnh nằm trong `train.tar.gz`/`test.tar.gz`, đường dẫn theo `file_name` của json), `load_wisard`
   (YOLO txt chuẩn hoá, lớp nào cũng → `person`; kiểm lại tên lớp trong file `.txt`/`classes` của bộ). Đếm ảnh, hộp, cỡ hộp.
   Cả ba bộ đều là "dương" cho verifier; HERIDAL là nguồn nền khó (đá, bụi cây) chính cho detector.
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
| WiSARD sample chỉ 264 ảnh RGB, một chuyến bay | high | low | Chỉ góp vào verifier + một phần train detector; không chia val từ nó (một cảnh) |
| Ảnh HERIDAL rất lớn (4000×3000, đã đo) làm người co về < 2 px khi thu về 640×480 | high | high | Cắt ô trước rồi mới làm xấu (bước 3); đo phân bố cỡ hộp sau cắt, ghi README |
| CPU quá tải khi làm xấu hàng loạt | medium | medium | `--workers 4`, ghi theo lô, chạy tiếp được |

Rollback: xoá nhánh `lane/data-ext`; dữ liệu ở `D:\DevCache` không vào git.
