# `data/real/labels/devpool/` — tập nhãn dev mở rộng + fold theo phiên (AI v2 phase 04)

Mã: `xla/labels/devpool.py`, test `tests/labels/test_dev_folds.py`. Hợp đồng: `docs/hop-dong-model.md` §14.2.
Seed 20261125. Ngày làm 25/11/2026. Nhãn do agent soát (phase 04 không cần chủ dự án duyệt).

```bash
$PY -m xla.labels.devpool select --json     # -> devpool.json (chọn khung)
$PY -m xla.labels.devpool prelabel --json   # -> review/prelabel.json (hộp hợp hai model; GPU chỉ cho khung chưa có)
$PY -m xla.labels.devpool sheets --scale 1 --cols 2 --rows 2 --json   # ảnh soát ra cache (không commit)
$PY -m xla.labels.devpool apply --json      # review/edits.csv -> <khung>.txt, review/review_log.csv, review/labels.json
$PY -m xla.labels.devpool folds --json      # -> ../folds.json + fold_info.json
```

## Tệp

| Tệp | Nội dung |
|---|---|
| `devpool.json` | `{version, seed, min_gap_s, frames, sessions, report}`: 160 khung, báo cáo chọn (bảng sức chứa, tầng, dev-val bị loại) |
| `<khung>.txt` | nhãn YOLO 640×480, một lớp `0` (người); khung không người là file rỗng |
| `review/prelabel.json` | hộp gán trước (RT-DETR-x + YOLO11x + SAHI, conf 0,05, hợp WBF), lấy từ `results/v2/neg-holdout/prelabel/raw.json` (không chạy GPU thêm) |
| `review/edits.csv` | dòng soát của agent: `frame, action (ok/drop/tighten/add/hard), box_id, x1..y2 px gốc, reason` |
| `review/review_log.csv` | edits + nguồn hộp gán trước (rtdetr/yolo/both) |
| `review/labels.json` | `{hard, boxes}` hộp cuối mọi khung (px gốc) |
| `fold_info.json` | người mỗi phiên, các k đã thử, phiên mỗi fold, số khung / khung có người mỗi fold |
| `../folds.json` | `{groups: "session", folds: [[khung…], …]}` (§14.2) |

## Chọn khung

Pool = khung `dev` của `split.json` v1 trong 6 phiên bay (phase 03), **trừ trọn hai khối cal** (khối 13 và 21,
`cal_mode: block`) và mọi khung test: **965 khung**, 619 khung có ứng viên người của prelabel. Khung mine của phase 03
được chọn sau devpool và loại devpool ra, nên hai tập không giao.

Sức chứa của pool theo khoảng cách tối thiểu giữa hai khung:

| Khoảng cách | Khung tối đa | Khung có ứng viên người |
|---|---|---|
| 1,0 s | 112 | 95 |
| **0,5 s** | **190** | **150** |
| 0,33 s | 253 | 193 |
| 0,2 s | 369 | 267 |

Đích của plan (500–700 khung, ≥ 150 khung có người, cách ≥ 1 s) **không đạt được ở bất kỳ khoảng cách nào**. Chọn
khoảng cách lớn nhất mà số khung có ứng viên còn ≥ 150: **0,5 s** (khoảng cách thật nhỏ nhất 0,501 s). Thứ tự lấy:
mọi khung dev-val v1 còn trong pool (bắt buộc) → khung có ứng viên, xoay vòng theo tầng (phiên × độ cao 3–8 / 8–14 /
> 14 m) → khung không ứng viên. Kết quả **160 khung** (86 dev-val v1 + 74 khung mới); 190 là sức chứa khi không có khung
bắt buộc, các khung dev-val cố định chiếm chỗ nên còn 160.

| Phiên | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Khung devpool | 26 | 15 | 25 | 12 | 82 |
| Khung có người (nhãn cuối) | 18 | 5 | 8 | 0 | 39 |

## Nhãn

- **86 khung dev-val v1**: chép nguyên byte nhãn cũ (115 hộp, 36 khung có người), test kiểm từng byte.
- **74 khung mới**: agent soát từng khung trên ảnh soát (lưới toạ độ, hộp gán trước tô theo nguồn), phóng 2–3× chỗ có
  người, vẽ lại hộp cuối và xem lại toàn bộ khung có người trước khi ghi. Có **103 hộp**, 34 khung có người, 9 khung `hard`.
  Mỗi khung có ít nhất một dòng trong `edits.csv` (200 dòng).
- Hành động: `ok` 24, `tighten` 16, `add` 76, `drop` 75, `hard` 9. Trên 102 hộp gán trước của khung mới chỉ 11 hộp giữ
  nguyên và 16 hộp sửa lại; 75 hộp bỏ (51 RT-DETR, 9 YOLO, 15 cả hai), chủ yếu là túi/tấm nhựa hồng, dải đen mép trái của
  ảnh và hộp to phủ cả mảng cỏ. **76 hộp người phải thêm tay**: prelabel gần như không thấy người nhìn từ trên xuống
  (cùng kết quả đo ở phase 03: 26 % hộp dev-val, conf ≥ 0,05).
- Quy ước theo dev-val v1 để hai phần nhất quán: người đứng sát nhau mỗi người một hộp; phần người bị mép khung cắt vẫn
  gán hộp; chỉ có bóng người (người ở ngoài khung) thì không gán hộp. Khung có điểm không chắc gán `hard`, lý do ghi trong
  `edits.csv`: vật tối ở cuối bóng có thể là người (004_01757, 01760, 01814, 01846), một hay hai người nằm chéo
  (007_00750), người bị cắt ở góc (007_01743, 01756), chỉ có bóng người sát mép (004_01860, 007_01771).

**Tổng devpool: 160 khung, 218 hộp, 70 khung có người.**

So với dev-val cũ (100 khung, 115 hộp, 36 khung có người): thêm 74 khung, 103 hộp, 34 khung có người; bớt 14 khung dev-val
nằm trong khối cal (cả 14 khung đều không người). Số khung có người gần gấp đôi (36 → 70).

## Fold (`folds.json`)

GroupKFold theo phiên vì có 4 phiên có người (≥ 3). Thử k từ 4 xuống, giữ k lớn nhất mà mọi fold có ≥ 20 khung có người:

| k | Khung có người mỗi fold | |
|---|---|---|
| 4 | 39, 18, 8, 5 | loại |
| 3 | 39, 18, 13 | loại |
| **2** | **39, 31** | chọn |

| Fold | Phiên | Khung | Khung có người |
|---|---|---|---|
| 0 | #5 | 82 | 39 |
| 1 | #1, #2, #3, #4 | 78 | 31 |

Đường lui `block5` (5 fold khối 30 s liền nhau) cũng không đạt: 70 khung có người chia 5 thì có fold < 20. Chạy lại
`folds` ra đúng file (test kiểm).

**Ràng buộc cho phase 05–07:** khi huấn luyện model cho fold i, loại khỏi dữ liệu huấn luyện mọi khung thật (mine lẫn nhãn
giả dev) thuộc các phiên của fold i, và loại trọn khối cal (`xla.camera.sessions.reserved_frames`).

## Lệch so với plan

1. **160 khung thay vì 500–700, 70 khung có người thay vì ≥ 150.** Chỉ một buổi bay có video (6 phiên, 1115 khung dev),
   trừ khối cal còn 965; sức chứa ở 1 s chỉ 112 khung. Muốn đạt đích phải bay thêm.
2. **Khoảng cách 0,5 s thay vì 1 s** (bảng sức chứa ở trên); hai khung liền nhau giống nhau hơn, phương sai giữa fold
   vì vậy đánh giá thấp hơn thật.
3. **14 khung dev-val v1 không nằm trong devpool** (nằm trong khối cal phase 03): dev-val cũ không còn là tập con trọn vẹn.
   Danh sách: `devpool.json` → `report.devval_not_in_devpool`; test kiểm đúng bằng tập dev-val trong khối cal.
4. **Chỉ 2 fold**, và fold 0 là một phiên (#5) chiếm 39/70 khung có người. Rủi ro "một phiên chiếm hầu hết người" của
   plan xảy ra thật; phương sai giữa fold chỉ có hai điểm.
5. Khung devpool đều là khung dev v1 mà detector-v1 đã thấy lúc huấn luyện (E2), như khối cal ở phase 03.
