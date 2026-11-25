# `results/v2/neg-holdout/` — phiên bay, chia `neg_flight` mine / cal / holdout (AI v2 phase 03)

Mã: `xla/camera/sessions.py` (+ `xla/camera/frames.py` nhập mọi bản ghi), test `tests/camera/test_sessions_neg_split.py`.
Hợp đồng: `docs/hop-dong-model.md` §14.2. Seed 20261125. Ngày chạy 25/11/2026.

```bash
$PY -m xla.camera.sessions index --json      # backend.flights --events (IOT-CV) -> data/real/sessions.json
$PY -m xla.camera.sessions prelabel --json   # RT-DETR-x + YOLO11x + SAHI, conf 0,05, trong gpu_lock (D6)
$PY -m xla.camera.sessions split --allow-unreviewed --json   # chỉ để lập kế hoạch (khối cal) trước phase 04 select
$PY -m xla.camera.sessions sheets --json     # ảnh ghép 4×4 mọi khung neg dự kiến -> agent soát
$PY -m xla.camera.sessions split --json      # -> data/real/neg_flight_split.json (exit 1 nếu còn khung chưa soát)
$PY -m xla.camera.sessions review --json     # trang duyệt holdout review/index.html
$PY -m xla.camera.sessions apply-review <file xuất> --json   # SAU khi chủ dự án duyệt (chưa chạy)
```

## 1. Phiên bay (`data/real/sessions.json`)

Nguồn: `uv run --no-sync python -m backend.flights --events` của IOT-CV (chạy subprocess, bỏ `UV_PROJECT_ENVIRONMENT`
và `VIRTUAL_ENV` khỏi env con), bản chép ở `flights_events.txt`. Một phiên = một chuyến bay (`FlightWindow`) cắt theo
khoảng giờ của một bản ghi camera, id `<rec>#<i>`. `xla/camera/frames.py` giờ quét **cả 11 bản ghi** (v1 chỉ đọc một),
mỗi dòng khung có thêm trường `session`; `manifest.csv` không đổi cột (trường mới bị bỏ qua lúc ghi).

| Bản ghi | Phiên | Khung giữ | rel_alt lớn nhất |
|---|---|---|---|
| 20261101-175552 … 20261101-191230 (6 bản ghi) | 0 | 0 | ≤ 1,79 m (đứng đất / trong nhà) |
| 20261102-033719, -040317, -042928, -043054 | 0 | 0 | ≤ 0,27 m hoặc không có GPS |
| **20261102-100941** | **6** | **2915** | 57,75 m |

Đếm nhập khung khớp v1 từng số: 17775 đoạn AVI, 11908 dòng CSV có JPEG, 4769 trong cửa sổ bay, 2944 cao > 3 m,
2915 sau khi bỏ 29 khung nhảy độ cao. Không khung nào bị bỏ sót (`not_ingested` 0), không khung cũ lạc trong manifest.

| Phiên | Chuyến | Khung | Cao (trung vị) | Sáng (trung vị) | split v1 |
|---|---|---|---|---|---|
| #0 | 10:10:46 | 371 | 17,3 m | 163 | test 43, ngoài split 328 |
| #1 | 10:14:35 | 616 | 9,3 m | 163 | dev 172, test 37 (dev-val 13) |
| #2 | 10:23:19 | 241 | 9,9 m | 163 | dev 93, test 23 (dev-val 9) |
| #3 | 10:26:36 | 323 | 10,1 m | 161 | dev 217, test 10 (dev-val 20) |
| #4 | 10:30:39 | 530 | 8,9 m | 165 | dev 80, test 61 (dev-val 6) |
| #5 | 10:40:39 | 834 | 13,9 m | 164 | dev 553, test 24 (dev-val 52) |

## 2. Chia `neg_flight` (`data/real/neg_flight_split.json`)

**Cả 6 phiên đều có khung test v1** (`split.json` v1 chia theo khối 30 s xen kẽ trong cùng buổi bay), nên không phiên
nào được làm holdout hay cal nguyên phiên: holdout 0 phiên, `cal_mode` = **`block`** (đường lui của plan). Trong phiên
khoá chỉ dùng khung `dev` v1; khung test v1 không bao giờ vào neg.

- **cal** = trọn hai khối dev v1: khối **13** (phiên #3, 77 khung) và khối **21** (phiên #5, 73 khung) = **150 khung**.
  Chọn khối mất ít khung nhất (ít dev-val + ít ứng viên người của prelabel), tối đa một khối mỗi phiên, mỗi phiên còn
  giữ ít nhất một khối dev. Thứ hạng và lý do loại từng khối: `split_report.json` → `cal_block_candidates`.
  **Mọi khung** của hai khối này (kể cả khung có người) bị giữ lại khỏi devpool, mine và fold (`reserved_frames`).
- **mine** = khung dev v1 còn lại, ngoài devpool (phase 04) và ngoài khối cal, cách nhau ≥ 0,5 s: dự kiến 171, bỏ 86
  khung có người sau khi soát → **85 khung** (#1: 6, #2: 12, #3: 15, #4: 13, #5: 39; #0 không có dev).
- **holdout = 0** khung.

| split | dự kiến | bỏ vì có người | còn |
|---|---|---|---|
| cal | 150 | 0 | **150** |
| mine | 171 | 86 | **85** |
| holdout | 0 | 0 | **0** |

`approved` là `false` ở mọi khung (chỉ có nghĩa với holdout). Khoá ngoài hợp đồng §14.2: `cal_mode` ở mức gốc (phase
03 và phase 09 cùng dùng). Chia tất định theo seed 20261125: chạy lại ra đúng file (test `test_split_is_deterministic`).

### Kill (a) — số holdout trước khi duyệt: **0 < 150 → "không đủ chứng minh"**

Ghi trước khi chủ dự án duyệt, đúng yêu cầu plan. Holdout rỗng nên phase 09 không có tập âm độc lập để đo tỷ lệ báo
nhầm. Đường duy nhất để có holdout là bay thêm một buổi trên bãi trống không người (phiên mới không dính test v1);
khi đó chạy lại `index` → `split` → `review` → `apply-review`.

## 3. Prelabel và soát của agent

Prelabel chạy RT-DETR-x @640 + YOLO11x @1280 + SAHI, conf 0,05, trên cả 1115 khung dev v1, trong
`xla.gpu_lock.gpu_lock` (nhả sau khi chạy). **GPU 2935 s (~49 phút)**. Kết quả thô: `prelabel/raw.json`.

**Prelabel sót người nhìn từ trên xuống rất nhiều.** Trên 115 hộp người của dev-val v1 (36 khung có người): mọi model
gộp lại, IoU ≥ 0,3 và conf ≥ 0,05, chỉ thấy 26,1 % (conf 0,3: 21,7 %; conf 0,5: 10,4 %); hộp đã gộp IoU ≥ 0,5 thấy
20 %; 5/36 khung không có hộp nào. Vì vậy prelabel **chỉ dùng để xếp thứ tự khối cal** (ít ứng viên trước), không dùng
để quyết khung nào sạch.

Quyết định "có người" do agent soát **từng khung** trên ảnh ghép 4×4 (0,5×) của cả 321 khung neg dự kiến, phóng to
(`xla.detect.review crop`, 2–3×) chỗ nghi ngờ. Chỉ mục ô → khung: `agent_review_sheets.csv`; kết quả:
`agent_review.csv` (frame, verdict, reason), 321 dòng, **86 `person`**. Ảnh ghép dựng lại bằng
`$PY -m xla.camera.sessions sheets --json` (không commit, nằm ở cache).

- Khối cal 13 và 21 (sân bê tông/cỏ, túi rác tối, mảnh nhựa hồng): không thấy người. Ô nghi ngờ duy nhất (khối 21, các
  khung 007_01477…01486, mảng hồng/xanh trong bụi) phóng to ra là rác nhựa → giữ.
- 86 khung mine bị loại: 67 có người rõ (nhóm người đi trên đường đất, prelabel sót gần hết; tính cả 2 khung có nửa người ở mép trên) và 19 khung **không chắc**
  (bóng người, có thể kèm một phần người, sát mép khung, một đống vật tối giống người ngồi). Loại cả nhóm không chắc vì tập neg
  phải sạch; cái giá chỉ là bớt khung mine.

## 4. Trang duyệt holdout và `apply-review`

- Trang: `results/v2/neg-holdout/review/index.html` (mở thẳng bằng trình duyệt). Lưới 4×4, mỗi khung một ô "có người",
  mỗi lưới một ô "Đã xem lưới này", nút **Xuất JSON**, trạng thái ở `localStorage` (khoá `xla.neg-holdout-review.v1`
  + 12 ký tự đầu sha256 của file chia). Hiện có **0 khung** (holdout rỗng), trang báo điều đó thay vì lưới trống.
- Đã kiểm trên Chrome headless với một file chia giả 20 khung: ảnh tải được, đánh dấu còn sau khi tải lại trang, file
  xuất được `apply-review` nhận.
- Sau khi duyệt: `$PY -m xla.camera.sessions apply-review <file xuất>.json --json` (thêm `--dry-run` để xem trước).
  Lệnh từ chối khi: sai `version`, sai sha256 của file chia, thiếu/thừa khung, còn lưới chưa đánh "đã xem" (trừ khi
  `--allow-unseen`), file đã áp rồi, dòng hỏng. Khung `person: false` → `approved: true`; khung `person: true` → bỏ khỏi
  holdout. Báo lại Kill (a) theo số đã duyệt. **Chưa chạy** (không có gì để duyệt).

## 5. Ràng buộc cho phase sau

- `split.json` v1 không đổi (sha256 `c1b01728…192c`, test kiểm).
- Khung cal và holdout không nằm trong bất kỳ manifest huấn luyện/mining nào của XLA. Test quét `data/`, `results/` và
  `XLA_DATA_ROOT`; ngoại lệ có tên: `e2_r1`/`e2_r2` (tập huấn luyện E2 của **v1**, `detector-v1.0.0` = `e2/r1`) chứa
  khung dev v1 của mọi khối, gồm cả khối cal. Đây là cái giá của `cal_mode: block` (không có khung dev nào model v1 chưa
  thấy); tương tự `results/realism/calibration.json` + `realism-smoke/dev_b` (hiệu chỉnh độ thật ảnh làm xấu v1 đo trên khung dev v1). `results/restore/export.json` (2 khung vàng kiểm parity ONNX) không phải dữ liệu huấn luyện. Companion test hỏng khi ngoại lệ thừa hoặc thiếu.
- **Phase 05 phải** loại `reserved_frames` (trọn khối cal) khỏi danh sách nhãn giả dev, và khi huấn luyện fold i thì
  loại mọi khung thật (mine lẫn dev giả) thuộc phiên của fold i.

## 6. Holdout yếu "cùng buổi" (26/11/2026) — **0 khung, B2 vẫn không đủ chứng minh**

**Nhãn bắt buộc: holdout yếu: cùng buổi bay với test, không độc lập; agent duyệt.** Quyết định chủ dự án 26/11 (không bay
thêm được): lấy khung bay của `20261102-100941` **không** thuộc `split.json` v1 (không dev, không test), không là
mine/cal/devpool/khung giữ lại, cách mọi khối dev/test v1 **≥ 30 s**, giãn cách ≥ 0,5 s trong phiên; agent soát từng khung
(thay cho chủ dự án duyệt: `approved_by: "agent"`, `owner_approved: false`). Không thay holdout chặt, không dùng cal.

```bash
$PY -m xla.camera.sessions weak-holdout --sheets --json        # -> weak/plan.json + khối holdout_weak (exit 1 nếu còn ứng viên chưa soát)
$PY -m xla.camera.sessions weak-holdout --min-dist-s 30 --prelabel --sheets --json   # thêm nhãn trước (GPU, khoá D6)
$PY -m xla.camera.sessions score-weak --iotcv D:/IOTCV-wt/iotcv --json   # -> results/v2/system/holdout_weak.json
```

Lưu ở khoá gốc `holdout_weak` của `neg_flight_split.json` (`kind: "weak_same_day"`, `label`, `rule`, `frames`), **ngoài**
`frames`: mọi bên đọc `frames` (system, train, verify, detection) và đường holdout chặt (`split: holdout`, `approved`) không
thấy nó; `split` / `apply-review` giữ nguyên khối này. Soát của agent: `weak/agent_review.csv` (cùng dạng
`frame,verdict,reason`), ảnh ghép 4×4 như mục 3 (`weak/agent_review_sheets.csv`, ảnh ở cache).

**Kết quả: 0 ứng viên.** 1 602 khung bay đã nhập không thuộc `split.json` v1, **cả 1 602 đều nằm trong hoặc sát một khối
dev/test v1**: chúng là khung không được lấy mẫu **bên trong** các khối test v1 (khối 30 s của test v1 chỉ lấy rải một phần khung
vào `split.json`), cộng vài khung ở khe giữa hai khối. Khoảng cách tới khối dev/test v1 gần nhất (khoảng giờ thật của các khung
trong khối), sau giãn cách 0,5 s (`weak/plan.json` → `dist_hist`, `dist_hist_by_session`):

| ≥ khoảng cách | 0 s | 1 s | 2 s | 5 s | 10 s | 15 s | 20 s | **30 s** |
|---|---|---|---|---|---|---|---|---|
| khung | 326 | 53 | 37 | 8 | 0 | 0 | 0 | **0** |

Theo phiên ở 0 s: #0 67, #1 74, #2 26, #3 20, #4 85, #5 54; ở 5 s chỉ còn #0 3, #3 3, #5 2. Không có ứng viên nên không chạy
nhãn trước, không soát ảnh, không lấy khoá GPU; `holdout_weak.frames` = `[]`, `results/v2/system/holdout_weak.json` →
**"không đủ chứng minh (0 khung)"** ở cả ba tầng.

- Khung trong cửa sổ bay nhưng **chưa nhập** vì cao ≤ 3 m (`ALT_MIN_M`) mà cách khối ≥ 30 s: 209 khung (đếm tay từ
  `frames.csv` của IOT-CV, giãn 0,5 s; #0 42, #1 38, #2 52, #3 39, #4 23, #5 15), độ cao lớn nhất 0,23–1,64 m = lúc hạ cánh,
  sát đất. Không phải khung bay (định nghĩa §14.2 / phase 03), nên không phải ứng viên.
- Muốn có số B2 từ dữ liệu này thì phải nới luật 30 s (chủ dự án quyết): khung còn lại ở 0–5 s là hàng xóm 0,2–5 s của khung
  test v1 (gần như cùng cảnh với 159 khung bay trống của test, đã chấm ở `results/v2/system/README.md` §4), nên số đó gần như
  chấm lại test, không phải tập giữ kín. Chạy lại: `weak-holdout --min-dist-s <s> --prelabel --sheets`, agent soát, rồi
  `score-weak`.

## Lệch so với plan

1. **Holdout 0 khung, cal theo khối (`cal_mode: block`)**: cả 6 phiên dính test v1. Kill (a) "không đủ chứng minh".
2. **Khung cal là khung dev v1 model v1 đã thấy lúc huấn luyện** (E2): không còn khung bay nào chưa dùng ngoài test.
   Ngưỡng hiệu chỉnh trên cal vì vậy có thể lạc quan với detector-v1; với detector v2 thì khối cal bị loại khỏi huấn luyện.
3. **14 khung dev-val v1 nằm trong khối cal** nên không vào devpool phase 04 (danh sách trong `split_report.json` →
   `devval_in_cal_blocks`).
4. **mine chỉ 85 khung** (kế hoạch không đặt số tối thiểu, nhưng nhỏ): phần lớn khung dev còn lại có người.
5. `manifest.csv` không thêm cột `session` (file không thuộc phạm vi lane); phiên của từng khung nằm trong
   `sessions.json` và `neg_flight_split.json`, tra bằng `sessions.session_of`.
