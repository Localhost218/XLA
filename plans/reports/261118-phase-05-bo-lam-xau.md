# Phase 05 — bộ máy làm xấu D0/D1/D2 + dữ liệu nguồn (lane degrade, 18/11/2026)

Trạng thái: máy dựng xong, Accept đạt. Lọc nadir viết lại theo nhãn góc nhìn soát tay (lệnh Owner, mục 1b). Cổng D2
chạy trên **hồ sơ thật** `configs/camera_profile.yaml` v1.1.0 (merge `origin/feat/ai-anh-chat-luong-thap` 98aee74).
Hồ sơ thật lộ một lỗi mô hình tông màu (vùng sáng ra xanh lá thay vì hồng), đã sửa (mục 2).

## 1. Dữ liệu nguồn (`configs/sources.yaml`)

| Nguồn | Kết quả | Cỡ | sha256 |
|---|---|---|---|
| VisDrone2019-DET train | tải 29 s (~53 MB/s), giải nén `D:\DevCache\datasets\src\visdrone` | 1 549 875 511 B, 6 471 ảnh | `86a77eba…6f84` |
| VisDrone2019-DET val | tải 5 s | 81 638 851 B, 548 ảnh | `abeea063…35ef9` |
| NOMAD (CC BY 4.0) | tải **một phần**: 420 ảnh (Actor001–009, a10 + a30, 1/4 khung), rồi Drive trả "Quota exceeded". **Loại khỏi nguồn nadir** (mục 1b) | ảnh gốc 5472×3078 ~8,4 MB; lưu hạ mẫu ×2, 1,1 GB | annotations.json `ec29299e…ad13`; sha256 từng ảnh gốc ở `fetch_manifest.jsonl` |
| POG | **bỏ** | — | cloud.cs.uni-tuebingen.de timeout 20 s; datasetninja chỉ trỏ lại trang Tübingen |

NOMAD bản đủ là 10 zip × 38 GB ≈ 380 GB, không vừa ổ D (còn ~57 GB), nên viết `xla/degrade/fetch_nomad.py` để tải
từng ảnh và hạ mẫu ngay. Mã tải và manifest vẫn giữ. Chạy lại lệnh dưới khi hạn mức Drive hồi, ảnh đã có sẽ được bỏ qua:
`python -m xla.degrade.fetch_nomad --out D:/DevCache/datasets/src/nomad --actors 1-100 --dist a10,a30 --stride 4 --workers 6`.

### 1a. Bộ lọc cũ (sky + CLIP zero-shot), đã thay

Bộ lọc này chấm `sky_frac` và `clip_nadir` (open_clip ViT-B-32 laion2b) cho 7 439 ảnh, cache ở
`D:\DevCache\datasets\src\nadir_scores.jsonl`. Soát tay 100 ảnh có người thì 28 ảnh là nadir. Với ngưỡng
`sky ≤ 0,05` và `clip ≥ 0,40`, precision 0,50, recall 0,54. Owner không nhận mức này, nên viết lại bộ lọc như mục 1b.

### 1b. Bộ lọc mới: nhãn góc nhìn soát tay theo chuỗi, có probe cho chuỗi trộn

Nhãn nằm ở `configs/visdrone_view.yaml`, soát ngày 18/11/2026. Mỗi chuỗi lấy 3 khung (đầu, giữa, cuối) xếp thành lưới
có ghi tên chuỗi, rồi agent xem bằng Read. Lệnh tạo lưới: `python -m xla.degrade.sources seq-sheet`.

- **Lớp góc nhìn:** `nadir` (≲ 20–25°, không thấy chân trời, thấy mái nhà hoặc đỉnh đầu), `near` (≲ 35°), `oblique`,
  `mixed`.
- **284 chuỗi** (khoá `train/…`, `val/…`, vì 24 mã chuỗi trùng giữa hai tập): nadir 14, near 21, oblique 160, mixed 89.
- **Chuỗi mixed** chủ yếu là các `99999xx`. Đây là tập ảnh rời, mỗi ảnh một góc, chiếm 5 873/7 019 ảnh VisDrone. Soát
  thêm từng ảnh 192 ảnh mixed (seed 20261119, lưới ở `D:\DevCache\cache\degrade\mixed_review\`): nadir 18, near 71,
  oblique 103.
- **Luật quyết** (`sources.is_nadir`), theo thứ tự:
  1. Nguồn trong `exclude_sources` bị loại.
  2. Có nhãn ảnh thì dùng nhãn ảnh, không thì dùng nhãn chuỗi. Lớp nằm trong `accept_views: [nadir, near]` thì nhận.
  3. Lớp `mixed` thì dùng probe: nhận khi `probe_nadir ≥ 0,8`.
  4. Chuỗi thiếu nhãn là lỗi, không ngầm cho qua.
- **Lệch so với lệnh Owner (dùng probe thay sky + CLIP cho mixed).** Trên 192 ảnh mixed đã soát:
  - sky + CLIP cũ chỉ đạt precision 0,15 (nadir chặt) và 0,52 (nadir + near); AUC CLIP 0,61 và 0,48, gần như đoán.
  - Probe là hồi quy logistic (C=1, cân lớp) trên đặc trưng ảnh CLIP ViT-B-32, học từ nhãn soát tay. Lệnh:
    `python -m xla.degrade.sources fit-probe`, ra file `nadir_probe_scores.jsonl`.
  - Kiểm chéo 5 phần trên 192 ảnh mixed, nadir + near:

    | t | precision | recall |
    |---|---|---|
    | 0,75 | 0,83 | 0,53 |
    | **0,80** (chọn) | **0,93** | **0,42** |
    | 0,85 | 0,97 | 0,32 |

  - Với nadir chặt, precision không vượt 0,43 nên không dùng được.
  - ViT-L-14 datacomp không hơn B-32 ở lớp near, giữ B-32.
- **Nới sang near (≲ 35°)** theo luật Owner. Nadir chặt chỉ khoảng 9 % ảnh mixed (18/192); cộng 45 ảnh của chuỗi nadir
  thì ước ~520 ảnh có người, dưới 800.
- **NOMAD loại khỏi nguồn nadir.** Ảnh a10/a30 nhìn xiên 45–60° (đã xem ảnh); lý do ghi ở `nadir.exclude_sources`.

| Sau lọc mới (t 0,8) | Ảnh | Ảnh có người | Người |
|---|---|---|---|
| VisDrone train | 1 768 | 1 537 | 26 066 |
| VisDrone val | 73 | 69 | 1 313 |
| NOMAD | 0 (loại) | 0 | 0 |

Nếu hạ t xuống 0,75 thì được train 2 267 / 1 957 / 33 248 và val 82 / 77 / 1 493.

**Đo lại trên đúng 100 ảnh đã soát lần trước.** Nhãn lần trước là "gần nadir" lỏng, 28 ảnh dương.

| Nhóm | n | Precision mới | Recall mới | Độc lập? |
|---|---|---|---|---|
| Tất cả | 100 | **0,69** (chọn 26, đúng 18) | **0,64** | cũ: P 0,50, R 0,54 |
| Ảnh thuộc chuỗi mixed | 72 | 0,68 | 0,71 | độc lập với nhãn đã dùng để học probe |
| Ảnh thuộc chuỗi có nhãn | 19 | 1,00 | 0,75 | **không độc lập**: nhãn chuỗi do chính agent soát |
| NOMAD | 8 | – | 0 | nay bị loại cả bộ |
| Ảnh có nhãn riêng | 1 | 0 | – | 1 dương giả, do định nghĩa hai lần soát lệch nhau |

Số độc lập đáng tin nhất là 0,68 / 0,71 trên 72 ảnh mixed. Kiểm chéo 0,93 / 0,42 dùng định nghĩa nadir + near mới, còn
100 ảnh cũ dùng nhãn lỏng, nên hai con số không so thẳng được với nhau.

## 2. Mã

**Các module trong `xla/degrade/`:**

- `sources.py`:
  - VisDrone: category 1 và 2 → người; category 0 → vùng mà cửa sổ không được chồng lên.
  - NOMAD: hộp COCO XYWH; visibility < 30 → vùng bỏ qua.
  - Nhãn góc nhìn, probe; lệnh `score`, `review-sheet`, `seq-sheet`, `fit-probe`, `stats`.
- `geometry.py`:
  - Thu nhỏ `INTER_AREA` để cạnh ngắn của người theo `person_box_px.short_side` của hồ sơ; không phóng to.
  - Cửa sổ 624×480; 20 % cửa sổ âm; giữ hộp còn ≥ 60 % diện tích.
- `d1_random.py`, `d2_camera.py`, `build.py`, `fetch_nomad.py`.

**Config:** `configs/sources.yaml`, `visdrone_view.yaml`, `degrade_d1.yaml`, `degrade_d2.yaml`. Mỗi khối D2 ghi
`matches:` là các trường hồ sơ nó khớp.

**Test:** `tests/degrade/test_degrade.py`.

**Các bước D2.** Bước (a)–(c) tính trong miền tuyến tính.

- (a) Màn mờ: chọn t riêng cho từng ảnh để dark channel đạt giá trị lấy mẫu từ hồ sơ.
- (b) PSF Gauss theo `psf_sigma_px_fit`, vignette cos⁴ theo `focal_px_fit`.
- (c) Tông màu: bỏ WB → phơi sáng → cắt từng kênh ở 1,0 → nhân lại WB → hệ số ra k (suy từ `magenta.highlight_y` p50)
  → ám màu. Vùng bão hoà thành hồng.
- (d) Làm mềm, rồi thêm nhiễu σ(Y) theo `nlf`, phổ lọc theo `psd_radial`.
- (e) Dải mép trái: khối tối Y 64 xen vạch đỏ chu kỳ 60 px, bắt đầu từ hàng ~150. Số đo từ 117 khung thật của phiên
  20261102-100941, ghi trong `degrade_d2.yaml`.
- (f) JPEG một lần: Pillow `qtables` + `subsampling="4:2:2"`.
  - q = 12 (trọng số 0,8): dùng bảng của hồ sơ.
  - q 19–40: Annex K × q/16; manifest ghi `qtable_verified: false`.

**Sửa sau khi chạy hồ sơ thật (bước c).** Trong hồ sơ v1.1.0, `color.wb_gain_r_fit` = 0,9753 và `wb_gain_b_fit` =
0,96423. Hai số này chính là p50 của `r_over_g` và `b_over_g` (`xla/camera/profile.py:212`), không phải hệ số WB của
ISP.

- **Lỗi:** dùng chúng làm hệ số ISP (< 1) thì điểm bão hoà ra G > R, B, tức xanh lá. Đo 50 khung D2 ra 0/50 khung có
  > 0,5 % điểm hồng, trong khi khung thật là 31 %.
- **Sửa:**
  - `stage_tone` lấy hệ số ISP từ cấu hình: `isp_wb_gain_r/b: 2.2`. Đây là giả định; chỉ cần lớn hơn brightness/k ≈ 2,0.
  - Ám màu của tông chưa bão hoà lấy p50 `r_over_g` / `b_over_g` của hồ sơ, nhân `cast_r` / `cast_b`.
  - D2 thôi đọc `wb_gain_*_fit`.
  - Test tông màu nay chạy cả với hồ sơ của repo: vùng trắng phải ra hồng, vùng xám phải mang đúng ám màu p50.
- Không sửa hồ sơ, vì file đó do lane data sinh ra.

## 3. Accept, smoke, cổng D2 hồ sơ thật

- `pytest tests/degrade/test_degrade.py` → **22 passed, exit 0**.
- `pytest -m "not gpu and not data"` → **119 passed, 8 deselected, exit 0**.
- `ruff check xla/degrade tests/degrade` → exit 0.
- **Cổng D2:** `python -m xla.degrade.build --set d2 --seed 20261118 --limit 50 --json --profile
  configs/camera_profile.yaml`.
  - exit 0, 50 ảnh ghi ra, 2 đơn vị bỏ qua, `manifest.jsonl` 50 dòng.
  - Tổng 1 841 ảnh nguồn, 3 682 đơn vị; 395 người.
  - Cỡ trung bình 15 996 B (khung thật p50 15,1 KB). Kết quả ở `D:\DevCache\datasets\xla\d2`.
- D0 và D1 chạy cùng lệnh: đều exit 0, 50 dòng.
- **Tất định:** chạy D2 1 worker và 8 worker cho tổng sha256 trùng nhau.

| Tập | giây/ảnh 1 worker | 8 worker | cỡ TB | ước tập đủ (3 682 ảnh = 1 841 nguồn × 2 cửa sổ) |
|---|---|---|---|---|
| D0 | 0,011 | — | 128 KB | ~40 s; ~470 MB |
| D1 | 0,062 | — | 46 KB | ~4 phút 1 worker; ~170 MB |
| D2 | 0,151 | 0,044 (50 ảnh, gồm cả thời gian khởi động) | 16,0 KB | ~9 phút 1 worker / ~2,7 phút 8 worker; ~59 MB |

**50 khung D2 so với hồ sơ thật**, đo bằng `xla.camera.measure.measure_image`. Phase 07 dùng bảng này làm điểm xuất phát
để hiệu chỉnh.

| Đại lượng | D2 p50 (p5–p95) | thật p50 (p5–p95) | Nhận xét |
|---|---|---|---|
| dark_channel_15px | 140 (126–152) | 144 (135–152) | khớp |
| y_p1 | 129 | 113 | chỗ tối của D2 còn sáng quá |
| saturation_mean | 12,7 | 16,2 | thiếu bão hoà màu |
| pink_frac / số khung > 0,5 % | 0,0001 / 14 % | 0,0017 / 31 % | đã có hồng, nhưng còn ít |
| pink_highlight_y | 212 | 222 | gần |
| r_over_g / b_over_g | 0,991 / 0,979 | 0,975 / 0,964 | lệch ~1,5 % do màu cảnh VisDrone |
| laplacian_var | 16,7 | 34,1 | **mềm quá ~2 lần** |
| sigma_immerkaer | 0,23 | 0,35 | nhiễu thiếu |
| vignette corner/center | 0,90 | 0,81 | vignette yếu |
| blocking_ratio_8x8 | 2,25 | 2,68 | khối JPEG yếu |
| mtf50 | 0,22 | 0,25 | gần |
| left_strip width | 14 | 14 | khớp |

`negative_windows` = 0/50 trong smoke: 50 đơn vị đầu đều là ảnh VisDrone đông người. Ảnh NOMAD không có người nay đã bị
loại. Cửa sổ âm vẫn được sinh ở tập đủ (20 % kế hoạch), nhưng tỉ lệ thực tế phải đo lại khi sinh tập đủ.

## 4. Quyết định tự chọn

1. NOMAD chỉ tải tập con vì bản đủ 380 GB; sau đó loại khỏi nguồn nadir vì ảnh nhìn xiên.
2. D1: thu hẹp dải resize (0,5–1,5 / 0,5–1,2) vì không có bước hạ ×4 cuối của Real-ESRGAN. Nhòe, nhiễu, JPEG giữ đúng
   khoảng gốc. Chỉ dùng kernel Gauss.
3. Hình học dùng chung `person_box_px.short_side` của hồ sơ cho cả ba tập. Hàm làm xấu D1 không đọc hồ sơ (có test AST).
4. D0 lưu Pillow q95 4:4:4. D1 ghi file bằng JPEG vòng hai (cv2), không nén thêm lần nữa.
5. D2 làm dải mép (e) sau ISP (d); dải được làm mềm σ 1, không thêm nhiễu.
6. `windows_per_image: 2`.
7. Chuỗi mixed lọc bằng probe CLIP học từ nhãn soát tay, thay cho sky + CLIP (số đo ở mục 1b). Chọn t = 0,8 và nới sang
   near.
8. Hệ số WB của ISP trong D2 lấy từ cấu hình (2,2), không lấy từ hồ sơ (mục 2).

## 5. Việc còn sót

- **Phase 07** hiệu chỉnh D2 theo bảng ở mục 3. Ưu tiên: độ nét (laplacian ×0,5), nhiễu, vignette, độ sáng chỗ tối, tỉ
  lệ hồng.
- **Hợp đồng hồ sơ (§6):** tên `color.wb_gain_*_fit` dễ hiểu nhầm là hệ số ISP. Lane data nên đổi tên hoặc ghi rõ nghĩa.
- **Tập nguồn:** sau lọc chỉ còn 69 ảnh val có người, ít. Hạ t xuống 0,75 được 77 ảnh. Hoặc tách val theo chuỗi, không
  theo split gốc của VisDrone.
- NOMAD còn lại chưa tải (Drive hết hạn mức); dù tải thêm cũng không dùng làm nguồn nadir.
- Chưa sinh tập đủ. Phase 07 sinh sau khi hiệu chỉnh D2.
