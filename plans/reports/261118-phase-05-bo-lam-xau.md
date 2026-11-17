# Phase 05 — bộ máy làm xấu D0/D1/D2 + dữ liệu nguồn (lane degrade, 18/11/2026)

Trạng thái: máy dựng xong, Accept đạt; smoke D2 chạy trên **hồ sơ camera tổng hợp** vì `configs/camera_profile.yaml`
(phase 04, lane data) chưa có lúc xong việc. Cổng D2 với hồ sơ thật còn chờ (lệnh ở mục 6).

## 1. Dữ liệu nguồn (`configs/sources.yaml`)

| Nguồn | Kết quả | Cỡ | sha256 |
|---|---|---|---|
| VisDrone2019-DET train | tải 29 s (~53 MB/s), giải nén `D:\DevCache\datasets\src\visdrone` | 1 549 875 511 B, 6 471 ảnh | `86a77eba…6f84` |
| VisDrone2019-DET val | tải 5 s | 81 638 851 B, 548 ảnh | `abeea063…35ef9` |
| NOMAD (CC BY 4.0) | **một phần**: 420 ảnh (Actor001–009, a10 + a30, 1/4 khung), rồi Drive trả "Quota exceeded" | ảnh gốc 5472×3078 ~8,4 MB; lưu hạ mẫu ×2, 1,1 GB | annotations.json `ec29299e…ad13`; sha256 từng ảnh gốc ở `fetch_manifest.jsonl` |
| POG | **bỏ** | — | cloud.cs.uni-tuebingen.de timeout 20 s; datasetninja chỉ trỏ lại trang Tübingen |

NOMAD bản đủ là 10 zip × 38 GB ≈ 380 GB, không vừa ổ D (còn ~57 GB), nên viết `xla/degrade/fetch_nomad.py` tải từng ảnh
và hạ mẫu ngay. Chạy lại cùng lệnh sau khi hạn mức Drive hồi thì lệnh tự bỏ qua ảnh đã có:
`python -m xla.degrade.fetch_nomad --out D:/DevCache/datasets/src/nomad --actors 1-100 --dist a10,a30 --stride 4 --workers 6`.
Ảnh NOMAD a10/a30 nhìn xiên ~45–60°, không thẳng xuống (đã xem ảnh).

### Lọc nadir (Deferred 3)

Điểm `sky_frac` (dải trên 30 %: sáng, phẳng, ngả trời) + `clip_nadir` (open_clip ViT-B-32 laion2b, hai nhóm câu
mô tả) cho 7 439 ảnh, cache `D:\DevCache\datasets\src\nadir_scores.jsonl` (180 s, GPU có `gpu_lock`). Soát tay 100
ảnh có người (lưới ảnh `D:\DevCache\cache\degrade\nadir_review\`): 28/100 nadir. AUC `clip_nadir` = 0,74; ViT-L-14
datacomp (0,69) và bộ câu khác (0,71–0,74) không hơn. Chọn `sky ≤ 0,05`, `clip ≥ 0,40`: precision 0,50, recall 0,54.

| Sau lọc | Ảnh | Ảnh có người | Người |
|---|---|---|---|
| VisDrone train | 1 550 | 1 406 | 35 257 |
| VisDrone val | 170 | 164 | 4 770 |
| NOMAD train (val rỗng: chưa tới diễn viên chia hết cho 10) | 420 | 319 | 319 |

## 2. Mã

`xla/degrade/`: `sources.py` (VisDrone cat 1+2 → người, cat 0 → vùng cửa sổ không chồng; NOMAD COCO XYWH, visibility < 30
→ vùng bỏ qua, khoảng cách theo `_aNN`, val theo diễn viên), `geometry.py` (thêm: thu nhỏ `INTER_AREA` để cạnh ngắn
người theo `person_box_px.short_side` của hồ sơ, không phóng to, cửa sổ 624×480, 20 % cửa sổ âm, giữ hộp ≥ 60 % diện
tích), `d1_random.py`, `d2_camera.py`, `build.py`, `fetch_nomad.py` (thêm). Config: `configs/sources.yaml`,
`degrade_d1.yaml`, `degrade_d2.yaml` (mỗi khối ghi `matches:` trường hồ sơ). Test: `tests/degrade/test_degrade.py`.

D2: (a) màn mờ chọn t theo từng ảnh để dark channel đạt giá trị lấy mẫu từ hồ sơ; (b) PSF Gauss `psf_sigma_px_fit` +
cos⁴ `focal_px_fit`; (c) bỏ WB → phơi sáng → cắt từng kênh ở 1,0 → nhân WB (`wb_gain_*_fit`) + hệ số ra k suy từ
`magenta.highlight_y` p50 → vùng bão hoà thành hồng; (d) làm mềm + nhiễu σ(Y) theo `nlf`, phổ lọc theo `psd_radial`;
(e) dải mép trái: khối tối Y 64 xen vạch đỏ chu kỳ 60 px từ hàng ~150 (đo 117 khung thật của 20261102-100941, ghi
trong `degrade_d2.yaml`); (f) JPEG một lần, Pillow `qtables` + `subsampling="4:2:2"`, q = 12 (trọng số 0,8, bảng của
hồ sơ) hoặc 19–40 (Annex K × q/16, `qtable_verified: false` trong manifest). (a)–(c) tính tuyến tính.

## 3. Accept và smoke

- `pytest tests/degrade/test_degrade.py` → **19 passed, exit 0**. Toàn bộ `pytest -m "not gpu and not data"` → 81 passed.
  `ruff check xla/degrade tests/degrade` sạch.
- `python -m xla.degrade.build --set d2 --seed 20261118 --limit 50 --json --profile <hồ sơ tổng hợp>` → exit 0,
  `manifest.jsonl` 50 dòng. Cùng lệnh d0, d1 cũng exit 0, 50 dòng. Ra `D:\DevCache\datasets\xla_smoke\`.
- Tất định: 50/50 sha256 trùng giữa chạy 1 worker và 8 worker.

| Tập | giây/ảnh 1 worker | 8 worker (đo d2 400 ảnh) | cỡ TB | ước tập đủ (4 280 ảnh = 2 140 nguồn × 2 cửa sổ) |
|---|---|---|---|---|
| D0 | 0,024 | — | 150 KB | ~1,7 phút 1 worker; ~640 MB |
| D1 | 0,100 | — | 46 KB | ~7 phút 1 worker; ~200 MB |
| D2 | 0,191 | 0,037 | 16,9 KB (khung thật p50 15,1 KB) | ~14 phút 1 worker / ~2,7 phút 8 worker; ~72 MB |

8 % đơn vị bị bỏ vì không đặt được cửa sổ hợp lệ (lấy đơn vị kế tiếp); cửa sổ âm thực tế 17,5 % (VisDrone đông người).

## 4. Quyết định tự chọn

1. NOMAD chỉ tải tập con (a10 + a30, 1/4 khung, hạ mẫu ×2 + JPEG q95) vì bản đủ 380 GB.
2. D1: dải resize thu hẹp (0,5–1,5 / 0,5–1,2) vì không có bước hạ ×4 cuối của Real-ESRGAN; nhòe/nhiễu/JPEG giữ đúng
   khoảng gốc; chỉ kernel Gauss.
3. Hình học dùng chung lấy `person_box_px.short_side` từ hồ sơ cho cả ba tập; hàm làm xấu D1 không nhận hồ sơ (test).
4. D0 lưu Pillow q95 4:4:4. D1 ghi file bằng JPEG vòng hai (cv2), không nén lại lần nữa.
5. D2 thứ tự phase: dải mép (e) sau ISP (d), dải được làm mềm σ 1, không thêm nhiễu.
6. `windows_per_image: 2`.

## 5. Việc còn sót

- Smoke D2 với hồ sơ thật (`D:\XLA-wt\data\configs\camera_profile.yaml` chưa có).
- NOMAD phần còn lại (Drive hết hạn mức); val NOMAD rỗng tới khi tải thêm Actor010+.
- Bộ lọc nadir yếu (precision 0,50): khoảng nửa ảnh giữ lại vẫn xiên.
- Chưa sinh tập đủ (phase 07 sinh sau khi hiệu chỉnh D2).

## 6. Lệnh chạy cổng D2 khi có hồ sơ thật

```
python -m xla.degrade.build --set d2 --seed 20261118 --limit 50 --json --profile D:/XLA-wt/data/configs/camera_profile.yaml
```
