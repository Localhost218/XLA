# Độ giống thật của ảnh tổng hợp (`results/realism/`)

Phase 06 viết phép đo, phase 07 dùng nó để hiệu chỉnh D2 và ghi bảng D1/D2 vào file này (thêm mục, không ghi đè mục dưới).

## Phép đo (phase 06)

`python -m xla.eval.realism --real <dir|split:dev> --synth <dir> --json [--out <file>]`, exit 0 đo xong, 1 lỗi đầu vào,
2 hai tập khác quy trình đọc (khác cỡ ảnh, ảnh không giải mã được). Ba tầng, xếp theo độ tin:

| Tầng | Cách tính | "Giống" là |
|---|---|---|
| Thống kê mức thấp | W1 giữa phân bố từng khoá `xla.camera.measure.measure_image` + từng bin `nlf`, `psd_radial` (log10), chia IQR tập thật; tổng = trung bình có trọng số (`--weights`, mặc định bằng nhau). DQT + lấy mẫu màu so khớp chính xác, báo riêng | gần nền nhiễu bên dưới |
| C2ST | CNN 60 929 tham số trên phần dư `ảnh − Gauss σ=2` của patch 64×64; chia train/val/test theo ảnh nguồn; CI 95 % bootstrap theo cụm ảnh | gần nền nhiễu (không phải 0,5, xem dưới) |
| KID-DINOv2 | DINOv2 ViT-L/14 token CLS trên patch 224×224 (không resize), MMD² không chệch nhân đa thức bậc 3, 10 tập con rời nhau, CI 95 % bootstrap | gần nền nhiễu |

Hai tập đi qua cùng `xla.eval.patches`: `cv2.imdecode(IMREAD_COLOR)`, bỏ cột `[0, 16)`, cắt patch có seed, không resize.
Đặc trưng KID cache theo sha256 ảnh ở `<cache_dir>/realism/` (mặc định `D:\DevCache\cache\realism\`); trích trong `xla.gpu_lock`.

### Bộ mã hoá KID: thay `dgm-eval` bằng `torch.hub`

`dgm-eval` (layer6ai-labs, commit 74c79f7) không cài được vào venv chung: nó ghim `pillow==9.2.0`, `numpy==1.23.3`,
`timm==0.8.19.dev0`, `transformers==4.26.0`, `xformers`, trái `pillow>=12` mà plan cần (ghi ở `D:\DevCache\venvs\xla.READY`).
Owner đã duyệt phương án thay (`plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-01-07.md` §2): cùng model
`dinov2_vitl14` nạp qua `torch.hub` (`TORCH_HOME=D:\DevCache\torch`, trọng số sha256 `d5383ea8…cbf428`), cùng tiền xử lý
của `dgm_eval/models/dinov2.py` (PIL RGB → resize 224×224 bicubic → /255 → chuẩn hoá ImageNet), KID tự viết. Nếu sau này
`dgm_eval` import được, `xla.eval.kid.Dinov2Features(backend="auto")` tự dùng encoder của nó.

### Smoke 18/11/2026 (`smoke.json`, RTX 4060 Laptop)

D2 `--limit 50` của phase 05 chưa có nên cặp chính là dev thật vs **D0 tạm** (ảnh VisDrone sạch, cắt về 640×480, không làm
xấu) — cận trên của khoảng cách, không phải số của D2. Cặp thứ hai là **nền nhiễu**: hai nửa dev cùng camera chia theo block 30 s.

| Cặp | Thống kê (W1/IQR) | DQT khớp | C2ST acc [CI95] | KID ± SE | Giây (thống kê / C2ST / KID) | VRAM đỉnh MB (C2ST / KID) |
|---|---|---|---|---|---|---|
| dev (470) vs D0 tạm (470) | 8,29 | không (PNG) | 0,974 [0,958; 0,986] | 7,55 ± 0,15 | 75 / 23 / 85 | 368 / 1 549 |
| dev nửa B (323) vs nửa A (147) | 0,84 | có | 0,767 [0,720; 0,807] | 1,33 ± 0,09 | 28 / 4 / 91 | 368 / 1 549 |

KID chạy lại có cache: 10,5 giây (so với 85), cùng giá trị.

Đọc các số smoke này như bản ghi lịch sử, không so trực tiếp với mục phase 07 bên dưới. Chúng đo trên lần chia đầu (dev 470
khung), trước khi chia lại thành dev 1 115 / test 198, và dùng một bản `xla/camera/measure.py` của lane data lúc chưa gộp.
Không chạy lại vì phase 07 đã có hàng thật–thật đo trên đúng split hiện tại.

**Nền nhiễu không về 0.** Hai nửa dev cùng camera vẫn cho C2ST 0,77 và KID 1,33: hai nửa khác cảnh và thời điểm bay, phần dư
cao tần vẫn mang dấu nội dung (độ cao, mặt đất). Phase 07 nên coi D2 "giống thật" khi các số rơi vào vùng nền này, không đặt
mục tiêu C2ST 0,5 hay KID 0.

## Phase 07: hiệu chỉnh D2 và so D1 / D2 / D2+Real-SR (18/11/2026)

Hồ sơ camera `configs/camera_profile.yaml` v1.2.0. Số liệu: `calibration.json` (hiệu chỉnh) và `d1_vs_d2.json` (so sánh).
Tập sinh nằm ở `D:\DevCache\datasets\xla\{d0,d1,d2,d2sr}`, không commit. Tham số hiện hành là **`params_version` 3** (vòng 2,
mục dưới); mục "Hiệu chỉnh trên khung dev" là vòng 1 (`params_version` 2), giữ làm bản ghi.

### Hiệu chỉnh trên khung dev

Lệnh: `python -m xla.degrade.calibrate --split dev --budget 155 --workers 16`.

- **Mục tiêu:** tổng W1/IQR của phase 06 (cùng `stat_distance`, trọng số bằng nhau), giữa D2 sinh từ 500 đơn vị nguồn cố
  định và 1 115 khung dev. Không dùng khung test nào; test `test_calibration_used_dev_frames_only` kiểm điều này.
- **Thuật toán:** CMA-ES (`cma`) trên 11 tham số chuẩn hoá về [0, 1], cận ghi ở khối `calibration` của
  `configs/degrade_d2.yaml`. 155 lượt, 14 thế hệ.
- **Thời gian:** 1,6 giờ CPU wall (16 tiến trình), có một lần dừng rồi chạy tiếp từ checkpoint. Không dùng GPU.
- **Kết quả:** khoảng cách **0,967 → 0,736**. `params_version` 1 → 2.
- Mức "trước" là cấu hình phase 05, riêng hệ số WB ISP đặt ở cận dưới của hồ sơ v1.2.0 (1,571 / 1,620), theo lệnh Owner.
- Để đối chiếu: hai nửa dev của cùng camera cho 0,84 (mục smoke ở trên). Nhưng tập so sánh khác nhau nên chỉ đọc thô.

| Tham số (khoá config) | Trước | Sau | Cận tìm |
|---|---|---|---|
| `haze.dark_scale` | 1,0 | 0,982 | 0,6–1,3 |
| tâm `haze.airlight_0_1` | 0,875 | 0,828 | 0,55–0,92 |
| tâm `lens.psf_sigma_scale` | 1,0 | 0,692 | 0,3–1,6 |
| tâm `lens.vignette_focal_scale` | 1,0 | **0,618** | 0,6–1,3 (sát cận dưới) |
| tâm `tone.brightness` | 1,025 | 1,326 | 0,7–1,5 |
| `tone.isp_wb_gain_r` | 1,571 | 1,988 | 1,571 (hồ sơ)–2,6 |
| `tone.isp_wb_gain_b` | 1,620 | 2,052 | 1,620 (hồ sơ)–2,6 |
| `tone.cast_r` | 1,0 | **0,960** | 0,96–1,04 (chạm cận dưới) |
| `tone.cast_b` | 1,0 | 0,993 | 0,96–1,04 |
| tâm `isp.soften_sigma` | 0,5 | 0,444 | 0–1,2 |
| tâm `isp.noise_scale` | 1,0 | 0,645 | 0,5–3,0 |

Khoảng cách từng đại lượng, trước → sau:

| Nhóm | Đại lượng |
|---|---|
| Giảm rõ | vignette 0,90 → 0,21; r/g 1,75 → 1,07; pink_frac 0,68 → 0,31; độ hồng vùng sáng 3,06 → 2,16; psd 0,70 → 0,40; dark channel 0,65 → 0,37; laplacian 0,59 → 0,40; nhiễu Immerkær 1,22 → 0,83; y_p1 0,88 → 0,57 |
| Tăng | b/g 1,13 → 1,21; blocking 0,81 → 1,05 (DQT cố định, không chỉnh); dải mép mean_y 0,40 → 0,54; nlf 1,40 → 1,46 |

Đo p50 trên 400 ảnh D2 của tập đủ, so với hồ sơ (cùng `measure_image`). Đây là các khâu đã lệch ở phase 05:

| Đại lượng | D2 phase 05 (50 ảnh) | D2 phase 07 | Thật (dev) |
|---|---|---|---|
| laplacian_var | 16,7 | 31,7 | 34,1 |
| sigma_immerkaer | 0,23 | 0,30 | 0,35 |
| vignette corner/center | 0,90 | 0,80 | 0,81 |
| y_p1 | 129 | 124 | 113 |
| tỉ lệ khung có > 0,5 % điểm hồng | 14 % | 25,5 % | 31 % |
| blocking_ratio_8x8 (không chỉnh) | 2,25 | 2,07 | 2,68 |

Cột phase 05 đo theo hồ sơ v1.1.0. Số thật của hai hồ sơ v1.1.0 và v1.2.0 trùng nhau ở các dòng trên.

### Vòng 2: nới cận, thêm mức đen dư (`params_version` 3)

Lệnh: `python -m xla.degrade.calibrate --split dev --budget 80 --workers 4` (khởi đầu từ `params_version` 2).

- **Lý do:** vòng 1 dừng sát cận ở vignette (0,618, cận 0,6) và `cast_r` (0,960 = cận). Vòng 2 nới cận vignette xuống
  0,45, `cast_r`/`cast_b` ra 0,88–1,08, brightness lên 1,6, `noise_scale` xuống 0,3, hệ số WB ISP lên 3,0, và thêm tham số `tone.black_level` (mức đen dư, khớp y_p1).
- **Kết quả:** khoảng cách trên dev **0,7359 → 0,7118** sau 80 lượt (12 tham số). Đạt điều kiện nhận (giảm so với vòng 1),
  nên `params_version` 2 → 3.
- **Đổi chỗ lệch:** b/g 1,21 → 0,51, pink_frac 0,31 → 0,20, y_p99 0,84 → 0,70 giảm; nhưng r/g 1,07 → 1,91 và vignette
  0,21 → 0,27 tăng. Tổng giảm nhẹ, không phải mọi đại lượng cùng tốt lên.
- **Máy:** chạy 16 rồi 8 worker thì máy sập 4 lần (kd đọc minidump: RAM lật bit dưới tải, `MEMORY_CORRUPTION_ONE_BIT`).
  Checkpoint ghi sau mỗi lượt nên không mất lượt nào; đoạn cuối chạy 4 worker, 37 phút, không sập.

| Tham số | Vòng 1 | Vòng 2 |
|---|---|---|
| `haze.dark_scale` | 0,982 | 0,962 |
| tâm `haze.airlight_0_1` | 0,828 | 0,869 |
| tâm `lens.psf_sigma_scale` | 0,692 | 0,655 |
| tâm `lens.vignette_focal_scale` | 0,618 | 0,679 |
| tâm `tone.brightness` | 1,326 | 1,313 |
| `tone.isp_wb_gain_r` / `_b` | 1,988 / 2,051 | 2,130 / 1,812 |
| `tone.cast_r` / `cast_b` | 0,960 / 0,993 | 0,998 / 0,963 |
| `tone.black_level` | 0 (chưa có) | 0,0017 |
| tâm `isp.soften_sigma` | 0,444 | 0,534 |
| tâm `isp.noise_scale` | 0,645 | 0,625 |

Hệ số Real-SR chọn lại với tham số mới (`calibrate --realsr-gains 0.3,0.4,0.5,0.6`): 0,3 → 0,666; **0,4 → 0,657**;
0,5 → 0,657; 0,6 → 0,673; D2 cùng đơn vị 0,711. Vẫn giữ 0,4.

### Tập sinh (seed 20261118)

Bốn tập D0, D1, D2, D2+Real-SR dùng **cùng** 3 493 đơn vị: 1 841 ảnh nguồn × 2 cửa sổ, bỏ 189 đơn vị không đặt được
cửa sổ.

- **Chia train/val:** train 3 006, val 487.
- **Có người:** train 2 394 ảnh (19 792 người), val 409 ảnh (2 716 người). Ngoài ra có 690 cửa sổ âm.
- **Val chia theo chuỗi:** chuyển 10/75 chuỗi train sang val (`configs/visdrone_valsplit.yaml`, seed 20261118). Val
  nguồn tăng từ 69 lên 238 ảnh có người.

| Tập | Cỡ ổ | sha256 manifest (16 ký tự đầu) |
|---|---|---|
| D0 | 433 MB | `c49dd534b60acb77` |
| D1 | 162 MB | `f9a765cd2191852b` |
| D2 (`params_version` 3) | 67 MB | `3ad4511041b070e1` |
| D2+Real-SR (`params_version` 3) | 74 MB | `1616c6c1f94a69a2` |

D2 và D2+Real-SR sinh lại ngày 18/11 sau vòng 2 (4 worker, 2,6 và 1,6 phút). Bản `params_version` 2 có manifest
`c7e2660690161f8c` (78 MB) và `0ee578bd9ea0f840` (85 MB), đã bị thay.

### D1 / D2 / D2+Real-SR so với khung test (`d1_vs_d2.json`)

Lệnh: `python -m xla.degrade.realism_compare --json`.

- **Phía thật:** 198 khung test, chỉ dùng ảnh, không dùng nhãn.
- **Phía tổng hợp:** 600 ảnh mỗi tập, chọn có seed; cùng tên ảnh ở mọi tập.
- **Tham số đo:** mỗi bên 1 000 patch, seed 20261118. KID tính 10 tập con rời nhau, mỗi tập 100 patch, CI 95 % bootstrap.
- **Hàng tham chiếu thật–thật:** 600 khung dev đặt vào chỗ tập tổng hợp, mọi thứ khác giữ nguyên.

| Hàng | Thống kê (W1/IQR) | DQT khớp | C2ST acc [CI95] | KID [CI95] |
|---|---|---|---|---|
| D1 (ngẫu nhiên) | 5,93 | không | 0,936 [0,909; 0,960] | 4,44 [4,29; 4,60] |
| D2 (`params_version` 3) | 0,80 | không* | 0,950 [0,930; 0,966] | 4,41 [4,24; 4,60] |
| D2 + Real-SR (`params_version` 3) | **0,66** | không* | 0,923 [0,897; 0,945] | **3,99 [3,82; 4,17]** |
| thật–thật (dev vs test) | 0,33 | có | 0,542 [0,494; 0,594] | 0,37 [0,29; 0,47] |

Bảng trên đo lại sau vòng 2. Với `params_version` 2, D2 là 0,81 / C2ST 0,960 / KID 4,37 [4,19; 4,56] và D2+Real-SR là 0,66 /
0,953 / 3,93 [3,75; 4,11]. Vòng 2 giảm khoảng cách trên dev nhưng trên test gần như không đổi: các kết luận dưới giữ nguyên.

\* 20 % ảnh D2 dùng bảng giả thuyết q 19–40 (`jpeg.real_q_weight` 0,8), nên tập bảng lượng tử khác tập thật.

**Kết luận, viết đúng như số ra:**

1. **Thống kê mức thấp:** D2 gần thật hơn D1 rất nhiều (0,81 so với 5,93). Real-SR kéo xuống thêm còn 0,66. Mức thật–thật
   là 0,33.
2. **KID:** D2 **không** thắng D1, vì CI chồng nhau (4,41 so với 4,44). D2+Real-SR thắng D1, CI không chồng
   ([3,82; 4,17] so với [4,29; 4,60]).
3. **Cả ba tập vẫn rất xa nền thật–thật:** KID khoảng 4 so với 0,37. C2ST ≥ 0,93 so với 0,54.
4. **C2ST không phân biệt ba tập:** D2 0,950, D1 0,936, D2+Real-SR 0,923 đều chồng CI. Bộ phân loại tách gần như hoàn toàn mọi tập tổng
   hợp khỏi khung thật.
5. **Câu hỏi §8 "Dữ liệu"** (KID(D2) < KID(D1), CI không chồng) có câu trả lời **không** với D2, **có** với D2+Real-SR.
   Phase 10 so trên nhận diện vẫn là phép thử chính.

Nền thật–thật ở đây (C2ST 0,54, KID 0,37) thấp hơn số smoke phase 06 (0,77 và 1,33). Hai phép đo khác cặp: ở đây là dev
(600 khung rải mọi block) so với test, còn smoke là nửa A so với nửa B chia theo block. Cặp ở đây trộn nhiều cảnh hơn.

### E3: D2 + Real-SR (Deferred 7, đã làm)

`xla/degrade/realsr.py` thay nhiễu tổng hợp ở bước (d) bằng phần dư lấy từ khung dev thật.

- **Pool:** 8 920 patch 32×32 từ 1 115 khung dev, 8 patch mỗi khung. Chỉ lấy ở vùng phẳng (30 % phẳng nhất của mỗi
  khung), thẳng lưới 8 px. Lệnh tạo: `python -m xla.degrade.realsr build --split dev`.
- **Phần dư** = patch trừ trung bình từng khối 8×8. Bước này bỏ bậc DC giữa các khối, là dấu khối JPEG chính ở vùng phẳng.
- **Khi dán:** lưới ô lệch khỏi lưới 8 px. Mỗi ô lấy patch cùng bin độ sáng. JPEG vẫn chỉ nén một lần ở cuối.
- **Hệ số nhân:** chọn trên dev bằng cùng hàm mục tiêu (`calibrate --realsr-gains`). Kết quả theo hệ số: 0,2 → 0,703;
  0,3 → 0,684; **0,4 → 0,672**; 0,5 → 0,672; 0,75 → 0,740; 1,0 → 0,830; 1,5 → 1,039. D2 cùng đơn vị là 0,736.

Cảnh báo khi đọc: phần dư lấy từ khung thật vốn đã qua JPEG q12. C2ST cũng nhìn đúng phần dư cao tần, nên một phần lợi thế
của Real-SR có thể đến từ việc mang dấu nén thật sang. Phase 10 (nhận diện) mới trả lời được nó có giúp hay không.

### Giờ GPU đo được (RTX 4060 Laptop)

| Lượt | Giờ GPU |
|---|---|
| Smoke: 1 lần KID 1 000 + 1 000 patch, không cache | 123 s (138 s tổng) |
| Lượt đủ, 4 hàng (C2ST + đặc trưng KID; đặc trưng phía thật dùng cache sau hàng đầu) | 235 s |
| Đo lại sau vòng 2 (`params_version` 3), 4 hàng | 208 s |

Cộng lại khoảng **0,16 giờ GPU**. VRAM đỉnh: 1,55 GB (KID), 0,37 GB (C2ST). Hiệu chỉnh và sinh tập chỉ chạy CPU.
