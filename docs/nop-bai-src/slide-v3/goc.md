<!-- Slide number: 1 -->
![uav_hero.png](Image 1)
ĐỒ ÁN XỬ LÝ ẢNH  ·  PTIT
Khôi phục
ảnh UAV
từ camera ESP32-CAM
Màu sắc
Nhiễu
Ánh sáng
Chi tiết
![pink_input.png](Image 14)
![dark_v2.png](Image 17)
![urban_roi_fhd.png](Image 20)
Sắc độ
Ánh sáng
Chi tiết
GVHD: PGS.TS. Phạm Văn Cường

### Notes:
Đồ án xây dựng pipeline thích nghi để sửa màu, giảm nhiễu và cải thiện ánh sáng, chi tiết ảnh UAV từ ESP32-CAM. Hình UAV trên bìa là minh họa ý tưởng; các ảnh nhỏ là dữ liệu và kết quả xử lý của dự án.

Nguồn: README.md; src/uav_pipeline.py; asset minh họa do image_gen tạo.

<!-- Slide number: 2 -->
NỘI DUNG
02 / 26
Nội dung trình bày
![uav_gt_001.jpg](Image 28)
![pink_mask.png](Image 33)
![dark_v2.png](Image 38)
![web_comparison.png](Image 43)
01
02
03
04
Bài toán và dữ liệu
Phương pháp
Thực nghiệm
Ứng dụng, kết luận
Dạng lỗi · 30 cặp GT · 60 ảnh UAV
Mặt nạ · lọc thích nghi · ánh sáng · SR
Ablation · PCI · PSNR · runtime
FastAPI · ảnh/video · demo · giới hạn

### Notes:
Bài trình bày gồm bốn phần: bài toán và dữ liệu; phương pháp; thực nghiệm; ứng dụng và kết luận. Phần phương pháp đi theo thứ tự xử lý của hệ thống, từ màu và nhiễu đến ánh sáng, sau đó mới tăng chi tiết.

Nguồn: Mạch nội dung của đồ án.

<!-- Slide number: 3 -->
01 · BÀI TOÁN
03 / 26
Ba nhóm suy giảm cần được xử lý riêng
![pink_heavy_input.png](Image 52)
![dark_input.png](Image 57)
![urban_roi_input.png](Image 62)
Ám hồng
Thiếu sáng
Mờ / JPEG
Vùng trung tính lệch sang hồng.
Mặt đường tối, dễ mất chi tiết.
Biên nhòe; nhiễu và khối nén.

### Notes:
Ba lỗi cần quan tâm là ám hồng, thiếu sáng và mờ hoặc nén JPEG. Chúng có thể xuất hiện đồng thời. Nếu tăng tương phản hoặc làm nét quá sớm, nhiễu và sai lệch màu cũng bị khuếch đại.

Nguồn: data/real_uav/esp32_real_013.jpg, 052.jpg; data/samples/sample_sim_pink_road_1.jpg.

<!-- Slide number: 4 -->
01 · DỮ LIỆU
04 / 26
Dữ liệu kiểm thử gồm hai nguồn đánh giá
30
60
cặp có ảnh chuẩn (GT)
ảnh UAV
![uav_gt_001.jpg](Image 73)
![uav_esp32_sim_001.jpg](Image 77)
Đa nguồn, không có GT đồng thời.
152
biến thể
GROUND TRUTH
SUY GIẢM MÔ PHỎNG
Dùng lại 30 ảnh GT, không phải 152 cảnh độc lập.

### Notes:
Tập có tham chiếu gồm 30 cặp ảnh chuẩn và ảnh suy giảm mô phỏng. Tập UAV gồm 60 ảnh không có ảnh chuẩn đồng thời. Có 152 biến thể được tạo lại từ 30 ảnh chuẩn, nên không coi đó là 152 cảnh độc lập. Manifest của tập UAV ghi 40 mẫu multisource_aug và 20 mẫu fpv_aug, không phải 60 lần chụp phần cứng độc lập.

Nguồn: data/dataset_manifest.csv; data/augmented/manifest.csv.

<!-- Slide number: 5 -->
01 · MÔ HÌNH SUY GIẢM
05 / 26
Mô hình suy giảm tạo cặp ảnh có tham chiếu
1
Veil / dải động
![uav_gt_001.jpg](Image 94)
![uav_esp32_sim_001.jpg](Image 98)
2
Lệch màu hồng
3
Blur + nhiễu
4
Nén JPEG
GT
ẢNH SUY GIẢM
![eq_veil.png](Image 121)
c: hệ số truyền   ·   A: ánh sáng môi trường   ·   V = (1−c)A: thành phần veil

### Notes:
Bộ sinh dữ liệu mô phỏng bốn nhóm tác động: veil hoặc co dải động, lệch hồng, blur và nhiễu, sau cùng là JPEG. Công thức mô hình hóa ảnh quan sát I từ ảnh sạch J. Đây là mô phỏng để tạo cặp có tham chiếu, không phải kết quả đo đặc tuyến cảm biến bằng thiết bị chuyên dụng.

Nguồn: src/data_augmentation.py; data/dataset_manifest.csv; data/augmented/manifest.csv.

<!-- Slide number: 6 -->
02 · PIPELINE V2
06 / 26
Pipeline v2 tổ chức thành ba giai đoạn
![processing_layers.png](Image 127)
01
02
03
Tiền xử lý
Ánh sáng
Chi tiết
Khử hồng · AWB
NLM / Bilateral / Guided
Veil · Retinex/LIME
Dark-channel · CLAHE
ESPCN · Richardson–Lucy
Sobel · kẹp min–max

### Notes:
Pipeline có ba giai đoạn. Giai đoạn 1 sửa ám hồng, cân bằng trắng có kiểm soát và giảm nhiễu. Giai đoạn 2 điều chỉnh ánh sáng, veil, khử sương có điều kiện và CLAHE. Giai đoạn 3 xử lý ESPCN trên Y, giải chập, làm nét có cổng biên và giới hạn viền giả. Hình lớp ảnh ở giữa là minh họa khái niệm.

Nguồn: src/uav_pipeline.py.

<!-- Slide number: 7 -->
02 · TIỀN XỬ LÝ
07 / 26
Hệ màu được chọn theo thao tác cần thực hiện
RGB / BGR
CIE LAB
YCrCb
![ch_R.png](Image 148)
![ch_G.png](Image 151)
![ch_B.png](Image 154)
![ch_L.png](Image 160)
![ch_la.png](Image 163)
![ch_lb.png](Image 166)
![ch_Y.png](Image 172)
![ch_Cr.png](Image 175)
![ch_Cb.png](Image 178)
R
G
B
L*
a*
b*
Y
Cr
Cb
Quan hệ giữa ba kênh để phát hiện dư sắc hồng.
CLAHE trên L*; kiểm soát độ bão hòa a*, b*.
Lọc và ESPCN trên Y; xử lý sắc độ riêng.
RGB: sửa màu      LAB: tương phản      YCrCb: lọc và siêu phân giải

### Notes:
Mỗi hệ màu phục vụ một thao tác. BGR là cách OpenCV sắp thứ tự các kênh RGB. LAB tách L* để điều chỉnh tương phản; YCrCb tách Y để lọc và siêu phân giải. L* của LAB và Y của YCrCb không phải cùng một đại lượng. Đổi hệ màu không tự tạo thêm chi tiết.

Nguồn: src/uav_pipeline.py: run_stage1_preprocess, run_stage2_adaptive, run_stage3_sharpen.

<!-- Slide number: 8 -->
02 · KHỬ ÁM HỒNG
08 / 26
Mặt nạ mềm chọn mức can thiệp vào vùng ám hồng
![pink_input.png](Image 187)
![pink_mask.png](Image 191)
![pink_stage1.png](Image 195)
ĐẦU VÀO
MẶT NẠ α ∈ [0,1]
SAU TIỀN XỬ LÝ
![eq_pink_e.png](Image 202)

### Notes:
Mức dư hồng e được tính bằng trung bình R, B trừ G, và chỉ lấy phần dương. Mặt nạ mềm alpha kết hợp điều kiện sáng và dư hồng, rồi được giãn và làm mượt. Alpha càng lớn thì mức sửa càng mạnh. Mã thực hiện hai lượt; đây vẫn là heuristic và có thể nhận nhầm vật thể hồng thật.

Nguồn: recover_pink_highlights trong src/uav_pipeline.py.

<!-- Slide number: 9 -->
02 · KHỬ ÁM HỒNG
09 / 26
Sửa sắc độ gần bảo toàn độ chói
ΔY ≈ 0
MỘT ĐIỂM ẢNH · MỘT LƯỢT SỬA
TRƯỚC
SAU
Y = 0,299R + 0,587G + 0,114B
Giữ độ chói ở phép dịch sắc độ, trước các bước kẹp và bù chi tiết.
180
140
180
158
156
158
R
G
B
R
G
B
![eq_pink_shift.png](Image 234)

### Notes:
Ví dụ một điểm ảnh có R, G, B là 180, 140, 180. Với alpha bằng 1, một lượt sửa cho kết quả xấp xỉ 158, 156, 158. Hệ số 0,704 bù vào G giúp gần bảo toàn độ chói Y trước kẹp giá trị, lượng tử hóa và bù high-pass. Pipeline đầy đủ còn có lượt sửa tiếp theo, AWB và lọc nhiễu.

Nguồn: recover_pink_highlights; hệ số độ chói BT.601 trong mã.

<!-- Slide number: 10 -->
02 · KHỬ NHIỄU
10 / 26
Ước lượng nhiễu từ Wavelet HH₁ để điều chỉnh NLM
MAD: trung vị của |HH₁|
TỪ ẢNH ĐẾN ƯỚC LƯỢNG NHIỄU
![eq_noise_sigma.png](Image 249)
![urban_input.png](Image 241)
![wavelet_hh.png](Image 242)
NLM: lọc theo mức nhiễu
Ảnh đầu vào
|HH₁| · Wavelet db1
![eq_noise_h.png](Image 252)
σ̂ ≈ 1.48   →   h ≈ 1.26

### Notes:
Ảnh bên trái và băng tần HH1 được lấy từ urban_input. Độ lớn HH1 được tăng tương phản chỉ để dễ nhìn. MAD cho sigma xấp xỉ 1,48 trên thang 0 đến 255; từ đó h của NLM xấp xỉ 1,26. Công thức làm cho cường độ lọc đi theo mức nhiễu ước lượng. Đây không phải phép đo trực tiếp nhiễu phần cứng.

Nguồn: estimate_noise_sigma; bilateral_denoise trong src/uav_pipeline.py.; presentation/prepare_v4_assets.py; assets/v4/method_figures.json.

<!-- Slide number: 11 -->
02 · KHỬ NHIỄU
11 / 26
Mặt nạ cấu trúc phân phối mức lọc theo từng vùng
Vùng phẳng
Biên / texture
Kênh Cr, Cb
NLM mạnh hơn
Bilateral nhẹ
Guided Filter theo Y
![denoise_input.png](Image 266)
![structure_w.png](Image 270)
w lớn
→ có cấu trúc, ưu tiên Bilateral nhẹ
w nhỏ
→ vùng phẳng, ưu tiên NLM; texture dư được giữ lại
TRƯỚC LỌC
TRỌNG SỐ w
![eq_noise_blend.png](Image 280)

### Notes:
Mặt nạ w được tính đúng theo nhánh lọc chính: gradient trên Y đã làm mượt, với ngưỡng phụ thuộc sigma của ảnh gốc. Màu vàng là w cao, ưu tiên Bilateral nhẹ; tím là w thấp, ưu tiên NLM. Mã còn giữ lại một phần texture và có nhánh mạnh hơn riêng cho vùng ám hồng. Công thức trên slide chỉ biểu diễn thành phần pha trộn chính. Cr và Cb dùng Guided Filter theo Y.

Nguồn: bilateral_denoise trong src/uav_pipeline.py.; presentation/prepare_v4_assets.py; assets/v4/method_figures.json.

<!-- Slide number: 12 -->
02 · ÁNH SÁNG
12 / 26
Thống kê độ sáng quyết định mức can thiệp
![low_light_003.jpg](Image 286)
![backlit_uneven_003.jpg](Image 291)
![haze_pink_003.jpg](Image 296)
Thiếu sáng
Ngược sáng
Sương / veil
p50 thấp · nhiều pixel tối
Vùng tối cùng tồn tại với highlight
Nền đen bị nâng · dải động hẹp
Các điểm số chỉ điều chỉnh cường độ; nhiều cơ chế có thể cùng hoạt động trên một ảnh.

### Notes:
Hệ thống dùng các phân vị sáng, tỷ lệ pixel tối, highlight và dark channel để tính điểm thiếu sáng, ngược sáng và sương. Các điểm này là trọng số liên tục; nhiều cơ chế có thể cùng hoạt động trên một ảnh. Ba ảnh minh họa được sinh từ cùng tập ảnh chuẩn.

Nguồn: analyze_scene; STAGE2_CFG trong src/uav_pipeline.py; data/augmented/.

<!-- Slide number: 13 -->
02 · ÁNH SÁNG
13 / 26
Bản đồ chiếu sáng nâng vùng tối và bảo vệ vùng sáng
![dark_stage1.png](Image 305)
![illumination_t.png](Image 308)
![dark_stage2.png](Image 311)
T thấp
T cao
TRƯỚC GIAI ĐOẠN 2
BẢN ĐỒ CHIẾU SÁNG T(x)
SAU GIAI ĐOẠN 2
![eq_illumination.png](Image 315)
T thấp → nâng vùng tối      T gần 1 → giữ vùng sáng      gain tối đa 5×

### Notes:
Ba ảnh lần lượt là đầu vào giai đoạn 2, bản đồ T thực sự được tính trước bước nâng sáng, và kết quả giai đoạn 2. Trên bản đồ T, tím biểu diễn chiếu sáng thấp, vàng biểu diễn chiếu sáng cao. T được làm mượt theo biên; gamma thích nghi theo phân vị 35 của T, còn gain giới hạn tối đa 5. Kết quả bên phải gồm cả các bước còn lại của giai đoạn 2, không phải chỉ một phép nhân gain.

Nguồn: _illumination_gain và run_stage2_adaptive trong src/uav_pipeline.py.; presentation/prepare_v4_assets.py; assets/v4/method_figures.json.

<!-- Slide number: 14 -->
02 · TƯƠNG PHẢN
14 / 26
CLAHE tác động trên L* và được pha trộn nhẹ
![clahe_demo_input.png](Image 333)
![clahe_demo_output.png](Image 337)
HISTOGRAM L* · 32 BINS
Trước
Sau

### Chart

| Category | Trước | Sau |
| --- | --- | --- |
| 0 | 0.0 | 0.0 |
| 8 | 0.0 | 0.0 |
| 16 | 0.0 | 0.0 |
| 24 | 0.0 | 0.0 |
| 32 | 0.0 | 0.0 |
| 40 | 2.0 | 13.0 |
| 48 | 18.0 | 259.0 |
| 56 | 365.0 | 971.0 |
| 64 | 1085.0 | 2529.0 |
| 72 | 2942.0 | 5673.0 |
| 80 | 7156.0 | 10498.0 |
| 88 | 12966.0 | 16894.0 |
| 96 | 19889.0 | 21707.0 |
| 104 | 25671.0 | 25597.0 |
| 112 | 32108.0 | 30490.0 |
| 120 | 36281.0 | 34930.0 |
| 128 | 41315.0 | 39018.0 |
| 136 | 45567.0 | 44946.0 |
| 144 | 50967.0 | 45975.0 |
| 152 | 44806.0 | 40746.0 |
| 160 | 35597.0 | 34687.0 |
| 168 | 28907.0 | 27399.0 |
| 176 | 21395.0 | 20616.0 |
| 184 | 20020.0 | 18340.0 |
| 192 | 19975.0 | 21186.0 |
| 200 | 8768.0 | 11635.0 |
| 208 | 11343.0 | 9352.0 |
| 216 | 12079.0 | 14518.0 |
| 224 | 743.0 | 1941.0 |
| 232 | 32.0 | 76.0 |
| 240 | 2.0 | 3.0 |
| 248 | 1.0 | 1.0 |

TRƯỚC CLAHE
PHA TRỘN 20%
0
255
![eq_clahe.png](Image 350)
clipLimit = 1,6   ·   lưới 8×8   ·   a*, b* được kiểm soát riêng

### Notes:
Ảnh trước và sau ở đây được tạo để tách riêng tác động CLAHE trên L của ảnh urban_stage1: clipLimit 1,6, lưới 8 nhân 8, pha trộn 20 phần trăm và giữ các kênh sắc độ. Histogram 32 bin lấy từ đúng hai ảnh này; trục ngang là L 8-bit của OpenCV. Trong pipeline đầy đủ, tỷ lệ pha trộn còn giảm ở ảnh tối. Không coi hình này là đầu ra toàn bộ giai đoạn 2.

Nguồn: prepare_expanded_assets.py; run_stage2_adaptive trong src/uav_pipeline.py.

<!-- Slide number: 15 -->
03 · CHI TIẾT
15 / 26
ESPCN xử lý Y; sắc độ được nội suy riêng
![urban_roi_stage2.png](Image 356)
![urban_roi_fhd.png](Image 360)
Y → ESPCN ×3
Cr / Cb → bicubic
Tile 224×224 · stride 200
Một batch cho toàn bộ tile
SAU GĐ1+2
SAU SIÊU PHÂN GIẢI
ESPCN ×3
Đầu ra giữ tỉ lệ 4:3
Pixel shuffle
960×720 / 1440×1080
Kênh Y
H × W × 9
3H × 3W × 1

### Notes:
ESPCN có sẵn dưới dạng ONNX, nhận Y đã chuẩn hóa của hệ YCrCb. Sơ đồ pixel shuffle minh họa cách 9 kênh đặc trưng cuối được sắp thành lưới có chiều cao, chiều rộng gấp 3. Mã gom tile 224 nhân 224, stride 200 vào một batch. Cr và Cb dùng bicubic. Sau SR còn kẹp quanh Lanczos, pha trộn và đổi kích thước đích; đầu ra 960 nhân 720 hoặc 1440 nhân 1080 giữ tỷ lệ 4:3.

Nguồn: _run_espcn_sr_luminance; upscale_to_hd trong src/uav_pipeline.py.

<!-- Slide number: 16 -->
03 · CHI TIẾT
16 / 26
Giải chập, cổng Sobel và giới hạn viền giả
![urban_stage2.png](Image 428)
![sobel_gradient.png](Image 432)
![sharpen_detail.png](Image 436)
ẢNH TRƯỚC HẬU XỬ LÝ
GRADIENT SOBEL
CHI TIẾT BÙ TRÊN Y
![eq_clamp.png](Image 441)
ΔY: lượng bù chi tiết   ·   N: vùng lân cận   ·   Richardson–Lucy: 2 vòng, PSF 5×5

### Notes:
Hình giữa là độ lớn gradient Sobel để minh họa cấu trúc, không phải toàn bộ mặt nạ active_w. Hình phải biểu diễn lượng bù chi tiết với mức 128 là không bù và độ lệch được khuếch đại 4 lần cho dễ nhìn. Richardson–Lucy mặc định 2 vòng với PSF Gaussian 5 nhân 5. Kẹp theo min, max lân cận giúp hạn chế overshoot và viền giả.

Nguồn: run_stage3_sharpen; prepare_expanded_assets.py.

<!-- Slide number: 17 -->
04 · THỰC NGHIỆM
17 / 26
Đối chứng và ablation tách vai trò từng giai đoạn
PSNR (dB) · 30 CẶP SYNTHETIC GỐC
PSNR / SSIM

### Chart

| Category | PSNR |
| --- | --- |
| Đầu vào | 20.35 |
| v1 | 27.42 |
| GĐ1 v2 | 20.43 |
| GĐ1+2 | 25.28 |
| v2 đầy đủ | 25.39 |

Độ gần ảnh chuẩn và tương đồng cấu trúc.
PCI / vùng đen
Dấu hiệu ám hồng và pixel có Y ≤ 3.

### Notes:
Biểu đồ ablation dùng 30 cặp synthetic gốc. Phần lớn mức tăng PSNR của v2 xuất hiện sau giai đoạn ánh sáng; giai đoạn chi tiết tăng từ 25,28 lên 25,39 dB. V1 vẫn có PSNR cao hơn trên tập này. PSNR và SSIM cần ảnh chuẩn; PCI và tỷ lệ vùng đen có thể tính khi không có ảnh chuẩn. Không kết luận mọi độ đo đều cải thiện ở mọi bước.

Nguồn: results/tables/table1_synthetic_metrics.csv; compute_psnr, compute_ssim, compute_pink_cast_index.

<!-- Slide number: 18 -->
04 · THỰC NGHIỆM
18 / 26
Trên 60 ảnh UAV, PCI trung bình giảm 92,1%
PCI TRUNG BÌNH (%) · 60 ẢNH UAV
![pink_input.png](Image 461)

### Chart

| Category | PCI (%) |
| --- | --- |
| Đầu vào | 17.45 |
| v1 | 7.03 |
| v2 | 1.38 |

TRƯỚC
![pink_v2.png](Image 465)
SAU

### Notes:
PCI trung bình trên 60 ảnh giảm từ 17,45 phần trăm xuống 1,38 phần trăm, tương đương giảm tương đối khoảng 92,1 phần trăm. V1 đạt 7,03 phần trăm. Hai ảnh bên phải là một trường hợp minh họa, không đại diện cho mọi cảnh. PCI là quy tắc phát hiện dư hồng và có thể tính cả vật thể hồng thật.

Nguồn: table5_before_after_real60.csv; data/real_uav/esp32_real_001.jpg.

<!-- Slide number: 19 -->
04 · TRƯỜNG HỢP KHÓ
19 / 26
Ảnh tối 052: v2 giữ lại vùng đường bị v1 làm mất
ĐẦU VÀO
v1
v2
![dark_input.png](Image 475)
![dark_v1.png](Image 481)
![dark_v2.png](Image 487)
0%
40,72%
0%
pixel đen · L* trung bình 22,14
pixel đen · L* trung bình 5,66
pixel đen · L* trung bình 36,31

### Notes:
Ở ảnh tối 052, v1 làm 40,72 phần trăm pixel về Y nhỏ hơn hoặc bằng 3; v2 là 0 phần trăm. L* trung bình lần lượt là 22,14 ở đầu vào, 5,66 với v1 và 36,31 với v2. Các số lấy từ CSV đã lưu; đây là một ca minh họa, không phải bảo đảm cho mọi ảnh tối.

Nguồn: table6_real_dark_pink_cases.csv.

<!-- Slide number: 20 -->
04 · THỰC NGHIỆM
20 / 26
V2 cải thiện các cảnh khó, nhưng chưa vượt mọi đầu vào
Đầu vào
v1
v2
PSNR (dB) ↑

### Chart

| Category | Đầu vào | v1 | v2 |
| --- | --- | --- | --- |
| Synthetic gốc | 20.35 | 27.42 | 25.39 |
| Sương + hồng | 15.84 | 20.36 | 25.22 |
| Thiếu sáng | 14.32 | 10.82 | 18.25 |
| Ngược sáng | 16.86 | 13.51 | 17.24 |
| Mờ + JPEG | 28.67 | 21.54 | 24.5 |
| Hình học | 18.63 | 16.1 | 21.0 |

+9,38 dB
Sương + hồng, so với đầu vào
Mờ / JPEG: đầu vào > v2

### Notes:
V2 có PSNR cao hơn v1 ở cả năm nhóm augmented. So với đầu vào, nhóm sương cộng hồng tăng 9,38 dB và nhóm thiếu sáng tăng 3,93 dB. Tuy nhiên, synthetic gốc có lợi cho v1; nhóm motion JPEG vẫn có đầu vào tốt hơn v2. Các biến thể dùng chung ảnh chuẩn nên không tuyên bố ý nghĩa thống kê hoặc độc lập dữ liệu.

Nguồn: table3_before_after_synthetic30.csv; table4_before_after_augmented.csv; table7_augmented_by_gt_brightness.csv.

<!-- Slide number: 21 -->
04 · HIỆU NĂNG
21 / 26
Benchmark CPU tuần tự: v2 mất khoảng 5,49 giây/ảnh
4,06 s
5,49 s
v1 · giây / ảnh
v2 · giây / ảnh
PHÂN BỔ THỜI GIAN CỦA V2
GĐ1 · 2.23 s
GĐ2 · 1.00 s
GĐ3 · 2.26 s
Tiền xử lý
Ánh sáng
Chi tiết
12 ảnh · 800×600 · một tiến trình · CPU tuần tự

### Notes:
Benchmark đo tuần tự 12 ảnh UAV, kích thước 800 nhân 600, trong một tiến trình. V1 mất trung bình 4,06 giây mỗi ảnh; v2 mất 5,49 giây. Ba giai đoạn của v2 lần lượt khoảng 2,23, 1,00 và 2,26 giây. Bảng không lưu đầy đủ cấu hình CPU; không so trực tiếp với thời gian hiển thị của một lượt demo trên web.

Nguồn: table8_runtime_serial.csv; per_image/runtime_serial.csv.

<!-- Slide number: 22 -->
05 · ỨNG DỤNG
22 / 26
FastAPI điều phối ảnh đồng bộ và video theo job nền
Upload  →  FastAPI
![web_comparison.png](Image 535)
Ảnh: xử lý đồng bộ
Video: job nền, tiến độ theo frame
Pipeline v2  →  PNG / MP4
OpenCV + ONNX Runtime CPU

### Notes:
Ứng dụng FastAPI xử lý ảnh theo yêu cầu đồng bộ và trả các kết quả từng giai đoạn. Video chạy bằng worker thread nền, cập nhật tiến độ theo frame; registry job nằm trong bộ nhớ. Đầu ra là PNG hoặc MP4. Đây là bản demo; sơ đồ không mô tả một hệ hàng đợi phân tán hoặc hệ thống triển khai production.

Nguồn: web/server.py; web/index.html; src/uav_pipeline.py.

<!-- Slide number: 23 -->
05 · DEMO ẢNH
23 / 26
Demo ảnh: đối sánh trực tiếp trên giao diện
![web_result.png](Image 550)
PCI · MẪU DEMO
32,53% → 0,97%
ĐỘ PHÂN GIẢI
800×600 → 960×720
Kéo đối sánh · xem từng giai đoạn · tải ảnh

### Notes:
Ảnh chụp giao diện do người dùng cung cấp, trùng với assets/web_result.png. Giữ nguyên đầy đủ ảnh, nút điều khiển và chỉ số. Bên trái là ảnh gốc ám hồng; bên phải là kết quả. PCI của riêng mẫu này giảm từ 32,53% xuống 0,97%; đầu ra 960×720. Các chỉ số trong ảnh thuộc lượt demo này; thời gian hiển thị trên giao diện không thay thế benchmark CPU tuần tự của slide hiệu năng.

Nguồn: Ảnh giao diện người dùng cung cấp; presentation/assets/web_result.png.

<!-- Slide number: 24 -->
05 · DEMO
24 / 26
Demo video: đối sánh ảnh trước / sau
![](Embedded comparison video)
demo_urban_uav_SideBySide_FullHD_1080p.mp4

### Notes:
Video được nhúng là D:\XLA\results\video\demo_urban_uav_SideBySide_FullHD_1080p.mp4. File có 40 frame ở 24 fps, dài khoảng 1,67 giây. Video dùng để minh họa đối sánh; nhãn thuật toán trong video có thể kế thừa phiên bản trước. Khi giải thích phương pháp, dùng nội dung v2 trên các slide. PDF chỉ hiển thị ảnh poster.

Nguồn: results/video/demo_urban_uav_SideBySide_FullHD_1080p.mp4.

<!-- Slide number: 25 -->
TÀI LIỆU THAM KHẢO
25 / 26
Tài liệu phương pháp và nguồn thực nghiệm
Lọc và tương phản
Ánh sáng và khử sương
Buades, Coll & Morel (2011). Non-Local Means Denoising. IPOL.
Zuiderveld (1994). Contrast Limited Adaptive Histogram Equalization.
Guo, Li & Ling (2017). LIME. IEEE TIP, 26(2), 982–993.
He, Sun & Tang (2009/2010). Dark Channel Prior; Guided Image Filtering.
Giải chập và siêu phân giải
Mã và dữ liệu
Richardson (1972). Bayesian-Based Iterative Method of Image Restoration.
Shi et al. (2016). Efficient Sub-Pixel CNN. CVPR, 1874–1883.
github.com/Localhost218/XLA
src/  ·  notebooks/  ·  results/tables/

### Notes:
Các thuật toán được tham khảo từ các công trình được ghi trên slide rồi điều chỉnh cho đồ án; không tuyên bố tái hiện nguyên bản mọi phương pháp. Số liệu đến từ các CSV trong dự án. Hình UAV và lớp ảnh chỉ là minh họa; các bản đồ Wavelet, cấu trúc và chiếu sáng được tạo từ dữ liệu bằng mã đi kèm.

Nguồn: Các tài liệu ghi trên slide; kho mã XLA.

<!-- Slide number: 26 -->
![uav_hero.png](Image 584)
KẾT LUẬN
26 / 26
Kết quả và hướng phát triển
−92,1%
Giới hạn
Không thắng mọi nhóm · dữ liệu còn dẫn xuất · CPU tuần tự chưa thời gian thực.
PCI trung bình · 60 ảnh UAV
+3,93 dB
Tiếp theo
Cảnh bay độc lập · tối ưu hiệu năng · ổn định video · đánh giá tác vụ sau xử lý.
PSNR nhóm thiếu sáng

### Notes:
Kết quả nổi bật là giảm PCI trung bình 92,1 phần trăm trên tập 60 ảnh và tăng PSNR 3,93 dB ở nhóm thiếu sáng so với đầu vào. Hệ thống đã có giao diện ảnh và video. Giới hạn gồm dữ liệu dẫn xuất, không thắng mọi nhóm và CPU tuần tự chưa đạt thời gian thực. Hướng tiếp theo là thêm cảnh bay độc lập, tối ưu hiệu năng, đánh giá ổn định video và tác vụ sau xử lý.

Nguồn: table3–table8; src/uav_pipeline.py; web/server.py.
