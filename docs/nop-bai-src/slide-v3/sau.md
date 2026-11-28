<!-- Slide number: 1 -->
![uav_hero.png](Image 1)
ĐỒ ÁN XỬ LÝ ẢNH  ·  PTIT  ·  NHÓM 10
Làm rõ ảnh UAV
và tìm người
từ camera ESP32-S3 + OV3660
Ám hồng
Mù sương
Khối JPEG
Tìm người
![pink_input.png](Image 14)
![dark_v2.png](Image 17)
![urban_roi_fhd.png](Image 20)
Khung gốc
Cổ điển
Mạng NAFNet
GVHD: PGS.TS. Phạm Văn Cường

### Notes:
Đồ án làm hai việc trên khung bay thật của camera ESP32-S3 + OV3660 (JPEG 640×480, chất lượng 12): làm rõ ảnh cho người vận hành và phát hiện người để cảnh báo. Ba ảnh nhỏ là cùng một khung bay ngày 02/11/2026: khung gốc, sau bộ lọc cổ điển và sau mạng NAFNet. Hình drone trên bìa chỉ là minh hoạ.

Nguồn: BAO_CAO_DO_AN.md (Tóm tắt); khung 20261102-100941_005_00934.

<!-- Slide number: 2 -->
NỘI DUNG
02 / 23
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
Tích hợp, kết luận
Camera thật · 3 kiểu lỗi · 198 khung test
D1/D2 · YOLO11s · cổ điển · NAFNet
E1 · E2 · A/B mù · E4
IOT-CV · ms · giới hạn · phân công

### Notes:
Bốn phần: bài toán và dữ liệu thật; phương pháp (bộ làm xấu D1/D2, bộ nhận diện YOLO11s, bộ lọc cổ điển, mạng NAFNet); thực nghiệm E1, E2, chấm mù A/B, E4; cuối cùng là chạy thật trong phần mềm drone IOT-CV, giới hạn và phân công.

Nguồn: BAO_CAO_DO_AN.md.

<!-- Slide number: 3 -->
01 · BÀI TOÁN
03 / 23
Camera thật làm hỏng ảnh theo ba kiểu đo được
![pink_heavy_input.png](Image 52)
![dark_input.png](Image 57)
![urban_roi_input.png](Image 62)
Ám hồng
Mù + tối góc
JPEG + mép trái
Kênh G sụt khi điểm ảnh gần bão hoà.
Dark channel p50 144/255; góc 0,81× tâm.
Khối 8×8, 4:2:2; dải tối ~14 px.

### Notes:
Hồ sơ camera đo trên 1115 khung dev: ám hồng ở vùng gần bão hoà (31 % khung có hơn 0,5 % điểm hồng), dark channel trung vị 143,7/255 (mù như sương), góc ảnh chỉ sáng 0,81 lần tâm, tỉ số khối 8×8 trung vị 2,68 do JPEG q12 lấy mẫu 4:2:2, và dải tối khoảng 14 px ở mép trái (hệ thống cắt 16 px). Ảnh phải là vùng mép trái phóng to 4 lần. Bản trước của nhóm viết cho ESP32-CAM OV2640 và chấm trên ảnh tự sinh; camera thật khác loại.

Nguồn: results/camera/profile_report.md; configs/camera_profile.yaml.

<!-- Slide number: 4 -->
01 · DỮ LIỆU
04 / 23
Chỉ chấm điểm trên khung bay thật, chia theo khối thời gian
198
100
khung test, 39 khung có người
khung dev-val
![uav_gt_001.jpg](Image 73)
![uav_esp32_sim_001.jpg](Image 77)
Chỉ để chọn model và ngưỡng.
459
khung âm
NHÃN NGƯỜI (GT)
KHUNG KHÔNG NGƯỜI
300 trong nhà + 159 khung bay, để đếm báo nhầm.

### Notes:
Chuyến bay 02/11/2026, phiên 20261102-100941. Khung chọn cách nhau ít nhất 1 giây và chia theo khối thời gian 30 giây, không chia ngẫu nhiên, để khung test không có khung gần giống trong tập huấn luyện. Tập test 198 khung (39 khung có người, 149 hộp người); dev-val 100 khung chỉ dùng chọn model và ngưỡng; thêm 300 khung âm trong nhà và 159 khung bay không người để đếm báo nhầm. Nhãn do nhiều model và SAHI gán trước, soát từng khung bằng mắt, chủ dự án duyệt lần cuối trên Label Studio.

Nguồn: BAO_CAO_DO_AN.md mục 2; data/real/split.json.

<!-- Slide number: 5 -->
02 · BỘ LÀM XẤU
05 / 23
D2 sao chép camera để làm xấu ảnh huấn luyện
1
Mù theo mô hình khí quyển
![uav_gt_001.jpg](Image 94)
![uav_esp32_sim_001.jpg](Image 98)
2
PSF + tối góc cos⁴
3
Cắt kênh → ám hồng, nhiễu
4
Mép trái + JPEG DQT q12
ẢNH NGUỒN
SAU D2
![eq_veil.png](Image 121)
D1: chuỗi Real-ESRGAN, tham số ngẫu nhiên   ·   D2: tham số hiệu chỉnh tự động cho khớp khung dev thật

### Notes:
D1 là cách làm phổ biến khi không biết camera đích: chuỗi suy giảm kiểu Real-ESRGAN với tham số rút ngẫu nhiên. D2 là bản sao camera: màn mờ theo mô hình khí quyển (công thức), PSF và tối góc cos⁴ theo tiêu cự đo được, cắt từng kênh rồi nhân cân bằng trắng nên vùng sáng thành hồng, nhiễu theo NLF, dải mép trái, rồi nén JPEG một lần bằng đúng bảng lượng tử của camera ở q12, 4:2:2. Tham số được hiệu chỉnh tự động (xla.degrade.calibrate). Ảnh nguồn huấn luyện là VisDrone; NOMAD đã cân nhắc nhưng loại vì ảnh chụp xiên 45–60°. Ảnh VisDrone không được đăng lại, nên ảnh minh hoạ ở đây lấy nguồn là một khung bay của nhóm sau bộ lọc cổ điển rồi cho qua D2.

Nguồn: xla/degrade/d2_camera.py; configs/degrade_d2.yaml; BAO_CAO_DO_AN.md mục 3.1.

<!-- Slide number: 6 -->
02 · ĐỘ GIỐNG THẬT
06 / 23
D2 khớp thống kê pixel, nhưng bộ phân loại vẫn tách được
Lệch thống kê
C2ST
KID ×100
5,93 → 0,80
0,94 · 0,95
4,44 · 4,41
D1 → D2, tổng độ lệch 18 thống kê mức thấp. Thật–thật: 0,33.
D1 · D2. Mức 0,5 là không phân biệt được. Thật–thật: 0,54.
D1 · D2, trên đặc trưng DINOv2. Thật–thật: 0,37.
Khớp thống kê là cần nhưng chưa đủ: nội dung ảnh VisDrone vẫn khác bãi bay của nhóm.

### Notes:
Ba thước đo chạy trên patch ảnh. Lệch thống kê: D1 5,928, D2 0,796, D2 + Real-SR 0,663, mức sàn thật–thật 0,330. C2ST là độ chính xác của bộ phân loại phân biệt patch thật với patch làm xấu: D1 0,936, D2 0,950, thật–thật 0,542. KID ×100: D1 4,443, D2 4,409, thật–thật 0,371. D2 sao chép camera đúng ở mức pixel, nhưng bộ phân loại vẫn tách được; nhiều khả năng do nội dung VisDrone (phố đô thị) khác bãi bay, giả thuyết này chưa kiểm riêng.

Nguồn: results/reproduce/tables/t2_do_giong_that.md.

<!-- Slide number: 7 -->
02 · HỆ THỐNG
07 / 23
Nhận diện chạy trên khung gốc, ảnh rõ chỉ cho người xem
![processing_layers.png](Image 127)
01
02
03
Hồ sơ camera
Nhận diện
Làm rõ
DQT q12 · tối góc · hồng
đo trên 1115 khung dev
YOLO11s học trên ảnh D2
tự huấn luyện · ONNX 1280
Cổ điển 9,2 ms trên CPU
NAFNet-w16 20,7 ms trên GPU

### Notes:
Hồ sơ camera là nền của mọi bước: bộ làm xấu D2, bộ lọc cổ điển và cả khung giả trong web dùng chung các số đo này. Bộ nhận diện YOLO11s học trên ảnh làm xấu D2 rồi tự huấn luyện trên khung bay chưa nhãn, xuất ONNX khung 1280. Nhánh làm rõ có bộ lọc cổ điển và mạng NAFNet bề rộng 16. Quy tắc D5: nhận diện luôn chạy trên khung gốc, ảnh rõ chỉ để người xem. Hình lớp ảnh ở giữa là minh hoạ khái niệm.

Nguồn: docs/hop-dong-model.md §2–3; BAO_CAO_DO_AN.md mục 3.

<!-- Slide number: 8 -->
02 · NHẬN DIỆN
08 / 23
YOLO11s học ảnh D2, rồi tự huấn luyện trên khung bay
![urban_roi_stage2.png](Image 356)
![urban_roi_fhd.png](Image 360)
YOLO11s · 1280 px
E1: D0 · D1 · D2 · D2+SR
Chọn model: mAP50 dev-val
Ngưỡng: F1 tối đa dev-val
NHÃN TAY
DETECTOR-V1.0.0
E2: thầy YOLO11m
Xuất ONNX, resize trong graph
Trò YOLO11s
ship trò vòng 1 theo dev-val
gán nhãn giả khung bay

### Notes:
Mốc là YOLO11s COCO có sẵn, khung 1280, bỏ 16 px mép trái. E1 tinh chỉnh YOLO11s trên VisDrone qua từng bộ làm xấu: D0 ảnh sạch, D1, D2, D2 + siêu phân giải; chọn model bằng mAP50 trên dev-val. E2: thầy YOLO11m gán nhãn giả cho khung bay chưa nhãn, trò YOLO11s học lại hai vòng; ship trò vòng 1 theo dev-val, xuất ONNX có cắt mép và resize trong graph, ngưỡng vận hành 0,077 chọn bằng F1 tối đa trên dev-val. Hình phải là hộp của chính bundle detector-v1.0.0 trên một khung test, hình trái là nhãn tay.

Nguồn: configs/train_e1.yaml; configs/selftrain_e2.yaml; models/release/detector-v1.0.0/meta.json.

<!-- Slide number: 9 -->
02 · CỔ ĐIỂN
09 / 23
Bộ lọc cổ điển chạy ngược chuỗi suy giảm của camera
![low_light_003.jpg](Image 286)
![backlit_uneven_003.jpg](Image 291)
![haze_pink_003.jpg](Image 296)
Gỡ khối + hồng
Bù tối góc
Khử mù + nét
Khử khối 8×8 ngay sau giải mã, rồi khử hồng.
Chia cho bản đồ cos⁴, tiêu cự 1016 px.
Dark channel 15 px, unsharp trên Y.
Bỏ siêu phân giải và giải chập của bản trước: trên khung thật chúng làm nổi khối JPEG.

### Notes:
Thứ tự bước đảo chuỗi suy giảm của camera: JPEG, rồi cắt kênh và cân bằng trắng của ISP, rồi tối góc, rồi màn mờ. Khử khối đặt ngay sau giải mã giảm tỉ số khối còn 2,15; khử hồng trước khử mù giảm 93 % điểm hồng, đặt sau chỉ giảm 46–68 % vì khử mù nhân chênh lệch màu lên. Ba ảnh là kết quả cộng dồn sau từng bước trên một khung bay có ám hồng. Bộ lọc giữ ý tưởng mặt nạ hồng và cân bằng trắng có loại trừ của bản trước.

Nguồn: xla/enhance/classic.py (docstring, STAGES); configs/enhance_classic.yaml.

<!-- Slide number: 10 -->
02 · KHỬ ÁM HỒNG
10 / 23
Mặt nạ mềm chỉ chọn vùng hồng gần bão hoà
![pink_input.png](Image 187)
![pink_mask.png](Image 191)
![pink_stage1.png](Image 195)
ĐẦU VÀO
MẶT NẠ α ∈ [0,1]
SAU KHỬ HỒNG
![eq_pink_e.png](Image 202)

### Notes:
Cổng gồm hai điều kiện: max(R, B) gần bão hoà (từ 230) và cả R lẫn B vượt G (từ 6 mức). Vật đỏ bão hoà có B thấp nên không bị đụng. Cổng được nở 21 px và làm mềm thành mặt nạ alpha; mỗi lượt sửa trừ delta khỏi R và B, cộng 0,704 delta vào G; chạy hai lượt. Ảnh giữa là alpha thật của lượt đầu trên khung này.

Nguồn: demagenta trong xla/enhance/classic.py; configs/enhance_classic.yaml (magenta).

<!-- Slide number: 11 -->
02 · KHỬ ÁM HỒNG
11 / 23
Sửa sắc độ gần bảo toàn độ chói
ΔY ≈ 0
MỘT ĐIỂM GẦN BÃO HOÀ · α = 1 · MỘT LƯỢT
TRƯỚC
SAU
Y = 0,299R + 0,587G + 0,114B
δ = 0,56·α·e; hệ số 0,704 bù vào G giữ Y gần như không đổi.
245
200
240
221
217
216
R
G
B
R
G
B
![eq_pink_shift.png](Image 234)

### Notes:
Ví dụ một điểm gần bão hoà R, G, B = 245, 200, 240, alpha bằng 1: e = (245 + 240)/2 − 200 = 42,5; delta = 0,56 × 42,5 ≈ 23,8. Sau một lượt: R ≈ 221, G ≈ 217, B ≈ 216. Độ chói Y trước 218,0, sau 218,1. Pipeline thật còn lượt thứ hai và bơm high-pass của G ở vùng bão hoà mạnh.

Nguồn: demagenta trong xla/enhance/classic.py.

<!-- Slide number: 12 -->
02 · KHỬ MÙ
12 / 23
Dark channel ước lượng độ truyền t(x) để khử mù
![dark_stage1.png](Image 305)
![illumination_t.png](Image 308)
![dark_stage2.png](Image 311)
t thấp
t cao
TRƯỚC KHỬ MÙ
BẢN ĐỒ t(x) THÔ
SAU KHỬ MÙ + LÀM NÉT
![eq_illumination.png](Image 315)
patch 15 px   ·   ω = 0,85   ·   A: phân vị 99,9 của dark channel   ·   t tinh chỉnh bằng guided filter

### Notes:
Dark channel là min của ba kênh rồi lấy min trong cửa sổ 15 px. Ánh sáng môi trường A lấy trung bình các điểm có dark channel từ phân vị 99,9, kẹp trong 180–250. Độ truyền t = 1 − 0,85·dc/A, sàn 0,4, tinh chỉnh bằng guided filter theo Y ở 1/4 độ phân giải. Bản đồ giữa là t thô trước guided filter; tím là t thấp (mù dày). Ảnh phải gồm cả bước làm nét unsharp trên Y. Trên tập test bộ cổ điển giảm dark channel trung vị 51,6 %.

Nguồn: dehaze trong xla/enhance/classic.py; He, Sun & Tang (2011).

<!-- Slide number: 13 -->
03 · THỰC NGHIỆM
13 / 23
E1: D2 hơn D1 về mAP50, nhưng CI còn chạm 0
MAP50 · 198 KHUNG TEST THẬT
Báo nhầm

### Chart

| Category | mAP50 |
| --- | --- |
| Mốc COCO | 0.126 |
| D0 sạch | 0.297 |
| D1 ngẫu nhiên | 0.143 |
| D2 camera | 0.236 |
| D2 + SR | 0.149 |

D2: 0,30 hộp/khung bay không người, D1: 0,99. Chênh có CI trọn dưới 0.
Chọn theo dev-val
Dev-val chọn D2 (0,329). D0 cao trên test nhưng báo nhầm 1,20.

### Notes:
mAP50 trên 198 khung test: mốc YOLO11s COCO 0,126; D0 0,297; D1 0,143; D2 0,236; D2 + siêu phân giải 0,149. D2 hơn D1 0,093 mAP50 nhưng CI 95 % [−0,073; +0,252] chạm 0, nên chưa đủ để nói D2 hơn về mAP. Chỗ D2 hơn rõ là báo nhầm: 0,302 so với 0,994 hộp sai mỗi khung bay không người, chênh −0,692 [−1,238; −0,246]. D0 có CI rất rộng và báo nhầm cao. CI bằng bootstrap theo khối thời gian, 2000 mẫu.

Nguồn: results/reproduce/tables/t1_nhan_dien.md; results/e1/README.md.

<!-- Slide number: 14 -->
03 · THỰC NGHIỆM
14 / 23
E2: dev-val chọn trò vòng 1, test lại nghiêng về vòng 2
MAP50 · 198 KHUNG TEST THẬT
Dev-val chọn sai

### Chart

| Category | mAP50 |
| --- | --- |
| E1 D2 | 0.236 |
| Thầy YOLO11m | 0.218 |
| Trò vòng 1 | 0.156 |
| Trò vòng 2 | 0.313 |
| Bundle ONNX | 0.12 |

Dev-val vòng 2 là 0,296, thấp hơn vòng 1 (0,379), nên theo luật ship vòng 1.
Bundle đã ship
detector-v1.0.0: mAP50 0,120, lệch bản .pt −0,003; 20,0 ms p50 ORT.

### Notes:
Thầy YOLO11m 0,218; trò YOLO11s vòng 1 0,156; vòng 2 0,313 cao nhất trên test nhưng dev-val của nó thấp hơn vòng 1 (0,296 so với 0,379), nên theo luật đã chốt nhóm ship vòng 1. Bundle ONNX chấm bằng chính đường ONNX (cắt mép, letterbox 1280 trong graph) đạt 0,120 [0,032; 0,197], báo nhầm 0,881 hộp mỗi khung bay không người. ORT CUDA chạy riêng 20,0 / 20,2 ms p50 / p95. Bài học chính: 100 khung dev-val quá ít để xếp hạng các model sát nhau.

Nguồn: results/reproduce/tables/t1_nhan_dien.md; results/e2/README.md; models/release/detector-v1.0.0/meta.json.

<!-- Slide number: 15 -->
03 · LÀM RÕ
15 / 23
Cổ điển và mạng trên cùng một khung bay
KHUNG GỐC
CỔ ĐIỂN
MẠNG NAFNET-W16
![dark_input.png](Image 475)
![dark_v1.png](Image 481)
![dark_v2.png](Image 487)
q12
9,2 ms
20,7 ms
JPEG 640×480 từ camera
CPU · hồng −89 % · mù −52 %
GPU · hồng −99,9 % · mù −37 %

### Notes:
Bộ cổ điển chạy 9,2 ms mỗi khung trên CPU (p50), giảm điểm hồng 89 % và dark channel 52 %. Mạng NAFNet bề rộng 16 học từ cặp ảnh sạch và ảnh làm xấu D2, chạy 20,7 ms trên GPU, xoá 99,9 % điểm hồng nhưng màu ra nguội và nhạt, dark channel chỉ giảm 37 % (cần 50 %). Nhấp nháy giữa hai khung liền nhau tăng ở cả hai: mạng 1,81 lần, cổ điển 2,18 lần. Mạng đạt 3/7 tiêu chí đăng ký trước.

Nguồn: results/reproduce/tables/t3_tieu_chi_anh_ro.md; thuc-thi-phase-10-14.md §7.

<!-- Slide number: 16 -->
03 · THỰC NGHIỆM
16 / 23
Chấm mù A/B: người chấm chọn bản cổ điển 24/25 lượt
LƯỢT MẠNG THẮNG / 25 · L1 CẢ KHUNG · L2 PHÓNG TO 2×
![pink_input.png](Image 461)

### Chart

| Category | Lượt thắng |
| --- | --- |
| L2 vs cổ điển | 5.0 |
| L2 vs gốc | 13.0 |
| L1 vs cổ điển | 1.0 |
| L1 vs gốc | 11.0 |

GỐC
![pink_v2.png](Image 465)
MẠNG

### Notes:
Người chấm thấy hai ảnh cạnh nhau, trái phải xáo trộn, chọn ảnh dễ nhìn hơn. Lần 1, cả khung: mạng thắng ảnh gốc 11/25 (0,44, Wilson [0,267; 0,629]), thắng cổ điển 1/25, tức cổ điển thắng 24/25. Lần 2 phóng to 2 lần: 13/25 với ảnh gốc, 5/25 với cổ điển. Ngưỡng đạt là 70 % với gốc và 55 % với cổ điển, nên mạng không được bật làm mặc định. Nếu chỉ đếm các lượt có người, mạng thắng gốc 17/21; đó là quan sát hậu kiểm, không dùng để đổi kết luận.

Nguồn: results/reproduce/tables/t4_ab.md; results/ab/README.md.

<!-- Slide number: 17 -->
03 · E4
17 / 23
E4: làm rõ trước khi nhận diện làm bộ đã tinh chỉnh kém đi
Khung gốc
Sau cổ điển
Sau mạng
mAP50 ↑

### Chart

| Category | Khung gốc | Sau cổ điển | Sau mạng |
| --- | --- | --- | --- |
| YOLO11s COCO (có sẵn) | 0.126 | 0.081 | 0.195 |
| detector-v1.0.0 (đã tinh chỉnh) | 0.119 | 0.007 | 0.035 |

−0,085
bộ đã tinh chỉnh + ảnh mạng, CI [−0,139; −0,011]
Quy tắc D5: nhận diện luôn chạy trên khung gốc

### Notes:
Chạy hai bộ nhận diện trên ba biến thể khung test và so mAP50 ghép cặp với khung gốc. Với YOLO11s COCO có sẵn, ảnh qua mạng tăng 0,069 nhưng CI [−0,033; +0,231] chạm 0; qua cổ điển giảm 0,046. Với bộ đã tinh chỉnh, ảnh qua mạng giảm 0,085 với CI trọn dưới 0, qua cổ điển giảm 0,112. Bộ đã tinh chỉnh học trên ảnh xấu kiểu D2 nên ảnh đẹp lại là ảnh lạ với nó. Luật chốt trước: chỉ chuyển sang ảnh đã làm rõ khi CI nằm trọn trên 0, nên giữ khung gốc.

Nguồn: results/reproduce/tables/t5_e4.md; results/e4/e4.json.

<!-- Slide number: 18 -->
04 · TÍCH HỢP
18 / 23
Trong IOT-CV: theo kịp camera, nhưng p95 vượt ngân sách
23,6 ms
37,5 ms
detector · p50 ở nhịp camera thật
tổng hai model · p50 · 7,2 khung/s
PHÂN BỔ p50 Ở NHỊP THẬT, CÓ NGƯỜI XEM ẢNH RÕ
Detector · 23,6 ms
Enhancer · 12,5 ms
YOLO11s, khung 1280
NAFNet-w16
Phát lại chuyến 02/11 · ORT CUDA, RTX 4060 Laptop · bỏ 0,3 % khung · p95 tổng 54,1 ms > ngân sách 50 ms

### Notes:
Hai model chạy trong backend IOT-CV qua ONNX Runtime CUDA, trên JPEG ghi được của chuyến 02/11, phát lại đúng nhịp. Có người xem Ảnh rõ: detector 23,6 / 33,4 ms p50 / p95, enhancer 12,5 / 19,0, tổng 37,5 / 54,1 ms, 7,2 khung mỗi giây, bỏ 0,3 % khung. Chạy hết tốc lực đạt 23,6 khung/s và đạt ngân sách (tổng p95 44,9 ms). Ở nhịp thật GPU rảnh khoảng 100 ms giữa hai khung và hạ xung, nên p95 vượt ngân sách 50 ms; nhóm chấp nhận và ghi là giới hạn.

Nguồn: results/reproduce/tables/t6_iotcv_ms.md; BAO_CAO_DO_AN.md mục 4.5.

<!-- Slide number: 19 -->
04 · TÍCH HỢP
19 / 23
Hai bundle ONNX nạp vào IOT-CV qua một hợp đồng cố định
Camera  →  backend/vision
![web_comparison.png](Image 535)
Detector trên khung gốc
Hộp gửi qua WebSocket
Enhancer cho luồng Ảnh rõ
Sai hợp đồng  →  từ chối
detector-v1.0.0 · enhancer-v1.0.0 · ORT CUDA

### Notes:
Hợp đồng model (docs/hop-dong-model.md) cố định đầu vào là khung 640×480 BGR uint8 nguyên khung, cắt mép 16 px nằm trong graph; detector trả boxes theo toạ độ khung gốc và scores; enhancer trả khung cùng cỡ. Bundle gồm file ONNX, meta.json có sha256, và 3 khung golden để kiểm sai số. IOT-CV đặt bundle ở models/vision, từ chối nạp model sai hợp đồng và báo lý do lên web. Ảnh trái là hộp của detector-v1.0.0 trên một khung bay.

Nguồn: docs/hop-dong-model.md; IOT-CV backend/vision.

<!-- Slide number: 20 -->
04 · GIỚI HẠN
20 / 23
Giới hạn của kết quả
Một bãi, một buổi
Tập test nhỏ
Mọi số đo từ chuyến 02/11/2026; ánh sáng, độ cao, nền khác có thể đổi kết quả.
39 khung có người, CI của mAP rộng khoảng ±0,1; dev-val 100 khung chọn sai ở E2.
Hồ sơ chỉ ở q12
Ngân sách ms
Bảng DQT đo ở q12; web hạ xuống q40 khi Wi-Fi yếu, mức đó chưa đo.
Ở nhịp camera, GPU hạ xung giữa hai khung: p95 tổng không đạt 50 ms.

### Notes:
Bốn giới hạn chính. Thứ nhất, mọi số đo đến từ một bãi, một buổi sáng 02/11/2026. Thứ hai, chỉ 39 khung test có người nên CI của mAP rộng; dev-val 100 khung đã không chọn đúng model ở E2. Thứ ba, bảng lượng tử chỉ đo ở q12 trong khi web hạ chất lượng xuống q40 khi Wi-Fi yếu. Thứ tư, ở nhịp camera thật trên laptop này p95 không đạt ngân sách 50 ms. Ngoài ra nhãn do máy gán rồi người duyệt, vẫn có thể sót người rất nhỏ; NR-IQA chỉ để mô tả.

Nguồn: BAO_CAO_DO_AN.md mục 6.

<!-- Slide number: 21 -->
PHÂN CÔNG
21 / 23
Phân công nhóm 10
Nguyễn Duy Nghĩa
Nguyễn Tự Kiên
B23DCCN600
Làm rõ ảnh (cổ điển + NAFNet), đánh giá A/B, tích hợp IOT-CV.
B23DCCN465
Bộ làm xấu D1/D2, đo độ giống thật (C2ST/KID), hiệu chỉnh D2.
Nguyễn Minh Toàn
Nguyễn Thành Trung
B23DCCN832
Nhận diện người: mốc, E1, E2 tự huấn luyện, xuất ONNX.
B23DCCN861
Thu khung bay thật, hồ sơ camera, chia tập, gán nhãn test.

### Notes:
Phân công theo phần việc của bản làm lại trên camera thật. Chủ dự án duyệt nhãn tập test và chấm A/B. Trợ lý AI (Claude Code) viết phần lớn mã và chạy thí nghiệm; nhóm đọc, duyệt và chịu trách nhiệm về kết luận.

Nguồn: phân công nhóm; BAO_CAO_DO_AN.md mục 9–10.

<!-- Slide number: 22 -->
TÀI LIỆU THAM KHẢO
22 / 23
Tài liệu phương pháp và nguồn thực nghiệm
Khử mù và làm xấu ảnh
Mạng và nhận diện
He, Sun & Tang (2011). Dark Channel Prior. IEEE TPAMI 33(12).
Wang et al. (2021). Real-ESRGAN. ICCVW.
Chen et al. (2022). NAFNet. ECCV.
Ultralytics YOLO11; RoyChowdhury et al. (2019). Self-training. CVPR.
Dữ liệu và độ giống thật
Mã và dữ liệu
Zhu et al. (2022). VisDrone. IEEE TPAMI 44(11).
Lopez-Paz (2017) C2ST; Bińkowski (2018) KID; Oquab (2024) DINOv2.
github.com/Localhost218/XLA
scripts/reproduce.py --check  ·  results/reproduce/

### Notes:
Các phương pháp tham khảo từ công trình ghi trên slide rồi điều chỉnh theo camera của nhóm. Mọi bảng và hình sinh bằng scripts/reproduce.py từ kết quả đã lưu; lệnh --check chấm lại 15 tệp dự đoán trên CPU và so sha256 của 9 bảng, hình. Danh sách đầy đủ ở BAO_CAO_DO_AN.md.

Nguồn: BAO_CAO_DO_AN.md, Tài liệu tham khảo.

<!-- Slide number: 23 -->
![uav_hero.png](Image 584)
KẾT LUẬN
23 / 23
Kết quả và hướng phát triển
0,30
Hai câu trả lời
D2 hơn D1 ở báo nhầm, mAP chưa chắc. Làm rõ trước khi nhận diện làm hại bộ đã tinh chỉnh.
báo nhầm/khung bay · D2 (D1: 0,99)
24/25
Tiếp theo
Sửa phơi sáng, WB ở cảm biến · thêm khung có người · ràng buộc màu, thời gian cho mạng · thử imgsz 960.
lượt A/B chọn bộ cổ điển

### Notes:
Câu 1: làm xấu theo số đo camera (D2) thắng làm xấu ngẫu nhiên ở độ giống thật mức pixel và ở báo nhầm (0,30 so với 0,99 hộp mỗi khung bay không người), còn mAP chưa kết luận được. Câu 2: làm rõ trước khi nhận diện không giúp bộ đã tinh chỉnh (giảm rõ), với bộ có sẵn có xu hướng giúp nhưng CI chạm 0. Ảnh rõ: người chấm chọn bộ cổ điển 24/25 lượt, mạng chưa bật mặc định. Việc tiếp: sửa tận gốc ở cảm biến (phơi sáng, cân bằng trắng, đo DQT q40), thu thêm khung có người nhiều buổi, thêm ràng buộc màu và thời gian khi huấn luyện mạng, thử detector imgsz 960 hoặc FP16.

Nguồn: BAO_CAO_DO_AN.md mục 5, 7.
