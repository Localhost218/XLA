# Brainstorm: AI xử lý ảnh chất lượng thấp cho drone IOT-CV

Ngày 18/11/2026. Viết cho: nhóm XLA và chủ dự án. Thiết kế đã được chủ dự án duyệt; bước tiếp theo là plan
(`plans/`), chưa viết code. Bằng chứng (ảnh, số liệu, script phân tích) ở `261118-bang-chung/`.

## 0. Kết luận ngắn

Camera thật của drone là **ESP32-S3 + OV3660 nhìn thẳng xuống**, không phải ESP32-CAM/OV2640 như repo đang
ghi. Repo hiện tại chưa chạm vào ảnh của camera này: dữ liệu "thật" là ảnh phố tô hồng nhân tạo, điểm số đo
trên chính dữ liệu do mô hình suy giảm của nhóm sinh ra, không có phần phát hiện người, và chạy 2,2 khung/s.

Hướng mới: dựng một **bản sao số của camera** (làm xấu ảnh sạch của dataset lớn theo đúng số đo của OV3660),
dùng nó nuôi hai nhánh: **ảnh rõ cho người xem** trên web và **phát hiện người**. Mọi điểm số chấm trên
**khung bay thật**. Phần cổ điển của nhóm (khử hồng, khử nhiễu theo σ, trừ màn mờ) được giữ làm phương pháp
đối chứng sau khi sửa.

## 1. Bài toán và ràng buộc

| Mục | Nội dung |
|---|---|
| Mục tiêu A | Luồng video trên web rõ hơn cho người lái: bớt màn mờ, mảng hồng, khối JPEG, ảnh mềm |
| Mục tiêu B | Phát hiện người chính xác trên khung bay thật (hộp + sự kiện cho web) |
| Yêu cầu | Cả A và B phải đạt; chạy kịp luồng camera (≈ 6–9 khung/s khi sóng tốt) trên laptop RTX 4060 8 GB |
| Dữ liệu | **Chưa thu thêm.** Chỉ dùng 11 lần ghi đã có (`IOT-CV/logs/recordings`, 40 969 khung) |
| Chấm điểm | Trên khung thật. Ảnh tổng hợp chỉ dùng để phát triển, báo riêng, không dùng làm kết luận |
| Khung yêu cầu | Đề đồ án của giảng viên (`docs/DoAn_XuLyAnh.pdf`): pipeline nhiều bước, có đối chứng, ≥ 50 ảnh đánh giá, độ đo chuẩn |

## 2. Hiện trạng đo được

### 2.1 Dữ liệu thật

Chi tiết: `261118-bang-chung/video/phan-tich-video.md`.

- 40 969 khung 640×480 MJPEG, nhưng **chỉ lần ghi `20261102-100941` có cảnh bay**: 6 chuyến, **2 944 khung
  bay** (trong cửa sổ chuyến của log FC và cao > 3 m). Ghi 01/11 tối đen, không trùng chuyến nào; bốn lần
  ghi đêm 02/11 quay trên bàn trong nhà.
- Camera nhìn thẳng xuống. Người nhìn từ đỉnh đầu, cao 12–20 m, hộp bao 21–139 px (trung vị 50 px). Ước
  ~370 khung có người.
- JPEG cố định IJG ≈ 62 (quality 12 của esp32-camera). Không khung nào hỏng; mất sóng là mất trắng khung,
  49 % thời gian bay nằm trong các khoảng hở > 1 s.
- Cột `armed`/`mode` trước 15/11 không tin được; `rel_alt_m` trôi 1–4 m khi nằm đất.

### 2.2 Lỗi ảnh, xếp theo mức hại cho nhận diện

| # | Lỗi | Số đo (khung bay, trung vị) |
|---|---|---|
| 1 | Người nhỏ, nhìn từ đỉnh đầu | 50 px; COCO gần như không có góc nhìn này |
| 2 | **Mảng hồng/tím ở vùng sáng** (bão hoà kênh rồi nhân hệ số cân bằng trắng) | 31 % khung > 0,5 % điểm hồng; Y vùng hồng 222 |
| 3 | **Dải đen ~13 px mép trái**, có vạch đỏ đều | Mọi khung, cả khung trong nhà; nguyên nhân chưa xác minh |
| 4 | **Màn mờ sữa**, tương phản thấp | Dark channel 142/255, mức đen p1 = 80, độ bão hoà màu 16 |
| 5 | Ảnh mềm + khối JPEG | Laplacian var 33; chênh biên khối 8×8 gấp 2,6 |
| 6 | Mất khung theo cụm | Hở dài nhất 14,8 s |
| 7 | Tối góc, nhoè chuyển động | Góc/giữa 0,77; 19 % khung nhoè vừa |

### 2.3 Mốc so sánh: YOLO có sẵn

YOLO11s (COCO), ảnh vào 1280, ngưỡng 0,25, trên 300 khung bay: 128 hộp, duyệt tay được 52 người thật, 43 hộp
sai do dải mép trái, 28 hộp sai do mảng hồng. **Precision 41 %** (61 % nếu bỏ dải mép). Recall chưa đo
(chưa có nhãn); đếm tay 4 khung được khoảng một nửa số người. Ảnh vào 640 chỉ bắt được người ở khoảng một
nửa số khung so với 1280.

## 3. Kiểm repo XLA (commit `3e9a764`)

Chi tiết: `261118-bang-chung/kiem-xla/kiem-xla.md`.

| Vấn đề | Mức | Bằng chứng |
|---|---|---|
| `data/real_uav` (60) và `synthetic_gt` (30) là ảnh drone phố Trung Quốc tô hồng nhân tạo, không phải camera này; manifest ghi sai nguồn | Cao | `real_grid.jpg`, `manifest_check.jpg` |
| `raw_videos` là video mã hoá lại (HEVC/H.264 30 khung/s), cảnh khác, không phải drone này | Cao | `kiem-xla.md` §1 |
| PSNR/SSIM/ΔE/PCI đo trên dữ liệu do chính mô hình suy giảm của nhóm sinh ra → vòng tròn | Cao | v1 thắng v2 vì "pivot 172 trùng mô hình sinh" (README tự ghi) |
| Entropy/Laplacian thưởng nhiễu (Global HE cao nhất dù PSNR 12,5) | Cao | Bảng của nhóm |
| Không có đánh giá phát hiện người | Cao | grep |
| Chậm: 448 ms/khung trên khung thật (≈ 2,2 khung/s) | Cao | `metrics.csv` |
| Mặt nạ khử hồng làm bạc vật đỏ/hồng thật (ảnh sạch → PSNR 24,75 dB, ΔE 12,2) | Trung bình | `gt001_clean_vs_v2.jpg` |
| Mô tả phần cứng OV2640/ESP32-CAM | Trung bình | README, báo cáo |

**Giữ lại (sau khi sửa):**
- khử hồng vùng sáng: chỉ kích hoạt khi gần bão hoà, bỏ ngưỡng sáng 110;
- ước lượng σ nhiễu và khử nhiễu theo σ;
- trừ màn mờ theo percentile, đặt sau bước bù tối góc;
- khung sinh dữ liệu có seed và manifest.

**Bỏ:**
- bảng 1–7 và các kết luận định lượng hiện tại;
- Stage 3 (ESPCN, Richardson-Lucy, nâng lên 720p/1080p): không thêm thông tin, tốn khoảng 40 % thời gian;
- CLAHE và bước bù bão hoà sắc độ;
- việc gọi `real_uav`/`raw_videos` là ảnh "camera ESP32 thật".

## 4. Các hướng đã cân nhắc

| Hướng | Ý chính | Ưu | Nhược | Quyết định |
|---|---|---|---|---|
| Sửa tại chỗ pipeline v2 | Chỉnh ngưỡng, đo lại | Ít việc | Vẫn không có dữ liệu thật, không có nhận diện, chậm | Bỏ; chỉ giữ phần đối chứng |
| Dùng YOLO có sẵn + tiền xử lý | Không huấn luyện | Rẻ | Precision 41 %, góc nhìn lạ, mảng hồng thành báo nhầm | Làm mốc so sánh |
| **H1. Bản sao số camera** | Làm xấu dataset lớn theo số đo thật | Dữ liệu cặp vô hạn, có nhãn; giải thích được; nuôi được cả A và B | Phải đo và hiệu chỉnh cẩn thận | **Xương sống** |
| H2. Học cách làm xấu (CUT/CycleGAN) | Học "phong cách" từ khung thật không cặp | Bắt được lỗi khó viết thành công thức | Có thể bịa chi tiết, khó kiểm | Nhánh so sánh nếu còn sức |
| **H3. Tự gán nhãn (self-training)** | Model thầy gán nhãn khung thật, model trò học lại | Thường tăng điểm mạnh nhất trên miền thật | Cần tập test gán tay riêng, dễ rò rỉ | **Dùng cho nhánh B** |
| **H4. Mạng khôi phục riêng camera** | Fine-tune mạng nhẹ trên cặp của H1 | Ảnh rõ cho người xem, chạy kịp luồng | Không có ảnh chuẩn thật để chấm | **Dùng cho nhánh A** |

Điều làm hướng này khác các bài báo: chúng (Real-ESRGAN, BSRGAN) phải **đoán** phân bố suy giảm; ở đây ta
**đo được** từ chính camera, có độ cao từng khung và chép được nguyên bảng lượng tử JPEG.

## 5. Giải pháp chọn

### 5.1 Kiến trúc chạy thật

```
khung MJPEG ─► cắt dải mép + bù tối góc ─┬─► A: mạng khôi phục riêng camera ─► video rõ trên web (bật/tắt)
                                         └─► B: bộ phát hiện người fine-tune ─► hộp + sự kiện
```

Tổng ≤ 50 ms/khung trên RTX 4060 (khung đến cách nhau 110–170 ms).

### 5.2 Bộ máy dữ liệu "bản sao số camera"

1. **Nguồn sạch:** VisDrone2019-DET (lọc ảnh nhìn thẳng xuống bằng CLIP zero-shot hoặc heuristic có trời),
   cộng dataset người nhìn thẳng xuống khác (chốt trong plan theo báo cáo tra cứu tài liệu).
2. **Chuỗi làm xấu theo đường vật lý**, mỗi tham số khớp một số đo ở §2.2:

   | Khâu | Mô hình | Khớp với |
   |---|---|---|
   | Tỉ lệ | Thu nhỏ để cỡ người theo phân bố thật | 21–139 px, trung vị 50 |
   | Màn mờ | I = J·t + A·(1−t) | dark channel 142, p1 80 |
   | Ống kính | PSF Gauss + tối góc cos⁴ | Laplacian 33, góc/giữa 0,77 |
   | Bão hoà + cân bằng trắng | Cắt từng kênh ở không gian tuyến tính **trước** khi nhân hệ số R/B → vùng sáng thành hồng | 31 % khung hồng, Y ≈ 222 |
   | ISP | Khử nhiễu + làm mềm | σ nhiễu 0,36 |
   | Nén | **Bảng DQT chép từ khung thật**, cùng kiểu lấy mẫu màu | IJG ≈ 62, blocking 2,6 |
   | Cảm biến | Dán dải mép trái 13 px | mọi khung |
   | Truyền | (không cần cho ảnh đơn; không có khung hỏng) | – |

3. **Bộ so sánh:** D0 sạch; D1 làm xấu ngẫu nhiên kiểu Real-ESRGAN bậc hai; **D2 bản sao camera**; D3 học
   bằng CUT (tuỳ chọn).
4. **Độ giống thật:** khoảng cách giữa phân bố các số đo §2.2 của ảnh tổng hợp và khung thật, cộng KID trên
   đặc trưng mạng. Hiệu chỉnh tham số trên các khung **dev**, không đụng khung test.

### 5.3 Nhánh B: phát hiện người

- YOLO (11s hoặc 26s, ảnh vào 1280) fine-tune trên D0 / D1 / D2, so trên tập test thật.
- Tự gán nhãn: model thầy (tốt nhất + SAHI + TTA) gán nhãn khung bay dev, model trò học lại. Khung trong nhà
  và khung không người làm mẫu âm (giảm báo nhầm do mảng hồng, dải mép).
- Đo: mAP50, mAP50-95, precision/recall ở ngưỡng vận hành, báo nhầm mỗi khung không người, ms/khung; khoảng
  tin cậy bootstrap.

### 5.4 Nhánh A: ảnh rõ cho người xem

- Đối chứng cổ điển: pipeline của nhóm sau khi sửa, gồm cắt dải mép, bù tối góc, khử mờ dark channel, khử
  hồng theo ngưỡng bão hoà, khử khối, làm nét nhẹ. Viết lại dạng vector, mục tiêu ≤ 30 ms/khung.
- Học sâu: mạng nhẹ (NAFNet, SCUNet hoặc FBCNN; chọn bằng đo tốc độ thật) fine-tune trên cặp D2, để học
  đảo đúng chuỗi lỗi của camera.
- Chấm trên khung thật (không có ảnh chuẩn):
  - chỉ số vật lý về mức tự nhiên: dark channel, % điểm hồng, blocking, độ bão hoà màu;
  - IQA không tham chiếu, loại khó bị làm nét quá tay đánh lừa (chốt trong plan);
  - độ nhấp nháy giữa các khung liên tiếp;
  - ms/khung;
  - **thử mù A/B**: chủ dự án chấm khoảng 50 cặp trên một trang HTML chạy tại máy;
  - tác động lên nhánh B.

### 5.5 Câu hỏi nghiên cứu (cho báo cáo môn)

1. Làm xấu theo số đo của chính camera (D2) có thắng làm xấu ngẫu nhiên (D1) trên khung thật không, xét cả
   nhận diện lẫn độ giống thật?
2. Khôi phục ảnh trước khi nhận diện có giúp không? Dự đoán: giúp bộ nhận diện có sẵn, không giúp bộ đã
   fine-tune (đã bền sẵn). Dữ liệu có thể bác bỏ dự đoán này, và kết quả nào cũng đáng báo cáo.

### 5.6 Tập test thật

- Lấy từ 2 944 khung bay, mỗi khung cách nhau ≥ 1 s; chia **theo khối thời gian** (không chia ngẫu nhiên
  theo khung), giữ cả hai phần dev/test đều có người ở các độ cao.
- Gán nhãn: nhiều model + SAHI gán trước → soát từng khung bằng mắt → chủ dự án duyệt lần cuối trên Label
  Studio chạy tại máy.
- Đạt yêu cầu ≥ 50 ảnh đánh giá của đề; mục tiêu 300–400 khung.

## 6. Repo và dữ liệu

- **XLA** là nơi nghiên cứu:
  - tái cấu trúc thành package `xla/` gồm `camera`, `degrade`, `enhance`, `detect`, `eval`;
  - giữ lịch sử git và notebook của nhóm;
  - dữ liệu và kết luận sai chuyển vào `archive/`, kèm ghi chú nói đúng sự thật;
  - viết lại README và báo cáo;
  - đẩy lên repo khung bay đã chọn kèm nhãn (khoảng 80 MB). VisDrone và các bộ ảnh tổng hợp tạo lại bằng
    script có seed.
- **IOT-CV** là nơi chạy thật:
  - `backend/vision` gọi hai model ONNX đã xuất từ XLA;
  - web có nút bật/tắt ảnh rõ và hiện hộp người;
  - `plans/ai` và `ml/` cũ đổi sang trỏ về XLA (kiểm tham chiếu trước khi xoá; khoảng 40 file đang nhắc tới
    chúng).

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Tập test chỉ 1 bãi, 1 buổi sáng; khung tương quan | Chia theo khối thời gian, khoảng tin cậy bootstrap, không nói quá |
| Hiệu chỉnh khớp riêng buổi 02/11 | Ghi rõ phạm vi; hồ sơ camera có phiên bản, đo lại khi đổi cấu hình |
| Nhãn do máy gán thiên về model gán | Dùng model khác họ với model được chấm, chủ dự án duyệt lần cuối, báo cáo khai rõ |
| Sửa camera tận gốc làm hồ sơ cũ lỗi thời | Hồ sơ theo phiên bản; giữ khả năng chịu lỗi cũ vẫn có ích |
| Mạng khôi phục "bịa" chi tiết | Ưu tiên mạng nhẹ, kiểm nhấp nháy và tác động lên nhận diện, không chỉ nhìn đẹp |
| VisDrone không có file giấy phép | Dùng cho nghiên cứu, trích dẫn bài gốc, không đăng lại ảnh thô |

## 8. Tiêu chí thành công (sửa lại sau khi đo mốc ở M2)

| Nhánh | Tiêu chí |
|---|---|
| B | mAP50 trên tập test thật ≥ 1,5 × mốc YOLO11s có sẵn; báo nhầm ≤ 0,05 hộp/khung trên khung không người; ≤ 30 ms/khung |
| A | % điểm hồng giảm ≥ 90 %; dark channel trung vị giảm ≥ 50 %; nhấp nháy không tăng; thắng ≥ 70 % lượt A/B so với ảnh gốc và ≥ 55 % so với đối chứng cổ điển; ≤ 30 ms/khung; không làm giảm mAP nhánh B |
| Dữ liệu | KID(D2, thật) < KID(D1, thật), khoảng tin cậy không chồng nhau |
| Tái lập | Một lệnh dựng lại toàn bộ bảng và hình từ dữ liệu + seed |

## 9. Bước tiếp

Plan chi tiết ở `plans/`, theo mốc:

1. M0: tái cấu trúc XLA, nhập dữ liệu, lập hồ sơ camera.
2. M1: bộ máy làm xấu ảnh và phép đo độ giống thật.
3. M2: tập test thật có nhãn, đo mốc so sánh.
4. M3: thí nghiệm nhận diện.
5. M4: thí nghiệm ảnh rõ, thử A/B.
6. M5: gắn vào IOT-CV.
7. M6: báo cáo, slide, README.

Mỗi lần huấn luyện đều chạy thử nhỏ trước để đo giờ GPU thật. Các phần độc lập chạy song song bằng
sub-agent, mỗi agent sở hữu file riêng.

## 10. Ngoài AI: sửa tận gốc ở lần bay tới (chưa làm vì chưa thu thêm dữ liệu)

Rẻ và có thể lợi hơn mọi mô hình:

- **Ống kính:** lau, kiểm xem còn film bảo vệ không, thêm chụp che nắng (cho màn mờ).
- **Mảng hồng:** hạ mức phơi sáng AE xuống −1 hoặc −2, khoá cân bằng trắng.
- **Dải mép:** chỉnh cửa sổ đọc cảm biến để bỏ dải đen bên trái.
- **Ghi dữ liệu:** thêm cột `jpeg_quality` (đã có từ 15/11).
- **Lần sau thu thêm:** bay 5/10/15/20 m có người đứng sẵn, chụp cặp với điện thoại.

Các việc này nằm ở `IOT-CV/firmware/camera`.
