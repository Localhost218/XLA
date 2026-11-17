# Nghiên cứu bổ sung: AI cho ảnh chất lượng thấp của camera ESP32-S3 + OV3660

Ngày 18/11/2026. Bổ sung (không thay) `IOT-CV/plans/reports/260921-research-ai-vision-pipeline.md` mục 2.4, 3.3–3.5.
Phương pháp: WebSearch + WebFetch (abstract, HTML, bảng kết quả), curl HEAD/GET cho từng link (mục 9–10), đo trực tiếp
trên dữ liệu camera đã ghi (mục 0). Chỗ chỉ thấy qua snippet ghi rõ "theo snippet" hoặc "chưa xác minh". Phần lớn
kết luận lấy từ abstract hoặc một bảng kết quả, chưa đọc toàn văn. Script đo (chỉ đọc): `jpeg_dqt_scan.py`,
`riff_count.py`, `annexk_check.py` cùng thư mục.

## TL;DR

1. **Camera thật ở q=12 dùng bảng lượng tử chuẩn Annex K × 75 % (≈ IJG Q62,5) + màu 4:2:2.** Không mức quality
   nguyên nào của IJG/OpenCV tái tạo đúng; Pillow `qtables=` + `subsampling="4:2:2"` tái tạo 100 %.
   `IOT-CV/backend/vision/fake_stream.py::cv2_quality()` coi q=12 là Q81 → ảnh camera giả nét hơn ảnh thật.
2. **Dataset đáng lấy thêm:** POG (người trên cỏ, 4–103 m, độ cao + góc từng ảnh, giấy phép chưa rõ), NOMAD (CC BY
   4.0, 10–90 m), Archangel (15–50 m, metadata đầy đủ, phải đăng nhập), SeaDronesSee (CC0, nền biển).
3. **Chưa có bài nào so trực tiếp** "suy giảm hiệu chỉnh theo một camera" với "suy giảm ngẫu nhiên kiểu Real-ESRGAN"
   cho detection trên ảnh thật → chỗ trống đề tài lấp được. Gần nhất (Sensor Transfer): tham số hiệu ứng cảm biến học
   từ ảnh đích +1,66/+5,35 AP; khoảng ngẫu nhiên tay chọn −2,07/+4,76; CycleGAN −2,25/−0,83.
4. **Khôi phục trước detection:** tài liệu 2019–2026 thống nhất "không chắc giúp, có khi hại". Fine-tune detector trên
   ảnh suy giảm là đối chứng mạnh; chỉ khôi phục huấn luyện theo tác vụ hoặc có cổng mới vượt nó.
5. **JPEG ở q=12 không phải nút thắt chính:** Faster R-CNN mất ~1,3 mAP ở Q50, ~15 mAP ở Q10 (COCO); vật nhỏ chịu nặng
   nhất. Đáng lo khi auto-quality đẩy lên q=40 (chưa biết tương đương IJG, chưa có khung mẫu).
6. **Thích nghi miền vừa 8 GB:** nhãn giả + tracker (RoyChowdhury 2019) và SimROD rút gọn (teacher lớn → student nhỏ,
   trộn nguồn/đích) dựng bằng Ultralytics. Adaptive Teacher / Efficient Teacher / SSDA-YOLO đòi 4–8 GPU.
7. **Dữ liệu đích ít hơn tưởng:** 49 027 khung JPEG trong AVI (frames.csv chỉ 40 969 dòng) nhưng chỉ ~3,5 nghìn khung
   ở ≥ 5 m, cùng một phiên (02/11 10:09).
8. **"Giống thật":** KID (không chệch) trên đặc trưng DINOv2 + bộ phân loại hai mẫu (C2ST) trên patch + thống kê mức
   thấp (DQT, nhiễu, MTF). FID chệch theo số mẫu và nhạy chính JPEG.
9. **NR-IQA** (NIQE, BRISQUE, MUSIQ, CLIP-IQA) chỉ để mô tả: đã có bằng chứng bị tấn công, bị tối ưu thành artifact,
   chấm sai ảnh GAN. Trọng tài cuối: mAP50 / AR-small trên khung thật có nhãn.
10. **Xếp hạng:** (1) suy giảm hiệu chỉnh tất định + dữ liệu có độ cao → (2) self-training trên khung thật → (3) học
    suy giảm không cặp làm ablation → (4) front-end khôi phục có cổng → (5) Mean-Teacher nặng: không làm.

## 0. Đo trực tiếp trên dữ liệu camera

| Đại lượng | Giá trị đo | Cách đo |
|---|---|---|
| Khung JPEG gốc | **49 027** khung, 581 file AVI, 11 phiên 01–02/11/2026 | đếm chunk `00dc` trong `LIST movi`, khớp `dwTotalFrames`; `avi_writer.py` chép nguyên khung, không nén lại |
| Cỡ khung | p05 / p50 / p95 ≈ 11,0 / 15,1 / 30,2 KB | khớp README camera: q=12 → 15,2 KB |
| Định dạng | baseline (SOF0) 640×480, Y 2×1, Cb/Cr 1×1 = **4:2:2** | đọc SOF |
| Bảng lượng tử | **một bộ duy nhất** trên 2 394 khung mẫu; luma hàng đầu `12 8 8 12 18 30 38 46`, chroma `13 14 18 35 74 74 74 74` | = Annex K × 75 % (làm tròn IJG), khớp 64/64 cả hai bảng; IJG Q62 lệch 32/64 ô luma, Q63 lệch 45/64, Q81 lệch 64/64 |
| Tái tạo | Pillow 12.1: `save(..., qtables=frame.quantization, subsampling="4:2:2")` → DQT trùng 100 % | mặc định Pillow ở Q62 là 4:2:0 |
| Độ cao khung | frames.csv 40 969 dòng; chỉ `20261102-100941` có khung bay: 3 454 dòng ≥ 5 m, 2 117 ≥ 10 m, max 57,8 m (điểm vọt) | cột `rel_alt_m` |

Hệ quả:
- Camera giả và augment `ImageCompression` (dự kiến trong `IOT-CV/ml/scripts/augment.py`) sẽ nhẹ tay hơn thật.
- Bảng cho q ≠ 12 (19…40 của auto-quality từ 15/11) chưa có mẫu; bản ghi mới có cột `jpeg_quality`.
- frames.csv ít dòng hơn số khung AVI (40 969 vs 49 027), nguyên nhân chưa rõ → ghép khung ↔ độ cao theo
  `segment` + `video_frame`, không theo thứ tự dòng.
- FOV ống kính OV3660 chưa đo.

## 1. Dataset người nhìn từ drone

| Dataset | Người / ảnh | Độ cao · góc | Phân giải | Giấy phép | Tải (kiểm 18/11) | Metadata độ cao |
|---|---|---|---|---|---|---|
| **POG – PeopleOnGrass** (Kiefer, ICAR 2021, arXiv 2101.12677) | ~2 900 ảnh, 13 713 người | **4–103 m · 0–90°** | 3840×2160 | **chưa rõ** | cloud Tübingen — máy chủ không trả lời | **Có, từng ảnh** |
| **NOMAD** (WACV 2024, arXiv 2309.09518) | 42 825 khung có hộp + 10 mức nhìn thấy; 100 diễn viên (đi, nằm, nấp) | 10/30/50/70/90 m | 5472×3078 | **CC BY 4.0** | Google Drive (200) | theo thư mục (`Actor001_a10`) |
| **Archangel** (IEEE Access, arXiv 2209.00128) | Real 41,4k ảnh; Mannequin 178,8k | 15–50 m, pitch −22,5/−45/−67,5° | 1304×978 | không ghi | a2i2-archangel.vision, đăng nhập | **Có, từng ảnh** + tư thế |
| **SeaDronesSee** (WACV 2022, arXiv 2105.01922) | OD v2: 14 227 ảnh | 5–260 m · 0–90° | — | **CC0 1.0** (README) | máy chủ từ chối kết nối | trong nhãn MOT; nền biển |
| **Manipal-UAV Person** (ISPRS JPRS 195, 2023) | 13 462 ảnh, 153 112 instance | 10–50 m | 1280×720 | GPL-3.0 + học thuật | MS Form + email | không |
| **WiSARD** (IROS 2022, arXiv 2309.04453) | 26 862 ảnh visual + 29 989 nhiệt | — | — | kiểu MIT | Google Drive 40,54 GB | không |
| DetReIDX (arXiv 2505.04793) | >13 triệu hộp | 5,8–120 m | — | CC BY-NC-ND 4.0 | không trả lời | chưa xác minh |
| AU-AIR (arXiv 2001.11737) | 32K ảnh giao thông | 5–30 m | — | — | **404** | có |
| MAVREC (CVPR 2024) | 537 030 khung *(snippet)* | 25–45 m | 2.7K *(snippet)* | chưa đọc | 206 | chưa xác minh |
| SynDrone (ICCVW 2023, CARLA) | 72k mẫu | 20/50/80 m *(snippet)* | 1920×1080 | CC-BY (dataset) | zip 113 GB | có (mô phỏng) |
| ForestPersons (ICLR 2026) | 96 482 ảnh | 1,5–2 m dưới tán | — | CC BY-NC-SA 4.0 | HF 200 | độ cao lệch xa |

Khảo sát arXiv 2511.17674: SARD 5–50 m, HERIDAL 40–65 m, VisDrone 30–100 m, UAV-Human 2–8 m.

**Xếp hạng:** (1) POG — cảnh gần nhất, metadata từng ảnh; giấy phép chưa rõ → chỉ dùng nội bộ. (2) NOMAD — CC BY 4.0,
10–30 m khớp, có tư thế nằm/nấp. (3) Archangel. (4) Manipal-UAV. (5) SeaDronesSee. Bỏ: DetReIDX, AU-AIR, ForestPersons,
VTSaR, SynDrone.

**Về GSD:** chiều cao người theo pixel nằm sẵn trong hộp nhãn, nên khớp kích thước không cần metadata. Metadata dùng để
(a) chọn ảnh có góc nhìn giống camera (xiên vs thẳng xuống), (b) phân tầng đánh giá. `h_px ≈ f_px · H / D`, `f_px` từ
hiệu chỉnh bàn cờ OV3660; hạ mẫu nguồn bằng `INTER_AREA` **trước** khi thêm nhòe/nhiễu/JPEG.

## 2. Suy giảm có hiệu chỉnh hoặc học từ ảnh thật không cặp

| Họ | Đại diện | Cần | Hợp OV3660? |
|---|---|---|---|
| Vật lý, nhiễu RAW | Unprocessing (Brooks, CVPR 2019); ELD (Wei, CVPR 2020) | RAW / biết ISP | Một phần: OV3660 có ISP + JPEG trên chip → đo nhiễu **sau ISP** (NLF + phổ công suất) |
| Thống kê từ ảnh đích | **Real-SR** (Ji, CVPRW 2020, thắng NTIRE 2020): kernel KernelGAN + gom nhiễu cao tần từ ảnh đích; Apache-2.0 | ảnh đích không nhãn | Tốt; chỉ lấy thống kê, không dán patch đã dính khối JPEG; JPEG đúng một lần ở cuối |
| Học tham số hiệu ứng cảm biến | **Sensor Transfer** (Carlson, RA-L 2019): học tham số 5 hiệu ứng để khớp ảnh đích | ảnh đích không nhãn | Rất hợp; không đổi hình học → nhãn giữ nguyên |
| Học phân phối suy giảm (GAN) | PDM (CVPR 2022), DASR, CUT/CycleGAN | GPU | Rủi ro đổi nội dung; CycleGAN hại (bảng dưới) |
| JPEG đúng bảng | Pillow `qtables` + `subsampling` | 1 khung thật | **Bắt buộc**; đã kiểm khớp 100 % |

Sensor Transfer, Faster R-CNN, train Sim10k, AP50 xe trên ảnh thật:

| Augment | KITTI | Cityscapes |
|---|---|---|
| Không | 51,01 | 30,13 |
| Hiệu ứng cảm biến, khoảng tay chọn | 48,94 (−2,07) | 34,89 (+4,76) |
| Hiệu ứng cảm biến, tham số học từ ảnh đích | **52,67 (+1,66)** | **35,48 (+5,35)** |
| CycleGAN / UNIT / MUNIT | 48,75 / 51,21 / 45,50 | 29,30 / 28,05 / 26,20 |

- MAET (ICCV 2021, YOLOv3, ExDark mAP): gốc 76,4; front-end tăng sáng 76,3–76,9; huấn luyện với suy giảm vật lý **77,7**.
- Taori et al., NeurIPS 2020: bền vững với dịch chuyển tổng hợp thường không chuyển sang dịch chuyển tự nhiên.
- Real-ESRGAN: JPEG ngẫu nhiên [30, 95], Gauss σ 1–30, Poisson 0,05–3, kernel 7–21 → phủ Q62,5 của camera nhưng đa số
  mẫu không giống nó.
- **Khoảng trống** (phạm vi: 3 truy vấn WebSearch + các bài trên, 18/11/2026): không thấy bài so detector huấn luyện
  bằng suy giảm hiệu chỉnh cho MỘT camera với Real-ESRGAN/BSRGAN ngẫu nhiên, đánh giá trên ảnh thật.

Pipeline đề xuất: hạ mẫu theo GSD → nhòe quang học theo MTF đo → nhòe chuyển động (tuỳ) → nhiễu theo NLF, lọc cho đúng
PSD → gamma/cân bằng trắng theo ảnh thật → JPEG 4:2:2 bằng DQT thật của mức q. Lưu tham số + hash để tái lập.

## 3. Khôi phục trước khi phát hiện có giúp không

| Nguồn | Suy giảm | Kết quả |
|---|---|---|
| Ehrlich et al., ICCVW 2021 | JPEG, COCO | Q10: gốc 35,37 → không xử lý 20,35; fine-tune có nhãn 29,09; khử artifact có sẵn 28,45; **khử artifact theo tác vụ (TTAC) 31,43**. Mất ~15 mAP Q10, ~5 Q20, ~2,8 Q30, ~1,3 Q50 |
| Gandor & Nalepa, Sensors 2022 | JPEG Q1–100, 9 detector | giảm chậm tới ~Q40, dốc nhất ~Q15; vật nhỏ chịu nặng nhất, mất chủ yếu recall |
| MAET, ICCV 2021 | thiếu sáng | front-end −0,1…+0,5 mAP; huấn luyện theo suy giảm +1,3 |
| Sun et al., NeurIPS 2022 | sương, thiếu sáng | khôi phục hiện có không cải thiện detector, có lúc giảm |
| Yang et al. (UG²+), TIP 2020 | sương, mưa, tối | tăng cường giúp detection "not always the case" |
| Wu et al., arXiv 2409.14461 | thiếu sáng | LLIE không nhất quán, có lúc hại |
| TaskGuard, arXiv 2609.08011 | nhòe, mưa, defocus | cổng chọn ảnh khôi phục/gốc: −37,0 % ca bị tệ đi, giữ 98,8 % AP |
| JR², arXiv 2609.13791 | nhiễu loạn khí quyển | ảnh đẹp hơn "neither necessary nor sufficient"; cổng bỏ qua ~70 % khung sạch |
| B2BDet, arXiv 2401.14661 | ảnh hàng không | SRGAN +2,6 % mAP; không so với fine-tune |

Kết luận: (1) front-end tối ưu PSNR/mắt người trước detector đóng băng: lợi nhỏ, không nhất quán, có thể âm. (2)
Fine-tune detector trên ảnh suy giảm giống thật là đối chứng bắt buộc. (3) Khôi phục chỉ thắng khi huấn luyện theo
tác vụ hoặc có cổng. (4) Với camera này JPEG Q62 phạt nhỏ; nút thắt là cỡ người, nhòe, màn mờ, ám hồng.

## 4. Self-training / thích nghi miền trên 1 GPU 8 GB

| Phương pháp | Nền | Vừa 8 GB? |
|---|---|---|
| **Nhãn giả + tracker** (RoyChowdhury, CVPR 2019): nhãn từ phát hiện tin cậy cao + hard example do tracker tìm theo thời gian | tự dựng bằng Ultralytics `predict(save_txt, save_conf)` + `track` (botsort/bytetrack, `persist=True`) | **Có** |
| **SimROD** (ICCV 2021): DomainMix 2×2 nguồn/đích, teacher lớn thích nghi dần → student | YOLOv5; dựng lại trên Ultralytics | Gần có |
| ConfMix (WACV 2023) | YOLOv5s, GPL-3.0 | Có, nhưng cũ |
| SF-YOLO (ECCVW 2024), source-free | YOLOv5l | Có thể; thừa vì có dữ liệu nguồn |
| SSDA-YOLO, Efficient Teacher, Adaptive Teacher | 4–8 GPU | Không |

SimROD (YOLOv5m, mPC50): Pascal-C 53,78 → 75,40; COCO-C 22,03 → 28,46; Cityscapes-C 11,53 → 18,01. SF-YOLO cảnh báo
mean teacher tự huấn luyện có thể mất chính xác nhanh khi nhãn giả nhiễu.

Quy trình (SimROD rút gọn): lọc khung bay, 1–2 khung/s, chia theo khối thời gian → teacher m/l imgsz 960–1280 + TTA →
nhãn giả ngưỡng cao làm sạch bằng track (giữ hộp thuộc track dài, cứu hộp thấp trong track cao, bỏ phát hiện lẻ) →
student n/s huấn luyện trên nguồn suy giảm hiệu chỉnh + khung đích có nhãn giả → lặp 2–3 vòng, chọn checkpoint bằng val
thật có nhãn. Rủi ro: thiên lệch xác nhận (người nhỏ bị sót bị củng cố). VRAM chưa đo.

## 5. Độ đo "giống thật" và chất lượng không tham chiếu

- FID chệch theo n và theo mô hình (Chong & Forsyth 1911.07023; Bińkowski ICLR 2018). **KID** có ước lượng không chệch;
  dùng n vừa và trung bình nhiều lần.
- Encoder: DINOv2-ViT-L/14 tương quan với người tốt hơn Inception (Stein et al., NeurIPS 2023), thư viện `dgm-eval`.
- FID có thể giảm khi ảnh sinh bị nén JPEG (Parmar, CVPR 2022) → giữ y hệt quy trình đọc/resize cho cả hai tập (clean-fid).
- Khác nội dung cảnh lấn khác biệt suy giảm (Kynkäänniemi 2203.06026) → so trên patch cùng loại cảnh hoặc phần dư cao tần.
- **C2ST** (Lopez-Paz & Oquab 1610.06545): CNN nhỏ phân biệt thật/giả trên patch 64×64 phần dư cao tần; ~50 % = không
  phân biệt được.
- Thống kê mức thấp (tất định, không lừa được): DQT + lấy mẫu màu, NLF, PSD nhiễu, MTF/ESF, độ khối JPEG.
- NR-IQA bị tấn công/bị tối ưu thành artifact (2210.00933, 2310.06958, 2405.20392, PIPAL 2007.12142) → chỉ mô tả, cùng crop,
  không dùng chọn checkpoint. `pyiqa` giấy phép PolyForm Noncommercial 1.0.0.

**Xếp hạng độ đo:** (1) mAP50 / AR-small trên khung thật có nhãn — trọng tài; (2) thống kê mức thấp; (3) C2ST trên patch;
(4) KID-DINOv2 (`dgm-eval`), 1 000–3 000 patch; (5) NR-IQA (MUSIQ + NIQE qua pyiqa) chỉ mô tả.

## 6. Khuyến nghị xếp hạng

| # | Hướng | Lợi kỳ vọng | Chi phí | Ghi chú |
|---|---|---|---|---|
| 1 | Suy giảm hiệu chỉnh tất định (DQT thật 4:2:2, NLF/PSD, MTF, GSD) + VisDrone-person + NOMAD/POG | trung bình–cao | thấp (CPU) | rất hợp |
| 2 | Self-training trên khung thật (nhãn giả + tracker, SimROD rút gọn) | cao nếu khung đích có người | trung bình | ~3,5k khung bay |
| 3 | Học suy giảm không cặp (Real-SR; CUT tuỳ chọn) | chưa rõ | trung bình | chỉ làm ablation |
| 4 | Front-end khôi phục (FBCNN/NAFNet) + cổng kiểu TaskGuard | nhỏ, không chắc | trung bình–cao | thí nghiệm, không đưa vào đường nhận diện nếu không thắng |
| 5 | Mean-Teacher nặng | — | 4–8 GPU | không làm |

## 7. Giới hạn

Không đọc full-text phần lớn bài. Chưa có: FOV OV3660, DQT cho q ≠ 12, VRAM thực đo. Không lấy được số: Hong et al. BMVC
2021, IA-YOLO, RoyChowdhury 2019, ConfMix (chỉ snippet). Giấy phép POG, Archangel, MAVREC chưa đọc được; SeaDronesSee CC0
lấy từ README GitHub.

## 8. URL đã kiểm (200/206 hoặc đọc được, 18/11/2026)

Dataset: https://arxiv.org/abs/2309.09518 · https://github.com/ArtRuss/NOMAD ·
https://drive.google.com/drive/folders/1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg · https://arxiv.org/abs/2105.01922 ·
https://github.com/Ben93kie/SeaDronesSee · https://arxiv.org/abs/2309.04453 · https://sites.google.com/uw.edu/wisard/ ·
https://arxiv.org/abs/2209.00128 · https://a2i2-archangel.vision/ · https://arxiv.org/abs/2101.12677 ·
https://datasetninja.com/pog · https://arxiv.org/abs/2001.11737 · https://arxiv.org/abs/2505.04793 ·
https://arxiv.org/abs/2603.02541 · https://huggingface.co/datasets/etri/ForestPersons · https://arxiv.org/abs/2308.10491 ·
https://github.com/LTTM/Syndrone · https://github.com/Akshathakrbhat/Manipal-UAV-Person-Dataset ·
https://arxiv.org/abs/2511.17674 · https://arxiv.org/abs/2312.04548 · https://mavrec.github.io

Suy giảm: https://arxiv.org/abs/1811.11127 · https://arxiv.org/abs/2003.12751 · https://github.com/Vandermode/NoiseModel ·
https://arxiv.org/abs/1809.06256 · https://arxiv.org/abs/1803.07721 · https://arxiv.org/abs/2205.03346 ·
https://github.com/cuiziteng/ICCV_MAET · https://github.com/Tencent/Real-SR · https://arxiv.org/abs/2203.04962 ·
https://arxiv.org/abs/2004.01178 · https://arxiv.org/abs/2007.00644 · https://arxiv.org/abs/2107.10833 ·
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html · https://arxiv.org/abs/2201.02314

Khôi phục ↔ detection: https://neurips.cc/virtual/2022/poster/54700 · https://arxiv.org/abs/1904.04474 ·
https://arxiv.org/abs/2011.08932 · https://arxiv.org/abs/2409.14461 · https://arxiv.org/abs/2609.13791 ·
https://arxiv.org/abs/2609.08011 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8840351/ · https://arxiv.org/abs/2112.08088 ·
https://arxiv.org/abs/2401.14661

Self-training: https://arxiv.org/abs/1904.07305 · https://arxiv.org/abs/2107.13389 · https://arxiv.org/abs/2111.13216 ·
https://github.com/facebookresearch/adaptive_teacher · https://arxiv.org/abs/2211.02213 · https://github.com/hnuzhy/SSDA-YOLO ·
https://arxiv.org/abs/2210.11539 · https://github.com/giuliomattolin/ConfMix · https://arxiv.org/abs/2409.16538 ·
https://github.com/vs-cv/sf-yolo · https://arxiv.org/abs/2302.07577 · https://github.com/AlibabaResearch/efficientteacher ·
https://docs.ultralytics.com/modes/predict/ · https://docs.ultralytics.com/modes/track/

Độ đo: https://arxiv.org/abs/2306.04675 · https://github.com/layer6ai-labs/dgm-eval · https://arxiv.org/abs/1801.01401 ·
https://arxiv.org/abs/1911.07023 · https://arxiv.org/abs/2104.11222 · https://github.com/GaParmar/clean-fid ·
https://arxiv.org/abs/2203.06026 · https://arxiv.org/abs/1610.06545 · https://arxiv.org/abs/2108.05997 ·
https://arxiv.org/abs/2207.12396 · https://github.com/chaofengc/IQA-PyTorch · https://arxiv.org/abs/2310.06958 ·
https://arxiv.org/abs/2210.00933 · https://arxiv.org/abs/2405.20392 · https://arxiv.org/abs/2007.12142 ·
https://arxiv.org/abs/2107.04510

## 9. URL chưa kiểm được

`seadronessee.cs.uni-tuebingen.de` và link POG trên `cloud.cs.uni-tuebingen.de` (từ chối kết nối 18/11); DetReIDX,
P-DESTRE (hết giờ); AU-AIR `bozcani.github.io/auairdataset` (404); VTSaR DOI (403); docs.opencv.org (403); MDPI (403,
dùng PMC); SynDrone zip (chỉ HEAD, 113 GB). Chỉ qua snippet: arXiv 2603.20086, RFSR, 2512.22647, số ConfMix.
