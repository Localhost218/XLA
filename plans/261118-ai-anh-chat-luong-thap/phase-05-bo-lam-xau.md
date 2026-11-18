# Phase 05: Bộ máy làm xấu D1 (ngẫu nhiên bậc hai) + D2 (bản sao camera), dữ liệu nguồn

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | phase 02, 04 | L (~2–3 ngày agent + thời gian tải), 0 giờ GPU (CPU) | Không |

Lane: degrade
Depends-on: 02, 04
Effort: L
Creates: `xla/degrade/sources.py`, `xla/degrade/d1_random.py`, `xla/degrade/d2_camera.py`, `xla/degrade/build.py`, `configs/sources.yaml`, `configs/degrade_d1.yaml`, `configs/degrade_d2.yaml`, `tests/degrade/test_degrade.py`
AX: `python -m xla.degrade.build --set d0|d1|d2 --seed 20261118 --limit N --json` sinh tập + manifest, `--limit` cho chạy thử nhỏ
Accept: `pytest tests/degrade/test_degrade.py`

## Mục tiêu

Dựng "bản sao số của camera": từ ảnh sạch có nhãn người, sinh ba tập cùng hình học 640×480: **D0** sạch, **D1** làm xấu
ngẫu nhiên kiểu Real-ESRGAN bậc hai, **D2** làm xấu theo đường vật lý khớp từng số đo của `camera_profile.yaml`. Mọi thứ
tái lập bằng seed + manifest. Phase này dựng máy và tham số ban đầu; phase 07 hiệu chỉnh D2.

## Đầu vào cần có

- `configs/camera_profile.yaml` (phase 04); hợp đồng §6.
- Thiết kế `plans/reports/261118-brainstorm-ai-anh-chat-luong-thap.md` §5.2 (bảng khâu ↔ số đo).
- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §1 (dataset), §1 "Về GSD" (khớp cỡ người bằng hộp
  nhãn; hạ mẫu bằng `INTER_AREA` **trước** nhòe/nhiễu/JPEG), §2 (nhiễu đo sau ISP theo NLF + PSD; JPEG đúng một lần ở
  cuối bằng DQT thật 4:2:2).
- Prior art: `src/data_augmentation.py:162` (`_rng_for`), `:174` (`generate_augmented_benchmark`, manifest), `:31` (`_jpeg`,
  thay bằng Pillow `qtables`); IOT-CV `ml/scripts/convert_visdrone_person.py` (category 1 pedestrian, 2 people → lớp 0);
  `d:\Coding\IOT-CV\plans\reports\260921-research-ai-vision-pipeline.md` §2.1–2.3 (VisDrone, công thức YOLO);
  `d:\Coding\IOT-CV\plans\ai\ai-phase-01-chuan-bi.md` A1.5 (thiết kế suy giảm bậc hai).
- `plan.md` Deferred 1, 2, 3.

## Việc theo thứ tự

1. **Tải nguồn** (ổ D): VisDrone2019-DET train/val (link của Ultralytics `VisDrone.yaml`) → `D:\IOT_Tools\downloads\visdrone\`;
   NOMAD (Google Drive, CC BY 4.0, bằng `gdown`) → `D:\IOT_Tools\downloads\nomad\`; POG thử một lần, máy chủ không trả lời
   thì ghi lý do và bỏ (Deferred 2). → 18/11/2026: NOMAD tải tập con rồi loại vì ảnh nhìn xiên; POG không trả lời (`plans/reports/261118-phase-05-bo-lam-xau.md`). Giải nén vào `D:\DevCache\datasets\src\`. Ghi sha256 file tải vào `configs/sources.yaml`.
   Tải > 2 phút: chạy thử 1 file trước, báo cỡ + thời gian ước.
2. **`xla/degrade/sources.py`:** VisDrone → một lớp person (category 1, 2), bỏ vùng `ignored` (category 0) bằng cách không
   đặt cửa sổ cắt chồng lên nó; NOMAD → hộp người theo nhãn của dataset, độ cao theo thư mục `ActorXXX_aNN`; lọc nadir theo
   Deferred 3 (metadata khi có; heuristic không-có-trời + điểm CLIP zero-shot cho phần còn lại, ngưỡng chọn trên 100 ảnh soát
   tay, ghi vào `configs/sources.yaml`). → đổi 18/11/2026: bộ lọc này precision ~0,5; thay bằng nhãn hướng nhìn theo chuỗi
   (`configs/visdrone_view.yaml`) + probe logistic trên đặc trưng CLIP, ngưỡng 0,8 (Deferred 3 của `plan.md`).
3. **Hình học chung D0/D1/D2:** cho mỗi ảnh nguồn, chọn hệ số thu nhỏ sao cho cạnh ngắn hộp người lấy mẫu theo phân bố
   `person_box_px` của hồ sơ (đo sơ bộ ở phase 04, cùng dải 21–139 px trung vị 50 của mẫu 300 khung), thu nhỏ bằng `INTER_AREA`, cắt
   cửa sổ 624×480 quanh người (và 20 % cửa sổ không người), giữ hộp còn ≥ 60 % diện tích. Kết quả 624×480 trước khi dán dải mép.
4. **D1** (`d1_random.py`, `configs/degrade_d1.yaml`): hai vòng [nhòe Gauss/anisotropic, resize ngẫu nhiên, nhiễu
   Gauss/Poisson, JPEG ngẫu nhiên Q 30–95 bằng cv2] theo khoảng của Real-ESRGAN (`nghien-cuu-tai-lieu.md` §2), kết thúc ở
   624×480 rồi dán 16 cột đen ngẫu nhiên → 640×480. Không đọc hồ sơ camera (đây là đối chứng "không đo").
5. **D2** (`d2_camera.py`, `configs/degrade_d2.yaml`, mỗi tham số ghi trường hồ sơ nó khớp): (a) màn mờ `I = J·t + A·(1−t)`;
   (b) PSF Gauss + tối góc cos⁴; (c) sang không gian tuyến tính, nhân độ sáng, **cắt từng kênh ở 1,0 rồi mới** nhân hệ số
   cân bằng trắng R/B → vùng sáng bão hoà thành hồng; về sRGB; (d) ISP: khử nhiễu + làm mềm, rồi nhiễu theo NLF + lọc cho
   đúng PSD; (e) đệm dải mép trái theo `left_strip` (độ sáng, vạch đỏ đều) → 640×480; (f) **JPEG một lần, cuối cùng** bằng
   Pillow `qtables` của hồ sơ + `subsampling="4:2:2"`, không truyền `quality`; tham số q ngẫu nhiên trong {12 thật, 19–40 giả
   thuyết} với trọng số trong config (mặc định 0,8 cho q12). Lưu tham số từng ảnh vào manifest.
6. **`build.py`:** sinh D0/D1/D2 ra `D:\DevCache\datasets\xla\{d0,d1,d2}\{train,val}\{images,labels}` + `manifest.jsonl`
   (nguồn, seed, tham số, sha256 ảnh ra) + `dataset.yaml` cho Ultralytics. Cùng ảnh nguồn + cùng cửa sổ cho cả ba tập (so
   sánh công bằng). Chạy thử `--limit 50`, đo giây/ảnh, ước tổng thời gian; > 2 phút thì báo con số trước khi chạy đủ.
7. **Test** `tests/degrade/test_degrade.py` (CPU, ảnh tổng hợp nhỏ): cùng seed → byte giống hệt; khác seed → khác; ảnh ra
   640×480 uint8; hộp nhãn biến đổi đúng theo phép thu nhỏ + cắt (một ca tính tay); DQT + SOF của ảnh D2 = bảng hồ sơ và
   4:2:2; khâu (c) biến một mảng sáng trắng thành hồng (R−G>25, B−G>25) còn mảng xám trung bình thì không; D1 không đọc hồ sơ.
8. Commit trên `lane/degrade`, đẩy, báo lead.

## Cổng pass

- Accept exit 0; `build --set d2 --limit 50` chạy hết, `manifest.jsonl` đủ 50 dòng.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| NOMAD tải chậm / Google Drive giới hạn lượt | medium | low | VisDrone đủ làm nguồn chính; NOMAD bổ sung sau |
| Nhãn VisDrone thiếu người rất nhỏ | medium | medium | Cửa sổ cắt quanh người có nhãn; báo cáo khai |
| Tập sinh chiếm ổ D quá lớn | low | low | Ghi cỡ ước ở bước 6 trước khi chạy đủ |

## Rollback

Xoá `D:\DevCache\datasets\xla\` (tái tạo được bằng seed) và revert commit.
