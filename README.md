# ĐỒ ÁN XỬ LÝ ẢNH (IMAGE PROCESSING) — PTIT
## Đề tài: Xây dựng Pipeline Khôi phục Màu sắc, Khử nhiễu và Nâng cao Chất lượng Ảnh thu nhận từ UAV sử dụng Camera giá rẻ ESP32-CAM (OV2640)

* **Đơn vị:** Khoa Trí tuệ Nhân tạo – Học viện Công nghệ Bưu chính Viễn thông (PTIT)
* **Giảng viên hướng dẫn:** PGS.TS. Phạm Văn Cường
* **Loại dự án:** Lựa chọn 2 – Dự án xử lý ảnh thực tế (kết hợp thử nghiệm mã nguồn mở có sẵn theo hướng dẫn tại Trang 7 `DoAn_XuLyAnh.pdf`)
* **Kiến thức học phần áp dụng (bám sát bộ bài giảng `docs/1_XLA1` $\rightarrow$ `9_XLA9`):**
  * **Chương 2 (`2_XLA2_Co ban ve XLA.pdf`):** Hệ thống thị giác người (mắt nhạy cảm độ sáng hơn màu sắc – Slide 40), thu nhận ảnh cảm quang, lấy mẫu & lượng tử hóa 8-bit ($256$ mức xám – Slide 23–25), độ phân giải không gian & mức cường độ sáng (Slide 26–38).
  * **Chương 3 (`3_XLA3_Nang cao chat luong anh.pdf`):** Toán tử điểm $g(x,y) = T[f(x,y)]$, phân ngưỡng, biến đổi tuyến tính tăng độ tương phản (Slide 9–21); Lược đồ xám (Histogram) & Cân bằng lược đồ xám HE/CLAHE (Slide 22–38); Toán tử láng giềng làm mịn & Khôi phục ảnh nhiễu bằng Bộ lọc trung vị (Median Filter) và Bộ lọc trung bình có điều kiện bảo toàn biên (Slide 43–80).
  * **Chương 4 (`4_XLA4_ Cac he mau va ky thuat in anh.pdf`):** Định dạng ảnh JPG/PNG (Slide 13–23), mô hình màu cộng RGB (Slide 28–31) và mô hình tách biệt Độ sáng (Value/Luminance) khỏi Độ màu & Độ bão hòa (Saturation – Slide 37–41).
  * **Chương 5 (`5_XLA5_Phat hien bien.pdf`):** Ba bước phát hiện biên, Đạo hàm bậc 1 – Bộ lọc Sobel & phân ngưỡng biên (Slide 11–25), Đạo hàm bậc 2 – Toán tử Laplacian & Tăng cường ảnh bằng Laplacian (Slide 27–35), Phân vùng biên vs. miền đồng nhất (Slide 55–58).
  * **Chương 6 (`6_XLA6_Nen anh.pdf`) & Chương 9 (`9_XLA9_Dao duc AI trong XLA.pdf`):** Chuẩn nén JPEG khối $8 \times 8$ DCT & lượng tử hóa (Slide 40–45), tiêu chí đánh giá khách quan MSE/SNR/PSNR (Slide 39), và đạo đức chỉnh sửa ảnh (Slide 5).
* **Báo cáo tổng quát:** Xem toàn văn tại [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md) và bản PDF chuẩn nộp bài tại [`docs/BaoCao_DoAn_XuLyAnh.pdf`](docs/BaoCao_DoAn_XuLyAnh.pdf).

---

## 1. Môi trường & Yêu cầu Hệ thống (Environment)

* **Hệ điều hành:** Linux (Ubuntu 20.04/22.04/24.04), Windows 10/11 hoặc macOS (Chạy 100% trên CPU thông thường).
* **Phiên bản Python:** Python `>= 3.10`.
* **Thư viện sử dụng (`requirements.txt`):** `opencv-python`, `scikit-image`, `PyWavelets`, `numpy`, `pandas`, `matplotlib`, `nbformat`, `onnxruntime`, `imageio-ffmpeg`.

---

## 2. Hướng dẫn Cài đặt (Installation)

```bash
pip install -r requirements.txt
```

---

## 3. Cấu trúc Thư mục & Đường dẫn Dữ liệu (Dataset & Directory Structure)

Toàn bộ dữ liệu kiểm thử (**90 ảnh tĩnh** gồm 60 ảnh bay UAV thực nghiệm + 30 cặp ảnh đối chứng Ground Truth, cùng **3 video bay FPV thực tế** và **6 ảnh mẫu**) đã được tích hợp sẵn trong thư mục `data/`:

```text
XLA/
├── BAO_CAO_DO_AN.md                        # Báo cáo tổng quát đồ án (bản Markdown đầy đủ)
├── docs/
│   ├── BaoCao_DoAn_XuLyAnh.pdf             # Báo cáo tổng quát đồ án (bản PDF chuẩn 4-10 trang để nộp)
│   ├── DoAn_XuLyAnh.pdf                    # File hướng dẫn đồ án của PGS.TS. Phạm Văn Cường
│   └── 1_XLA1... -> 9_XLA9...pdf           # Bộ bài giảng gốc của học phần Xử lý ảnh
├── data/                                   # [ĐƯỜNG DẪN DỮ LIỆU ĐÃ TÍCH HỢP SẴN]
│   ├── samples/                            # 6 ảnh mẫu tiêu biểu phục vụ chạy demo nhanh
│   ├── real_uav/                           # 60 ảnh bay UAV thực nghiệm đa bối cảnh (>= 50 ảnh theo quy định)
│   ├── synthetic_gt/                       # 30 ảnh UAV gốc sạch làm nhãn chuẩn (Ground Truth)
│   ├── synthetic_degraded/                 # 30 ảnh suy giảm mô phỏng phần cứng OV2640 tương ứng
│   ├── raw_videos/                         # 3 video bay FPV thực tế từ camera ESP32-CAM
│   └── dataset_manifest.csv                # Bảng log chi tiết nguồn và tham số của toàn bộ 90 ảnh
├── src/
│   ├── __init__.py
│   ├── uav_pipeline.py                     # Pipeline v2 (thích nghi: hồng/AWB/nhiễu/Retinex/ESPCN-Y)
│   ├── uav_pipeline_v1.py                  # Bản gốc đóng băng để so sánh "Trước"
│   ├── data_augmentation.py                # Sinh tập augmented đa kịch bản có seed (data/augmented/)
│   └── build_notebooks_and_results.py      # Script chạy thực nghiệm 90 ảnh, xuất hình 300 DPI, bảng CSV & Video
├── notebooks/
│   ├── 00_Full_Pipeline_Master.ipynb       # [Tổng hợp] Notebook báo cáo End-to-End có sẵn kết quả inline
│   ├── 01_Data_Prep_and_Degradation.ipynb  # [TV2 & TV4] Chuẩn bị 90 ảnh đa nguồn & Mô hình suy giảm OV2640
│   ├── 02_Stage1_Preprocess_PinkFix.ipynb  # [TV1] Toán tử điểm & Phân ngưỡng mềm khử hồng + Lọc nhiễu (Ch.3, 4, 6)
│   ├── 03_Stage2_LAB_CLAHE_Baselines.ipynb # [TV3] Tách kênh sáng, Kéo giãn tương phản & CLAHE + Histogram (Ch.3, 4)
│   ├── 04_Stage3_Sharpen_Upscale720p.ipynb # [TV2] Phát hiện biên Sobel + Tăng cường Laplacian & Nâng HD (Ch.2, 5)
│   └── 05_Evaluation_50Images_Tables.ipynb # [TV4] Đánh giá định lượng 90 ảnh & Xuất Bảng 1, Bảng 2 (Ch.3, 5, 6)
├── results/
│   ├── figures/                            # 4 hình minh họa chuẩn 300 DPI cho báo cáo
│   ├── hd_outputs/                         # Bộ ảnh đầu ra độ phân giải thực 1:1 (HD 720p & Full HD 1080p)
│   ├── tables/                             # Bảng số liệu định lượng CSV (table1_synthetic, table2_real_uav)
│   └── video/                              # 4 video demo kết quả khôi phục và đối sánh Side-by-Side (720p & 1080p)
├── web/
│   ├── server.py                           # Backend FastAPI xử lý upload Ảnh/Video & gọi Pipeline 3 giai đoạn
│   └── index.html                          # Giao diện Web kéo thả (Drag & Drop), Slider so sánh & Player MP4
├── requirements.txt
├── PLAN.md
└── README.md
```

---

## 4. Lệnh chạy Chương trình (Execution Commands)

### Chạy ứng dụng Web kéo thả Ảnh/Video (`XLA UAV Restoration Studio`):
```bash
python3 web/server.py
```
Sau khi khởi động, mở trình duyệt tại **`http://localhost:8000`** (có thể đổi cổng bằng biến môi trường `PORT=8080 python3 web/server.py`):
* **Kéo & thả (Drag & Drop)** hoặc chọn bất kỳ file Ảnh (`.jpg`, `.png`, `.webp`, `.bmp`) hoặc Video (`.mp4`, `.avi`, `.mov`, `.mkv`) — pipeline tự động chạy ngay và xuất kết quả mới.
* Hoặc bấm trực tiếp vào các **mẫu UAV có sẵn** (`data/samples/`, `data/raw_videos/`) để chạy thử nhanh.
* Hỗ trợ **Slider so sánh Trước / Sau**, xem từng giai đoạn (Stage 1 / Stage 2 / Output), theo dõi tiến trình xử lý video theo frame, xem các chỉ số định lượng (Pink Cast Index, Entropy, Laplacian Variance) và tải file kết quả về máy.

### Chạy kịch bản tái tạo toàn bộ Bảng số liệu, Hình minh họa 300 DPI, Video Demo & Notebooks:
```bash
python3 src/build_notebooks_and_results.py
```

### Mở báo cáo tương tác trên Jupyter Notebook / VS Code:
```bash
jupyter notebook notebooks/00_Full_Pipeline_Master.ipynb
```

---

## 5. Kết quả Trước vs Sau cải thiện (pipeline v2)

Môi trường chạy: `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`. Tạo tập augmented: `python -m src.data_augmentation --seed 2026`. Chạy benchmark: `python -m src.build_notebooks_and_results` (bản "Trước" là `src/uav_pipeline_v1.py`). Bảng đầy đủ ở `results/tables/` (table3..table8), hình `results/figures/fig5..fig8`.

| Nhóm | Chỉ số | Đầu vào | Trước (v1) | Sau (v2) |
|---|---|---|---|---|
| 30 cặp synthetic gốc | PSNR / SSIM / ΔE | 20.35 / 0.702 / 24.74 | **27.42 / 0.818 / 10.06** | 25.39 / 0.700 / 14.55 |
| haze_pink | PSNR / ΔE | 15.84 / 42.38 | 20.36 / 25.03 | **25.22 / 15.98** |
| low_light | PSNR / ΔE / L̄ / % đen | 14.32 / 52.32 / 16.6 / 0.8 | 10.82 / 79.96 / 4.3 / 51.4 | **18.25 / 32.51 / 30.9 / 0.06** |
| backlit_uneven | PSNR / ΔE / % đen | 16.86 / 35.01 / 2.3 | 13.51 / 53.62 / 29.4 | **17.24 / 34.13 / 1.5** |
| motion_jpeg | PSNR / ΔE | 28.67 / 7.21 | 21.54 / 19.37 | 24.50 / 15.16 |
| 60 ảnh thật | PCI % / % đen | 17.45 / 0 | 7.03 / 1.94 | **1.38 / 0.38** |
| Ảnh tối 051–056 | L̄ | 22–35 | 5.7–19.6 (052/053: 41–69 % đen) | **36–44, 0 % đen** |

Lưu ý trung thực: trên 30 cặp synthetic gốc, v1 vẫn tốt hơn v2 về PSNR/SSIM/ΔE vì v1 dùng pivot cố định 172 trùng đúng mô hình sinh suy giảm của tập này; v2 là bản thích nghi nên đổi lấy việc không phá ảnh tối. Với `motion_jpeg`, v2 không vượt ảnh đầu vào về PSNR/ΔE (các cảnh GT vốn tối bị nâng sáng, xem `table7`). Thời gian chạy tuần tự v2 ≈ 5.5 s/ảnh 800×600 so với v1 ≈ 4.1 s.
