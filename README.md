# ĐỒ ÁN XỬ LÝ ẢNH (IMAGE PROCESSING) — PTIT
## Đề tài: Xây dựng Pipeline Khôi phục Màu sắc, Khử nhiễu Đa miền và Nâng cao Chất lượng Ảnh thu nhận từ UAV sử dụng Camera giá rẻ ESP32-CAM (OV2640)

* **Đơn vị:** Khoa Trí tuệ Nhân tạo – Học viện Công nghệ Bưu chính Viễn thông (PTIT)
* **Giảng viên hướng dẫn:** PGS.TS. Phạm Văn Cường
* **Loại dự án:** Lựa chọn 2 – Dự án xử lý ảnh thực tế (kết hợp tích hợp mô hình học sâu mã nguồn mở có sẵn **ESPCN** ở bước nâng cấp siêu phân giải HD 720p / Full HD 1080p)
* **Kiến thức học phần áp dụng:** Chương 2, Chương 3, Chương 4, Chương 5, Chương 6 và Chương 9
* **Báo cáo chi tiết:** Xem toàn văn tại [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md) và bản in PDF chuẩn học thuật tại [`docs/BaoCao_DoAn_XuLyAnh.pdf`](docs/BaoCao_DoAn_XuLyAnh.pdf)

---

## 1. Môi trường & Yêu cầu Hệ thống (Environment)

* **Hệ điều hành:** Linux (Ubuntu 20.04/22.04/24.04), Windows 10/11 hoặc macOS (Chạy 100% trên CPU thông thường, không bắt buộc GPU rời).
* **Phiên bản Python:** Python `>= 3.10` (Khuyến nghị Python `3.11` hoặc `3.12`).
* **Các thư viện cốt lõi (`requirements.txt`):**
  * `opencv-python >= 4.8.0`: Biến đổi không gian màu CIE $L^*a^*b^*$, CLAHE, lọc Non-Local Means, Median, Bilateral, toán tử Sobel/Laplacian, nội suy Lanczos-4 và xử lý video.
  * `scikit-image >= 0.21.0` & `PyWavelets >= 1.4.1`: Lọc khử nhiễu đa phân giải Wavelet ngưỡng mềm BayesShrink (`denoise_wavelet`), giải chập lặp quang học Richardson-Lucy PSF (`richardson_lucy`) và tính độ đo chuẩn `PSNR`, `SSIM`.
  * `onnxruntime >= 1.16.0`: Thực thi mô hình học sâu siêu phân giải mã nguồn mở **ESPCN (Sub-Pixel CNN $3\times$)** (`models/espcn_super_res.onnx`).
  * `numpy`, `pandas`, `matplotlib`, `nbformat`, `jupyter`, `imageio-ffmpeg`: Tính toán ma trận, xuất biểu đồ 300 DPI, tạo bảng CSV, thực thi Notebook và mã hóa video H.264.

---

## 2. Hướng dẫn Cài đặt (Installation)

Từ thư mục gốc của dự án (`XLA/`), chạy lệnh cài đặt toàn bộ thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

---

## 3. Cấu trúc Thư mục & Đường dẫn Dữ liệu (Dataset & Directory Structure)

Toàn bộ dữ liệu kiểm thử (**90 ảnh tĩnh** gồm 60 ảnh bay UAV thực nghiệm + 30 cặp ảnh đối chứng Ground Truth, cùng **3 video bay FPV thực tế** và **6 ảnh mẫu**) đã được đóng gói sẵn trực tiếp trong thư mục `data/` của kho lưu trữ (không cần tải thủ công từ bên ngoài):

```text
XLA/
├── BAO_CAO_DO_AN.md                        # Báo cáo tổng quát đồ án (bản Markdown đầy đủ)
├── docs/
│   ├── BaoCao_DoAn_XuLyAnh.pdf             # Báo cáo tổng quát đồ án (bản PDF chuẩn 4-10 trang để nộp)
│   └── DoAn_XuLyAnh.pdf                    # File quy định yêu cầu đồ án của PGS.TS. Phạm Văn Cường
├── data/                                   # [ĐƯỜNG DẪN DỮ LIỆU ĐÃ TÍCH HỢP SẴN]
│   ├── samples/                            # 6 ảnh mẫu tiêu biểu phục vụ chạy demo nhanh
│   ├── real_uav/                           # 60 ảnh bay UAV thực nghiệm đa bối cảnh (>= 50 ảnh theo quy định)
│   ├── synthetic_gt/                       # 30 ảnh UAV gốc sạch làm nhãn chuẩn (Ground Truth)
│   ├── synthetic_degraded/                 # 30 ảnh suy giảm mô phỏng phần cứng OV2640 tương ứng
│   ├── raw_videos/                         # 3 video bay FPV thực tế từ camera ESP32-CAM
│   └── dataset_manifest.csv                # Bảng log chi tiết nguồn và tham số của toàn bộ 90 ảnh
├── models/
│   ├── espcn_super_res.onnx                # Mô hình học sâu mã nguồn mở ESPCN 3x Super-Resolution (ONNX)
│   └── realesrgan_vulkan/                  # Bộ công cụ mã nguồn mở Real-ESRGAN NCNN Vulkan
├── src/
│   ├── __init__.py
│   ├── uav_pipeline.py                     # Module lõi chứa toàn bộ Pipeline 3 giai đoạn, Baselines & Độ đo
│   └── build_notebooks_and_results.py      # Script chạy thực nghiệm 90 ảnh, xuất hình 300 DPI, bảng CSV & Video
├── notebooks/
│   ├── 00_Full_Pipeline_Master.ipynb       # [Tổng hợp] Notebook báo cáo End-to-End có sẵn kết quả inline
│   ├── 01_Data_Prep_and_Degradation.ipynb  # [TV2 & TV4] Chuẩn bị 90 ảnh đa nguồn & Mô hình suy giảm OV2640
│   ├── 02_Stage1_Preprocess_PinkFix.ipynb  # [TV1] Mặt nạ khử cháy hồng α(x,y) + Masked AWB + Wavelet/NLM Denoise
│   ├── 03_Stage2_LAB_CLAHE_Baselines.ipynb # [TV3] CIE LAB Haze-Pivot + CLAHE + Đối chứng Global HE + Histogram
│   ├── 04_Stage3_Sharpen_Upscale720p.ipynb # [TV2] Giải chập Richardson-Lucy + Sobel Gate + ESPCN 720p/1080p
│   └── 05_Evaluation_50Images_Tables.ipynb # [TV4] Đánh giá định lượng 90 ảnh & Xuất Bảng 1, Bảng 2
├── results/
│   ├── figures/                            # 4 hình minh họa chuẩn 300 DPI cho báo cáo
│   ├── hd_outputs/                         # Bộ ảnh đầu ra độ phân giải thực 1:1 (HD 720p & Full HD 1080p)
│   ├── tables/                             # Bảng số liệu định lượng CSV (table1_synthetic, table2_real_uav)
│   └── video/                              # 4 video demo kết quả khôi phục và đối sánh Side-by-Side (720p & 1080p)
├── requirements.txt                        # Danh sách thư viện cài đặt
├── PLAN.md                                 # Kế hoạch thiết kế toán học và lộ trình thực hiện
└── README.md                               # Tài liệu hướng dẫn sử dụng này
```

---

## 4. Lệnh chạy Chương trình (Execution Commands)

### Cách 1: Chạy kịch bản tự động tái tạo toàn bộ Bảng số liệu, Hình minh họa 300 DPI, Video Demo & Notebooks
```bash
python3 src/build_notebooks_and_results.py
```

### Cách 2: Chạy khôi phục và nâng cấp siêu phân giải cho một ảnh bất kỳ bằng Python CLI
```bash
python3 -c "
import cv2
from src.uav_pipeline import run_full_pipeline
img = cv2.imread('data/samples/sample_sim_pink_road_1.jpg')
res = run_full_pipeline(img, return_intermediates=True)
cv2.imwrite('results/demo_output_720p.png', res['output_720p'])
cv2.imwrite('results/demo_output_1080p.png', res['output_1080p'])
print('Đã xuất ảnh khôi phục HD 720p và Full HD 1080p thành công!')
"
```

### Cách 3: Mở và tương tác trực tiếp trên Jupyter Notebook / VS Code / Google Colab
Toàn bộ 6 tệp trong thư mục `notebooks/` đã được thực thi sẵn và nhúng đầy đủ biểu đồ, hình ảnh, bảng số liệu. Để khởi động Jupyter:
```bash
jupyter notebook notebooks/00_Full_Pipeline_Master.ipynb
```

---

## 5. Đầu ra Mong đợi (Expected Outputs)

Sau khi chạy lệnh thực nghiệm, hệ thống tự động sinh ra các đầu ra chuẩn xác sau:

1. **Bảng kết quả định lượng (`results/tables/table1_synthetic_metrics.csv` & `table2_real_uav_metrics.csv`):**
   * **Trên 30 cặp ảnh đối chứng (Synthetic Ground Truth):**
     * **PSNR:** Tăng từ **$20.35\text{ dB}$** (Ảnh suy giảm) và **$12.53\text{ dB}$** (Đối chứng Global HE RGB) lên **$27.66\text{ dB}$** (Giai đoạn 1+2) và **$27.42\text{ dB}$** (Full Pipeline) — cải thiện **$+7.07\text{ dB}$** so với ảnh gốc.
     * **SSIM:** Tăng từ **$0.7015$** lên **$0.8295$** (Giai đoạn 1+2) và **$0.8180$** (Full Pipeline).
     * **Sai lệch màu CIE LAB ($\Delta E_{ab}^*$):** Giảm **$60\%$**, từ **$24.74$** xuống **$9.89 - 10.06$**.
   * **Trên 60 ảnh bay UAV thực nghiệm (`data/real_uav/`):**
     * **Pink Cast Index (PCI %):** Giảm từ **$17.45\%$** xuống **$2.55\%$** sau Giai đoạn 1 và duy trì ở **$7.01\% - 7.05\%$** sau khi kích tương phản & độ bão hòa toàn cảnh.
     * **Shannon Entropy:** Tăng từ **$6.364\text{ bits}$** lên **$6.850 - 6.860\text{ bits}$**.
     * **Độ sắc nét biên (Laplacian Variance):** Tăng gấp **$1.87$ lần**, từ **$121.1$** lên **$226.5$**.
2. **Hình ảnh minh họa báo cáo (`results/figures/` & `results/hd_outputs/`):**
   * `fig1_stage1_pink_mask_recovery.png`: Minh họa mặt nạ trọng số mềm $\alpha(x,y)$ và khôi phục vùng cháy hồng.
   * `fig2_stage2_lab_clahe_histograms.png`: Đối sánh ảnh đầu vào – đầu ra và biểu đồ Histogram 3 kênh RGB.
   * `fig3_4col_comparison_grid.png`: Lưới so sánh 4 cột trên 3 cảnh bay UAV khác nhau.
   * `fig4_upscale_720p_zoom.png`: Phóng to vùng chi tiết (Zoom-in ROI) chứng minh mặt đường sạch nhiễu hạt và tán cây/xe cộ sắc nét ở chuẩn HD 720p & Full HD 1080p.
3. **Video Demo thực tế (`results/video/`):**
   * `demo_esp32_restoration_720p.mp4` & `demo_urban_uav_SideBySide_FullHD_1080p.mp4`: Video đối sánh trực tiếp Side-by-Side (Trái: Gốc ESP32-CAM, Phải: Đã khôi phục qua Pipeline).
