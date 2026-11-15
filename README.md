# ĐỒ ÁN XỬ LÝ ẢNH (IMAGE PROCESSING)
## Đề tài: Xây dựng pipeline khôi phục màu sắc và nâng cao chất lượng ảnh thu nhận từ UAV sử dụng camera giá rẻ ESP32-CAM

**Giảng viên hướng dẫn:** ThS/TS. Phạm Văn Cường  
**Nhóm thực hiện:** 4 thành viên (Mỗi thành viên đóng góp 25%)

---

## 1. Giới thiệu tổng quan
Dự án xây dựng một pipeline xử lý ảnh cổ điển (bám sát kiến thức **Chương 2, Chương 3 và Chương 4** trong học phần Xử lý ảnh) nhằm khắc phục 3 suy giảm chất lượng ảnh đặc trưng của module camera giá rẻ **ESP32-CAM (cảm biến OV2640)** gắn trên thiết bị bay không người lái (UAV) truyền hình ảnh qua Wi-Fi ($2.5 - 6.5\text{ fps}$):
1. **Lệch màu cháy sáng hồng/tím (Pink/Magenta Highlight Cast):** Do bão hòa kênh Đỏ ($R$) và Xanh dương ($B$) trong khi kênh Xanh lá ($G$) bị sụt giảm tại vùng phản xạ mạnh (mặt đường bê tông, mái nhà, xe sáng màu, bầu trời).
2. **Độ tương phản thấp và phủ sương mờ (Low Contrast & Haze):** Biểu đồ mức xám (Histogram) bị co cụm ở dải giữa do thấu kính vi hình và điều kiện ngoài trời.
3. **Mờ nhòe chuyển động và nhiễu khối nén JPEG (Motion Blur & JPEG Block Artifacts):** Do rung động cơ học từ cánh quạt UAV và nén JPEG mạnh khi truyền qua Wi-Fi.

Toàn bộ thuật toán chạy **100% trên CPU thông thường** (không cần GPU, không cần huấn luyện mạng học sâu), đạt tốc độ thời gian thực **$> 40\text{ FPS}$** trên độ phân giải gốc và hỗ trợ **Nội suy Bicubic nâng chuẩn hiển thị lên HD 720p**.

---

## 2. Cấu trúc thư mục dự án

```text
XLA/
├── data/
│   ├── raw_videos/                         # 3 video bay FPV thực tế từ ESP32-CAM (OV2640)
│   ├── samples/                            # 6 ảnh mẫu tiêu biểu để chạy demo nhanh
│   ├── real_uav/                           # 60 ảnh UAV kiểm thử đa nguồn đã qua Data Augmentation
│   ├── synthetic_gt/                       # 30 ảnh UAV ngoài trời sạch (Ground Truth) từ 30 chuyến bay độc lập
│   ├── synthetic_degraded/                 # 30 ảnh giả lập suy giảm phần cứng OV2640 ngẫu nhiên
│   └── dataset_manifest.csv                # Bảng log chi tiết tham số Augmentation của toàn bộ 90 ảnh
├── notebooks/
│   ├── 00_Full_Pipeline_Master.ipynb       # [Tổng hợp] Notebook báo cáo hoàn chỉnh (có sẵn toàn bộ ảnh & bảng)
│   ├── 01_Data_Prep_and_Degradation.ipynb  # [TV2 & TV4] Chuẩn bị dữ liệu đa nguồn & Augmentation chống Overfit
│   ├── 02_Stage1_Preprocess_PinkFix.ipynb  # [TV1] Lọc Bilateral + Mặt nạ nội suy mềm khử hồng + Gray World AWB
│   ├── 03_Stage2_LAB_CLAHE_Baselines.ipynb # [TV3] CIE LAB CLAHE + Gamma + Đối chứng Global HE + Histogram RGB
│   ├── 04_Stage3_Sharpen_Upscale720p.ipynb # [TV2] Làm sắc nét biên Unsharp Masking + Nội suy Bicubic nâng 720p
│   └── 05_Evaluation_50Images_Tables.ipynb # [TV4] Đánh giá thực nghiệm 90 ảnh & Xuất Bảng 1 (PSNR/SSIM), Bảng 2 (Entropy/FPS)
├── src/
│   ├── __init__.py
│   └── uav_pipeline.py                     # Module Python chứa toàn bộ hàm xử lý 3 giai đoạn, Baselines & Độ đo
├── results/
│   ├── figures/                            # Hình ảnh minh họa xuất chuẩn báo cáo (mặt nạ hồng, Histogram, lưới 4 cột)
│   ├── tables/                             # File CSV kết quả định lượng (table1_synthetic_metrics.csv, table2_real_uav_metrics.csv)
│   └── video/                              # Video Demo Side-by-Side 720p (demo_esp32_restoration_720p.mp4)
├── requirements.txt
├── PLAN.md
└── README.md
```

---

## 3. Hướng dẫn cài đặt và chạy thử nghiệm

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Mở và chạy trên Jupyter Notebook / VS Code
* Toàn bộ 6 file trong thư mục `notebooks/` **đã được chạy sẵn và lưu kèm kết quả hiển thị (inline plots & tables)**.
* Để xem báo cáo tổng hợp từ đầu đến cuối, mở ngay file:
  * [`notebooks/00_Full_Pipeline_Master.ipynb`](notebooks/00_Full_Pipeline_Master.ipynb)
* Hoặc mở từng Notebook thành phần từ `01` đến `05` tương ứng với phần việc của 4 thành viên.
# XLA
