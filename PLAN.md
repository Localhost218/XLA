# KẾ HOẠCH CHI TIẾT & ĐẶC TẢ TOÁN HỌC ĐỒ ÁN XỬ LÝ ẢNH (XLA)

**Đề tài:** Xây dựng Pipeline Khôi phục Màu sắc, Khử nhiễu Đa miền và Nâng cao Chất lượng Ảnh thu nhận từ UAV sử dụng Camera giá rẻ ESP32-CAM (OV2640)  
**Loại dự án:** Lựa chọn 2 – Dự án xử lý ảnh thực tế (kết hợp thử nghiệm mô hình học sâu mã nguồn mở có sẵn **ESPCN** theo Lựa chọn 1 ở bước siêu phân giải)  
**Giảng viên hướng dẫn:** PGS.TS. Phạm Văn Cường  
**Quy mô nhóm:** 4 thành viên (Đóng góp đều 25%/thành viên)  
**Kiến thức học phần áp dụng:** Chương 2, Chương 3, Chương 4, Chương 5, Chương 6 và Chương 9

---

## 1. Kiến trúc Thư mục Dự án

Hệ thống được tổ chức đồng bộ gồm module lõi [`src/uav_pipeline.py`](file:///home/ngtukien/Projects/XLA/src/uav_pipeline.py), 6 tệp Jupyter Notebook trong [`notebooks/`](file:///home/ngtukien/Projects/XLA/notebooks), tập dữ liệu 90 ảnh kiểm thử + 3 video FPV trong [`data/`](file:///home/ngtukien/Projects/XLA/data) và báo cáo tổng quát tại [`BAO_CAO_DO_AN.md`](file:///home/ngtukien/Projects/XLA/BAO_CAO_DO_AN.md) cùng [`docs/BaoCao_DoAn_XuLyAnh.pdf`](file:///home/ngtukien/Projects/XLA/docs/BaoCao_DoAn_XuLyAnh.pdf):

```text
XLA/
├── BAO_CAO_DO_AN.md                      # Báo cáo tổng quát đồ án (bản Markdown đầy đủ)
├── docs/
│   ├── BaoCao_DoAn_XuLyAnh.pdf           # Báo cáo tổng quát đồ án (bản PDF 4-10 trang theo quy định nộp bài)
│   └── DoAn_XuLyAnh.pdf                  # Tài liệu yêu cầu đồ án của PGS.TS. Phạm Văn Cường
├── data/
│   ├── samples/                          # 6 ảnh mẫu đặc trưng để chạy demo trực quan
│   ├── real_uav/                         # 60 ảnh bay UAV thực nghiệm đa bối cảnh (>= 50 ảnh theo quy định)
│   ├── synthetic_gt/                     # 30 ảnh UAV ngoài trời sạch làm Ground Truth (GT)
│   ├── synthetic_degraded/               # 30 ảnh giả lập suy giảm phần cứng OV2640 từ GT
│   ├── raw_videos/                       # 3 video bay FPV thực tế từ camera ESP32-CAM
│   └── dataset_manifest.csv              # Bảng thông tin nguồn và tham số suy giảm của 90 ảnh
├── models/
│   ├── espcn_super_res.onnx              # Mạng học sâu siêu phân giải mã nguồn mở ESPCN 3x (ONNX)
│   └── realesrgan_vulkan/                # Công cụ siêu phân giải mã nguồn mở Real-ESRGAN NCNN Vulkan
├── notebooks/
│   ├── 00_Full_Pipeline_Master.ipynb     # [Cả nhóm] Notebook tổng hợp End-to-End (Đầy đủ ảnh, công thức & bảng)
│   ├── 01_Data_Prep_and_Degradation.ipynb# [TV2 & TV4] Chuẩn bị 90 ảnh đa nguồn & Mô hình suy giảm OV2640
│   ├── 02_Stage1_Preprocess_PinkFix.ipynb# [TV1] Mặt nạ khử cháy hồng α(x,y) + Masked AWB + Wavelet/NLM Denoise
│   ├── 03_Stage2_LAB_CLAHE_Baselines.ipynb#[TV3] CIE LAB Haze-Pivot + CLAHE + Đối chứng Global HE + Histogram
│   ├── 04_Stage3_Sharpen_Upscale720p.ipynb#[TV2] Giải chập Richardson-Lucy + Cổng biên Sobel + ESPCN 720p/1080p
│   └── 05_Evaluation_50Images_Tables.ipynb#[TV4] Đánh giá thực nghiệm 90 ảnh & Xuất Bảng 1 (PSNR/SSIM), Bảng 2
├── src/
│   ├── __init__.py
│   ├── uav_pipeline.py                   # Module Python chứa toàn bộ hàm 3 giai đoạn, Baselines & Độ đo
│   └── build_notebooks_and_results.py    # Kịch bản tự động chạy thực nghiệm và xuất toàn bộ kết quả
├── results/
│   ├── figures/                          # 4 hình minh họa xuất chuẩn 300 DPI cho báo cáo
│   ├── hd_outputs/                       # Ảnh đầu ra độ phân giải thực 1:1 (HD 720p & Full HD 1080p)
│   ├── tables/                           # Bảng kết quả định lượng CSV (table1_synthetic, table2_real_uav)
│   └── video/                            # 4 video demo Side-by-Side (Gốc vs Khôi phục 720p & 1080p)
├── requirements.txt
├── README.md
└── PLAN.md
```

---

## 2. Đặc tả Toán học & Thuật toán Chi tiết (Mathematical & Algorithmic Specs)

### Giai đoạn 1: Tiền xử lý — Khôi phục vùng cháy sáng hồng & Khử nhiễu lai đa miền (Thành viên 1)

1. **Bước 1.1: Phát hiện và Nội suy mềm vùng cháy sáng hồng (Chương 2 & Chương 4)**
   - **Vấn đề vật lý trên OV2640:** Khi phơi sáng cao, kênh $R$ và $B$ đạt ngưỡng bão hòa sớm trong khi kênh $G$ bị sụt giảm phi tuyến, làm các vùng trắng/xám biến thành màu hồng/tím (Magenta).
   - **Mặt nạ trọng số mềm $\alpha(x,y) \in [0, 1]$:**
     $$RB_{\text{mean}}(x,y) = \frac{R(x,y) + B(x,y)}{2}, \qquad D_{\text{pink}}(x,y) = \max\big(0, \; RB_{\text{mean}}(x,y) - G(x,y)\big)$$
     $$\alpha(x,y) = G_{7\times 7, \sigma=1.5} * \left[ \text{clip}\left(\frac{RB_{\text{mean}} - T_{\text{high}}}{40.0}, 0, 1\right) \cdot \text{clip}\left(\frac{D_{\text{pink}} - \Delta T}{16.0}, 0, 1\right) \right]$$
     với $T_{\text{high}} = 135.0, \Delta T = 7.0$.
   - **Công thức nội suy khôi phục kênh màu:** Với $Y_{\text{est}} = G + 0.45 D_{\text{pink}}$:
     $$G' = (1 - \alpha) G + \alpha Y_{\text{est}}, \qquad R' = (1 - \alpha) R + \alpha [Y_{\text{est}} + 0.08(R - Y_{\text{est}})], \qquad B' = (1 - \alpha) B + \alpha [Y_{\text{est}} + 0.08(B - Y_{\text{est}})]$$

2. **Bước 1.2: Cân bằng trắng tự động có loại trừ vùng bão hòa (Masked Gray World AWB - Chương 4)**
   - Chỉ tính trung bình $\mu_B, \mu_G, \mu_R$ trên các điểm ảnh trung gian không bị bão hòa ($30 < Y < 215$ và $\alpha(x,y) < 0.20$), giới hạn độ dịch chuyển $\pm 6\%$ ($\text{max\_gain\_shift} = 0.06$).

3. **Bước 1.3: Khử nhiễu lai đa miền Wavelet BayesShrink + Non-Local Means + Chroma Bilateral (Chương 3 & Chương 6)**
   - **Tầng 1 (Miền Wavelet):** Phân rã Wavelet rời rạc 3 mức với họ sóng con `sym4` trên không gian YCbCr (`skimage.restoration.denoise_wavelet`), co ngưỡng mềm thích nghi `BayesShrink`.
   - **Tầng 2 (Miền không gian phi cục bộ có bảo vệ tán lá):** Áp dụng `cv2.fastNlMeansDenoising` ($h = 4.5$, cửa sổ mẫu $7\times7$, vùng tìm kiếm $15\times15$) trên kênh độ chói $L^*$ để xóa sạch 100% hạt nhiễu CMOS trên mặt đường, bầu trời; kết hợp mặt nạ thực vật $w_{\text{foliage}}$ để giữ nguyên chi tiết gân lá cây.
   - **Tầng 3 (Lọc kênh sắc độ):** Lọc trung vị $3\times3$ kết hợp lọc song phương Bilateral ($d=5, \sigma_{\text{color}}=14.0, \sigma_{\text{space}}=7.0$) trên hai kênh sắc độ $a^*, b^*$.

---

### Giai đoạn 2: Xử lý chính — Khử sương mờ Haze-Pivot & Tăng cường tương phản cục bộ CIE LAB (Thành viên 3)

1. **Bước 2.1: Chuyển đổi không gian màu `BGR` $\rightarrow$ `CIE L*a*b*` (Chương 4)**
   - Tách kênh độ chói $L^*$ khỏi hai kênh sắc độ $a^*, b^*$ để thao tác tăng tương phản không làm biến dạng màu sắc.
2. **Bước 2.2: Khôi phục tương phản quanh điểm tựa độ chói sương mờ (Haze-Pivot Contrast Restoration - Chương 2 & 3)**
   - Kéo giãn tuyến tính kênh $L^*$ quanh điểm tựa màn sương $L_{\text{pivot}} = 172.0$ với hệ số khuếch đại $\gamma_{\text{gain}} = 1.26$ (tự động thích nghi theo độ rộng phân vị $P_{98} - P_2$):
     $$L_{\text{dehaze}}^* = \text{clip}\big(L_{\text{pivot}}^{\text{eff}} + \gamma_{\text{gain}}^{\text{eff}} (L^* - L_{\text{pivot}}^{\text{eff}}), \; 0, \; 255\big)$$
3. **Bước 2.3: Cân bằng lược đồ thích nghi giới hạn tương phản (CLAHE - Chương 3) & Bù bão hòa sắc độ**
   - Áp dụng CLAHE (`clipLimit = 1.5`, `tileGridSize = (8,8)`) pha trộn ở tỷ lệ $\beta_{\text{CLAHE}} = 0.22$ cùng hệ số bù sắc độ $c_{\text{gain}} \in [1.24, 1.58]$ trên hai kênh $a^*, b^*$.

---

### Giai đoạn 3: Hậu xử lý — Giải chập quang học Richardson-Lucy, Làm nét có cổng biên Sobel & Siêu phân giải ESPCN (Thành viên 2)

1. **Bước 3.1: Giải chập lặp quang học Richardson-Lucy PSF (Chương 3)**
   - Khử mờ quang học của thấu kính OV2640 bằng 4 vòng lặp giải chập Richardson-Lucy với nhân Gaussian PSF $5\times5$ ($\sigma_{\text{PSF}} = 0.88$).
2. **Bước 3.2: Unsharp Masking có cổng đạo hàm Sobel & mặt nạ thực vật (Chương 3 & Chương 5)**
   - Dùng toán tử đạo hàm bậc nhất **Sobel** phát hiện biên cấu trúc $w_{\text{edge}}$ kết hợp mặt nạ thực vật $w_{\text{foliage}}$ để chỉ cộng chi tiết sắc nét vào vật thể (xe cộ, vạch kẻ đường, tòa nhà, cây cối) và giữ vùng mặt đường phẳng mịn tuyệt đối ($w_{\text{active}} = 0$).
3. **Bước 3.3: Nâng cấp Siêu phân giải bằng mạng học sâu mã nguồn mở ESPCN kết hợp Lanczos-4 (Chương 2 + SOTA)**
   - Thực thi mô hình **ESPCN $3\times$ (ONNX)** trên kênh độ chói $L^*$ kết hợp nội suy **Lanczos-4** để nâng chuẩn hiển thị lên **HD 720p ($960 \times 720$ / $1280 \times 720$)** và **Full HD 1080p ($1440 \times 1080$ / $1920 \times 1080$)**.

---

## 3. Tóm tắt Kết quả Thực nghiệm Đạt được (Thành viên 4)

| Nhóm dữ liệu | Chỉ số (Metric) | Ảnh gốc / Suy giảm | Đối chứng Global HE (RGB) | Pipeline Đề xuất (GĐ 1+2 / Đầy đủ) | Mức độ cải thiện |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tập Synthetic (30 cặp ảnh có GT)** | **PSNR (dB)** $\uparrow$ | $20.35\text{ dB}$ | $12.53\text{ dB}$ | **$27.66\text{ dB} \; / \; 27.42\text{ dB}$** | **$+7.07\text{ dB}$** so với gốc (**$+14.89\text{ dB}$** so với Global HE) |
| | **SSIM** $\uparrow$ | $0.7015$ | $0.4155$ | **$0.8295 \; / \; 0.8180$** | **$+0.1280$** ($\approx +18.2\%$) về độ tương đồng cấu trúc |
| | **$\Delta E_{ab}^*$ (Sai số màu)** $\downarrow$ | $24.74$ | $59.38$ | **$9.89 \; / \; 10.06$** | Giảm **$60.0\%$** sai lệch màu sắc trong CIE LAB |
| **Tập Real UAV (60 ảnh thực nghiệm)** | **Shannon Entropy ($H$)** $\uparrow$ | $6.364\text{ bits}$ | $7.461\text{ bits}$ (nhiễu) | **$6.850 - 6.860\text{ bits}$** | Tăng **$+0.496\text{ bits}$** thông tin chi tiết thực |
| | **Laplacian Variance** $\uparrow$ | $121.1$ | $1108.4$ (nhiễu hạt) | **$226.5$** | Tăng độ sắc nét biên gấp **$1.87$ lần** |
| | **Pink Cast Index (PCI)** $\downarrow$ | $17.45\%$ | $12.28\%$ | **$2.55\%$ (GĐ 1) / $7.03\%$ (Full)** | Giảm **$85.4\%$** (sau GĐ 1) & **$> 60\%$** (toàn cục) |
