# KẾ HOẠCH CHI TIẾT THỰC HIỆN ĐỒ ÁN XỬ LÝ ẢNH (XLA)

**Đề tài:** Xây dựng pipeline khôi phục màu sắc và nâng cao chất lượng ảnh thu nhận từ UAV sử dụng camera giá rẻ ESP32-CAM  
**Giảng viên hướng dẫn:** ThS/TS. Phạm Văn Cường  
**Quy mô nhóm:** 4 thành viên  

---

## 1. Kiến trúc thư mục dự án (Thiết kế theo Jupyter Notebook `.ipynb`)

Để thuận tiện nhất cho môn **Xử lý ảnh** (vừa viết công thức Markdown, vừa tinh chỉnh tham số và nhìn thấy ngay ảnh kết quả, mặt nạ lỗi hồng, biểu đồ Histogram ngay dưới mỗi ô code, đồng thời chạy được trên cả **JupyterLab / VS Code / Google Colab**), dự án được tổ chức xoay quanh **hệ thống Notebook `.ipynb`** như sau:

```text
XLA/
├── data/
│   ├── samples/                          # 2-5 ảnh mẫu đặc trưng để chạy demo trực quan trên Notebook
│   ├── real_uav/                         # >= 50 ảnh trích xuất từ video UAV thực tế (640x480 / 800x600)
│   ├── synthetic_gt/                     # 20-30 ảnh ngoài trời sạch làm Ground Truth (GT)
│   └── synthetic_degraded/               # Ảnh giả lập suy giảm từ GT (ám hồng + mờ + giảm tương phản)
├── notebooks/
│   ├── 01_Data_Prep_and_Degradation.ipynb # [TV2 & TV4] Cắt frame từ video UAV & Tạo ảnh giả lập suy giảm OV2640
│   ├── 02_Stage1_Preprocess_PinkFix.ipynb # [TV1] Lọc Bilateral + Mặt nạ khử cháy sáng hồng + Gray World AWB
│   ├── 03_Stage2_LAB_CLAHE_Baselines.ipynb# [TV3] CIE LAB CLAHE + Gamma + Đối chứng Global HE + Vẽ Histogram
│   ├── 04_Stage3_Sharpen_Upscale720p.ipynb# [TV2] Làm nét Unsharp Masking + Nội suy Bicubic nâng lên 720p (960x720)
│   ├── 05_Evaluation_50Images_Tables.ipynb# [TV4] Chạy thực nghiệm >= 50 ảnh, xuất bảng PSNR, SSIM, Entropy, FPS
│   └── 00_Full_Pipeline_Master.ipynb     # [Cả nhóm] Notebook tổng hợp End-to-End (Có sẵn toàn bộ ảnh & bảng để nộp)
├── src/                                  # (Tùy chọn) Chứa file hàm dùng chung để import gọn vào các Notebook
│   └── uav_pipeline.py                   # Gom các hàm đã chốt từ Notebook 02, 03, 04 để chạy batch & xuất Video Demo
├── results/
│   ├── figures/                          # Ảnh so sánh 4 cột, ảnh từng bước, biểu đồ Histogram xuất tự động (300 DPI)
│   ├── tables/                           # File kết quả định lượng (metrics_synthetic.csv, metrics_real.csv)
│   └── video/                            # Video demo Side-by-Side (Gốc 480p vs Khôi phục 720p) (.mp4)
├── docs/
│   ├── report/                           # Báo cáo PDF (4-10 trang)
│   └── slides/                           # Slide thuyết trình (10-15 slides)
├── requirements.txt                      # Thư viện: opencv-python, numpy, scikit-image, matplotlib, pandas, jupyter
├── README.md                             # Hướng dẫn mở Notebook và chạy thử nghiệm
└── PLAN.md                               # Tài liệu kế hoạch chi tiết này
```

---

## 2. Thiết kế Toán học & Thuật toán Chi tiết (Mathematical & Algorithmic Specs)

Phần này quy định rõ công thức toán học và đầu vào/đầu ra của từng hàm để các thành viên code khớp 100% với lý thuyết Chương 2, 3, 4 và viết vào Mục 3 của báo cáo.

### Giai đoạn 1: Tiền xử lý — Khử nhiễu nén & Khôi phục vùng cháy sáng hồng (`src/stage1_preprocess.py` - Thành viên 1)

1. **Bước 1.1: Khử nhiễu khối JPEG bảo toàn biên (Bilateral Filtering - Chương 3)**
   - **Công thức:**
     $$I_{\text{bf}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \, f_r\left(\|I(x_i) - I(x)\|\right) \, g_s\left(\|x_i - x\|\right)$$
     Trong đó $f_r$ là nhân Gaussian theo miền cường độ sáng ($\sigma_{\text{color}} = 50$) và $g_s$ là nhân Gaussian theo miền không gian ($\sigma_{\text{space}} = 50$), đường kính lân cận $d = 5$.
   - **Ý nghĩa:** Làm mịn nhiễu khối $8\times8$ do nén JPEG của ESP32 nhưng giữ nguyên biên sắc nét giữa mặt đường, người và cỏ.

2. **Bước 1.2: Phát hiện và Nội suy mềm vùng cháy sáng hồng (Highlight Color Recovery - Chương 2 & 3)**
   - **Vấn đề vật lý trên OV2640:** Khi phơi sáng cao, kênh $R$ và $B$ đạt ngưỡng bão hòa ($\approx 220-255$) trong khi kênh $G$ bị sụt giảm bất thường hoặc bão hòa lệch, làm vùng trắng/xám biến thành màu hồng/tím (Magenta).
   - **Xây dựng mặt nạ trọng số mềm (Soft Highlight Mask) $\alpha(x,y) \in [0, 1]$:**
     Thay vì dùng ngưỡng cứng gây viền giả (halo artifacts), ta dùng mặt nạ mềm dựa trên độ lệch màu hồng $D_{\text{pink}}(x,y)$ và cường độ sáng:
     $$RB_{\text{mean}}(x,y) = \frac{R(x,y) + B(x,y)}{2}$$
     $$D_{\text{pink}}(x,y) = \max\left(0, \; RB_{\text{mean}}(x,y) - G(x,y)\right)$$
     Điểm ảnh được xác định thuộc vùng cháy hồng khi $\min(R, B) > T_{\text{high}}$ (mặc định $T_{\text{high}} = 180$) và $D_{\text{pink}} > \Delta T$ (mặc định $\Delta T = 15$).
     Trọng số nội suy mềm được làm mượt bằng bộ lọc Gaussian $G_{\sigma=2.0}$:
     $$\alpha(x,y) = G_{\sigma=2.0} * \text{clip}\left( \frac{\min(R, B) - T_{\text{high}}}{255 - T_{\text{high}}} \cdot \frac{D_{\text{pink}} - \Delta T}{60}, \; 0, \; 1 \right)$$
   - **Công thức khôi phục kênh màu:**
     Tại các vùng có $\alpha(x,y) > 0$, trung hòa sắc hồng về mức xám sáng tự nhiên $Y_{\text{neutral}}(x,y) = 0.299R + 0.587G + 0.114B$ (hoặc đẩy kênh $G$ lên tiệm cận $RB_{\text{mean}}$ và hạ nhẹ $R, B$):
     $$G'(x,y) = (1 - \alpha(x,y)) \cdot G(x,y) + \alpha(x,y) \cdot RB_{\text{mean}}(x,y)$$
     $$R'(x,y) = (1 - 0.3\alpha(x,y)) \cdot R(x,y) + 0.3\alpha(x,y) \cdot Y_{\text{neutral}}(x,y)$$
     $$B'(x,y) = (1 - 0.3\alpha(x,y)) \cdot B(x,y) + 0.3\alpha(x,y) \cdot Y_{\text{neutral}}(x,y)$$

3. **Bước 1.3: Cân bằng trắng tự động Gray World có loại trừ vùng bão hòa (Masked Gray World AWB)**
   - Chỉ tính trung bình $\mu_R, \mu_G, \mu_B$ trên các điểm ảnh **không bị bão hòa** ($\alpha(x,y) < 0.2$ và $I \in [20, 235]$) để tránh vùng cháy sáng làm sai lệch hệ số cân bằng trắng:
     $$\mu_{\text{gray}} = \frac{\mu_R + \mu_G + \mu_B}{3}, \quad k_c = \text{clip}\left(\frac{\mu_{\text{gray}}}{\mu_c}, 0.8, 1.25\right), \quad c \in \{R, G, B\}$$

---

### Giai đoạn 2: Xử lý chính — Tăng cường tương phản cục bộ trong không gian CIE LAB (`src/stage2_contrast.py` - Thành viên 3)

1. **Bước 2.1: Chuyển đổi không gian màu `RGB` $\rightarrow$ `CIE LAB` (Chương 3)**
   - Tách kênh độ chói $L \in [0, 255]$ (đã chuẩn hóa 8-bit trong OpenCV) khỏi hai kênh sắc độ $a$ (Xanh lá – Đỏ) và $b$ (Xanh dương – Vàng).
   - **Ưu điểm học thuật:** Thao tác kéo giãn lược đồ trên kênh $L$ không làm biến đổi tỷ lệ sắc độ (Hue/Saturation), ngăn ngừa hiện tượng lệch màu thứ cấp thường gặp khi cân bằng Histogram trực tiếp trên 3 kênh RGB.

2. **Bước 2.2: Cân bằng lược đồ thích nghi có giới hạn độ tương phản (CLAHE trên kênh $L$)**
   - Chia kênh $L$ thành lưới $8 \times 8$ ô cục bộ (`tileGridSize = (8, 8)`).
   - Cắt ngưỡng tần suất tại `clipLimit = 2.5` và phân phối đều phần dư cho toàn bộ dải mức xám trước khi tính hàm phân bố tích lũy (CDF) nội suy song tuyến tính (bilinear interpolation) giữa các ô.

3. **Bước 2.3: Hiệu chỉnh Gamma khử sương mờ ($\gamma$-Correction)**
   - Áp dụng biến đổi lũy thừa trên kênh $L_{\text{CLAHE}}$ để tăng độ sâu vùng tối/trung gian, loại bỏ lớp mù sáng (haze):
     $$L_{\text{out}}(x,y) = 255 \times \left( \frac{L_{\text{CLAHE}}(x,y)}{255} \right)^{\gamma}, \quad \gamma = 1.15$$
   - Chuyển ngược `CIE LAB` $(L_{\text{out}}, a, b) \rightarrow$ `RGB`.

---

### Giai đoạn 3: Hậu xử lý — Làm sắc nét biên ảnh (`src/stage3_sharpen.py` - Thành viên 2)

1. **Bước 3.1: Kỹ thuật Unsharp Masking có ngưỡng chống nhiễu (Chương 3 & 4)**
   - **Công thức:**
     $$M_{\text{detail}}(x,y) = I_{\text{CLAHE}}(x,y) - (G_{\sigma} * I_{\text{CLAHE}})(x,y)$$
     $$I_{\text{out}}(x,y) = \text{clip}\left( I_{\text{CLAHE}}(x,y) + k \cdot M_{\text{detail}}(x,y), \; 0, \; 255 \right)$$
   - **Tham số chuẩn:** $\sigma = 1.2$, hệ số khuếch đại biên $k = 1.0$.
   - **Cải tiến nhỏ (Thresholding):** Chỉ cộng $M_{\text{detail}}(x,y)$ tại những vị trí có $|M_{\text{detail}}(x,y)| > T_{\text{sharp}}$ (ví dụ $T_{\text{sharp}} = 3$) để chỉ làm sắc nét biên thật (cỏ, người, mép đường) mà không làm nổi hạt nhiễu ở vùng phẳng.

---

### Các phương pháp đối chứng (`src/baselines.py` - Thành viên 3)

Để bảng kết quả và hình ảnh so sánh trong Mục 4 thuyết phục hội đồng chấm, nhóm cài đặt **2 phương pháp đối chứng (Baselines)** và **Ablation Study (Đánh giá từng bước)**:
1. **Baseline 1 (Global HE - RGB):** Cân bằng Histogram toàn cục độc lập trên 3 kênh $R, G, B$ (minh họa rõ hiện tượng biến dạng màu nặng).
2. **Baseline 2 (Global HE - YCrCb/LAB):** Cân bằng Histogram toàn cục trên kênh độ sáng $Y$ hoặc $L$ (không làm lệch màu thêm nhưng không sửa được lỗi ám hồng và gây cháy sáng cục bộ).
3. **Pipeline từng phần (Ablation):**
   - Chỉ Bước 1 (Bilateral + Khử hồng).
   - Bước 1 + Bước 2 (Khử hồng + CLAHE LAB).
   - Pipeline đầy đủ (Bước 1 + Bước 2 + Bước 3 Unsharp Masking).

---

## 3. Kế hoạch Dữ liệu & Kịch bản Thực nghiệm (`src/degradation_model.py` & `src/metrics.py` - Thành viên 4)

### 3.1. Xây dựng tập dữ liệu giả lập có tham chiếu (Synthetic Reference Dataset: 20–30 ảnh)
Vì ảnh chụp từ ESP32-CAM bay thực tế không có ảnh gốc sạch tương ứng ở cùng tọa độ (No Ground Truth), để tính được **PSNR** và **SSIM**, Thành viên 4 sẽ viết hàm `simulate_esp32cam_degradation(img_gt)` mô phỏng chính xác 4 bước suy giảm vật lý của OV2640:
1. **Nén dải động & phủ sương (Low Contrast & Haze):** Co hẹp histogram về khoảng $[35, 220]$ bằng biến đổi tuyến tính kết hợp $\gamma = 0.85$.
2. **Mô phỏng cháy sáng hồng (Pink Highlight Cast):** Tại các điểm có độ sáng $Y > 185$, tăng kênh $R$ và $B$ thêm $+35 \sim +55$ (clip ở 255) và giảm kênh $G$ đi $-25 \sim -40$.
3. **Làm mờ chuyển động / quang học (Blur):** Nhân chập với Gaussian kernel $\sigma = 1.1$.
4. **Nhiễu nén JPEG:** Mã hóa và giải mã lại ảnh bằng chuẩn JPEG với `quality = 45` (`cv2.imencode('.jpg', ...)`).

### 3.2. Bộ chỉ số đánh giá định lượng (Evaluation Metrics)

| Nhóm dữ liệu | Chỉ số (Metric) | Công thức / Ý nghĩa | Kỳ vọng kết quả của Pipeline |
| :--- | :--- | :--- | :--- |
| **Tập Synthetic (20–30 ảnh có GT)** | **PSNR (dB)** $\uparrow$ | $10 \log_{10} \left(\frac{255^2}{\text{MSE}}\right)$ — Đo sai số tái tạo màu sắc & cường độ sáng so với ảnh chuẩn. | Cao hơn ảnh suy giảm và Global HE từ $+3\text{ dB}$ đến $+6\text{ dB}$. |
| | **SSIM** $\uparrow$ | Độ tương đồng cấu trúc $[0, 1]$ giữa ảnh khôi phục và GT. | Đạt $\ge 0.82$, vượt trội so với Global HE. |
| | **$\Delta E_{ab}$ (CIE LAB Color Error)** $\downarrow$ | Sai lệch màu sắc trung bình trong không gian LAB (đặc biệt tại vùng cháy sáng). | Giảm $> 50\%$ độ lệch màu hồng so với ảnh đầu vào. |
| **Tập Real UAV ($\ge 50$ ảnh thực tế)** | **Shannon Entropy ($H$)** $\uparrow$ | $H = -\sum_{i=0}^{255} p_i \log_2(p_i)$ — Lượng thông tin và độ phong phú chi tiết mức xám. | Tăng từ $\sim 6.5\text{ bits}$ lên $\sim 7.3 - 7.6\text{ bits}$. |
| | **Laplacian Variance** $\uparrow$ | $\text{Var}(\nabla^2 I)$ — Độ sắc nét của biên ảnh (càng cao biên càng rõ). | Tăng rõ rệt sau bước Unsharp Masking (gấp $1.5 - 2.5$ lần ảnh gốc). |
| | **Pink Cast Index (PCI)** $\downarrow$ | Tỷ lệ $\%$ số điểm ảnh bị ám hồng ($\min(R,B)>180$ và $\frac{R+B}{2} - G > 15$). | Giảm từ $5\% - 20\%$ (ảnh gốc) về $\approx 0.0\% - 0.5\%$. |
| | **Thời gian xử lý (ms/frame & FPS)** | Thời gian chạy trung bình trên CPU cho ảnh $640\times480$. | $< 25\text{ ms/frame}$ ($\ge 40\text{ FPS}$), dư sức đáp ứng tốc độ $2.5 - 6.5\text{ FPS}$ của ESP32-CAM. |

---

## 4. Lộ trình Thực hiện Chi tiết (Timeline 7 Ngày / 4 Giai đoạn)

### Giai đoạn A: Khởi tạo & Chuẩn bị dữ liệu (Ngày 1 - Ngày 2)
- **Thành viên 1:** Khởi tạo Git repo / cấu trúc thư mục, thống nhất file `src/config.py` và interface hàm giữa các thành viên (nhận ảnh `np.ndarray` chuẩn `BGR` hoặc `RGB` kiểu `uint8`, trả về `uint8`).
- **Thành viên 2:** Trích xuất $\ge 50$ khung hình thực tế từ video bay UAV (`data/real_uav/`), chọn ra 5 ảnh tiêu biểu nhất (có cả góc cao 2.5 fps và góc thấp 6.5 fps, có mặt đường cháy hồng, bãi cỏ, người) đưa vào `data/samples/`.
- **Thành viên 4:** Thu thập 20–30 ảnh ngoài trời chuẩn (`data/synthetic_gt/`) và viết xong script `generate_synthetic.py`.

### Giai đoạn B: Lập trình Core Pipeline & Baselines (Ngày 3 - Ngày 4)
- **Thành viên 1:** Code và tinh chỉnh `src/stage1_preprocess.py` (Bilateral + Soft Pink Highlight Recovery + Masked Gray World). Kiểm tra trực quan mặt nạ $\alpha(x,y)$ trên 5 ảnh mẫu.
- **Thành viên 3:** Code `src/stage2_contrast.py` (LAB CLAHE + Gamma) và `src/baselines.py` (Global HE). Viết hàm xuất biểu đồ Histogram RGB trước/sau.
- **Thành viên 2:** Code `src/stage3_sharpen.py` (Unsharp Masking) và tích hợp vào `src/pipeline.py` cùng Thành viên 1.
- **Thành viên 4:** Code `src/metrics.py` đầy đủ các hàm tính PSNR, SSIM, Entropy, Laplacian Variance, Pink Cast Index và bộ đếm thời gian `time.perf_counter()`.

### Giai đoạn C: Chạy thực nghiệm & Tinh chỉnh tham số (Ngày 5)
- **Cả nhóm (TV4 chủ trì chạy script `run_experiments.py`):**
  - Chạy toàn bộ 30 ảnh Synthetic $\rightarrow$ Xuất **Bảng 1** (`results/tables/metrics_synthetic.csv`).
  - Chạy toàn bộ $\ge 50$ ảnh Real UAV $\rightarrow$ Xuất **Bảng 2** (`results/tables/metrics_real.csv`).
  - **TV3:** Xuất bộ ảnh minh họa chất lượng cao (300 DPI) cho báo cáo:
    1. Hình minh họa sơ đồ khối Pipeline tổng thể.
    2. Hình mặt nạ phát hiện vùng cháy sáng hồng $\alpha(x,y)$ và kết quả trước/sau khử hồng.
    3. Hình so sánh 4 cột trên 3–4 cảnh UAV khác nhau: `(a) Ảnh gốc ESP32-CAM` | `(b) Đối chứng Global HE` | `(c) Khử hồng + CLAHE` | `(d) Pipeline hoàn chỉnh`.
    4. Hình so sánh biểu đồ Histogram 3 kênh R, G, B trước và sau khi qua Pipeline.

### Giai đoạn D: Hoàn thiện Báo cáo, Slide & Video Demo (Ngày 6 - Ngày 7)
- **Thành viên 2:** Chạy `scripts/demo_video.py` ghép song song (Side-by-Side: Trái = Gốc ESP32-CAM, Phải = Đã xử lý + hiển thị FPS realtime) và quay/dựng video demo $\le 3$ phút.
- **Thành viên 4:** Điền số liệu thực nghiệm vào Báo cáo (4–10 trang) và thiết kế Slide thuyết trình (10–15 trang) bám sát 5 mục bắt buộc của thầy Phạm Văn Cường.
- **Thành viên 1:** Viết `README.md`, rà soát toàn bộ code chạy trơn tru chỉ với 1 câu lệnh, kiểm duyệt nội dung báo cáo và phần Khai báo sử dụng AI.

---

## 5. Bố cục Chi tiết Báo cáo (4–10 trang) & Slide Thuyết trình (12 Slides)

### 5.1. Cấu trúc Báo cáo chuẩn (6–8 trang là đẹp nhất)
1. **Trang 1: Thông tin chung & Mục 1 - Định nghĩa Bài toán (Problem Definition)**
   - Bối cảnh hệ thống UAV gắn ESP32-CAM (OV2640) truyền ảnh Wi-Fi $2.5 - 6.5\text{ fps}$.
   - Phân tích 3 suy giảm vật lý (Cháy sáng hồng, Mờ sương tương phản thấp, Nhòe rung động & nhiễu khối JPEG) kèm hình ảnh phóng to vùng lỗi thực tế.
2. **Trang 2: Mục 2 - Tập dữ liệu (Dataset Description)**
   - Bảng thông số kỹ thuật 2 tập dữ liệu: Real UAV Dataset ($\ge 50$ ảnh) và Synthetic Reference Dataset ($20-30$ ảnh).
   - Mô tả mô hình suy giảm giả lập phục vụ đánh giá có tham chiếu.
3. **Trang 3–4: Mục 3 - Phương pháp Đề xuất (Proposed Pipeline)**
   - Sơ đồ khối kiến trúc 3 giai đoạn.
   - Trình bày công thức toán học và lý giải lựa chọn tham số cho từng bước (liên hệ trực tiếp kiến thức Chương 2, Chương 3, Chương 4).
4. **Trang 5–6: Mục 4 - Kết quả Thực nghiệm (Experimental Results)**
   - **4.1. Kết quả định tính (Qualitative):** Lưới ảnh so sánh 4 cột, ảnh phóng to vùng mặt đường/người (Zoom-in patches), và biểu đồ Histogram RGB trước/sau.
   - **4.2. Kết quả định lượng (Quantitative):** Bảng 1 (PSNR, SSIM, $\Delta E$ trên tập Synthetic) và Bảng 2 (Entropy, Laplacian Variance, Pink Cast Index, Thời gian xử lý ms/frame trên 50 ảnh UAV thật).
5. **Trang 7–8: Mục 5 - Thảo luận, Phân công & Khai báo sử dụng AI**
   - Phân tích ưu điểm (tốc độ CPU thời gian thực, khử triệt để lỗi hồng) và hạn chế (vùng bão hòa trắng toàn phần $R=G=B=255$).
   - Bảng phân công nhiệm vụ 4 thành viên (đóng góp 25% mỗi người).
   - Khung **Khai báo sử dụng AI (AI Declaration)** đúng quy định Trang 8 slide hướng dẫn.

### 5.2. Cấu trúc Slide Thuyết trình (12 Slides)
- **Slide 1:** Trang bìa (Tên đề tài, Giảng viên hướng dẫn, Danh sách 4 thành viên & MSSV).
- **Slide 2 [Mục 1]:** Bối cảnh thực tế — Hệ thống UAV tự chế & Giới hạn phần cứng camera giá rẻ ESP32-CAM (OV2640).
- **Slide 3 [Mục 1]:** Ba suy giảm chất lượng ảnh cốt lõi (Minh họa trực quan bằng ảnh chụp thực tế + Histogram bị dồn kênh $R, B$).
- **Slide 4 [Mục 2]:** Tập dữ liệu thực nghiệm (50+ ảnh bay thực tế đa góc chụp + 30 ảnh giả lập có Ground Truth).
- **Slide 5 [Mục 3]:** Tổng quan Kiến trúc Pipeline 3 Giai đoạn (Sơ đồ khối trực quan).
- **Slide 6 [Mục 3.1]:** Giai đoạn 1 — Lọc nhiễu Bilateral & Thuật toán Nội suy mềm khử vùng cháy sáng hồng (Kèm ảnh mặt nạ $\alpha(x,y)$).
- **Slide 7 [Mục 3.2 & 3.3]:** Giai đoạn 2 & 3 — Tăng cường tương phản cục bộ CLAHE trên không gian CIE LAB & Làm sắc nét biên Unsharp Masking.
- **Slide 8 [Mục 4.1]:** Kết quả trực quan 1 — So sánh 4 cột trên các cảnh bay thực tế (Gốc vs Global HE vs CLAHE vs Full Pipeline).
- **Slide 9 [Mục 4.1]:** Kết quả trực quan 2 — Phân tích biểu đồ Histogram RGB & Vùng phóng to chi tiết (Zoom-in).
- **Slide 10 [Mục 4.2]:** Kết quả định lượng — Bảng số liệu PSNR, SSIM, Entropy, Laplacian Variance & Tốc độ thực thi ($\text{FPS}$).
- **Slide 11 [Mục 5]:** Thảo luận (Ưu điểm, Hạn chế khi cháy sáng 3 kênh) & Hướng phát triển.
- **Slide 12 [Mục 5]:** Bảng phân công nhiệm vụ (4 thành viên) & Khai báo sử dụng AI (AI Declaration).
