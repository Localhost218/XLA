# Thực thi phase 01–07 (phiên 18/11/2026, chủ dự án ngủ, agent tự quyết)

Owner (phiên chính) chốt các điểm dưới đây để các lane chạy **song song ngay**, không chờ nhau. Đây là phần bổ sung cho
`plan.md` và `contract.md`; chỗ nào trái nhau thì `contract.md` thắng, rồi tới file này, rồi tới `plan.md`.

## 1. Worktree và nhánh (D12)

| Lane | Phase | Thư mục | Nhánh |
|---|---|---|---|
| serial | 01, 02 | `D:\XLA` | `feat/ai-anh-chat-luong-thap` |
| data | 03, 04 | `D:\XLA-wt\data` | `lane/data` |
| degrade | 05, 07 | `D:\XLA-wt\degrade` | `lane/degrade` |
| realism | 06 | `D:\XLA-wt\realism` | `lane/realism` |

Lane chỉ commit trên nhánh của mình (pathspec: `git add <đường dẫn cụ thể>` cho file mới rồi `git commit -m ... -- <đường dẫn>`),
đẩy `git push -u origin lane/<id>`. Không `git switch`/`checkout -b` trong worktree, không mở PR. Owner gộp lane vào nhánh tích hợp.
Khi nhánh tích hợp có commit mới cần dùng: `git fetch origin && git merge origin/feat/ai-anh-chat-luong-thap` (không rebase
nhánh đã đẩy). Lane **không** sửa file do phase 01 sở hữu: `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`,
`tests/conftest.py`, `xla/__init__.py`, `xla/*/__init__.py`, `xla/paths.py`, `xla/gpu_lock.py`, `configs/paths.yaml`.

## 2. Môi trường Python (D8)

- Venv dùng chung: `D:\DevCache\venvs\xla` (Python 3.12). **Chỉ lane serial (phase 01) được chạy `uv lock`/`uv sync`/`uv add`.**
  Lane khác gọi thẳng `D:\DevCache\venvs\xla\Scripts\python.exe` (hoặc `uv run --no-sync`), không bao giờ để uv tự sync từ
  worktree của mình (sẽ đè venv chung).
- Venv sẵn sàng khi có file `D:\DevCache\venvs\xla.READY` (phase 01 ghi sau khi bước 4 kiểm GPU đạt). Trước đó lane viết code,
  chưa chạy được test cần scipy/torch thì để sau.
- Owner duyệt trước danh sách gói của phase 01 (thêm vào danh sách gốc): `open_clip_torch` (CLIP zero-shot lọc nadir, phase 05),
  `cma` (CMA-ES, phase 07), `scikit-learn`, `timm`, `tqdm`, `requests`, `huggingface_hub`.
- `dgm-eval`: thử `git+https://github.com/layer6ai-labs/dgm-eval`. Không cài được thì Owner **đã duyệt** phương án thay của
  phase 06: DINOv2 ViT-L/14 qua `torch.hub` (`facebookresearch/dinov2`, cache `TORCH_HOME=D:\DevCache\torch`), tiền xử lý
  giống dgm-eval (resize 224 bicubic, chuẩn hoá ImageNet), KID tự viết (đa thức bậc 3, ước lượng không chệch). Ghi rõ trong
  `results/realism/README.md` là đã thay và vì sao.
- Biến môi trường: dot-source `D:\XLA\scripts\env.ps1` (phase 01 tạo). Trước khi có: tự đặt `TORCH_HOME`, `HF_HOME`,
  `YOLO_CONFIG_DIR` về `D:\DevCache\...` như phase 01 bước 2. Không tải gì về ổ C.
- Lệnh có mạng (git push/fetch, gh, tải dataset) chạy ngoài sandbox.

## 3. API phase 01 tạo, các lane dùng ngay

```python
# xla/paths.py — đọc configs/paths.yaml; biến môi trường cùng tên (viết hoa, tiền tố XLA_) ghi đè
repo_root() -> Path            # gốc repo (thư mục chứa pyproject.toml của worktree đang chạy)
data_root() -> Path            # XLA_DATA_ROOT, mặc định D:/DevCache/datasets/xla (tập sinh D0/D1/D2)
src_datasets_dir() -> Path     # D:/DevCache/datasets/src (dataset nguồn đã giải nén)
downloads_dir() -> Path        # D:/IOT_Tools/downloads
cache_dir() -> Path            # D:/DevCache/cache
iotcv_root() -> Path           # d:/Coding/IOT-CV
gpu_lock_file() -> Path        # D:/DevCache/locks/gpu.lock

# xla/gpu_lock.py
@contextmanager
def gpu_lock(label: str = "", timeout_s: float | None = None, poll_s: float = 2.0): ...
# chờ khoá; quá timeout_s -> TimeoutError; ghi pid + label + giờ vào file khoá; nhả kể cả khi lỗi
```

## 4. API `xla/camera/measure.py` (phase 04 viết, phase 05/06/07 dùng)

```python
CROP_LEFT_DEFAULT = 16
def measure_image(img_bgr: np.ndarray, crop_left: int = 16) -> dict[str, float]
#   khoá (đại lượng một số / ảnh, bỏ cột [0, crop_left) trừ left_strip_width_px; NaN khi không xác định):
#   dark_channel_15px, y_p1, y_p99, saturation_mean, pink_frac, pink_highlight_y, blocking_ratio_8x8,
#   laplacian_var, sigma_immerkaer, sobel_x2_over_y2, vignette_corner_center_ratio, r_over_g, b_over_g,
#   mtf50_cyc_per_px, left_strip_width_px, left_strip_mean_y
def nlf(img_bgr, crop_left=16, n_bins=8) -> list[tuple[float, float]]      # (mức Y 0–255, σ 0–255) vùng phẳng
def psd_radial(img_bgr, crop_left=16, n_bins=16) -> list[tuple[float, float]]  # (tần số chu kỳ/px, công suất) phần dư cao tần
def read_jpeg_header(data: bytes) -> dict  # {"qtable_luma": [64 int, thứ tự tự nhiên], "qtable_chroma": [64], "subsampling": "4:2:2", "baseline": bool}
```

Phase 06 tính khoảng cách thống kê trên đúng các khoá của `measure_image` (+ đường `nlf`, `psd_radial`), không viết lại phép đo.

## 5. Dữ liệu dùng chéo giữa lane trước khi Owner gộp

- Lane đọc (chỉ đọc) sản phẩm của lane khác qua đường dẫn worktree tuyệt đối, ví dụ `D:\XLA-wt\data\data\real\split.json`,
  `D:\XLA-wt\data\configs\camera_profile.yaml`, để chạy smoke sớm; code không gõ cứng đường dẫn đó (truyền qua tham số CLI).
- Gộp chính thức: Owner gộp `lane/data` vào nhánh tích hợp ngay khi phase 04 xong, rồi báo lane degrade/realism `git merge`.

## 6. Quyết định Owner đã chốt thay chủ dự án (để không dừng giữa chừng)

- Mọi "dừng, báo Owner" trong phase 01–07 → báo **phiên chính** (Owner) trong báo cáo cuối của agent; Owner quyết, không chờ người.
- Chọn phương án đầu tiên được khuyến nghị trong plan khi có nhiều lựa chọn; ghi lại trong báo cáo cuối của lane.
- Tải dataset > 2 phút: chạy thử 1 file, ghi cỡ + thời gian ước vào báo cáo rồi **cứ tải tiếp** (Owner duyệt trước).
- Ổ D còn ~57 GB lúc bắt đầu: giữ tối thiểu 10 GB trống; NOMAD giải nén xong thì được xoá file zip nếu ổ D < 15 GB (ghi sha256 trước).

## 7. Trạng thái khi tạm dừng (18/11/2026, chủ dự án tắt máy)

Phase 01–09 **xong**, tất cả nằm trên nhánh tích hợp `feat/ai-anh-chat-luong-thap` (đã đẩy; toàn bộ test 168 passed,
ruff sạch). PR #1 (phase 01+02) đã merge vào `main`; PR #2 theo D12 là sau phase 14. Worktree lane còn ở `D:\XLA-wt\`
(data, degrade, realism, labels, baseline) — đều đã gộp, sạch; giữ lại để tiếp tục, xoá khi không cần.

Số chính:
- Split: dev 1 115 / test 198 / neg 300 (seed 20261118). Nhãn test chủ dự án duyệt: 39 khung có người, 149 hộp; dev-val
  100 khung, 115 hộp (chỉ agent soát; chủ dự án sửa 25,8 % hộp test của agent, chủ yếu ranh giới).
- Hồ sơ camera v1.2.0. D2 `params_version` 2 (CMA-ES 155 lượt, khoảng cách 0,967 → 0,736). Tập D0/D1/D2/D2+RealSR
  ở `D:\DevCache\datasets\xla\` (3 493 ảnh/tập; train 3 006, val 487).
- Độ giống thật trên khung test: thống kê D1 5,93 / D2 0,81 / D2+RealSR 0,66 / thật–thật 0,33; KID D1 4,44 / D2 4,37
  (CI chồng D1) / D2+RealSR 3,93 (CI không chồng) / thật–thật 0,37.
- Mốc `edge16` (YOLO11s COCO, bỏ hộp chạm mép): mAP50 0,126 [0,030–0,224], 23,3 ms p95 lúc máy rảnh. Tiêu chí nhánh B
  sau M2: `results/baseline/tieu-chi-sau-m2.md` (B1 ≥ 0,227; B2 ≤ 0,05 hộp/khung âm ở ngưỡng dev-val; B3 ≤ 30 ms p95).

**Việc tiếp theo khi bật máy (theo thứ tự, trước khi vào phase 10):**
1. **Hiệu chỉnh D2 vòng 2** (lane degrade): ba tham số dừng sát biên khoảng tìm (vignette focal 0,618/biên 0,6,
   `cast_r` 0,960/biên, hệ số WB ISP). Nới biên, khởi đầu từ `params_version` 2, ngân sách ~80 lượt; **chỉ nhận** nếu
   khoảng cách trên dev giảm, khi đó `params_version` 3, sinh lại D2 + D2+RealSR, đo lại `d1_vs_d2`. ~1–1,6 giờ CPU.
2. **So hiệu ghép cặp** trong `xla/eval/detection.py`: cùng mẫu khối bootstrap cho hai file dự đoán, CI của hiệu mAP50
   (và báo nhầm/khung âm); thêm báo riêng 159 khung bay không người cho B2. Phase 10 cần lệnh này để kết luận B1.
3. Label Studio không tự chạy lại sau khi tắt máy (nhãn đã xuất, không cần nữa); mở lại nếu cần:
   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\XLA-wt\labels\scripts\labelstudio_start.ps1`.
4. Rồi phase 10 (E1 fine-tune D0/D1/D2, nên thêm D2+RealSR vì nó thắng KID) và các phase song song được: 12 (cổ điển,
   CPU), 13 (mạng khôi phục), 15 + 17 (IOT-CV với bundle giả; nhớ bẫy ORT CUDA EP ở phase 15).
