# Độ giống thật của ảnh tổng hợp (`results/realism/`)

Phase 06 viết phép đo, phase 07 dùng nó để hiệu chỉnh D2 và ghi bảng D1/D2 vào file này (thêm mục, không ghi đè mục dưới).

## Phép đo (phase 06)

`python -m xla.eval.realism --real <dir|split:dev> --synth <dir> --json [--out <file>]`, exit 0 đo xong, 1 lỗi đầu vào,
2 hai tập khác quy trình đọc (khác cỡ ảnh, ảnh không giải mã được). Ba tầng, xếp theo độ tin:

| Tầng | Cách tính | "Giống" là |
|---|---|---|
| Thống kê mức thấp | W1 giữa phân bố từng khoá `xla.camera.measure.measure_image` + từng bin `nlf`, `psd_radial` (log10), chia IQR tập thật; tổng = trung bình có trọng số (`--weights`, mặc định bằng nhau). DQT + lấy mẫu màu so khớp chính xác, báo riêng | gần nền nhiễu bên dưới |
| C2ST | CNN 60 929 tham số trên phần dư `ảnh − Gauss σ=2` của patch 64×64; chia train/val/test theo ảnh nguồn; CI 95 % bootstrap theo cụm ảnh | gần nền nhiễu (không phải 0,5, xem dưới) |
| KID-DINOv2 | DINOv2 ViT-L/14 token CLS trên patch 224×224 (không resize), MMD² không chệch nhân đa thức bậc 3, 10 tập con rời nhau, CI 95 % bootstrap | gần nền nhiễu |

Hai tập đi qua cùng `xla.eval.patches`: `cv2.imdecode(IMREAD_COLOR)`, bỏ cột `[0, 16)`, cắt patch có seed, không resize.
Đặc trưng KID cache theo sha256 ảnh ở `<cache_dir>/realism/` (mặc định `D:\DevCache\cache\realism\`); trích trong `xla.gpu_lock`.

### Bộ mã hoá KID: thay `dgm-eval` bằng `torch.hub`

`dgm-eval` (layer6ai-labs, commit 74c79f7) không cài được vào venv chung: nó ghim `pillow==9.2.0`, `numpy==1.23.3`,
`timm==0.8.19.dev0`, `transformers==4.26.0`, `xformers`, trái `pillow>=12` mà plan cần (ghi ở `D:\DevCache\venvs\xla.READY`).
Owner đã duyệt phương án thay (`plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-01-07.md` §2): cùng model
`dinov2_vitl14` nạp qua `torch.hub` (`TORCH_HOME=D:\DevCache\torch`, trọng số sha256 `d5383ea8…cbf428`), cùng tiền xử lý
của `dgm_eval/models/dinov2.py` (PIL RGB → resize 224×224 bicubic → /255 → chuẩn hoá ImageNet), KID tự viết. Nếu sau này
`dgm_eval` import được, `xla.eval.kid.Dinov2Features(backend="auto")` tự dùng encoder của nó.

### Smoke 18/11/2026 (`smoke.json`, RTX 4060 Laptop)

D2 `--limit 50` của phase 05 chưa có nên cặp chính là dev thật vs **D0 tạm** (ảnh VisDrone sạch, cắt về 640×480, không làm
xấu) — cận trên của khoảng cách, không phải số của D2. Cặp thứ hai là **nền nhiễu**: hai nửa dev cùng camera chia theo block 30 s.

| Cặp | Thống kê (W1/IQR) | DQT khớp | C2ST acc [CI95] | KID ± SE | Giây (thống kê / C2ST / KID) | VRAM đỉnh MB (C2ST / KID) |
|---|---|---|---|---|---|---|
| dev (470) vs D0 tạm (470) | 8,29 | không (PNG) | 0,974 [0,958; 0,986] | 7,55 ± 0,15 | 75 / 23 / 85 | 368 / 1 549 |
| dev nửa B (323) vs nửa A (147) | 0,84 | có | 0,767 [0,720; 0,807] | 1,33 ± 0,09 | 28 / 4 / 91 | 368 / 1 549 |

KID chạy lại có cache: 10,5 giây (so với 85), cùng giá trị.

**Nền nhiễu không về 0.** Hai nửa dev cùng camera vẫn cho C2ST 0,77 và KID 1,33: hai nửa khác cảnh và thời điểm bay, phần dư
cao tần vẫn mang dấu nội dung (độ cao, mặt đất). Phase 07 nên coi D2 "giống thật" khi các số rơi vào vùng nền này, không đặt
mục tiêu C2ST 0,5 hay KID 0.
