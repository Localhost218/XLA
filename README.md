# XLA: ảnh chất lượng thấp từ camera drone

Đồ án Xử lý ảnh (PTIT). Camera ESP32-S3 + OV3660 trên drone S500 phát JPEG 640×480 q12 qua Wi-Fi; ảnh ám hồng, mù, vỡ
khối, tối góc. Repo này nghiên cứu hai việc trên **khung bay thật**: phát hiện người và làm rõ ảnh cho người xem. Model xuất ra
ONNX và chạy trong phần mềm drone (repo IOT-CV, `backend/vision/`).

Bản trước của nhóm viết cho camera ESP32-CAM OV2640, ghi nhầm loại camera; báo cáo và mã đó giữ ở `archive/v2-nhom/`.

Báo cáo: [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md). Slide: mở `docs/slide/index.html` trong trình duyệt (phím ← →).

## Kết quả chính (198 khung test thật, chuyến 02/11/2026)

| | Kết quả |
|---|---|
| Nhận diện, mốc YOLO11n COCO | mAP50 0,126 |
| E1, YOLO11n học trên ảnh làm xấu theo camera (D2) | mAP50 0,236; báo nhầm 0,30 hộp/khung bay không người (D1: 0,99) |
| Bundle ship `detector-v1.0.0` (ONNX) | mAP50 0,120; 23,6 ms/khung trung vị trong IOT-CV (RTX 4060 Laptop) |
| Làm rõ: cổ điển vs mạng `enhancer-v1.0.0` | người chấm mù chọn cổ điển 24/25 lượt; mạng không bật mặc định |
| Làm rõ trước khi nhận diện (E4) | bộ đã tinh chỉnh giảm mAP50 0,085 [−0,139; −0,011] → nhận diện luôn chạy trên khung gốc |

Bảng đầy đủ kèm CI 95 %: `results/reproduce/tables/`. CI GitHub chạy `pytest -m "not gpu and not data"` trên mọi push.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `xla/camera/` | trích khung bay, đo hồ sơ camera, chia tập theo khối thời gian |
| `xla/degrade/` | bộ làm xấu D1 (ngẫu nhiên), D2 (bản sao camera), hiệu chỉnh, so độ giống thật |
| `xla/detect/` | mốc, huấn luyện E1, tự huấn luyện E2, xuất ONNX |
| `xla/enhance/` | bộ lọc cổ điển, mạng NAFNet, xuất ONNX |
| `xla/eval/` | chấm nhận diện (bootstrap theo khối), C2ST, KID, A/B, E4 |
| `xla/contract.py` | kiểm bundle theo hợp đồng với IOT-CV |
| `models/release/` | `detector-v1.0.0`, `enhancer-v1.0.0` (ONNX + `meta.json` + golden) |
| `results/` | mọi kết quả đo; `results/reproduce/` là bảng/hình của báo cáo |
| `src/`, `web/`, `notebooks/` | mã của bản trước, giữ để tham khảo |

## Cài đặt (Windows, mọi thứ ở ổ D)

```powershell
. .\scripts\env.ps1          # đặt cache uv/pip/torch/HF và venv ở D:\DevCache
uv sync --locked
uv run --no-sync pytest -m "not gpu and not data"
```

Dữ liệu ảnh nguồn (VisDrone, NOMAD) không nằm trong repo; đường dẫn ở `configs/paths.yaml`.

## Tái lập

```powershell
uv run --no-sync python scripts/reproduce.py --check --json   # CPU: chấm lại 15 tệp dự đoán + so sha256 9 bảng/hình
uv run --no-sync python scripts/reproduce.py --write          # dựng lại bảng/hình, chép bảng vào báo cáo, ghi manifest
uv run --no-sync python scripts/reproduce.py --full --dry-run # các bước GPU (huấn luyện lại), khoảng 6,6 giờ GPU
```

## Hợp đồng với IOT-CV

`docs/hop-dong-model.md` (v1.0). Detector: input `frame` uint8 [480, 640, 3] BGR, output `boxes` [N, 4] + `scores` [N].
Enhancer: output `frame_out` uint8 [480, 640, 3]. IOT-CV kiểm sha256, phiên bản hợp đồng, tên và kiểu I/O rồi mới nạp.

Tải bundle từ GitHub Release **`models-v1.0.0`**:

```powershell
gh release download models-v1.0.0 --repo Localhost218/XLA --dir D:\DevCache\tmp\models
Expand-Archive D:\DevCache\tmp\models\detector-v1.0.0.zip D:\DevCache\tmp\models\detector-v1.0.0
uv run --no-sync python -m xla.contract validate D:\DevCache\tmp\models\detector-v1.0.0   # in []
```

`enhancer-v1.0.0` đúng hợp đồng nhưng không đạt A/B, nên IOT-CV để "Ảnh rõ" tắt mặc định.

## Giấy phép dữ liệu và mã

- VisDrone: chỉ dùng nghiên cứu; ảnh không được đăng lại trong repo.
- NOMAD: CC BY 4.0, ghi công tác giả bộ dữ liệu.
- `pyiqa` (MUSIQ, NIQE): PolyForm Noncommercial 1.0.0; chỉ dùng mô tả trong đồ án.
- Ultralytics YOLO11: AGPL-3.0. NAFNet: MIT. DINOv2: Apache-2.0.
- Khung bay trong `data/real/` là của nhóm.
