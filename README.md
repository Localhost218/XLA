# XLA: ảnh chất lượng thấp từ camera drone

Đồ án Xử lý ảnh (PTIT). Camera ESP32-S3 + OV3660 trên drone S500 phát JPEG 640×480 q12 qua Wi-Fi; ảnh ám hồng, mù, vỡ
khối, tối góc. Repo này nghiên cứu hai việc trên **khung bay thật**: phát hiện người và làm rõ ảnh cho người xem. Model xuất ra
ONNX và chạy trong phần mềm drone (repo IOT-CV, `backend/vision/`).

Bản trước của nhóm viết cho camera ESP32-CAM OV2640, ghi nhầm loại camera; báo cáo và mã đó giữ ở `archive/v2-nhom/`.

Báo cáo: [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md). Slide: mở `docs/slide/index.html` trong trình duyệt (phím ← →).
Báo cáo ngắn (PDF) và slide nộp bài chỉ lưu trên Google Drive của nhóm, không đưa vào repo.

## Kết quả chính (198 khung test thật, chuyến 02/11/2026)

| | Kết quả |
|---|---|
| Nhận diện, mốc YOLO11s COCO | mAP50 0,126 |
| E1, YOLO11s học trên ảnh làm xấu theo camera (D2) | mAP50 0,236; báo nhầm 0,30 hộp/khung bay không người (D1: 0,99) |
| Bundle ship `detector-v1.0.0` (ONNX) | mAP50 0,120; 23,6 ms/khung trung vị trong IOT-CV (RTX 4060 Laptop) |
| Làm rõ: cổ điển vs mạng `enhancer-v1.0.0` | người chấm mù chọn cổ điển 24/25 lượt; mạng không bật mặc định |
| Làm rõ trước khi nhận diện (E4) | bộ đã tinh chỉnh giảm mAP50 0,085 [−0,139; −0,011] → nhận diện luôn chạy trên khung gốc |

Bảng đầy đủ kèm CI 95 %: `results/reproduce/tables/`. Detector là YOLO11s (`configs/train_e1.yaml`), thầy E2 là YOLO11m (`configs/selftrain_e2.yaml`). CI GitHub chạy `pytest -m "not gpu and not data"` trên mọi push.

Ngân sách thời gian trong IOT-CV (p95 tổng ≤ 50 ms, detector ≤ 30 ms) **không đạt ở nhịp camera thật** trên laptop này: GPU hạ
xung giữa hai khung, tổng p95 54,1 ms (lần đo tốt nhất). Chạy hết tốc lực thì đạt (44,9 ms, 23,6 khung/s). Nhóm chấp nhận giới
hạn này; chi tiết ở báo cáo mục 4.5.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `xla/camera/` | trích khung bay, đo hồ sơ camera, chia tập theo khối thời gian |
| `xla/degrade/` | bộ làm xấu D1 (ngẫu nhiên), D2 (bản sao camera), hiệu chỉnh, so độ giống thật |
| `xla/detect/` | mốc, huấn luyện E1 (YOLO11s), tự huấn luyện E2 (thầy YOLO11m, trò YOLO11s), xuất ONNX |
| `xla/enhance/` | bộ lọc cổ điển, mạng NAFNet, xuất ONNX |
| `xla/eval/` | chấm nhận diện (bootstrap theo khối), C2ST, KID, A/B, E4 |
| `xla/contract.py` | kiểm bundle theo hợp đồng với IOT-CV |
| `models/release/` | `meta.json` + golden của `detector-v1.0.0`, `enhancer-v1.0.0`; file `.onnx` lấy từ GitHub Release |
| `results/` | mọi kết quả đo; `results/reproduce/` là bảng/hình của báo cáo |
| `configs/` | đường dẫn dùng chung (`paths.yaml`), hồ sơ camera, cấu hình huấn luyện |
| `tests/` | pytest; đánh dấu `gpu` (cần CUDA) và `data` (cần dữ liệu local) |
| `src/`, `web/`, `notebooks/` | mã của bản trước, giữ để tham khảo |

## Yêu cầu hệ thống

| Thứ | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 11 (máy nhóm) hoặc Linux x86_64 (CI). `uv.lock` chỉ giải cho hai môi trường này |
| Python | 3.12 (`requires-python = ">=3.12,<3.13"`); uv tự tải nếu máy chưa có |
| uv | bản mới; đã dùng 0.13.8 |
| GPU | không bắt buộc cho test và tái lập CPU. Huấn luyện, xuất ONNX, test `gpu`: GPU NVIDIA + driver hỗ trợ CUDA 12.4 (máy nhóm: RTX 4060 Laptop 8 GB) |
| CUDA | không cần cài CUDA Toolkit riêng: torch 2.6.0+cu124 mang sẵn DLL CUDA 12.4 và cuDNN 9 |
| Ổ đĩa | venv khoảng vài GB (torch CUDA); mọi cache để ở ổ D |

Gói GPU chọn theo hệ điều hành, đã ghim trong `pyproject.toml`:

| | Windows | Linux (CI) |
|---|---|---|
| torch, torchvision | 2.6.0+cu124, 0.21.0+cu124 (index `download.pytorch.org/whl/cu124`) | 2.6.0+cpu, 0.21.0+cpu |
| ONNX Runtime | `onnxruntime-gpu==1.24.4` | `onnxruntime` (CPU) |

`onnxruntime-gpu` 1.24.4 chứ không mới hơn: 1.32.0 đòi CUDA 13 (`cublasLt64_13.dll`). Bản 1.24.4 chạy CUDA EP bằng DLL
CUDA 12.4/cuDNN 9 của torch, nên phải `import torch` trước khi tạo session.

## Cài đặt

Đường chính là uv, đọc đúng `uv.lock`:

```powershell
cd D:\XLA
. .\scripts\env.ps1          # đặt cache uv/pip/torch/HF, venv và file tạm ở D:\DevCache
uv sync --locked             # cài đúng uv.lock vào D:\DevCache\venvs\xla
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Trên Windows có GPU dòng cuối in `2.6.0+cu124 True`; trên Linux CI in `2.6.0+cpu False`.

Đường dự phòng là pip với `requirements.txt` (Python 3.12):

```powershell
py -3.12 -m venv D:\DevCache\venvs\xla-pip
D:\DevCache\venvs\xla-pip\Scripts\python -m pip install -r requirements.txt
D:\DevCache\venvs\xla-pip\Scripts\python -m pip install --no-deps -e .
```

`requirements.txt` sinh từ `uv.lock`, không sửa tay. Một file dùng cho cả Windows lẫn Linux: mỗi dòng có điều kiện
`sys_platform`, nên pip tự chọn torch cu124 hay cpu. File có dòng `--extra-index-url` tới index PyTorch, vì PyPI không có bản
`+cu124`. Cài bằng `uv pip` thì thêm `--index-strategy unsafe-best-match` (uv mặc định chỉ lấy gói ở index đầu tiên có tên đó,
và index PyTorch có bản `certifi` cũ). Sinh lại sau mỗi lần đổi `uv.lock`:

```powershell
uv export --format requirements-txt --no-hashes --no-emit-project --emit-index-url --locked -o requirements.txt
```

Dữ liệu ảnh nguồn (VisDrone, NOMAD) không nằm trong repo; đường dẫn ở `configs/paths.yaml`.

## Môi trường

`scripts/env.ps1` (dot-source trong PowerShell, mỗi cửa sổ mới chạy lại) đặt các biến sau và tạo thư mục nếu chưa có:

| Biến | Giá trị |
|---|---|
| `UV_CACHE_DIR` | `D:\DevCache\uv` |
| `UV_PROJECT_ENVIRONMENT` | `D:\DevCache\venvs\xla` (venv ngoài repo, dùng chung giữa các worktree) |
| `UV_TOOL_DIR`, `UV_PYTHON_INSTALL_DIR` | `D:\DevCache\uv-tools`, `D:\DevCache\uv-python` |
| `PIP_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `YOLO_CONFIG_DIR` | `D:\DevCache\pip`, `\torch`, `\hf`, `\ultralytics` |
| `XLA_DATA_ROOT` | `D:\DevCache\datasets\xla` |
| `PYTHONUTF8` | `1` (tên file và log tiếng Việt) |
| `TMP`, `TEMP` | `D:\DevCache\tmp` (file tạm của pytest cũng ở ổ D) |

Quy ước: tải về để ở `D:\IOT_Tools\downloads\`, cache và venv ở `D:\DevCache\`, không để gì ở ổ C. Đường dẫn dùng chung ở
`configs/paths.yaml`; ghi đè từng khoá bằng biến `XLA_<KHOÁ VIẾT HOA>` (ví dụ `XLA_DATA_ROOT`, `XLA_GPU_LOCK_FILE`). Repo không
có file `.env`.

Sau `uv sync` lần đầu, chạy mọi lệnh bằng `uv run --no-sync` để uv không tự sửa venv dùng chung.

## Chạy

| Việc | Lệnh | GPU |
|---|---|---|
| Test CPU (giống CI) | `uv run --no-sync pytest -m "not gpu and not data"` | không |
| Tái lập bảng/hình của báo cáo | `uv run --no-sync python scripts/reproduce.py --check --json` | không |
| Dựng lại bảng/hình, chép vào báo cáo, ghi manifest | `uv run --no-sync python scripts/reproduce.py --write` | không |
| Liệt kê các bước huấn luyện lại | `uv run --no-sync python scripts/reproduce.py --full --dry-run` | không (chỉ in) |
| Kiểm một bundle theo hợp đồng | `uv run --no-sync python -m xla.contract validate <thư mục bundle>` | không bắt buộc |

`--check` làm hai việc: chấm lại 15 tệp dự đoán đã lưu và so sha256 của 9 bảng/hình với `results/reproduce/manifest.json`. Mã
thoát: 0 đạt, 1 lệch, 2 thiếu dữ liệu. Bảng và hình trong `BAO_CAO_DO_AN.md` chỉ lấy từ đây, không dán tay.

`--full` chạy lại 17 bước GPU (hồ sơ camera, hiệu chỉnh D2, E1, E2, xuất ONNX, NAFNet, chấm làm rõ), ước 6,6 giờ trên RTX 4060
Laptop. Thêm `--dry-run` để xem danh sách trước.

## Kết quả mong đợi

Test CPU (khoảng 30 s):

```
290 passed, 45 deselected in 26.20s
```

`reproduce.py --check --json` (khoảng 10 s) in một object, `pass` là kết luận:

```json
{"pass": true,
 "rescore": [{"name": "baseline_edge16", "mAP50": 0.126482, "ref": "results/baseline/baseline.json",
              "max_abs_diff": 0.0, "pass": true}, "... 15 mục ..."],
 "outputs_mismatch": [], "committed_stale": [], "assets_mismatch": [], "n_outputs": 9}
```

Không `--json` thì in từng dòng `OK  chấm lại <tên>: <mAP50>` và dòng cuối `ĐẠT`.

`reproduce.py --full --dry-run` mở đầu bằng `17 bước GPU, ước 6.6 giờ GPU (RTX 4060 Laptop, 18–19/11/2026)` rồi một dòng mỗi
bước. `python -m xla.contract validate <bundle>` in `[]` khi bundle đúng hợp đồng.

Số chính cần ra đúng (từ `results/reproduce/tables/`):

| Đại lượng | Giá trị |
|---|---|
| mAP50 test: mốc / E1 D2 / bundle `detector-v1.0.0` | 0,126 / 0,236 / 0,120 |
| Báo nhầm trên khung bay không người: D1 / D2 | 0,994 / 0,302 hộp/khung |
| A/B lần 1, mạng thắng gốc / thắng cổ điển | 11/25 / 1/25 (ngưỡng 70 % / 55 %) |
| Enhancer trong IOT-CV, nhịp thật p50 / p95 | 12,5 / 19,0 ms |
| Detector trong IOT-CV, nhịp thật p50 / p95 | 23,6 / 33,4 ms |
| Hết tốc lực hai model | 23,6 khung/s, tổng p95 44,9 ms |

## Test

```powershell
uv run --no-sync pytest -m "not gpu and not data"   # CPU, CI chạy đúng lệnh này
uv run --no-sync pytest -m gpu                      # cần GPU CUDA thật
uv run --no-sync pytest                             # tất cả, cần cả GPU lẫn dữ liệu local
```

Hai đánh dấu khai ở `tests/conftest.py`: `gpu` cần CUDA, `data` cần dữ liệu local (thiếu thì FAIL chứ không bỏ qua).
`tests/integration/test_iotcv_replay.py::test_ngan_sach_ms` đỏ có chủ đích trên laptop nhóm (ngân sách ms ở nhịp thật không đạt,
xem trên); nó cần GPU và bản ghi của IOT-CV nên CI không chạy.

## Gỡ lỗi

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| ORT báo có `CUDAExecutionProvider` nhưng chạy chậm như CPU | thiếu DLL cuDNN thì ORT vẫn liệt kê CUDA rồi lặng lẽ chạy CPU. `import torch` trước khi tạo session, và đọc `session.get_providers()` **sau** một lần chạy thử, không đọc trước |
| `cublasLt64_13.dll` không tìm thấy | đã cài `onnxruntime-gpu` mới hơn 1.24.4 (đòi CUDA 13). Về đúng bản ghim: `uv sync --locked` |
| Số ms của bench lệch nhiều giữa hai lần đo | GPU laptop hạ xung khi rảnh hoặc khi nóng. Đọc khối `machine` (nhiệt độ, xung) trong file bench trước khi so; đo xong ghi kèm điều kiện |
| Venv XLA biến mất sau khi chạy lệnh từ IOT-CV | subprocess `uv` gọi sang repo khác mà còn giữ `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` của repo này thì uv xoá venv bên kia. Bỏ hai biến đó khi gọi chéo |
| `uv run` tự cài lại gói, xung đột giữa các cửa sổ | venv dùng chung. Dùng `uv run --no-sync`; chỉ một nơi chạy `uv sync` |
| Máy treo hoặc màn xanh khi huấn luyện | máy nhóm từng lỗi RAM dưới tải nặng. Tối đa 4 worker, lưu checkpoint mỗi lượt |
| Test `data` FAIL trên máy mới | thiếu dữ liệu local (VisDrone, NOMAD, khung bay). Chạy `-m "not gpu and not data"`, hoặc trỏ `XLA_DATA_ROOT` tới dữ liệu |
| `uv pip install -r requirements.txt` báo không có `certifi==...` | uv chỉ lấy gói ở index đầu tiên có tên đó. Thêm `--index-strategy unsafe-best-match`, hoặc dùng pip |

## Hợp đồng với IOT-CV

`docs/hop-dong-model.md` (v1.0). Detector: input `frame` uint8 [480, 640, 3] BGR, output `boxes` [N, 4] + `scores` [N].
Enhancer: output `frame_out` uint8 [480, 640, 3]. IOT-CV kiểm sha256, phiên bản hợp đồng, tên và kiểu I/O rồi mới nạp.

Tải bundle từ GitHub Release **`models-v1.0.0`**:

```powershell
gh release download models-v1.0.0 --repo Localhost218/XLA --dir D:\DevCache\tmp\models
Expand-Archive D:\DevCache\tmp\models\detector-v1.0.0.zip D:\DevCache\tmp\models\detector-v1.0.0
uv run --no-sync python -m xla.contract validate D:\DevCache\tmp\models\detector-v1.0.0   # in []
```

Phía IOT-CV chép bundle bằng `scripts/vision/lay_model.py --from <thư mục bundle>` vào `models/vision/<bundle>/`.
`enhancer-v1.0.0` đúng hợp đồng nhưng không đạt A/B, nên IOT-CV để "Ảnh rõ" tắt mặc định.

## Tài liệu

| Tài liệu | File |
|---|---|
| Báo cáo đồ án | [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md) |
| Slide HTML | [`docs/slide/index.html`](docs/slide/index.html) |
| Hợp đồng model với IOT-CV | [`docs/hop-dong-model.md`](docs/hop-dong-model.md) |
| Plan nghiên cứu | [`plans/261118-ai-anh-chat-luong-thap/plan.md`](plans/261118-ai-anh-chat-luong-thap/plan.md) |
| Trạng thái và kết quả phase 10–14 (mục 7) | [`plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-10-14.md`](plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-10-14.md) |
| Báo cáo brainstorm gốc | [`plans/reports/261118-brainstorm-ai-anh-chat-luong-thap.md`](plans/reports/261118-brainstorm-ai-anh-chat-luong-thap.md) |
| Bảng kết quả của báo cáo | [`results/reproduce/tables/`](results/reproduce/tables/) |
| A/B làm rõ ảnh, quyết định không ship enhancer | [`results/ab/README.md`](results/ab/README.md) |
| E1, E2, mạng làm rõ | [`results/e1/README.md`](results/e1/README.md), [`results/e2/README.md`](results/e2/README.md), [`results/restore/README.md`](results/restore/README.md) |
| Mã và báo cáo bản cũ của nhóm | [`archive/README.md`](archive/README.md) |

## Giấy phép dữ liệu và mã

- VisDrone: chỉ dùng nghiên cứu; ảnh không được đăng lại trong repo.
- NOMAD: CC BY 4.0, ghi công tác giả bộ dữ liệu.
- `pyiqa` (MUSIQ, NIQE): PolyForm Noncommercial 1.0.0; chỉ dùng mô tả trong đồ án.
- Ultralytics YOLO11 (YOLO11s, thầy YOLO11m): AGPL-3.0. NAFNet: MIT. DINOv2: Apache-2.0.
- Khung bay trong `data/real/` là của nhóm.
