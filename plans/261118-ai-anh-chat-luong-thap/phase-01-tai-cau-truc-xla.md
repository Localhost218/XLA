# Phase 01: Tái cấu trúc XLA thành package `xla/`, archive phần sai, môi trường, CI

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | không | M (~1 ngày agent), 0 giờ GPU | Không |

Lane: serial
Depends-on: none
Effort: M
Owns: `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`, `web/server.py`, `archive/**`, `configs/paths.yaml`, `scripts/env.ps1`, `.github/workflows/tests.yml`, `tests/conftest.py`
Creates: `xla/__init__.py`, `xla/paths.py`, `xla/gpu_lock.py`, `xla/camera/__init__.py`, `xla/degrade/__init__.py`, `xla/enhance/__init__.py`, `xla/detect/__init__.py`, `xla/eval/__init__.py`, `tests/test_package_layout.py`
AX: n/a because this phase creates only package scaffolding and helpers, no command-line entry point
Accept: `pytest tests/test_package_layout.py`

## Mục tiêu

Dựng nền cho mọi phase sau: package `xla/` với năm gói `camera`, `degrade`, `enhance`, `detect`, `eval`; môi trường uv
Python 3.12 ở ổ D có GPU chạy thật; dữ liệu và kết luận sai của nhóm chuyển vào `archive/` kèm ghi chú đúng sự thật;
khoá GPU dùng chung (D9); CI chạy pytest CPU. Không viết thuật toán nào ở phase này.

## Đầu vào cần có

- `plans/reports/261118-brainstorm-ai-anh-chat-luong-thap.md` §3 (giữ/bỏ), §6 (repo).
- `plans/reports/261118-bang-chung/kiem-xla/kiem-xla.md` (vì sao dữ liệu cũ sai).
- `plan.md` cạnh file này: Decision register D7, D8, D9, D12; Deferred decisions 13, 14, 18.
- Repo `D:\XLA` (remote `Localhost218/XLA`, nhánh `main`, commit gốc `3e9a764`). Toàn quyền commit/push trong XLA.
- Máy: Windows, uv 0.13.8, RTX 4060 Laptop 8 GB, venv sẵn `D:\DevCache\venvs\xla` (Python 3.12.16).

## Việc theo thứ tự

1. **Nhánh.** `git fetch origin`. Nhánh `feat/ai-anh-chat-luong-thap` đã có trên origin (phiên lập plan 18/11/2026 đã commit
   plan lên đó) thì `git switch feat/ai-anh-chat-luong-thap` + `git pull --ff-only`; chưa có thì
   `git switch -c feat/ai-anh-chat-luong-thap origin/main` rồi commit plan (`plans/261118-ai-anh-chat-luong-thap/`,
   `plans/reports/261118-*`) trước mọi việc khác (đường dẫn plan phải đọc được từ worktree của lane).
2. **Biến môi trường ổ D** trong `scripts/env.ps1` (dot-source): `UV_CACHE_DIR=D:\DevCache\uv`,
   `UV_PROJECT_ENVIRONMENT=D:\DevCache\venvs\xla`, `UV_TOOL_DIR=D:\DevCache\uv-tools`, `PIP_CACHE_DIR=D:\DevCache\pip`,
   `TORCH_HOME=D:\DevCache\torch`, `HF_HOME=D:\DevCache\hf`, `YOLO_CONFIG_DIR=D:\DevCache\ultralytics`,
   `XLA_DATA_ROOT=D:\DevCache\datasets\xla`. Không tải gì về ổ C.
3. **`pyproject.toml`** (uv, `requires-python = ">=3.12,<3.13"`, `package = true` cho `xla`). Khai **một lần** đủ phụ thuộc
   cho cả plan (D8): numpy, opencv-python-headless, pillow ≥ 12 (cần `qtables` + `subsampling`), scipy, scikit-image,
   PyWavelets, pandas, pyyaml, jsonschema, matplotlib, torch + torchvision bản CUDA (cu124, khớp venv `vidstats` đã chạy;
   khai `[tool.uv.sources]` + index PyTorch), ultralytics, sahi, ensemble-boxes, pycocotools, onnx, onnxruntime-gpu,
   onnxsim, pyiqa, dgm-eval (nếu không cài được qua PyPI thì ghi nguồn git cụ thể vào pyproject, không cài tay),
   gdown, pytest, ruff. Phiên bản chốt bằng `uv lock`; ghi nguyên lỗi nếu gói nào không có wheel cho Python 3.12 Windows
   và dừng báo, không tự đổi Python.
4. **Kiểm GPU thật:** `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   phải in `True` và tên RTX 4060; `onnxruntime.get_available_providers()` có `CUDAExecutionProvider`. Không đạt thì dừng.
5. **Package:** `xla/__init__.py` (version), năm gói con rỗng có docstring nói gói làm gì; `xla/paths.py` đọc
   `configs/paths.yaml` (gốc repo, `XLA_DATA_ROOT`, gốc IOT-CV `d:/Coding/IOT-CV`, thư mục tải `D:/IOT_Tools/downloads`) — không
   đường dẫn nào gõ cứng trong code khác; `xla/gpu_lock.py`: context manager khoá file `D:\DevCache\locks\gpu.lock`
   (tạo thư mục nếu thiếu, chờ có timeout cấu hình được, ghi pid + lệnh vào file khoá, nhả khi thoát kể cả khi lỗi).
6. **Archive** (Deferred 13): `git mv` `data/real_uav`, `data/synthetic_gt`, `data/synthetic_degraded`, `data/raw_videos`,
   `data/augmented`, `results/tables`, `results/figures` vào `archive/v2-nhom/` giữ cấu trúc. Viết `archive/README.md`:
   ảnh "real_uav"/GT là ảnh drone phố có biển chữ Hán cộng hồng nhân tạo, không phải camera này; `raw_videos` là video đã mã
   hoá lại cảnh khác; bảng 1–7 đo trên dữ liệu do chính mô hình suy giảm sinh ra nên không dùng làm kết luận; cách xem bản cũ
   `git checkout 3e9a764`. `src/`, `notebooks/`, `web/`, `models/` giữ nguyên chỗ (notebook vẫn chạy).
7. **`web/server.py`:** chỉ đổi `host="0.0.0.0"` → `"127.0.0.1"` (dòng 446). Không sửa gì khác.
8. **`README.md`:** thêm khối đầu file: camera thật là ESP32-S3 + OV3660 nhìn thẳng xuống (không phải ESP32-CAM/OV2640),
   repo đang tái cấu trúc theo plan này, phần v2 cũ nằm ở `src/` + `archive/`. Viết lại toàn bộ README là việc của phase 18.
9. **`.gitignore`:** thêm `models/release/**/*.onnx` (Deferred 21: file model đi qua GitHub Release), `runs/`,
   `results/**/runs/`, `*.pt` ngoài `models/`, `.pytest_cache/`. **Không** ignore khung thật (`data/real/`,
   `results/ab/img/`, `models/release/**/golden/`): chủ dự án đã cho commit (Deferred 14).
10. **CI** `.github/workflows/tests.yml` (Deferred 18): push + pull_request, `timeout-minutes: 10`, không `paths:`,
    `concurrency` hủy run cũ, cài torch CPU, chạy `pytest -m "not gpu"`. Đăng ký marker `gpu` và `data` (cần dữ liệu local)
    trong `tests/conftest.py`; test có marker `data` **fail** (không skip) khi chạy local mà thiếu dữ liệu, còn CI loại chúng
    bằng `-m`.
11. **Test** `tests/test_package_layout.py`: import đủ năm gói; `xla.paths` không trả đường dẫn ổ C; thư mục cũ đã nằm dưới
    `archive/v2-nhom/` và không còn ở chỗ cũ; `archive/README.md` có các câu ở bước 6; `web/server.py` không còn `0.0.0.0`;
    `.gitignore` có các dòng ở bước 9 và `git check-ignore` không chặn `data/real/frames/a.jpg`, `results/ab/img/a.jpg`,
    `models/release/x-v0.0.0/golden/000.jpg`; `gpu_lock` khoá được, khoá lần hai trong cùng tiến trình khác chờ rồi timeout đúng.
12. Commit pathspec, push nhánh, CI xanh.

## Cổng pass

- `pytest tests/test_package_layout.py` exit 0 (Accept).
- `uv run pytest -m "not gpu"` xanh trên CI của nhánh.
- Bước 4 in đúng GPU và provider CUDA.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Một gói (dgm-eval, pyiqa, onnxruntime-gpu) không có wheel Py3.12 Windows | medium | medium | Ghi lỗi nguyên văn, dừng báo Owner; không đổi Python âm thầm |
| `git mv` làm notebook trỏ đường dẫn dữ liệu cũ | high | low | Notebook là bản lưu trữ; `archive/README.md` chỉ cách xem bản cũ |
| torch bản CPU bị kéo vào thay bản CUDA | medium | medium | Bước 4 chặn; `[tool.uv.sources]` ghim index CUDA |

## Rollback

Xoá nhánh `feat/ai-anh-chat-luong-thap`; `main` không bị đụng tới phase 02 mới mở PR.
