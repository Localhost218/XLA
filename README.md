# XLA: ảnh chất lượng thấp từ camera drone

Đồ án Xử lý ảnh (PTIT), nhóm 10. Bản hiện hành: **AI v2** (26/11/2026).

## Bài toán

Drone S500 của dự án IOT-CV mang camera ESP32-S3 N16R8 + OV3660, phát MJPEG 640×480 chất lượng 12 qua Wi-Fi về laptop.
Khung nhận được ám hồng, mù, vỡ khối JPEG và tối góc. Repo này làm hai việc trên **khung bay thật**, không phải ảnh tự sinh:

1. **Phát hiện người** trên khung gốc, cho người lái biết có người dưới drone.
2. **Làm rõ ảnh** ("Ảnh rõ") cho người xem trên web.

Model xuất ra ONNX theo một hợp đồng cố định và chạy trong phần mềm drone (repo IOT-CV, `backend/vision/`). XLA lo nghiên
cứu, huấn luyện và chấm điểm; IOT-CV chỉ nạp bundle đã xuất.

AI v2 xếp ba tầng cho nhận diện (detector YOLO11s, verifier EfficientNet-B0 chấm lại từng hộp, lọc track k1/n1) và đưa bộ lọc
cổ điển (deblock, khử hồng, bù tối góc, khử mù, unsharp) thành một graph ONNX chạy trực tiếp.

Báo cáo: [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md) (v2 ở mục 6). Slide HTML: mở `docs/slide/index.html` trong trình duyệt (phím
← →).

## Kết quả chính

Chấm trên 198 khung test thật (chuyến 02/11/2026, 39 khung có người, 149 người), CI 95 % bootstrap theo khối 30 s.

| | v1 | v2 | Hiệu [CI 95 %] |
|---|---|---|---|
| mAP50 cả hệ | 0,120 (`detector-v1.0.0`) | 0,210 (detector → verifier → track) | +0,090 [+0,026; +0,189] |
| Báo nhầm / khung bay trống (159 khung) | 0,881 | 0,289 | −0,591 [−1,033; −0,273] |
| "Ảnh rõ" classic ONNX vs gốc, A/B mù cả khung | | 25/25, Wilson [0,867; 1,000] | lặp đổi bên 5/5 |
| p95 tổng có "Ảnh rõ" trong IOT-CV, cùng lúc đo | 99,1 ms | 89,5 ms | −9,65 ms |
| p95 verifier trong IOT-CV | | 8,9 ms (có Ảnh rõ), 9,4 ms (chỉ nhận diện) | |

Hàng rào báo nhầm của ảnh rõ offline qua (−0,033 [−0,123; +0,039] hộp/khung trên 150 khung `cal`). Golden của ba bundle ship
khớp trên cả ORT CUDA lẫn CPU trong IOT-CV. Phần cải thiện mAP50 và báo nhầm đến từ verifier; detector vẫn là ONNX của v1.

So sánh phương pháp đã chạy trước khi chọn bản đang ship (chi tiết `results/v2/README.md` § "Thư mục con"):

| Đã thử | Lý do chọn bản đang chạy |
|---|---|
| Thầy YOLO11m làm nhãn giả | không rõ hơn detector v1 trên devpool, nên giữ detector v1 |
| Trò YOLO11s C3 (k-fold theo phiên) | tốt nhất OOF nhưng trên cùng đường ONNX không hơn v1 |
| Real-ESRGAN x2plus offline | A/B cả khung không phân biệt rõ với classic (16/25, Wilson [0,445; 0,798]); classic chạy trực tiếp, offline 0,404 s/khung |
| SwinIR | quá chậm cho một chuyến bay |
| BasicVSR++ | ONNX Runtime thiếu kernel `DeformConv` |
| Lai classic → NAFNet | không qua bước lọc của phase 10, không xuất |

Bảng đầy đủ: `results/reproduce/tables/t7_v2_nhan_dien.md`, `t8_v2_ab.md`, `t9_v2_ms.md`, hình
`results/reproduce/fig/f4_v2_bao_nham_recall.svg`. Bảng v1 (t1 đến t6) cùng thư mục.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `xla/camera/` | trích khung bay, đo hồ sơ camera, chia tập theo khối thời gian và theo phiên bay |
| `xla/degrade/` | bộ làm xấu D1 (ngẫu nhiên), D2 (bản sao camera), hiệu chỉnh CMA-ES, dữ liệu ngoài (`sources.py`), dựng tập `d02x` |
| `xla/detect/` | mốc COCO, huấn luyện E1 (YOLO11s), tự huấn luyện E2, k-fold theo phiên (v2), xuất ONNX |
| `xla/verify/` | v2: khai thác âm khó, huấn luyện verifier, xuất ONNX có RoiAlign |
| `xla/enhance/` | bộ lọc cổ điển (`classic.py`, SSOT), classic thành ONNX (`export_classic.py`), NAFNet, siêu phân giải ×2, video |
| `xla/eval/` | chấm nhận diện (bootstrap theo khối), cả hệ v2 (`system.py`), C2ST, KID, trang A/B mù, hàng rào báo nhầm |
| `xla/contract.py` | kiểm bundle theo hợp đồng với IOT-CV |
| `xla/gpu_lock.py` | khoá GPU dùng chung giữa các tiến trình |
| `models/release/` | `meta.json` + `golden/` của mọi bundle; file `.onnx` lấy từ GitHub Release |
| `data/real/` | khung bay thật của nhóm (JPEG nguyên byte), `manifest.csv`, `split.json` |
| `results/` | mọi kết quả đo; `results/v2/` cho AI v2; `results/reproduce/` là bảng/hình của báo cáo |
| `configs/` | đường dẫn (`paths.yaml`), nguồn dữ liệu (`sources.yaml`), hồ sơ camera, cấu hình huấn luyện |
| `scripts/` | `env.ps1` (biến môi trường), `reproduce.py` (tái lập), `labelstudio_start.ps1` |
| `tests/` | pytest; đánh dấu `gpu` (cần CUDA) và `data` (cần dữ liệu local) |
| `plans/` | plan v1 (`261118-ai-anh-chat-luong-thap/`), plan v2 (`261125-ai-v2/`), báo cáo nghiên cứu |
| `archive/`, `src/`, `web/`, `notebooks/` | mã và dữ liệu bản cũ của nhóm, giữ để tham khảo |

## Yêu cầu hệ thống

| Thứ | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 11 (máy nhóm) hoặc Linux x86_64 (CI). `uv.lock` chỉ giải cho hai môi trường này |
| Python | 3.12 (`requires-python = ">=3.12,<3.13"`); uv tự tải nếu máy chưa có |
| uv | bản mới; đã dùng 0.13.8 |
| GPU | không bắt buộc cho test CPU và `reproduce.py --check`. Huấn luyện, xuất ONNX, bench, test `gpu`: GPU NVIDIA + driver hỗ trợ CUDA 12.4 (máy nhóm: RTX 4060 Laptop 8 GB) |
| CUDA | không cần cài CUDA Toolkit riêng: torch 2.6.0+cu124 mang sẵn DLL CUDA 12.4 và cuDNN 9 |
| Khoá GPU | mọi việc GPU giữ khoá file `D:\DevCache\locks\gpu.lock` (`gpu_lock_file` trong `configs/paths.yaml`, `xla/gpu_lock.py`). Hai tiến trình GPU chạy cùng lúc làm lệch số ms và có thể hết VRAM; tiến trình chết thì hệ điều hành tự nhả khoá |
| Ổ đĩa | venv vài GB (torch CUDA); dữ liệu ngoài v2 khoảng 10 GB sau giải nén; mọi cache để ở ổ D |

Gói GPU chọn theo hệ điều hành, đã ghim trong `pyproject.toml`:

| | Windows | Linux (CI) |
|---|---|---|
| torch, torchvision | 2.6.0+cu124, 0.21.0+cu124 (index `download.pytorch.org/whl/cu124`) | 2.6.0+cpu, 0.21.0+cpu |
| ONNX Runtime | `onnxruntime-gpu==1.24.4` | `onnxruntime` (CPU) |

`onnxruntime-gpu` 1.24.4 chứ không mới hơn: 1.32.0 đòi CUDA 13 (`cublasLt64_13.dll`). Bản 1.24.4 chạy CUDA EP bằng DLL
CUDA 12.4/cuDNN 9 của torch, nên phải `import torch` trước khi tạo session.

## Cài đặt

1. Lấy mã:

   ```powershell
   git clone git@github.com:Localhost218/XLA.git D:\XLA
   cd D:\XLA
   ```

2. Đặt biến môi trường (dot-source, mỗi cửa sổ PowerShell mới chạy lại):

   ```powershell
   . .\scripts\env.ps1
   ```

3. Cài đúng `uv.lock` vào venv dùng chung `D:\DevCache\venvs\xla`:

   ```powershell
   uv sync --locked
   ```

4. Kiểm torch thấy GPU:

   ```powershell
   uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   Windows có GPU in `2.6.0+cu124 True`; Linux CI in `2.6.0+cpu False`.

5. Lấy file `.onnx` của bundle (mục "Đầu ra mong đợi") nếu cần chạy test `gpu` hay bench.

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

Lệnh trên đã chạy lại ngày 26/11/2026 trên `uv.lock` hiện tại và cho đúng file đang commit.

## Môi trường

`scripts/env.ps1` đặt các biến sau và tạo thư mục nếu chưa có:

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

## Dữ liệu

| Nguồn | Ở đâu | Dùng cho | Giấy phép |
|---|---|---|---|
| Khung bay thật của nhóm | `data/real/frames/` trong repo: 2 915 khung bay + 300 khung âm trong nhà, JPEG nguyên byte camera | dev, test (198 khung), `cal`, khung âm | của nhóm |
| VisDrone2019-DET | tải theo `configs/sources.yaml` khối `downloads.visdrone` (url + sha256) | ảnh sạch cho D0/D1/D2 (v1) | chỉ nghiên cứu phi thương mại |
| NOMAD (tập con) | `python -m xla.degrade.fetch_nomad` theo `downloads.nomad` | ảnh sạch nhìn từ drone (v1) | CC BY 4.0 |
| HERIDAL | Zenodo 5662351, `downloads.heridal` | dương + nền cho `d02x` và verifier (v2): 1 684 ảnh sau lọc góc nhìn | CC BY 3.0 |
| WiSARD Multi-Modal Sample | trang WiSARD, `downloads.wisard` | như trên: 110 ảnh | MIT |
| TinyPerson | Google Drive trong README TinyBenchmark, `downloads.tinyperson` | lớp dương của verifier: 173 ảnh | bộ dữ liệu không ghi giấy phép (mã MIT) |

Ảnh của các bộ ngoài không có trong repo. Repo chỉ giữ cấu hình, sha256 và mã đọc. Model v2 học từ các bộ này và từ trọng số
ImageNet (torchvision), nên **chỉ dùng cho nghiên cứu**.

Lấy dữ liệu ngoài v2:

1. Tải từng file trong `configs/sources.yaml` khối `downloads.<bộ>.files` vào `D:\IOT_Tools\downloads\datasets\<bộ>\`.
2. Giải nén và kiểm sha256:

   ```powershell
   uv run --no-sync python -m xla.degrade.sources extract           # dừng nếu sha256 file nén khác cấu hình
   uv run --no-sync python -m xla.degrade.sources extract --verify  # băm lại từng file đã giải nén
   ```

3. Lọc góc nhìn (GPU, giữ khoá) và dựng tập trộn:

   ```powershell
   uv run --no-sync python -m xla.degrade.sources probe-external
   uv run --no-sync python -m xla.degrade.sources stats --json
   uv run --no-sync python -m xla.degrade.build --set d02x --workers 4 --json
   ```

Ảnh giải nén nằm ở `D:\DevCache\datasets\xla\external\<bộ>\`, tập sinh ở `D:\DevCache\datasets\xla\d02x\`. Chi tiết cách
chia và lọc: `results/v2/data/README.md`. Khung bay và cách chia: `data/real/README.md`.

## Tái lập

| Việc | Lệnh | GPU |
|---|---|---|
| Chấm lại dự đoán đã lưu, so bảng/hình với manifest | `uv run --no-sync python scripts/reproduce.py --check --json` | không |
| Dựng lại bảng/hình, chép vào báo cáo, ghi manifest | `uv run --no-sync python scripts/reproduce.py --write` | không |
| Liệt kê các bước huấn luyện lại của v1 | `uv run --no-sync python scripts/reproduce.py --full --dry-run` | không (chỉ in) |
| Chạy lại các bước GPU của v1 | `uv run --no-sync python scripts/reproduce.py --full` | có |
| Kiểm một bundle theo hợp đồng | `uv run --no-sync python -m xla.contract validate <thư mục bundle>` | không |

`--check` chấm lại 18 tệp dự đoán đã lưu (15 của v1, 3 tầng cả hệ v2) và so sha256 của 13 bảng/hình với
`results/reproduce/manifest.json`. Mã thoát: 0 đạt, 1 lệch, 2 thiếu dữ liệu. Bảng và hình trong `BAO_CAO_DO_AN.md` chỉ lấy
từ đây, không dán tay. Sau khi một kết quả trong `results/` đổi, chạy `--write` rồi commit cả bảng lẫn manifest.

`--full` chạy 17 bước GPU của v1 (hồ sơ camera, hiệu chỉnh D2, E1, E2, xuất ONNX, NAFNet, chấm làm rõ), ước 6,6 giờ trên
RTX 4060 Laptop. Các bước v2 chạy tay theo thứ tự dưới đây; mỗi bước GPU tự giữ khoá `gpu.lock`. Lệnh đầy đủ kèm tham số ở
README của từng thư mục `results/v2/*/`.

| Bước v2 | Lệnh chính | Đầu ra |
|---|---|---|
| Dữ liệu ngoài | `python -m xla.degrade.sources extract`, `python -m xla.degrade.build --set d02x` | `results/v2/data/` |
| Verifier: khai thác âm khó | `python -m xla.verify.mine detect` → `sheet` → `build` | `results/v2/verifier/` |
| Verifier: huấn luyện, chọn | `python -m xla.verify.train prep` → `run` → `select` | `select.json` |
| Verifier: xuất ONNX | `python -m xla.verify.export_onnx --json` | `models/release/verifier-v1.0.0` |
| Cả hệ: hiệu chỉnh ngưỡng | `python -m xla.eval.system calibrate --iotcv <IOT-CV> --workers 4 --json` → `apply` | `detector-v1.0.1`, ngưỡng verifier 0,35 |
| Cả hệ: chấm test một lần | `python -m xla.eval.system score --iotcv <IOT-CV> --json` | `results/v2/system/score.json`, bảng t7 |
| Classic thành ONNX | `python -m xla.enhance.export_classic --params configs/enhance_classic.yaml --out models/release/enhancer-v2.0.0 --json` | `models/release/enhancer-v2.0.0` |
| Siêu phân giải ×2 (so sánh) | `python -m xla.enhance.sr bench\|train\|select\|export --config configs/sr_x2.yaml --json` | `results/v2/sr/` |
| A/B mù, hàng rào báo nhầm | `python -m xla.eval.guard_fp ...`, `python -m xla.eval.ab_page build ...` | `results/ab-v2/`, bảng t8 |
| Bench trong IOT-CV | `results/v2/iotcv/bench_v1_v2.py` (chạy bằng venv của IOT-CV) | `results/v2/iotcv/bench.json`, bảng t9 |

Cả hệ (`xla.eval.system`) gọi sang IOT-CV bằng `uv run --no-sync` trong venv **của IOT-CV** và bỏ
`UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` của XLA khỏi môi trường con.

## Đầu ra mong đợi

Test CPU (khoảng 2 phút), lần chạy 26/11/2026 trên clone chưa tải `.onnx` (3 test bỏ qua vì thiếu file `.onnx` của Release):

```
479 passed, 3 skipped, 72 deselected, 25 warnings in 133.44s (0:02:13)
```

`reproduce.py --check --json` (khoảng 10 s) in một object, `pass` là kết luận:

```json
{"pass": true,
 "rescore": [{"name": "baseline_edge16", "mAP50": 0.126482, "ref": "results/baseline/baseline.json",
              "max_abs_diff": 0.0, "pass": true}, "... 18 mục ..."],
 "outputs_mismatch": [], "committed_stale": [], "assets_mismatch": [], "n_outputs": 13}
```

Không `--json` thì in từng dòng `OK  chấm lại <tên>: <mAP50>` (ba dòng cuối là `v2_system_detector: 0.119471`,
`v2_system_verifier: 0.209782`, `v2_system_track: 0.209782`) và dòng cuối `ĐẠT`. `python -m xla.contract validate <bundle>`
in `[]` khi bundle đúng hợp đồng.

Bảng t7 (`results/reproduce/tables/t7_v2_nhan_dien.md`), từng tầng của cả hệ trên 198 khung test:

| Tầng | mAP50 [CI 95 %] | Báo nhầm / khung bay trống (159) [CI] | / khung trong nhà (300) | Ngưỡng |
|---|---|---|---|---|
| 1 detector `detector-v1.0.1` (ONNX của v1.0.0) | 0,119 [0,032; 0,197] | 0,874 [0,333; 1,486] | 0,003 | 0,077 |
| 2 + `verifier-v1.0.0` @0,350 | 0,210 [0,122; 0,312] | 0,289 [0,036; 0,631] | 0,003 | 0,350 |
| 3 + lọc track k1/n1 (cả hệ) | 0,210 [0,122; 0,312] | 0,289 [0,036; 0,631] | 0,003 | 0,350 |

Bảng t8 (`t8_v2_ab.md`), A/B mù lượt cả khung, 25 cặp mỗi so sánh:

| So sánh | Bên thứ nhất thắng | Tỉ lệ [Wilson 95 %] |
|---|---|---|
| classic vs gốc | 25/25 | 1,000 [0,867; 1,000] |
| offline ×2 vs gốc | 25/25 | 1,000 [0,867; 1,000] |
| offline ×2 vs classic | 16/25 | 0,640 [0,445; 0,798] |

Cặp lặp đổi bên: 5/5 chọn như lần đầu.

Bảng t9 (`t9_v2_ms.md`), trong IOT-CV ở nhịp camera thật, bản ghi 20261102-100941, 4 lượt xen kẽ v1-v2-v2-v1, xung GPU trung
vị 780 đến 802 MHz:

| Luồng có "Ảnh rõ" | v1 p50 / p95 ms | v2 p50 / p95 ms | Hiệu p95 |
|---|---|---|---|
| detector | 53,2 / 63,3 | 57,4 / 63,4 | +0,15 |
| verifier | | 0,0 / 8,9 | |
| enhancer | 29,7 / 34,4 | 15,4 / 19,3 | −15,12 |
| tổng | 84,5 / 99,1 | 76,3 / 89,5 | −9,65 |

Bundle (sha256 của `.onnx`, ghi trong `models/release/<bundle>/meta.json`):

| Bundle | Ship trong IOT-CV | sha256 `.onnx` |
|---|---|---|
| `detector-v1.0.1` | có | `23b1da69165ffe05760bda23b4a41c858930e8a8fd8b0aebc039f1e0c5c59aef` |
| `verifier-v1.0.0` | có, ngưỡng 0,35 | `d566343a5487775b9f6334d9df0bb0a1d9c9b2d2b76e2b8bfa4738facc012d58` |
| `enhancer-v2.0.0` | có, "Ảnh rõ" bật sẵn | `83d96e8b0f628d0e6cc271ed9550220962de897cac2d57ed306a1db51cb5477b` |
| `enhancer-offline-v1.0.0` | không (bản so sánh) | `e93b3eaa714a003df0c6a77c11dacefc7b22666143dc8c490957dc1a5b53afc8` |

File `.onnx` không vào git. Cả bốn bundle v2 nằm ở GitHub Release **`models-v2.0.0`**; bundle v1 (`detector-v1.0.0`,
`enhancer-v1.0.0`) ở **`models-v1.0.0`**:

```powershell
gh release download models-v2.0.0 --repo Localhost218/XLA --dir D:\DevCache\tmp\models
Expand-Archive D:\DevCache\tmp\models\verifier-v1.0.0.zip D:\DevCache\tmp\models\verifier-v1.0.0
uv run --no-sync python -m xla.contract validate D:\DevCache\tmp\models\verifier-v1.0.0   # in []
```

## Kết nối với IOT-CV

Hợp đồng: [`docs/hop-dong-model.md`](docs/hop-dong-model.md), `contract_version` 1.1 (bundle 1.0 vẫn nạp được).

| Kind | Input | Output |
|---|---|---|
| detector | `frame` uint8 [480, 640, 3] BGR | `boxes` [N, 4] pixel khung gốc, `scores` [N]; meta 1.1 có thể thêm `recommended_track` |
| verifier | `frame` + `boxes` [N, 4] | `scores` [N] |
| enhancer | `frame` | `frame_out` uint8 [480, 640, 3] |
| enhancer-offline | `frame` | ×2, [960, 1280, 3] |

Mỗi bundle là một thư mục `<kind>-v<MAJOR.MINOR.PATCH>/` gồm `.onnx`, `meta.json` (có `onnx_sha256`) và `golden/` (3 khung dev
thật + `expected.json`). IOT-CV kiểm sha256, phiên bản hợp đồng, tên và kiểu I/O rồi mới nạp. Phía IOT-CV, trong venv của
IOT-CV:

```powershell
uv run --no-sync python scripts/vision/lay_model.py --from D:/XLA/models/release/<bundle>    # chép vào models/vision/<bundle>/
uv run --no-sync python scripts/vision/kiem_bundle.py models/vision/<bundle> --json          # chạy golden, thêm --cpu cho ORT CPU
```

`lay_model.py` cần file `.onnx` đã nằm trong thư mục nguồn (tải từ Release như trên). Bundle mặc định của IOT-CV ghi ở
`backend/config.py` bên đó. Ảnh rõ offline không ship nên API offline của IOT-CV trả 503 và web ghi lý do.

Gọi IOT-CV từ XLA (hay ngược lại) bằng `uv` thì bỏ `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` của repo gọi, không thì uv xoá venv
của repo kia.

## Test

```powershell
uv run --no-sync pytest -m "not gpu and not data"   # CPU, CI GitHub chạy đúng lệnh này trên mọi push
uv run --no-sync pytest -m gpu                      # cần GPU CUDA thật
uv run --no-sync pytest                             # tất cả, cần cả GPU lẫn dữ liệu local
```

Hai đánh dấu khai ở `tests/conftest.py`: `gpu` cần CUDA, `data` cần dữ liệu local (thiếu thì FAIL chứ không bỏ qua). Test
nghiệm thu v2: `tests/integration/test_system_v2.py` (cả hệ), `tests/integration/test_iotcv_v2.py` (bench trong IOT-CV),
`tests/verify/test_verifier.py`, `tests/enhance/test_classic_onnx.py`, `tests/contract/test_model_contract.py`.
`tests/test_reproduce.py` kiểm `--check` đạt, mọi bảng t7 đến t9 có khối trong báo cáo và số v2 trên slide lấy nguyên văn từ
bảng.

`tests/integration/test_iotcv_replay.py::test_ngan_sach_ms` so ms với ngân sách của plan v1 ở nhịp camera thật, nên kết quả
theo xung GPU của máy; trên laptop nhóm nó báo đỏ. Test này cần GPU và bản ghi của IOT-CV, CI không chạy.

## Gỡ lỗi

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| ORT báo có `CUDAExecutionProvider` nhưng chạy chậm như CPU | thiếu DLL cuDNN thì ORT vẫn liệt kê CUDA rồi lặng lẽ chạy CPU. `import torch` trước khi tạo session, và đọc `session.get_providers()` sau một lần chạy thử, không đọc trước |
| `cublasLt64_13.dll` không tìm thấy | đã cài `onnxruntime-gpu` mới hơn 1.24.4 (đòi CUDA 13). Về đúng bản ghim: `uv sync --locked` |
| Số ms của bench lệch nhiều giữa hai lần đo | GPU laptop hạ xung khi rảnh hoặc khi nóng. Đọc khối `machine` (nhiệt độ, xung) trong file bench trước khi so; đo khi không có job GPU khác (khoá `gpu.lock`); so v1 và v2 xen kẽ trong một tiến trình |
| Việc GPU đứng chờ mãi | tiến trình khác đang giữ `D:\DevCache\locks\gpu.lock`. Đợi nó xong; khoá tự nhả khi tiến trình giữ khoá chết |
| Venv XLA biến mất sau khi chạy lệnh từ IOT-CV | subprocess `uv` gọi sang repo khác mà còn giữ `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` của repo này thì uv xoá venv bên kia. Bỏ hai biến đó khi gọi chéo |
| `uv run` tự cài lại gói, xung đột giữa các cửa sổ | venv dùng chung. Dùng `uv run --no-sync`; chỉ một nơi chạy `uv sync` |
| Máy treo hoặc màn xanh khi huấn luyện | máy nhóm từng lỗi RAM dưới tải nặng. Tối đa 4 worker, lưu checkpoint mỗi lượt |
| Test `data` FAIL trên máy mới | thiếu dữ liệu local (VisDrone, NOMAD, dữ liệu ngoài v2). Chạy `-m "not gpu and not data"`, hoặc trỏ `XLA_DATA_ROOT` tới dữ liệu |
| `uv pip install -r requirements.txt` báo không có `certifi==...` | uv chỉ lấy gói ở index đầu tiên có tên đó. Thêm `--index-strategy unsafe-best-match`, hoặc dùng pip |

## Tài liệu

| Tài liệu | File |
|---|---|
| Báo cáo đồ án (v1 mục 1 đến 5, v2 mục 6, tái lập mục 9) | [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md) |
| Slide HTML | [`docs/slide/index.html`](docs/slide/index.html) |
| Hợp đồng model với IOT-CV | [`docs/hop-dong-model.md`](docs/hop-dong-model.md) |
| Plan AI v2 (18 phase) | [`plans/261125-ai-v2/plan.md`](plans/261125-ai-v2/plan.md) |
| Tổng hợp kết quả v2, bundle, thư mục con | [`results/v2/README.md`](results/v2/README.md) |
| Cả hệ detector → verifier → track | [`results/v2/system/README.md`](results/v2/system/README.md) |
| Verifier | [`results/v2/verifier/README.md`](results/v2/verifier/README.md) |
| Classic thành ONNX | [`results/v2/classic/README.md`](results/v2/classic/README.md) |
| IOT-CV chạy bundle v2, bench | [`results/v2/iotcv/README.md`](results/v2/iotcv/README.md) |
| Dữ liệu ngoài v2 | [`results/v2/data/README.md`](results/v2/data/README.md) |
| A/B mù v2 và quyết định ship | [`results/ab-v2/README.md`](results/ab-v2/README.md) |
| Bảng và hình của báo cáo | [`results/reproduce/tables/`](results/reproduce/tables/), [`results/reproduce/fig/`](results/reproduce/fig/) |
| Khung bay thật | [`data/real/README.md`](data/real/README.md) |
| Plan v1 | [`plans/261118-ai-anh-chat-luong-thap/plan.md`](plans/261118-ai-anh-chat-luong-thap/plan.md) |
| E1, E2, mạng làm rõ (v1) | [`results/e1/README.md`](results/e1/README.md), [`results/e2/README.md`](results/e2/README.md), [`results/restore/README.md`](results/restore/README.md) |
| Mã và dữ liệu bản cũ của nhóm | [`archive/README.md`](archive/README.md) |

## Giấy phép dữ liệu và mã

- Khung bay trong `data/real/` là của nhóm.
- VisDrone: chỉ dùng nghiên cứu; ảnh không được đăng lại trong repo.
- NOMAD: CC BY 4.0, ghi công tác giả bộ dữ liệu.
- HERIDAL: CC BY 3.0. WiSARD (Multi-Modal Sample): MIT. TinyPerson: bộ dữ liệu không ghi giấy phép (mã MIT). Ảnh của ba bộ
  không có trong repo; model v2 học từ chúng và từ trọng số ImageNet (torchvision), nên chỉ dùng nghiên cứu.
- `pyiqa` (MUSIQ, NIQE): PolyForm Noncommercial 1.0.0; chỉ dùng mô tả trong đồ án.
- Ultralytics YOLO11 (YOLO11s, thầy YOLO11m): AGPL-3.0. NAFNet: MIT. DINOv2: Apache-2.0.
- Real-ESRGAN: BSD-3-Clause; kiến trúc RRDBNet vendor từ BasicSR (Apache-2.0).
