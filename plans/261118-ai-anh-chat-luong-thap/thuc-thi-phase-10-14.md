# Thực thi phase 10–14 (18/11/2026, đêm) — luật chung, hợp đồng giữa lane, ngân sách

Owner = phiên chính. Chủ dự án **đang ngủ, không trả lời được**: không agent nào hỏi người. Mọi chỗ plan ghi "dừng, báo
lead/Owner" → tự quyết theo phương án đầu tiên được khuyến nghị trong plan, ghi quyết định vào README kết quả của lane, báo
trong báo cáo cuối. Chỉ dừng thật khi không thể làm tiếp (ghi rõ vì sao + bằng chứng).

Đọc trước: `plan.md` (D1–D13, Deferred), `docs/hop-dong-model.md` (SSOT hợp đồng, **không sửa**), phase file của lane,
`thuc-thi-phase-01-07.md` §6–7 (trạng thái hiện tại), file này.

## 1. Lane, worktree, người sở hữu file

| Lane | Phase | Worktree / nhánh | GPU | Sở hữu (chỉ lane này ghi) |
|---|---|---|---|---|
| detect | 10, rồi 11 | `D:\XLA-wt\detect` / `lane/detect` | có | `xla/detect/train.py`, `xla/detect/selftrain.py`, `configs/train_e1.yaml`, `configs/selftrain_e2.yaml`, `results/e1/**`, `results/e2/**`, `tests/detect/test_finetune_runs.py`, `models/release/detector-*/**` (phase 11) |
| detect-export | 11 bước 5–7 (công cụ) | `D:\XLA-wt\detect-export` / `lane/detect-export` | ngắn | `xla/detect/export_onnx.py`, `tests/detect/test_detector_export.py` |
| classic | 12 | `D:\XLA-wt\classic` / `lane/classic` | không | `xla/enhance/classic.py`, `configs/enhance_classic.yaml`, `results/enhance/classic_bench.json`, `tests/enhance/test_classic.py` |
| restore | 13 | `D:\XLA-wt\restore` / `lane/restore` | có | `xla/enhance/arch/**`, `xla/enhance/net.py`, `xla/enhance/train_net.py`, `xla/enhance/export_onnx.py`, `configs/restore.yaml`, `results/restore/**`, `tests/enhance/test_enhancer_export.py`, `models/release/enhancer-*/**` |
| abeval | 14 (code, chạy thử bằng bundle giả) | `D:\XLA-wt\abeval` / `lane/abeval` | ngắn | `xla/eval/enhance_eval.py`, `xla/eval/ab_page.py`, `xla/eval/ab_score.py`, `tests/eval/test_enhance_eval.py`, `results/ab/README.md` (khung) |

Không lane nào sửa: `pyproject.toml`, `uv.lock`, `plans/**`, `docs/hop-dong-model.md`, `xla/contract.py`,
`xla/eval/temporal.py`, `xla/eval/detection.py`, `tests/conftest.py`, file của lane khác. Cần gói Python mới → không cài;
viết lại phần cần (vd. `einops.rearrange` → `reshape`/`permute`), ghi lý do. Cần đổi hợp đồng → không đổi, ghi + báo.
Phase 14 (gộp, chạy thật, trang A/B, PR #2) do Owner điều phối sau khi các lane xong.

## 2. Môi trường (Bash)

```bash
cd /d/XLA-wt/<lane>
export UV_CACHE_DIR='D:\DevCache\uv' UV_PROJECT_ENVIRONMENT='D:\DevCache\venvs\xla' TORCH_HOME='D:\DevCache\torch' \
  HF_HOME='D:\DevCache\hf' YOLO_CONFIG_DIR='D:\DevCache\ultralytics' XLA_DATA_ROOT='D:\DevCache\datasets\xla' \
  PYTHONUTF8=1 TMP='D:\DevCache\tmp' TEMP='D:\DevCache\tmp'
PY=/d/DevCache/venvs/xla/Scripts/python.exe
$PY -c "import xla; print(xla.__file__)"   # PHẢI in đường dẫn trong worktree của lane
```

- venv chung cài `xla` dạng editable trỏ về `D:\XLA`. Vì vậy **luôn** chạy `$PY -m pytest ...` và `$PY -m xla....` từ gốc
  worktree (cờ `-m` đặt thư mục hiện tại lên đầu `sys.path`); không gọi `pytest` trần, không `uv sync`, không `uv run` có sync.
- Không để gì ở ổ C. Run/trọng số/cache: `D:\DevCache\runs\xla\<lane>\...`, `D:\DevCache\weights\...`. Ổ D còn ~36 GB: giữ
  ≥ 10 GB trống (xoá checkpoint trung gian không cần khi gần chạm).
- Mạng (git push, tải mã/trọng số): lệnh Bash với `dangerouslyDisableSandbox: true`. Remote là SSH.

## 3. Máy và GPU

- RAM lật bit dưới tải nặng (4 lần màn xanh 18/11): job CPU nặng **tối đa 4 worker** (kể cả `workers` của dataloader);
  checkpoint mỗi epoch/lượt để máy sập thì chạy tiếp được (`resume`).
- GPU RTX 4060 Laptop 8 GB, **một job một lúc** qua `xla.gpu_lock.gpu_lock(label, timeout_s=...)` (D9). Lane khác có thể
  giữ khoá tới ~1 giờ: đặt `timeout_s` ≥ 6 giờ. Mỗi lần chạy giữ khoá **một** job (một lần train / một lần đo), nhả ngay
  khi xong để lane khác chen được giữa các lần.
- Job dài (> 8 phút): chạy tách tiến trình (PowerShell `Start-Process` hoặc `cmd /c start /b`), log ra file, có resume;
  chờ **trong lượt** bằng vòng đọc log ≤ 9 phút mỗi lệnh. Không kết thúc lượt khi job của mình còn chạy.

## 4. Ngân sách GPU đêm nay (thay ước 34–84 giờ của plan; Owner chốt, ghi Deferred trong README kết quả)

| Việc | Trần GPU | Cách giữ trần |
|---|---|---|
| Phase 10: smoke + 4 lần đủ (D0, D1, D2, D2+RealSR) | ~3 giờ tổng, ≤ 40 phút/lần | số epoch chọn từ smoke, **giống nhau cho 4 tập**; giữ imgsz 1280 nếu vừa, không thì 960 |
| Phase 11: thầy + ≤ 2 vòng trò + nhãn giả | ~2 giờ | thầy YOLO11m (không l), dừng vòng khi dev-val không tăng |
| Phase 13: đo tốc độ + smoke + huấn luyện | ~1,25 giờ | số bước chọn từ smoke |
| Xuất ONNX, đo ms, parity, phase 14 | ~1 giờ | |

D2+RealSR thêm vào E1 (tập `d2sr`) vì thắng KID ở phase 07 (`thuc-thi-phase-01-07.md` §7).

## 5. Hợp đồng giữa các lane (chép đúng, không đổi tên/kiểu)

**5.1 `xla.enhance.classic` (classic → abeval, phase 14)**

```python
def load_params(path: str | Path | None = None) -> dict          # None → configs/enhance_classic.yaml
def enhance(frame_bgr: np.ndarray, params: dict | None = None) -> np.ndarray
```
`frame_bgr` uint8 [480, 640, 3] BGR nguyên khung camera; trả uint8 [480, 640, 3] BGR; cột `[0, crop_left)` = lặp cột
`crop_left` (như `docs/hop-dong-model.md` §3); hàm thuần, tất định, không đổi mảng vào; sai kiểu/cỡ → `ValueError`.
`crop_left` lấy từ `configs/camera_profile.yaml` (qua `xla.contract.load_profile`).

**5.2 `xla.eval.temporal` (đã có trên nhánh tích hợp, commit 688894d)** — `consecutive_runs(frames, split, max_gap_s=0.25)`,
`flicker_ratio(raw, processed, crop_left)`. Restore dùng để chọn checkpoint, abeval dùng để chấm. Không viết hàm nhấp
nháy thứ hai.

**5.3 Bundle `detector-v1.0.0`, `enhancer-v1.0.0`** — đúng `docs/hop-dong-model.md` §2–§4, kiểm bằng
`python -m xla.contract validate <dir>` → `[]`. Bundle giả để thử: `xla.contract.make_dummy_bundle`. File `.onnx` không vào
git (Deferred 21); `meta.json` + `golden/` commit.

**5.4 `xla.detect.export_onnx` (detect-export → detect phase 11)**

```
python -m xla.detect.export_onnx --weights <pt> --out <bundle_dir> --version 1.0.0 --conf <float> \
    --metrics <json file> --train-run <str> [--imgsz 1280] [--parity-out <json>] [--skip-cuda] --json
```
```python
def export_bundle(weights: str | Path, out_dir: str | Path, *, version: str, conf_threshold: float,
                  metrics: dict, train_run: str, imgsz: int = 1280, parity_out: str | Path | None = None,
                  skip_cuda: bool = False) -> dict   # báo cáo: errors (list[str], rỗng = đạt), parity, ms_per_frame, onnx_sha256
```
Mã thoát 0 = bundle hợp lệ + parity đạt (|ΔmAP50| ≤ 0,005 giữa `.pt` và ORT CPU, ORT CUDA trên dev-val); 1 = lệch
hợp đồng/parity; 2 = sai cách gọi. Chạy được với cả model COCO 80 lớp (lấy lớp person = 0) lẫn model một lớp.

**5.5 Kết quả E1 cho phase 11** — `results/e1/e1.json` có khoá `best`:
`{"set": "d0|d1|d2|d2sr", "weights": "<đường dẫn tuyệt đối .pt ngoài repo>", "devval_map50": float, "conf_threshold": float, "threshold_source": "devval"}`.
CSV dự đoán (D11, conf ≥ 0,001, bỏ hộp chạm 16 px mép trái như mốc) commit ở `results/e1/preds/<set>_{test,neg,devval}.csv`
nếu mỗi file < 5 MB.

## 6. Git và báo cáo

- Commit dạng pathspec: file mới `git add <đường dẫn cụ thể>` rồi `git commit -m "..." -- <đường dẫn>`; không `git add -A`/`.`.
  Message kiểu log hiện có (`feat(detect): phase 10 - ...`), dòng cuối `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Đẩy `git push -u origin lane/<lane>`; không mở PR, không merge nhánh khác vào nhánh mình (trừ lane detect phase 11 gộp
  `lane/detect-export`).
- Xong = lệnh Accept của phase exit 0 + `$PY -m pytest -m "not gpu" -q` xanh trong worktree (trừ test `data` của bundle
  chưa tồn tại, ghi rõ tên) + `$PY -m ruff check .` sạch.
- Báo cáo cuối: Status (DONE / DONE_WITH_CONCERNS / BLOCKED), file đã ghi, số đo chính (giờ GPU thật, ms, mAP + CI), quyết
  định đã tự chốt, SHA commit đọc lại bằng `git log -1 --format=%H`.

## 6b. Quyết định Owner sau báo cáo detect-export (cho phase 11)

- `meta.metrics` của bundle detector và số báo cáo cho bundle phải chấm bằng **chính đường ONNX** (cắt mép trong graph,
  letterbox vuông 1280) trên test/neg, không lấy số của đường `.pt` nguyên khung + bỏ hộp mép: stand-in COCO cho thấy cắt
  trong graph thấp hơn 0,012 mAP50 trên dev-val. `e2.json` ghi cả hai đường, nói rõ dòng nào là bundle.
- `ms_per_frame` của bundle thật đo không `--skip-cuda`, khi `% Processor Performance` ≥ 80 % liên tục ~1 phút; ghi
  `cpu_perf_pct`. Hai test `gpu` của `tests/detect/test_detector_export.py` phải chạy xanh ở phase 11.
- Thư mục `--out` phải tên đúng `detector-v<version>` (công cụ trả exit 2 nếu sai).

## 7. Trạng thái (Owner cập nhật)

| Lane | Trạng thái |
|---|---|
| detect (10) | đang chạy (máy tắt 13:24 vì ổ sạc lỏng; chạy lại 13:35, workers 2 vì hết commit memory lúc 4 worker) |
| detect-export | **xong** `0934f1e`: parity ORT CPU/CUDA −0,0029 mAP50 so `.pt` (đạt); resize trong graph trùng bit với `LetterBox` |
| classic (12) | **xong** `e859e22`: hồng −91,6 %, dark channel 142,7→65,3, blocking 2,73→2,08; bench 95 ms lúc máy quá nhiệt (khoẻ 13–18 ms), đo lại ở phase 14 |
| restore (13) | đang chạy |
| abeval (14 code) | **xong** `fbaf1dc`: chạy giả đầu-cuối đạt; E4 COCO×gốc 0,1265 khớp mốc phase 09; lệnh chạy thật ở `results/ab/README.md` + báo cáo lane |
| detect (11) | chờ 10 + detect-export |
| 14 gộp + PR #2 | chờ tất cả |
