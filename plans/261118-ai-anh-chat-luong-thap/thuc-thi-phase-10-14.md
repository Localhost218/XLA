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

## 7. Trạng thái (Owner cập nhật; bảng cuối 19/11/2026)

Mọi lane xong và đã gộp vào `feat/ai-anh-chat-luong-thap`; SHA là commit cuối của lane / merge commit trên nhánh tích hợp.

| Lane | Trạng thái |
|---|---|
| detect (10) | **xong** 18/11 `d1a13a2`, gộp `81a4898`: test mAP50 mốc 0,126 / D0 0,297 / D1 0,143 / D2 0,236 / D2sr 0,149; D2−D1 +0,093 [−0,073; +0,252]; best = D2 (dev-val 0,329); B1 chỉ D0 đạt, B2 B3 không model nào đạt |
| detect-export | **xong** 18/11 `0934f1e`, gộp `63ed406`: parity ORT CPU/CUDA −0,0029 mAP50 so `.pt` (đạt); resize trong graph trùng bit với `LetterBox`; mốc cổng đổi sang `.pt rect=False` ở `8746726` (§9) |
| classic (12) | **xong** 18/11 `e859e22`, gộp `dcc04d4`: hồng −91,6 %, dark channel 142,7→65,3, blocking 2,73→2,08 (200 khung dev); bench đo lại ở phase 14 lúc máy khoẻ p50 9,18 / p95 10,21 ms (thay số 95 ms lúc quá nhiệt) |
| abeval (14 code) | **xong** 18/11 `fbaf1dc`, gộp `52ec195`: chạy giả đầu-cuối đạt; E4 COCO×gốc 0,1265 khớp mốc phase 09 |
| restore (13) | **xong** 18/11 `868d881`, gộp `9650338`: vòng 1 `01a93c9` NAFNet-width32 p95 82,1 ms (Accept FAIL); vòng 2 NAFNet-w16, bundle `enhancer-v1.0.0` p50 15,4 / p95 21,0 ms, Accept 7/7 PASS |
| detect (11) | **xong** 19/11 `b0eec03`, gộp `5a4a05f`: dừng sau r2 (dev-val 0,296 < r1 0,379), ship trò r1; bundle `detector-v1.0.0` ONNX test mAP50 0,120, B1 B2 không đạt, B3 đạt p95 20,2 ms |
| 14 gộp + chạy thật | **xong** 19/11 trên nhánh tích hợp: `9fbdbba` (bước 1–7, model thật, E4, trang A/B), `e142767` (trang A/B lần 2 phóng to), `779985a` (bước 9–11, chấm A/B lần 1, tiêu chí §8), `8ed4930` (kết quả A/B lần 2) |
| PR #2 | `feat/ai-anh-chat-luong-thap` → `main`, **Owner mở** 19/11 (https://github.com/Localhost218/XLA/pull/2), merge khi CI xanh; file này đi trong chính PR đó, có trên `main` là đã merge |

### Kết quả cuối phase 10–14

- **Phase 10 (E1):** best theo dev-val = D2 (dev-val mAP50 0,329). Trên test D2 − D1 +0,093 [−0,073; +0,252], CI chạm 0:
  chưa đủ để nói D2 hơn D1 về mAP; báo nhầm khung bay D2 thấp hơn D1 rõ (−0,692 [−1,238; −0,246]). `results/e1/README.md`.
- **Phase 11 (E2):** ship trò r1 theo dev-val. Bundle `detector-v1.0.0` đường ONNX test mAP50 0,120 [0,032–0,197]: B1 không
  đạt, B2 không đạt (0,881 hộp/khung bay không người), **B3 đạt** (ORT CUDA EP p95 20,2 ms). Dev-val không dự đoán được test
  lần này (r1 kém E1 D2 trên test, r2 cao nhất 0,313 nhưng không ship). `results/e2/README.md`.
- **Phase 12 (cổ điển):** `xla.enhance.classic` p50 9,18 ms CPU. Trên test: hồng −89,2 %, dark channel −51,6 % (đạt), nhấp
  nháy 2,18 (không đạt); thắng mạng 24/25 lượt A/B lần 1. `results/enhance/classic_bench.json`, `results/ab/README.md` §4.2.
- **Phase 13 (mạng):** NAFNet-w16 (`enhancer-v1.0.0`), p95 21,0 ms. Trên test: hồng −99,9 % (đạt), dark channel −36,8 %
  (không đạt), nhấp nháy 1,81 (không đạt). `results/restore/README.md` §8.
- **Phase 14:** E4 → **D5 giữ `raw`** (bundle × mạng − gốc −0,085 [−0,139; −0,011]; hợp đồng không đổi). A/B lần 1 (cả khung):
  mạng vs gốc 11/25 = 0,44 [0,267; 0,629], vs cổ điển 1/25 = 0,04 [0,007; 0,195]. A/B lần 2 (phóng to 2×): 13/25 = 0,52
  [0,335; 0,700] và 5/25 = 0,20 [0,089; 0,391]. Quanh người gộp hai lần mạng thắng gốc 17/21 = 0,81 [0,60; 0,92]; đây là mô
  tả hậu kiểm, không phải tiêu chí đã đăng ký. Mạng đạt 3/7 tiêu chí §8 nhánh A → **không ship `enhancer-v1.0.0` làm ảnh rõ
  mặc định** (bundle vẫn hợp lệ hợp đồng, Kill (c) không kích hoạt). Hướng sửa (màu nguội/nhạt, khử mù yếu, nhấp nháy, chấm
  lại bằng trang mới): `results/ab/README.md` §4.6. Chi tiết: `results/ab/README.md` §4, `results/ab-zoom/README.md`,
  `results/e4/e4.json`, `results/enhance/enhance_eval.json`.
- **Giờ GPU thật** (so ngân sách §4): phase 10 2,85 (trần ~3); phase 11 2,49 (trần ~2, vượt ~0,5); phase 13 ~1,7 gồm cả hai
  vòng (vòng 1 1,03 trong trần ~1,25; vòng 2 ~0,67 trong trần riêng 45 phút); phase 14 + xuất/đo ms: không đo.

**Việc còn lại sau phase 14:** phase 15–18 theo `plan.md` (15 IOT-CV tích hợp code, 16 IOT-CV chạy model thật, 17 IOT-CV
trỏ về XLA, 18 báo cáo + một lệnh tái lập + PR cuối). Chưa làm. Phase 15–16 cần biết quyết định ở `results/ab/README.md`
§4.6: enhancer chưa đạt nên mặc định không có ảnh rõ khuyến nghị; bật thử hay dùng cổ điển là quyết định của Owner.

## 8. Chạy tiếp sau khi bật máy (tạm dừng lần 2: 18/11 22:15)

Mọi thứ đã commit + push; trọng số, nhãn giả, run ở `D:\DevCache\runs\xla\` (ổ D, không mất khi tắt máy). Thứ tự làm tiếp:

1. Phiên Owner: `git -C D:\XLA status` + `git worktree list`, xác nhận worktree `D:\XLA-wt\detect` và `D:\XLA-wt\restore`
   sạch và trùng `origin/lane/*`; khoá GPU trống.
2. Song song: lane detect làm tiếp phase 11 theo `D:\XLA-wt\detect\results\e2\RESUME.md`; lane restore làm tiếp phase 13
   vòng 2 theo `D:\XLA-wt\restore\results\restore\RESUME.md`. Hai lane dùng chung GPU qua `gpu_lock`. Brief gốc vẫn đúng:
   §1–6, §6b của file này; lệnh chủ dự án: không chờ/hoãn/để nợ vì CPU-nhiệt.
3. Hai lane xong (Accept exit 0, push) → phase 14: gộp `lane/restore`, `lane/detect` vào nhánh tích hợp (detect-export,
   abeval, classic, phase 10 đã gộp), chép `detector.onnx`/`enhancer.onnx` từ worktree vào bundle của nhánh tích hợp,
   `validate` từng bundle, `pytest -m "not gpu"`, rồi chạy các lệnh ở `results/ab/README.md` (enhance_eval all, ab_page
   build), mở `results/ab/ab-test.html` cho chủ dự án chấm (~10 phút), ab_score, README kết luận, PR #2 → CI xanh → merge.

## 9. Quyết định Owner trong lúc chạy phase 11 (18/11 tối)

- **Mốc cổng parity của detector = `.pt` với `rect=False`** (cùng letterbox vuông 1280×1280 như graph theo
  `docs/hop-dong-model.md` §2), không phải `.pt` mặc định (`rect=True`, 1280×992). Bản ONNX của trò r1 khớp mốc này Δ 0,0000;
  so với `rect=True` lệch −0,034, đổi dấu giữa các model (E1 D2 +0,021, r2 −0,016, thầy −0,005) nên là nhiễu của 100 khung
  dev-val. Hợp đồng không đổi; `.pt rect=True` và `full_edge16` giữ trong `parity.json` làm số tham khảo.
- Phase 11 dừng ở vòng 2 (r2 kém r1 trên dev-val), model ship = trò r1.
- Từ 23:05 mọi lần train còn lại chạy 4 worker (chủ dự án yêu cầu đẩy nhanh); lỗi bộ nhớ thật mới quay về 2.

## 10. Phase 15–18 (19/11/2026, một phiên, không sub-agent)

- **15 (IOT-CV code):** `backend/vision/{model_bundle,detector,enhancer,pipeline,runtime,jpeg_tables}.py`, `/api/video/stream?variant=enhanced`
  (503 `enhancer_unavailable` khi không có model), sự kiện `vision.model_rejected` / `vision.person_detected`, nút "Ảnh rõ" trên web
  (tắt mặc định), khung giả q12 nén bằng DQT camera thật. ORT CUDA trên Windows cần thêm `site-packages/nvidia/*/bin` vào PATH và đọc
  provider SAU một lần chạy thử (không thì lặng lẽ về CPU). Backend 411 test, frontend 278 test xanh.
- **16 (chạy thật):** bench chính thức `results/iotcv/bench-20261119-022705.json` (GPU 72 °C): nhịp thật tổng p50 37,5 / p95 54,1 ms,
  detector p95 33,4 (vượt 50 / 30), enhancer p95 19,0 (đạt), bỏ 0,3 % khung; hết tốc lực 23,6 khung/s, tổng p95 44,9, detector p95
  28,1 (đạt). Nguyên nhân vượt: GPU hạ xung giữa hai khung camera; ngân sách giữ nguyên, `tests/integration/test_iotcv_replay.py::test_ngan_sach_ms`
  đỏ có chủ đích tới khi đo lại sau bước người "Prefer maximum performance" (plan.md Human steps). Golden GPU hai bundle đạt; web kiểm
  ở 1707×900 cả ba trạng thái (gốc có box, Ảnh rõ, về gốc).
- **17 (trỏ về XLA):** `plans/ai/ai-phase-0{1..4}*.md` có khối "Đã thay thế", `ml/README.md` mới, 353 tham chiếu phân loại ở
  `results/iotcv_refs.json`; `ml/scripts/*.py` chưa xoá (chờ chủ dự án duyệt).
- **18:** `scripts/reproduce.py --check` chấm lại 15 tệp dự đoán (lệch 0) + so sha256 9 bảng/hình; báo cáo, README, slide viết
  lại; PR #3 + Release `models-v1.0.0`.
