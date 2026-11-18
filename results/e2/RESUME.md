# Phase 11 (E2) — trạng thái khi tạm dừng lần 2, 18/11/2026 ~22:16 (chủ dự án tắt máy)

Lần dừng 1: 19:46, chạy tiếp 21:31. Lần dừng 2: 22:16. Lúc dừng lần 2 không job nào của lane detect chạy: tiến trình
`--round 2` (PID 24112) đang **chờ khoá GPU** (lane restore giữ từ 22:08), chưa bắt đầu nhãn giả vòng 2, đã bị dừng;
`D:\DevCache\runs\xla\e2\pseudo_r2\` rỗng, `r2\` chưa có.

## 1. Đã xong

| Bước | Kết quả | Giờ GPU thật |
|---|---|---|
| Tải YOLO11m COCO | `D:\IOT_Tools\downloads\yolo-weights\yolo11m.pt` (sha256 `d5ffc1a6…5b95`) | — |
| Smoke thầy batch 4 | VRAM 8,37 GB > 8 GB, tràn RAM chung, 1,15 s/batch → bỏ (`smoke_teacher_batch4.json`) | 0,046 h |
| Smoke thầy batch 2 | 4,46 GB, 0,64 s/batch, 0,28 h/epoch → 4 epoch vừa trần 75 phút (`smoke_teacher.json`) | 0,048 h |
| **Thầy YOLO11m @1280 batch 2, 4 epoch, D2** | dev-val 0,025 / 0,062 / 0,188 / **0,267** (epoch 4); test mAP50 0,218 [0,107–0,349] (đường `.pt`) | **1,196 h** (4 305 s) |
| Nhãn giả vòng 1 (GPU) | phát hiện thô + TTA của 2 ứng viên thầy trên 1 115 khung dev (47 chuỗi, cách < 1 s) | 0,157 h (405 s + 161 s) |
| Nhãn giả vòng 1 (CPU) | thầy chọn = **e1_d2** (YOLO11s D2 của E1): τ_cao 0,55, precision nhãn giả dev-val 1,0, recall 0,084 (9/107); YOLO11m: τ 0,45, P 1,0, R 0,075. Tập `D:\DevCache\datasets\xla\e2_r1`: 3 006 ảnh D2 + 578 khung dev ×2 (86 khung có hộp, 87 hộp) + 300 `neg` = 4 462 ảnh | — |
| Smoke trò r1 (YOLO11s từ E1 D2, batch 4) | 4,30 GB, 0,364 s/batch, 1 116 batch/epoch; ước 480 s/epoch (trung vị batch) hoặc 965 s (ngây thơ, có khởi động) | 0,055 h |
| **Trò r1 đủ** (YOLO11s từ E1 D2, 4 epoch, batch 4, tập `e2_r1`) | dev-val 0,250 / 0,265 / **0,379** / 0,339: tốt nhất 0,379 (epoch 3) > E1 D2 0,329; trung bình 3 epoch cuối **0,328** > E1 D2 0,229 → **cả hai cách đều tăng → làm vòng 2**. Trọng số `D:\DevCache\runs\xla\e2\r1\weights\best_devval.pt` (sha256 `7e374765…ade5`); CSV `results/e2/preds/r1_{test,neg,devval}.csv` | **0,602 h** (2 167 s) |
| Chấm sơ bộ (`--score`, CPU) | `results/e2/e2.json` với mốc, E1 D2, thầy; cache bootstrap ở `D:\DevCache\runs\xla\e2\score\cache\` | — |

## 2. Còn lại (theo thứ tự)

1. ~~Trò vòng 1~~ XONG (bảng trên).
2. **Vòng 2**: `--round 2` = nhãn giả bằng r1 + TTA (~3 phút GPU) → tập `e2_r2` → trò r2 4 epoch (~36 phút như r1). Chưa có gì
   của vòng 2: chạy lại nguyên lệnh (bước xong tự bỏ qua, train resume từ `last.pt` nếu sập giữa chừng).
3. So r2 với r1 (0,379 / 0,328). Cả hai cách đều tăng → vòng 3 (`--round 3`, vòng cuối); không → dừng.
4. `--score` (CPU, ~15 phút nhờ cache; chạy SAU khi train xong, không chạy hai job nặng cùng lúc) → `e2.json` chọn model
   ship theo dev-val trong {E1 D2, r1, r2, ...}.
5. Xuất nháp (`--skip-cuda`) để có `detector.onnx` → `--onnx-score` (metrics đường ONNX + ngưỡng F1 dev-val của đường ONNX) → xuất thật (CUDA, parity) với `metrics_bundle.json`.
6. Kiểm sha256 `detector.onnx` lần xuất thật = lần chấm ONNX (khác → chạy lại `--onnx-score`), validate `[]`, pytest `tests/detect/` (cả 2 test `gpu`), `-m "not gpu"`, ruff, README, commit, push.

## 3. Lệnh chạy tiếp (PowerShell, từ `D:\XLA-wt\detect`)

```powershell
. D:\XLA\scripts\env.ps1
$py = 'D:\DevCache\venvs\xla\Scripts\python.exe'
# hoặc tách tiến trình + log: D:\DevCache\runs\xla\e2\logs\run.ps1 -Log <log> <tham số selftrain>
# --round 1 ĐÃ XONG. Bước tiếp:
& $py -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round 2 --json   # r1 đã tăng dev-val: chạy
& $py -m xla.detect.selftrain --config configs/selftrain_e2.yaml --round 3 --json   # chỉ khi r2 tăng so với r1
& $py -m xla.detect.selftrain --config configs/selftrain_e2.yaml --score --json
# SHIP = e2.json["ship"]["weights"], CONF = e2.json["ship"]["conf_threshold_pt"]
& $py -m xla.detect.export_onnx --weights <SHIP> --out models/release/detector-v1.0.0 --version 1.0.0 `
    --conf <CONF> --metrics results/e2/e2.json-ship-metrics.json --train-run e2/<tên> --skip-cuda --json   # nháp
& $py -m xla.detect.selftrain --config configs/selftrain_e2.yaml --onnx-score models/release/detector-v1.0.0 --json
& $py -m xla.detect.export_onnx --weights <SHIP> --out models/release/detector-v1.0.0 --version 1.0.0 `
    --conf <metrics_bundle.json conf_threshold> --metrics results/e2/metrics_bundle.json --train-run e2/<tên> `
    --parity-out results/e2/parity.json --json                                                           # thật, CUDA
& $py -m xla.contract validate models/release/detector-v1.0.0
& $py -m pytest tests/detect -q ; & $py -m pytest -m "not gpu" -q ; & $py -m ruff check .
```

File metrics nháp: object có 5 khoá `map50, map50_95, precision, recall, fp_per_neg_frame` — lấy từ
`e2.json["models"][<ship>]["test_bundle_keys"]` ghi ra một file JSON.

## 4. Đường dẫn trên ổ D

- Thầy: `D:\DevCache\runs\xla\e2\teacher\weights\best_devval.pt` (epoch 4), `last.pt`, `devval_history.jsonl`, `run_info.json`, `train_done.json`, `preds\`.
- E1 D2 (thầy vòng 1 được chọn, khởi tạo trò): `D:\DevCache\runs\xla\e1\d2\weights\best_devval.pt`.
- Nhãn giả vòng 1: `D:\DevCache\runs\xla\e2\pseudo_r1\` (`raw_dets_teacher.json`, `raw_dets_e1_d2.json`, `pseudo.json`); tập `D:\DevCache\datasets\xla\e2_r1\` (`dataset.yaml`, `train.txt`, `train\images|labels`, `labels.cache`).
- Smoke: `D:\DevCache\runs\xla\e2\teacher_smoke\`, `D:\DevCache\runs\xla\e2\r1_smoke\`.
- Log: `D:\DevCache\runs\xla\e2\logs\`. Cache chấm: `D:\DevCache\runs\xla\e2\score\`.
- Trò r1: `D:\DevCache\runs\xla\e2\r1\` (`weights\best_devval.pt`, `last.pt`, `devval_history.jsonl`, `run_info.json`, `train_done.json`, `preds\`).
- Vòng 2 (chưa tạo): `D:\DevCache\runs\xla\e2\pseudo_r2\` (đang rỗng), `D:\DevCache\datasets\xla\e2_r2\`, `D:\DevCache\runs\xla\e2\r2\`.

## 5. Quyết định đã tự chốt (ghi lại vào README cuối)

1. **Thầy YOLO11m batch 2** (batch 4 tràn VRAM), 4 epoch (smoke 0,28 h/epoch, trần 75 phút; đo thật 71,8 phút), close_mosaic 1, cùng siêu tham số khác với E1.
2. **Thầy vòng 1 chọn trong {YOLO11m, YOLO11s D2 của E1}** bằng chất lượng nhãn giả trên dev-val (đạt precision ≥ 0,9 trước, rồi recall): YOLO11m 4 epoch chỉ đạt dev-val 0,267 < 0,329 của E1 D2; recall nhãn giả 0,075 so với 0,084. Thắng: e1_d2.
3. **ByteTrack gọi thẳng** (`BYTETracker.update`, tracker mới mỗi chuỗi, giữ trạng thái suốt chuỗi = `track(persist=True)`), phát hiện thô (GPU) tách khỏi track (CPU). `fuse_score: false`, `match_thresh 0.9`, `track_high 0.05`, `new_track 0.1`, `track_buffer 15`: với `fuse_score` mặc định, điểm người ~0,15 làm cost > 0,8, không ghép được track nào (thử trên 3 chuỗi).
4. **Gói `lap` không cài**: ultralytics tự cài nó lúc thử đầu, đã gỡ lại (`uv pip uninstall lap`); ghép Hungarian bằng `use_lap=False`.
5. Khung dev-val và khung dev cách khung dev-val < 0,3 s không vào train (428 khung bị loại vì đệm); chỉ split `dev` được đọc cho nhãn giả, test/khối test không bao giờ; khung `neg` vào train với nhãn rỗng (plan bước 2) → **báo nhầm `neg_indoor` và số gộp `fp_per_neg_frame` của model E2 lạc quan** (300 khung đó đã thấy lúc train); B2 đo trên `neg_flight` không bị ảnh hưởng.
6. Khung có hộp ≥ τ_cao bị bỏ (lẻ / track ngắn) không vào train (20 khung vòng 1). Nhận xét: precision ≥ 0,9 ép recall nhãn giả còn ~8 %, nên phần lớn người trong khung đích bị gán nền — rủi ro thiên lệch xác nhận của plan đang hiện thật; giữ đúng luật plan cho vòng 1, dev-val sẽ quyết có làm vòng 2.
7. Trò khởi từ YOLO11s D2 của E1 (không từ COCO) để vừa trần 45 phút; khung đích lặp 2 lần trong `train.txt`.
8. Tập nhãn giả tự chứa (ảnh D2 liên kết cứng vào `e2_r<K>`) để `labels.cache` không ghi đè cache của D2.
