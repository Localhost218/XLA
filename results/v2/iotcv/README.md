# IOT-CV chạy bundle v2 thật (phase 15, plan 261125-ai-v2)

Ngày đo: 26/11/2026. IOT-CV nhánh `feat/ai-v2` (gộp `lane/iotcv` 98b8676 và `lane/iotcv-debt` f2c562d), máy RTX 4060
Laptop 8 GB. Accept: `$PY -m pytest tests/integration/test_iotcv_v2.py`.

```bash
# trong worktree/clone IOT-CV, venv CỦA IOT-CV (không phải venv XLA)
uv run --no-sync python scripts/vision/lay_model.py --from D:/XLA/models/release/<bundle>      # 3 bundle ship
uv run --no-sync python scripts/vision/kiem_bundle.py models/vision/<bundle> --json [--cpu]     # -> kiem_bundle.jsonl
UV_PROJECT_ENVIRONMENT=D:/Coding/IOT-CV/.venv uv run --no-sync python D:/XLA/results/v2/iotcv/bench_v1_v2.py \
  --iotcv D:/IOTCV-wt/iotcv --recording D:/Coding/IOT-CV/logs/recordings/20261102-100941 \
  --out D:/XLA/results/v2/iotcv/bench.json                                                     # -> bench.json
```

## 1. Bundle ship

Theo `results/ab-v2/decision.json` và Kill (b)–(d) của plan:

| Bundle | sha256 `.onnx` | Golden CUDA | Golden CPU |
|---|---|---|---|
| `detector-v1.0.1` (cùng ONNX v1.0.0, thêm `recommended_track` k1/n1/iou 0,3/max_gap 1) | `23b1da69165ffe05760bda23b4a41c858930e8a8fd8b0aebc039f1e0c5c59aef` | đạt (IoU ≥ 0,9998) | đạt |
| `verifier-v1.0.0` (ngưỡng 0,35) | `d566343a5487775b9f6334d9df0bb0a1d9c9b2d2b76e2b8bfa4738facc012d58` | đạt (Δscore ≤ 0,0003) | đạt |
| `enhancer-v2.0.0` (classic ONNX) | `83d96e8b0f628d0e6cc271ed9550220962de897cac2d57ed306a1db51cb5477b` | đạt (max_abs ≤ 2) | đạt |
| `enhancer-offline-v1.0.0` | — | **không ship** | — |

Không ship offline vì lượt cả khung của A/B mù: offline thắng classic 16/25 (Wilson [0,445; 0,798], cận dưới < 0,5),
dù thắng gốc 25/25 và qua hàng rào báo nhầm. IOT-CV để `ENHANCER_OFFLINE_ONNX` rỗng: `POST /api/flights/enhance-offline`
trả **503** `enhancer_offline_unavailable` (đã gọi thật, mục 3), web tắt nút dựng kèm lý do. Bước 4 của phase (chạy một
chuyến offline thật, `manifest.wall_s`) **bỏ qua**: không có bundle offline để chạy; test `test_manifest_offline_xong`
skip kèm lý do này.

Ba bundle chép vào cả worktree `D:\IOTCV-wt\iotcv\models\vision\` và clone chính `d:\Coding\IOT-CV\models\vision\`
(gitignored), sha256 hai nơi trùng bảng trên. Mặc định đường dẫn + sha256 trong `backend/config.py`,
`ENHANCER_DEFAULT_ON = true` (classic, quyết định chủ dự án; classic thắng gốc 25/25).

## 2. ms trực tiếp — `bench.json`

`bench_v1_v2.py`: nạp năm bundle trong **một** tiến trình, chạy bốn lượt xen kẽ ABBA (v1, v2, v2, v1), mỗi lượt warmup
30 khung rồi `replay_bench._realtime` (nhịp thật của bản ghi, ~7,2 khung/s) trên khung 284 → 884 của
`20261102-100941` (600 khung, như phase 09), có người xem "Ảnh rõ" và chỉ detector. Phân vị gộp mọi khung của hai lượt
mỗi cấu hình. v1 = `detector-v1.0.0` + `enhancer-v1.0.0` (track k = n = 3 như trước 25/11); v2 = `detector-v1.0.1` +
`verifier-v1.0.0` + `enhancer-v2.0.0` (track k1/n1). Không giữ khoá GPU XLA; khoá `free` lúc chạy.

**Máy lúc đo** (nvidia-smi mỗi 2 s suốt từng lượt): xung SM median **780–802 MHz** (tối đa 3 105), công suất median
19–22 W, 70–72 °C, P0/P4 — GPU hạ xung mạnh, như lúc đo phase 09 (780 MHz). Đây là số chính thức kèm điều kiện, không
đo lại chờ máy mát. Lúc chụp web (mục 3) GPU lên xung, detector p95 26 ms.

| p95 ms (p50) | v1 nhịp thật | v2 nhịp thật | v1 chỉ detector | v2 chỉ detector |
|---|---|---|---|---|
| detector | 63,3 (53,2) | 63,4 (57,4) | 63,3 (60,0) | 63,3 (59,6) |
| verifier | — | **8,9 (0,03)** | — | **9,4 (0,03)** |
| enhancer | 34,4 (29,7) | **19,3 (15,4)** | — | — |
| **tổng** | **99,1 (84,5)** | **89,5 (76,3)** | **64,0 (60,8)** | **72,4 (61,1)** |
| khung bỏ (2 lượt) | 7,8 % · 8,2 % | 4,2 % · 4,7 % | 0,5 % · 0,3 % | 0,3 % · 0,3 % |
| lỗi pipeline | 0 | 0 | 0 | 0 |

- **Trần +5 ms, luồng có "Ảnh rõ" (mặc định từ nay): đạt**, v2 − v1 = **−9,6 ms** p95. Classic ONNX nhẹ hơn NAFNet
  ~15 ms p95, bù hết verifier.
- **Luồng chỉ detector (người xem tắt "Ảnh rõ"): không đạt**, v2 − v1 = **+8,4 ms** p95 (trần +5). Phần thêm là
  verifier: p95 ~9 ms ở xung 780 MHz khi khung có hộp qua ngưỡng detector; p50 0,03 ms vì phần lớn khung không có hộp.
- So phase 09 (173,3 ms p95 cả hệ nhịp thật, trước khi IOT-CV sửa verifier theo bậc ở 78a16af): nay **89,5 ms**, verifier
  p95 76,3 → 8,9 ms. Cùng điều kiện xung (~780 MHz) nên so được.
- Từng lượt (tổng p95 nhịp thật): v1 98,2 · v2 87,9 · v2 91,3 · v1 99,4 ms; lệch giữa hai lượt cùng cấu hình ≤ 3,4 ms,
  nhỏ hơn chênh v1↔v2.

## 3. Web 1707×900 nền ngày (Chrome headless, hồ sơ mới) — `web_facts.json`

Backend worktree trên drone ảo (SITL headless, `CAMERA_FAKE=1`), web do backend phục vụ từ `frontend/dist` vừa build.
Ảnh ở `D:\IOTCV-wt\iotcv\plans\reports\` (vào repo IOT-CV cùng PR):

| Ảnh | Thấy |
|---|---|
| `261126-phase15-trang-bay-1707x900.png` | trang Bay vừa 900 px (`scrollHeight` = 900, không cuộn cột), radar vật cản + bàn phím lái hiện; "Ảnh rõ" **bật sẵn** (`aria-pressed=true`, `localStorage` trống, luồng `variant=enhanced`) |
| `261126-phase15-vision-model-1707x900.png` | detector v1.0.1, verifier v1.0.0, enhancer v2.0.0 đều "đang chạy" CUDA; Ảnh rõ offline "tắt" kèm lý do |
| `261126-phase15-lich-su-bay-classic-1707x900.png` | Lịch sử bay chuyến 02/11 10:10: gốc ↔ "ẢNH RÕ · enhancer-v2.0.0", ghi chú đọc phiên bản + mặc định từ trạng thái |
| `261126-phase15-lich-su-bay-offline-tat-1707x900.png` | chọn nguồn offline: nút dựng tắt, chip "Không dựng được: ENHANCER_OFFLINE_ONNX chưa đặt (tắt)", bên phải về ảnh rõ trực tiếp |

API thật: `GET /api/flights/enhance-offline?rec=20261102-100941` → 200 `bundle_version: null` + lý do;
`POST` → **503** `{"error":"enhancer_offline_unavailable", …}`.

## 4. Giới hạn

- Bench chỉ trên một đoạn 600 khung của một bản ghi, GPU hạ xung cả buổi đo; số tuyệt đối ở xung đầy đủ thấp hơn nhiều
  (detector ~24–26 ms lúc chụp web) nhưng chênh v1↔v2 là thứ trần +5 ms đo.
- Trang Lịch sử bay chụp ở khung #84 (trước cất cánh, sát đất), không có người trong khung.
- Clone chính `d:\Coding\IOT-CV\.env` còn ghim `DETECTOR_ONNX`/`ENHANCER_ONNX` (+ sha) của bundle v1: biến môi trường
  thắng mặc định, nên backend thật của chủ dự án vẫn chạy enhancer v1 cho tới khi bỏ các dòng đó (không sửa `.env` thay
  chủ dự án).

## 5. Tệp

`bench_v1_v2.py` (driver bench), `bench.json` (từng lượt + gộp + khối máy), `kiem_bundle.jsonl` (golden CUDA và CPU ×3),
`web_facts.json` (số đọc từ trang lúc chụp).

## 6. Kết quả

- Accept `$PY -m pytest tests/integration/test_iotcv_v2.py` trên clone chính `d:\Coding\IOT-CV` sau merge: **7 passed,
  1 skipped** (`test_manifest_offline_xong`: offline không ship). Trước merge, chính test config đỏ trên clone chính
  (`config.py` cũ không có `VERIFIER_ONNX`), tức test bắt được bản v1.
- PR IOT-CV NguyenTuKien/IOT-CV#61 (`feat/ai-v2` → `main`), CI lint/test/frontend xanh, merge kiểu squash `c82509a`
  (repo chặn merge commit trên `main`). Workflow deploy EC2 chạy theo lọc đường dẫn và kết thúc `skipped` (job `checks`
  không đủ điều kiện, như hai lần push trước).
