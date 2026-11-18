# Phase 13 vòng 2 — trạng thái lúc tạm dừng (lần 2: 18/11/2026 22:15; lần 1: ~19:39; chủ dự án tắt máy)

> **ĐÃ XONG 18/11/2026 23:34.** Đã chạy hết các bước dưới đây: train tới 4 500, select chọn step_004500, bundle
> `enhancer-v1.0.0` = nafnet_w16 (p95 21,0 ms), validate `[]`, Accept 7/7 exit 0. Kết quả: `README.md` mục 8. Phần dưới
> giữ làm bản ghi.

Vòng 1 (width32) đã xong, xem `README.md`; bundle `models/release/enhancer-v1.0.0` hiện vẫn là **width32** (p95 82,1 ms,
Accept FAIL ở `test_bundle_ms_p95`). Bản sao width32 ngoài repo: `D:\DevCache\tmp\enhancer-width32-v1.0.0\` (cả
`enhancer.onnx` + `export_width32.json`).

## Đã làm ở vòng 2

1. **Đo tốc độ biến thể** (`speed_round2.json`; 480×624 fp16 ORT CUDA EP; CPU 134 %/128 % trong/sau khi đo, cờ
   `throttled: true` vì cổng trước khoá thấy 22 %):

   | Biến thể | Tham số | Khởi tạo | p50 | p95 |
   |---|---|---|---|---|
   | nafnet_tiny (width 32, 1 khối/tầng) | 4,5 M | k khối đầu SIDD-w32 | 34,9 ms | 36,9 ms |
   | nafnet_lite (width 32) | 6,8 M | k khối đầu SIDD-w32 | 54,8 ms | 73,9 ms |
   | **nafnet_w16** (width 16, 1 khối/tầng) | 1,1 M | ngẫu nhiên | 19,0 ms | 32,9 ms |

   Dò thêm (`D:\DevCache\runs\xla\restore\probe\ln_probe.json`, cùng lúc): tiny LN fp32 32,6/34,9 ms; tiny LN fp16 chia
   tỉ lệ 31,1/32,2 ms; width16 14,2/16,4 ms. Chi phí nằm ở tầng độ phân giải đủ (∝ width), nên tiny width 32 không thể
   xuống dưới 30 ms ở máy này. Một bản w16 nhiều khối hơn (enc 1,1,2,4, mid 4) đo p95 64,5 ms → bỏ
   (`D:\DevCache\runs\xla\restore\speed\speed_round2_w16deep.json`).
2. **Chốt huấn luyện nafnet_w16** (biến thể `VARIANTS["nafnet_w16"]` trong `xla/enhance/net.py`): khởi tạo ngẫu nhiên
   (width khác 32 nên không nạp được trọng số công bố), batch 16, lr 1e-3 (công thức gốc NAFNet), cùng cặp D2→D0, cùng
   loss, cùng luật chọn checkpoint. Trần GPU vòng 2: 45 phút (speed + smoke + train).
3. **Smoke nafnet_w16** (`D:\DevCache\runs\xla\restore\nafnet_w16-smoke\smoke.json`): 500 bước, 0,454 s/bước, VRAM
   3,46 GB, 257 s GPU; val tổng hợp PSNR 10,22 → 19,99 dB, SSIM 0,255 → 0,767. Số bước theo ngân sách: **4 500**
   (ckpt/val mỗi 1 000).
4. **Train nafnet_w16** (`D:\DevCache\runs\xla\restore\nafnet_w16-train\`): lần 1 dừng ở bước 2 200 (979 s GPU);
   22:08:35 chạy tiếp từ `last.pt` (workers 2 vì máy ảo VMware giữ 16,5 GB RAM; trước đó chờ khoá GPU vì detect
   phase 11 giữ từ 21:32), dừng lần 2 ngay sau checkpoint bước **3 000** (22:14:48, PID 29032, ~373 s GPU). Hiện có
   `last.pt` = bước 3 000 (đã đọc lại), `step_001000.pt`, `step_002000.pt`, `step_003000.pt`. Val tổng hợp: bước
   1 000 PSNR 20,74 / SSIM 0,782; 2 000 PSNR 22,03 / 0,803; 3 000 PSNR 22,17 / 0,805 (width32 vòng 1 cuối:
   22,83 / 0,813). `train_summary.json` và `runs.nafnet_w16-train` trong `train.json` chỉ ghi khi chạy hết 4 500 bước.

Giờ GPU vòng 2 tới lúc dừng: speed 92 + 82 s, dò LN ~120 s (ước), smoke 257 s, train 979 + ~373 s ≈ 32 phút.

## Chạy tiếp (sau khi bật máy, đúng thứ tự)

```bash
cd /d/XLA-wt/restore
export UV_CACHE_DIR='D:\DevCache\uv' UV_PROJECT_ENVIRONMENT='D:\DevCache\venvs\xla' TORCH_HOME='D:\DevCache\torch' \
  XLA_DATA_ROOT='D:\DevCache\datasets\xla' PYTHONUTF8=1 TMP='D:\DevCache\tmp' TEMP='D:\DevCache\tmp' XLA_ALLOW_THROTTLED=1
PY=/d/DevCache/venvs/xla/Scripts/python.exe
# 1) train tiếp (tự resume từ last.pt bước 3000; giữ nguyên mọi tham số để lịch lr cosine khớp; workers ≤ 2 khi VM chạy)
$PY -m xla.enhance.train_net train --candidate nafnet_w16 --run-id nafnet_w16-train --batch 16 --lr 1e-3 \
    --max-steps 4500 --ckpt-every 1000 --workers 2 --json
# 2) chọn checkpoint trên dev thật (exit 1 chỉ khi không có checkpoint; luật dự phòng flicker có sẵn)
$PY -m xla.enhance.train_net select --run-id nafnet_w16-train --json
# 3) xuất bundle thay width32 (ckpt = chosen_path trong results/restore/train.json khoá select)
$PY -m xla.enhance.export_onnx --ckpt "$($PY -c "import json;print(json.load(open('results/restore/train.json',encoding='utf-8'))['select']['chosen_path'])")" \
    --out models/release/enhancer-v1.0.0 --json
$PY -m xla.contract validate models/release/enhancer-v1.0.0
# 4) Accept + bộ test + lint
$PY -m pytest tests/enhance/test_enhancer_export.py -q -p no:cacheprovider
$PY -m pytest -m "not gpu" -q -p no:cacheprovider
$PY -m ruff check .
```

Hoặc chạy nền một lệnh cho 1–3: `$PY D:/DevCache/runs/xla/restore/driver/resume_round2.py` (đã có `--workers 2`; từ gốc worktree, cùng biến
môi trường). Job dài: chạy tách tiến trình, chờ trong lượt; dừng tiến trình theo đúng PID.

Sau đó: xem 3 khung golden trước/sau, cập nhật `README.md` (bảng width32 vs w16: chỉ số dev, flicker, ms; flicker
1,65 của width32 là kết quả thật, không đổi định nghĩa; flicker của w16), commit, push `lane/restore`, báo cáo.

## Quyết định đã chốt

- Bỏ nafnet_tiny width 32 và nafnet_lite: p95 36,9 / 73,9 ms > 30 ngay cả riêng phần mạng.
- nafnet_w16 khởi tạo ngẫu nhiên, batch 16, lr 1e-3, 4 500 bước, ckpt mỗi 1 000.
- `select` lấy kiến trúc từ checkpoint (biến thể có `arch_kwargs` riêng); khởi tạo ngẫu nhiên thì bỏ dòng `pretrained`.
- `train.json`: khoá gốc = lần mới nhất, `runs.<run_id>` giữ mọi vòng (width32: `nafnet-smoke`, `nafnet-train`).
- Trần Accept 30 ms giữ nguyên; nếu bundle w16 đo p95 > 30 ms thì ghi số đo, coi là Kill (c), không làm giả.
