# Phase 10 — Classic thành ONNX `enhancer-v2.0.0` (+ lai có giới hạn) (lane classic)

Lane: classic
Depends-on: 01
Effort: L
Repo: XLA, worktree `D:\XLA-wt\classic`, nhánh `lane/classic` từ `feat/ai-v2`
Owns: `xla/enhance/export_classic.py`, `configs/restore_hybrid.yaml`, `models/release/enhancer-v2.0.0/**`, `models/release/enhancer-v2.1.0/**`, `results/v2/classic/**`, `tests/enhance/test_classic_onnx.py`
Creates: `xla/enhance/export_classic.py`, `configs/restore_hybrid.yaml`, `tests/enhance/test_classic_onnx.py`
AX: AX-01 `python -m xla.enhance.export_classic --params configs/enhance_classic.yaml --out models/release/enhancer-v2.0.0 --json`
Accept: `pytest tests/enhance/test_classic_onnx.py`

## Mục tiêu

Đưa `xla.enhance.classic` (thắng NAFNet 24/25, ~9 ms CPU) tới IOT-CV qua hợp đồng enhancer §3 **có sẵn**: một graph ONNX
dựng từ cùng tham số, parity với numpy **đo trong ORT**, để `Enhancer` IOT-CV chạy không sửa code. numpy `classic.py` vẫn
là SSOT. Không sửa `xla/enhance/export_onnx.py` hay `train_net.py` (chỉ gọi hàm có sẵn; cần đổi → dừng, báo).

## Đầu vào

`xla/enhance/classic.py` (`load_params` :89, `deblock` :227, `demagenta` :247, `flat_field` :296, `dehaze` :316 — nhánh
`if lo_out < floor` :358, `sharpen` :372, `enhance` :450), `configs/enhance_classic.yaml`, `xla/enhance/export_onnx.py`
(golden, MAE ≤ 1,0 — chỉ đọc), `docs/hop-dong-model.md` §3, `xla/eval/{enhance_eval,temporal}.py`.

## Bước

1. Parity từng tầng: viết mỗi tầng bằng op xuất được ở opset 17 — box/Gaussian = conv cố định, erode/dilate = max-pool,
   percentile = `topk` + nội suy tuyến tính (red-team 25/11: `kthvalue`/`quantile` không xuất được ở opset 17), mọi nhánh phụ
   thuộc dữ liệu (`if lo_out < floor`, `float(np.mean(...))`, cổng `countNonZero`) = `where`/tensor, không Python `if`.
   Xuất từng tầng, đo **trong ORT** (CPU + CUDA) so numpy trên 50 khung dev **khác** khung dùng để trace: MAE, max. Tầng
   nào > 1,0 MAE: viết lại bằng op tương đương, ghi sai số còn lại (không đổi `classic.py`).
2. Ghép graph: `frame` uint8 [480,640,3] → crop → 5 tầng → pad dải mép (§3) → `frame_out`; opset 17; fp32 (fp16 nếu parity
   vẫn ≤ 1,0). Meta `network: "classic"`, `classic_params_sha256`.
3. Bundle `models/release/enhancer-v2.0.0`, golden 3 khung; `xla.contract validate`; p50/p95 ORT CUDA và CPU (`machine`);
   nhấp nháy `temporal.py` so numpy classic.
4. Lai (tuỳ chọn, trần **2 giờ GPU**, Deferred 16): NAFNet-w16 sửa màu sau classic, gọi `python -m xla.enhance.train_net`
   có sẵn với `configs/restore_hybrid.yaml` (đầu vào = output classic trên cặp D2). Ứng viên A/B chỉ khi hơn classic trên dev
   (dark channel trung vị, % hồng, nhấp nháy) và tổng ms ≤ 30. Qua thì xuất `models/release/enhancer-v2.1.0` (`network: "hybrid"`).
5. `results/v2/classic/README.md`: parity từng tầng (ORT), ms, nhấp nháy, kết quả bước lai, giờ GPU.

## Tiêu chí xong

Accept xanh: parity golden (ORT) MAE ≤ 1,0 / max ≤ 8 với `classic.enhance` trên khung không dùng để trace; `classic_params_sha256`
khớp yaml; graph đúng IO §3; test đỏ khi đổi một tham số yaml mà không xuất lại; test đỏ nếu graph còn chứa nhánh đã bị trace
cứng (so output trên một khung kích hoạt nhánh kia).

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Nhánh dữ liệu bị trace cứng, PyTorch đúng mà ONNX sai | high | medium | Parity đo trong ORT trên khung khác khung trace; test nhánh |
| Percentile / deblock không parity | medium | medium | `topk` + nội suy; A/B chấm đúng bundle xuất, không chấm numpy |
| Hai cài đặt trôi nhau | medium | medium | Test sha tham số + parity golden trong CI CPU |

Rollback: IOT-CV giữ `ENHANCER_ONNX` cũ; xoá bundle v2.
