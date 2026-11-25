# AI v2 phase 10 — classic thành ONNX `enhancer-v2.0.0`

Ngày 25/11/2026, lane `classic` (worktree `D:\XLA-wt\classic`, nhánh `lane/classic`). Plan:
`plans/261125-ai-v2/phase-10-classic-onnx.md`. Hợp đồng: `docs/hop-dong-model.md` §3, §4, §12 (không đổi).

```
python -m xla.enhance.export_classic --params configs/enhance_classic.yaml --out models/release/enhancer-v2.0.0 --json
python -m xla.enhance.export_classic hybrid --config configs/restore_hybrid.yaml --json     # bước 4 (lai, có trần)
python -m pytest tests/enhance/test_classic_onnx.py                                          # Accept
```

Số đo đầy đủ: `export.json` (bundle), `hybrid.json` (bước 4). Bundle trên đĩa: `models/release/enhancer-v2.0.0/`
(`enhancer.onnx` ngoài git, `meta.json` + `golden/` trong git). `onnx_sha256` 83d96e8b…477b, `classic_params_sha256`
2b1addee…3730 (= sha256 của `configs/enhance_classic.yaml`).

## 1. Graph

`numpy xla/enhance/classic.py` vẫn là SSOT. `xla/enhance/export_classic.py` dựng lại đúng năm tầng (cùng thứ tự
`classic.STAGES`: khử khối → khử hồng → bù tối góc → khử mờ → làm nét) từ cùng `classic.load_params`, xuất TorchScript
opset 17, fp32, IO đúng §3 (`frame` uint8 [480, 640, 3] BGR → `frame_out` uint8 [480, 640, 3], cột `[0, 16)` lặp cột 16).

| Phép của classic | Trong graph | Vì sao |
|---|---|---|
| `cv2` uint8/int16 (`addWeighted`, `multiply(scale)`, `subtract`, `add` có bão hoà) | float rồi `Round` (nửa về chẵn = `cvRound`) + `Clip` đúng chỗ | tái tạo từng lần làm tròn |
| `np.percentile` (phân vị 99,9 / 0,5 trên lưới 1/4) | `TopK` lấy 2 phần tử kề + nội suy như `numpy._lerp` | `kthvalue`/`quantile` không xuất được ở opset 17 |
| `if lo_out < floor` (kéo giãn mức đen) | `Where` | không Python `if` lúc trace |
| `break` theo `countNonZero` của khử hồng | bỏ hẳn | lượt có cổng toàn 0 là phép đồng nhất (α = 0 ⇒ Δ = 0, bơm chi tiết 0) |
| erode / dilate | `MaxPool` | biên bỏ qua như cv2 |
| `blur` / `boxFilter`, `resize INTER_AREA` ×1/4 | `AveragePool` trên ảnh đệm phản chiếu | đệm `BORDER_REFLECT_101` = một `Gather` chỉ số hằng mỗi trục: `F.pad` reflect sinh `If` + pads tính ra (mất shape tĩnh) |
| `GaussianBlur` | tổng lát dịch có trọng số, hàng rồi cột (như `sepFilter2D`), cạnh nhân như cv2 (8U: σ·3·2+1, float: σ·4·2+1) | `Conv`/`MatMul` có thể chạy TF32 trên ORT CUDA EP (~1e-3 tương đối, đủ đổi làm tròn uint8); bản `Conv` tách trọng số đúng TF32 đã thử: parity như nhau, chậm hơn 3 ms (§3) |
| `resize INTER_LINEAR` ×4 | `Resize` linear half-pixel | |

Graph không có `If`/`Loop`/`Scan`/`Conv`/`MatMul`/`Gemm` (test kiểm). fp16 không xuất: phép nguyên của cv2 cần fp32 — ví dụ
cổng khử hồng tính `max(R, B)·25,5 − 5865`, mà fp16 ở 5865 chỉ có bước 4,0, mất cả phép làm tròn uint8.

## 2. Parity từng tầng (đo trong ORT)

50 khung dev cách đều (`export_onnx.dev_frames(50)`), khác khung trace (`20261102-100941_002_00390.jpg`) và khác 3 khung
golden. Mỗi tầng chạy graph riêng (`planes` uint8 [3, 480, 624] → `planes_out`), đầu vào = ra numpy của tầng trước (sai
số không cộng dồn). Sai khác tuyệt đối thang 0–255; "lệch" = tỉ lệ điểm khác numpy ở khung tệ nhất. Dung sai Accept:
MAE ≤ 1,0, lớn nhất ≤ 8.

| Tầng | ORT CPU MAE TB / tệ nhất | CPU max | CPU lệch | ORT CUDA MAE TB / tệ nhất | CUDA max | CUDA lệch |
|---|---|---|---|---|---|---|
| deblock | 0 / 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| magenta | 0,00028 / 0,0024 | 1 | 0,24 % | 0,00028 / 0,0024 | 1 | 0,24 % |
| vignette | 0 / 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| dehaze | 0,0012 / 0,0144 | 1 | 1,44 % | 0,00059 / 0,0069 | 1 | 0,69 % |
| sharpen | 0,000002 / 0,000016 | 1 | 0,002 % | 0,000002 / 0,000016 | 1 | 0,002 % |

Đủ graph (khung nguyên → khung nguyên):

| So | MAE TB | MAE tệ nhất | max | lệch tệ nhất |
|---|---|---|---|---|
| ORT CPU ↔ numpy | 0,0020 | 0,0179 | 8 | 1,69 % |
| ORT CUDA ↔ numpy | 0,0012 | 0,0100 | 8 | 0,98 % |
| ORT CUDA ↔ ORT CPU (dung sai golden IOT-CV §4) | 0,0013 | 0,0155 | 2 | 1,47 % |
| golden 3 khung, ORT CPU ↔ numpy | 0,0011 | 0,0017 | 6 | 0,14 % |

Sai số còn lại: khử hồng lệch ≤ 1 mức ở ≤ 0,24 % điểm (cv2 `GaussianBlur` 8U tính dấu phẩy tĩnh, graph tính float);
khử mờ lệch ≤ 1 mức ở ≤ 1,44 % điểm (A, t, mức đen tính fp32, numpy tính một phần fp64). Lệch lớn nhất 8 của đủ graph là
vài điểm lẻ: 1 mức lệch ở khử hồng bị khử mờ nhân 1/t (t sàn 0,4) rồi làm nét khuếch đại; MAE vẫn ≤ 0,018. Không tầng
nào vượt 1,0 nên không phải viết lại tầng nào.

Nhánh dữ liệu trên 50 khung dev: khử mờ không kéo giãn mức đen ở cả 50 khung (`lo_out ≥ floor`); cổng khử hồng mở ở 46,
đóng ở 4. Phía nhánh còn lại (kéo giãn mức đen) không có trong dev, nên test CI dựng khung tổng hợp phủ đủ hai phía của
cả hai nhánh, và một graph ghim cứng nhánh mức đen (mô phỏng Python `if` lúc trace) phải ĐỎ trên khung `dark`
(MAE 4,78, max 7 — khung `haze` chỉ lệch MAE 0,88 nên không dùng làm khung bắt lỗi).

## 3. ms/khung

Máy: AMD Ryzen 9 7945HX, RTX 4060 Laptop GPU, onnxruntime-gpu 1.24.4, torch 2.6.0+cu124, opencv 5.1.0, Windows 11.
`measure_ort_ms`: warmup 30, đo 300 lần `sess.run` (gồm chép vào/ra), một khung dev. `% Processor Performance` lúc đo CPU
164–171 % (không hạ xung). GPU lúc vào đo: 86 °C, 30 W, SM 2 145 / 3 105 MHz, P0 (`export.json` `gpu_state_before`).

| Đường chạy (bundle cuối) | p50 ms | p95 ms |
|---|---|---|
| ORT CUDA EP, A/B sạch (`meta.ms_per_frame`, `ms_ab.json`) | 19,3 | 25,3 |
| ORT CPU EP, 4 luồng (`meta.ms_per_frame_cpu`) | 29,3 | 30,6 |
| numpy classic, cv2 4 luồng (tham chiếu) | 9,6 | 10,2 |

**A/B sạch với NAFNet v1** (26/11, GPU rảnh, `python -m xla.enhance.export_classic ab-ms --a models/release/enhancer-v2.0.0
--b D:\XLA\models\release\enhancer-v1.0.0`): hai session ORT CUDA trong cùng tiến trình, cùng khung dev, warm-up 30 lần
mỗi model, rồi 10 vòng xen kẽ (vòng chẵn v2 → v1, lẻ v1 → v2), mỗi khối 5 lần làm nóng + 100 lần đo, giữ `gpu_lock` suốt
lượt (95,5 s). Mỗi vòng ghi `gpu_state` và `% Processor Performance` lúc đầu và lúc cuối (`ms_ab.json` `rows`). GPU 66–68 °C,
CPU 172–188 %, không hạ xung.

| Bundle | p50 ms | p95 ms | n |
|---|---|---|---|
| enhancer-v2.0.0 (classic ONNX) | 19,27 | 25,35 | 1 000 |
| enhancer-v1.0.0 (NAFNet fp16) | 20,61 | 27,89 | 1 000 |
| hiệu v2 − v1 | −1,34 | −2,55 | |

Thứ tự trong vòng ảnh hưởng mạnh: mỗi vòng bắt đầu sau 2 s đọc `% Processor Performance`, GPU xuống xung (SM 270–945 MHz
lúc vào vòng, 1 140–1 665 MHz lúc ra), nên model chạy trước chịu phần tăng xung. Trung vị p50 theo vị trí: chạy trước v2
22,2 / v1 21,9 ms (ngang nhau); chạy sau, GPU đã nóng xung, v2 16,2 / v1 19,9 ms. Xen kẽ ABAB triệt phần này trong số gộp.
Hiệu p50 từng vòng dao động −7,8 … +3,3 ms tuỳ thứ tự. Kết luận: classic ONNX không chậm hơn NAFNet v1 (nhanh hơn
1,3 ms p50, 2,5 ms p95), khác với số cao 26,7–30,1 ms đo lúc GPU bị các lane khác tranh. Số 15,4 / 18,4 ms của lần xuất là
lúc GPU vừa xong job khác, đang xung cao, nên không phải số đại diện; `meta.ms_per_frame` đã thay bằng số A/B gộp ở trên
(`ms_per_frame_source` ghi số cũ).

Ba cách dựng graph đã đo trên ORT CUDA (parity như nhau, MAE ≤ 0,018):

| Graph | Nút tính | p50 / p95 ms | Đo |
|---|---|---|---|
| Gauss = lát dịch, đệm = lát + lật + nối | 795 | 26,9 / 27,6 | cùng lần giữ khoá với dòng dưới |
| Gauss = `Conv` tách đúng TF32, đệm = lát + lật + nối | 627 | 30,1 / 31,7 | như trên |
| **Gauss = lát dịch, đệm = một `Gather` mỗi trục (bundle)** | 703 | 15,4 / 18,4 (A/B sạch: 19,3 / 25,3) | lần giữ khoá khác, GPU đang ở xung cao |

Hai dòng đầu là A/B sạch. Dòng cuối không cùng lần đo với hai dòng trên: lúc đó GPU vừa chạy job khác nên đang ở SM
2 145 MHz (hai lần trước bắt đầu ở 780 MHz), nên phần giảm từ 26,9 xuống 15,4 ms là gộp cả việc bỏ `Concat`/`Slice`
của đệm (profile ORT: ~4,5 ms/khung) lẫn xung GPU; chưa tách được hai phần. Số chính thức của bundle là số A/B sạch ở trên. Graph bị chi phối bởi số lần gọi kernel (~700 nút nhỏ trên ảnh 480×624; profile: tổng thời gian kernel
14,1 ms trong 30 ms/lần ở bản `Conv`), không phải phép tính.

## 4. Nhấp nháy và chỉ số dev

`flicker_ratio` (`xla.eval.temporal`) gộp trên mọi 60 chuỗi dev liền kề (1 086 khung):

| | flicker_ratio (so khung gốc) |
|---|---|
| numpy classic | 2,19312 |
| ONNX (ORT CPU) | 2,19307 |

ONNX nhấp nháy đúng như numpy (lệch 0,00005). Classic tự nó tăng nhấp nháy ~2,2× (khử mờ tăng tương phản nên |Δ| giữa
khung tăng theo), như mọi checkpoint NAFNet v1 (1,65–2,0).

Trên 50 khung dev parity (ONNX = numpy tới 4 chữ số): dark channel trung vị 141,5 → 56,0 (giảm 60,4 %), % điểm hồng
0,70 % → 0,060 % (giảm 91,4 %). `meta.metrics` (tập test thật) để `null`: lane không đụng khung test, phase 14 đo.

## 5. Bước 4 — lai classic → NAFNet-w16 (trần 30 phút GPU)

`python -m xla.enhance.export_classic hybrid --config configs/restore_hybrid.yaml` (số đo: `hybrid.json`).

- Đầu vào mạng = `classic(D2)` dựng sẵn cho 3 006 cặp train + 487 cặp val (JPEG q100 4:4:4, `XLA_DATA_ROOT/v2_classic_d2`,
  30 s CPU, 4 tiến trình), đích = D0. Gọi `xla.enhance.train_net.train` có sẵn (không sửa `train_net.py`): NAFNet-w16
  (`VARIANTS.nafnet_w16`, 1,14 triệu tham số, khởi tạo ngẫu nhiên), batch 8, patch 256, lr 1e-3, AMP bf16.
- Smoke 200 bước đo 0,188 s/bước → còn chỗ cho 7 400 bước (trần − smoke − dự phòng val cuối 120 s − dự phòng đo 240 s);
  chạy tiếp từ `last.pt` tới 7 600 bước, 0,151 s/bước. Val tổng hợp cuối (487 cặp): PSNR 19,51 → 24,60 dB, SSIM 0,736 → 0,812.
- Đo dev mỗi checkpoint: `mạng(classic(khung))` trên 50 khung dev parity (dark channel trung vị, % hồng) + flicker trên mọi
  60 chuỗi dev liền kề (1 086 khung). Luật "hơn classic": không kém ở cả ba chỉ số và tốt hơn ít nhất một.

| | dark channel trung vị (gốc 141,5) | % điểm hồng (gốc 0,70 %) | flicker_ratio |
|---|---|---|---|
| classic | **56,0** | 0,060 % | 2,193 |
| lai bước 2 400 | 84,0 | **0** | 2,176 |
| lai bước 4 800 | 103,0 | **0** | 2,067 |
| lai bước 7 200 | 111,0 | **0** | 1,859 |
| lai bước 7 600 | 111,5 | **0** | **1,848** |

Kết quả: **không đạt trong trần**. Mạng học theo đích D0 nên trả lại phần lớn màn mờ classic đã khử (dark channel tăng
56 → 84–111,5): hết hồng và bớt nhấp nháy nhưng kém classic ở chỉ số chính. Không đo tổng ms, không xuất
`enhancer-v2.1.0`. Giờ GPU đã dùng (thời gian giữ khoá): smoke 45,7 s + huấn luyện 1 149,0 s + đo dev 306,7 s =
**1 501,4 s = 25,0 phút** ≤ 30 phút (chờ khoá không tính: 1 644 + 1 342 + 2 168 s).

## 6. Test (Accept)

`python -m pytest tests/enhance/test_classic_onnx.py`: 15 test (12 CPU chạy ở CI + 3 `data` cần bundle và khung thật).
CPU: thứ tự tầng khớp `classic.STAGES`; phân vị `TopK` = `np.percentile`; Gauss = `cv2.GaussianBlur`; chữ ký §3, opset 17,
không `If`/`Loop`/`Scan`/`Conv`/`MatMul`/`Gemm`, có `Where` + `TopK`; parity ORT CPU trên khung tổng hợp khác khung trace,
phủ hai phía nhánh mức đen và cổng khử hồng; graph ghim cứng nhánh mức đen thì parity đỏ; đổi `dehaze.omega` trong yaml mà
không xuất lại thì `stale_reasons` báo và graph cũ lệch numpy quá dung sai. Data: `validate_bundle` sạch,
`classic_params_sha256` + `camera_profile_sha256` khớp file hiện tại, golden ORT CPU ↔ numpy trong dung sai, báo cáo đủ
năm tầng cho cả ORT CPU và CUDA.
