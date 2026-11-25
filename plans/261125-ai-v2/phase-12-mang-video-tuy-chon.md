# Phase 12 — Tuỳ chọn: mạng video BasicVSR++ fine-tune D2 (lane sr)

Lane: sr
Depends-on: 03, 11
Effort: M
Repo: XLA, worktree `D:\XLA-wt\sr` (tiếp sau phase 11)
Owns: `xla/enhance/arch/basicvsrpp.py`, `xla/enhance/vsr.py`, `configs/vsr_x2.yaml`, `models/release/enhancer-offline-v1.1.0/**`, `results/v2/vsr/**`, `tests/enhance/test_video_sr.py`
Creates: `xla/enhance/arch/basicvsrpp.py`, `xla/enhance/vsr.py`, `configs/vsr_x2.yaml`, `tests/enhance/test_video_sr.py`
AX: AX-01 `python -m xla.enhance.vsr probe|train|export --config configs/vsr_x2.yaml --json`
Accept: `pytest tests/enhance/test_video_sr.py`

## Mục tiêu

Thử ứng viên 2 của thiết kế: mạng nhiều khung dùng khung kề để hồi chi tiết và bớt nhấp nháy. Có giới hạn: **1 ngày làm việc,
4 giờ GPU tính cả bước probe**; dừng sớm theo tiêu chí dưới, phase 14 chạy được không cần phase này.

## Đầu vào

`contract.md` §C (biến thể video `frames` [T,480,640,3]); BasicVSR++ (Apache-2.0, trọng số REDS4 trong MMEditing/mmagic);
cặp video: chuỗi khung liên tiếp (VisDrone-VID nếu tải được, không thì chuỗi `d02x`), làm xấu D2 từng khung (seed theo chuỗi);
`xla/eval/temporal.py`; ứng viên ảnh của phase 11 để so; `data/real/neg_flight_split.json` (chỉ `cal`).

## Bước

1. `probe` (≤ 1 giờ): `basicvsrpp.py` vendor (deformable conv qua `torchvision.ops.deform_conv2d`, không mmcv); xuất thử ONNX
   opset 19 (`DeformConv`) với T = 5 ở 640×480; chạy ORT CUDA. Dừng (ghi `killed: onnx|vram|speed`) nếu ORT không chạy op,
   VRAM > 7 GB với tile, hoặc > 1,0 s/khung.
2. Fine-tune ngắn từ trọng số công bố trên cặp video D2 (×2: hạ mẫu đầu ra ×4 hoặc nhánh ×2), bf16, checkpoint + resume,
   `workers ≤ 4`, khoá D6.
3. So với ứng viên ảnh phase 11 trên khung dev thật: nhấp nháy, thống kê mức thấp, báo nhầm detector v1 trên `cal` sau SR.
   Chỉ thay ứng viên khi hơn ở nhấp nháy mà không tăng báo nhầm; khi đó xuất `enhancer-offline-v1.1.0` (`temporal_window: 5`),
   giữ v1.0.0.
4. `results/v2/vsr/README.md` + `results/v2/vsr/status.json`: kết quả hoặc lý do dừng, giờ GPU.

## Tiêu chí xong

Accept xanh: test kiểm hoặc bundle video hợp lệ §C, hoặc `status.json` ghi `killed: <lý do>` kèm số đo (không có kết quả im lặng).

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `DeformConv` không chạy trên ORT | high | medium | Bước 1 dừng sớm; phase 14 dùng ứng viên ảnh |
| VRAM 8 GB không đủ với T = 5 | medium | medium | T = 3, tile; dừng nếu vẫn quá |
| Ngốn lịch | medium | low | Trần 1 ngày / 4 giờ GPU (Straggler) |

Rollback: không ship; `enhancer-offline-v1.0.0` của phase 11 giữ nguyên.
