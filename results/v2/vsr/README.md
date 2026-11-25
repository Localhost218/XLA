# Phase 12 AI v2 (tuỳ chọn): BasicVSR++, dừng ở probe: `killed: onnx`

Lệnh `python -m xla.enhance.vsr probe --config configs/vsr_x2.yaml --json`; kết quả đầy đủ `status.json`, sổ giờ
`gpu_ledger.jsonl`. Trần **1 giờ GPU** cả phase tính cả probe (quyết định chủ dự án 25/11, plan ghi 4 h).

## Probe (bước 1)

- Kiến trúc vendor `xla/enhance/arch/basicvsrpp.py` (Apache-2.0, MMEditing), deformable conv bằng
  `torchvision.ops.deform_conv2d`, không mmcv (D8). Trọng số REDS4 `basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth`
  nạp `strict`; sha256 đủ khớp 8 ký tự `db622b2f` OpenMMLab đặt trong tên file.
- Graph §11 biến thể video: `frames` uint8 [5,480,640,3] → `frame_out` [960,1280,3] của khung giữa (mạng ×4, avg-pool 2
  về ×2, dải mép lặp cột 32).

| Đo | Giá trị |
|---|---|
| PyTorch fp16, T = 5, 640×480 | 3,34 s/khung ra, VRAM đỉnh 3,08 GB |
| Xuất ONNX opset 19 | đạt sau khi tự đăng ký symbolic `torchvision::deform_conv2d` → `DeformConv` (torchvision 0.21 không có, exporter báo `UnsupportedOperatorError`); graph có node `DeformConv` |
| ORT 1.24.4 CPU | `NOT_IMPLEMENTED: Failed to find kernel for DeformConv(19)` |
| ORT 1.24.4 CUDA | node `DeformConv` rơi về CPU EP, cùng lỗi không có kernel |

Kết luận: **`killed: onnx`**, ORT không chạy được op. Hai lý do dừng khác cũng đã đúng dù ORT có op:
3,34 s/khung (PyTorch) > trần 1,0 s/khung, và `DeformConv` cần opset 19 trong khi hợp đồng §4 chốt opset 17 (đổi phải sửa
`docs/hop-dong-model.md`, lane không được sửa). Bước 2–3 (fine-tune D2, so với ứng viên ảnh) không chạy;
`enhancer-offline-v1.1.0` không xuất; `enhancer-offline-v1.0.0` của phase 11 giữ nguyên.

Giờ GPU: **0,013 h** (47 s giữ khoá; chờ khoá 1 236 s không tính).
