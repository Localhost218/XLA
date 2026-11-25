# `results/v2/`: kết quả AI v2 (plan `plans/261125-ai-v2/plan.md`)

Đo ngày 25–26/11/2026, nhánh `feat/ai-v2`. Bảng cho báo cáo dựng bằng `python scripts/reproduce.py --write` từ file trong
thư mục này (`results/reproduce/tables/t7_v2_nhan_dien.md`, `t8_v2_ab.md`, `t9_v2_ms.md`, hình
`results/reproduce/fig/f4_v2_bao_nham_recall.svg`); `--check` chấm lại ba CSV cả hệ và so sha256. Báo cáo: `BAO_CAO_DO_AN.md`
mục 6. Số trong file này chép từ các README con, mỗi dòng ghi nguồn.

## Chỉ số §5 (đích chốt ở `plans/reports/261125-brainstorm-cai-thien-ai-v2.md` §5)

| Chỉ số | Đo | Đích | Kết luận | Nguồn |
|---|---|---|---|---|
| B1: test mAP50, cả hệ detector → verifier | 0,210 [0,122; 0,312] | ≥ 0,30 | không đạt | `system/score.json`, bảng t7 |
| B1 tối thiểu | 0,210; hiệu với edge16 +0,083 [−0,071; +0,219] | ≥ 0,227 và cận dưới hiệu > 0 | không đạt | như trên |
| B2: báo nhầm / khung trên holdout giữ kín | 0 khung (holdout chặt và holdout yếu) | ≤ 0,05 | không đủ chứng minh | `system/score.json`, `system/holdout_weak.json` |
| Recall ở ngưỡng vận hành (test) | 0,396 [0,271; 0,500] | ≥ 0,6 | không đạt | bảng t7 |
| ms trực tiếp, p95 tổng có "Ảnh rõ", v2 − v1 cùng lúc đo | −9,65 ms (99,1 → 89,5) | ≤ +5 ms | đạt | `iotcv/bench.json`, bảng t9 |
| ms trực tiếp, chỉ nhận diện, v2 − v1 | +8,39 ms (64,0 → 72,4) | ≤ +5 ms | không đạt | như trên |
| Ảnh rõ trực tiếp (classic ONNX) vs gốc, A/B mù | 25/25, Wilson [0,867; 1,000] | cận dưới > 0,5 | đạt | `../ab-v2/ab_score.json`, bảng t8 |
| Ảnh rõ offline vs classic, A/B mù | 16/25, Wilson [0,445; 0,798] | cận dưới > 0,5 | không đạt → không ship | `../ab-v2/decision.json` |
| Hàng rào: offline không tăng báo nhầm (`cal`) | −0,033 [−0,123; +0,039] | cận dưới ≤ 0 | qua | `../ab-v2/guard_fp.json` |
| Offline một chuyến 20 phút ≤ 60 phút GPU | 0,758 giờ (0,404 s/khung) | ≤ 1 giờ | đạt (không ship) | `sr/export.json`, `sr/bench.json`, bảng t9 |

Cải thiện có CI của hiệu không chạm 0, so với v1 trên cùng 198 khung test: mAP50 +0,090 [+0,026; +0,189], báo nhầm trên
khung bay trống 0,881 → 0,289 (−0,591 [−1,033; −0,273]). Phần này là của verifier; detector vẫn là v1 (`system/README.md` §4).

## Bundle

| Bundle | Ship | sha256 `.onnx` | Ghi chú |
|---|---|---|---|
| `detector-v1.0.1` | có | `23b1da69165ffe05760bda23b4a41c858930e8a8fd8b0aebc039f1e0c5c59aef` | cùng ONNX với `detector-v1.0.0`, thêm `recommended_track` k1/n1 |
| `verifier-v1.0.0` | có | `d566343a5487775b9f6334d9df0bb0a1d9c9b2d2b76e2b8bfa4738facc012d58` | EfficientNet-B0 crop 96, `conf_threshold` 0,35 |
| `enhancer-v2.0.0` | có | `83d96e8b0f628d0e6cc271ed9550220962de897cac2d57ed306a1db51cb5477b` | classic ONNX, "Ảnh rõ" mặc định |
| `enhancer-offline-v1.0.0` | không | `e93b3eaa714a003df0c6a77c11dacefc7b22666143dc8c490957dc1a5b53afc8` | Real-ESRGAN x2plus tinh chỉnh; thua classic ở A/B |
| `detector-v2.0.0` | không có | — | Kill (b): trò mới không hơn v1 trên cùng đường ONNX |

Nguồn sha256: `meta.json` của từng bundle ở `models/release/` và `iotcv/README.md` §1. `.onnx` không vào git; file của bundle
ship đính GitHub Release `models-v2.0.0`.

## Thư mục con

| Thư mục | Phase | Kết luận một dòng |
|---|---|---|
| `data/` | 02 | HERIDAL 1 684 ảnh, WiSARD 110, TinyPerson 173 (chỉ làm dương cho verifier) sau lọc góc nhìn; SARD bỏ; giấy phép chỉ nghiên cứu |
| `neg-holdout/` | 03 | 6 phiên bay, cả 6 dính khung test v1 → holdout 0 khung (Kill (a)); `cal` 150 khung theo khối, `mine` 85; holdout yếu cùng buổi cũng 0 khung |
| `teacher/` | 05 | thầy YOLO11m không rõ hơn v1 trên devpool → không chạy ứng viên nhãn giả C2 |
| `detect/` | 06 | Kill (b): trò C3 tốt nhất OOF nhưng kém v1 trên cùng đường ONNX; không có `detector-v2.0.0` |
| `verifier/` | 07 | `effb0_96`: báo nhầm `cal` 15 → 1–2 hộp, recall OOF giữ 0,266; ms N = 5 không đạt đích < 3 ms |
| `system/` | 09 | điểm vận hành ngưỡng 0,35, k = n = 1; chấm test một lần (bảng t7); bench phase 09 trước khi IOT-CV sửa verifier |
| `classic/` | 10 | classic ONNX khớp numpy (MAE ≤ 0,018); lai classic → NAFNet không qua lọc, không xuất |
| `sr/` | 11 | Real-ESRGAN x2plus tinh chỉnh L1 + perceptual được chọn; SwinIR quá chậm; nhấp nháy 1,35 > 1,15 |
| `vsr/` | 12 | BasicVSR++ dừng ở probe: ORT không có kernel `DeformConv` |
| `iotcv/` | 15 | IOT-CV chạy ba bundle ship, golden CUDA + CPU đạt; bench ABBA (bảng t9); web 1707×900 |
| `../ab-v2/` | 14 | hàng rào qua; A/B lượt cả khung (bảng t8); `offline_ship = false`; lượt phóng to không chấm |

## Giới hạn chung

Một buổi bay (bản ghi 20261102-100941) cho mọi khung có người; test 39 khung có người nên CI rộng; vật dễ nhầm lặp giữa các
phiên cùng buổi; detector không có bản OOF nên recall hiệu chỉnh lạc quan; GPU hạ xung ~780 MHz lúc bench (số chính thức kèm
điều kiện). Chi tiết: `system/README.md` §7, `BAO_CAO_DO_AN.md` mục 6.7.
