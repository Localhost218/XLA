| Tầng (CSV của IOT-CV) | mAP50 [CI 95 %] | Recall @ ngưỡng [CI] | Báo nhầm / khung bay trống (159) [CI] | / khung trong nhà (300) | Ngưỡng |
|---|---|---|---|---|---|
| 1 detector `detector-v1.0.1` (ONNX của v1.0.0) | 0,119 [0,032; 0,197] | 0,396 [0,271; 0,500] | 0,874 [0,333; 1,486] | 0,003 | 0,077 |
| 2 + `verifier-v1.0.0` @0,350 | 0,210 [0,122; 0,312] | 0,396 [0,271; 0,500] | 0,289 [0,036; 0,631] | 0,003 | 0,350 |
| 3 + lọc track k1/n1 (cả hệ) | 0,210 [0,122; 0,312] | 0,396 [0,271; 0,500] | 0,289 [0,036; 0,631] | 0,003 | 0,350 |

Test v1: 198 khung (39 có người, 149 người). CI 95 % bootstrap theo khối 30 s.

| Mốc → cả hệ (tầng 3): hiệu [CI 95 %] | mAP50 | Recall @ ngưỡng | Báo nhầm / khung bay trống |
|---|---|---|---|
| `detector-v1.0.0` (v1, chỉ detector) | 0,120 → 0,210: +0,090 [+0,026; +0,189] | 0,396 → 0,396: +0,000 [+0,000; +0,000] | 0,881 → 0,289: -0,591 [-1,033; -0,273] |
| mốc COCO + cắt mép 16 px | 0,126 → 0,210: +0,083 [-0,071; +0,219] | 0,329 → 0,396: +0,067 [-0,219; +0,244] | 0,597 → 0,289: -0,308 [-0,690; +0,091] |

| Chỉ số §5 (nhánh B) | v2 | Đích | Kết luận |
|---|---|---|---|
| B1: mAP50 cả hệ | 0,210 [0,122; 0,312] | ≥ 0,300 | **không đạt** |
| B1 tối thiểu: mAP50 và hiệu với mốc edge16 | 0,210; +0,083 [-0,071; +0,219] | ≥ 0,227 và cận dưới hiệu > 0 | **không đạt** |
| B2: báo nhầm / khung, holdout giữ kín (tầng 3) | không đủ chứng minh (0 khung) | ≤ 0,050 | **không đủ chứng minh** |
| B2 trên holdout yếu (cùng buổi, agent duyệt) | không đủ chứng minh (0 khung) | ≤ 0,050 | **không đủ chứng minh** |
| Recall @ ngưỡng vận hành | 0,396 [0,271; 0,500] | ≥ 0,600 | **không đạt** |
