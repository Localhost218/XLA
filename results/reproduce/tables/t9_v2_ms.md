| Luồng, nhịp thật | Bước | v1 p50 / p95 ms | v2 p50 / p95 ms | Hiệu p95 | Trần | Kết luận |
|---|---|---|---|---|---|---|
| có người xem Ảnh rõ | detector | 53,2 / 63,3 | 57,4 / 63,4 | +0,15 |  |  |
|  | verifier | – / – | 0,0 / 8,9 | – |  |  |
|  | enhancer | 29,7 / 34,4 | 15,4 / 19,3 | -15,12 |  |  |
|  | tổng | 84,5 / 99,1 | 76,3 / 89,5 | -9,65 | +5,0 | đạt |
| chỉ nhận diện | detector | 60,0 / 63,3 | 59,6 / 63,3 | -0,01 |  |  |
|  | verifier | – / – | 0,0 / 9,4 | – |  |  |
|  | tổng | 60,8 / 64,0 | 61,1 / 72,4 | +8,39 | +5,0 | **không đạt** |

v1 = `detector-v1.0.0` + `enhancer-v1.0.0`; v2 = `detector-v1.0.1` + `verifier-v1.0.0` + `enhancer-v2.0.0`. Bản ghi 20261102-100941, khung 284–884, 4 lượt xen kẽ v1-v2-v2-v1. Máy: xung GPU trung vị mỗi lượt 780–802 MHz (tối đa 3105), 70–72 °C; bỏ khung có Ảnh rõ: v1 7,8 · 8,2 %, v2 4,2 · 4,7 %.

| Ảnh rõ offline ×2 (`enhancer-offline-v1.0.0`, không ship) | s / khung TB (p95) | Khung / s của bản ghi | Giờ GPU cho chuyến 20 phút |
|---|---|---|---|
| ORT CUDA fp16, 20 khung dev | 0,404 (0,526) | 5,6 | 0,758 |
