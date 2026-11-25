# AI v2 phase 14: hàng rào báo nhầm + thử mù A/B v2

Plan `plans/261125-ai-v2/phase-14-ab-mu-v2.md`, D13 (`plan.md`). Bước 1–5 xong 26/11/2026: chủ dự án chấm lượt cả khung
(lượt phóng to không chấm, không quyết định), `ab_score` + `decision.json` ở §4 (`offline_ship = false`).

```
python -m xla.eval.guard_fp --bundle models/release/enhancer-offline-v1.0.0 --write-meta --json
python -m xla.eval.ab_page build --comparisons offline_vs_raw,offline_vs_classic,classic_vs_raw \
    --per-comparison 25 --repeats 5 --seed 20261125 --json                                        # lượt 1 (trang này)
python -m xla.eval.ab_page build --comparisons offline_vs_raw,offline_vs_classic,classic_vs_raw \
    --per-comparison 25 --repeats 5 --seed 20261126 --zoom 2 --out-dir results/ab-v2-zoom --json  # lượt 2
python -m pytest tests/eval/test_ab_v2.py                                                         # Accept
```

## 1. Hàng rào báo nhầm (`guard_fp.json`)

`detector-v1.0.1` (`conf_threshold` 0,07669) chạy trên 150 khung `cal` không người (2 phiên, 16 khối 10 khung) ở dạng gốc,
và ở dạng sau `enhancer-offline-v1.0.0` (ONNX đúng bản sẽ ship, ORT CUDA, ảnh ra 1280×960 hạ ×½ `INTER_AREA`).

| | Hộp | Báo nhầm / khung | Khung có báo nhầm |
|---|---|---|---|
| gốc | 15 | 0,100 | 9 |
| offline | 10 | 0,067 | 9 |
| hiệu (offline − gốc) | | **−0,033**, CI 95 % [−0,123; +0,039] | |

CI tính bằng bootstrap ghép cặp theo khối, lấy lại khối trong từng phiên (10 000 lần, seed 20261125). Cận dưới ≤ 0 nên
**pass**: offline không tăng báo nhầm có ý nghĩa và được vào A/B. Không nói được là offline *giảm* báo nhầm, vì CI chứa 0.
Số này khớp phép đo PyTorch ở phase 11 (0,100 → 0,067). Holdout có 0 khung, nên `fp_holdout_*` = null, ghi là "0 khung /
không đủ chứng minh". Holdout không được đọc. Kết quả ghi vào `metrics` của `models/release/enhancer-offline-v1.0.0/meta.json`.
Điều kiện đo: GPU 87 °C, 2 025 MHz lúc kết thúc, 64,9 s GPU.

## 2. Trang thử mù

| Lượt | Trang | Seed | Cặp |
|---|---|---|---|
| 1, cả khung | `results/ab-v2/ab-test.html` | 20261125 | 80 = 25 offline_vs_raw + 25 offline_vs_classic + 25 classic_vs_raw + 5 lặp (3/1/1) |
| 2, phóng ×2 | `results/ab-v2-zoom/ab-test.html` | 20261126 | 80 = 25 + 25 + 25 + 5 lặp (1/3/1) |

- Lai classic→NAFNet không có so sánh nào vì không được xuất (phase 10 không qua lọc).
- Chỉ dùng khung **dev** có nhãn người (`data/real/labels/devpool`, `devval`, có độ cao; 174 khung), phân tầng theo phiên × bin độ cao × có
  người. Không dùng khung test.
- Mỗi biến thể giữ độ phân giải gốc của nó: gốc và classic 624×480, offline 1248×960, đều đã cắt dải mép, JPEG q95 4:4:4.
  Trình duyệt phóng mọi ảnh về cùng một cỡ hiển thị (CSS `width: 100%`), giống trang Lịch sử bay. Ở 1707×900 CSS
  (150 %), mỗi ảnh hiện 810×623 CSS, tức khoảng 1215×935 điểm màn hình. Lượt 2 cắt vùng 1/2 (tâm là người lớn nhất theo
  nhãn dev) ở độ phân giải gốc rồi để trình duyệt phóng, không nội suy riêng.
- Tên ảnh là mã hex. Biến thể chỉ nằm trong `ab_key.json`. Cặp lặp đổi bên và cách lần đầu ít nhất 10 lượt.
- Lượt 2 dùng seed + 1 như lượt phóng to của v1 (`results/ab-zoom`), để khung và bên không lặp lại lượt 1.

Kiểm bằng Chrome headless (Playwright `channel: "chrome"`, 1707×900, scale 1,5): ở cả hai trang, 80/80 cặp tải được cả hai
ảnh, hai bên cùng cỡ hiển thị, không có lỗi trang, không URL nào chứa tên biến thể. Nút Xuất tải về JSON
`xla-ab-export-v1` đủ 80 lượt, đúng `key_id`. `ab_page.check_page` cho cả hai trang: không lộ tên biến thể, không thiếu ảnh,
cặp lặp đúng.

## 3. Tiêu chí (cố định trước khi chấm)

Mỗi so sánh đạt khi cận dưới Wilson 95 % > 0,5. Với 25 cặp, 18/25 là đạt, còn 17/25 cho cận dưới 0,484. Luật quyết định
nằm trong `ab_score.DECISION_RULE`: offline được ship khi guard pass và cả offline_vs_raw lẫn offline_vs_classic cùng đạt
ở lượt **cả khung** (sửa 26/11 trước khi chấm, commit b545b64: lượt phóng to chỉ báo cáo; luật "ít nhất một lượt" làm lỏng tiêu chí). classic_vs_raw chỉ là chỉ số §5. Lai không thay được classic. Một giới hạn cần biết: ảnh SR ×2
không có lưới khối JPEG nên người chấm có thể nhận ra được.

## 4. Kết quả (chủ dự án chấm lượt cả khung, 26/11/2026 01:49) — `ab_result.json`, `ab_score.json`, `decision.json`

| So sánh | Thắng (bên thứ nhất) | Tỉ lệ | Wilson 95 % | Đạt? |
|---|---|---|---|---|
| classic vs gốc | 25/25 | 1,00 | [0,867; 1,000] | đạt (chỉ số §5) |
| offline vs gốc | 25/25 | 1,00 | [0,867; 1,000] | đạt |
| offline vs classic | 16/25 | 0,64 | [0,445; 0,798] | **không đạt** |

Cặp lặp đổi bên: 5/5 chọn như lần đầu (nhất quán 1,00).

**Quyết định (`decision.json`): `offline_ship = false`** — lượt cả khung không thắng cả gốc lẫn classic. Offline rõ hơn gốc
rõ rệt nhưng không chứng minh được hơn classic (64 %, cận dưới 0,445), trong khi đắt hơn (0,40 s/khung, ~0,76 h/chuyến 20
phút), nhấp nháy 1,35 (> 1,15) và màu ngả lạnh. IOT-CV không cài `enhancer-offline-*`; API offline trả 503, web ẩn nút.
Classic thắng gốc 25/25: củng cố quyết định Owner bật "Ảnh rõ" classic mặc định (Deferred 17), không đổi quyết định đó.

Lượt phóng to (`results/ab-v2-zoom/`) **không chấm**: theo luật chốt trước nó không quyết định; trang vẫn để nguyên nếu
muốn chấm sau (`ab_score` lượt `zoom` chỉ thêm vào báo cáo).
