# Tiêu chí nhánh B sau M2 (phase 09 bước 4)

Ngày 18/11/2026. Viết lại tiêu chí nhánh B của §8 (`plans/reports/261118-brainstorm-ai-anh-chat-luong-thap.md`, bản ghi đã
duyệt, không sửa) bằng số mốc thật. §8 cho phép sửa sau M2. Plan dẫn tới file này.

Nguồn số: `results/baseline/baseline.json`. Đó là số chính thức: nhãn test chủ dự án đã duyệt (`approved_by_owner: true`,
198 khung, 149 hộp), ba biến thể, CI 95 % bootstrap 2 000 lần theo khối 30 s, seed 20261118. Lệnh dựng lại:

```powershell
. D:\XLA\scripts\env.ps1
$py = 'D:\DevCache\venvs\xla\Scripts\python.exe'
& $py -m xla.detect.baseline run       # CSV dự đoán (GPU, gpu_lock)
& $py -m xla.detect.baseline time      # ms/khung lúc máy rảnh -> run_meta.json
& $py -m xla.detect.baseline score --out results/baseline/baseline.json
```

## 1. Mốc YOLO11s COCO @1280 trên tập test thật

Ngưỡng vận hành là ngưỡng tối đa F1, chọn trên **dev-val** (100 khung, nhãn agent soát, không qua chủ dự án) rồi áp lên
test (D10). Khung âm gồm 159 khung test không người và 300 khung `neg`, tổng 459.

| Biến thể | mAP50 [CI 95 %] | mAP50-95 | AP-small | Ngưỡng dev-val | P / R / F1 @ ngưỡng | Báo nhầm / khung âm @ ngưỡng | ms p50 / p95 (rảnh) |
|---|---|---|---|---|---|---|---|
| `full` | 0,067 [0,016–0,130] | 0,034 | 0,053 | 0,186 | 0,089 / 0,094 / 0,091 | 0,253 [0,197–0,309] | 20,9 / 23,3 |
| **`edge16`** | **0,126 [0,030–0,224]** | 0,063 | 0,054 | 0,068 | 0,193 / 0,329 / 0,243 | 0,283 [0,211–0,374] | 20,9 / 23,3 |
| `sahi_edge16` | 0,113 [0,024–0,188] | 0,055 | 0,056 | 0,184 | 0,155 / 0,154 / 0,155 | 0,231 [0,174–0,297] | 542,6 / 615,3 |

ms đo bằng `python -m xla.detect.baseline time` lúc máy rảnh. Lúc đó không còn tiến trình `xla.degrade`, khoá GPU trống,
CPU toàn máy 20 % trong lúc đo. Đo trên 498 khung test + neg, khởi động 5 khung trước. Đây là ultralytics PyTorch FP32 với
conf 0,001; ms gồm tiền/hậu xử lý, không gồm đọc JPEG. `edge16` dùng lại ms của `full` vì chỉ lọc hộp sau suy luận.
SAHI chậm chủ yếu do hậu xử lý trên rất nhiều hộp điểm thấp ở conf 0,001.

**Mốc chuẩn: `edge16`.** Mọi model sau đều bỏ dải mép trái (phase 03 đo ~13 px), nên so với `full` là so với một mốc
yếu một cách giả tạo: chỉ bỏ hộp ở mép đã gần gấp đôi mAP50. SAHI không hơn `edge16` (0,113 so với 0,126, CI chồng
nhau gần hết) và chậm hơn 26 lần, nên không dùng làm mốc.

## 2. Tiêu chí mới

| | Tiêu chí | Số mốc (`edge16`) | Cách đo |
|---|---|---|---|
| **B1** | mAP50 test ≥ **0,227** | 0,126 [0,030–0,224] | `xla.eval.detection --split test`, mAP50 (IoU 0,5, maxDets 100) |
| **B2** | báo nhầm ≤ **0,05** hộp/khung âm, **ở ngưỡng vận hành** (ngưỡng F1 chọn trên dev-val); kèm recall ở ngưỡng đó | 0,283 [0,211–0,374], recall 0,329 | `fp_per_neg_frame@<ngưỡng dev-val>` |
| **B3** | detector ≤ **30 ms/khung p95** trên RTX 4060, đo lúc máy rảnh, đúng đường chạy thật (ONNX Runtime CUDA EP, phase 11/16) | 23,3 ms p95 (PyTorch) | `time` cho PyTorch; ONNX đo ở phase 11/16 |

- **B1:** max(1,5 × 0,1265; 0,1265 + 0,10) = max(0,190; 0,226) = 0,226 (làm tròn lên 0,227). Vế "+ 0,10" thắng vì mốc
  quá thấp: 1,5 × mốc chỉ hơn mốc 0,06, nhỏ hơn nhiều so với nửa độ rộng CI.
- **B2:** giữ ngưỡng 0,05 của §8, nhưng nay buộc vào ngưỡng vận hành. Nếu không buộc thì tiêu chí đạt "miễn phí": ở
  conf 0,25 mốc `edge16` đã có 0,037 hộp/khung âm, nhưng recall chỉ 0,054. Báo B2 mà không kèm recall thì vô nghĩa.
- **B3:** mốc PyTorch đã đạt (p95 23,3 ms). Phần dư 6,7 ms phải đủ cho model fine-tune và cho ORT CUDA EP trên máy
  IOT-CV; số ONNX hiện chưa có. SAHI (p95 615 ms) loại khỏi mọi đường chạy thời gian thực.

**Kill (a) không kích hoạt.** Mốc 0,126 thấp hơn nhiều so với 0,7, nên không cần đặt mục tiêu khác cho phase 10.

## 3. Giới hạn thống kê — đọc trước khi kết luận

Tập test nhỏ:

- 13 khối 30 s bay, 39 khung có người, 149 hộp. Hộp dồn vào ít khối và ít khung: 106/149 hộp ở bin > 14 m.
- Phía âm có 213 khối. Phần lớn là khung trong nhà (`neg`), không phải khung bay không người.

Hệ quả cho từng tiêu chí:

| Tiêu chí | Kết luận được với tập này? |
|---|---|
| B1, điểm ước lượng | Được, nhưng CI rộng. CI của mốc rộng ~0,19 (0,030–0,224), nên một model đạt đúng 0,227 vẫn có CI chồng lên CI mốc. Mục tiêu 0,227 nằm sát cận trên CI mốc: đạt B1 nghĩa là điểm ước lượng vượt cả cận trên đó, nhưng **không** chứng minh được "tốt hơn mốc có ý nghĩa thống kê" bằng so CI không ghép cặp. Đề xuất phase 10 báo thêm **hiệu ghép cặp** (cùng mẫu khối bootstrap cho hai model, CI của hiệu > 0). Bộ chấm hiện chưa có lệnh này. Cùng seed và cùng tập khối thì mẫu khối đã trùng nhau, nên chỉ cần thêm phép trừ. |
| B2 | Kết luận khá được. 459 khung âm, 213 khối âm, CI ±0,08 quanh 0,28. Với mục tiêu 0,05 (~23 hộp trên 459 khung), CI sẽ hẹp hơn nhiều nhờ ít sự kiện. Lưu ý: 300/459 khung âm là trong nhà, không đại diện đốm hồng/trắng lúc bay. Nên báo thêm riêng 159 khung bay không người. |
| B3 | Kết luận được (đo trực tiếp, 498 khung, p95 ổn định), miễn là đo lúc máy rảnh. Số đo lúc CPU 100 % do lane khác là 139 ms p50 cho cùng model: phải luôn ghi kèm % CPU (`cpu_percent_during`). |
| Tách theo độ cao | **Không kết luận được** cho bin 3–8 m: 4 khung có người, 7 hộp, CI mAP50 của `edge16` là 0,09–1,0. Bin 8–14 m (10 khung, 36 hộp) chỉ để mô tả. Chỉ bin > 14 m (25 khung, 106 hộp) có CI đủ hẹp để so. |
| Cờ `hard` | **Chỉ mô tả.** 5 khung, 18 hộp. |

Thêm hai lưu ý:

- **Nhãn dev-val chỉ do agent soát.** Chủ dự án đã sửa 25,8 % hộp test của agent, chủ yếu ranh giới hộp. Ngưỡng chọn
  trên dev-val nên có thể lệch nhẹ; điều này ảnh hưởng P/R/F1/B2 ở ngưỡng, không ảnh hưởng mAP.
- **AR-small của mốc là ~0,53, cao hơn mAP rất nhiều.** Ở conf 0,001, model có tìm thấy khoảng một nửa số người nhưng
  xếp chúng dưới rất nhiều hộp sai. Phần còn lại cho fine-tune nằm ở việc xếp hạng / báo nhầm, không chỉ ở recall.

## 4. Số tạm → chính thức

Số tạm (`baseline_provisional.json`) chấm trên nhãn agent soát, trước khi chủ dự án duyệt. Số chính thức lệch rất ít:

| Biến thể | mAP50 tạm → chính thức | AR-small |
|---|---|---|
| `full` | 0,0674 → 0,0673 | 0,527 → 0,526 |
| `edge16` | 0,1272 → 0,1265 | 0,531 → 0,531 |
| `sahi_edge16` | 0,1093 → 0,1133 | 0,549 → 0,534 |

Số hộp test 151 → 149, số khung có người không đổi (39). Ngưỡng dev-val chọn ra trùng với bản tạm ở cả ba biến thể (sha256 file nhãn dev-val khác bản tạm, nhưng ngưỡng không đổi). Báo
nhầm/khung âm không đổi: chủ dự án không thêm/bớt khung có người, nên tập khung âm giữ nguyên.
