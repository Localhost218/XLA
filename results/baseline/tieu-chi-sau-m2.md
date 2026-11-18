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
test (D10). Khung âm gồm 159 khung test bay không người (`neg_flight`) và 300 khung `neg` trong nhà (`neg_indoor`),
gộp 459.

| Biến thể | mAP50 [CI 95 %] | mAP50-95 | AP-small | Ngưỡng dev-val | P / R / F1 @ ngưỡng | ms p50 / p95 (rảnh) |
|---|---|---|---|---|---|---|
| `full` | 0,067 [0,016–0,130] | 0,034 | 0,053 | 0,186 | 0,089 / 0,094 / 0,091 | 20,9 / 23,3 |
| **`edge16`** | **0,126 [0,030–0,224]** | 0,063 | 0,054 | 0,068 | 0,193 / 0,329 / 0,243 | 20,9 / 23,3 |
| `sahi_edge16` | 0,113 [0,024–0,188] | 0,055 | 0,056 | 0,184 | 0,155 / 0,154 / 0,155 | 542,6 / 615,3 |

Báo nhầm (hộp/khung âm) ở ngưỡng dev-val, CI 95 %:

| Biến thể | `neg_flight` (159) | `neg_indoor` (300) | gộp (459) |
|---|---|---|---|
| `full` | 0,547 [0,438–0,648] | 0,097 [0,062–0,135] | 0,253 [0,197–0,309] |
| **`edge16`** | **0,597 [0,397–0,884]** | 0,117 [0,077–0,160] | 0,283 [0,211–0,374] |
| `sahi_edge16` | 0,428 [0,281–0,614] | 0,127 [0,091–0,166] | 0,231 [0,174–0,297] |

Khung bay không người báo nhầm gấp ~5 lần khung trong nhà: đốm hồng/trắng, bóng, vật trên mặt đất lúc bay không có
trong khung `neg`. Số gộp bị khung trong nhà kéo xuống, nên không dùng số gộp để kết luận B2.

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
| **B1** | mAP50 test ≥ **0,227** (điểm ước lượng) **và** CI 95 % của hiệu ghép cặp (model − `edge16`) > 0 | 0,126 [0,030–0,224] | `xla.eval.detection compare --pred-a results/baseline/pred/edge16.csv --pred-b <model> --split test --json`: `metrics.mAP50.b` ≥ 0,227 và `metrics.mAP50.ci95[0]` > 0 |
| **B2** | báo nhầm ≤ **0,05** hộp/khung trên **khung bay không người** (`neg_flight`), **ở ngưỡng vận hành** (ngưỡng F1 chọn trên dev-val); báo kèm `neg_indoor`, số gộp và recall ở ngưỡng đó | `neg_flight` 0,597 [0,397–0,884]; `neg_indoor` 0,117; gộp 0,283; recall 0,329 | `fp_per_neg_flight@<ngưỡng dev-val>`; `compare` cho hiệu ghép cặp |
| **B3** | detector ≤ **30 ms/khung p95** trên RTX 4060, đo lúc máy rảnh, đúng đường chạy thật (ONNX Runtime CUDA EP, phase 11/16) | 23,3 ms p95 (PyTorch) | `time` cho PyTorch; ONNX đo ở phase 11/16 |

- **B1:** max(1,5 × 0,1265; 0,1265 + 0,10) = max(0,190; 0,226) = 0,226 (làm tròn lên 0,227). Vế "+ 0,10" thắng vì mốc
  quá thấp: 1,5 × mốc chỉ hơn mốc 0,06, nhỏ hơn nhiều so với nửa độ rộng CI.
- **B1 kết luận bằng hiệu ghép cặp:** cả hai model chấm trên cùng một bộ 2 000 mẫu khối bootstrap. Biến thiên do chọn
  khối test (phần lớn độ rộng CI của từng model) bị triệt tiêu trong hiệu, nên hiệu có CI hẹp hơn nhiều. Ví dụ ở mục 2a:
  CI riêng của `full` và `edge16` chồng nhau gần hết, nhưng CI hiệu ghép cặp vẫn > 0.
- **B2:** giữ ngưỡng 0,05 của §8, nhưng nay buộc vào ngưỡng vận hành và đo trên khung bay không người. Nếu không buộc
  ngưỡng thì tiêu chí đạt "miễn phí": ở conf 0,25 mốc `edge16` có 0,037 hộp/khung âm gộp (0,107 trên `neg_flight`, 0 trên
  `neg_indoor`), nhưng recall chỉ 0,054. Báo B2 mà không kèm recall thì vô nghĩa.
- **B3:** mốc PyTorch đã đạt (p95 23,3 ms). Phần dư 6,7 ms phải đủ cho model fine-tune và cho ORT CUDA EP trên máy
  IOT-CV; số ONNX hiện chưa có. SAHI (p95 615 ms) loại khỏi mọi đường chạy thời gian thực.

### 2a. Ví dụ hiệu ghép cặp: `edge16` − `full`

`results/baseline/compare_full_vs_edge16.json` (A = `full`, B = `edge16`, mỗi model dùng ngưỡng dev-val của chính nó,
2 000 mẫu, seed 20261118):

| Chỉ số | `full` | `edge16` | Hiệu B − A [CI 95 %] | P(B tốt hơn) |
|---|---|---|---|---|
| mAP50 | 0,067 | 0,126 | +0,059 [+0,013 … +0,117] | 1,00 |
| mAP50-95 | 0,034 | 0,063 | +0,029 [+0,009 … +0,055] | 1,00 |
| AP-small | 0,0534 | 0,0539 | +0,001 [+0,000 … +0,001] | 1,00 |
| recall @ ngưỡng | 0,094 | 0,329 | +0,235 [+0,091 … +0,366] | 0,99 |
| báo nhầm `neg_flight` @ ngưỡng | 0,547 | 0,597 | +0,050 [−0,142 … +0,327] | 0,32 |
| báo nhầm `neg_indoor` @ ngưỡng | 0,097 | 0,117 | +0,020 [−0,034 … +0,075] | 0,22 |

Hợp lý: bỏ hộp chạm mép chỉ xoá hộp sai ở dải tối, nên xếp hạng tốt lên và mAP tăng chắc chắn, kể cả khi CI riêng của
hai model chồng nhau. Báo nhầm ở ngưỡng **không** giảm. Lý do: ngưỡng dev-val của `edge16` thấp hơn nhiều (0,068 so với
0,186), nên hiệu báo nhầm ở đây là hiệu của hai điểm vận hành, không phải tác dụng của việc bỏ mép. CI hiệu vắt qua 0,
nên không kết luận được.

**Kill (a) không kích hoạt.** Mốc 0,126 thấp hơn nhiều so với 0,7, nên không cần đặt mục tiêu khác cho phase 10.

## 3. Giới hạn thống kê — đọc trước khi kết luận

Tập test nhỏ:

- 13 khối 30 s bay, 39 khung có người, 149 hộp. Hộp dồn vào ít khối và ít khung: 106/149 hộp ở bin > 14 m.
- Phía âm: 159 khung bay không người nằm trong 13 khối bay; 300 khung `neg` trong nhà nằm trong 213 khối riêng.

Hệ quả cho từng tiêu chí:

| Tiêu chí | Kết luận được với tập này? |
|---|---|
| B1 | Được, nhờ **hiệu ghép cặp** (lệnh `compare`). CI riêng của mốc rộng ~0,19 (0,030–0,224), nên so hai CI không ghép cặp gần như không bao giờ tách nhau. Hiệu ghép cặp triệt tiêu biến thiên chọn khối: chênh +0,059 giữa `edge16` và `full` đã có CI hiệu > 0. Một model đạt 0,227 hơn mốc +0,10, gần gấp đôi chênh đó, nên điều kiện CI hiệu > 0 nhiều khả năng đi cùng điểm ước lượng. Vẫn phải kiểm cả hai, vì hiệu do vài khối dày người kéo lên sẽ có CI rộng. |
| B2 (`neg_flight`) | Khó hơn B1. Chỉ 159 khung bay không người, nằm trong cùng 13 khối bay, nên CI của mốc rất rộng (0,397–0,884). Ở mục tiêu 0,05 (~8 hộp trên 159 khung), vài khối có đốm hồng là đủ đẩy số lên. Kết luận đạt B2 khi cả điểm ước lượng ≤ 0,05 và cận trên CI không quá ~0,1. Nếu không thì ghi "chưa đủ dữ liệu", đừng ghi "đạt". `neg_indoor` (300 khung, 213 khối) có CI hẹp nhưng không đại diện cảnh bay, chỉ dùng để phát hiện model báo nhầm ở cảnh lạ. |
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
