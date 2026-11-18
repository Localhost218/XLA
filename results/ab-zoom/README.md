# `results/ab-zoom/` — thử mù A/B lần 2, phóng to 2× (phase 14, 19/11/2026)

Lý do: ở A/B lần 1 (`results/ab/`, cả khung 624×480) chủ dự án thấy hai ảnh của một cặp "không khác nhau nhiều". Lần 2
vẫn dùng **khung bay thật** của tập test (không làm xấu thêm, không ảnh tự sinh). Mỗi cặp chỉ hiện một vùng 1/2 mỗi chiều
(312×240), phóng 2× theo điểm ảnh gần nhất để nhìn rõ khối JPEG, ám hồng, chi tiết người.

```bash
python -m xla.eval.ab_page build --pairs 55 --seed 20261119 --zoom 2 --out-dir results/ab-zoom --json
python -m xla.eval.ab_score results/ab-zoom/ab_result.json --key results/ab-zoom/ab_key.json --json > results/ab-zoom/ab_score.json
```

- Model giống hệt lần 1: `enhancer-v1.0.0` (NAFNet-w16, sha256 `1728de3e…`), cổ điển `xla.enhance.classic:enhance` 1.0.0.
- Cấu trúc như lần 1: 25 cặp mạng vs gốc + 25 cặp mạng vs cổ điển + 5 cặp lặp, cân tầng độ cao × có/không người. Seed khác
  (20261119) nên phần lớn là khung khác lần 1.
- Vùng phóng: khung có người → tâm là hộp người lớn nhất theo **nhãn test đã duyệt**; không người → vùng ngẫu nhiên theo seed.
  Vùng chọn trước khi xử lý ảnh, giống nhau hai bên, ghi ở `ab_key.json` → `pairs[].region`. Không chọn theo ảnh của biến
  thể nào nên không thiên vị.
- Chấm: chủ dự án, 55/55 lượt, key `fbd458e6754590f0`, xuất 18/11/2026 18:28 UTC.

## Kết quả (`ab_score.json`)

| So sánh | Lượt | Mạng thắng | Tỉ lệ | CI Wilson 95 % | Tiêu chí §8 | Đạt? |
|---|---|---|---|---|---|---|
| mạng vs gốc | 25 | 13 | 0,52 | [0,335; 0,700] | ≥ 70 % | **không** (CI chạm 50 %: chưa phân định) |
| mạng vs cổ điển | 25 | 5 | 0,20 | [0,089; 0,391] | ≥ 55 % | **không** (CI trọn dưới 50 %: mạng thua rõ) |
| độ nhất quán (5 cặp lặp) | 5 | — | 3/5 = 0,60 | — | — | — |

Lệch bên: chọn trái 44 % (50 lượt không lặp).

Theo tầng (chỉ mô tả, n nhỏ; tầng có/không người là tầng thiết kế sẵn của trang, không chọn sau khi xem số):

| | có người | không người | 3–8 m | 8–14 m | > 14 m |
|---|---|---|---|---|---|
| mạng vs gốc | **10/11** | 3/14 | 2/7 | 6/9 | 5/9 |
| mạng vs cổ điển | 4/12 | 1/13 | 1/6 | 2/10 | 2/9 |

Gộp hai lần, mạng vs gốc trên khung có người: 7/10 (lần 1) + 10/11 (lần 2) = **17/21 = 0,81**, CI Wilson [0,60; 0,92].
Trên khung không người: 4/15 + 3/14 = 7/29 = 0,24 [0,12; 0,42].

## Kết luận

- **Quyết định ship không đổi**: lần 2 cũng không đạt cả hai tiêu chí A/B §8 (0,52 < 0,70; 0,20 < 0,55). `enhancer-v1.0.0`
  không phải ảnh rõ khuyến nghị mặc định (`results/ab/README.md` §4.6).
- Phóng to làm mạng nhích lên (vs gốc 0,44 → 0,52; vs cổ điển 0,04 → 0,20) nhưng CI hai lần chồng nhau rộng, không kết luận
  được là phóng to làm đổi kết quả.
- **Phát hiện có ích**: quanh người, mạng thắng ảnh gốc ổn định ở cả hai lần (17/21, CI dưới 0,60), khớp với số vật lý
  (hồng −99,9 %: áo/da người bị ám hồng ở ảnh gốc). Trên nền không người mạng thua gốc (7/29), khớp với lỗi màu nguội, nhạt
  và khử mù yếu ở `results/ab/README.md` §4.6 hướng sửa 1–2. Đây là mô tả hậu kiểm, không phải tiêu chí đã đăng ký; muốn
  dùng làm kết luận phải chấm một trang mới chỉ gồm khung có người.
- Cổ điển vẫn thắng mạng ở mọi tầng.
