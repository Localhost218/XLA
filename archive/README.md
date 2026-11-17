# archive/ — dữ liệu và kết luận v2 cũ của nhóm (không dùng làm kết luận)

Chuyển vào đây ngày 18/11/2026 theo plan `plans/261118-ai-anh-chat-luong-thap/` (phase 01, Deferred 13). Chỉ dữ liệu
và bảng kết luận bị chuyển; mã `src/`, `notebooks/`, `web/`, `models/` vẫn ở chỗ cũ. Notebook nào đọc dữ liệu ở
`data/real_uav` … giờ phải trỏ sang `archive/v2-nhom/data/...`, hoặc xem bản cũ nguyên vẹn bằng `git checkout 3e9a764`.

Bằng chứng: `plans/reports/261118-bang-chung/kiem-xla/kiem-xla.md`.

## Đã chuyển

| Cũ | Mới |
|---|---|
| `data/real_uav/` | `archive/v2-nhom/data/real_uav/` |
| `data/synthetic_gt/` | `archive/v2-nhom/data/synthetic_gt/` |
| `data/synthetic_degraded/` | `archive/v2-nhom/data/synthetic_degraded/` |
| `data/raw_videos/` | `archive/v2-nhom/data/raw_videos/` |
| `data/augmented/` | `archive/v2-nhom/data/augmented/` |
| `results/tables/` | `archive/v2-nhom/results/tables/` |
| `results/figures/` | `archive/v2-nhom/results/figures/` |

## Sự thật về dữ liệu này

- Ảnh "real_uav"/GT là ảnh drone phố có biển chữ Hán cộng hồng nhân tạo, không phải camera này (ESP32-S3 + OV3660 của
  drone IOT-CV). Manifest `data/dataset_manifest.csv` ghi sai nguồn.
- `raw_videos` là video đã mã hoá lại cảnh khác (HEVC 800×456 / H.264 1280×720, encoder Lavf58: cánh đồng, rừng,
  giông), không phải video drone này.
- Bảng 1–7 đo trên dữ liệu do chính mô hình suy giảm sinh ra nên không dùng làm kết luận: thuật toán được viết để đảo
  đúng mô hình suy giảm đó, phép đo thành vòng tròn. Entropy/Laplacian còn thưởng cho nhiễu.
- Không có đánh giá phát hiện người nào trong bản v2.

## Cách xem bản cũ

```
git checkout 3e9a764
```

(commit gốc trước khi tái cấu trúc; quay lại nhánh đang làm bằng `git switch -`).
