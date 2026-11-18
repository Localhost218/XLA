# `data/real/labels/` — nhãn người tập test thật + dev-val (phase 08)

| File | Nội dung |
|---|---|
| `manifest.json` | danh sách khung test / dev-val / hard, số hộp, model gán trước (sha256), trạng thái duyệt của chủ dự án |
| `devval/<stem>.txt` | 100 khung dev-val, nhãn máy gán + agent soát (không qua chủ dự án; dùng chọn checkpoint / ngưỡng, D10) |
| `test/<stem>.txt` | 198 khung test, nhãn **chủ dự án đã duyệt** trên Label Studio; chỉ có sau `labelstudio export` |

Định dạng (hợp đồng §7): mỗi dòng `0 cx cy w h`, chuẩn hoá theo 640×480 khung gốc, 6 chữ số thập phân; khung không người
= file rỗng (vẫn có file). `<stem>` = tên khung trong `split.json` bỏ `.jpg`. Khung `neg` không có file (mặc định không
người). Nhãn test agent soát (trước khi chủ dự án duyệt) ở `results/labels/agent_reviewed_test/`, để đối chiếu số sửa.

## Quy trình (lệnh, theo thứ tự)

```powershell
. D:\XLA\scripts\env.ps1
$py = 'D:\DevCache\venvs\xla\Scripts\python.exe'
& $py -m xla.detect.devval select --json        # 1. dev-val 100 khung dev, cách >= 1 s, phân tầng như test
& $py -m xla.detect.prelabel --split all --json # 2. RT-DETR-x + YOLO11x, SAHI, WBF (GPU, trong xla.gpu_lock)
& $py -m xla.detect.review render --frames <khung.jpg> ...   # 3. ảnh soát 2x (ảnh ở D:\DevCache\cache\xla-labels-review)
& $py -m xla.detect.review apply --json         #    gộp results/labels/review_parts/*.csv -> review_log.csv + nhãn
& $py -m xla.detect.labelstudio setup --json    # 4. project xla-test-v1 + nạp khung test kèm nhãn đã soát
& $py -m xla.detect.labelstudio status --json   #    số khung đã / chưa Submit
& $py -m xla.detect.labelstudio export --json   # 5. sau khi chủ dự án duyệt: test/*.txt + manifest (exit 1 nếu còn khung chưa duyệt)
& $py -m pytest tests/labels/test_testset_labels.py
```

Lệnh `labelstudio` gọi API ở `http://127.0.0.1:8080` nên chạy ngoài sandbox của agent.

## Label Studio local

- Cài: `uv tool install --python 3.12 label-studio` với `UV_TOOL_DIR=D:\DevCache\uv-tools`,
  `UV_TOOL_BIN_DIR=D:\DevCache\uv-tools\bin`, `UV_CACHE_DIR=D:\DevCache\uv` (bản 1.23.2, 18/11/2026).
- Dữ liệu (SQLite, log, tài khoản): `D:\DevCache\label-studio`. Tài khoản `owner@xla.local`; mật khẩu và token API nằm
  trong `D:\DevCache\label-studio\credentials.json` (ngoài repo, không commit).
- Ảnh: local files serving, `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo>\data\real`, kho local storage của project trỏ
  `<repo>\data\real\frames`; task dùng URL `/data/local-files/?d=frames/<khung>.jpg`.
- Chỉ nghe `127.0.0.1` (`--internal-host 127.0.0.1`); kiểm: `netstat -ano | findstr :8080` phải ra
  `127.0.0.1:8080 ... LISTENING`, không có `0.0.0.0:8080`. Tắt thống kê gửi về (`COLLECT_ANALYTICS=false`).

**Khởi động lại server** (sau khi máy khởi động lại hoặc server tắt):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\XLA-wt\labels\scripts\labelstudio_start.ps1
# dừng: thêm -Stop ; cổng khác: -Port 8081 ; lần đầu khởi tạo DB mất ~2-3 phút
```

## Kết quả duyệt 18/11/2026 (bước 5–6)

Chủ dự án Submit đủ 198/198 khung test (mỗi task một annotation, không annotation huỷ). `labelstudio export --json`
exit 0: test **149 hộp** (agent soát 151), dev-val **115 hộp** (không qua chủ dự án).

| Sửa của chủ dự án so với nhãn agent (ghép IoU ≥ 0,5) | Hộp |
|---|---|
| xoá | 5 |
| thêm | 3 |
| chỉnh (cặp ghép IoU < 0,95) | 31 (IoU ≥ 0,9: 7; 0,8–0,9: 7; < 0,8: 17; trung vị 0,787) |
| **tổng `owner_edits`** | **39 / 151 = 25,8 %** |

**25,8 % > ngưỡng 15 %: chất lượng soát của agent chưa đủ** để dùng nhãn agent thay cho người duyệt. Phần lớn là chỉnh
hộp (31), 17 cặp lệch nhiều (IoU < 0,8) — ranh giới người / bóng / người dính nhau là chỗ agent đoán sai; số hộp gần như
giữ nguyên (xoá 5, thêm 3). Hệ quả: nhãn **dev-val** (chỉ agent soát, cùng quy trình) có thể lệch hộp cỡ tương tự; dùng để
chọn checkpoint/ngưỡng thì được, đừng báo số đo trên dev-val như số trên nhãn người duyệt.

**Cờ `hard`:** `manifest.json` → `hard_frames` lấy **đúng** các khung chủ dự án tick (5 khung test), không gộp cờ của agent.
Agent đã cờ 78 khung (45 test, 33 dev-val); chủ dự án cố ý bỏ tick 41/45 khung test agent cờ và thêm 1 khung mới (Owner đã
hỏi trực tiếp, xác nhận là cố ý). Danh sách cờ của agent giữ để tham khảo ở `results/labels/agent_hard_frames.json` (và
trường `hard` trong `results/labels/agent_reviewed.json`).

Trang project: `http://localhost:8080/projects/<id>/data` (id in ra ở `labelstudio setup`). Chủ dự án: mở từng khung, sửa
hộp nếu sai (kéo cạnh, xoá hộp thừa, vẽ hộp người sót với nhãn `person`), tick `hard` nếu nhóm người / mờ, bấm **Submit**
(khung không người cũng Submit).
