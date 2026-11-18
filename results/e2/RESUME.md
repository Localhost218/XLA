# Phase 11 (E2): XONG 18/11/2026

Không còn việc dở. Kết quả, bảng số, quyết định và giờ GPU: `results/e2/README.md`.

Lịch sử: tạm dừng 19:46 (chạy tiếp 21:31) và 22:16 (chạy tiếp 22:58) theo lệnh chủ dự án. Sau lần chạy tiếp thứ hai:
trò r2 (4 worker) xong, r2 không vượt r1 ở cả hai thước đo nên không chạy vòng 3; ship r1 theo dev-val; xuất
`models/release/detector-v1.0.0` có CUDA, parity đạt với mốc `.pt rect=False` (Owner chốt A); `validate` trả `[]`.
