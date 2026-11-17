# Phase 06: Phép đo độ giống thật — thống kê mức thấp, KID-DINOv2, C2ST

| Trạng thái | Phụ thuộc | Ước lượng | Cần phần cứng |
|---|---|---|---|
| xong 18/11/2026 | phase 04 | M (~1,5 ngày agent), ~0,5 giờ GPU cho smoke (chưa xác minh) | Không |

Lane: realism
Depends-on: 04
Effort: M
Creates: `xla/eval/realism.py`, `xla/eval/c2st.py`, `results/realism/smoke.json`, `tests/eval/test_realism.py`
AX: `python -m xla.eval.realism --real <dir|split:dev> --synth <dir> --json` in khoảng cách thống kê, KID-DINOv2 (trung bình ± CI), độ chính xác C2ST, exit 2 khi hai tập khác quy trình đọc
Accept: `pytest tests/eval/test_realism.py`

## Mục tiêu

Một bộ đo trả lời "ảnh tổng hợp có giống khung thật của camera này không", dùng cho cả hiệu chỉnh (phase 07, trên dev)
lẫn kết luận (báo cáo, trên khung thật giữ riêng). Ba tầng, xếp theo độ tin: thống kê mức thấp tất định → C2ST trên patch
→ KID trên đặc trưng DINOv2. Không dùng FID làm chính.

## Đầu vào cần có

- `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §5 (KID không chệch, DINOv2 qua `dgm-eval`,
  1 000–3 000 patch, trung bình nhiều lần; C2ST CNN nhỏ trên patch 64×64 phần dư cao tần; cùng quy trình đọc/resize cho hai
  tập; so trên patch/phần dư để cảnh không lấn suy giảm).
- `xla/camera/measure.py` (phase 04): mọi thống kê mức thấp lấy từ đây, không viết lại.
- `plan.md` Deferred 4; D9 (khoá GPU).

## Việc theo thứ tự

1. **Nạp ảnh chung một đường:** một hàm `load_patches(paths, size=…, seed)` dùng cho cả thật lẫn tổng hợp: giải mã cùng
   thư viện, bỏ cột `[0, crop_left)`, cắt patch ngẫu nhiên có seed, không resize khác nhau giữa hai tập.
2. **Thống kê mức thấp:** với mỗi đại lượng trong hồ sơ (dark channel, Y p1, độ bão hoà, % hồng, blocking, Laplacian, NLF,
   PSD, MTF50, tối góc), khoảng cách Wasserstein-1 giữa hai phân bố chia cho IQR của tập thật; tổng có trọng số (trọng số
   trong config, mặc định bằng nhau); DQT + lấy mẫu màu so khớp chính xác (khác → báo riêng, không trộn vào tổng).
3. **KID-DINOv2:** dùng `dgm-eval` (encoder DINOv2-ViT-L/14) nếu cài được ở phase 01; trích đặc trưng trên 1 000–3 000 patch
   mỗi tập, KID đa thức bậc 3 ước lượng không chệch, lặp 10 tập con có seed, báo trung bình + CI 95 % bootstrap. Trích đặc
   trưng trong `xla.gpu_lock`; cache đặc trưng theo sha256 ảnh vào `D:\DevCache\cache\realism\`.
4. **C2ST** (`xla/eval/c2st.py`): phần dư cao tần (ảnh − Gauss σ=2) của patch 64×64; CNN nhỏ (≤ 100 k tham số) phân biệt
   thật/tổng hợp; chia train/test theo **ảnh nguồn** (patch cùng ảnh không rơi vào hai phía); báo độ chính xác test + CI;
   ~50 % = không phân biệt được.
5. **Smoke** (đo thời gian thật): 1 000 patch mỗi tập, dev thật vs `d2 --limit 50` của phase 05 nếu đã có, không thì vs D0
   tạm sinh ngay; ghi giây/bước, VRAM đỉnh vào `results/realism/smoke.json`.
6. **Test** `tests/eval/test_realism.py` (CPU, đặc trưng giả bằng hàm trích đơn giản thay DINOv2 qua tham số): KID(X, X') ≈ 0
   khi X, X' cùng phân bố (|KID| < 3 SE); KID tăng đơn điệu khi cộng nhiễu tăng dần; C2ST ≈ 0,5 trên hai tập cùng phân bố và
   > 0,9 trên hai tập khác rõ; khoảng cách thống kê = 0 khi hai tập giống hệt; `load_patches` trả cùng patch cho cùng seed.
7. Commit trên `lane/realism`, đẩy, báo lead.

## Cổng pass

- Accept exit 0; `results/realism/smoke.json` có thời gian từng tầng.

## Rủi ro

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `dgm-eval` không cài được trên Windows | medium | medium | Dừng báo Owner; phương án thay (DINOv2 qua `torch.hub` + KID tự viết) cần Owner duyệt vì đổi Deferred 4 |
| C2ST học nội dung cảnh thay vì suy giảm | medium | medium | Phần dư cao tần + chia theo ảnh nguồn |

## Rollback

Revert commit; xoá cache đặc trưng.
