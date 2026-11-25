# Plan: AI v2 — nhận diện người lọc nhiều tầng + ảnh rõ hai tầng (XLA → IOT-CV)

Hardness: phases=18 new=3 modules=0 lanes=8
Flags: --auto
PRs: 3

Ngày lập: 25/11/2026. Thiết kế đã duyệt (SSOT phạm vi, chỉ số thành công §5, rủi ro §4), **không mở lại quyết định**:
`plans/reports/261125-brainstorm-cai-thien-ai-v2.md`. Plan v1 (cấu trúc, lane, quy ước worktree, kết quả):
`plans/261118-ai-anh-chat-luong-thap/plan.md`, `thuc-thi-phase-10-14.md` (§2 môi trường, §3 GPU, §5 hợp đồng lane, §7, §10).
Vì sao v1 không đạt: `results/e2/README.md` §2, `results/ab/README.md` §4.6, `results/e1/README.md`.

Hợp đồng chung XLA ↔ IOT-CV cho v2: `contract.md` cạnh file này (phần **thêm** vào `docs/hop-dong-model.md`, nâng
`contract_version` 1.0 → 1.1). Phase 01 gộp nó vào `docs/hop-dong-model.md`; từ đó file đó là SSOT. Mỗi phase là một
file tự đủ: agent chỉ cầm `phase-NN-*.md` + `contract.md` + mục "Luật chung khi thực thi" dưới đây là bắt đầu được.

## Forecast

Cap: 70 % tối đa cho mọi dòng khi giả định chính (dữ liệu ngoài + lớp phủ định thật làm detector tổng quát hơn trên
camera này; số phiên bay đủ để chia mine/holdout) còn UNMEASURED.

| Outcome (đo được) | Confidence | Basis | Đo (re-score 26/11/2026, phase 17) |
|---|---|---|---|
| B1: test mAP50 (đường ONNX, cả hệ detector → verifier) ≥ 0,30 | 35 % | ESTIMATED: E1-D0 đã đạt 0,297 [cận dưới hiệu +0,016] (MEASURED, `results/e1/README.md` §2); v2 thêm dữ liệu ngoài + lớp âm, nhưng test chỉ 74 khung có người nên CI rộng ±0,08 (MEASURED, CI bundle v1 [0,032–0,197]) | MEASURED 0,210 [0,122; 0,312] → **không đạt** (`results/reproduce/tables/t7_v2_nhan_dien.md`) |
| B1 tối thiểu: test mAP50 ≥ 0,227 và cận dưới CI hiệu ghép cặp với `edge16` > 0 | 55 % | ESTIMATED: một trong bốn biến thể E1 v1 đã đạt cả hai vế (MEASURED); luật chọn mới (k-fold theo phiên thay dev-val 100 khung) bỏ đúng lỗi đã chọn sai E2 r1 | MEASURED 0,210 < 0,227; vs edge16 +0,083 [−0,071; +0,219] → **không đạt** cả hai vế (t7) |
| B2: báo nhầm ≤ 0,05 hộp/khung trên `neg_flight` phần **giữ kín** sau tầng 2 + 3 | 45 % | ESTIMATED: thấp nhất v1 0,302 (E1-D2, MEASURED) → cần giảm ~6×; bộ phân loại vùng cắt + lọc track k/n thường cắt phần lớn báo nhầm rời rạc, nhưng chưa đo trên túi rác / bụi cây thật (UNMEASURED) | MEASURED holdout 0 khung (Kill (a)); holdout yếu cùng buổi cũng 0 khung → **không đủ chứng minh** (t7, `results/v2/system/holdout_weak.json`) |
| Recall ở ngưỡng vận hành ≥ 0,6 | 40 % | ESTIMATED: v1 0,396 (MEASURED, brainstorm §5); tầng 2 + 3 hạ báo nhầm đồng thời hạ recall; D7 ưu tiên B2 nên recall có thể không đạt | MEASURED 0,396 [0,271; 0,500] → **không đạt**; verifier không mất người nào, recall chặn bởi detector v1 (t7) |
| ms trực tiếp không tăng quá +5 ms so với p95 nhịp thật v1 (~54 ms) | 60 % | ESTIMATED: verifier ~96 px × vài hộp ước < 3 ms (brainstorm §4, chưa đo); classic ONNX thay NAFNet (bundle v1 p95 21,0 ms, MEASURED) có thể còn nhanh hơn | MEASURED phase 15 (ABBA cùng tiến trình, GPU ~780 MHz): có Ảnh rõ 99,1 → 89,5 ms, −9,65 → **đạt**; chỉ nhận diện 64,0 → 72,4 ms, +8,39 → **không đạt** (`results/reproduce/tables/t9_v2_ms.md`). Phase 09 đo +70,5 ms trước khi IOT-CV sửa verifier theo bậc |
| Ảnh rõ trực tiếp (classic ONNX) A/B mù vs gốc: cận dưới Wilson 95 % > 0,5 | 45 % | ESTIMATED: classic thắng NAFNet 24/25 (MEASURED, `results/ab/README.md` §4.6) nhưng NAFNet vs gốc chỉ 11/25–13/25, nên classic vs gốc chưa ai đo (UNMEASURED) | MEASURED lượt cả khung 25/25, Wilson [0,867; 1,000] → **đạt** (`results/reproduce/tables/t8_v2_ab.md`) |
| Ảnh rõ offline A/B mù vs classic: cận dưới Wilson 95 % > 0,5 | 40 % | ESTIMATED: SR ×2 nhìn rõ hơn khi phóng to là kết quả thường thấy, nhưng camera này nén JPEG nặng, mạng sinh có thể bịa chi tiết (rủi ro brainstorm §4) | MEASURED lượt cả khung 16/25, Wilson [0,445; 0,798] → **không đạt**, `offline_ship = false` (t8, `results/ab-v2/decision.json`); lượt phóng to không chấm |
| Hàng rào: detector chạy trên ảnh offline không tăng báo nhầm (holdout) | 60 % | ESTIMATED: chưa đo; Real-ESRGAN nổi tiếng làm mịn nền, ít tạo vật thể mới, nhưng chưa thử trên khung này | MEASURED trên `cal` (holdout 0 khung): 0,100 → 0,067 hộp/khung, hiệu −0,033 [−0,123; +0,039] → **qua** (t8, `results/ab-v2/guard_fp.json`) |
| Một chuyến 20 phút xử lý offline xong ≤ 60 phút trên RTX 4060 Laptop 8 GB | 55 % | UNKNOWN cho s/khung của RRDBNet ×2 ở 640×480 trên ORT CUDA: insufficient evidence; 55 % là ước theo cỡ mạng (ESTIMATED). Phase 11 bước 1 đo | MEASURED 0,404 s/khung (p95 0,526) × 5,6 khung/s → 0,758 giờ → **đạt** (t9, `results/v2/sr/export.json`); bundle không ship |

Success rate: phase 02 dữ liệu ngoài 80 % (HERIDAL + TinyPerson + WiSARD sample đã trên đĩa 25/11, MEASURED; SARD bỏ) · 03 chia theo phiên 60 % (UNKNOWN số phiên đủ) · 05–06 detector 55 % · 07 verifier 60 % · 08 IOT-CV track 80 % · 09 B2 đạt 45 % · 10 classic ONNX 65 % (parity percentile/deblock chưa thử) · 11 SR offline 55 % · 12 mạng video 25 % (tuỳ chọn, ONNX của deformable conv chưa chắc) · 13 IOT-CV offline 80 % · 18 web 85 % · 14 A/B 45 %. Overall (chuỗi B: 02 → 06 → 07 → 09 ≈ 0,7 × 0,55 × 0,6 × 0,45 ≈ 10 %; chuỗi A: 10 hoặc 11 → 14 ≈ 25 %), dưới mắt xích yếu nhất (09, 45 %). Đây là xác suất mọi tiêu chí đạt ngay lần đầu, không phải xác suất có kết quả báo cáo được (mọi phase đều cho số đo kèm CI dù đạt hay không).
Effect (low/likely/high): giờ GPU 10 / 15 / 25 giờ (ESTIMATED, cộng theo phase: thầy 3 / 4,5 / 6 (trần phase 05) + trò k-fold 2,8 / 4,2 / 6 (1–6 lần × ~0,7 giờ, v1 E1 39,6–45,6 phút/lần MEASURED; trần phase 06) + verifier 0,5 / 0,7 / 1 + gán nhãn trước 03–04 0,5 / 0,8 / 1 + lai classic 0 / 1 / 2 + SR 2 / 3 / 4,5 + video 0 (dừng ở probe) / 0,5 / 4 + offline một chuyến 1 / 1 / 1 ≈ 9,8 / 15,7 / 25,5) · lịch theo đường găng 6 / 9 / 14 ngày làm việc (ESTIMATED, cộng Effort đường găng) · chất lượng: báo nhầm/khung bay 0,881 → ≤ 0,05 (đích). May be slower: tải SARD (4,43 GB, cần đăng nhập IEEE DataPort, DOCUMENTED), duyệt holdout và chấm A/B phụ thuộc lúc chủ dự án rảnh, cài BasicVSR++ (mmcv) trên Windows (UNKNOWN). Trực tiếp: verifier cộng ms, bù lại classic nhẹ hơn NAFNet (ESTIMATED).
Uncertainties (by impact): 1. Số phiên bay + số khung không người ngoài bản ghi `20261102-100941` (mọi khung dev/test v1 đều từ bản ghi này, MEASURED từ `data/real/split.json`) → phase 03 bước 1 đếm theo `python -m backend.flights --events`, 30 phút CPU, trước phase 05/07/09. 2. Tỉ lệ ảnh gần nadir trong SARD/TinyPerson → phase 02 bước 3 đếm sau lọc góc, 1 giờ, trước phase 05. 3. VRAM YOLO11l @1280 → phase 05 bước 1 smoke 15 phút. 4. Parity ONNX của classic (percentile, deblock) → phase 10 bước 1 parity từng tầng, 2 giờ. 5. s/khung và VRAM của SR ×2 → phase 11 bước 1 đo trọng số có sẵn, 30 phút. 6. ORT có chạy DeformConv (BasicVSR++) không → phase 12 bước 1 thử xuất, 1 giờ.
Kill / no-regression: (a) phase 03 thấy holdout < 150 khung không người → vẫn chấm nhưng báo "không đủ chứng minh", thêm human step ghi phiên bay trống; không bao giờ lấy khung holdout ra hiệu chỉnh. (b) Không ứng viên detector nào hơn `detector-v1.0.0` trên OOF k-fold → không xuất `detector-v2.0.0`, giữ v1, verifier + track vẫn ship. (c) Verifier không hạ báo nhầm trên `cal` ≥ 50 % ở recall OOF giảm ≤ 0,05 → không ship `verifier-v1.0.0`, IOT-CV chạy detector + track. (d) Offline: không ứng viên qua hàng rào báo nhầm hoặc > 0,5 s/khung → không xuất `enhancer-offline-v1.0.0`, web ẩn nút offline (API trả 503 đúng hợp đồng). (e) Classic vs gốc không đạt cận dưới > 0,5 → ghi "không đạt" ở chỉ số §5; "Ảnh rõ" classic vẫn bật mặc định theo quyết định Owner (brainstorm §3.A1), người dùng tắt được. Rollback: mỗi phase một nhánh lane; bundle có phiên bản, IOT-CV đổi về bundle v1 bằng `DETECTOR_ONNX`/`ENHANCER_ONNX` trong `.env`; `test_ngan_sach_ms` đỏ có chủ đích, không động vào.
Re-score (26/11/2026, phase 09; bổ sung phase 17 cùng ngày): mọi dòng Forecast nay đã đo, cột "Đo" ở bảng trên, số lấy từ `results/reproduce/tables/t7`–`t9` (dựng bằng `scripts/reproduce.py --write`). Theo phase: 02 — HERIDAL giữ cả 1 684 ảnh sau lọc góc, WiSARD 110, TinyPerson chỉ 173 (< 200 → chỉ làm dương cho verifier), SARD bỏ (`results/v2/data/README.md` §3) · 03 — 6 phiên bay, phiên nào cũng dính khung test v1 → **holdout 0 khung** (Kill (a)), cal 150 khung theo khối, mine 85; holdout yếu cùng buổi (luật 30 s) cũng 0 khung (`results/v2/neg-holdout/README.md` §2, §6) · 04 — devpool 160 khung / 70 có người / 218 người, 2 fold theo phiên (đích 500–700 khung không đạt, cần bay thêm) · 05 — thầy YOLO11m không rõ hơn v1 trên devpool (−0,031 [−0,092; +0,052]) → C2 không chạy (`results/v2/teacher/README.md`) · 06 — **Kill (b)**: trò C3 OOF 0,321 nhưng cùng đường ONNX kém v1 −0,041 [−0,078; +0,018], không có `detector-v2.0.0`; trên test C3 `.pt` hơn v1 +0,207 [+0,019; +0,308] nên Kill (b) có thể là âm tính giả, không đảo quyết định (`results/v2/detect/README.md` §1) · 07 — verifier effb0_96: cal 0,100 → 0,013 hộp/khung, recall OOF giữ 0,266, ms N = 5 p50 5,96 (đích < 3 không đạt) (`results/v2/verifier/README.md` §1) · 09 — điểm vận hành 0,35, k/n 1/1, `detector-v1.0.1`; chỉ số nhánh B ở bảng t7 (`results/v2/system/README.md` §1) · 10 — classic ONNX MAE ≤ 0,018 với numpy, lai không qua lọc · 11 — Real-ESRGAN x2plus tinh chỉnh 0,404 s/khung, nhấp nháy 1,35 (> 1,15) (`results/v2/sr/README.md`) · 12 — `killed: onnx` (ORT 1.24.4 không có kernel `DeformConv`, `results/v2/vsr/README.md`) · 14 — guard qua, A/B lượt cả khung chủ dự án chấm 26/11 01:49, `offline_ship = false` · 15 — IOT-CV chạy ba bundle ship, ms bảng t9 (`results/v2/iotcv/README.md`). Cải thiện có CI ≠ 0 so với v1: mAP50 +0,090 [+0,026; +0,189], báo nhầm khung bay trống test 0,881 → 0,289. Đạt: ms có Ảnh rõ, classic vs gốc, hàng rào offline, giờ offline/chuyến. Không đạt: B1, B1 tối thiểu, recall, ms chỉ nhận diện, offline vs classic. Không đủ chứng minh: B2.
Not checked: giấy phép dataset của TinyPerson và VisDrone (README không ghi; repo TinyPerson ghi MIT cho mã), bản ghi bay 19/11 có video hay không (`logs/tlog/20261119-*` có, `logs/recordings/` không có thư mục 19/11). Đã kiểm từ lúc lập plan (bỏ khỏi danh sách): góc nhìn HERIDAL/TinyPerson/WiSARD (soát tay 100 ảnh mỗi bộ, `results/v2/data/README.md` §3), ORT DeformConv (không có kernel, `results/v2/vsr/README.md`), VRAM của thầy, trò, verifier, SR, BasicVSR++ (đo trong README từng lane).

## Finish ETA

`done by 2026-11-25 20:21 UTC+07:00 [parallel; p80-chain; 3 gates x 10m; delivery 52m42s] n=173 basis=global validated=unvalidated (0 of 10 plans measured; backtest of 10 past plans: took 1.78x longer than this figure, hit 2/10)`
(sequential: 2026-11-26 02:36; đường găng của CLI: 1 > 3 > 4 > 5 > 6 > 7 > 9 > 14 > 15 > 17). Con số chỉ tính thời gian agent +
cổng người theo dữ liệu chung của kit; **không** tính giờ GPU (10 / 15 / 25 giờ, Forecast), thời gian tải dữ liệu ngoài, hay
lúc chủ dự án rảnh để duyệt holdout và chấm A/B. Lịch thực tế theo Forecast: 6 / 9 / 14 ngày làm việc.

## Phases

| # | Phase | Nhánh | Lane | Depends-on | Effort | File |
|---|---|---|---|---|---|---|
| 01 | Hợp đồng v1.1: verifier, enhancer-offline, `recommended_track`; validator, bundle giả | chung | serial | none | M | `phase-01-hop-dong-v2.md` |
| 02 | Dữ liệu ngoài có giấy phép (HERIDAL, TinyPerson, WiSARD sample; SARD bỏ), lọc nadir, làm xấu D2 v3 | B | data-ext | 01 | L | `phase-02-du-lieu-ngoai.md` |
| 03 | Nhập mọi bản ghi, chỉ mục phiên bay, chia `neg_flight` theo phiên → mine / cal / holdout | B | realdata | 01 | M | `phase-03-phien-bay-chia-neg.md` |
| 04 | Nhãn dev mở rộng + fold theo phiên cho chọn model k-fold | B | realdata | 03 | L | `phase-04-nhan-dev-kfold.md` |
| 05 | Thầy YOLO11l/m @1280 trên D0+D2 (VisDrone + ngoài) + lớp âm mine, nhãn giả dev | B | detect | 01, 02, 03, 04 | L | `phase-05-thay-nhan-gia.md` |
| 06 | Trò YOLO11s: ≤ 3 ứng viên, chọn bằng k-fold theo phiên, ngưỡng OOF, xuất `detector-v2.0.0`; thử cắt ô offline | B | detect | 01, 04, 05 | L | `phase-06-tro-kfold-xuat.md` |
| 07 | Verifier: khai thác hard negative trên mine, ngưỡng trên cal, MobileNetV3/EfficientNet-B0, RoiAlign trong graph, `verifier-v1.0.0` | B | verifier | 01, 03, 06 | L | `phase-07-bo-xac-minh.md` |
| 08 | IOT-CV: nạp verifier, lọc track k/n, sự kiện, `replay_pred.py` | B | iotcv | 01 | L | `phase-08-iotcv-xac-minh-track.md` |
| 09 | Hiệu chỉnh k/n trên cal (blank test + holdout), chấm cả hệ trên test + holdout, `results/v2/` | B | serial | 03, 06, 07, 08 | M | `phase-09-hieu-chinh-cham-he.md` |
| 10 | Classic thành ONNX `enhancer-v2.0.0` (parity với `xla.enhance.classic`); thử lai classic → NAFNet có giới hạn giờ | A | classic | 01 | L | `phase-10-classic-onnx.md` |
| 11 | Offline SR ×2: Real-ESRGAN x2plus (zero-shot + fine-tune D2), SwinIR-M x2 real; chọn, xuất `enhancer-offline-v1.0.0` | A | sr | 01, 03 | L | `phase-11-sieu-phan-giai-offline.md` |
| 12 | Tuỳ chọn: mạng video BasicVSR++ fine-tune D2, có tiêu chí dừng | A | sr | 03, 11 | M | `phase-12-mang-video-tuy-chon.md` |
| 13 | IOT-CV backend: job offline + cache + API (dừng khi armed / vừa mất liên kết), hộp sau verifier ở Lịch sử bay | A | iotcv | 01, 08 | L | `phase-13-iotcv-offline-web.md` |
| 14 | Hàng rào báo nhầm trên ảnh offline + thử mù A/B v2 (cả khung + phóng to) | A | serial | 06, 10, 11, 12 | M | `phase-14-ab-mu-v2.md` |
| 15 | IOT-CV chạy bundle thật: golden, ms, một chuyến offline thật, web 1707×900, PR IOT-CV | chung | serial | 09, 13, 14, 16, 18 | M | `phase-15-iotcv-chay-that.md` |
| 16 | IOT-CV: xoá `ml/scripts/` sau grep tham chiếu, sửa tài liệu sống trỏ tới nó | nợ | iotcv-debt | 01 | S | `phase-16-iotcv-don-no.md` |
| 17 | Báo cáo: `results/v2`, `reproduce.py --write`, `BAO_CAO_DO_AN.md`, slide `docs/slide/`, Release `models-v2.0.0`, PR cuối | chung | serial | 09, 14, 15, 16 | M | `phase-17-bao-cao-phat-hanh.md` |
| 18 | IOT-CV web: nguồn ảnh offline, ô trạng thái model (KeyValue), mặc định "Ảnh rõ" theo `enhancer_default_on` | A | iotcv | 13 | M | `phase-18-iotcv-web.md` |

Đường găng: 01 → 02 → 05 → 06 → 07 → 09 → 15 → 17, nhánh song song 01 → 03 → 04 → 05. Nhánh A (10, 11 → 12 → 14)
và IOT-CV (08 → 13 → 18) chạy song song với nhánh B từ ngày đầu vì chỉ cần hợp đồng (01). Hai cổng người (duyệt holdout ở
03, chấm A/B ở 14) không nằm trên đường găng của nhánh B; cổng A/B nằm trên đường tới 15.

## Trạng thái phase (26/11/2026)

| # | Trạng thái | Kết quả chính / lệch so với phase file |
|---|---|---|
| 01 | done | hợp đồng v1.1 trong `docs/hop-dong-model.md`; bundle giả sinh trong tmp, không có `tests/contract/fixtures/` |
| 02 | done | HERIDAL 1 684, WiSARD 110, TinyPerson 173 (chỉ verifier); SARD bỏ |
| 03 | done, Kill (a) | holdout 0 khung; cal 150 (theo khối), mine 85 |
| 04 | done, lệch | devpool 160 khung / 70 có người (đích 500–700 / ≥ 150); 2 fold theo phiên |
| 05 | done | thầy YOLO11m (l dừng epoch 1), trần GPU 1 h; C2 không chạy |
| 06 | done, Kill (b) | không có `detector-v2.0.0`; hệ dùng `detector-v1.0.1` |
| 07 | done | `verifier-v1.0.0` effb0_96; ms N = 5 không đạt đích < 3 ms |
| 08 | done | IOT-CV `track_filter.py`, verifier, `replay_pred.py` (lane/iotcv, gộp ở PR IOT-CV #61) |
| 09 | done | điểm vận hành 0,35 k1/n1; chấm test một lần; B2 không đủ chứng minh |
| 10 | done | `enhancer-v2.0.0` classic ONNX; lai không qua lọc, không xuất |
| 11 | done | `enhancer-offline-v1.0.0` (không ship sau phase 14) |
| 12 | done, dừng | `killed: onnx` ở probe |
| 13 | done | job offline + API (503 khi không có bundle); cache mặc định `logs/.cache/flights-ai` |
| 14 | done | guard qua; A/B lượt cả khung; `offline_ship = false` |
| 15 | done | IOT-CV chạy ba bundle ship; PR IOT-CV #61 merged (`c82509a`) |
| 16 | done | `ml/scripts/` đã xoá |
| 17 | đang làm | bước 1–4, 6 xong (bảng t7–t9, f4, báo cáo mục 6, slide, `results/v2/README.md`, re-score); bước 5 (Release `models-v2.0.0`) và 7 (PR cuối) do phiên chính |
| 18 | done | web: nguồn offline, ô trạng thái model, "Ảnh rõ" mặc định theo `enhancer_default_on` |

## Human steps (do first)

| Step | Unblocks | Where / exact command | Why an agent cannot | Timing |
|---|---|---|---|---|
| ~~Tải dữ liệu ngoài~~ **xong 25/11**: HERIDAL, TinyPerson, WiSARD sample ở `D:\IOT_Tools\downloads\datasets\{heridal,tinyperson,wisard}\`; SARD bỏ | phase 02 | — | — | done |
| ~~Duyệt tập `neg_flight` giữ kín~~ **không cần** (holdout 0 khung, phase 03 Kill (a); holdout yếu cũng 0 khung) | phase 03 | Trang `results/v2/neg-holdout/review/index.html` báo 0 khung | — | done: không có gì để duyệt; chạy lại khi có phiên bay trống |
| ~~Chấm thử mù A/B v2~~ **xong lượt cả khung 26/11/2026 01:49** (lượt quyết định); lượt phóng to không chấm, không quyết định (luật chốt trước khi chấm, b545b64) | phase 14 | `results/ab-v2/ab-test.html` (đã chấm); `results/ab-v2-zoom/ab-test.html` (tuỳ chọn, chỉ thêm vào báo cáo) | Cảm nhận người xem là chính đại lượng cần đo | done (lượt phóng to: tuỳ chọn, không chặn) |
| (Tuỳ chọn, không chặn) Buổi bay tới: bật camera bay 1–2 chuyến ở bãi KHÔNG có người để có phiên holdout | phase 03 (holdout, B2) | Bay như thường; agent chạy backend ghi video, sau đó chạy lại phase 03 bước 1–4 và `xla.eval.system score` | Cần phần cứng + người lái | later: buổi bay tới; là đường duy nhất để đo B2 |

## Luật chung khi thực thi

- **Môi trường XLA (Bash):** y hệt `plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-10-14.md` §2: `export
  UV_CACHE_DIR='D:\DevCache\uv' UV_PROJECT_ENVIRONMENT='D:\DevCache\venvs\xla' TORCH_HOME='D:\DevCache\torch'
  HF_HOME='D:\DevCache\hf' YOLO_CONFIG_DIR='D:\DevCache\ultralytics' XLA_DATA_ROOT='D:\DevCache\datasets\xla'
  PYTHONUTF8=1 TMP='D:\DevCache\tmp' TEMP='D:\DevCache\tmp'`; `PY=/d/DevCache/venvs/xla/Scripts/python.exe`; luôn
  `$PY -m pytest ...` / `$PY -m xla....` từ gốc worktree (Accept `pytest X` nghĩa là `$PY -m pytest X`). Không `uv sync`.
- **Môi trường IOT-CV:** `uv run --no-sync pytest ...` từ gốc worktree IOT-CV; pytest/backend có lúc bị sandbox chặn
  socket (WinError 10013) → chạy qua PowerShell `Start-Process` với `PYTHONUTF8=1`. Web: `pnpm build` bằng **PowerShell**,
  không Git Bash. Màu/nhãn lấy từ `lib/tone.ts`, `lib/eventStyle.ts`, `components/kit`; kiểm bố cục ở 1707×900 nền ngày
  bằng Chrome headless (`chromium.launch({ channel: "chrome" })` trong `frontend/`).
- **Gọi chéo repo:** subprocess `uv` từ XLA sang IOT-CV (và ngược lại) phải **bỏ** `UV_PROJECT_ENVIRONMENT` và
  `VIRTUAL_ENV` khỏi env con, không thì uv xoá venv của repo kia (AGENTS.md IOT-CV, bẫy 19/11).
- **ORT GPU:** thư mục `site-packages/nvidia/*/bin` phải có trong PATH; chỉ đọc `get_providers()` sau một lần chạy thử
  (thiếu cuDNN thì ORT vẫn báo CUDA nhưng chạy CPU). Số ms phụ thuộc GPU có hạ xung không: ghi khối `machine` kèm số đo.
- **Máy:** job CPU nặng **tối đa 4 worker** (kể cả `workers` dataloader; `calibrate.py` mặc định 16 → truyền `--workers 4`);
  checkpoint mỗi epoch/lượt + resume (RAM lật bit dưới tải). Không hoãn việc vì nhiệt/hạ xung: đo xong ghi kèm điều kiện.
  **Không đụng máy ảo VMware của chủ dự án** (không tắt/treo tiến trình `vmware*`, không đổi cấu hình ảo hoá).
- **GPU:** RTX 4060 Laptop 8 GB, một job một lúc qua `xla.gpu_lock.gpu_lock(label, timeout_s≥21600)` (D6); job IOT-CV
  không giữ khoá XLA nhưng không chạy khi một job XLA giữ khoá (đọc `read_holder`).
- **Tải về:** `D:\IOT_Tools\downloads\`; cache/trọng số/run: `D:\DevCache\...` (`D:\DevCache\runs\xla\v2\<lane>`,
  `D:\DevCache\weights\`). Ổ D giữ ≥ 10 GB trống. Lệnh mạng: Bash `dangerouslyDisableSandbox: true`; remote SSH.
- **Không đụng Google Drive, không Plane, không gửi gì lên repo kit.** Không sửa `test_ngan_sach_ms` (đỏ có chủ đích),
  không khoá xung GPU để "chữa" ngân sách ms.

## Planning waves

Group A: run
Group B: run
Review: 1 reviewer
Fan-out: 8 lanes
Facts: lanes=8 new-rows=3

Nhóm A chạy song song một lượt: `t1k-corpus-scout` (MCP knowledge-retrieval **không có** trong phiên → corpus `unverified`,
mọi kết luận "không có" dưới đây là grep trong phạm vi ghi ở Reuse Ledger), hai `t1k-investigator` (XLA, IOT-CV, có
`file:line`), `t1k-fact-fetcher` (giấy phép + nguồn dataset/model, URL kiểm 200 ngày 25/11). Nhóm B + C chạy một lượt sau
khi bản nháp có trên đĩa (`t1k-list-classifier` theo rubric R1–R6, `t1k-data-verifier` số liệu, `t1k-review-correctness`
mọi câu "luôn/không bao giờ" có đo thử RoiAlign + xuất ONNX, `t1k-tester` chạy validator); mọi phát hiện đã sửa vào plan
(chia `cal`, `--blank`, sửa Depends-on, tách phase 18, trần giờ GPU, classic mặc định bật theo Owner). Reviewer: không spawn (chủ dự án không yêu cầu review; luật `no-unrequested-review`).

## Verification topology

| Lane | Verifier | Concurrent? |
|------|----------|-------------|
| data-ext | `pytest tests/degrade/test_external_sources.py` | yes, CPU (≤ 4 worker); CLIP probe dùng GPU, xếp hàng D6 |
| realdata | `pytest tests/camera/test_sessions_neg_split.py tests/labels/test_dev_folds.py` | yes, CPU; gán nhãn trước dùng GPU, xếp hàng D6 |
| detect | `pytest tests/detect/test_teacher_pseudo.py tests/detect/test_student_v2.py` | no: huấn luyện GPU, xếp hàng D6 |
| verifier | `pytest tests/verify/test_verifier.py` | no: huấn luyện GPU ngắn, xếp hàng D6 |
| iotcv | `pytest backend/tests/test_vision_v2.py backend/tests/test_flights_offline.py` + `bunx vitest run tests/unit/modelStatus.test.tsx tests/unit/flightsOffline.test.tsx` (IOT-CV `frontend/`) | yes, bundle giả, CPU |
| classic | `pytest tests/enhance/test_classic_onnx.py` | yes cho parity (CPU); bước lai dùng GPU, xếp hàng D6 |
| sr | `pytest tests/enhance/test_offline_sr.py tests/enhance/test_video_sr.py` | no: fine-tune GPU, xếp hàng D6 |
| iotcv-debt | `pytest backend/tests/test_repo_hygiene.py` (IOT-CV) | yes, chỉ xoá file + sửa Markdown |

Lane sizing: tier=opus UNMEASURED; lanes=8; p=0.8; P(all land)=0.17 (information, not a cap)

p=0,8 là ước (ESTIMATED). GPU là tài nguyên chung duy nhất: bốn lane GPU (detect, verifier, classic bước lai, sr) mở
song song nhưng chạy job GPU nối tiếp qua khoá D6.

Seam check (25/11, `cochange-seams.cjs`, 6 tháng, 78 commit XLA dùng được): mọi cặp lane XLA (data-ext `xla/degrade/**`, realdata
`xla/camera/**`, detect `xla/detect/**`, verifier `xla/verify/**`, classic `xla/enhance/export_classic.py`, sr `xla/enhance/arch/**`)
`independent`. Lane IOT-CV ở repo khác; ranh giới do Owns từng phase + D10 giữ.

## Contract

Owner: tác giả plan (phiên chính); lane cần đổi thì dừng và báo, Owner sửa `contract.md` / `docs/hop-dong-model.md`, tăng `contract_version`, báo lại mọi lane
File: contract.md
Test: tests/contract/test_model_contract.py

`contract.md` cạnh plan này là bản đã chốt của phần thêm v1.1 (verifier §A, `recommended_track` + sự kiện §B,
enhancer-offline §C, API offline IOT-CV §D, CSV cả hệ §E, dữ liệu §F). Phase 01 chép nó vào `docs/hop-dong-model.md`
(SSOT từ đó), để lại một dòng con trỏ ở `contract.md`; IOT-CV trỏ tới file XLA, không chép.

## Decision register

| # | Decision | Taken |
|---|---|---|
| D1 | Hợp đồng v1.1 | `contract_version` "1.1" (thêm, không phá): kind mới `verifier`, `enhancer-offline`; trường tuỳ chọn `recommended_track` trong meta detector. IOT-CV `SUPPORTED_CONTRACT_MAJOR=1` giữ nguyên, bundle v1.0 vẫn nạp được. Bản E4 `input_variant: "enhanced"` (từng giữ chỗ 1.1) dời sang 1.2 nếu có. `contract.md` |
| D2 | Vào/ra `verifier.onnx` | Vào `frame` uint8 [480,640,3] BGR + `boxes` float32 [N,4] xyxy px khung gốc (N ≥ 0); nới hộp, cắt, RoiAlign về `crop_px`, chuẩn hoá đều **trong graph**; ra `scores` float32 [N] = P(người). IOT-CV không viết phép cắt. `contract.md` §A |
| D3 | Vào/ra `enhancer-offline.onnx` | Ảnh đơn: `frame` [480,640,3] → `frame_out` uint8 [960,1280,3] BGR (×2). Mạng video: `frames` uint8 [T,480,640,3] → `frame_out` của khung giữa. Dải mép trái xử lý như enhancer §3, nhân 2. `contract.md` §C |
| D4 | Lọc theo thời gian (tầng 3) | **Một** cài đặt duy nhất của lọc sự kiện ở IOT-CV `backend/vision/track_filter.py` (ghép IoU tham lam, numpy, không kéo ultralytics/torch vào runtime); `EventThrottle` thành lớp tương thích gọi nó; XLA hiệu chỉnh + chấm tầng 3 bằng `scripts/vision/replay_pred.py` của IOT-CV (§E). ByteTrack của `selftrain.py` chỉ làm sạch nhãn giả. k/n từ `recommended_track`, env `DETECTION_TRACK_K/N` ghi đè; không có thì k = n = `DETECTION_MIN_FRAMES`, cooldown toàn cục (gần đúng hành vi cũ, ba khác biệt ghi ở `contract.md` §B) |
| D5 | Dữ liệu thật, chia tập | `data/real/split.json` (dev/test/neg v1) **chỉ đọc**. Mới: `data/real/sessions.json` (phiên bay = cửa sổ `python -m backend.flights --events`), `data/real/neg_flight_split.json` (`mine` huấn luyện / `cal` hiệu chỉnh / `holdout` chấm, theo phiên), `data/real/labels/devpool/` + `folds.json`. Khung thật vào huấn luyện theo danh sách cho phép. Holdout không bao giờ vào huấn luyện, khai thác, hiệu chỉnh, chọn model hay quyết định ship; phát lại cả hệ luôn `--blank` test + holdout. Test v1 chỉ dùng để chấm. Chi tiết `contract.md` §E, §F |
| D6 | GPU dùng chung | Một job GPU một lúc, khoá `D:\DevCache\locks\gpu.lock` (`xla/gpu_lock.py:76`), `timeout_s` ≥ 6 giờ, nhả ngay sau mỗi lần chạy |
| D7 | Ưu tiên điểm vận hành | Ngưỡng verifier + k/n chọn để báo nhầm/khung trên **cal** ≤ 0,03 (chừa biên cho 0,05 ở holdout), trong đó lấy recall OOF dev lớn nhất. Recall < 0,6 thì báo "không đạt", không nới đích báo nhầm |
| D8 | Phụ thuộc Python | Không thêm gói vào `pyproject.toml` của XLA hay IOT-CV mà không dừng báo Owner. Backbone verifier từ `torchvision` (đã có), kiến trúc SR/video **vendor** vào `xla/enhance/arch/` như NAFNet v1 (ghi giấy phép đầu file), không cài `basicsr`/`realesrgan`/`mmagic` vào venv chung; mmcv (nếu phase 12 cần) cài vào venv riêng `D:\DevCache\venvs\xla-vsr` |
| D9 | Nhánh, worktree, PR | XLA: nhánh tích hợp `feat/ai-v2` từ `main`; lane worktree `D:\XLA-wt\<lane>` nhánh `lane/<lane>`. IOT-CV: tích hợp `feat/ai-v2`; lane worktree `D:\IOTCV-wt\<lane>` nhánh `lane/<lane>`. Lane đẩy nhánh và báo, không mở PR; phase serial gộp. PR: XLA sau phase 09, XLA cuối ở phase 17, IOT-CV ở phase 15 (nhánh `iotcv-debt` gộp vào `feat/ai-v2` IOT-CV trước đó). Commit dạng pathspec |
| D10 | File IOT-CV dùng chung | `backend/config.py`, `backend/schemas.py`, `backend/api.py`, `backend/app.py`, `backend/ws.py`, `frontend/src/**` chỉ lane `iotcv` sửa (08 → 13 → 18, nối tiếp trong cùng worktree). Lane `iotcv-debt` chỉ sửa `ml/**`, Markdown nêu ở phase 16 và `backend/tests/test_repo_hygiene.py`. Trước khi sửa: `git status --short` + hỏi phiên đang sống |
| D11 | Định dạng dự đoán | CSV v1 `frame,x1,y1,x2,y2,conf` (px khung 640×480); tên khung `frame_name(rec, segment, video_frame)` (`xla/camera/frames.py:77`). CSV cả hệ từ IOT-CV có cột thêm `stage` theo §E; chấm bằng `xla.eval.detection` |
| D12 | Tên bundle | `detector-v2.0.0`, `verifier-v1.0.0`, `enhancer-v2.0.0` (classic ONNX, `network: "classic"`), `enhancer-offline-v1.0.0`. `.onnx` không vào git (`.gitignore:9`), đính vào Release `models-v2.0.0` ở phase 17 |
| D13 | A/B v2 | 25 cặp không lặp mỗi so sánh + 5 cặp lặp; so sánh: classic vs gốc, offline vs classic, offline vs gốc (+ lai vs classic nếu lai qua lọc ở phase 10). Đạt = cận dưới Wilson 95 % > 0,5 (25 cặp: ≥ 18/25; 17/25 cho 0,484). Khung **dev** (không test). Ảnh offline 1280×960; gốc và classic phóng bằng trình duyệt như trang Lịch sử bay. Hàng rào báo nhầm quyết định trên `cal`, holdout chỉ báo cáo |

Straggler: lane CPU về trễ quá ~2× trung vị thời gian lane thì gộp các lane đã xong, giao phần còn lại cho lane trễ bằng re-brief; không chờ. Lane GPU không tính là trễ khi đang chờ khoá D6. Phase 12 (tuỳ chọn) trễ quá 1 ngày làm việc thì dừng theo tiêu chí dừng của nó, phase 14 chạy không có nó.

## Reuse Ledger

Phạm vi rà (25/11/2026): `D:\XLA` (`xla/**`, `configs/**`, `scripts/**`, `tests/**`, `models/release/**`, `results/{e1,e2,ab}`,
plan 261118), `d:\Coding\IOT-CV` (`backend/vision/**`, `backend/flights/**`, `backend/{api,app,ws,events,schemas,config}.py`,
`scripts/vision/**`, `ml/**`, `frontend/src/{components/VideoPanel,components/flights,components/kit,lib,hooks,store,pages}`,
`.github/workflows`, `infra`). Corpus `knowledge-retrieval`: **unverified** (MCP không có trong phiên 25/11) — ba dòng NEW
bên dưới là kết quả grep, không phải quét corpus.

| # | Capability | Verdict | Evidence |
|---|---|---|---|
| 1 | Bản sao số camera D2 (`params_version` 3) | REUSE | `degrade` @ xla/degrade/d2_camera.py:267, `encode_final` @ :287; `params_version: 3` @ configs/degrade_d2.yaml:7 |
| 2 | Dựng tập D0/D2 + nhãn YOLO + `dataset.yaml` | EXTEND | `xla/degrade/build.py:112-114` (gọi D2), `:273-274` (`dataset.yaml`); delta: nguồn `external` trong `configs/sources.yaml`, tập trộn `d02x` (cặp SR ×2 dựng riêng ở `xla/enhance/sr_pairs.py`, phase 11, gọi `degrade` của dòng 1) |
| 3 | Nạp VisDrone + lọc nadir | EXTEND | `load_visdrone` @ xla/degrade/sources.py:99, `is_nadir` @ :243, `cmd_fit_probe` @ :508, `configs/visdrone_view.yaml`; delta: `load_heridal`, `load_tinyperson`, `load_wisard` + nhãn góc nhìn từng bộ |
| 4 | Nhập khung bay thật + cửa sổ chuyến | EXTEND | `extract_flight` @ xla/camera/frames.py:188, `parse_flight_events` @ :89, `FlightWindow` @ :69, `extract_neg` @ :229; delta: mọi bản ghi `logs/recordings/*`, trường `session` |
| 5 | Chia dev/test theo khối 30 s | REUSE | `xla/camera/split.py`, `data/real/split.json` (1115/198/300, chỉ đọc) |
| 6 | Gán nhãn trước (RT-DETR-x + YOLO11x + SAHI, WBF) | REUSE | `run` @ xla/detect/prelabel.py:169, `fuse` @ :80 |
| 7 | Soát nhãn bằng ảnh ghép + áp sửa | REUSE | `cmd_render` @ xla/detect/review.py:152, `cmd_apply` @ :245 |
| 8 | Trang ảnh cắt để soát | EXTEND | `cmd_crop` @ xla/detect/review.py:173; delta: chế độ hard-negative (vùng cắt hộp báo nhầm trên mine) và trang duyệt holdout |
| 9 | Ngưỡng F1 tối đa, chặn rò test | EXTEND | `select_f1_threshold` @ xla/eval/detection.py:254-271 (`TestLeakError`); delta: nhận dự đoán OOF trên `devpool`, chặn cả khung cal + holdout |
| 10 | Bộ chấm nhận diện, báo nhầm/khung âm, bootstrap khối | EXTEND | `NEG_KINDS` @ xla/eval/detection.py:277, `fp_per_neg_flight` :27, :513; delta: loại `cal`, `holdout` đọc `neg_flight_split.json` |
| 11 | Fine-tune YOLO, chấm mỗi epoch, resume | EXTEND | `EpochTracker` @ xla/detect/train.py:128-186, resume @ :214-224, chặn khung thật @ :84-86; delta: danh sách cho phép khung thật (§F), ảnh nền (mine, nhãn rỗng), loại fold, dự đoán OOF |
| 12 | Thầy + nhãn giả + làm sạch bằng ByteTrack | REUSE | `xla/detect/selftrain.py` (`_bytetrack_without_lap` @ :253, `track_sequences` @ :296), `configs/selftrain_e2.yaml` (`pseudo.tracker`); chỉ đổi trọng số thầy + nguồn |
| 13 | Xuất detector ONNX + bundle (tiền/hậu xử lý trong graph) | REUSE | `xla/detect/export_onnx.py:5-26` (`--version`, `--conf`, `--imgsz`); delta duy nhất: ghi `recommended_track` vào meta |
| 14 | Kiểm bundle theo hợp đồng | EXTEND | `validate_bundle` @ xla/contract.py:394, `BUNDLE_DIR_RE` @ :82, `CONTRACT_VERSION` @ :35; delta: kind `verifier`, `enhancer-offline`, 1.1 |
| 15 | Khoá GPU | REUSE | `gpu_lock` @ xla/gpu_lock.py:76 (`configs/paths.yaml:13`) |
| 16 | Ảnh rõ cổ điển | REUSE | `enhance` @ xla/enhance/classic.py:450, `load_params` @ :89, `configs/enhance_classic.yaml` (SSOT; graph ONNX phase 10 là đường dẫn xuất, khoá bằng parity) |
| 17 | Xuất enhancer ONNX + parity | EXTEND | `xla/enhance/export_onnx.py:5-12, 225` (opset 17, fp16, MAE ≤ 1,0); delta: không sửa file này; graph classic (`export_classic.py`, phase 10) và ×2 offline (`sr.py`, phase 11) gọi lại hàm ghi bundle/golden của nó |
| 18 | Huấn luyện mạng khôi phục | EXTEND | `xla/enhance/train_net.py` (`last.pt` mỗi 200 bước, resume, bf16; `gpu_lock` @ :234), ứng viên @ xla/enhance/net.py:42; delta: đích ×2, loss perceptual + GAN tuỳ chọn |
| 19 | Kiến trúc vendor | EXTEND | `xla/enhance/arch/{nafnet,fbcnn,scunet}.py`; delta: `rrdbnet.py` (BSD-3), `swinir.py` (Apache-2.0), `basicvsrpp.py` (Apache-2.0, phase 12) |
| 20 | Đo nhấp nháy | REUSE | `xla/eval/temporal.py` |
| 21 | Trang thử mù A/B | EXTEND | `build` @ xla/eval/ab_page.py:175, `plan_pairs` @ :84, `pick_region` @ :135; delta: nhiều so sánh một trang, ảnh 1280×960, phóng Lanczos cho biến thể 640×480 |
| 22 | Chấm A/B, Wilson | EXTEND | `wilson` @ xla/eval/ab_score.py:37, `score` @ :60, `CRITERIA` @ :28-29; delta: tiêu chí cận dưới > 0,5 theo từng so sánh (D13) |
| 23 | Một lệnh tái lập + đồng bộ báo cáo | EXTEND | `build` @ scripts/reproduce.py:302, `sync_report` @ :323 (khối `<!-- bang: -->` trong `BAO_CAO_DO_AN.md`); delta: bảng t7–t9 v2 |
| 24 | Nạp bundle ở IOT-CV (sha, meta, IO, warm-up, provider) | EXTEND | `load_bundle` @ d:/Coding/IOT-CV/backend/vision/model_bundle.py:175-260, `IO_SPEC` @ :38-50, `_preload_cuda_dlls` @ :108-137; delta: IO verifier + enhancer-offline, trạng thái `ready` |
| 25 | Model bị từ chối không làm sập backend | REUSE | `VisionRuntime._try` @ backend/vision/runtime.py:53-66 (sự kiện `vision.model_rejected`) |
| 26 | Xác nhận trước khi báo sự kiện | EXTEND | `EventThrottle.should_emit` @ backend/vision/events.py:62-72 (N khung liên tiếp, cooldown); delta: thành lớp tương thích gọi `TrackFilter` theo track, k/n (`track_filter.py`) |
| 27 | Sự kiện `vision.person_detected` | EXTEND | backend/vision/runtime.py:88-94 (detail `confidence, uav_lat, uav_lon, uav_alt`); delta: `track_id`, `hits`, `window`, `verifier_score` |
| 28 | Đo ms từng bước | EXTEND | backend/vision/pipeline.py:34-35, 245-260 (`decode, detector, enhancer, encode, total`); delta: bước `verifier` |
| 29 | Ảnh AI cho Lịch sử bay + cache | EXTEND | `FlightFrameAIService` @ backend/flights/ai.py:82-95, `_enhance_now` @ :149-172, LRU @ :26; delta: job offline theo lô + cache đĩa |
| 30 | API khung lịch sử bay | EXTEND | `GET /api/flights/frame` @ backend/api.py:277-313 (`variant` @ :281); delta: `variant=offline`, route job §D |
| 31 | Schema trạng thái model | EXTEND | `VisionModelStatus`, `VisionStatus` @ backend/schemas.py:200-218; delta: `verifier`, `enhancer_offline`, `enhancer_default_on` |
| 32 | Nút "Ảnh rõ" trực tiếp | EXTEND | `VideoPanel.tsx:41-57, 199-219` (localStorage `iot-cv.video-enhanced`); delta: mặc định theo `enhancer_default_on` khi người dùng chưa chọn |
| 33 | So gốc ↔ ảnh rõ trên Lịch sử bay | EXTEND | `loadCompare` @ frontend/src/components/flights/AiCompare.tsx:62-92, khoá localStorage @ :33-34; `AiViewer.tsx` (chỉ hiển thị); delta: nguồn `offline`, tiến độ job, nhãn "ảnh do AI dựng lại" |
| 34 | Ô nhãn–giá trị của kit | REUSE | `KeyValue` @ frontend/src/components/kit/index.tsx:181-243, `KeyValueList` @ :249, `lib/tone.ts` |
| 35 | Chép + kiểm bundle ở IOT-CV | EXTEND | `scripts/vision/lay_model.py:29-69`, `scripts/vision/kiem_bundle.py:27-131`; delta: tên env `-`→`_`, golden verifier/offline |
| 36 | Phát lại bản ghi qua pipeline, đo ms | EXTEND | `scripts/vision/replay_bench.py` (`BUDGET_MS` @ :49, log `logs/vision-bench`); delta: `replay_pred.py` dùng chung vòng phát lại, xuất CSV §E |
| 37 | Cắt ô kiểu SAHI cho offline | REUSE | `repo: obss/sahi` (MIT) như v1 (gán nhãn trước, `plans/261118-.../plan.md` Deferred 9); chỉ chấm trong XLA (phase 06) |
| 38 | Khung sườn `ml/scripts` cũ | REPLACE | `ml/scripts/*.py` (6 file, toàn `NotImplementedError`) đã được module XLA thay (v1 Reuse #29); tham chiếu ngoài `ml/`: chỉ tài liệu (phase 16) |
| 39 | Bộ phân loại xác minh từng hộp (vùng cắt 64–96 px) | NEW | không có bộ phân loại vùng cắt nào; `review.cmd_crop` chỉ làm ảnh soát, `detector.py` chỉ lọc theo điểm<br>reuse-search: not-found-after-corpus-and-grep<br>scope: xla, configs, repo:IOT-CV (backend/vision, ml) \| swept: 2026-11-25 |
| 40 | Siêu phân giải offline ×2 (RRDBNet / SwinIR) fine-tune D2 | NEW | repo chỉ có NAFNet/FBCNN/SCUNet cùng cỡ (`xla/enhance/arch/`); bản Real-ESRGAN-anime cũ ở `models/` của nhóm bị bỏ theo thiết kế v1<br>reuse-search: not-found-after-corpus-and-grep<br>scope: xla/enhance, models, src, repo:IOT-CV (backend) \| swept: 2026-11-25 |
| 41 | Khôi phục video nhiều khung (BasicVSR++) | NEW | không có mạng nhiều khung nào; `temporal.py` chỉ đo nhấp nháy<br>reuse-search: not-found-after-corpus-and-grep<br>scope: xla, src, repo:IOT-CV (backend) \| swept: 2026-11-25 |

Nguồn sự thật bên ngoài (fact-fetcher 25/11, URL kiểm 200): SARD cần thuê bao IEEE DataPort trả phí (kiểm 25/11 khi đã đăng nhập) → bỏ; HERIDAL CC BY 3.0 (Zenodo 5662351, 7,74 GB, md5 khớp); WiSARD MIT; TinyPerson
không ghi giấy phép dataset (mã MIT); VisDrone không ghi giấy phép; máy chủ gốc HERIDAL từ chối kết nối (dùng bản Zenodo); Ultralytics AGPL-3.0;
Real-ESRGAN BSD-3 (có `RealESRGAN_x2plus.pth` v0.2.1); SwinIR Apache-2.0 (có `SwinIR-M_x2` real GAN/PSNR); BasicVSR++
Apache-2.0; RVRT CC-BY-NC (không dùng, D8 + Deferred 11); SAHI MIT; ByteTrack MIT; timm Apache-2.0 (trọng số ImageNet
chỉ cho nghiên cứu, áp cho cả torchvision). Đồ án là nghiên cứu, báo cáo ghi rõ (phase 17).

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Không đủ phiên bay ngoài `20261102-100941` cho holdout | medium | high | Phase 03 bước 1 đếm trước mọi phase GPU; Kill (a); human step tuỳ chọn ghi phiên trống; báo CI, không nói quá |
| Rò holdout/test vào huấn luyện, khai thác, ngưỡng, quyết định ship | medium | high | D5; danh sách cho phép ở `train.py`; `test_sessions_neg_split.py` cấm holdout/cal trong manifest huấn luyện; `select_f1_threshold` chặn cal + holdout; `replay_pred.py --blank` + xoá track ở biên đoạn (red-team 25/11: test xen kẽ dev trong `100941`) |
| Verifier + track hạ recall dưới 0,6 | high | medium | D7 chọn điểm vận hành minh bạch; báo đường cong báo nhầm–recall; ngưỡng ghi đè được bằng env |
| Giấy phép dữ liệu ngoài chỉ cho nghiên cứu / không rõ | medium | medium | Phase 02 ghi giấy phép từng bộ vào `configs/sources.yaml`; ảnh ngoài không đẩy lên repo (chỉ manifest + mã tải); báo cáo ghi rõ |
| SR bịa chi tiết / người giả | medium | high | Hàng rào báo nhầm holdout ở phase 14 trước A/B; nút xem gốc luôn có; nhãn "ảnh do AI dựng lại"; Kill (d) |
| VRAM 8 GB không đủ (thầy l @1280, SR, video) | medium | medium | Smoke ở bước 1 của 05, 11, 12; hạ về m / tile trong graph / bỏ phase 12 |
| Parity ONNX classic không đạt | medium | medium | Phase 10 parity từng tầng; tầng không đạt viết lại bằng op tương đương và ghi sai số; A/B chấm đúng bundle ship |
| Nhiều phiên cùng sửa file IOT-CV / XLA | medium | high | D9 worktree riêng; D10 sở hữu file; `git status` + nhắn phiên đang sống |
| Máy sập giữa job dài | medium | medium | ≤ 4 worker, checkpoint mỗi epoch/200 bước, resume; job tách tiến trình, log ra file |
| Job offline chiếm GPU lúc bay | low | high | §D: job từ chối/tạm dừng khi `armed`; một job một lúc |

## Self-improvement loop

- **Signal:** `tests/contract/test_model_contract.py` (XLA) và `backend/tests/test_vision_v2.py` (IOT-CV) đỏ khi bundle lệch hợp đồng v1.1; lúc chạy IOT-CV phát `vision.model_rejected` kèm lý do; `test_sessions_neg_split.py` đỏ khi khung holdout lọt vào manifest huấn luyện/khai thác; ô trạng thái model trên web hiện `rejected` + lý do
- **Loop:** CI pytest có sẵn của hai repo (`.github/workflows/tests.yml` XLA, Dockerfile CI IOT-CV) trên mọi PR, và mục "Bẫy đã gặp" trong `AGENTS.md` IOT-CV + memory agent mà mọi phiên đã đọc
- **Fix:** sửa `docs/hop-dong-model.md` (tăng `contract_version`), xuất lại bundle, ghi bẫy mới vào `AGENTS.md` mục "Bẫy đã gặp" và memory
- **Guard:** `golden/` mỗi bundle kiểm ở cả hai repo (`xla.contract validate`, `kiem_bundle.py`); test rò holdout chạy trong CI CPU (`-m "not gpu and not data"` phải có bản dữ liệu giả)

## Deferred decisions

| # | Question | Options | Choice taken | Why | How to reverse |
|---|----------|---------|--------------|-----|----------------|
| 1 | Tầng 3 cài ở đâu | ultralytics BYTETracker trong IOT-CV / numpy ghép IoU trong IOT-CV / ở XLA | numpy ghép IoU tham lam trong IOT-CV `backend/vision/track_filter.py` (hộp đã lọc ngưỡng nên không có tập điểm thấp cho kiểu hai ngưỡng), XLA gọi qua `replay_pred.py` (D4) | IOT-CV không có tracker nào (grep 0 hit); ByteTrack hiện có chỉ ở XLA và kéo ultralytics + torch; AGENTS.md: không chép code hai nơi | Thay bằng `BYTETracker` của ultralytics nếu phase 09 thấy bản numpy kém hơn trên mine |
| 2 | Verifier cắt vùng ở đâu | IOT-CV cắt / RoiAlign trong graph | Trong graph (D2) | Một SSOT cho phép cắt như letterbox của detector v1 | Sửa `contract.md` §A, tăng `contract_version` |
| 3 | "Ảnh rõ" classic tới IOT-CV bằng gì | ONNX `enhancer-v2.0.0` / IOT-CV import `xla` / chép mã | ONNX qua hợp đồng enhancer §3 có sẵn, parity MAE ≤ 1,0 với numpy | IOT-CV chỉ nhận ONNX (AGENTS.md luật 10); `Enhancer` IOT-CV không phải sửa | Phase 10 Kill: nếu parity không đạt mà A/B classic thắng, Owner mở plan riêng |
| 4 | `contract_version` | 2.0 / 1.1 | 1.1 (D1) | Chỉ thêm kind và trường tuỳ chọn; bundle v1 vẫn nạp | Nâng 2.0 nếu lane nào buộc IOT-CV đổi hành vi bắt buộc |
| 5 | Holdout lấy phiên nào | phiên của `100941` / bản ghi khác | Ưu tiên phiên của bản ghi khác `20261102-100941` (01/11 tối và bản ghi mới); phiên nào có khung dev/test/huấn luyện không vào holdout. Dưới 150 khung thì vẫn chấm, báo "không đủ chứng minh" | Holdout phải chưa từng chạm huấn luyện; dev v1 phủ mọi khối không-test của `100941` | Chạy lại phase 03 khi có phiên mới (human step tuỳ chọn) |
| 6 | k-fold chia theo gì | phiên / khối 30 s | GroupKFold theo phiên có khung `devpool`; < 3 phiên thì nhóm khối 30 s liền nhau thành 5 fold | Thiết kế §3.B5; dev hiện từ một bản ghi | Sửa `folds.json` (phase 04) |
| 7 | Điểm vận hành | ưu tiên báo nhầm / recall | Báo nhầm cal ≤ 0,03, max recall (D7) | Lỗi gốc v1 là báo nhầm (túi rác thành người) | Env `VERIFIER_CONFIDENCE`, `DETECTION_TRACK_K/N` |
| 8 | Cỡ thầy | YOLO11l / m | l nếu smoke vừa batch ≥ 2 @1280 + AMP trong 8 GB, không thì m | v1 batch 8 YOLO11s đã tràn 8,52 GB | Đổi `configs/teacher_v2.yaml` |
| 9 | Ứng viên trò | nhiều / ≤ 3 | C1 D0+D2 trộn (VisDrone + ngoài) + nền mine; C2 = C1 + nhãn giả dev; C3 = D0 + nền mine (giống E1-D0) | E1-D0 v1 tốt nhất trên test; k-fold chỉ cần huấn luyện lại theo fold cho C2 | Thêm ứng viên trong `configs/student_v2.yaml` |
| 10 | Backbone verifier | timm / torchvision | torchvision `mobilenet_v3_small`, `efficientnet_b0`, crop 64 và 96 chọn bằng k-fold | Không thêm gói (D8) | Thêm timm sau khi Owner duyệt gói |
| 11 | Ứng viên offline | Real-ESRGAN / SwinIR / BasicVSR++ / RVRT | Real-ESRGAN x2plus (zero-shot + fine-tune D2), SwinIR-M x2 real (zero-shot); BasicVSR++ là phase 12 tuỳ chọn; không RVRT | RVRT CC-BY-NC, không có mã huấn luyện trong mmagic, VRAM nặng | Thêm RVRT bằng phase mới |
| 12 | Cỡ ra offline | ×1 / ×2 | ×2 (1280×960) (D3) | Thiết kế §3.A2 | Xuất lại ở ×1 nếu A/B cho thấy ×2 không giúp |
| 13 | Loss fine-tune SR | L1 / +perceptual / +GAN | L1 + perceptual + GAN trọng số 0,1 (mặc định fine-tune Real-ESRGAN), cấu hình được; thêm biến thể không GAN để so | Giữ độ nét mà vẫn có bản an toàn; hàng rào báo nhầm chặn bịa | `configs/sr_x2.yaml` |
| 14 | Cache offline ở IOT-CV | trong repo / D:\DevCache | `FLIGHT_AI_CACHE_DIR`, mặc định `logs/flights-ai/` + dòng `.gitignore` | Cùng chỗ với `logs/recordings`, `logs/vision-bench`; Owner cho toàn quyền IOT-CV 25/11 | Đặt env sang `D:\DevCache\iotcv\flights-ai` |
| 15 | Cắt ô SAHI cho offline | tích hợp IOT-CV / chỉ đo | Chỉ đo trong XLA (phase 06), không tích hợp IOT-CV trong plan này | YAGNI tới khi có số đo thắng | Plan sau nếu cận dưới CI hiệu > 0 |
| 16 | Lai classic → NAFNet | luôn thử / có giới hạn | Tối đa 2 giờ GPU ở phase 10; vào A/B chỉ khi hơn classic trên dev (dark channel, nhấp nháy) | Thiết kế: chỉ giữ khi thắng | Bỏ bước trong phase 10 |
| 17 | Mặc định "Ảnh rõ" | luôn bật / theo A/B | Bật: phase 15 đặt `ENHANCER_DEFAULT_ON = true` cho classic; A/B classic vs gốc chỉ là chỉ số §5, không đổi mặc định; lựa chọn đã lưu của người dùng luôn thắng | Owner đã chốt (brainstorm §3.A1); gắn cổng A/B là mở lại quyết định (list-classifier R6 25/11) | `.env` `ENHANCER_DEFAULT_ON=false` |
| 18 | Ô trạng thái model đặt đâu | trang Bay / trang Vision | `VisionPage.tsx` | Trang Bay phải vừa một màn hình không cuộn (AGENTS.md) | Chuyển component sang trang khác |
| 19 | Release `models-v2.0.0` | có / không | Tạo như v1 D21 | Owner cho toàn quyền XLA (AGENTS.md luật 10) | Xoá Release trên GitHub |
| 20 | HERIDAL | máy chủ gốc / bản sao | Bản Zenodo 5662351 (CC BY 3.0, keras-retinanet VOC), agent tải bằng aria2c 8 luồng, md5 khớp | Máy chủ gốc từ chối kết nối 25/11 | — |
| 21 | Tải TinyPerson | agent / chủ dự án | Chủ dự án tự tải 4 file (train/test.tar.gz + tiny_set_train/test.json), xong 25/11 | Chủ dự án muốn tự bấm tải file lớn | — |
| 22 | Ô trạng thái `enhancer-offline` | nạp khi khởi động / kiểm không nạp | Kiểm sha + meta + mở session **CPU** không chạy rồi đóng (state `ready`, màu vàng); session GPU chỉ mở khi job chạy | Không chiếm VRAM luồng trực tiếp; sha + meta không chứng minh graph mở được (red-team 25/11) | Mở session GPU sẵn nếu job khởi động chậm |
| 23 | Box vẽ trên luồng trực tiếp | trước / sau verifier | Sau verifier; sự kiện sau track | Người lái thấy đúng thứ hệ thống tin; payload `detection` không đổi | Thêm cờ hiển thị hộp chưa xác minh |
| 24 | Plane Work-Item Gate | chạy / bỏ | Bỏ hẳn (opt-out ghi nhận) | Chủ dự án cấm (luật chủ máy + AGENTS.md luật 4) | Không đảo ngược |
| 25 | Commit báo cáo brainstorm | có / không | Commit cùng plan (`plans/reports/261125-brainstorm-cai-thien-ai-v2.md`) | Là SSOT plan trỏ tới; worktree lane đọc được | `git rm --cached` |
| 26 | Chia khung bay không người | mine / holdout / ba phần | Ba phần theo phiên: `mine` (huấn luyện), `cal` (hiệu chỉnh + hàng rào), `holdout` (chấm); thiếu phiên thì `cal` = khối 30 s trong phiên mine | Hiệu chỉnh trên chính khung đã huấn luyện làm biên 0,03 / 0,05 vô nghĩa (red-team 25/11) | Gộp `cal` vào `mine` trong `neg_flight_split.json` |
| 27 | A/B v2 lấy khung nào | test / dev | Dev | A/B quyết định ship (lai, offline) là chọn model, không làm trên test (D10 v1) | Đổi `ab_page --split` |
| 28 | Hàng rào báo nhầm ảnh offline đo trên | holdout / cal | Quyết định trên `cal`, holdout chỉ ghi vào meta + báo cáo; không có ứng viên dự phòng sau khi bị loại | Holdout không được chọn gì (D5) | Sửa `guard_fp` |
| 29 | Phần web IOT-CV | gộp phase 13 / phase riêng | Phase 18 riêng, Accept bằng vitest | Một Accept pytest không chứng minh được web (list-classifier R1) | Gộp lại |
| 30 | Recall OOF khi C2 thắng | dùng model cuối / model từng fold | Xuất ONNX từng fold, phase 09 đo recall fold i bằng detector fold i | Model cuối đã thấy nhãn giả của mọi fold | Bỏ khi C1/C3 thắng |
| 31 | Bỏ SARD, thêm WiSARD sample | SARD / WiSARD bản đủ / WiSARD sample | SARD bỏ (cần thuê bao IEEE trả phí, không bản sao công khai); WiSARD bản đủ 40,5 GB chủ dự án thấy quá nặng → dùng Multi-Modal Sample 971 MB (264 ảnh RGB, MIT); nguồn nền khó chính là HERIDAL | Kiểm 25/11 | Thêm SARD lại nếu có thuê bao IEEE |
| 32 | TinyPerson dùng vào đâu | `d02x` / chỉ verifier | Chỉ làm mẫu dương cho verifier (`use: [verifier]`, `d02x: false`), 725 hộp | Sau lọc góc chỉ còn 173 ảnh (< 200, Risk phase 02), phần lớn video bãi biển thấy chân trời (`results/v2/data/README.md` §3) | Đặt `d02x: true` trong `configs/sources.yaml`, dựng lại `d02x` |
| 33 | Trần giờ GPU từng phase | theo plan (05 6 h, 06 6 h, 07 1 h, 11 4,5 h, 12 4 h, lai 2 h) / trần chủ dự án | Trần chủ dự án 25/11 (vì tốc độ): 05 **1 h**, 06 **2 h** (nới 45 phút một lần nếu val còn lên, không dùng), 07 **30 phút**, 11 **1,5 h**, 12 **1 h**, lai classic → NAFNet (phase 10) **30 phút**. Đã dùng: 05 0,834 h, 06 1,314 h, 07 0,175 h, lai 25,0 phút, 12 0,013 h | Chủ dự án quyết; số dùng ghi ở README từng lane | Chạy lại lane với trần mới (`kfold extend`, GAN SR thêm giờ) |
| 34 | Owns `detector-v2.0.0/meta.json` (phase 06, 09) khi Kill (b) | giữ tên / bundle mới | Không có `detector-v2.0.0`; `recommended_track` ghi vào bundle mới **`detector-v1.0.1`** (cùng `detector.onnx` sha256 `23b1da69…aef` + golden của v1.0.0, meta thêm `recommended_track`, `contract_version` 1.1). Owns của phase 09 đổi thành `models/release/detector-v1.0.1/**` | Bundle đã phát hành `detector-v1.0.0` không được sửa (phiên chính / chủ dự án 25/11, `results/v2/system/README.md` §2) | IOT-CV đặt `DETECTOR_ONNX` về `detector-v1.0.0` |
| 35 | B2 khi holdout chặt rỗng | bỏ trống / holdout yếu cùng buổi | Thêm holdout yếu (chủ dự án 26/11): khung bay không thuộc `split.json` v1, cách mọi khối dev/test v1 ≥ 30 s, agent duyệt, nhãn "không độc lập". Kết quả **0 khung** (1 602 khung ngoài split đều cách khối < 10 s) → B2 vẫn "không đủ chứng minh" | Không bay thêm được trước hạn; holdout yếu không thay holdout chặt (`results/v2/system/README.md` §5b) | Nới luật 30 s (tập gần trùng cảnh test) hoặc bay buổi bãi trống |
| 36 | Lượt A/B nào quyết định ship offline | ít nhất một lượt / chỉ lượt cả khung | Chỉ lượt **cả khung** (đúng cách web hiển thị); lượt phóng to chỉ báo cáo. Sửa trước khi chấm, commit `b545b64` (`ab_score.DECISION_RULE`) | "Đạt ở lượt nào cũng được" làm lỏng tiêu chí (`results/ab-v2/README.md` §3) | Sửa `DECISION_RULE` trước một lượt chấm mới, không sau khi đã thấy kết quả |
