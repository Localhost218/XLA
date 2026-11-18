# Plan: AI xử lý ảnh chất lượng thấp cho camera drone IOT-CV (ESP32-S3 + OV3660)

Hardness: phases=18 new=6 modules=0 lanes=9
Flags: --auto
PRs: 3

Ngày lập: 18/11/2026. Thiết kế đã duyệt (SSOT mục tiêu, kiến trúc, tiêu chí §8, mốc §9):
`plans/reports/261118-brainstorm-ai-anh-chat-luong-thap.md`. Bằng chứng số đo: `plans/reports/261118-bang-chung/`
(`video/phan-tich-video.md`, `kiem-xla/kiem-xla.md`, `nghien-cuu/nghien-cuu-tai-lieu.md` và các script `.py`).
Hợp đồng chung XLA ↔ IOT-CV: `contract.md` (cạnh file này; phase 02 chuyển thành `docs/hop-dong-model.md`).

Plan này viết trước khi có code (→ 18/11/2026: phase 01–09 đã thực thi; trạng thái và các quyết định đổi so với plan ở `thuc-thi-phase-01-07.md` §6–7. → 19/11/2026: phase 10–14 xong, trạng thái + kết quả cuối ở `thuc-thi-phase-10-14.md` §7; còn phase 15–18). → 19/11/2026: phase 15–18 xong, kết quả ở `thuc-thi-phase-10-14.md` §10. Mỗi phase là một file tự đủ: agent chỉ cầm `phase-NN-*.md` + `contract.md` là bắt đầu được.

## Forecast

Cap: 70 % tối đa cho mọi dòng khi giả định chính (bản sao số camera D2 giống khung thật hơn D1) còn UNMEASURED.

| Outcome (đo được) | Confidence | Basis |
|---|---|---|
| B1: mAP50 trên tập test thật ≥ 1,5 × mốc YOLO11s COCO @1280 | 60 % | ESTIMATED: mốc precision chỉ 41 % (MEASURED, `video/phan-tich-video.md`), 43/128 hộp sai do dải mép (cắt 16 px là bỏ được) + fine-tune trên ảnh suy giảm là đối chứng mạnh nhất trong tài liệu (`nghien-cuu-tai-lieu.md` §3). → đổi 18/11/2026 sau M2: mốc `edge16` mAP50 0,126 [0,030–0,224] trên 39 khung / 149 hộp đã duyệt; tiêu chí mới mAP50 ≥ 0,227 **và** cận dưới CI hiệu ghép cặp > 0 (`results/baseline/tieu-chi-sau-m2.md`). → kết quả 18–19/11/2026: **không đạt** với model ship (bundle `detector-v1.0.0` = E2 r1, đường ONNX: 0,120 [0,032–0,197], hiệu với `edge16` −0,007 [−0,138; 0,067]); E1 best D2 0,236 nhưng cận dưới CI hiệu −0,017; chỉ E1 D0 đạt cả hai vế (0,297; cận dưới +0,016) nhưng không được chọn vì luật chọn trên dev-val (D10) (`results/e1/README.md` §2, `results/e2/README.md` §2) |
| B2: báo nhầm ≤ 0,05 hộp/khung trên khung không người | 50 % | ESTIMATED: 28 hộp sai do đốm hồng/trắng trên 300 khung (MEASURED) cần mẫu âm + self-training mới hạ được. → kết quả: **không đạt** — bundle 0,881 [0,333–1,495] hộp/khung bay không người; thấp nhất là E1 D2 0,302 [0,218–0,376], vẫn gấp 6 lần mục tiêu (`results/e2/README.md` §2) |
| B3: detector ≤ 30 ms/khung trên RTX 4060 | 60 % | UNKNOWN cho ONNX Runtime CUDA EP trên Python 3.13 Windows: insufficient evidence; 60 % là ước theo cỡ YOLO11s @1280 (ESTIMATED). → kết quả: **đạt** — bundle `detector-v1.0.0` ORT CUDA EP (`onnxruntime-gpu` 1.24.4, venv XLA Python 3.12) p50 19,985 / p95 20,223 ms, 300 khung, không hạ xung (`models/release/detector-v1.0.0/meta.json`); Python 3.13 của IOT-CV vẫn kiểm ở phase 15 |
| A1: % điểm hồng giảm ≥ 90 %, dark channel trung vị giảm ≥ 50 % | 65 % | ESTIMATED: Stage 1 của nhóm đã khử hồng tốt trên khung thật (MEASURED, `kiem-xla.md` §6). → kết quả trên test (mạng `enhancer-v1.0.0`): hồng −99,9 % **đạt**, dark channel trung vị −36,8 % [−45,9; −31,0] **không đạt** → dòng **không đạt** (cổ điển ngược lại: hồng −89,2 %, dark channel −51,6 %) (`results/ab/README.md` §4.2) |
| A2: thắng ≥ 70 % lượt A/B so với ảnh gốc | 50 % | ESTIMATED: chưa có mạng nào chạy trên khung này. → kết quả: **không đạt** — lần 1 (cả khung) 11/25 = 0,44 [0,267; 0,629]; lần 2 (phóng to 2×) 13/25 = 0,52 [0,335; 0,700]; CI cả hai lần chạm 50 % (`results/ab/README.md` §4.1, `results/ab-zoom/README.md`) |
| A3: thắng ≥ 55 % lượt A/B so với đối chứng cổ điển | 40 % | ESTIMATED: tài liệu thống nhất khôi phục tối ưu PSNR thường không hơn rõ (`nghien-cuu-tai-lieu.md` §3). → kết quả: **không đạt** — lần 1 1/25 = 0,04 [0,007; 0,195]; lần 2 5/25 = 0,20 [0,089; 0,391]; CI trọn dưới 50 %, mạng thua rõ |
| A4: enhancer ≤ 30 ms/khung, nhấp nháy không tăng | 55 % | ESTIMATED: chưa đo tốc độ NAFNet/FBCNN ở 624×480. → kết quả: **không đạt** — tốc độ đạt (p50 20,7 ms trong `enhance_eval all`, 11,1 ms đo riêng; bundle p95 21,0 ms) nhưng nhấp nháy tăng 1,81 [1,74; 1,92] > 1 (`results/ab/README.md` §4.2) |
| Dữ liệu: KID(D2, thật) < KID(D1, thật), CI không chồng nhau | 45 % | ESTIMATED: Sensor Transfer +1,66/+5,35 AP khi tham số học từ ảnh đích (DOCUMENTED, `nghien-cuu-tai-lieu.md` §2); chưa ai so trực tiếp cho một camera. → đo 18/11/2026 (phase 06–07, khung test): KID D1 4,44 / D2 4,37 (CI chồng D1, **chưa đạt**) / D2+RealSR 3,93 (CI không chồng, **đạt**); thống kê mức thấp D1 5,93 / D2 0,81 / D2+RealSR 0,66 (`results/realism/d1_vs_d2.json`) |
| Tái lập: một lệnh dựng lại bảng + hình | 70 % | ESTIMATED: chỉ là kỷ luật seed + manifest, khung sinh dữ liệu có seed đã có ở `src/data_augmentation.py:162` |

Success rate: M0 hạ tầng 85 % · M1 bộ làm xấu 70 % · M2 tập test 75 % (cần chủ dự án duyệt) · M3 nhận diện 55 % · M4 ảnh rõ 45 % · M5 tích hợp 65 % · M6 báo cáo 85 %. Overall (nhân chuỗi bảy mốc) ≈ 6 %, dưới mắt xích yếu nhất (M4, 45 %); riêng nhánh B (M0 → M3) ≈ 25 %. Đây là xác suất mọi việc chạy đúng ý ngay lần đầu, không phải xác suất có kết quả báo cáo được.
Effect (low/likely/high): giờ GPU 34 / 50 / 84 giờ (ESTIMATED = cộng khoảng ước của các phase, chưa xác minh: mọi phase huấn luyện có bước smoke đo giờ thật trước) · lịch theo đường găng 12 / 16 / 24 ngày làm việc, tổng công ~30 ngày-agent (ESTIMATED, cộng Effort các phase) · chất lượng: precision trên khung bay 41 % → 70–85 % (ESTIMATED). May be slower: tải NOMAD (Google Drive, 10+ GB, chưa đo), dựng `dgm-eval` + DINOv2 trên Windows, onnxruntime-gpu trên Python 3.13 (UNKNOWN). → 18/11/2026: NOMAD bị loại (ảnh nhìn xiên), `dgm-eval` không cài được (thay DINOv2 qua `torch.hub` + KID tự viết, Owner duyệt trước); `onnxruntime-gpu` 1.24.4 chạy CUDA EP trong venv XLA (Python 3.12) chỉ khi `import torch` trước, 1.32 đòi CUDA 13; Python 3.13 của IOT-CV vẫn kiểm ở phase 15. Phase 07 thực tế ~0,1 giờ GPU (hiệu chỉnh chạy CPU). → giờ GPU thật phase 10–14 (giữ khoá, đọc từ results/): 10: 2,85 (`e1.json`; thêm ~0,5 giờ lần chạy hỏng, ước theo log); 11: 2,49 (`e2.json`); 12: 0; 13: ~1,7 (vòng 1 1,03 + vòng 2 ~0,67, `results/restore/README.md` §6, §8; chọn checkpoint/xuất ngoài số này); 14: không đo. Cộng ≈ 7,0 giờ, so với trần đêm 18/11 ~7,25 giờ (`thuc-thi-phase-10-14.md` §4).
Uncertainties (by impact): 1. Tập test quá ít người (ước ~370 khung có người trong 2 944, MEASURED trên mẫu 300) → phase 03 đếm thật trước khi chia, 1 giờ CPU, trước phase 08. → đếm 18/11/2026: trần khả thi 362 khung test / 74 có người; chia lại test 198 khung, nhãn duyệt 39 khung có người (phase 03 mục "Quyết định chia lại"). 2. ONNX Runtime CUDA EP không chạy trên Python 3.13 Windows → phase 15 bước 1 cài thử + đo với model giả, 30 phút, trước phase 16. 3. Giờ GPU huấn luyện YOLO @1280 → smoke 1 epoch trên 10 % dữ liệu ở phase 10 bước 1, 30 phút. → giải quyết 18/11/2026: smoke batch 4 ước 0,119 giờ/epoch, đo thật 0,106–0,120; giữ imgsz 1280 batch 4 (batch 8 tràn VRAM 8,52 GB), 6 epoch cho cả bốn tập, 40–46 phút/lần, E1 tổng 2,85 giờ; Kill (b) không kích hoạt (`results/e1/README.md` §4–5). 4. Mạng khôi phục không kịp 30 ms → phase 13 bước 1 đo tốc độ 3 ứng viên bằng trọng số có sẵn, 1 giờ. → giải quyết 18/11/2026: cả ba bản công bố đều > 30 ms (NAFNet-width32 bundle p95 82,1 ms lúc máy không hãm); vòng 2 thu nhỏ thành NAFNet-w16 (1,14 M tham số, huấn luyện từ đầu) → bundle p95 21,0 ms, Accept đạt; Kill (c) không kích hoạt (`results/restore/README.md` §1, §8).
Kill / no-regression: (a) sau phase 09 mà mốc YOLO11s đã ≥ 0,7 mAP50 thì sửa lại tiêu chí 1,5× (§8 cho phép sửa sau M2) trước khi huấn luyện; (b) phase 10 smoke cho thấy một lần huấn luyện đủ > 12 giờ GPU thì hạ imgsz 1280 → 960 hoặc giảm epoch, ghi Deferred; (c) nhánh A: không ứng viên mạng nào ≤ 30 ms → không xuất `enhancer.onnx`, IOT-CV ẩn nút ảnh rõ (luồng `enhanced` trả 503 đúng hợp đồng), đối chứng cổ điển và mạng chỉ còn là kết quả thí nghiệm trong báo cáo. Rollback: mỗi phase một nhánh/commit riêng trong XLA; IOT-CV chỉ ghi file (không commit) nên `git -C d:\Coding\IOT-CV diff` là toàn bộ thay đổi, hoàn tác bằng `git checkout -- <file>` sau khi chủ dự án đồng ý.
Re-score: 55 % → 60 % (B1) sau báo cáo `nghien-cuu-tai-lieu.md`: fine-tune trên ảnh suy giảm là đối chứng mạnh, JPEG Q62,5 phạt nhỏ (~1–3 mAP). 50 % → 40 % (A3) sau cùng báo cáo: khôi phục tối ưu PSNR thường không hơn. 18/11/2026 sau M2: B1 đổi tiêu chí nên % cũ không còn chấm đúng đại lượng; chưa chấm lại, chờ smoke phase 10. Dòng "Dữ liệu" không còn là dự báo (đã đo, xem bảng). → 19/11/2026: không chấm lại %; B1–B3, A1–A4 đã có kết quả đo (xem bảng), không còn là dự báo: B3 đạt; B1, B2, A1–A4 không đạt.
Not checked: FOV OV3660; bảng DQT ở q ≠ 12 (auto-quality 19–40); VRAM thực khi huấn luyện; tốc độ ORT CUDA EP; giấy phép POG; nguyên nhân dải mép trái và màn mờ; chất lượng nhãn VisDrone ở người nhỏ.

## Finish ETA

`done by 2026-11-18 07:00 UTC+07:00 [parallel; p80-chain; 3 gates x 10m; agent+gates only] n=88 basis=global validated=unvalidated (0 of 10 plans measured; backtest of 10 past plans: took 1.78x longer than this figure, hit 2/10)`
(sequential: 09:25). Con số của CLI chỉ tính thời gian agent + cổng người theo dữ liệu chung của kit; **không** tính giờ GPU
huấn luyện (34–84 giờ, chưa xác minh), thời gian tải dataset, hay lúc chủ dự án rảnh để duyệt. Lịch thực tế theo Forecast:
12 / 16 / 24 ngày làm việc.

## Phases

| # | Phase | Mốc | Lane | Depends-on | Effort | File |
|---|---|---|---|---|---|---|
| 01 | Tái cấu trúc XLA thành package `xla/`, archive phần sai, môi trường, CI | M0 | serial | none | M | `phase-01-tai-cau-truc-xla.md` |
| 02 | Hợp đồng model ONNX + hồ sơ camera: file, hằng số, test, bundle giả | M0 | serial | 01 | S | `phase-02-hop-dong-model.md` |
| 03 | Nhập khung bay thật, ghép độ cao, chia dev/test theo khối thời gian | M0 | data | 01 | M | `phase-03-khung-bay-that.md` |
| 04 | Hồ sơ camera `camera_profile.yaml` v1 đo trên khung dev | M0 | data | 02, 03 | M | `phase-04-ho-so-camera.md` |
| 05 | Bộ máy làm xấu D1 (ngẫu nhiên bậc hai) + D2 (bản sao camera), dữ liệu nguồn | M1 | degrade | 02, 04 | L | `phase-05-bo-lam-xau.md` |
| 06 | Phép đo độ giống thật: thống kê mức thấp, KID-DINOv2, C2ST | M1 | realism | 04 | M | `phase-06-do-do-giong-that.md` |
| 07 | Hiệu chỉnh D2 trên dev, báo KID/C2ST D1 vs D2 | M1 | degrade | 05, 06 | M | `phase-07-hieu-chinh-d2.md` |
| 08 | Nhãn tập test thật: máy gán trước, agent soát, chủ dự án duyệt | M2 | labels | 03 | L | `phase-08-nhan-tap-test.md` |
| 09 | Bộ đánh giá nhận diện + mốc YOLO11s có sẵn, sửa tiêu chí §8 | M2 | labels | 08 | M | `phase-09-moc-so-sanh.md` |
| 10 | Thí nghiệm E1: fine-tune YOLO trên D0/D1/D2 | M3 | detect | 07, 09 | L | `phase-10-tinh-chinh-nhan-dien.md` |
| 11 | Thí nghiệm E2: self-training trên khung thật, xuất `detector.onnx` | M3 | detect | 02, 10 | L | `phase-11-tu-gan-nhan-xuat-onnx.md` |
| 12 | Đối chứng cổ điển viết lại, vector hoá, ≤ 30 ms | M4 | classic | 04 | M | `phase-12-doi-chung-co-dien.md` |
| 13 | Mạng khôi phục riêng camera trên cặp D2, xuất `enhancer.onnx` | M4 | restore | 02, 07 | L | `phase-13-mang-khoi-phuc.md` |
| 14 | Đánh giá nhánh A trên khung thật, E4 khôi phục-trước-nhận-diện, thử mù A/B | M4 | serial | 09, 11, 12, 13 | M | `phase-14-danh-gia-anh-ro-ab.md` |
| 15 | IOT-CV: nạp bundle ONNX, pipeline, luồng ảnh rõ, nút web, camera giả đúng DQT | M5 | iotcv | 02 | L | `phase-15-iotcv-tich-hop-code.md` |
| 16 | IOT-CV: chạy model thật trên bản ghi 02/11, đo ms, kiểm web 1707×900 | M5 | serial | 11, 13, 14, 15 | M | `phase-16-iotcv-chay-model-that.md` |
| 17 | IOT-CV: `plans/ai`, `ml/` trỏ về XLA (không xoá) | M5 | iotcv-docs | 02 | S | `phase-17-iotcv-tro-ve-xla.md` |
| 18 | Báo cáo, slide, README, một lệnh tái lập, PR cuối | M6 | serial | 07, 10, 11, 14, 16, 17 | M | `phase-18-bao-cao-tai-lap.md` |

Đường găng: 01 → 03 → 04 → 05 → 07 → 10 → 11 → 14 → 16 → 18. Phase 08 (cần chủ dự án duyệt nhãn) chỉ phụ thuộc 03 nên
chạy song song với nhánh làm xấu, để bước người làm không nằm trên đường găng. Phase 15 và 17 chỉ cần hợp đồng (02) nên
làm sớm với bundle giả.

## Human steps (do first)

| Step | Unblocks | Where / exact command | Why an agent cannot | Timing |
|---|---|---|---|---|
| ~~Duyệt lần cuối nhãn tập test (~20–30 phút)~~ → xong 18/11/2026: chủ dự án duyệt 39 khung, 149 hộp, cờ hard 5 khung (`data/real/labels/manifest.json`) | phase 08 | Label Studio local `http://localhost:8080`, project `xla-test-v1`, agent mở sẵn đúng trang, nhãn gán trước + đã soát đã nạp; bấm **Submit** từng khung, xong báo "đã duyệt" | Nhãn cuối của tập test là phán quyết của chủ dự án (ràng buộc đã chốt) | later: cần nhãn gán trước + agent soát xong (phase 08 bước 1–4) |
| ~~Chấm thử mù A/B ~55 lượt (~10 phút)~~ → xong 19/11/2026: chủ dự án chấm 55/55 lượt hai lần, cả khung và phóng to 2× (`results/ab/ab_result.json`, `results/ab-zoom/ab_result.json`) | phase 14 | Mở file `results/ab/ab-test.html` trong trình duyệt (agent mở sẵn), chọn ảnh rõ hơn mỗi cặp, bấm **Xuất kết quả** | Cảm nhận của người xem là chính đại lượng cần đo | later: cần ảnh của phase 12 + 13 |
| ~~Bảo commit/PR các file IOT-CV agent đã ghi ở phase 15–17~~ → 19/11/2026 chủ dự án giao toàn quyền trong phiên cook phase 15–18, agent tự commit + mở PR | phase 18 | `git -C d:\Coding\IOT-CV status --short` (phase 18 bước 6: agent đưa danh sách file + kết quả test) | Luật AGENTS.md số 2: IOT-CV chỉ ghi file khi chưa được bảo | later: sau khi phase 16 pass |
| Duyệt xoá các khung sườn cũ `ml/scripts/*.py` của IOT-CV (tuỳ chọn) | phase 17 | Danh sách file + kết quả quét tham chiếu từ phase 17 | Xoá file không tự quyết (luật autonomous) | later: sau phase 17 |
| NVIDIA app → Graphics → Program settings → `python.exe` của IOT-CV (`.venv`) → Power management mode = **Prefer maximum performance**, rồi `uv run python scripts/vision/replay_bench.py --recording logs/recordings/20261102-100941` ở IOT-CV | phase 16 (ngân sách p95 nhịp thật) | Bảng điều khiển NVIDIA là GUI có quyền admin, đổi cấu hình máy của chủ dự án | later: trước buổi bay có bật Ảnh rõ |
| Khi camera bật lại lần tới: cho agent ghi vài giây ở mỗi mức auto-quality (19–40) để kiểm giả thuyết bảng DQT | phase 05 (giả thuyết q ≠ 12) | Bật drone + camera, không cần bay; agent đổi `quality` qua `/set` và đọc khung | Cần cắm pin + bật phần cứng bằng tay | later: lần dùng phần cứng tới, không chặn plan |

## Planning waves

Group A: run
Group B: run
Review: 1 reviewer
Fan-out: 9 lanes
Facts: lanes=9 new-rows=6

Lúc lập plan, planner đọc phải bản `AGENTS.md` cũ của IOT-CV, trong đó luật 5 cấm spawn sub-agent (chủ dự án đã bỏ lệnh
cấm ngày 18/11/2026; luật 5 bản mới cho spawn sub-agent Opus, cùng mức effort với phiên chính). Vì vậy nhóm A (rà prior-art:
XLA, IOT-CV `ml/`, `backend/vision`, `backend/flights`, `frontend/.../VideoPanel`, `plans/ai`, báo cáo 21/09) và nhóm B
(đối chiếu số liệu với `phan-tich-video.md`, `kiem-xla.md`, `nghien-cuu-tai-lieu.md`; soát từng "luôn/không bao giờ")
được làm **inline, đơn luồng**; kết quả vẫn ghi đủ ở Reuse Ledger. Không có reviewer nào được spawn (chủ dự án không yêu
cầu review). Khi thực thi, dùng sub-agent Opus theo lane bên dưới.

## Verification topology

| Lane | Verifier | Concurrent? |
|------|----------|-------------|
| data | `pytest tests/camera/test_frames_split.py tests/camera/test_camera_profile.py` | yes, CPU |
| degrade | `pytest tests/degrade/test_degrade.py tests/degrade/test_calibration_report.py` | yes cho 05 (CPU); 07 dùng GPU, xếp hàng GPU (D9) |
| realism | `pytest tests/eval/test_realism.py` | yes, test dùng đặc trưng giả lập CPU |
| labels | `pytest tests/labels/test_testset_labels.py tests/eval/test_detection_eval.py` | yes; gán nhãn trước dùng GPU, xếp hàng GPU (D9) |
| detect | `pytest tests/detect/test_finetune_runs.py tests/detect/test_detector_export.py` | no: huấn luyện GPU, xếp hàng GPU (D9) |
| classic | `pytest tests/enhance/test_classic.py` | yes, CPU |
| restore | `pytest tests/enhance/test_enhancer_export.py` | no: huấn luyện GPU, xếp hàng GPU (D9) |
| iotcv | `pytest tests/integration/test_iotcv_vision.py` | yes, chạy test IOT-CV bằng bundle giả, không cần GPU |
| iotcv-docs | `pytest tests/integration/test_iotcv_refs.py` | yes, chỉ đọc + sửa Markdown |

Lane sizing: tier=opus UNMEASURED; lanes=9; p=0.8; P(all land)=0.13 (information, not a cap)

p=0,8 là ước (ESTIMATED), chưa có số đo cho tier này. GPU là tài nguyên chung duy nhất: lane GPU (degrade ở phase 07, labels lúc gán trước, detect, restore) mở song song nhưng chạy job GPU nối tiếp qua khoá D9.

Seam check: `cochange-seams.cjs` không có lịch sử để đọc cho `xla/**` (package chưa tồn tại) và lane IOT-CV nằm ở repo khác; ranh giới lane do bảng sở hữu file trong từng phase giữ, các file dùng chung được giữ chỗ theo tên ở D7, D8.

## Contract

Owner: tác giả plan (phiên chính); lane cần đổi thì dừng và báo, Owner sửa `contract.md`/`docs/hop-dong-model.md`, tăng `contract_version`, rồi báo lại mọi lane
File: docs/hop-dong-model.md
Test: tests/contract/test_model_contract.py

Nội dung đầy đủ (đang ở trạng thái đã chốt): `contract.md` cạnh plan này. Phase 02 `git mv` nó thành `docs/hop-dong-model.md`
và để lại con trỏ; từ đó `docs/hop-dong-model.md` là SSOT, IOT-CV trỏ tới nó, không chép.

## Decision register

| # | Decision | Taken |
|---|---|---|
| D1 | Vào/ra `detector.onnx` | Vào `frame` uint8 [480,640,3] BGR 0–255 nguyên khung (kể cả dải mép); cắt dải mép, letterbox, chuẩn hoá, NMS, quy toạ độ về khung gốc đều nằm **trong graph**. Ra `boxes` float32 [N,4] xyxy px khung gốc 640×480 + `scores` float32 [N]. Lý do: một SSOT tiền/hậu xử lý, IOT-CV không phải chép lại phép letterbox. Chi tiết: `contract.md` §2 |
| D2 | Vào/ra `enhancer.onnx` | Vào giống D1; ra `frame_out` uint8 [480,640,3] BGR cùng cỡ; cột `[0, crop_left)` lấp bằng cách lặp cột `crop_left` để hộp của D1 vẫn khớp toạ độ. `contract.md` §3 |
| D3 | Bundle + nạp ở IOT-CV | Thư mục `<model>-v<semver>/` gồm `<model>.onnx`, `meta.json`, `golden/` (3 khung + kết quả mong đợi). IOT-CV đọc đường dẫn + sha256 từ `backend/config.py` (`DETECTOR_ONNX`, `DETECTOR_SHA256`, `ENHANCER_ONNX`, `ENHANCER_SHA256`); sai sha256 hoặc sai `contract_version` major → không nạp, log lỗi + sự kiện, backend bay vẫn chạy. `contract.md` §4–5 |
| D4 | Hồ sơ camera | `configs/camera_profile.yaml`, `schema_version` + `profile_version` semver, mọi trường có đơn vị và p5/p50/p95, đo **chỉ trên khung dev**. `contract.md` §6 |
| D5 | Đường nhận diện chạy trên ảnh gì | Khung **gốc** (không qua enhancer), trừ khi thí nghiệm E4 (phase 14) cho mAP50 cao hơn với CI không chồng nhau; khi đó mới đổi và tăng `contract_version` minor |
| D6 | Luồng ảnh rõ trên web | `GET /api/video/stream?variant=enhanced` (mặc định `variant=raw`, hành vi cũ giữ nguyên); nút bật/tắt chỉ đổi `src` của `<img>`; box vẫn đi qua message `detection` hiện có, không đổi `DetectionPayload` |
| D7 | CLI | Mỗi module có lệnh riêng `python -m xla.<gói>.<module> ... --json`, mã thoát 0 = đạt, 1 = lỗi dữ liệu/kiểm, 2 = sai cách gọi. Không có file điều phối CLI chung (tránh file dùng chung giữa lane). Lệnh tái lập toàn bộ là `scripts/reproduce.py` của phase 18 |
| D8 | Phụ thuộc Python XLA | Phase 01 khai đủ một lần trong `pyproject.toml` (danh sách ở phase 01). Lane cần gói mới thì dừng và báo, không tự sửa `pyproject.toml` |
| D9 | GPU dùng chung | Một job GPU tại một thời điểm, khoá bằng file `D:\DevCache\locks\gpu.lock` (script `xla/gpu_lock.py`, phase 01); job chờ khoá, không chạy chen |
| D10 | Chia dev/test | `data/real/split.json` do phase 03 tạo, sau đó chỉ đọc. Không phase nào hiệu chỉnh, chọn ngưỡng hay chọn checkpoint trên khung test |
| D11 | Định dạng dự đoán để chấm | CSV `frame,x1,y1,x2,y2,conf` (px khung gốc 640×480), một file một model; nhãn YOLO txt chuẩn hoá theo 640×480. `contract.md` §7 |
| D12 | Nhánh và PR (XLA) | Nhánh tích hợp `feat/ai-anh-chat-luong-thap` cắt từ `main`; mỗi lane một worktree + nhánh `lane/<id>`, đẩy nhánh và báo, không mở PR; phase serial gộp lane. 3 PR vào `main`: sau phase 02, sau phase 14, sau phase 18 |
| D13 | IOT-CV | Chỉ ghi file, không commit/push/PR khi chủ dự án chưa bảo; trước khi sửa file dùng chung (`backend/app.py`, `backend/config.py`, `backend/ws.py`, `pyproject.toml`, `AGENTS.md`, `plans/`) chạy `git status --short` và hỏi phiên đang sống qua ListAgents/SendMessage |

Straggler: lane CPU về trễ quá ~2× trung vị thời gian lane thì gộp các lane đã xong, giao lại phần còn lại cho lane trễ; không chờ. Lane GPU không tính là trễ khi đang chờ khoá GPU (D9).

## Reuse Ledger

Phạm vi rà (18/11/2026 giờ VN; cột `swept:` ghi 2026-11-17 vì validator so theo UTC): `D:\XLA` toàn repo (src, web, notebooks, data, plans/reports), `d:\Coding\IOT-CV\ml\**`,
`backend\vision\**`, `backend\flights\**`, `backend\schemas.py`, `backend\ws.py`, `backend\config.py`, `pyproject.toml`,
`frontend\src\components\VideoPanel\**`, `frontend\src\lib\detection.ts`, `plans\ai\*.md`,
`plans\reports\260921-research-ai-vision-pipeline.md`. Không có MCP knowledge-retrieval cho hai dự án này: mọi kết luận
"không có" dưới đây là kết quả grep trong phạm vi trên, không phải quét corpus.

| # | Capability | Verdict | Evidence |
|---|---|---|---|
| 1 | Khử hồng vùng sáng | EXTEND | `recover_pink_highlights` @ src/uav_pipeline.py:98; delta: bỏ cổng sáng `t_high=110`, chỉ kích hoạt khi max(R,B) gần bão hoà (≳ 240), viết lại vector hoá trong `xla/enhance/classic.py` (`kiem-xla.md` §6) |
| 2 | Ước lượng σ nhiễu, khử nhiễu theo σ | REUSE | `estimate_noise_sigma` @ src/uav_pipeline.py:84, `bilateral_denoise` @ src/uav_pipeline.py:180 |
| 3 | Trừ màn mờ / dark channel | EXTEND | `_dehaze_dark_channel` @ src/uav_pipeline.py:341; delta: chạy sau bù tối góc (flat-field), màn mờ theo percentile |
| 4 | Khung sinh dữ liệu có seed + manifest | EXTEND | `_rng_for` @ src/data_augmentation.py:162, `generate_augmented_benchmark` @ src/data_augmentation.py:174; delta: thay mô hình hồng cộng màu bằng bão hoà kênh tuyến tính rồi nhân hệ số WB |
| 5 | Nén JPEG đúng camera | EXTEND | `_jpeg` @ src/data_augmentation.py:31 (cv2 quality); delta: Pillow `qtables=` chép từ khung thật + `subsampling="4:2:2"`, không truyền `quality` (`plans/reports/261118-bang-chung/nghien-cuu/annexk_check.py`, `jpeg_dqt_scan.py`) |
| 6 | Đọc khung từ bản ghi AVI | REUSE | `Recording` @ d:/Coding/IOT-CV/backend/flights/video.py:74, `avi_chunk_index` @ backend/flights/video.py:49 (repo IOT-CV) |
| 7 | Cửa sổ chuyến bay | REUSE | `python -m backend.flights --events` @ d:/Coding/IOT-CV/backend/flights/__main__.py, `backend/flights/index.py` (repo IOT-CV) |
| 8 | Đặc trưng suy giảm (dark channel, % hồng, blocking, quality từ DQT, dải mép…) | EXTEND | `ijg_quality` và đặc trưng @ plans/reports/261118-bang-chung/video/features.py:22; delta: thành module `xla/camera/measure.py` có test, ghép khung↔độ cao theo `segment`+`video_frame` (`nghien-cuu-tai-lieu.md` §0) |
| 9 | VisDrone → một lớp person | EXTEND | khung sườn `convert_visdrone_person.py` @ d:/Coding/IOT-CV/ml/scripts/convert_visdrone_person.py:20 (thân `NotImplementedError`), mapping category 1, 2; delta: hiện thực trong `xla/degrade/sources.py` |
| 10 | Chia theo khối thời gian | EXTEND | nguyên tắc chia theo session `plans/ai/ai-phase-01-chuan-bi.md:371` (IOT-CV), `split_dataset.py` (khung sườn); delta: khối 30 s phân tầng theo độ cao + có người |
| 11 | Suy giảm ngẫu nhiên bậc hai D1 | NEW | thiết kế A1.5 `plans/ai/ai-phase-01-chuan-bi.md:408` (IOT-CV) chỉ có trên giấy, không có code; `basicsr` không dùng (import `torchvision.transforms.functional_tensor` đã bị bỏ ở torchvision mới)<br>reuse-search: not-found-after-corpus-and-grep<br>scope: src/, web/, notebooks/, plans/reports/, d:/Coding/IOT-CV/ml, d:/Coding/IOT-CV/backend \| swept: 2026-11-17 |
| 12 | Huấn luyện / đánh giá / xuất YOLO | REUSE | `repo: ultralytics` trong venv `D:\DevCache\venvs\vidstats` (đã chạy, `plans/reports/261118-bang-chung/video/yolo_run.py`) |
| 13 | Suy luận lát cắt SAHI | REUSE | `repo: obss/sahi` (chưa cài); thiết kế A2.3 `plans/ai/ai-phase-02-huan-luyen-thi-nghiem.md:213` (IOT-CV) |
| 14 | Gán nhãn local | REUSE | `repo: label-studio` qua `uv tool install`; quy trình + `label_config.xml` A1.7 `plans/ai/ai-phase-01-chuan-bi.md:677` (IOT-CV) |
| 15 | Self-training nhãn giả + tracker | REUSE | `repo: ultralytics` `track(persist=True)` botsort/bytetrack; quy trình SimROD rút gọn `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md` §4 |
| 16 | Xuất ONNX detector | EXTEND | A2.4 `plans/ai/ai-phase-02-huan-luyen-thi-nghiem.md:246` (IOT-CV) + `ultralytics export`; delta: bọc tiền/hậu xử lý vào graph (D1) |
| 17 | Đọc MJPEG, giữ khung mới nhất, fan-out | REUSE | `MjpegLatestFrameReader` @ d:/Coding/IOT-CV/backend/vision/stream.py:199, `mjpeg_multipart` @ backend/vision/stream.py:304 |
| 18 | Gửi box lên web, vẽ box | REUSE | `DetectionPayload` @ d:/Coding/IOT-CV/backend/schemas.py:239, `broadcast("detection")` @ backend/ws.py:603, `DetectionOverlay` @ frontend/src/components/VideoPanel/DetectionOverlay.tsx, `scaleBox` @ frontend/src/lib/detection.ts |
| 19 | Bộ phát hiện người trong backend | EXTEND | khung sườn `PersonDetector` @ d:/Coding/IOT-CV/backend/vision/detector.py:29 (`NotImplementedError`); delta: nạp bundle ONNX theo D3, bỏ ultralytics khỏi backend |
| 20 | Chống spam sự kiện phát hiện | EXTEND | `EventThrottle.should_emit` @ d:/Coding/IOT-CV/backend/vision/events.py:51 (khung sườn) |
| 21 | Camera giả đúng nén thật | EXTEND | `cv2_quality()` @ d:/Coding/IOT-CV/backend/vision/fake_stream.py (q=12 → OpenCV Q81, nhẹ tay hơn thật); delta: Pillow `qtables` từ `camera_profile.yaml` + 4:2:2 |
| 22 | Web demo của nhóm | REUSE | `web/server.py:446` giữ nguyên chức năng; chỉ sửa bind `0.0.0.0` → `127.0.0.1` |
| 23 | Mạng khôi phục riêng camera | NEW | ứng viên NAFNet/FBCNN/SCUNet là mã ngoài; repo chỉ có ESPCN/FSRCNN/Real-ESRGAN-anime @ models/ (nâng cỡ, bị bỏ theo thiết kế §3)<br>reuse-search: not-found-after-corpus-and-grep<br>scope: src/, models/, web/, d:/Coding/IOT-CV/ml, d:/Coding/IOT-CV/backend/vision \| swept: 2026-11-17 |
| 24 | Độ đo giống thật (KID-DINOv2, C2ST, thống kê mức thấp) | NEW | thư viện `dgm-eval` (ngoài); repo không có KID/FID/C2ST nào<br>reuse-search: not-found-after-corpus-and-grep<br>scope: src/, notebooks/, plans/reports/, d:/Coding/IOT-CV/ml \| swept: 2026-11-17 |
| 25 | Trang thử mù A/B local | NEW | `web/index.html` là trang tải ảnh lên, không có chấm mù<br>reuse-search: not-found-after-corpus-and-grep<br>scope: web/, src/, d:/Coding/IOT-CV/frontend/src \| swept: 2026-11-17 |
| 26 | Hồ sơ camera có phiên bản | NEW | số đo rời rạc trong `video/phan-tich-video.md`, `nghien-cuu-tai-lieu.md` §0; `ml/configs/own_camera.yaml` (IOT-CV) chỉ là cấu hình dataset<br>reuse-search: not-found-after-corpus-and-grep<br>scope: plans/reports/, d:/Coding/IOT-CV/ml/configs, d:/Coding/IOT-CV/firmware/camera \| swept: 2026-11-17 |
| 27 | Hợp đồng bundle model (meta.json, sha256, golden) | NEW | `YOLO_WEIGHTS` @ d:/Coding/IOT-CV/backend/config.py:172 chỉ là một đường dẫn, không kiểm gì<br>reuse-search: not-found-after-corpus-and-grep<br>scope: d:/Coding/IOT-CV/backend, d:/Coding/IOT-CV/docs, src/ \| swept: 2026-11-17 |
| 28 | Đánh giá nhận diện (mAP, AR-small, FP/khung âm, bootstrap theo khối) | EXTEND | khung sườn `evaluate.py` @ d:/Coding/IOT-CV/ml/scripts/evaluate.py (giao diện in AR-small + độ trễ theo A1.6 `plans/ai/ai-phase-01-chuan-bi.md:562`); delta: hiện thực trong `xla/eval/detection.py` trên CSV D11 |
| 29 | Cấu hình dataset + script cũ của IOT-CV | REPLACE | `ml/configs/*.yaml`, `ml/scripts/*.py` (IOT-CV, toàn khung sườn) thay bằng module XLA; phase 17 chỉ thêm con trỏ, không xoá |

Báo cáo tham chiếu (không chép lại nội dung): `plans/reports/261118-bang-chung/nghien-cuu/nghien-cuu-tai-lieu.md`
(dataset §1, suy giảm §2, khôi phục-trước-nhận-diện §3, self-training §4, độ đo §5),
`d:\Coding\IOT-CV\plans\reports\260921-research-ai-vision-pipeline.md` (§1.3 cấu hình 8 GB, §2.1–2.3 VisDrone, §5.2 ONNX,
§5.3 SAHI, §5.4 gán nhãn; phần nói OV2640 và đề tài A/B/C/D đã lỗi thời).

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tập test ít người, khung tương quan (1 bãi, 1 buổi) | high | high | Phase 03 đếm thật trước khi chia; khối thời gian 30 s, bootstrap theo khối, báo CI, không nói quá (thiết kế §7) |
| Nhãn máy gán thiên về model gán | medium | high | Model gán khác họ YOLO (RT-DETR + SAHI), agent soát, chủ dự án duyệt; báo cáo khai rõ (phase 08) |
| Rò rỉ test vào hiệu chỉnh / chọn checkpoint | medium | high | D10: `split.json` chỉ đọc; test `test_frames_split.py` cấm khối test trong mọi manifest huấn luyện/hiệu chỉnh |
| ORT CUDA EP không chạy trên Python 3.13 Windows | medium | high | Phase 15 bước 1 thử với bundle giả; dự phòng `onnxruntime-directml` rồi CPU kèm cảnh báo; hợp đồng không đổi |
| Bảng DQT ở q ≠ 12 sai giả thuyết | medium | medium | Cờ "chưa xác minh" trong `camera_profile.yaml`; human step kiểm khi bật camera |
| GPU 8 GB không đủ cho YOLO @1280 batch hợp lý | medium | medium | Smoke đo VRAM + giờ; hạ imgsz 960 hoặc batch, ghi Deferred |
| Mạng khôi phục bịa chi tiết / nhấp nháy | medium | medium | Loss L1 + SSIM, không GAN; đo nhấp nháy; đường nhận diện chạy khung gốc (D5) |
| Nhiều phiên cùng sửa file IOT-CV | medium | high | D13: `git status` + nhắn phiên đang sống trước khi sửa file dùng chung |
| NOMAD/POG tải chậm hoặc không tải được | medium | low | VisDrone-person đủ làm nguồn chính; NOMAD là bổ sung. → 18/11/2026: NOMAD loại (nhìn xiên), POG máy chủ không trả lời; nguồn duy nhất là VisDrone lọc nadir |

## Self-improvement loop

- **Signal:** `tests/contract/test_model_contract.py` (XLA) và `backend/tests/test_vision_ai.py` (IOT-CV) đỏ khi bundle lệch hợp đồng; lúc chạy, IOT-CV phát sự kiện `vision.model_rejected` kèm lý do (sha256, `contract_version`, cỡ khung) thay vì im lặng
- **Loop:** workflow pytest GitHub Actions có sẵn của IOT-CV chạy test đó trên mọi PR, và lệ "Bẫy đã gặp" trong `AGENTS.md` + memory của agent mà mọi phiên IOT-CV đã đọc
- **Fix:** sửa `docs/hop-dong-model.md` hoặc tăng `profile_version` trong `configs/camera_profile.yaml`, xuất lại bundle, ghi bẫy mới vào `AGENTS.md` mục "Bẫy đã gặp"
- **Guard:** bộ ba khung `golden/` trong mỗi bundle được kiểm ở cả hai repo; test hợp đồng đỏ là bundle không được nạp

## Deferred decisions

| # | Question | Options | Choice taken | Why | How to reverse |
|---|----------|---------|--------------|-----|----------------|
| 1 | Bảng DQT cho auto-quality q ≠ 12 (19–40) | Chỉ q12 / Annex K × q/16 / chờ đo | q12 đo thật (Annex K × 0,75, 4:2:2) làm chính; q 19–40 theo giả thuyết Annex K × q/16, cờ `verified: false` | Báo cáo nghiên cứu §0: chưa có khung mẫu q ≠ 12 | Đo khi camera bật lại (human step), sửa `configs/camera_profile.yaml`, tăng `profile_version` |
| 2 | Dataset nguồn | VisDrone / +NOMAD / +POG / +Archangel | VisDrone-person lọc nadir (chính) + NOMAD (CC BY 4.0); POG chỉ dùng nội bộ nếu tải được, không đẩy ảnh POG lên repo public; bỏ AU-AIR, DetReIDX, ForestPersons, SynDrone. → đổi 18/11/2026: chỉ VisDrone-person lọc nadir; NOMAD loại vì ảnh nhìn xiên ~45–60°, POG máy chủ không trả lời (`configs/sources.yaml`, `plans/reports/261118-phase-05-bo-lam-xau.md`) | Báo cáo nghiên cứu §1 | Sửa `configs/sources.yaml` (phase 05) |
| 3 | Lọc ảnh nadir | Heuristic / CLIP zero-shot / metadata | Metadata khi có (NOMAD theo thư mục), còn lại heuristic không-có-trời + điểm CLIP zero-shot, ngưỡng chọn trên 100 ảnh soát tay. → đổi 18/11/2026: heuristic không-có-trời + CLIP zero-shot chỉ đạt precision ~0,5 trên ảnh soát tay; thay bằng nhãn hướng nhìn theo chuỗi (`configs/visdrone_view.yaml`) + probe logistic trên đặc trưng CLIP cho chuỗi lẫn, ngưỡng 0,8 | Báo cáo §1 "Về GSD" | Sửa ngưỡng trong `configs/sources.yaml` |
| 4 | Độ đo giống thật | FID / KID-Inception / KID-DINOv2 + C2ST | KID trên DINOv2 (`dgm-eval` → đổi 18/11/2026: `dgm-eval` ghim pillow 9.2 / numpy 1.23, không cài được; dùng DINOv2 ViT-L/14 qua `torch.hub` + KID tự viết, tiền xử lý như `dgm-eval`, xem `results/realism/README.md`), 1 000–3 000 patch, trung bình nhiều lần + C2ST patch 64×64 phần dư cao tần + thống kê mức thấp (DQT, NLF, PSD, MTF, blocking, dark channel, % hồng); không dùng FID làm chính; cùng quy trình đọc/resize cho hai tập | Báo cáo §5 | Sửa `xla/eval/realism.py` (phase 06) |
| 5 | NR-IQA | Tối ưu theo / chỉ mô tả | MUSIQ + NIQE qua `pyiqa` chỉ mô tả, không chọn checkpoint/bộ khôi phục; ghi giấy phép PolyForm Noncommercial | Báo cáo §5 | Phase 14 |
| 6 | Self-training | Mean-Teacher / nhãn giả + tracker / SimROD rút gọn | Nhãn giả ngưỡng cao làm sạch bằng tracker Ultralytics (`persist=True`) + SimROD rút gọn (teacher m/l @1280 + TTA → student s, mosaic trộn nguồn/đích), 2–3 vòng, chọn checkpoint bằng val thật có nhãn; không Mean-Teacher nặng | Báo cáo §4: vừa 8 GB | Phase 11 |
| 7 | Học suy giảm không cặp | CUT / CycleGAN / Real-SR | Real-SR (tiêm thống kê nhiễu từ ảnh đích) là ablation tuỳ chọn E3 trong phase 07 nếu còn giờ; không CycleGAN. → 18/11/2026: đã làm, tập D2+RealSR thắng KID (3,93, CI không chồng D1) | Báo cáo §2: CycleGAN −2,25/−0,83 AP | Phase 07 bước tuỳ chọn |
| 8 | Khôi phục trước nhận diện | Luôn / không / có cổng | Đường nhận diện chạy khung gốc (D5); E4 đo: detector có sẵn + enhancer, detector fine-tune + enhancer, biến thể có cổng kiểu TaskGuard là tuỳ chọn | Báo cáo §3 | Đổi D5 khi E4 thắng có ý nghĩa |
| 9 | Model gán nhãn trước | YOLO lớn / RT-DETR / zero-shot | RT-DETR-x COCO + SAHI làm chính, YOLO11x @1280 + SAHI phiếu thứ hai, hợp bằng WBF; hộp chỉ một model thấy được đánh dấu cho agent soát kỹ | Thiết kế §7: dùng model khác họ với model được chấm | Phase 08 |
| 10 | Kiến trúc detector | YOLO11s / YOLO26s / n | YOLO11s @1280 làm chính (đã đo mốc); YOLO26s chỉ thêm nếu smoke cho thấy có sẵn trong Ultralytics đã cài và không chậm hơn | Đã có số mốc cho 11s | Phase 10 |
| 11 | Loss mạng khôi phục | L1 / +SSIM / +GAN | L1 + (1 − SSIM), không GAN; đích = ảnh nguồn sau cùng phép hạ mẫu, không suy giảm | Tránh bịa chi tiết (thiết kế §7) | Phase 13 |
| 12 | Runtime ONNX ở IOT-CV | onnxruntime-gpu / directml / TensorRT | `onnxruntime-gpu` CUDA EP trong nhóm optional `vision`; lỗi thì `onnxruntime-directml`; CPU chỉ kèm cảnh báo to; không TensorRT (gắn driver) | Chưa đo trên Py 3.13 | Phase 15 bước 1 |
| 13 | Mã + notebook cũ của nhóm | Chuyển vào archive / giữ chỗ | `src/`, `notebooks/`, `web/` giữ nguyên chỗ (notebook còn chạy), README ghi rõ là bản v2 cũ; chỉ chuyển dữ liệu sai (`data/real_uav`, `data/synthetic_*`, `data/raw_videos`, `data/augmented`) và bảng kết luận sai (`results/tables`, `results/figures`) vào `archive/`, kèm ghi chú sự thật | Giữ lịch sử, không làm hỏng notebook | Phase 01 `git mv` ngược |
| 14 | Khung bay thật trong repo `Localhost218/XLA` | Đẩy ngay / chờ đồng ý | Đẩy ngay: commit khung bay + mẫu âm (`data/real/frames/`), nhãn, manifest, `split.json`, `golden/` của bundle, ảnh thử A/B (`results/ab/img/`). Chủ dự án đã đồng ý ngày 18/11/2026 ("cứ làm như 1 repo private ... tôi cho phép bạn đủ mọi quyền"), kể cả khung có người và toạ độ GPS | Repo tái lập được mà không cần bản ghi IOT-CV; không còn bước người hỏi công bố | Đã đẩy thì không thu hồi hết được; chỉ xoá khỏi lịch sử (`git filter-repo` + force-push) khi chủ dự án bảo |
| 15 | Chia dev/test | Theo chuyến / khối thời gian | Khối 30 s, phân tầng theo (bin độ cao, có người theo YOLO đếm sơ bộ), seed 20261118, bỏ 5 s đệm giữa khối dev và khối test kề nhau; test lấy khung cách nhau ≥ 1 s, mục tiêu 300–400 → **đổi 18/11/2026: 200–400** vì trần khả thi 362 khung / 74 có người (phase 03 "Quyết định chia lại") | Thiết kế §5.6 | Phase 03, tạo lại `split.json` (chỉ trước phase 08) |
| 16 | Cấu trúc thử A/B | 50 cặp một loại / chia đôi | 25 cặp mạng vs gốc + 25 cặp mạng vs cổ điển + 5 cặp lặp lại để đo độ nhất quán; trái/phải ngẫu nhiên; báo CI Wilson | Thiết kế §8 cần cả hai tỉ lệ | Phase 14 |
| 17 | `ml/` cũ của IOT-CV | Xoá / trỏ | Không xoá; thêm con trỏ tới XLA; xoá là human step tuỳ chọn | Xoá không tự quyết | Phase 17 |
| 18 | CI XLA | Không / GitHub Actions CPU | GitHub Actions chạy pytest CPU (`timeout-minutes: 10`, không `paths:`), không test GPU | Có vòng chặn hồi quy rẻ | Xoá `.github/workflows/tests.yml` |
| 19 | Ước lượng giờ GPU | — | Mọi số giờ GPU trong plan ghi "chưa xác minh" tới khi smoke đo; tổng hiện ước 34–84 giờ | Ràng buộc đã chốt | Cập nhật Forecast "Re-score" sau mỗi smoke |
| 20 | Bước Plane Work-Item Gate của skill | Chạy / bỏ | Bỏ hẳn | Luật chủ máy + AGENTS.md luật 4 | Không đảo ngược |
| 21 | Chia sẻ bundle model đã xuất cho nhóm | Commit `.onnx` vào git / Git LFS / GitHub Release | `meta.json` + `golden/` commit vào git; file `.onnx` không vào lịch sử git (`.gitignore` phase 01) mà đính vào GitHub Release `models-v<ver>` của XLA ở phase 18 bước 9; `meta.onnx_sha256` trong git dùng để kiểm bản tải về. Giữa các worktree lane, `.onnx` được chép tay ở phase 14 bước 1 | `.onnx` cỡ chục MB, mỗi lần xuất lại mà commit vào git thì lịch sử phình mãi; LFS cần cài thêm và có hạn mức; Release giữ đúng từng bản, tải lại được | Bỏ dòng `models/release/**/*.onnx` trong `.gitignore` rồi commit file |
