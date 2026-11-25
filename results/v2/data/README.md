# Dữ liệu ngoài + tập trộn `d02x` (AI v2 phase 02, lane data-ext)

Ngày 25/11/2026. Nhánh `lane/data-ext` (từ `feat/ai-v2` @ c3d68f9). Không ảnh ngoài nào trong git: repo chỉ giữ mã đọc
(`xla/degrade/sources.py`), cấu hình (`configs/sources.yaml` khối `downloads` + `external`, `configs/view_external.yaml`) và
file này. Ảnh giải nén ở `D:\DevCache\datasets\xla\external\<bộ>\`, tập sinh ở `D:\DevCache\datasets\xla\d02x\`.

Dựng lại (Bash, môi trường `plans/261118-ai-anh-chat-luong-thap/thuc-thi-phase-10-14.md` §2):

```bash
$PY -m xla.degrade.sources extract                 # kiểm sha256 file nén, giải nén, ghi manifest.jsonl + extract.json
$PY -m xla.degrade.sources extract --verify        # băm lại từng file trên đĩa, so với manifest
$PY -m xla.degrade.sources probe-external          # probe góc nhìn (GPU trong gpu_lock) → external/probe_scores.jsonl
$PY -m xla.degrade.sources seq-sheet --source heridal --n 100 --seed 20261125 --out D:/DevCache/cache/degrade/external_review
$PY -m xla.degrade.sources stats --json            # khối "external": số trước/sau lọc, cỡ hộp, precision probe
$PY -m xla.degrade.build --set d02x --workers 4 --json
```

## 1. Nguồn, giấy phép, file nén

Giấy phép ghi trước khi dùng (`configs/sources.yaml` khối `downloads`, `use: research`, `retrieved: 2026-11-25`). SARD
**bỏ**: IEEE DataPort đòi gói thuê bao trả phí kể cả khi đã đăng nhập (`downloads.sard.status: unavailable`, Deferred 31).

| Bộ | Giấy phép | Nguồn | File nén (sha256 đầu) | Giải nén (file, byte) | sha256 `manifest.jsonl` |
|---|---|---|---|---|---|
| HERIDAL | CC BY 3.0 (Božić-Štulić et al., IJCV 2019) | Zenodo 5662351, bản VOC keras-retinanet | `heridal_keras_retinanet_voc.zip` 8 311 735 812 B, `63ea51b2…` | 3 338 (1 684 JPG + 1 650 XML + 4 ImageSets), 8,38 GB | `f513bf0c…` |
| TinyPerson | dataset không ghi giấy phép; mã MIT (`ucas-vg/PointTinyBenchmark`) | Google Drive trong README TinyBenchmark | `train.tar.gz` `8ae1ba95…`, `test.tar.gz` `73af8f53…`, `tiny_set_train.json` `4c2683bf…`, `tiny_set_test.json` `0cfcbbcc…` | 1 534 (1 532 ảnh có trong json + 2 json), 0,46 GB | `ea6225c3…` |
| WiSARD Multi-Modal Sample | MIT (Broyles et al., IROS 2022) | https://sites.google.com/uw.edu/wisard | `WiSARD_Multi_Modal_Sample.zip` `fd9bd786…` | 529 (thư mục `*_VIS_0003`: 264 ảnh + 265 txt; `*_IR_*` không giải nén), 0,99 GB | `291cfcba…` |

sha256 đủ 64 ký tự của file nén ở `configs/sources.yaml`; của manifest ở `<external>/<bộ>/extract.json`. Mỗi dòng
`manifest.jsonl` = `{rel, sha256, bytes, archive}` của một file đã giải nén. `extract` dừng nếu sha256 file nén khác cấu
hình; tên thành viên tuyệt đối / có `..` / ổ đĩa, hay symlink trong tar → lỗi (file nén là dữ liệu ngoài).

## 2. Trình đọc và cách chia

| Bộ | Nhãn → người | Vùng bỏ qua | Nhóm (chia train/val theo nhóm) | Ghi chú |
|---|---|---|---|---|
| HERIDAL | VOC `<object><name>person` (không phân biệt hoa thường; lớp lạ → lỗi) | hộp bị ô cắt cụt < 50 % | **nơi bay** trong tên (`train_BRA_0001` → `BRA`), 17 nơi | ảnh không XML = nền không người (lớp âm) |
| TinyPerson | `sea_person`, `earth_person` | hộp có cờ `ignore` / `uncertain` / `logo` | video (`bb_V0032`); ảnh rời (bing…) một nhóm | chỉ ảnh có trong json (labeled + pure_bg) |
| WiSARD | YOLO chuẩn hoá, lớp nào cũng là người (bộ không có file `classes`; `count.txt` ghi 1 022 người / 264 ảnh, khớp) | — | thư mục = một chuyến bay | chỉ `*_VIS_*` |

Chia: HERIDAL val = nơi `BLI, CAB, GRO, MOB` (chỉ có trong test gốc) + `RAK, SB, TRS` = 217/1 684 ảnh (12,9 %).
`ImageSets/Main/val.txt` của bộ **không dùng**: đó là ảnh ngẫu nhiên lấy từ train cùng nơi bay (rò cảnh). TinyPerson:
json `train` → train, `test` → val (đã kiểm: 0 video chung giữa hai json). WiSARD: một chuyến, toàn bộ train (Risk:
không chia val từ một cảnh). `sources.check_group_split` chặn mọi nhóm nằm ở cả hai split (chạy cả lúc dựng `d02x`).

## 3. Lọc góc nhìn

Thứ tự quyết (`sources.is_nadir_external`): nhãn soát tay từng ảnh → nhãn nhóm (`configs/view_external.yaml`) → probe CLIP
≥ 0,8, `accept_views: [nadir, near]` như VisDrone. Probe = đúng probe VisDrone (`fit-probe`: logistic C=1 cân lớp trên đặc
trưng CLIP ViT-B-32 laion2b, học từ nhãn chuỗi + nhãn ảnh VisDrone), không thêm nhãn ngoài nào vào huấn luyện.

**Soát tay 100 ảnh mỗi bộ** (seed 20261125, lưới 4 tờ × 25 ô 360×270, `seq-sheet --source <bộ>`; agent lane data-ext xem
từng tờ, lúc soát lưới chưa có điểm probe). Precision/recall của **probe** trên mẫu đó (dương = nadir/near theo mắt):

| Bộ | Nhãn tay (nadir / near / oblique) | Probe chọn | Precision probe | Recall probe |
|---|---|---|---|---|
| HERIDAL | 93 / 7 / 0 | 77 | **1,00** (77/77) | 0,77 |
| TinyPerson | 11 / 19 / 70 | 11 | **1,00** (11/11) | 0,37 |
| WiSARD | 38 / 24 / 38 | 25 | **1,00** (25/25) | 0,40 |

Probe chặt: không nhận nhầm ảnh xiên nào trong 300 ảnh mẫu, nhưng bỏ sót 23–63 % ảnh tốt. Cỡ mẫu 100 nên precision 1,00
có cận dưới Wilson 95 % ≈ 0,95 (HERIDAL, 77 ảnh chọn), ≈ 0,74 (TinyPerson, 11 ảnh), ≈ 0,87 (WiSARD, 25 ảnh).

Quyết định sau khi soát (tự chốt, ghi ở `configs/view_external.yaml`):
- **HERIDAL gán nhãn nhóm `nadir` cho 16/17 nơi bay**: 100/100 ảnh mẫu nadir/near trên đủ 16 nơi đó, còn probe (học
  trên VisDrone đô thị) bỏ 23 % ảnh tốt. Bộ chụp có chủ đích thẳng xuống (SAR 40–65 m). Nơi `CAB` (8 ảnh, không có
  trong mẫu) vẫn qua probe (8/8 qua). Rủi ro: 4 nơi chỉ có 1 ảnh mẫu (BLA, BLI, MOB, SB).
- **WiSARD** nhóm `mixed` (mẫu có 38 % xiên) → từng ảnh qua probe; 100 ảnh mẫu dùng nhãn tay.
- **TinyPerson** không gán nhãn nhóm (video bãi biển, phần lớn thấy chân trời).

| Bộ | Ảnh trước lọc (có người / người) | Probe ≥ 0,8 (chưa tính nhãn tay) | Sau lọc: ảnh (có người / người) | Nhóm sau lọc | split train / val sau lọc |
|---|---|---|---|---|---|
| HERIDAL | 1 684 (1 086 / 3 229) | 1 256 | **1 684** (1 086 / 3 229) | 17 | 1 467 / 217 |
| TinyPerson | 1 532 (1 454 / 32 120; 11 676 vùng bỏ qua) | 154 | **173** (171 / 1 682) | 42 | 65 / 108 |
| WiSARD | 264 (261 / 1 022) | 73 | **110** (110 / 424) | 1 | 110 / 0 |

Kế hoạch ghi HERIDAL 3 235 hộp: 6 hộp nằm trong 3 file XML không có ảnh tương ứng (1 650 XML, 1 647 khớp ảnh) nên trình
đọc đếm 3 229.

**TinyPerson < 200 ảnh sau lọc → chỉ làm lớp dương cho verifier** (Risk phase 02): `use: [verifier]`, `d02x: false`.
Phase 07 lấy bằng `sources.select_external(cfg, for_d02x=False)` lọc `source == "tinyperson"` và dựng cửa sổ bằng
`build.render_unit` (cùng D2 v3). Owner chép vào Deferred của plan.

## 4. Cắt ô HERIDAL và cỡ hộp

Ảnh 4000×3000 (4 ảnh `test_MED` 4000×2250) cắt ô 1280×960 (4:3 như khung) chồng mép 160 px **trước** khi đặt cửa sổ và
làm xấu (`sources.tile_item`; `build` đọc ảnh gốc rồi cắt ô theo `crop`). Hộp còn ≥ 50 % diện tích trong ô → người; còn
ít hơn → vùng bỏ qua. Mỗi hộp có đúng một ô "chủ" (tâm ô gần tâm hộp nhất); chỉ giữ ô làm chủ ít nhất một hộp, cộng 1 ô
không chạm người nào mỗi ảnh (nền đá/bụi cây khó cho detector). Mỗi ô một cửa sổ.

Kết quả (1 684 ảnh): **3 913 ô** = 2 230 ô có người + 1 683 ô nền; 3 882 hộp trong ô (3 229 người khác nhau, người ở
dải chồng mép xuất hiện ở hai ô); 155 hộp bị cắt cụt thành vùng bỏ qua; train/val 3 296 / 617 ô.

Cạnh ngắn hộp người, px ảnh nguồn (HERIDAL: px trong ô, cũng là px ảnh gốc vì ô không co):

| Bộ | n | p5 | p25 | p50 | p75 | p95 | < 8 px | < 16 px |
|---|---|---|---|---|---|---|---|---|
| HERIDAL ảnh gốc | 3 229 | 32 | 42 | 52 | 63 | 82 | 0 | 0 |
| HERIDAL sau cắt ô | 3 882 | 31 | 42 | 52 | 63 | 82 | 0 | 0 |
| TinyPerson trước lọc | 32 120 | 3,8 | 6,5 | 10,4 | 17,9 | 43,7 | 11 450 | 22 668 |
| TinyPerson sau lọc | 1 682 | 5,6 | 10,0 | 15,1 | 25,2 | 57,3 | 262 | 900 |
| WiSARD sau lọc | 424 | 42 | 54 | 63 | 75 | 105 | 0 | 0 |
| đích camera (`camera_profile.yaml` `person_box_px.short_side`) | — | 26,7 | — | 53,5 | — | 64,7 | — | — |

HERIDAL đã đúng cỡ người của camera này (p50 52 so với 53,5) nên hình học giữ tỉ lệ 1 (không thu nhỏ, không phóng to):
nỗi lo "người co về < 2 px khi thu ảnh 4000×3000 về 640×480" (Risk) không xảy ra vì cửa sổ cắt từ ô, không thu cả ảnh.

## 5. Tập trộn `d02x`

`python -m xla.degrade.build --set d02x --workers 4 --json` (seed 20261118, hồ sơ `configs/camera_profile.yaml`
`ov3660-esp32s3-q12-20261102` v1.2.0, D2 `configs/degrade_d2.yaml` `params_version: 3`), ra
`D:\DevCache\datasets\xla\d02x\{train,val}\{images,labels}` + `manifest.jsonl` + `dataset.yaml` (`names: {0: person}`).
Nguồn = VisDrone (cùng lọc + cùng chia chuỗi với d0/d2 v1) + HERIDAL (ô) + WiSARD. Mỗi đơn vị (ảnh/ô, cửa sổ) ra **hai**
ảnh cùng cửa sổ: `d0__<đơn vị>.jpg` (sạch, q95 4:4:4) và `d2__<đơn vị>.jpg` (D2 v3); bản ghi manifest có `variant`,
`group`, `crop` (ô HERIDAL). Cùng hình học 624×480 + dải mép trái 16 cột như v1.

| Nguồn | Ảnh/ô nguồn | Đơn vị ghi được (bỏ: không đặt được cửa sổ) | Ảnh ra (d0 + d2) | train / val | Hộp người (ảnh ra) | Cửa sổ không người |
|---|---|---|---|---|---|---|
| VisDrone | 1 841 ảnh × 2 cửa sổ | 3 493 | 6 986 | 6 012 / 974 | 45 016 | 1 380 |
| HERIDAL | 3 913 ô × 1 cửa sổ | 3 895 | 7 790 | 6 556 / 1 234 | 5 712 | 3 830 |
| WiSARD | 110 ảnh × 2 cửa sổ | 220 | 440 | 440 / 0 | 958 | 90 |
| **Tổng** | 5 864 | 7 608 (414 đơn vị bỏ) | **15 216** | 13 008 / 2 208 | 51 686 | 5 300 |

1,28 GB, trung bình 84 KB/ảnh. Nguồn ngoài chiếm 8 230/15 216 ảnh (54 %); nền HERIDAL (đá, bụi cây, đường mòn) chiếm
3 830 cửa sổ không người.

Cạnh ngắn hộp người trong khung 640×480 (biến thể d0; d2 cùng hộp):

| Nguồn | n | p5 | p25 | p50 | p75 | p95 | < 8 px | < 16 px |
|---|---|---|---|---|---|---|---|---|
| VisDrone | 22 508 | 5,0 | 9,0 | 14,0 | 20,0 | 34,0 | 3 361 | 13 332 |
| HERIDAL | 2 856 | 25,7 | 34,3 | 42,1 | 52,0 | 62,4 | 0 | 7 |
| WiSARD | 479 | 23,4 | 36,5 | 46,5 | 54,8 | 66,0 | 0 | 3 |

HERIDAL/WiSARD gần phân bố người của camera thật (p5 26,7 / p50 53,5 / p95 64,7) hơn VisDrone nhiều: hình học thu nhỏ
về cỡ đích nhưng không phóng to, nên VisDrone vẫn nhỏ như v1.

Kiểm (đều chạy được lại):
- Phần VisDrone của d02x **trùng từng byte** với d0/d2 v1: 3 493/3 493 ảnh d0 và 3 493/3 493 ảnh d2 cùng sha256 theo
  đơn vị (RNG theo đơn vị gốc, không theo tên tập) → d0/d1/d2 v1 và E1 không bị ảnh hưởng.
- `pytest tests/degrade/test_external_sources.py` (19 test, gồm 2 test `data` trên ổ D): mỗi đơn vị có đủ d0 + d2, không
  nhóm nào ở cả train lẫn val, không khung bay thật, số ảnh trên đĩa = số dòng manifest; manifest giải nén khớp
  `sources.yaml` và không file ảnh nào trong git trùng sha256 với ảnh ngoài.
- `extract --verify`: băm lại 5 401 file giải nén, 0 lệch.

## 6. Thời gian máy

| Bước | Thời gian | Ghi chú |
|---|---|---|
| `extract` (3 bộ) | ≤ 4 phút | băm 10,3 GB file nén + ghi 9,8 GB |
| giải mã ảnh cho CLIP (CPU, 4 luồng, ngoài khoá GPU) | ~30 phút CPU (~13 phút thực) | JPEG 12 MP HERIDAL ~1,7 s/ảnh lúc máy đang tải nặng (25/11 14:20) |
| CLIP ViT-B-32 3 480 ảnh (GPU, `gpu_lock` "degrade:clip-emb", timeout 6 giờ) | ≤ 3 phút giữ khoá (nhả 15:02:29) | sau đó chờ khoá lane realdata (prelabel) ~15 phút |
| `build --set d02x --workers 4` | 551 s thực, 0,036 s/ảnh | ảnh lớn giải mã một lần cho mọi ô của nó trong cùng tiến trình |

## 7. Lệch so với plan

1. **Manifest từng file không vào git.** Phase ghi "chỉ manifest (đường dẫn tương đối + sha256) và mã đọc" lên repo;
   danh sách file lane được sở hữu không có chỗ cho manifest trong repo, nên `manifest.jsonl` (5 401 dòng) nằm cạnh dữ
   liệu ở `D:\DevCache\datasets\xla\external\<bộ>\`, còn repo giữ sha256 file nén (`configs/sources.yaml`) và sha256 của
   từng manifest (bảng §1). Giải nén là tất định nên manifest dựng lại được từ file nén. Owner muốn manifest trong git
   thì chép ba file đó vào một đường dẫn lane được cấp.
2. **TinyPerson không vào `d02x`** (173 ảnh sau lọc < 200): chỉ làm lớp dương verifier (§3). Owner chép vào Deferred.
3. **HERIDAL lọc bằng nhãn nhóm theo nơi bay** chứ không chỉ probe (§3): probe giữ precision 1,00 nhưng recall 0,77.
4. **Chia HERIDAL theo nơi bay**, không theo `ImageSets` của bộ (val gốc rò cảnh). 4 ảnh `test_MED` có `<size>` sai
   trong XML (4000×3000, ảnh thật 4000×2250, hộp đều nằm trong ảnh): dùng cỡ ảnh thật.
5. **`clip_embeddings` giải mã ảnh trước khi giữ khoá GPU** (4 luồng) và đặt `timeout_s` 6 giờ (D6); trước đây giải mã
   trong khối khoá, không timeout. Kết quả VisDrone cũ không đổi (đặc trưng VisDrone đọc từ cache).
6. `xla/degrade/sources.py` giờ ~1 300 dòng (quy ước ≤ 200 dòng/file): phần nguồn ngoài nằm chung file vì lane chỉ được
   sở hữu file này. Nên tách `xla/degrade/external.py` ở một phase sau.
7. Không thêm gói Python nào (D8): giải nén bằng `zipfile`/`tarfile`, XML bằng `xml.etree` (expat 2.9.0 của venv chặn bom
   mở rộng thực thể), thumbnail bằng PIL/cv2 có sẵn.
