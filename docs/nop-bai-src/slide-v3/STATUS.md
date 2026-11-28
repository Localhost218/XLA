# Slide v3: trạng thái khi dừng (19/11/2026)

Deck dở dang: `XLA_UAV_Nhom10.v3-wip.pptx` (bằng `deck/v8b.pptx`, 28 slide, dựng lại lúc dừng, CHƯA render để soát).
Bản đã upload (v2) vẫn nằm nguyên ở `docs/nop-bai/XLA_UAV_Nhom10.pptx/.pdf`, không bị ghi đè.
Chưa upload gì lên Drive trong vòng này: hai file Drive 1BmC8PqVL5zfbYhrmaWa_1RECoyxNwu4i (pptx) và
1cIMhViZsT3CzKqA8qWA4RA3DtVnp2Wdj (pdf) vẫn là bản v2.

## Thư mục

- `deck/`: script dựng deck. `build3.py` là bản đang dùng (v7 → v8). Nội dung từng slide ở `slides1.py` (mục 1–6),
  `slides1b.py` (mục 7–10), `slides2.py` (mục 11–20). Hàm chung: `common.py` (nhân bản slide mẫu, header, nền, notes),
  `kit.py` (chữ, bảng, biểu đồ native, tách run thiếu glyph sang Cambria Math), `stage.py` (sơ đồ giai đoạn kiểu ảnh mẫu 1.png:
  panel, node, bar, arrow, thẻ chi tiết viền đỏ, chèn công thức PNG). `v7.pptx` là đầu vào của build3.
  `r8/` là ảnh render của v8 (bản trước các sửa cuối). `v2..v6`, `edited.pptx`, `smallfix.pptx`, `backup-before.pptx` là bản cũ.
- `tex/`: `eqlist.py` (63 công thức LaTeX, khoá → mã), `mkeq.py` (render ra `tex/eq/*.png`, 300 dpi, nền trong),
  `mkfig.py` (biên dịch riêng 4 hình TikZ/pgfplots của báo cáo ra `tex/fig_png/`: vig3d, dc3d, rgbcube, timeline),
  `tex/fig/` là bản chép thư mục fig của báo cáo (ảnh jpg d2_*, art_*, enh*...). `neweq.txt`: cách thêm công thức (key<TAB>latex).
- Gốc: `render.ps1`, script phụ của các vòng trước (`make_assets.py`, `edit_deck.py`, `serve_one.py`...).

## Lệnh dựng lại (Git Bash, đứng trong thư mục tương ứng)

```
# công thức (cần MiKTeX xelatex + pdftocairo; font Be Vietnam Pro ở D:/IOT_Tools/downloads/fonts/)
cd tex && rm -f eq/*.png && PYTHONUTF8=1 py -3.13 -I mkeq.py        # in "rendered 63"
PYTHONUTF8=1 py -3.13 -I mkfig.py                                    # 4 hình vào fig_png/
# deck
cd ../deck && PYTHONUTF8=1 py -3.13 -I build3.py v7.pptx v8.pptx "$PWD"   # in "slides 28 glyph splits N"
# render (PowerPoint COM, đường dẫn tuyệt đối)
powershell.exe -NoProfile -File render.ps1 -Deck "<abs>\v8.pptx" -OutDir "<abs>\r8" -Slides "1 2 3 ... 28"
PYTHONUTF8=1 py -3.13 -I sheets.py r8 2,3 out.png                    # ghép 2 slide để soát
PYTHONUTF8=1 py -3.13 -I dumptext.py v8.pptx > v8text.txt            # toàn bộ chữ + notes để grep
# PDF
powershell.exe -NoProfile -File pdf.ps1 -Deck "<abs>\v8.pptx" -Pdf "<abs>\XLA_UAV_Nhom10.pdf"
```

## Đã xong: 28 slide (≤ 30), số slide → tiêu đề → mục dàn ý v3

Nhãn badge trên slide đánh số khác số mục dàn ý; cột cuối là số mục trong `dan-y-slide-v3.md` / `transcript.yaml`.

| # | Tiêu đề | Mục |
|---|---|---|
| 1 | Bìa | 1 |
| 2 | Năm lỗi đo được trên khung thật, nặng nhất là ám hồng | 2 |
| 3 | Một khung đi qua bốn chặng, nhận diện luôn đọc khung gốc (sơ đồ hệ thống) | 2 |
| 4 | Mù và tối góc là hai phép biến đổi lên ảnh sạch J | 3 |
| 5 | Khối 8×8 lộ ra vì q12; vùng sáng hồng vì G bão hoà trước | 3 |
| 6 | Chia theo khối 30 s để khung test không có bản sao trong dev | 4 |
| 7 | D2 dựng lại sáu khâu hỏng, 12 tham số do CMA-ES khớp (sơ đồ + thanh mục tiêu) | 5 |
| 8 | D2 lệch thống kê ít hơn D1 7 lần, C2ST vẫn tách được cả hai (sơ đồ 3 thước + 3 biểu đồ) | 6 |
| 9 | YOLO11s đọc ba tầng P3–P5 và học hộp bằng CIoU cộng DFL | 7 |
| 10 | mAP50 chấm trên 198 khung thật, CI 95 % bootstrap theo khối (sơ đồ IoU → PR → AP → bootstrap) | 8 |
| 11 | Bộ nhận diện qua bốn giai đoạn, từ COCO tới gói ONNX | 9–10 |
| 12 | Dev-val xếp vòng 1 đầu, test lại xếp vòng 2 đầu (biểu đồ) | 9–10 |
| 13 | Vòng 2 cao nhất (0,313), nhưng CI của 9 bộ chồng lên nhau (biểu đồ + bảng t1 đủ 9 dòng) | 11 |
| 14 | (giữ từ v7) biểu đồ mAP50 với báo nhầm | 11 |
| 15 | (giữ từ v7) khung ví dụ ba bộ nhận diện | 11 |
| 16 | Bộ lọc cổ điển đi ngược chuỗi hỏng: 9,2 ms trên CPU (sơ đồ chuỗi theo thứ tự code) | 12 |
| 17 | Khử mù: ước A và t̃ từ dark channel, làm mịn bằng guided filter | 12 |
| 18 | NAFNet-w16 không dùng hàm kích hoạt, 1,14 triệu tham số | 13 |
| 19 | (giữ từ v7) ô ảnh gốc / cổ điển / NAFNet | 14 |
| 20 | Mạng đạt 3/7 tiêu chí, cổ điển giảm mù 52 % so với 37 % | 14 |
| 21 | Người chấm mù chọn bản cổ điển 24/25 lượt (biểu đồ Wilson) | 14 |
| 22 | (giữ từ v7) E4 | 15 |
| 23 | IOT-CV chỉ nạp gói ONNX qua được ba phép kiểm | 16 |
| 24 | (giữ từ v7) Ở nhịp thật p95 tổng 54,1 ms, vượt ngân sách 50 ms | 16 |
| 25 | (giữ từ v7) Chọn gì và vì sao | 17 |
| 26 | Dev-val chọn sai, mạng thua cổ điển, mọi số từ một buổi bay | 18 |
| 27 | Kết luận | 19 |
| 28 | Tài liệu tham khảo (16 mục của báo cáo) | 20 |

Đã bỏ slide "Phân công nhóm 10".

## Còn lại

### A. Sửa theo đính chính của điều phối (code là chuẩn)

1. Khử hồng, slide 16: công thức `cl_gate` đã đúng code (ramp(max(R,B); 230; 10) · ramp(min(R,B) − G; 6; 14)).
   Còn phải: ghi nhãn rõ "cổng sửa (bộ lọc)" cạnh `cl_gate` và đặt `pinkrule` (R−G > 25 và B−G > 25) cạnh nó với nhãn
   "luật đo (chỉ để đếm điểm hồng)"; node "mặt nạ α từ max(R,B) ≥ 230" phải nêu cả min(R,B) − G ≥ 6; câu bên phải
   "g bật khi R hoặc B gần trần và vượt G" → "R hoặc B gần trần và cả hai vượt G". Slide 20 dòng "p: tỉ lệ điểm R−G > 25..."
   ghi thêm "(luật đo)". 0,704 giữ Y đã có.
2. Thứ tự tầng cổ điển: đã theo code (khử khối → khử hồng → bù tối góc → khử mù → làm nét). Không cần sửa.
   Comment đầu `configs/enhance_classic.yaml` ghi thứ tự khác, sai (chưa sửa file đó, không thuộc phần slide).
3. Khử mù, slide 17: công thức `trans` đã đúng (t̃ = 1 − ω·dark/Ā, ω 0,85, t ≥ 0,4). Câu cuối thẻ trái
   "kéo mức đen về ≤ 8" phải đổi thành "kéo giãn khi mức đen ra thấp hơn min(mức đen vào; 8)" (classic.py:351-357).
4. Khoảng cách thống kê là TRUNG BÌNH 18 số hạng W1/IQR: công thức `obj_w`/`stat1` đã là 1/|K| Σ. Slide 7, dòng nhỏ trên thanh
   mục tiêu "K: 16 thống kê mức thấp + đường NLF + phổ nhiễu" → "K: 18 đại lượng (màu, dark channel, nhiễu, độ nét, khối, tối góc,
   dải mép, phổ nhiễu)". Không dùng chữ "tổng".
5. E2: tiêu đề và sơ đồ GĐ3 đã ghi "thầy vòng 1: E1 D2". Còn: đổi nhãn biểu đồ "E2 thầy YOLO11m" thành "YOLO11m (thầy dự kiến)"
   ở `slides2.T1`, `slides1b.s_train_results` (cats), và kiểm slide 14 (giữ từ v7) có nhãn cũ không.
6. Slide 25 (giữ từ v7, sửa trong build3 hoặc một hàm mới): ô quyết định của hàng "Bộ lọc cổ điển" đang là "không ship" →
   "đề xuất"; lý do → "đề xuất cho luồng người xem, chưa vào hợp đồng ONNX"; chú thích "Tốc độ chỉ đo cho model đã ship" → "Tốc độ: p50 mỗi khung".
7. Notes slide 13 còn chuỗi cấm ("ghi nhầm 'YOLO11n'") → xoá. Notes slide 8 câu "'×100' trong bảng t2 không khớp code" → xoá
   (giữ làm mối lo báo điều phối; transcript ghi "KID nhân 100").

### B. Sửa bố cục đã lên danh sách sau khi soát render r8

Đã làm (chưa render lại): NBSP cho số có dấu cách nghìn trong `kit.add_runs`; chú thích ảnh slide 2; chữ sơ đồ slide 3
(tên panel "1. Nén và gửi"..., ô đầu vào, thẻ camera, thẻ chặng 3); chữ thẻ slide 4 (bỏ gạch dưới).
Chưa làm:
- slide 5: thẻ phải đặt `pinkrule` 0,9 lên đầu, rgbcube rộng ~3,1 in bên dưới, chữ bên (Y 222/255, 440 điểm sáng), `tone`/`tone_v` ở y ~5,38/5,72;
  thẻ trái rút ngắn hai dòng giải thích, DQT ở 5,35, `block` ở 5,72.
- slide 6: cột bảng [2.45, 0.75, 0.95, 2.8], chữ "dev (gồm dev-val 100)", "hồ sơ, hiệu chỉnh D2, nhãn giả", "chấm điểm, 39 khung có người", tiêu đề cột "Hộp"; thẻ timeline cao hơn chút.
- slide 7: thẻ D1 → ["Real-ESRGAN", "bậc hai"].
- slide 8: panel 1,85–4,6, thanh ở 4,18; thước 1 dùng `stat1` + `w1` (~0,75); C2ST: `hp` 0,75 + node "CNN 4 lớp" cùng hàng, `c2st` 0,75,
  thanh "mảnh 64×64; CNN 60 929 tham số"; KID: hàng node DINOv2, `kernel` 0,7, `kid` ~0,6; biểu đồ 4,72–6,95; rút thẻ "Đọc bảng".
- slide 9: thẻ Head → ["mỗi ô đoán 4 khoảng cách,", "điểm lớp qua sigmoid; NMS sau"]; chữ cạnh `tal` dời x 11,15, rộng 1,5.
- slide 10: dựng lại: đầu vào rộng 1,5; 4 panel rộng 2,52 từ x 2,24, node bằng chữ, 1,85–3,75; thẻ 3,88–5,75 chứa `iou`, `pr2`, `ap2`, `boot2`
  (scale 0,8) + 1 dòng; dải dưới `fp` + `map` + tiêu chí. Các công thức tách đã render sẵn trong `tex/eq/`.
- slide 11: thẻ cao 1,2, chữ 12–13 pt, dòng ≤ 24 ký tự (nội dung đã chốt: Dữ liệu ["VisDrone", "3 006 + 487"]; GĐ1 ["ảnh vào 1 280 px;",
  "bỏ dải mép: 0,067 → 0,126"]; GĐ2 ["6 epoch, batch 4, 1 280 px", "AdamW lr 0,002, tắt HSV"]; GĐ3 ["τ cao 0,55 rồi 0,75", "vệt ≥ 3 khung, 4 epoch"];
  GĐ4 ["ngưỡng 0,0767", "NMS 0,5, ≤ 100 hộp"]); dời nhãn vòng lặp.
- slide 12: chip dev-val/test dời khỏi tiêu đề (góc dưới phải thẻ biểu đồ).
- slide 13: cột bảng [1.4, 1.1, 0.8, 0.9, 0.8], tiêu đề cột "FP" (chú thích giải nghĩa).
- slide 16: độ rộng chuỗi khử mù [0.56, 0.32, 0.32, 0.42, 0.32] cách 0,08; thẻ khử khối ["α 10, β 4, s 0,5, λ 0,5", "lưới còn nguyên"]; rút chữ 0,704.
- slide 18: ảnh cặp nhỏ lại, nhãn "vào: D2" và "đích: D0".
- slide 21: chữ → "p: tỉ lệ thắng, n = 25. Cận dưới > 0,5 là thắng, cận trên < 0,5 là thua."
- slide 26: cỡ chữ 15. Slide 28: 13 pt.

### C. Kiểm và xuất (chưa chạy trên bản mới)

- Render cả 28 slide và soát từng slide (kể cả 1, 14, 15, 19, 22, 24, 25 giữ từ v7), sửa tràn chữ.
- Đếm typeface (Be Vietnam Pro, SemiBold, Cambria Math cho ký hiệu); grep chuỗi cấm = 0: ESP32-CAM, OV2640, YOLO11n, "bản trước",
  "bản cũ", "ghi nhầm", archive, v2-nhom, em-dash trong văn xuôi, từ sáo. Đối chiếu chữ và công thức với `transcript/transcript.yaml` và báo cáo.
- Notes ngắn, chỉ gạch đầu dòng.
- Xuất PDF (pdf.ps1) + `pdffonts` xác nhận font nhúng; chép sang `docs/nop-bai/` chỉ khi được bảo; dựng lại `render-sau/`.
- Upload: pptx thành phiên bản mới của 1BmC8PqVL5zfbYhrmaWa_1RECoyxNwu4i, PDF thành phiên bản mới của 1cIMhViZsT3CzKqA8qWA4RA3DtVnp2Wdj
  (Manage versions, KHÔNG tạo file mới), rồi kiểm sha256 qua `page.context().request.get('https://drive.google.com/uc?export=download&id=<id>')`.

## Bẫy đã gặp

- Heredoc của Bash nuốt `\\` của LaTeX (mất xuống dòng): ghi công thức bằng file (`neweq.txt`, Write tool), không bằng heredoc.
- `re.findall` với chuỗi chứa `\e` báo bad escape: tách bằng `str.find`.
- `common.set_bg` lỗi NoneType khi slide là chính slide mẫu của nó: đã thêm `if src is slide: return`.
- Be Vietnam Pro thiếu chữ Hy Lạp, mũi tên, ∈, √, ′, ↓, ↑, ■: `kit.add_runs` + `build3.split_runs` đẩy các ký tự đó sang run Cambria Math
  (phải báo khi đếm typeface).
- Nhãn trục pgfplots ra font toán: thêm `\pgfkeys{/pgf/number format/assume math mode=true}`. Timeline chồng chữ ở 12 pt: dùng 10 pt.
- Công thức rộng quá 5 in: tách bản `aligned` hai dòng (dct, kid, ciou, dfl, gf) hoặc bản tách (stat1, w1, pr2, ap2, boot2).
- Kích thước công thức: `stage.eq_size(name, scale)` (ảnh 300 dpi, scale mặc định 0,9).
- Upload Drive bằng Playwright: `page.bringToFront()`; cả chuỗi More actions → File information → Manage versions → Upload new version
  trong MỘT lần `run_code` (menuitem ẩn: dispatch mouseover/mousedown/mouseup/click lên `closest('[role=menuitem]')`; "Upload new version"
  bấm bằng toạ độ từ boundingBox; `waitForEvent('filechooser')`, gán File qua DataTransfer rồi phát `change`; sau đó gọi `browser_file_upload`
  không đối số để gỡ modal). File phục vụ bằng `py -3.13 -m http.server <port> --bind 127.0.0.1` trong thư mục tạm, xong thì tắt server và xoá thư mục.
- Mối lo cần báo: nhãn "YOLO11n" trong bảng t1/t5 gốc (mốc thật là YOLO11s @1280); nhãn "KID ×100" của t2 cần đối chiếu lại `xla/eval/kid.py`;
  comment thứ tự sai đầu `configs/enhance_classic.yaml`; chữ thẻ sơ đồ 12–13 pt (dưới mức 14 pt của thân bài); font người dùng cài ở ổ C.
