import sys
from pptx import Presentation
src, dst = sys.argv[1], sys.argv[2]
prs = Presentation(src)
slides = list(prs.slides)

EDITS = [
    # (slide_no, where, old, new)
    (3, "notes",
     " Bản trước của nhóm viết cho ESP32-CAM OV2640 và chấm trên ảnh tự sinh; camera thật khác loại.",
     " Mọi số đo lấy từ khung bay thật của camera ESP32-S3 N16R8 + OV3660 (JPEG 640×480, q12), chuyến 20261102-100941."),
    (9, "body",
     "Bỏ siêu phân giải và giải chập của bản trước: trên khung thật chúng làm nổi khối JPEG.",
     "Không dùng siêu phân giải và giải chập: trên khung thật chúng làm nổi khối JPEG."),
    (9, "notes",
     " Bộ lọc giữ ý tưởng mặt nạ hồng và cân bằng trắng có loại trừ của bản trước.",
     " Khử hồng chỉ dịch sắc độ, giữ Y, và chỉ chạy ở vùng gần bão hoà có cả R lẫn B vượt G, nên vật đỏ bão hoà không bị đụng."),
]

def runs_of_slide(s, where):
    if where == "notes":
        tf = s.notes_slide.notes_text_frame
        for p in tf.paragraphs:
            yield from p.runs
    else:
        def walk(shapes):
            for sh in shapes:
                if sh.shape_type == 6:
                    yield from walk(sh.shapes)
                elif sh.has_text_frame:
                    for p in sh.text_frame.paragraphs:
                        yield from p.runs
        yield from walk(s.shapes)

for no, where, old, new in EDITS:
    hits = [r for r in runs_of_slide(slides[no-1], where) if old in r.text]
    assert len(hits) == 1, (no, where, len(hits))
    r = hits[0]
    assert r.text.count(old) == 1
    r.text = r.text.replace(old, new)
    print(f"S{no} {where}: OK")

cp = prs.core_properties
assert "ESP32-CAM" in cp.title, cp.title
cp.title = "Làm rõ ảnh UAV và tìm người từ camera ESP32-S3 + OV3660"
print("core title:", cp.title)
prs.save(dst)
