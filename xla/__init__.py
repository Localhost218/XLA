"""XLA: xử lý ảnh chất lượng thấp cho camera drone ESP32-S3 + OV3660 (nhìn thẳng xuống).

Năm gói con:

- ``xla.camera``  — đo đặc trưng camera thật, hồ sơ camera.
- ``xla.degrade`` — chuỗi làm xấu ảnh sạch cho giống camera thật.
- ``xla.enhance`` — làm rõ ảnh (cổ điển và mạng khôi phục).
- ``xla.detect``  — nhận diện người.
- ``xla.eval``    — chấm điểm, độ giống thật, so sánh A/B.

Plan: ``plans/261118-ai-anh-chat-luong-thap/plan.md``.
"""

__version__ = "0.1.0"
