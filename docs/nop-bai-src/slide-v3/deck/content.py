# Text rewrite map keyed by ORIGINAL slide number (1-based). Body: {old paragraph text: new text}.
# NOTES: replaces notes paragraph 0 (the source line paragraph stays).
DROP = [2]  # agenda slide: topic labels only
BODY = {
3: {
 "Camera thật làm hỏng ảnh theo ba kiểu đo được": "Ám hồng ở 31 % khung bay, kèm mù và vỡ khối JPEG",
 "Kênh G sụt khi điểm ảnh gần bão hoà.": "Điểm gần bão hoà mất kênh G nên ngả hồng.",
 "Dark channel p50 144/255; góc 0,81× tâm.": "Dark channel trung vị 144/255. Góc ảnh chỉ sáng bằng 0,81 tâm.",
 "Khối 8×8, 4:2:2; dải tối ~14 px.": "Tỉ số khối 8×8 là 2,68. Mép trái tối một dải 14 px.",
},
4: {
 "Chỉ chấm điểm trên khung bay thật, chia theo khối thời gian": "198 khung test lấy từ chuyến bay thật, chia theo khối 30 giây",
 "Chỉ để chọn model và ngưỡng.": "Chỉ dùng để chọn model và ngưỡng.",
 "300 trong nhà + 159 khung bay, để đếm báo nhầm.": "300 khung trong nhà và 159 khung bay không có người.",
},
5: {
 "D2 sao chép camera để làm xấu ảnh huấn luyện": "D2 làm xấu ảnh VisDrone theo số đo của chính camera",
 "Cắt kênh → ám hồng, nhiễu": "Cắt kênh ra ám hồng, nhiễu",
 "D1: chuỗi Real-ESRGAN, tham số ngẫu nhiên   ·   D2: tham số hiệu chỉnh tự động cho khớp khung dev thật":
   "D1 rút tham số ngẫu nhiên theo kiểu Real-ESRGAN. D2 tự hiệu chỉnh tham số theo khung dev thật.",
},
6: {
 "D2 khớp thống kê pixel, nhưng bộ phân loại vẫn tách được": "D2 kéo lệch thống kê từ 5,93 xuống 0,80, C2ST vẫn 0,95",
 "D1 → D2, tổng độ lệch 18 thống kê mức thấp. Thật–thật: 0,33.": "D1 → D2, tổng lệch của 18 thống kê mức thấp. Sàn thật với thật 0,33.",
 "D1 · D2. Mức 0,5 là không phân biệt được. Thật–thật: 0,54.": "D1 · D2. Bộ phân biệt thật/giả đúng 95 % với D2. Sàn 0,54.",
 "D1 · D2, trên đặc trưng DINOv2. Thật–thật: 0,37.": "D1 · D2, đo trên đặc trưng DINOv2. Hai bộ gần như bằng nhau.",
 "Khớp thống kê là cần nhưng chưa đủ: nội dung ảnh VisDrone vẫn khác bãi bay của nhóm.":
   "Nhóm đoán bộ phân biệt nhận ra cảnh phố VisDrone, khác bãi bay. Giả thuyết này chưa kiểm riêng.",
},
7: {
 "Nhận diện chạy trên khung gốc, ảnh rõ chỉ cho người xem": "Bộ nhận diện chạy trên khung gốc, ảnh rõ chỉ để người xem",
 "DQT q12 · tối góc · hồng": "bảng lượng tử q12, tối góc",
},
8: {
 "YOLO11s học ảnh D2, rồi tự huấn luyện trên khung bay": "YOLO11s học ảnh D2, rồi học thêm nhãn giả của YOLO11m",
},
9: {
 "Bộ lọc cổ điển chạy ngược chuỗi suy giảm của camera": "Khử hồng trước khi khử mù thì sạch 93 % điểm hồng",
 "Khử khối 8×8 ngay sau giải mã, rồi khử hồng.": "Khử khối ngay sau giải mã đưa tỉ số khối về 2,15.",
 "Dark channel 15 px, unsharp trên Y.": "Dark channel cửa sổ 15 px, rồi làm nét kênh Y.",
 "Không dùng siêu phân giải và giải chập: trên khung thật chúng làm nổi khối JPEG.":
   "Siêu phân giải và giải chập bị bỏ vì làm nổi khối JPEG trên khung thật.",
},
10: {"Mặt nạ mềm chỉ chọn vùng hồng gần bão hoà": "Mặt nạ chỉ bắt vùng hồng sáng từ mức 230"},
11: {
 "Sửa sắc độ gần bảo toàn độ chói": "Một lượt sửa đổi độ chói từ 218,0 thành 218,1",
 "δ = 0,56·α·e; hệ số 0,704 bù vào G giữ Y gần như không đổi.": "δ = 0,56·α·e. G được cộng 0,704·δ nên độ chói gần như giữ nguyên.",
},
12: {"Dark channel ước lượng độ truyền t(x) để khử mù": "Bộ cổ điển giảm dark channel 51,6 % trên tập test"},
13: {
 "E1: D2 hơn D1 về mAP50, nhưng CI còn chạm 0": "D2 giảm báo nhầm từ 0,99 xuống 0,30 hộp mỗi khung",
 "Báo nhầm": "mAP50 chưa tách",
 "D2: 0,30 hộp/khung bay không người, D1: 0,99. Chênh có CI trọn dưới 0.": "D2 hơn D1 0,093, nhưng CI [−0,073; +0,252] vẫn chứa 0.",
 "Dev-val chọn D2 (0,329). D0 cao trên test nhưng báo nhầm 1,20.": "Dev-val xếp D2 đầu (0,329). D0 điểm test cao hơn nhưng báo nhầm 1,20.",
},
14: {
 "E2: dev-val chọn trò vòng 1, test lại nghiêng về vòng 2": "Dev-val chọn vòng 1, nhưng trên test vòng 2 gấp đôi",
 "Dev-val chọn sai": "Luật đã chốt",
 "Dev-val vòng 2 là 0,296, thấp hơn vòng 1 (0,379), nên theo luật ship vòng 1.": "Trên dev-val vòng 2 chỉ được 0,296, vòng 1 được 0,379, nên nhóm ship vòng 1.",
 "Bundle đã ship": "Bundle ONNX",
 "detector-v1.0.0: mAP50 0,120, lệch bản .pt −0,003; 20,0 ms p50 ORT.": "detector-v1.0.0 đạt mAP50 0,120 và chạy 20,0 ms p50 trên ORT CUDA.",
},
15: {"Cổ điển và mạng trên cùng một khung bay": "Mạng xoá gần hết hồng, cổ điển khử mù tốt hơn"},
16: {"Chấm mù A/B: người chấm chọn bản cổ điển 24/25 lượt": "Người chấm mù chọn bản cổ điển 24/25 lượt"},
17: {
 "E4: làm rõ trước khi nhận diện làm bộ đã tinh chỉnh kém đi": "Cho ảnh qua mạng trước, bộ đã tinh chỉnh mất 0,085 mAP50",
 "Quy tắc D5: nhận diện luôn chạy trên khung gốc": "Nên bộ nhận diện luôn chạy trên khung gốc",
},
18: {
 "Trong IOT-CV: theo kịp camera, nhưng p95 vượt ngân sách": "Trong IOT-CV detector chạy 23,6 ms p50, p95 tổng vượt 50 ms",
 "Phát lại chuyến 02/11 · ORT CUDA, RTX 4060 Laptop · bỏ 0,3 % khung · p95 tổng 54,1 ms > ngân sách 50 ms":
   "Phát lại chuyến 02/11 trên RTX 4060 Laptop, ORT CUDA. Bỏ 0,3 % khung. p95 tổng 54,1 ms.",
},
19: {"Hai bundle ONNX nạp vào IOT-CV qua một hợp đồng cố định": "IOT-CV từ chối bundle nào sai hợp đồng 640×480 BGR"},
20: {
 "Giới hạn của kết quả": "Số đo chỉ từ một buổi bay, 39 khung có người",
 "Mọi số đo từ chuyến 02/11/2026; ánh sáng, độ cao, nền khác có thể đổi kết quả.": "Chỉ có chuyến sáng 02/11/2026. Ánh sáng hay nền khác có thể cho số khác.",
 "39 khung có người, CI của mAP rộng khoảng ±0,1; dev-val 100 khung chọn sai ở E2.": "CI của mAP rộng khoảng ±0,1. Dev-val 100 khung đã chọn sai ở E2.",
 "Hồ sơ chỉ ở q12": "Chỉ đo ở q12",
 "Bảng DQT đo ở q12; web hạ xuống q40 khi Wi-Fi yếu, mức đó chưa đo.": "Web hạ xuống q40 khi Wi-Fi yếu. Bảng lượng tử ở q40 chưa đo.",
 "Ngân sách ms": "Quá 50 ms",
 "Ở nhịp camera, GPU hạ xung giữa hai khung: p95 tổng không đạt 50 ms.": "GPU hạ xung giữa hai khung, nên p95 tổng lên 54,1 ms.",
},
22: {"Tài liệu phương pháp và nguồn thực nghiệm": "Tài liệu tham khảo và mã nguồn"},
23: {
 "Kết quả và hướng phát triển": "D2 giúp bớt báo nhầm, làm rõ ảnh không giúp nhận diện",
 "Hai câu trả lời": "Hai câu hỏi",
 "D2 hơn D1 ở báo nhầm, mAP chưa chắc. Làm rõ trước khi nhận diện làm hại bộ đã tinh chỉnh.":
   "mAP50 của D2 và D1 chưa tách được. Ảnh qua mạng làm bộ đã tinh chỉnh mất 0,085.",
 "Sửa phơi sáng, WB ở cảm biến · thêm khung có người · ràng buộc màu, thời gian cho mạng · thử imgsz 960.":
   "Chỉnh phơi sáng và cân bằng trắng ngay ở cảm biến, đo bảng lượng tử q40, thu thêm khung có người, thử imgsz 960.",
},
}

NOTES = {
1: "Nhóm làm hai việc trên khung bay thật của camera ESP32-S3 + OV3660, JPEG 640×480 ở chất lượng 12. Việc đầu là làm ảnh dễ nhìn hơn cho người lái. Việc sau là tìm người để cảnh báo. Ba ảnh nhỏ là cùng một khung bay ngày 02/11/2026: khung gốc, sau bộ lọc cổ điển và sau mạng NAFNet. Hình drone trên bìa chỉ để minh hoạ.",
3: "Nhóm đo hồ sơ camera trên 1115 khung dev của chuyến 20261102-100941. Cứ 100 khung thì 31 khung có hơn 0,5 % điểm hồng, luôn ở vùng gần bão hoà. Dark channel trung vị 143,7/255, ngang ảnh chụp trong sương, và góc ảnh chỉ sáng bằng 0,81 lần tâm. Camera nén JPEG q12, lấy mẫu 4:2:2, nên tỉ số khối 8×8 trung vị là 2,68. Mép trái có một dải tối khoảng 14 px; hệ thống cắt bỏ 16 px. Ảnh phải là vùng mép trái phóng to 4 lần.",
4: "Toàn bộ khung lấy từ chuyến bay 02/11/2026, phiên 20261102-100941. Hai khung chọn cách nhau ít nhất 1 giây, rồi chia theo khối 30 giây thay vì chia ngẫu nhiên, để tập huấn luyện không có khung gần giống khung test. Tập test có 198 khung, 39 khung có người, tổng 149 hộp. Dev-val 100 khung chỉ dùng chọn model và ngưỡng. 300 khung trong nhà và 159 khung bay không người dùng để đếm báo nhầm. Nhãn do nhiều model và SAHI gán trước, nhóm soát từng khung bằng mắt, chủ dự án duyệt lần cuối trên Label Studio.",
5: "D1 là cách làm quen thuộc khi chưa biết camera đích: chuỗi suy giảm kiểu Real-ESRGAN, tham số rút ngẫu nhiên. D2 bám camera thật. Nó phủ màn mờ theo mô hình khí quyển, làm nhoè bằng PSF và tối góc cos⁴ theo tiêu cự đo được, cắt từng kênh rồi nhân cân bằng trắng nên vùng sáng hoá hồng, thêm nhiễu theo NLF và dải tối mép trái, cuối cùng nén JPEG một lần bằng đúng bảng lượng tử q12, 4:2:2 của camera. Tham số do xla.degrade.calibrate tự hiệu chỉnh. Ảnh nguồn là VisDrone. NOMAD bị loại vì ảnh chụp xiên 45–60°. Không được đăng lại ảnh VisDrone, nên ảnh minh hoạ ở đây là một khung bay của nhóm đã qua bộ lọc cổ điển rồi cho qua D2.",
6: "Ba thước đo, cùng chạy trên patch ảnh. Lệch thống kê: D1 5,928, D2 0,796, D2 cộng Real-SR 0,663; hai tập thật so với nhau cho 0,330. C2ST là độ chính xác của bộ phân loại tách patch thật khỏi patch làm xấu: D1 0,936, D2 0,950, thật với thật 0,542. KID nhân 100: D1 4,443, D2 4,409, thật với thật 0,371. Vậy D2 giống camera ở mức pixel nhưng bộ phân loại vẫn tách được. Nhóm nghĩ lý do là cảnh phố của VisDrone khác bãi bay; giả thuyết này chưa kiểm riêng.",
7: "Mọi bước dùng chung hồ sơ camera: bộ làm xấu D2, bộ lọc cổ điển, cả khung giả trong web. YOLO11s học trên ảnh D2, rồi tự huấn luyện thêm trên khung bay chưa nhãn, xuất ONNX ở khung 1280. Nhánh làm rõ có bộ lọc cổ điển và NAFNet bề rộng 16. Quy tắc D5 giữ bộ nhận diện trên khung gốc; ảnh rõ chỉ để người xem. Hình lớp ảnh ở giữa chỉ minh hoạ ý tưởng.",
8: "Mốc so sánh là YOLO11s COCO có sẵn, khung 1280, bỏ 16 px mép trái. E1 tinh chỉnh YOLO11s trên VisDrone qua từng bộ làm xấu: D0 ảnh sạch, D1, D2, D2 cộng siêu phân giải; model được chọn bằng mAP50 trên dev-val. E2 dùng YOLO11m làm thầy, gán nhãn giả cho khung bay chưa nhãn, trò YOLO11s học lại hai vòng. Nhóm ship trò vòng 1 theo dev-val, xuất ONNX có cắt mép và resize trong graph. Ngưỡng vận hành 0,077, chọn bằng F1 lớn nhất trên dev-val. Hình trái là nhãn tay, hình phải là hộp do chính bundle detector-v1.0.0 vẽ trên một khung test.",
9: "Thứ tự bước đi ngược đường ảnh bị hỏng trong camera: JPEG, rồi cắt kênh và cân bằng trắng của ISP, rồi tối góc, rồi màn mờ. Khử khối ngay sau giải mã đưa tỉ số khối về 2,15. Khử hồng trước khử mù giảm 93 % điểm hồng; đặt sau thì chỉ giảm 46–68 %, vì khử mù khuếch đại chênh lệch màu. Ba ảnh là kết quả cộng dồn sau từng bước trên một khung bay có ám hồng. Bước khử hồng chỉ dịch sắc độ, giữ Y, và chỉ chạy ở chỗ gần bão hoà mà cả R lẫn B vượt G, nên vật màu đỏ không bị đụng.",
10: "Cổng có hai điều kiện: max(R, B) từ 230 trở lên, và cả R lẫn B vượt G từ 6 mức. Vật đỏ bão hoà có B thấp nên lọt qua. Cổng được nở 21 px rồi làm mềm thành mặt nạ alpha. Mỗi lượt trừ delta khỏi R và B, cộng 0,704 delta vào G; bộ lọc chạy hai lượt. Ảnh giữa là alpha thật của lượt đầu trên khung này.",
11: "Lấy một điểm gần bão hoà R, G, B = 245, 200, 240, alpha bằng 1. Khi đó e = (245 + 240)/2 − 200 = 42,5 và delta = 0,56 × 42,5 ≈ 23,8. Sau một lượt: R ≈ 221, G ≈ 217, B ≈ 216. Độ chói Y trước là 218,0, sau là 218,1. Bộ lọc thật còn chạy lượt hai và bơm thêm chi tiết của kênh G vào chỗ bão hoà nặng.",
12: "Dark channel là min của ba kênh, rồi lấy min trong cửa sổ 15 px. Ánh sáng môi trường A là trung bình các điểm có dark channel từ phân vị 99,9, kẹp trong khoảng 180–250. Độ truyền t = 1 − 0,85·dc/A, sàn 0,4, sau đó guided filter theo Y ở 1/4 độ phân giải làm mịn t. Bản đồ giữa là t thô; màu tím là chỗ mù dày. Ảnh phải đã qua cả bước làm nét unsharp trên Y. Trên tập test, bộ cổ điển giảm dark channel trung vị 51,6 %.",
13: "mAP50 trên 198 khung test: YOLO11s COCO có sẵn 0,126, D0 0,297, D1 0,143, D2 0,236, D2 cộng siêu phân giải 0,149. D2 hơn D1 0,093 nhưng CI 95 % là [−0,073; +0,252], còn chứa 0, nên chưa nói được D2 hơn về mAP. Chỗ D2 hơn rõ là báo nhầm: 0,302 so với 0,994 hộp sai mỗi khung bay không người, chênh −0,692, CI [−1,238; −0,246]. D0 có CI rất rộng và báo nhầm cao. CI tính bằng bootstrap theo khối thời gian, 2000 mẫu.",
14: "Thầy YOLO11m đạt 0,218. Trò YOLO11s vòng 1 đạt 0,156, vòng 2 đạt 0,313, cao nhất trên test. Nhưng trên dev-val vòng 2 thấp hơn (0,296 so với 0,379), và luật chốt từ trước là chọn theo dev-val, nên nhóm ship vòng 1. Bundle ONNX chấm bằng đúng đường ONNX (cắt mép, letterbox 1280 trong graph) đạt 0,120, CI [0,032; 0,197], báo nhầm 0,881 hộp mỗi khung bay không người. ORT CUDA chạy riêng mất 20,0 ms p50, 20,2 ms p95. Bài học: 100 khung dev-val quá ít để xếp hạng các model sát nhau.",
15: "Bộ cổ điển chạy 9,2 ms mỗi khung trên CPU (p50), giảm điểm hồng 89 % và dark channel 52 %. NAFNet bề rộng 16 học từ cặp ảnh sạch và ảnh D2, chạy 20,7 ms trên GPU. Nó xoá 99,9 % điểm hồng nhưng màu ra nguội và nhạt, dark channel chỉ giảm 37 % trong khi tiêu chí là 50 %. Cả hai đều làm nhấp nháy giữa hai khung liền nhau tăng lên: mạng 1,81 lần, cổ điển 2,18 lần. Mạng đạt 3 trên 7 tiêu chí đăng ký trước.",
16: "Người chấm thấy hai ảnh cạnh nhau, trái phải xáo trộn, và chọn ảnh dễ nhìn hơn. Lần 1 xem cả khung: mạng thắng ảnh gốc 11/25 lượt (0,44, Wilson [0,267; 0,629]) và chỉ thắng bộ cổ điển 1/25, tức bộ cổ điển thắng 24/25. Lần 2 phóng to 2 lần: mạng thắng ảnh gốc 13/25, thắng cổ điển 5/25. Ngưỡng đạt là 70 % khi so với gốc và 55 % khi so với cổ điển, nên mạng không được bật mặc định. Nếu chỉ đếm lượt có người, mạng thắng ảnh gốc 17/21; đây là quan sát sau khi chấm, không dùng để đổi kết luận.",
17: "Hai bộ nhận diện chạy trên ba bản của khung test, so mAP50 ghép cặp với khung gốc. Với YOLO11s COCO có sẵn, ảnh qua mạng tăng 0,069 nhưng CI [−0,033; +0,231] chứa 0; ảnh qua cổ điển giảm 0,046. Với bộ đã tinh chỉnh, ảnh qua mạng giảm 0,085, CI [−0,139; −0,011] nằm trọn dưới 0; ảnh qua cổ điển giảm 0,112. Bộ này học trên ảnh xấu kiểu D2 nên ảnh đã làm đẹp lại là ảnh lạ với nó. Luật chốt trước chỉ cho chuyển sang ảnh đã làm rõ khi CI nằm trọn trên 0, nên giữ khung gốc.",
18: "Hai model chạy trong backend IOT-CV bằng ONNX Runtime CUDA, trên JPEG ghi được của chuyến 02/11, phát lại đúng nhịp camera. Khi có người xem Ảnh rõ: detector 23,6 / 33,4 ms p50 / p95, enhancer 12,5 / 19,0 ms, tổng 37,5 / 54,1 ms, 7,2 khung mỗi giây, bỏ 0,3 % khung. Chạy hết tốc lực thì được 23,6 khung/s và p95 tổng 44,9 ms, trong ngân sách. Ở nhịp thật, GPU rảnh khoảng 100 ms giữa hai khung nên hạ xung, đẩy p95 qua 50 ms. Nhóm chấp nhận và ghi vào giới hạn.",
19: "Hợp đồng model (docs/hop-dong-model.md) cố định đầu vào: khung 640×480 BGR uint8 nguyên vẹn, việc cắt mép 16 px nằm trong graph. Detector trả hộp theo toạ độ khung gốc kèm điểm; enhancer trả khung cùng cỡ. Mỗi bundle có file ONNX, meta.json ghi sha256 và 3 khung golden để kiểm sai số. IOT-CV đặt bundle ở models/vision, không nạp model sai hợp đồng và báo lý do lên web. Ảnh trái là hộp của detector-v1.0.0 trên một khung bay.",
20: "Mọi số đo đến từ một bãi, một buổi sáng 02/11/2026. Tập test chỉ có 39 khung có người nên CI của mAP rộng, và dev-val 100 khung đã chọn sai model ở E2. Bảng lượng tử mới đo ở q12, trong khi web hạ chất lượng xuống q40 khi Wi-Fi yếu. Ở nhịp camera thật, laptop này không giữ được p95 dưới 50 ms. Nhãn do máy gán rồi người duyệt nên vẫn có thể sót người rất nhỏ. Các chỉ số NR-IQA chỉ để mô tả, không dùng để kết luận.",
21: "Mỗi người phụ trách phần việc ghi trên slide. Chủ dự án duyệt nhãn tập test và chấm A/B. Trợ lý AI (Claude Code) viết phần lớn mã và chạy thí nghiệm; nhóm đọc, duyệt và chịu trách nhiệm về kết luận.",
22: "Các phương pháp lấy từ những công trình trên rồi chỉnh theo camera của nhóm. Mọi bảng và hình sinh bằng scripts/reproduce.py từ kết quả đã lưu. Lệnh --check chấm lại 15 tệp dự đoán trên CPU và so sha256 của 9 bảng, hình. Danh sách đầy đủ ở BAO_CAO_DO_AN.md.",
23: "Câu 1: làm xấu theo số đo camera (D2) hơn làm xấu ngẫu nhiên ở độ giống thật mức pixel và ở báo nhầm, 0,30 so với 0,99 hộp mỗi khung bay không người; về mAP thì chưa kết luận được. Câu 2: làm rõ trước khi nhận diện làm bộ đã tinh chỉnh kém đi rõ; với bộ có sẵn thì có vẻ giúp nhưng CI còn chứa 0. Về ảnh rõ, người chấm chọn bộ cổ điển 24/25 lượt, nên mạng chưa bật mặc định. Việc tiếp theo: chỉnh phơi sáng và cân bằng trắng ở cảm biến, đo bảng lượng tử q40, thu thêm khung có người qua nhiều buổi, thêm ràng buộc màu và thời gian khi huấn luyện mạng, thử detector imgsz 960 hoặc FP16.",
}
