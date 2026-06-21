# ViMamba-SER: Nhận dạng cảm xúc giọng nói tiếng Việt bằng Mamba và Text-enhancement

> Tài liệu ngữ cảnh cho project môn Speech Processing (SLP)
> Tác giả: Quang Huy Ngo (SE190631), Tan Dat Vo (SE190008), Ly Huynh Khanh (SE193800)
> Trạng thái thực tế: chỉ mình Quang Huy triển khai trong giai đoạn midterm
> Deadline midterm: 3 ngày kể từ ngày tạo file này

---

## 1. Bối cảnh đề tài

Đề tài đề xuất một kiến trúc kết hợp âm thanh và văn bản cho bài toán nhận dạng cảm xúc giọng nói tiếng Việt, dùng WavLM-Base làm bộ mã hóa âm thanh, PhoWhisper sinh transcript tự động, PhoBERT-v2 mã hóa văn bản, một module Text-aware Modality Enhancement (TME) căn chỉnh hai luồng đặc trưng theo thời gian bằng độ tương đồng cosine, và một mạng Mamba hai chiều làm tầng fusion cuối trước lớp phân loại. Bài toán phân loại 5 nhãn cảm xúc: Vui, Buồn, Giận, Lo lắng, Trung tính.

Hướng nghiên cứu này không cần thay đổi. Lý do nằm ở phần xác minh dưới đây.

---

## 2. Sự kiện đã xác minh (qua web search thực tế, không phải suy đoán của agent)

### 2.1. Về dataset VNEMOS

VNEMOS có tồn tại thật, công bố tại IEEE Conference 2024 (DOI tham chiếu: ieeexplore.ieee.org/document/10616411). Dữ liệu gồm khoảng 250 đoạn ghi âm, tổng cộng xấp xỉ 30 phút, lấy từ phim, series phim và chương trình truyền hình thực tế, chia đều cho 5 lớp cảm xúc, tức khoảng 50 đoạn mỗi lớp. Một số bài báo phái sinh ghi nhãn lớp thứ năm là "anxious", một số khác ghi là "fear", cần đối chiếu kỹ với bài gốc trước khi chốt tên nhãn trong báo cáo.

Điểm cảnh báo quan trọng nhất: đây là dataset rất nhỏ. Huấn luyện một mạng sâu phức tạp như Bi-Mamba fusion từ đầu trên 250 mẫu gần như chắc chắn overfit. Cách tiếp cận đúng là dùng các mô hình nền tảng pretrained làm bộ trích đặc trưng cố định (frozen feature extractor), chỉ huấn luyện phần đầu phân loại nhỏ phía sau.

Khả năng truy cập dữ liệu chưa được xác nhận chắc chắn. Các bài báo liên quan chỉ ghi link dạng bit.ly/VNEMOS, chưa rõ có repository công khai đầy đủ hay cần liên hệ tác giả. Có một repo GitHub liên quan đến nghiên cứu phái sinh dùng VNEMOS (github.com/fiyud/Emotional-Vietnamese-Speech-Based-Depression-Diagnosis-Using-Dynamic-Attention-Mechanism) có thể là nguồn tham khảo, nhưng không có gì đảm bảo repo này chứa sẵn audio gốc.

**Việc đầu tiên cần làm trước khi lập kế hoạch chi tiết hơn**: xác nhận có tải được file âm thanh VNEMOS hay không. Nếu trong vài giờ đầu không lấy được, cần phương án dự phòng (dataset SER tiếng Việt khác, hoặc dataset SER ngôn ngữ khác để chứng minh pipeline chạy được, có ghi chú rõ ràng đây là tập thay thế tạm thời).

### 2.2. Về trích dẫn TF-Mamba (2025)

Bài báo TF-Mamba: Text-enhanced Fusion Mamba with Missing Modalities, công bố tại EMNLP Findings 2025 (aclanthology.org/2025.findings-emnlp.602), đã xác minh là có thật. Trích dẫn này trong phần related work của đề xuất là hợp lệ, có thể giữ nguyên.

Ngoài ra trong quá trình search còn phát hiện một số công trình liên quan khác về Mamba cho multimodal emotion recognition công bố 2025 (ví dụ MaTAV, SALM, DepMamba), tất cả đều dùng tiếng Anh hoặc tiếng Trung, không có công trình nào áp dụng cho ngôn ngữ thanh điệu như tiếng Việt. Điều này củng cố thêm cho Research Gap 1 trong đề xuất, cần xác nhận lại bằng search riêng trước khi viết câu khẳng định "chưa có nghiên cứu nào" trong bản final, vì lĩnh vực này đang phát triển rất nhanh.

---

## 3. Quyết định phạm vi cho Midterm (3 ngày, một người làm)

Vì lý do thời gian và nhân lực, **không triển khai Mamba fusion và module TME đầy đủ cho midterm**. Đây là quyết định phạm vi có chủ đích, không phải bỏ cuộc với hướng nghiên cứu.

Phạm vi midterm gồm 3 phase rút gọn:

| Phase | Nội dung | Mục tiêu |
|---|---|---|
| 1 | Xác nhận và chuẩn bị dữ liệu VNEMOS | Có audio + nhãn sẵn sàng cho training |
| 2 | Baseline đơn phương thức (audio-only) | WavLM-Base embedding (mean-pooled) → MLP nhỏ → accuracy thật |
| 3 | Baseline song phương thức đơn giản (audio + text, fusion bằng concatenation) | PhoWhisper transcript → PhoBERT-v2 embedding → nối với audio embedding → MLP → accuracy thật, so sánh trực tiếp với Phase 2 |

Phase 4 (TME module), Phase 5 (Bi-Mamba fusion), Phase 6 (so sánh Mamba với fusion truyền thống, multi-seed) đều chuyển sang kế hoạch cho bản final report, có lý do kỹ thuật rõ ràng: cài đặt mamba-ssm yêu cầu phiên bản CUDA và PyTorch khớp chặt, thường mất nhiều giờ xử lý lỗi môi trường trên Colab, đây là rủi ro thật chứ không phải lý do hình thức.

Tiêu chí thành công cho midterm: có ít nhất một con số accuracy thật từ mô hình audio-only, và một con số accuracy thật từ mô hình audio+text fusion đơn giản, đủ để trả lời sơ bộ cho mục tiêu nghiên cứu thứ hai trong đề xuất (kiểm chứng chiến lược text-enhancement), dù chưa dùng đến Mamba.

---

## 4. Quy tắc kiểm định trích dẫn (bắt buộc với agent)

Đây là quy tắc quan trọng nhất rút ra từ project trước (RUL Prediction), nơi agent từng tự bịa ra các tên paper không tồn tại (TMSCNN, SiMBA-PINN, ShapTST, SHAPformer) và phải xóa thẩm định lại toàn bộ. Quy tắc dưới đây nhằm ngăn lặp lại sai lầm đó.

Agent chỉ được trích dẫn một bài báo khi vừa thực sự gọi công cụ tìm kiếm hoặc tải trang trong phiên làm việc hiện tại để xác nhận bài đó tồn tại, có tên tác giả, venue, năm công bố khớp với những gì sẽ ghi vào báo cáo. Không được lấy tên paper, số liệu, hoặc tên tác giả từ trí nhớ huấn luyện của mô hình rồi trình bày như đã xác minh.

Mọi trích dẫn đưa vào AGENTS.md hoặc báo cáo chính thức phải được phân vào một trong hai nhóm rõ ràng, đã xác minh hoặc cần tự kiểm tra lại, không được trộn lẫn hai nhóm này trong cùng một danh sách không phân biệt.

Khi agent không tìm thấy thông tin xác thực cho một tuyên bố (ví dụ "đây là nghiên cứu đầu tiên áp dụng Mamba cho tiếng Việt"), agent phải nói rõ là chưa xác minh được, thay vì khẳng định một chiều. Đây cũng là yêu cầu đã có sẵn trong GEMINI.md toàn cục.

Trước khi đưa bất kỳ con số benchmark nào từ paper khác vào báo cáo, agent phải tự mở bài báo gốc và đối chiếu, không được suy ra số liệu từ tên paper hoặc từ phần tóm tắt mà chưa đọc full text.

---

## 5. Tài liệu tham khảo

### Đã xác minh (có thể trích dẫn)

1. VNEMOS dataset, IEEE Conference 2024. DOI: ieeexplore.ieee.org/document/10616411. Quy mô: 250 đoạn, 30 phút, 5 lớp cảm xúc, baseline accuracy 89%.
2. TF-Mamba: Text-enhanced Fusion Mamba with Missing Modalities. EMNLP Findings 2025. aclanthology.org/2025.findings-emnlp.602.

### Cần tự verify trước khi trích dẫn vào báo cáo chính thức

3. Các công trình Mamba cho multimodal emotion recognition khác xuất hiện trong search 2025 (MaTAV, SALM, DepMamba, Quality-Controlled MERC with MAMBA Fusion) — cần đọc kỹ từng bài để xác nhận đây có phải tiền lệ trực tiếp cạnh tranh với research gap 1 hay không, vì lĩnh vực đang phát triển nhanh, khẳng định "chưa có ai làm" cần cẩn trọng.
4. Các nghiên cứu kết hợp HuBERT và PhoBERT cho SER tiếng Việt được nhắc trong phần related work của bản đề xuất gốc — chưa có tên tác giả cụ thể, cần tìm và xác minh trước khi giữ lại câu này trong bản final.

---

## 6. Lưu ý khi triển khai thực tế

Vì dataset rất nhỏ (50 mẫu mỗi lớp), nên dùng k-fold cross-validation thay vì chia train/test một lần duy nhất, để con số accuracy đáng tin cậy hơn và tránh báo cáo một kết quả may rủi do cách chia ngẫu nhiên.

Toàn bộ mô hình nền tảng (WavLM-Base, PhoWhisper, PhoBERT-v2) nên giữ frozen ở giai đoạn midterm, chỉ train phần classifier head, vừa tiết kiệm thời gian tính toán vừa tránh overfit trên tập dữ liệu nhỏ.

Ghi rõ trong báo cáo phiên bản cụ thể của từng mô hình pretrained dùng từ HuggingFace (ví dụ tên checkpoint chính xác như "microsoft/wavlm-base", "vinai/phowhisper-base", "vinai/phobert-v2"), để có thể tái lập lại kết quả sau này.
