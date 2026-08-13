Kiến trúc Hệ thống: Multimodal Hybrid Search & Temporal Alignment in Video Retrieval
Mục lục
Tổng quan kiến trúc & Tóm tắt mục tiêu
Sơ đồ luồng dữ liệu tổng thể
Phase 1 – Data Pipeline Module
Phase 2 – Metadata Module
Phase 3 – Indexing Module
Phase 4 – Retrieval & Alignment Module
Phase 5 – QA & UI Module
Luồng dữ liệu minh họa (Data Flow Demo)
Giao diện tích hợp giữa các Module (API Contracts)
Thiết kế phi chặn (Non-blocking Integration)
Quyết định thiết kế & Trade-offs
1. Tổng quan kiến trúc & Tóm tắt mục tiêu
Hệ thống được thiết kế theo mô hình Multimodal Hybrid Pipeline, cho phép xử lý luồng dữ liệu đa phương thức (video, audio, text) một cách tuần tự nhưng có khả năng tìm kiếm kết hợp và suy luận chuỗi thời gian khắt khe.
Tóm tắt các mục tiêu cần làm:
Tiền xử lý thông minh: Loại bỏ khung hình rác, chỉ giữ lại các khung hình có giá trị ngữ nghĩa thực sự thông qua mạng nơ-ron và đo lường khoảng cách vector.
Khai phá đa phương thức: Chuyển hóa toàn bộ pixel hình ảnh và luồng âm thanh thành vector toán học (dense embeddings) và chuỗi văn bản (OCR, ASR).
Lập chỉ mục siêu tốc: Xây dựng cơ sở dữ liệu kép để phục vụ tìm kiếm ngữ nghĩa (FAISS/Milvus) và tìm kiếm từ khóa chính xác (Elasticsearch).
Truy xuất & Căn chỉnh thời gian: Áp dụng quy hoạch động (DANTE) để giải quyết loại câu hỏi chuỗi sự kiện TRAKE, bắt buộc tính toán trật tự thời gian.
Tương tác & Đệ trình tự động: Triển khai vòng lặp phản hồi Rocchio để tối ưu hóa truy vấn và tự động xuất file .csv chuẩn CodaBench.
2. Sơ đồ luồng dữ liệu tổng thể
[Raw Video Data]
        │
        ▼
┌───────────────────┐
│ Phase 1: Pipeline │  (Member 1 – Data Architect)
│ TransNetV2 +      │
│ Adaptive Sampling │
└────────┬──────────┘
         │  List[KeyframeData]
         ▼
┌───────────────────┐
│ Phase 2: Metadata │  (Member 2 – AI & Metadata Eng)
│ SigLIP/BEiT-3     │
│ Qwen2.5-VL, Whisper
└────────┬──────────┘
         │  List[KeyframeData + Embeddings + Text]
         ▼
┌───────────────────┐
│ Phase 3: Indexing │  (Member 3 – Backend & DB Admin)
│ Milvus (Dense)    │
│ ES (Sparse/Text)  │
└────────┬──────────┘
         │  Indexed Database (Linked by keyframe_id)
         ▼
┌───────────────────┐
│ Phase 4: Retrieval│  (Member 4 – AI Retrieval Sci)
│ WRRF Fusion       │
│ DANTE DP Algorithm│
└────────┬──────────┘
         │  List[RetrievalResult] (Top-100 Ranked)
         ▼
┌───────────────────┐
│ Phase 5: QA & UI  │  (Member 5 – Query Ops & QA)
│ Rocchio Feedback  │
│ ZIP Formatter     │
└────────┬──────────┘
         │
         ▼
 [Final submission.zip]

3. Phase 1 – Data Pipeline Module (Tiền xử lý)
Phụ trách: Member 1 – Data Pipeline Architect
3.1 Mục tiêu
Cắt nhỏ video thô thành các cảnh quay (shots) có tính liên kết và trích xuất khung hình đại diện nhằm giảm tải tính toán dư thừa nhưng vẫn không làm sót sự kiện.
3.2 Luồng xử lý
Shot Detection: Sử dụng mạng TransNetV2 để phát hiện các ranh giới cắt cảnh (hard cuts) và chuyển cảnh mượt (dissolves).
Adaptive Filtering: Thay vì lấy khung hình cố định theo giây, hệ thống tính toán khoảng cách chuẩn $L_2$ giữa khung hình hiện tại $e_{current}$ và khung hình trước đó $e_{prev}$. Chỉ giữ lại khung hình nếu $\frac{\Vert{}e_{current} - e_{prev}\Vert{}_2}{\Vert{}e_{prev}\Vert{}_2} > 0.4$.
4. Phase 2 – Metadata Module (Khai phá siêu dữ liệu)
Phụ trách: Member 2 – AI & Metadata Engineer
4.1 Mục tiêu
Nhận các khung hình từ Phase 1 và biến chúng thành các đại diện toán học và văn bản có thể tìm kiếm được.
4.2 Luồng xử lý
Visual Embeddings: Dùng SigLIP2 và BEiT-3 mã hóa hình ảnh thành vector đa chiều. Khác với CLIP dùng softmax, SigLIP dùng hàm suy hao sigmoid (sigmoid loss) cho không gian biểu diễn hội tụ tốt hơn.
Text & Speech (OCR/ASR): Gọi Qwen2.5-VL bóc tách chữ trên video (bảng hiệu, thời sự); dùng Whisper bóc tách lời thoại thành văn bản có gắn nhãn thời gian (timestamped speech-to-text).
5. Phase 3 – Indexing Module (Lập chỉ mục)
Phụ trách: Member 3 – Backend & DB Administrator
5.1 Mục tiêu
Lưu trữ toàn bộ dữ liệu từ Phase 2 với kiến trúc tối ưu cho độ trễ truy vấn siêu thấp, liên kết chặt chẽ qua khóa chính keyframe_id.
5.2 Luồng xử lý
Milvus (Dense Vectors): Nạp vector SigLIP/BEiT-3. Sử dụng đồ thị HNSW để tìm kiếm lân cận gần nhất (ANN) siêu tốc.
Elasticsearch (Sparse/Text): Nạp dữ liệu OCR/ASR. Chạy thuật toán BM25 kết hợp mô hình nhúng thưa để đối sánh từ vựng chính xác (exact lexical matching).
6. Phase 4 – Retrieval & Alignment Module (Truy xuất)
Phụ trách: Member 4 – AI & Retrieval Scientist
6.1 Mục tiêu
Xử lý truy vấn của người dùng, phân tách luồng tìm kiếm và giải quyết bài toán phức tạp nhất: chuỗi thời gian TRAKE.
6.2 Luồng xử lý
Hybrid Search: Trộn điểm số từ Milvus và Elasticsearch bằng thuật toán Weighted Reciprocal Rank Fusion (WRRF): $\text{WRRF}(q, d) = \frac{\alpha_d}{r_{text}(q, d) + k} + \frac{1 - \alpha_d}{r_{vision}(q, d) + k}$.
DANTE Algorithm (Cho TRAKE): Sử dụng quy hoạch động (Dynamic Programming). Tính ma trận tương đồng $S[i, t] = \text{cosine\_similarity}(u_i, E[t])$, sau đó rà soát bảng quy hoạch để tìm chuỗi $N$ khung hình đúng trình tự vật lý, có áp dụng hệ số phạt thời gian $\lambda$ để tránh các khung hình quá xa nhau.
7. Phase 5 – QA & UI Module (Tương tác & Đóng gói)
Phụ trách: Member 5 – Query Ops & QA Manager
7.1 Mục tiêu
Quản lý tương tác con người, tinh chỉnh truy vấn và xuất file đáp ứng format khắt khe của CodaBench.
7.2 Luồng xử lý
Relevance Feedback (Rocchio): Người dùng nhãn Đúng/Sai trực quan. Thuật toán Rocchio sẽ tự động dịch chuyển vector truy vấn $q_m$ về gần với các kết quả đúng ($C_r$) và đẩy ra xa bối cảnh sai ($C_{nr}$).
Format Validator: Code tự động xóa đuôi .mp4 khỏi ID, bọc ngoặc kép cho Q&A, và gói tệp CSV vào thư mục trung gian submission trước khi zip.
8. Luồng dữ liệu minh họa (Data Flow Demo)
8.1. Truy vấn Textual KIS (Khung hình đơn lẻ)
Input mẫu: "Một người đàn ông mặc áo khoác đỏ đang dắt chó đi dạo trong công viên."
Bước 1 – Phase 4 xử lý truy vấn: Hệ thống mã hóa nguyên câu thành vector ngữ nghĩa và tập hợp từ khóa.
Bước 2 – Hybrid Search (Milvus + ES): Trả về hàng ngàn khung hình ứng viên chứa người, chó, công viên.
Bước 3 – Thuật toán WRRF: Quét danh sách, hợp nhất điểm. Tìm được Frame 3450 của video L01_V015 có điểm cao nhất.
Bước 4 – JSON/CSV Assembly (Phase 5):
JSON
[
  {
    "video_id": "L01_V015",
    "frame_ids": [3450],
    "query_type": "Textual_KIS",
    "rank": 1
  }
]

8.2. Truy vấn Q&A (Hỏi đáp thị giác)
Input mẫu: "Tìm cảnh một chiếc xe cứu thương đang đỗ trước cổng bệnh viện. Biển số của chiếc xe cứu thương đó là gì?"
Bước 1 – Phase 4 xử lý truy vấn: Hệ thống tách thành $q_{search}$: "cảnh xe cứu thương đỗ..." và $q_{reasoning}$: "Biển số xe là gì?".
Bước 2 – Hybrid Search (Milvus + ES): Dùng $q_{search}$ tra cứu. WRRF trộn điểm và chốt được Frame 5120 của video L02_V030.
Bước 3 – Đọc hiểu hình ảnh (Phase 2): Đẩy ảnh Frame 5120 và $q_{reasoning}$ vào MLLM (Qwen2.5-VL). Mô hình đọc OCR và xuất kết quả "59A-123.45".
Bước 4 – JSON/CSV Assembly (Phase 5):
JSON
[
  {
    "video_id": "L02_V030",
    "frame_ids": [5120],
    "query_type": "Q&A",
    "answer": "59A-123.45",
    "rank": 1
  }
]

8.3. Truy vấn TRAKE (Chuỗi sự kiện)
Input mẫu: "Người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm."
Bước 1 – Phase 4 xử lý truy vấn: Hệ thống tự động tách thành $N=3$ sự kiện:
$u_1$: "chạy đà"
$u_2$: "giậm nhảy qua xà"
$u_3$: "tiếp đất trên nệm"
Bước 2 – Hybrid Search (Milvus + ES): Trả về hàng ngàn khung hình ứng viên từ các video khác nhau cho mỗi sự kiện $u_1, u_2, u_3$.
Bước 3 – Thuật toán DANTE: Quét qua video L01_V025. Tìm được:
Frame 1050 (điểm cao cho $u_1$)
Frame 1080 (điểm cao cho $u_2$)
Frame 1120 (điểm cao cho $u_3$) Thuật toán xác nhận thứ tự $1050 < 1080 < 1120$ hợp lệ về thời gian (không đứt gãy sự kiện).
Bước 4 – JSON/CSV Assembly (Phase 5):
JSON
[
  {
    "video_id": "L01_V025",
    "frame_ids": [1050, 1080, 1120],
    "query_type": "TRAKE",
    "rank": 1
  }
]

9. Giao diện tích hợp giữa các Module (API Contracts)
Để 5 thành viên có thể code độc lập, các TypedDict/Dataclass được định nghĩa nghiêm ngặt.
Python
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class KeyframeData:
    """Contract Output Phase 1 -> Input Phase 2"""
    keyframe_id: str          # Format: {video_id}_{frame_idx}
    video_id: str
    frame_idx: int
    timestamp_sec: float
    image_matrix: object      # Numpy array của ảnh

@dataclass
class MultimodalMetadata:
    """Contract Output Phase 2 -> Input Phase 3"""
    keyframe_id: str
    dense_vector: List[float] # Từ SigLIP/BEiT-3
    ocr_text: str             # Từ Qwen2.5-VL
    asr_transcript: str       # Từ Whisper

@dataclass
class RetrievalResult:
    """Contract Output Phase 4 -> Input Phase 5"""
    video_id: str
    frame_ids: List[int]      # Sẽ có N phần tử nếu là TRAKE, 1 phần tử nếu KIS
    answer: Optional[str]     # Chỉ dùng cho Q&A
    wrrf_score: float

10. Thiết kế phi chặn (Non-blocking Integration)
Nhóm áp dụng chiến lược tạo Mock Interfaces (Dữ liệu giả lập) ngay từ ngày đầu tiên để không ai phải đợi ai.
Ví dụ Mock Database (cho Member 4 & 5 code trước khi Member 2 & 3 làm xong):
Python
# mock_search_engine.py 
def mock_milvus_search(query_vector: List[float]) -> List[RetrievalResult]:
    return [
        RetrievalResult(video_id="L01_V001", frame_ids=[1500], answer=None, wrrf_score=0.95),
        RetrievalResult(video_id="L01_V002", frame_ids=[2000], answer=None, wrrf_score=0.82)
    ]

Tuần 1: Thống nhất API Contracts (như Mục 9), lập trình các hàm Mock.
Tuần 2: Member 1 train pipeline; Member 2 generate vector; Member 4 viết logic toán học DANTE.
Tuần 3: Thay Mock bằng dữ liệu thật, test luồng End-to-End.
11. Quyết định thiết kế & Trade-offs
Quyết định
Phương án được chọn
Lý do (Trade-offs)
Cắt khung hình
TransNetV2 + DCNN cells
Mặc dù tính toán nặng hơn việc lấy mẫu tĩnh mỗi 1 giây, nhưng đảm bảo không mất mát các sự kiện lướt qua nhanh.
Backbone Vision
SigLIP2 & BEiT-3
SigLIP giải quyết bài toán hàm suy hao softmax của CLIP bị thắt nút cổ chai (bottleneck), giúp vector hội tụ tốt hơn.
Database
Tách Milvus & Elasticsearch
ElasticSearch tìm từ khóa BM25 cực mạnh nhưng yếu toán học vector; Milvus xử lý HNSW siêu tốc. Việc tách biệt gây khó trong đồng bộ hóa keyframe_id, nhưng tối ưu hóa tốc độ độ trễ truy vấn.
Temporal Alignment
DANTE (Quy hoạch động)
Các phương pháp sinh tổ hợp (Brute-force) sẽ sập hệ thống khi $N$ lớn. DP đảm bảo độ phức tạp đa thức và thời gian thực thi có thể kiểm soát.


