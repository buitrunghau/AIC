# Multimodal Hybrid Search & Temporal Alignment in Video Retrieval

Hệ thống truy xuất video đa phương thức, kết hợp tìm kiếm ngữ nghĩa (dense embeddings), tìm kiếm từ khóa (BM25) và căn chỉnh chuỗi sự kiện theo thời gian (temporal alignment).

Hỗ trợ 3 loại truy vấn chính:

| Loại truy vấn | Mô tả |
|----------------|-------|
| **Textual KIS** | Tìm khung hình đơn lẻ theo mô tả văn bản |
| **Q&A**        | Tìm khung hình chứa câu trả lời thị giác (OCR) |
| **TRAKE**      | Tìm chuỗi khung hình đúng trình tự sự kiện thời gian |

---

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Chi tiết các Phase](#chi-tiết-các-phase)
- [Luồng dữ liệu minh họa](#luồng-dữ-liệu-minh-họa)
- [Giao diện tích hợp (API Contracts)](#giao-diện-tích-hợp-api-contracts)
- [Thiết kế phi chặn (Non-blocking Integration)](#thiết-kế-phi-chặn-non-blocking-integration)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [API Endpoints](#api-endpoints)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Quyết định thiết kế & Trade-offs](#quyết-định-thiết-kế--trade-offs)
- [Đóng góp](#đóng-góp)
- [License](#license)

---

## Tính năng chính

- **Tiền xử lý video thông minh**: Phát hiện shot bằng TransNetV2, lọc khung hình thích ứng dựa trên khoảng cách vector L2, loại bỏ khung hình rác.
- **Trích xuất đa phương thức**:
  - Visual embeddings: SigLIP2, BEiT-3.
  - OCR (Visual Text): Qwen2.5-VL.
- **Lập chỉ mục kép**:
  - Milvus (dense vectors, HNSW) cho tìm kiếm ngữ nghĩa.
  - Elasticsearch (sparse/text, BM25) cho tìm kiếm từ khóa OCR chính xác.
- **Tìm kiếm hybrid**: Kết hợp điểm số bằng Weighted Reciprocal Rank Fusion (WRRF).
- **Căn chỉnh thời gian**: Thuật toán DANTE (Dynamic Programming) cho truy vấn TRAKE.
- **Tương tác phản hồi**: Relevance feedback Rocchio để tinh chỉnh truy vấn.
- **Đóng gói kết quả**: Tự động tạo file CSV chuẩn CodaBench và nén thành `submission.zip`.

---

## Kiến trúc hệ thống

```text
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
│ SigLIP2/BEiT-3    │
│ Qwen2.5-VL (OCR)  │
└────────┬──────────┘
         │  List[KeyframeData + Embeddings + OCR]
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
```

---

## Chi tiết các Phase

### Phase 1 – Data Pipeline Module

**Phụ trách:** Member 1 – Data Pipeline Architect

**Mục tiêu:** Cắt nhỏ video thô thành các cảnh quay (shots) có tính liên kết và trích xuất khung hình đại diện nhằm giảm tải tính toán dư thừa nhưng vẫn không làm sót sự kiện.

**Luồng xử lý:**

- **Shot Detection**: Sử dụng mạng TransNetV2 để phát hiện các ranh giới cắt cảnh (hard cuts) và chuyển cảnh mượt (dissolves).
- **Adaptive Filtering**: Thay vì lấy khung hình cố định theo giây, hệ thống tính toán khoảng cách chuẩn L2 giữa khung hình hiện tại `e_current` và khung hình trước đó `e_prev`. Chỉ giữ lại khung hình nếu:

```text
||e_current - e_prev||_2 / ||e_prev||_2 > 0.4
```

### Phase 2 – Metadata Module

**Phụ trách:** Member 2 – AI & Metadata Engineer

**Mục tiêu:** Nhận các khung hình từ Phase 1 và biến chúng thành các đại diện toán học và văn bản có thể tìm kiếm được.

**Luồng xử lý:**

- **Visual Embeddings**: Dùng SigLIP2 và BEiT-3 mã hóa hình ảnh thành vector đa chiều. Khác với CLIP dùng softmax, SigLIP dùng hàm suy hao sigmoid (sigmoid loss) cho không gian biểu diễn hội tụ tốt hơn.
- **Visual Text (OCR)**: Gọi Qwen2.5-VL bóc tách chữ xuất hiện trên video (bảng hiệu, dòng chữ thời sự, bảng tên,...).

### Phase 3 – Indexing Module

**Phụ trách:** Member 3 – Backend & DB Administrator

**Mục tiêu:** Lưu trữ toàn bộ dữ liệu từ Phase 2 với kiến trúc tối ưu cho độ trễ truy vấn siêu thấp, liên kết chặt chẽ qua khóa chính `keyframe_id`.

**Luồng xử lý:**

- **Milvus (Dense Vectors)**: Nạp vector SigLIP2/BEiT-3. Sử dụng đồ thị HNSW để tìm kiếm lân cận gần nhất (ANN) siêu tốc.
- **Elasticsearch (Sparse/Text)**: Nạp dữ liệu văn bản OCR. Chạy thuật toán BM25 kết hợp mô hình nhúng thưa để đối sánh từ vựng chính xác (exact lexical matching).

### Phase 4 – Retrieval & Alignment Module

**Phụ trách:** Member 4 – AI & Retrieval Scientist

**Mục tiêu:** Xử lý truy vấn của người dùng, phân tách luồng tìm kiếm và giải quyết bài toán phức tạp nhất: chuỗi thời gian TRAKE.

**Luồng xử lý:**

- **Hybrid Search**: Trộn điểm số từ Milvus và Elasticsearch bằng thuật toán Weighted Reciprocal Rank Fusion (WRRF):

```text
WRRF(q, d) = α_d / (r_text(q, d) + k) + (1 - α_d) / (r_vision(q, d) + k)
```

- **DANTE Algorithm (Cho TRAKE)**: Sử dụng quy hoạch động (Dynamic Programming). Tính ma trận tương đồng `S[i, t] = cosine_similarity(u_i, E[t])`, sau đó rà soát bảng quy hoạch để tìm chuỗi N khung hình đúng trình tự vật lý, có áp dụng hệ số phạt thời gian λ để tránh các khung hình quá xa nhau.

### Phase 5 – QA & UI Module

**Phụ trách:** Member 5 – Query Ops & QA Manager

**Mục tiêu:** Quản lý tương tác con người, tinh chỉnh truy vấn và xuất file đáp ứng format khắt khe của CodaBench.

**Luồng xử lý:**

- **Relevance Feedback (Rocchio)**: Người dùng gắn nhãn Đúng/Sai trực quan. Thuật toán Rocchio sẽ tự động dịch chuyển vector truy vấn `q_m` về gần với các kết quả đúng (`C_r`) và đẩy ra xa bối cảnh sai (`C_nr`).
- **Format Validator**: Code tự động kiểm tra loại bỏ đuôi `.mp4` khỏi ID, bọc ngoặc kép cho Q&A, và gói tệp CSV vào thư mục trung gian `submission` trước khi zip.

---

## Luồng dữ liệu minh họa

### 8.1. Truy vấn Textual KIS (Khung hình đơn lẻ)

**Input mẫu:** `"Một người đàn ông mặc áo khoác đỏ đang dắt chó đi dạo trong công viên."`

- **Bước 1 – Phase 4 xử lý truy vấn:** Hệ thống mã hóa nguyên câu thành vector ngữ nghĩa và tập hợp từ khóa.
- **Bước 2 – Hybrid Search (Milvus + ES):** Trả về hàng ngàn khung hình ứng viên chứa người, chó, công viên.
- **Bước 3 – Thuật toán WRRF:** Quét danh sách, hợp nhất điểm. Tìm được `Frame 3450` của video `L01_V015` có điểm cao nhất.
- **Bước 4 – JSON/CSV Assembly (Phase 5):**

```json
[
  {
    "video_id": "L01_V015",
    "frame_ids": [3450],
    "query_type": "Textual_KIS",
    "rank": 1
  }
]
```

### 8.2. Truy vấn Q&A (Hỏi đáp thị giác)

**Input mẫu:** `"Tìm cảnh một chiếc xe cứu thương đang đỗ trước cổng bệnh viện. Biển số của chiếc xe cứu thương đó là gì?"`

- **Bước 1 – Phase 4 xử lý truy vấn:** Hệ thống tách thành `q_search`: `"cảnh xe cứu thương đỗ..."` và `q_reasoning`: `"Biển số xe là gì?"`.
- **Bước 2 – Hybrid Search:** Dùng `q_search` tra cứu. WRRF trộn điểm và chốt được `Frame 5120` của video `L02_V030`.
- **Bước 3 – Đọc hiểu hình ảnh (Phase 2):** Đẩy ảnh `Frame 5120` và `q_reasoning` vào MLLM (Qwen2.5-VL). Mô hình đọc OCR và xuất kết quả `"59A-123.45"`.
- **Bước 4 – JSON/CSV Assembly (Phase 5):**

```json
[
  {
    "video_id": "L02_V030",
    "frame_ids": [5120],
    "query_type": "Q&A",
    "answer": "59A-123.45",
    "rank": 1
  }
]
```

### 8.3. Truy vấn TRAKE (Chuỗi sự kiện)

**Input mẫu:** `"Người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm."`

- **Bước 1 – Phase 4 xử lý truy vấn:** Hệ thống tự động tách thành `N = 3` sự kiện:
  - `u1`: `"chạy đà"`
  - `u2`: `"giậm nhảy qua xà"`
  - `u3`: `"tiếp đất trên nệm"`
- **Bước 2 – Hybrid Search:** Trả về hàng ngàn khung hình ứng viên từ các video khác nhau cho mỗi sự kiện `u1, u2, u3`.
- **Bước 3 – Thuật toán DANTE:** Quét qua video `L01_V025`. Tìm được:
  - `Frame 1050` (điểm cao cho `u1`)
  - `Frame 1080` (điểm cao cho `u2`)
  - `Frame 1120` (điểm cao cho `u3`)

  Thuật toán xác nhận thứ tự `1050 < 1080 < 1120` hợp lệ về thời gian (không đứt gãy sự kiện).

- **Bước 4 – JSON/CSV Assembly (Phase 5):**

```json
[
  {
    "video_id": "L01_V025",
    "frame_ids": [1050, 1080, 1120],
    "query_type": "TRAKE",
    "rank": 1
  }
]
```

---

## Giao diện tích hợp (API Contracts)

Để 5 thành viên có thể code độc lập, các `dataclass` được định nghĩa nghiêm ngặt trong `shared_contracts/contracts.py`:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class KeyframeData:
    """Contract Output Phase 1 -> Input Phase 2"""
    keyframe_id: str          # Format: {video_id}_{frame_idx}
    video_id: str
    frame_idx: int
    timestamp_sec: float
    image_matrix: Any         # Numpy array hoac PIL Image

@dataclass
class MultimodalMetadata:
    """Contract Output Phase 2 -> Input Phase 3"""
    keyframe_id: str
    dense_vector: List[float] # Tu SigLIP2/BEiT-3
    ocr_text: str             # Tu Qwen2.5-VL

@dataclass
class RetrievalResult:
    """Contract Output Phase 4 -> Input Phase 5"""
    video_id: str
    frame_ids: List[int]      # Sẽ có N phần tử nếu là TRAKE, 1 phần tử nếu KIS/Q&A
    answer: Optional[str]     # Chỉ dùng cho Q&A
    wrrf_score: float
```

---

## Thiết kế phi chặn (Non-blocking Integration)

Nhóm áp dụng chiến lược tạo **Mock Interfaces** (Dữ liệu giả lập) ngay từ ngày đầu tiên để không ai phải đợi ai.

---

## Yêu cầu hệ thống

- **Python**: 3.10+
- **GPU** (khuyến nghị): CUDA 11.8+ cho các model deep learning.
- **Docker & Docker Compose**: chạy Milvus và Elasticsearch.
- **RAM**: tối thiểu 16GB (32GB khuyến nghị).
- **Ổ cứng**: phụ thuộc kích thước dataset video và vector embeddings.

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/nhuutri2000-max/AIC.git
cd AIC
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

---

## Cấu hình

Tạo file `.env` từ mẫu:

```bash
cp .env.example .env
```

Các biến môi trường quan trọng:

```env
# Đường dẫn dữ liệu
RAW_VIDEO_PATH=./data/raw_videos
KEYFRAME_PATH=./data/keyframes
METADATA_PATH=./data/metadata

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=video_frames

# Elasticsearch
ES_HOST=localhost
ES_PORT=9200
ES_INDEX=video_text

# Model
VISION_MODEL=siglip2
TEXT_MODEL=beit3
OCR_MODEL=qwen2.5-vl

# Retrieval
WRRF_ALPHA=0.6
WRRF_K=60
DANTE_LAMBDA=0.1
TOP_K=100
```

---

## Hướng dẫn sử dụng

### Chạy toàn bộ pipeline

#### Phase 1: Data Pipeline

```bash
python -m phase1_pipeline.extract_keyframes \
    --video_dir $RAW_VIDEO_PATH \
    --output_dir $KEYFRAME_PATH \
    --threshold 0.4
```

#### Phase 2: Metadata Module

```bash
python -m phase2_metadata.generate_metadata \
    --keyframe_dir $KEYFRAME_PATH \
    --output_dir $METADATA_PATH \
    --vision_model $VISION_MODEL \
    --ocr_model $OCR_MODEL
```

#### Phase 3: Indexing Module

```bash
python -m phase3_indexing.index_data \
    --metadata_path $METADATA_PATH \
    --milvus_host $MILVUS_HOST \
    --milvus_port $MILVUS_PORT \
    --es_host $ES_HOST \
    --es_port $ES_PORT
```

#### Phase 4: Retrieval & Alignment

Chạy CLI:

```bash
# Truy vấn KIS
python -m phase4_retrieval.search \
    --query "Một người đàn ông mặc áo khoác đỏ đang dắt chó đi dạo trong công viên." \
    --query_type KIS \
    --top_k 10

# Truy vấn TRAKE
python -m phase4_retrieval.search \
    --query "Người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm." \
    --query_type TRAKE \
    --top_k 5
```

#### Phase 5: QA UI & Đóng gói kết quả

Chạy giao diện Web:

```bash
streamlit run phase5_qa_ui/streamlit_app.py
```

Xuất file submission:

```bash
python -m phase5_qa_ui.export_submission \
    --results_path ./output/results.json \
    --output_dir ./submission
```

Tự động tạo file `submission.zip` chuẩn CodaBench.

---

## Cấu trúc thư mục

```text
.
├── phase1_pipeline/           # Phase 1: TransNetV2 & Adaptive Sampler
│   ├── transnet_segmentation.py
│   ├── adaptive_sampler.py
│   ├── extract_keyframes.py
│   └── task.md
├── phase2_metadata/           # Phase 2: Visual Embeddings & OCR
│   ├── siglip_beit_embedder.py
│   ├── qwen_ocr_extractor.py
│   ├── generate_metadata.py
│   └── task.md
├── phase3_indexing/           # Phase 3: Milvus & Elasticsearch Indexing
│   ├── milvus_indexer.py
│   ├── elastic_indexer.py
│   ├── index_data.py
│   └── task.md
├── phase4_retrieval/          # Phase 4: Hybrid Search (WRRF) & DANTE (TRAKE)
│   ├── hybrid_search_wrrf.py
│   ├── dante_trake_solver.py
│   ├── search.py
│   └── task.md
├── phase5_qa_ui/              # Phase 5: Rocchio Feedback, UI & Submission Exporter
│   ├── rocchio_feedback.py
│   ├── csv_formatter.py
│   ├── export_submission.py
│   ├── streamlit_app.py
│   └── task.md
├── shared_contracts/          # Contract DataClasses dùng chung giữa 5 Phase
│   └── contracts.py
├── tests/                     # Test cases cho 5 Phase
├── docker-compose.yml
├── requirements.txt
├── architecture_overview.md
└── README.md
```

---

## Quyết định thiết kế & Trade-offs

| Quyết định         | Phương án được chọn          | Lý do (Trade-offs)                                                                                                   |
|--------------------|------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Cắt khung hình     | TransNetV2 + DCNN cells      | Tính toán nặng hơn lấy mẫu tĩnh mỗi giây, nhưng đảm bảo không mất sự kiện lướt qua nhanh.                            |
| Backbone Vision    | SigLIP2 & BEiT-3             | SigLIP giải quyết bottleneck của softmax trong CLIP, vector hội tụ tốt hơn.                                          |
| Database           | Tách Milvus & Elasticsearch  | Elasticsearch mạnh BM25, Milvus mạnh HNSW. Khó đồng bộ `keyframe_id` nhưng tối ưu độ trễ.                          |
| Temporal Alignment | DANTE (Quy hoạch động)       | Brute-force tổ hợp sẽ sập khi N lớn. DP đảm bảo độ phức tạp đa thức, thời gian thực thi kiểm soát được.             |

---

## Đóng góp

1. Fork repository.
2. Tạo branch tính năng: `git checkout -b feature/ten-tinh-nang`.
3. Commit thay đổi.
4. Push branch: `git push origin feature/ten-tinh-nang`.
5. Tạo Pull Request.

---

## License

Dự án được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.
