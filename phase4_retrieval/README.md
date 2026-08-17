# Phase 4: Module Truy xuất & Căn chỉnh Chuỗi sự kiện (Retrieval & Alignment)

**Phụ trách:** Member 4 – AI & Retrieval Scientist

Module Phase 4 xử lý truy vấn của người dùng, phân tách luồng tìm kiếm và giải quyết bài toán phức tạp nhất: chuỗi thời gian TRAKE. Kết quả đầu ra là danh sách `RetrievalResult` (Top-K đã xếp hạng) để chuyển sang Phase 5.

## 1. Trách nhiệm cốt lõi

- **Hybrid Search:** Trộn điểm số từ Milvus (dense vector ANN) và Elasticsearch (BM25 lexical) bằng thuật toán **Weighted Reciprocal Rank Fusion (WRRF)** theo công thức:

$$\text{WRRF}(q, d) = \frac{\alpha_d}{r_{text}(q, d) + k} + \frac{1 - \alpha_d}{r_{vision}(q, d) + k}$$

- **DANTE Algorithm (cho TRAKE):** Sử dụng quy hoạch động (Dynamic Programming). Tính ma trận tương đồng $S[i, t] = \text{cosine\_similarity}(u_i, E[t])$ giữa sự kiện $i$ và vector khung hình $t$, sau đó rà soát bảng quy hoạch để tìm chuỗi $N$ khung hình đúng trình tự thời gian, có áp dụng hệ số phạt thời gian $\lambda$ để tránh các khung hình quá xa nhau.

- **Q&A Reasoning:** Tách truy vấn thành $q_{search}$ (phần tìm kiếm) và $q_{reasoning}$ (phần suy luận). Dùng $q_{search}$ để tra cứu, sau đó đẩy ảnh kết quả + $q_{reasoning}$ vào MLLM (Qwen2.5-VL) để đọc hiểu và trả lời. *(Hiện tại đang dùng placeholder, chờ tích hợp Qwen2.5-VL thật từ Phase 2/5.)*

## 2. Cấu trúc thư mục (Deliverables)

```text
phase4_retrieval/
├── search.py                 # Core retriever interface (entry point chính)
├── hybrid_search_wrrf.py     # Thuật toán lai WRRF (trộn rank Milvus & ES)
├── dante_trake_solver.py     # Thuật toán DANTE (quy hoạch động căn chỉnh sự kiện)
├── event_segmenter.py        # (Bản nháp PhoBERT - hiện không sử dụng)
├── mock_database.py          # Giả lập Phase 3 (Milvus/ES) cho môi trường dev/test
├── __init__.py               # Export public API: search, calculate_wrrf, solve_dante
├── README.md                 # Tài liệu hướng dẫn (file này)
└── tests/
    └── test_retrieval.py     # Bộ Unit & Integration test (28 test cases)
```

## 3. Data Contract (Đầu vào / Đầu ra)

### Đầu vào
Phase 4 nhận kết quả tra cứu thô từ Phase 3 (Milvus + Elasticsearch), được liên kết qua khóa chính `keyframe_id`.

### Đầu ra → Phase 5
Tuân thủ chuẩn `RetrievalResult` định nghĩa trong `shared_contracts/contracts.py`:

```python
@dataclass
class RetrievalResult:
    """Contract Output Phase 4 -> Input Phase 5"""
    video_id: str
    frame_ids: List[int]       # N phần tử nếu là TRAKE, 1 phần tử nếu KIS/Q&A
    query_type: str            # "KIS", "Q&A", hoặc "TRAKE"
    answer: Optional[str]      # Chỉ dùng cho Q&A
    wrrf_score: float
```

### Ví dụ output cho từng loại truy vấn

**KIS:**
```json
{"video_id": "L01_V015", "frame_ids": [3450], "query_type": "KIS", "answer": null, "wrrf_score": 0.942}
```

**Q&A:**
```json
{"video_id": "L02_V030", "frame_ids": [5120], "query_type": "Q&A", "answer": "59A-123.45", "wrrf_score": 0.891}
```

**TRAKE:**
```json
{"video_id": "L01_V025", "frame_ids": [1050, 1080, 1120], "query_type": "TRAKE", "answer": null, "wrrf_score": 0.885}
```

## 4. Yêu cầu hệ thống & Cài đặt

1. Đảm bảo bạn đang ở thư mục gốc của dự án (`PythonProject`).
2. Cài đặt thư viện cần thiết:
   ```bash
   pip install google-genai
   ```
3. **Cấu hình API Key (BẮT BUỘC):**
   Hệ thống sử dụng Gemini API (`google-genai`, model `gemini-3.7-flash`) để phân tách sự kiện cho truy vấn TRAKE.
   Hãy liên hệ với **Member 4 (AI & Retrieval Scientist)** để lấy `GEMINI_API_KEY`. Sau đó cấu hình biến môi trường:

   **Windows (PowerShell):**
   ```powershell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
   **Linux/macOS:**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

4. Bạn chưa cần cài `torch` hay các mô hình ngôn ngữ lớn (BERT) tại môi trường local. Hệ thống đang sử dụng `APIEventSegmenter` (gọi Gemini API) được nhúng trực tiếp trong `search.py`.

## 5. Hướng dẫn sử dụng & Kiểm thử

### Chạy CLI Entrypoint
```bash
# KIS (Known-Item Search)
python -m phase4_retrieval.search --query "người đàn ông mặc áo đỏ" --query_type KIS --top_k 5

# Q&A (Question & Answering)
python -m phase4_retrieval.search --query "xe cứu thương đỗ trước bệnh viện. Biển số xe là gì?" --query_type "Q&A" --top_k 3

# TRAKE (Chuỗi sự kiện thời gian)
python -m phase4_retrieval.search --query "người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm" --query_type TRAKE --top_k 5
```

### Chạy Unit / Integration Tests
```bash
python -m pytest phase4_retrieval/tests/test_retrieval.py -v
```

## 6. Hướng dẫn Tích hợp

### 6.1. Tích hợp với Phase 3 (Indexing Module)
Hiện tại, `search.py` đang import từ `mock_database.py` (giả lập).
Khi Phase 3 hoàn thiện, sửa lại import trong `phase4_retrieval/search.py`:

```python
# Thay vì import mock:
# from .mock_database import search_milvus_db, search_elastic_db

# Import module thật từ Phase 3:
from phase3_indexing.milvus_indexer import search_milvus_db
from phase3_indexing.elastic_indexer import search_elastic_db
```

### 6.2. Tích hợp với Phase 5 (QA & UI Module)
Phase 5 import hàm `search` và truyền vào dictionary truy vấn:

```python
from phase4_retrieval import search

payload = {
    "query": "người đàn ông đi xe đạp",
    "query_type": "KIS",
    "top_k": 100
}

# results là List[Dict] tuân thủ chuẩn RetrievalResult
results = search(payload)

for r in results:
    print(f"Video: {r['video_id']}, Frames: {r['frame_ids']}, Score: {r['wrrf_score']}")
```

### 6.3. Tích hợp Q&A Reasoning (Phase 2 / Phase 5)
Với `query_type = "Q&A"`, hàm `search` tự động tách truy vấn thành `q_search` và `q_reasoning`.
Hiện tại câu trả lời đang là placeholder (`"MOCK_ANSWER"`).
Để kích hoạt reasoning thật, nhóm Phase 2/5 cần tích hợp MLLM (Qwen2.5-VL) vào luồng Q&A trong `search.py` — đẩy ảnh frame kết quả + `q_reasoning` vào model để sinh câu trả lời.
