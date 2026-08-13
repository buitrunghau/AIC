# Nhiệm vụ: Xây dựng Module Truy xuất & Căn chỉnh Chuỗi sự kiện (Phase 4)

**Người phụ trách:** Member 4 – AI & Retrieval Scientist

**Vai trò:** Xử lý câu truy vấn của người dùng, thực hiện thuật toán trộn điểm tìm kiếm không gian, và quan trọng nhất là lập trình thuật toán giải bài toán chuỗi thời gian (TRAKE).

## 1. Trách nhiệm cốt lõi (Core Responsibility)

- Điều phối hai luồng tìm kiếm song song: tìm kiếm theo độ tương đồng vector (trên Milvus) và tìm kiếm theo từ vựng (trên Elasticsearch).
- Áp dụng thuật toán hợp nhất điểm số Weighted Reciprocal Rank Fusion (WRRF) để tạo ra bảng xếp hạng Top-K ứng viên.
- Viết giải thuật quy hoạch động DANTE để căn chỉnh thứ tự thời gian cho các bài toán chuỗi sự kiện nhiều khoảnh khắc (TRAKE).

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

**Đầu vào (Input):**  
- Chuỗi văn bản truy vấn \(Q\) (ví dụ: "Người đàn ông chạy đà, giậm nhảy...").
- Loại truy vấn `query_type`: `"KIS"`, `"Q&A"`, hoặc `"TRAKE"`.
- Kết quả tra cứu thô từ Milvus và Elasticsearch.

**Ví dụ Input (Cho truy vấn TRAKE với 3 sự kiện):**
```json
{
  "query": "Người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm.",
  "query_type": "TRAKE",
  "sub_queries": [
    "chạy đà",
    "giậm nhảy qua xà",
    "tiếp đất trên nệm"
  ],
  "retrieved_candidates_per_event": [
    {"event_idx": 0, "candidates": [{"video_id": "L01_V025", "frame_idx": 1050, "score": 0.91}]},
    {"event_idx": 1, "candidates": [{"video_id": "L01_V025", "frame_idx": 1080, "score": 0.89}]},
    {"event_idx": 2, "candidates": [{"video_id": "L01_V025", "frame_idx": 1120, "score": 0.88}]}
  ]
}
```

**Đầu ra (Output):**  
Danh sách đối tượng `RetrievalResult`:

**Ví dụ Output:**
```json
[
  {
    "video_id": "L01_V025",
    "frame_ids": [1050, 1080, 1120],
    "query_type": "TRAKE",
    "answer": null,
    "wrrf_score": 0.942
  }
]
```

## 3. Các công việc chi tiết (Key Tasks)

### 3.1. Lai ghép điểm số tìm kiếm (Hybrid Search & WRRF)

- Khi nhận kết quả danh sách từ tìm kiếm ngữ nghĩa và từ khoá, cài đặt thuật toán Weighted Reciprocal Rank Fusion (WRRF) theo công thức:
  \[
  \text{WRRF}(q, d) = \frac{\alpha_d}{r_{text}(q, d) + k} + \frac{1 - \alpha_d}{r_{vision}(q, d) + k}
  \]
- Trả về danh sách xếp hạng đã được trộn thứ hạng, lọc ra Top-100 khung hình tiềm năng nhất cho câu hỏi Textual KIS hoặc Q&A.

### 3.2. Thuật toán Quy hoạch động (DANTE cho TRAKE)

Triển khai thuật toán DANTE (Dynamic Alignment of Narrative Temporal Events):

- **Mã hóa chuỗi:** Cắt câu truy vấn thành \(N\) sự kiện \(U = [u_1, u_2, ..., u_N]\) và chuyển thành vector bằng BEiT-3 / SigLIP2.
- **Tính ma trận tương đồng:** Viết hàm tính độ tương đồng cosine \(S[i, t] = \text{cosine\_similarity}(u_i, E[t])\) giữa sự kiện \(i\) và vector khung hình \(t\).
- **Lập bảng DP:** Xây dựng mảng hai chiều \(DP[i, t]\), tính toán điểm tối ưu để khớp sự kiện vào các khung hình. Phải tích hợp một hệ số phạt thời gian \(\lambda\) (penalty factor) trừ điểm nếu 2 khung hình cách nhau quá xa hoặc sai trật tự thời gian.
- **Quay lui (Backtracking):** Viết vòng lặp dò ngược từ điểm cao nhất của bảng \(DP\) để trích xuất ra đúng một mảng chứa \(N\) phần tử `frame_ids` tăng dần theo thời gian.

## 4. Danh sách chuyển giao (Deliverables Checklist)

- [ ] `hybrid_search_wrrf.py`: Thuật toán trộn rank 2 luồng kết quả (Milvus & ES).
- [ ] `dante_trake_solver.py`: Logic quy hoạch động tính toán và căn chỉnh chuỗi sự kiện theo thời gian.
- [ ] `search.py`: Core retriever interface nhận query từ API/UI và trả về `RetrievalResult`.
- [ ] `tests/test_retrieval.py`: Test case giả lập \(N=3\), cung cấp 3 vector giả có thứ tự lộn xộn để khẳng định hàm DANTE luôn xuất ra mảng có frame index tăng dần và loại bỏ nhiễu.

> ⚠️ **Lưu ý:** File stub hiện tại `hybrid_retriever.py` cần được **xóa và thay thế** bằng 3 file kể trên. Giữ nguyên `hybrid_retriever.py` sẽ khiến lệnh `python -m phase4_retrieval.search` lỗi `ModuleNotFoundError`.
