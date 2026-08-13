# Nhiệm vụ: Xây dựng Module Truy xuất & Căn chỉnh Chuỗi sự kiện (Phase 4)

**Người phụ trách:** Member 4 – AI & Retrieval Scientist

**Vai trò:** Xử lý câu truy vấn của người dùng, thực hiện thuật toán trộn điểm tìm kiếm không gian, và quan trọng nhất là lập trình thuật toán giải bài toán chuỗi thời gian (TRAKE).

## 1. Trách nhiệm cốt lõi (Core Responsibility)

- Điều phối hai luồng tìm kiếm song song: tìm kiếm theo độ tương đồng vector (trên Milvus) và tìm kiếm theo từ vựng (trên Elasticsearch).
- Áp dụng thuật toán hợp nhất điểm số để tạo ra bảng xếp hạng Top-K ứng viên.
- Viết giải thuật quy hoạch động DANTE để căn chỉnh thứ tự thời gian cho các bài toán chuỗi sự kiện nhiều khoảnh khắc.

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

**Đầu vào (Input):**  
- Chuỗi văn bản truy vấn \(Q\) (vd: "Người đàn ông chạy đà, giậm nhảy...").
- Danh sách điểm số thô và `keyframe_id` trả về từ DB Phase 3.

**Ví dụ Input:**
```json
{
  "query": "Người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm.",
  "query_type": "TRAKE",
  "dense_hits": [
    {"keyframe_id": "L01_V025_1050", "rank": 1, "score": 0.91},
    {"keyframe_id": "L01_V025_1080", "rank": 2, "score": 0.89},
    {"keyframe_id": "L01_V025_1120", "rank": 3, "score": 0.88}
  ],
  "sparse_hits": [
    {"keyframe_id": "L01_V025_1080", "rank": 1, "score": 13.2},
    {"keyframe_id": "L01_V025_1050", "rank": 2, "score": 12.7},
    {"keyframe_id": "L01_V025_1120", "rank": 4, "score": 11.4}
  ]
}
```

**Đầu ra (Output):**  
Danh sách đối tượng `RetrievalResult` (Sẽ được Phase 5 đóng gói):

**Ví dụ Output:**
```json
[
  {
    "video_id": "L01_V025",
    "frame_ids": [1050, 1080, 1120],
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

- **Mã hóa chuỗi:** Cắt câu truy vấn thành \(N\) sự kiện \(U = [u_1, u_2, ..., u_N]\) và chuyển thành vector bằng BEiT-3.
- **Tính ma trận tương đồng:** Viết hàm tính độ tương đồng cosine \(S[i, t] = \text{cosine\_similarity}(u_i, E[t])\) giữa sự kiện \(i\) và vector khung hình \(t\).
- **Lập bảng DP:** Xây dựng mảng hai chiều \(DP[i, t]\), tính toán điểm tối ưu để khớp sự kiện vào các khung hình. Phải tích hợp một hệ số phạt thời gian \(\lambda\) (penalty factor) trừ điểm nếu 2 khung hình cách nhau quá xa hoặc sai trật tự thời gian.
- **Quay lui (Backtracking):** Viết vòng lặp dò ngược từ điểm cao nhất của bảng \(DP\) để trích xuất ra đúng một mảng chứa \(N\) phần tử `frame_ids` tăng dần theo thời gian.

## 4. Danh sách chuyển giao (Deliverables Checklist)

- [ ] `hybrid_search_wrrf.py`: Thuật toán trộn rank 2 luồng kết quả.
- [ ] `dante_trake_solver.py`: Toàn bộ logic toán học của hệ thống quy hoạch động tính toán chuỗi sự kiện.
- [ ] `tests/test_retrieval.py`: Test case giả lập \(N=3\), cung cấp 3 vector giả có thứ tự lộn xộn để khẳng định hàm DANTE luôn xuất ra mảng có index lớn dần và loại bỏ được nhiễu.
