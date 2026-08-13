# Nhiệm vụ: Xây dựng Module QA & UI (Phase 5)

**Người phụ trách:** Member 5 – Query Ops & QA Manager

**Vai trò:** Quản lý tương tác với người dùng cuối, tinh chỉnh truy vấn dựa trên phản hồi và đóng gói kết quả theo định dạng yêu cầu.

## 1. Trách nhiệm cốt lõi (Core Responsibility)

- Tiếp nhận và xử lý phản hồi từ người dùng để cải thiện chất lượng truy vấn.
- Tự động hóa quy trình xuất file kết quả đạt chuẩn CodaBench.
- Đóng gói toàn bộ câu trả lời thành tệp `submission.zip`.

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

**Đầu vào (Input):**  
- Danh sách `RetrievalResult` từ Phase 4.
- Phản hồi của người dùng (nhãn Đúng/Sai) nếu có.

**Ví dụ Input:**
```json
{
  "retrieval_results": [
    {
      "video_id": "L01_V025",
      "frame_ids": [1050, 1080, 1120],
      "answer": null,
      "wrrf_score": 0.942
    },
    {
      "video_id": "L02_V030.mp4",
      "frame_ids": [5120],
      "answer": "59A-123.45",
      "wrrf_score": 0.931
    }
  ],
  "feedback_labels": [
    {"video_id": "L01_V025", "frame_ids": [1050, 1080, 1120], "label": "relevant"},
    {"video_id": "L02_V030.mp4", "frame_ids": [5120], "label": "relevant"}
  ]
}
```

**Đầu ra (Output):**  
Tệp `submission.zip` chứa file CSV chuẩn CodaBench.

**Ví dụ Output:**
```json
{
  "zip_file": "/submission/submission.zip",
  "csv_file": "/submission/results.csv",
  "csv_preview": [
    {
      "video_id": "L01_V025",
      "frame_ids": "1050 1080 1120",
      "answer": ""
    },
    {
      "video_id": "L02_V030",
      "frame_ids": "5120",
      "answer": "\"59A-123.45\""
    }
  ]
}
```

## 3. Các công việc chi tiết (Key Tasks)

### 3.1. Relevance Feedback (Rocchio)

- Người dùng gắn nhãn Đúng/Sai trực quan cho các kết quả truy vấn.
- Thuật toán Rocchio tự động dịch chuyển vector truy vấn \(q_m\) về gần với các kết quả đúng (\(C_r\)) và đẩy ra xa bối cảnh sai (\(C_{nr}\)).

### 3.2. Format Validator & ZIP Formatter

- Code tự động xóa đuôi `.mp4` khỏi `video_id`.
- Bọc ngoặc kép cho câu trả lời Q&A.
- Gói tệp CSV vào thư mục trung gian `submission` trước khi nén thành `.zip`.

## 4. Danh sách chuyển giao (Deliverables Checklist)

- [ ] `rocchio_feedback.py`: Cài đặt vòng lặp phản hồi Rocchio để tinh chỉnh vector truy vấn.
- [ ] `csv_formatter.py`: Chuyển đổi kết quả truy xuất thành file CSV đúng chuẩn, xử lý mọi quy tắc định dạng.
- [ ] `tests/test_qa_ui.py`: Kiểm tra định dạng CSV, kiểm tra file zip xuất ra đúng cấu trúc và nội dung.
