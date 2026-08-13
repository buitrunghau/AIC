# Nhiệm vụ: Xây dựng Module Lập chỉ mục (Phase 3)

**Người phụ trách:** Member 3 – Backend & DB Administrator

**Vai trò:** Quản lý và vận hành cơ sở dữ liệu vector và văn bản, đảm bảo khả năng truy vấn nhanh, chính xác và liên kết chặt chẽ.

## 1. Trách nhiệm cốt lõi (Core Responsibility)

- Nạp toàn bộ dữ liệu metadata từ Phase 2 vào hai hệ quản trị cơ sở dữ liệu chuyên biệt.
- Đồng bộ hóa khóa chính `keyframe_id` giữa hai hệ thống để phục vụ việc hợp nhất kết quả sau này.
- Tối ưu cấu hình index nhằm đạt độ trễ truy vấn thấp nhất.

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

**Đầu vào (Input):**  
Danh sách `MultimodalMetadata` từ Phase 2.

**Ví dụ Input:**
```json
[
  {
    "keyframe_id": "L01_V025_1050",
    "dense_vector": [0.12, -0.45, 0.89, "..."],
    "ocr_text": "BỆNH VIỆN CHỢ RẪY",
    "asr_transcript": "Bệnh nhân đã được cấp cứu kịp thời."
  }
]
```

**Đầu ra (Output):**  
Indexed Database gồm:
- Milvus collection chứa các dense vectors (SigLIP/BEiT-3).
- Elasticsearch index chứa dữ liệu văn bản (OCR/ASR).

**Ví dụ Output:**
```json
{
  "milvus_collection": {
    "name": "video_frames",
    "records": 1,
    "sample": {
      "keyframe_id": "L01_V025_1050",
      "dense_vector_dim": 1024
    }
  },
  "elasticsearch_index": {
    "name": "video_text",
    "documents": 1,
    "sample": {
      "keyframe_id": "L01_V025_1050",
      "ocr_text": "BỆNH VIỆN CHỢ RẪY",
      "asr_transcript": "Bệnh nhân đã được cấp cứu kịp thời."
    }
  }
}
```

## 3. Các công việc chi tiết (Key Tasks)

### 3.1. Milvus (Dense Vectors)

- Tạo collection với trường `keyframe_id` là khóa chính và trường vector có số chiều tương ứng (ví dụ 768 hoặc 1024).
- Sử dụng thuật toán đồ thị HNSW cho phép tìm kiếm lân cận gần nhất (ANN) siêu tốc.
- Thiết lập index với các tham số `M`, `efConstruction` phù hợp để cân bằng giữa tốc độ và độ chính xác.

### 3.2. Elasticsearch (Sparse/Text)

- Tạo index với trường `keyframe_id` và các trường văn bản `ocr_text`, `asr_transcript`.
- Cấu hình analyzer phù hợp cho tiếng Việt/Anh.
- Sử dụng BM25 kết hợp mô hình nhúng thưa để đối sánh từ vựng chính xác (exact lexical matching).

## 4. Danh sách chuyển giao (Deliverables Checklist)

- [ ] `milvus_indexer.py`: Script tạo collection và nạp vector vào Milvus.
- [ ] `elastic_indexer.py`: Script tạo index và nạp văn bản vào Elasticsearch.
- [ ] `tests/test_indexing.py`: Kiểm tra số lượng bản ghi, tính nhất quán `keyframe_id` giữa hai DB.
