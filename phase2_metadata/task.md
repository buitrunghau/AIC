# Nhiệm vụ: Xây dựng Module Khai phá Siêu dữ liệu (Phase 2)

**Người phụ trách:** Member 2 – AI & Metadata Engineer

**Vai trò:** Nhận các khung hình thô từ Phase 1 và chuyển hóa chúng thành các vector toán học (dense embeddings) và văn bản thị giác (OCR) có thể tìm kiếm được.

## 1. Trách nhiệm cốt lõi (Core Responsibility)

- Chuyển đổi ma trận pixel của khung hình thành các vector đặc trưng dày đặc (dense embeddings) bằng SigLIP2 và BEiT-3.
- Trích xuất chữ viết xuất hiện trong khung hình (OCR) bằng Qwen2.5-VL.
- Đóng gói siêu dữ liệu xuất ra dưới dạng `MultimodalMetadata`.

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

**Đầu vào (Input):**  
- Danh sách `KeyframeData` từ Phase 1.

**Ví dụ Input:**
```json
[
  {
    "keyframe_id": "L01_V025_1050",
    "video_id": "L01_V025",
    "frame_idx": 1050,
    "timestamp_sec": 35.0,
    "image_matrix": "<numpy_array_data>"
  }
]
```

**Đầu ra (Output):**  
Danh sách đối tượng `MultimodalMetadata`:

**Ví dụ Output:**
```json
[
  {
    "keyframe_id": "L01_V025_1050",
    "video_id": "L01_V025",
    "dense_vector": [0.12, -0.45, 0.89, "..."],
    "ocr_text": "BỆNH VIỆN CHỢ RẪY"
  }
]
```

## 3. Các công việc chi tiết (Key Tasks)

### 3.1. Trích xuất đặc trưng thị giác (Visual Embedding Generation)

- Tích hợp và chạy suy luận mô hình BEiT-3 và SigLIP2.
- Đối với SigLIP2, đảm bảo sử dụng hàm suy hao sigmoid (sigmoid loss) cho biểu diễn không gian hội tụ tốt hơn so với CLIP thuần túy.
- Chuẩn hóa vector đặc trưng theo độ dài L2.

### 3.2. Bóc tách văn bản thị giác (OCR Extraction)

- Gọi mô hình Qwen2.5-VL xử lý ma trận ảnh của khung hình để trích xuất các chuỗi ký tự xuất hiện trên màn hình (bảng hiệu, dòng chữ thời sự, bảng tên,...).
- Lưu thông tin OCR vào trường `ocr_text` trong `MultimodalMetadata`.

## 4. Danh sách chuyển giao (Deliverables Checklist)

- [ ] `siglip_beit_embedder.py`: Chứa các hàm encode hình ảnh thành vector đa chiều chuẩn hóa L2.
- [ ] `qwen_ocr_extractor.py`: Tích hợp mô hình MLLM để trích xuất văn bản thị giác (OCR).
- [ ] `generate_metadata.py`: Script pipeline chính thực thi Phase 2 end-to-end.
- [ ] `tests/test_metadata.py`: Đảm bảo vector xuất ra đúng số chiều (vd: 768 hoặc 1024) và text OCR xử lý tốt tiếng Việt.

> ⚠️ **Lưu ý:** File stub hiện tại `multimodal_extractor.py` cần được **xóa và thay thế** bằng 3 file kể trên. Giữ nguyên `multimodal_extractor.py` sẽ khiến lệnh `python -m phase2_metadata.generate_metadata` lỗi `ModuleNotFoundError`.
