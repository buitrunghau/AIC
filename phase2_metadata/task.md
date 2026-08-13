# Nhiệm vụ: Xây dựng Module Khai phá Siêu dữ liệu (Phase 2)

**Người phụ trách:** Member 2 – AI & Metadata Engineer

**Vai trò:** Nhận các khung hình/âm thanh thô và chuyển hóa chúng thành các vector toán học và văn bản (Metadata) có thể tìm kiếm được.

## 1. Trách nhiệm cốt lõi (Core Responsibility)

- Chuyển đổi ma trận pixel của khung hình thành các vector đặc trưng dày đặc (dense embeddings).
- Trích xuất chữ viết xuất hiện trong khung hình (OCR).
- Phiên mã âm thanh trong video thành văn bản lời thoại (ASR).

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

**Đầu vào (Input):**  
- Danh sách `KeyframeData` từ Module 1.
- Luồng âm thanh `.wav` hoặc `.mp3` bóc tách từ video gốc.

**Đầu ra (Output):**  
Danh sách đối tượng `MultimodalMetadata`:

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

## 3. Các công việc chi tiết (Key Tasks)

### 3.1. Trích xuất đặc trưng thị giác (Visual Embedding Generation)

- Tích hợp và chạy suy luận mô hình BEiT-3 và SigLIP2.
- Đối với SigLIP2, phải đảm bảo sử dụng hàm suy hao sigmoid (sigmoid loss) trong quá trình cấu hình (nếu có fine-tune) để biểu diễn không gian hội tụ tốt hơn so với CLIP thuần túy.
- Cắt giảm số chiều vector cho phù hợp với bộ nhớ.

### 3.2. Bóc tách văn bản và âm thanh (OCR & ASR)

- **OCR:** Gọi mô hình Qwen2.5-VL (hoặc Gemini API nếu có) xử lý ma trận ảnh của khung hình để trích xuất các chuỗi ký tự trên màn hình, lưu thành các cặp key-value (`keyframe_id` : `ocr_text`).
- **ASR:** Sử dụng mô hình Whisper quét qua tệp âm thanh, trích xuất lời thoại tiếng Việt/Anh và gắn nhãn thời gian (timestamp). Cần viết logic đối chiếu timestamp của âm thanh với `timestamp_sec` của khung hình để gán đoạn text đúng chỗ.

## 4. Danh sách chuyển giao (Deliverables Checklist)

- [ ] `siglip_beit_embedder.py`: Chứa các hàm encode hình ảnh thành vector đa chiều chuẩn hóa L2.
- [ ] `qwen_ocr_whisper_asr.py`: Tích hợp mô hình ngôn ngữ và âm thanh để sinh siêu dữ liệu dạng chữ.
- [ ] `tests/test_metadata.py`: Đảm bảo vector xuất ra đúng số chiều (vd: 768 hoặc 1024) và text OCR/ASR xử lý tốt bảng mã tiếng Việt.
