# Multimodal Hybrid Search & Temporal Alignment in Video Retrieval

Hệ thống truy xuất video đa phương thức kết hợp tìm kiếm ngữ nghĩa (dense vectors), tìm kiếm từ khóa (sparse/text) và căn chỉnh chuỗi thời gian (temporal alignment) cho các loại truy vấn KIS, Q&A và TRAKE.

## Mục lục

1. [Tổng quan kiến trúc & Tóm tắt mục tiêu](#1-tổng-quan-kiến-trúc--tóm-tắt-mục-tiêu)
2. [Sơ đồ luồng dữ liệu tổng thể](#2-sơ-đồ-luồng-dữ-liệu-tổng-thể)
3. [Phase 1 – Data Pipeline Module](#3-phase-1--data-pipeline-module)
4. [Phase 2 – Metadata Module](#4-phase-2--metadata-module)
5. [Phase 3 – Indexing Module](#5-phase-3--indexing-module)
6. [Phase 4 – Retrieval & Alignment Module](#6-phase-4--retrieval--alignment-module)
7. [Phase 5 – QA & UI Module](#7-phase-5--qa--ui-module)
8. [Luồng dữ liệu minh họa](#8-luồng-dữ-liệu-minh-họa)
9. [Giao diện tích hợp giữa các Module (API Contracts)](#9-giao-diện-tích-hợp-giữa-các-module-api-contracts)
10. [Thiết kế phi chặn (Non-blocking Integration)](#10-thiết-kế-phi-chặn-non-blocking-integration)
11. [Quyết định thiết kế & Trade-offs](#11-quyết-định-thiết-kế--trade-offs)

---

## 1. Tổng quan kiến trúc & Tóm tắt mục tiêu

Hệ thống được thiết kế theo mô hình **Multimodal Hybrid Pipeline**, cho phép xử lý luồng dữ liệu đa phương thức (video, audio, text) một cách tuần tự nhưng có khả năng tìm kiếm kết hợp và suy luận chuỗi thời gian khắt khe.

### Mục tiêu chính

- **Tiền xử lý thông minh**: Loại bỏ khung hình rác, chỉ giữ lại các khung hình có giá trị ngữ nghĩa thực sự thông qua mạng nơ-ron và đo lường khoảng cách vector.
- **Khai phá đa phương thức**: Chuyển hóa toàn bộ pixel hình ảnh và luồng âm thanh thành vector toán học (dense embeddings) và chuỗi văn bản (OCR, ASR).
- **Lập chỉ mục siêu tốc**: Xây dựng cơ sở dữ liệu kép để phục vụ tìm kiếm ngữ nghĩa (FAISS/Milvus) và tìm kiếm từ khóa chính xác (Elasticsearch).
- **Truy xuất & Căn chỉnh thời gian**: Áp dụng quy hoạch động (DANTE) để giải quyết loại câu hỏi chuỗi sự kiện TRAKE, bắt buộc tính toán trật tự thời gian.
- **Tương tác & Đệ trình tự động**: Triển khai vòng lặp phản hồi Rocchio để tối ưu hóa truy vấn và tự động xuất file `.csv` chuẩn CodaBench.

---

## 2. Sơ đồ luồng dữ liệu tổng thể

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
