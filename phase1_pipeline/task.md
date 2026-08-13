# Nhiệm vụ 1: Xây dựng Module Tiền xử lý Dữ liệu Video (Phase 1)

**Người phụ trách:** Member 1

**Vai trò:**
Phát triển luồng nạp video gốc, cắt cảnh và trích xuất khung hình thông minh để giảm tải cho các mô hình AI phía sau mà không làm mất thông tin sự kiện.

## 1. Trách nhiệm cốt lõi (Core Responsibility)
- Phân đoạn video nguyên bản thành các cảnh quay (shots) có tính nhất quán về ngữ nghĩa.
- Lọc và loại bỏ các khung hình rác/trùng lặp để tối ưu hóa dung lượng tính toán.
- Đóng gói dữ liệu đầu ra thành chuẩn chung để chuyển giao sang Phase 2.

## 2. Đầu vào & Đầu ra (Inputs & Outputs)

### Đầu vào (Input)
- Các tệp video thô nguyên bản định dạng `.mp4` được cung cấp từ bộ dữ liệu.

### Đầu ra (Output)
Danh sách đối tượng `KeyframeData` chuẩn hóa:

```json
[
  {
    "keyframe_id": "L01_V025_1050",
    "video_id": "L01_V025",
    "frame_idx": 1050,
    "timestamp_sec": 35.0,
    "image_matrix": "..."
  }
]
```
