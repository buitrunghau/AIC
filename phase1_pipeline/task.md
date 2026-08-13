## 1. Module Tiền xử lý Dữ liệu Video (Phase 1)

**Người phụ trách:** Member 1 – Data Pipeline Architect  
**Vai trò:** Phát triển luồng nạp video gốc, cắt cảnh và trích xuất khung hình thông minh để giảm tải cho các mô hình AI phía sau mà không làm mất thông tin sự kiện.

### 1.1. Trách nhiệm cốt lõi

- Phân đoạn video nguyên bản thành các cảnh quay (shots) có tính nhất quán về ngữ nghĩa.
- Lọc và loại bỏ các khung hình rác/trùng lặp để tối ưu hóa dung lượng tính toán.
- Đóng gói dữ liệu đầu ra thành chuẩn chung để chuyển giao sang Phase 2.

### 1.2. Đầu vào & Đầu ra

**Đầu vào:**  
Các tệp video thô nguyên bản định dạng `.mp4` được cung cấp từ bộ dữ liệu.

**Đầu ra:**  
Danh sách đối tượng `KeyframeData` chuẩn hóa:

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
1.3. Các công việc chi tiết
3.1. Phân đoạn cảnh quay (Shot Boundary Detection)
Ứng dụng mô hình học sâu TransNetV2 để quét qua luồng video.

Cấu hình mạng tích chập mở rộng (DCNN) của mô hình để bắt chính xác cả các cú cắt cảnh đột ngột (hard cuts) lẫn các đoạn chuyển cảnh mượt (dissolves/transitions).

Trả về danh sách ranh giới [start_frame, end_frame] của mỗi cảnh quay.

3.2. Trích xuất khung hình thích ứng (Adaptive Keyframe Sampling)
Thuật toán lấy mẫu thô: Lấy 4 khung hình tại các vị trí phân bổ đều trong một đoạn cắt cảnh (đầu, 2 giữa, cuối) dựa trên công thức:

text
k_extract = { K_{a + floor(i * (b - a) / 3)} }, với i ∈ {0, 1, 2, 3}
Thuật toán lọc tinh (Khoảng cách L2): Không lấy mẫu mù quáng. Lập trình hàm tính khoảng cách vector pixel giữa khung hiện tại và khung trước đó. Chỉ giữ lại nếu:

text
||e_current - e_prev||_2 / ||e_prev||_2 > 0.4
1.4. Danh sách chuyển giao (Deliverables Checklist)
□ transnet_segmentation.py: Class tải trọng số mô hình TransNetV2 và hàm chia video thành các đoạn cắt cảnh.
□ adaptive_sampler.py: Logic toán học tính khoảng cách L2 để lọc khung hình giữ lại.
□ tests/test_pipeline.py: Unit test đảm bảo ID khung hình xuất ra đúng thứ tự, hàm cắt không làm mất khung hình chứa nội dung quan trọng.
