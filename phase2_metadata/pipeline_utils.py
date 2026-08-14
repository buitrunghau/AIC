import glob
import json
import os
import numpy as np


def get_all_keyframe_paths(keyframes_dir):
    """Quét toàn bộ danh sách đường dẫn ảnh keyframe trong thư mục dữ liệu."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(
            glob.glob(os.path.join(keyframes_dir, "*", ext), recursive=True)
        )

    image_paths = sorted(image_paths)
    print(f"📁 [utils] Tìm thấy tổng cộng {len(image_paths)} ảnh keyframe.")
    return image_paths


def parse_frame_info(image_path):
    """Trích xuất video_id và frame_id từ đường dẫn file ảnh."""
    parts = os.path.normpath(image_path).split(os.sep)
    video_id = parts[-2]
    file_name = parts[-1]
    frame_id = os.path.splitext(file_name)[0]
    return video_id, frame_id


def save_json(data, output_path):
    """Hàm ghi dữ liệu cấu trúc ra file JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 [utils] Đã lưu file JSON tại: {output_path}")


def load_json(input_path):
    """Hàm đọc dữ liệu từ file JSON."""
    if not os.path.exists(input_path):
        print(f"⚠️ [utils] File không tồn tại: {input_path}")
        return None
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(array, output_path):
    """Hàm lưu ma trận NumPy vector ra file .npy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, array)
    print(f"💾 [utils] Đã lưu ma trận NPY tại: {output_path}")