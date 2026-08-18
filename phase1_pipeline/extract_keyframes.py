"""
Phase 1: Main pipeline script for end-to-end keyframe extraction.

Deliverable: extract_keyframes.py
Entry point: python -m phase1_pipeline.extract_keyframes
    --video_dir   Path to raw video directory
    --output_dir  Path to save extracted keyframes
    --threshold   L2-distance threshold for adaptive filtering (default: 0.4)

Output: List[KeyframeData] serialized to output_dir
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
import torch
from typing import List

from phase1_pipeline.transnet_segmentation import ShotBoundaryDetector
from phase1_pipeline.adaptive_sampler import AdaptiveSampler, SamplingConfig
from shared_contracts.contracts import KeyframeData


def process_video(video_path: Path, output_dir: Path, detector: ShotBoundaryDetector, sampler: AdaptiveSampler) -> List[
    KeyframeData]:
    """Process a single video through the Phase 1 pipeline."""
    video_id = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_list = []
    transnet_frames = []

    # Đọc toàn bộ frame và chuẩn bị dữ liệu đầu vào
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame_rgb)

        # TransNetV2 yêu cầu input có kích thước (H:27, W:48, C:3)
        resized_for_transnet = cv2.resize(frame_rgb, (48, 27))
        transnet_frames.append(resized_for_transnet)

    cap.release()

    if not frames_list:
        return []

    # Chuyển đổi sang numpy array
    frames_np = np.array(frames_list, dtype=np.uint8)
    transnet_np = np.array(transnet_frames, dtype=np.uint8)

    # 1. Phát hiện cắt cảnh (Shot Boundary Detection)
    segments = detector.segment_video(transnet_np)

    # 2. Lấy mẫu khung hình thích ứng (Adaptive Keyframe Sampling)
    keyframe_indices = sampler.extract_keyframes_from_segments(frames_np, segments)

    # 3. Đóng gói kết quả và lưu hình ảnh
    results = []
    for idx in keyframe_indices:
        kf_id = f"{video_id}_{idx}"

        kf_data = KeyframeData(
            keyframe_id=kf_id,
            video_id=video_id,
            frame_idx=idx,
            timestamp_sec=float(idx / fps),
            image_matrix=frames_np[idx]
        )
        results.append(kf_data)

        # Lưu hình ảnh vật lý ra ổ cứng
        out_img_path = output_dir / f"{kf_id}.jpg"
        cv2.imwrite(str(out_img_path), cv2.cvtColor(frames_np[idx], cv2.COLOR_RGB2BGR))

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Extract Keyframes")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to raw video directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save extracted keyframes")
    parser.add_argument("--threshold", type=float, default=0.4, help="L2-distance threshold")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo thiết bị và model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ShotBoundaryDetector(device=device)

    config = SamplingConfig(threshold=args.threshold)
    sampler = AdaptiveSampler(config=config)

    all_metadata = []

    # Quét và xử lý tất cả video .mp4
    for video_file in video_dir.glob("*.mp4"):
        print(f"Processing video: {video_file.name} ...")
        kf_list = process_video(video_file, output_dir, detector, sampler)

        # Serialize các thông tin non-matrix vào JSON
        for kf in kf_list:
            all_metadata.append({
                "keyframe_id": kf.keyframe_id,
                "video_id": kf.video_id,
                "frame_idx": kf.frame_idx,
                "timestamp_sec": kf.timestamp_sec
            })

    # Xuất metadata log
    out_json_path = output_dir / "metadata.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[DONE] Pipeline finished. Extracted {len(all_metadata)} keyframes to {output_dir}")


if __name__ == "__main__":
    main()