"""
Phase 1: Main pipeline script for end-to-end keyframe extraction (Memory-Optimized for 8GB RAM).
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


def extract_candidate_frames(video_path: Path, candidate_indices: List[int]) -> dict:
    """Reads only the specific frame indices needed for L2 filtering and output."""
    if not candidate_indices:
        return {}

    sorted_indices = sorted(set(candidate_indices))
    frame_dict = {}
    cap = cv2.VideoCapture(str(video_path))

    current_idx = 0
    target_ptr = 0
    total_targets = len(sorted_indices)

    while target_ptr < total_targets:
        target_idx = sorted_indices[target_ptr]

        # If target frame is far ahead, use seek; otherwise read sequentially
        if target_idx - current_idx > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            current_idx = target_idx

        ret, frame = cap.read()
        if not ret:
            break

        if current_idx == target_idx:
            frame_dict[target_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_ptr += 1

        current_idx += 1

    cap.release()
    return frame_dict


def process_video(
        video_path: Path,
        output_dir: Path,
        detector: ShotBoundaryDetector,
        sampler: AdaptiveSampler
) -> List[KeyframeData]:
    """Process a single video through Phase 1 without loading all high-res frames to RAM."""
    video_id = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    transnet_frames = []

    # PASS 1: Read video stream and store ONLY downscaled (27x48) frames for TransNetV2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize immediately to free the high-res buffer
        resized = cv2.resize(frame, (48, 27))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        transnet_frames.append(resized_rgb)

    cap.release()

    if not transnet_frames:
        return []

    transnet_np = np.array(transnet_frames, dtype=np.uint8)
    del transnet_frames  # Free list container

    # 1. Shot Boundary Detection
    segments = detector.segment_video(transnet_np)
    del transnet_np  # Free downscaled array after segmentation

    # 2. Generate candidate frame indices uniformly per shot
    candidate_indices = []
    for start_frame, end_frame in segments:
        candidate_indices.extend(sampler.sample_uniform_in_shot(start_frame, end_frame))
    candidate_indices = sorted(set(candidate_indices))

    # PASS 2: Fetch only the candidate frames from disk
    candidate_frames = extract_candidate_frames(video_path, candidate_indices)

    # 3. Adaptive L2 Distance Filtering per shot
    final_keyframe_indices = []
    for start_frame, end_frame in segments:
        shot_candidates = sampler.sample_uniform_in_shot(start_frame, end_frame)
        shot_keyframes = sampler.filter_candidates_by_l2(candidate_frames, shot_candidates)
        final_keyframe_indices.extend(shot_keyframes)

    final_keyframe_indices = sorted(set(final_keyframe_indices))

    # 4. Save results to disk
    results = []
    for idx in final_keyframe_indices:
        if idx not in candidate_frames:
            continue

        frame_rgb = candidate_frames[idx]
        kf_id = f"{video_id}_{idx}"

        kf_data = KeyframeData(
            keyframe_id=kf_id,
            video_id=video_id,
            frame_idx=idx,
            timestamp_sec=float(idx / fps),
            image_matrix=frame_rgb
        )
        results.append(kf_data)

        # Write frame to disk
        out_img_path = output_dir / f"{kf_id}.jpg"
        cv2.imwrite(str(out_img_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ShotBoundaryDetector(device=device)

    config = SamplingConfig(threshold=args.threshold)
    sampler = AdaptiveSampler(config=config)

    all_metadata = []

    for video_file in video_dir.glob("*.mp4"):
        print(f"Processing video: {video_file.name} ...")
        kf_list = process_video(video_file, output_dir, detector, sampler)

        for kf in kf_list:
            all_metadata.append({
                "keyframe_id": kf.keyframe_id,
                "video_id": kf.video_id,
                "frame_idx": kf.frame_idx,
                "timestamp_sec": kf.timestamp_sec
            })

    out_json_path = output_dir / "metadata.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[DONE] Pipeline finished. Extracted {len(all_metadata)} keyframes to {output_dir}")


if __name__ == "__main__":
    main()