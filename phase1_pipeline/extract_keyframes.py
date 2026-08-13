"""Phase 1: Main pipeline script for end-to-end keyframe extraction.

Deliverable: extract_keyframes.py
Entry point: python -m phase1_pipeline.extract_keyframes
    --video_dir   Path to raw video directory
    --output_dir  Path to save extracted keyframes
    --threshold   L2-distance threshold for adaptive filtering (default: 0.4)

Output: List[KeyframeData] serialized to output_dir
"""
