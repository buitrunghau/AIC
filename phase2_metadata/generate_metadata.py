"""Phase 2: Main pipeline script for end-to-end metadata generation.

Deliverable: generate_metadata.py
Entry point: python -m phase2_metadata.generate_metadata
    --keyframe_dir  Path to keyframes from Phase 1
    --output_dir    Path to save MultimodalMetadata output
    --vision_model  Vision model to use: siglip2 or beit3
    --ocr_model     OCR model to use: qwen2.5-vl

Output: List[MultimodalMetadata] serialized to output_dir
"""
