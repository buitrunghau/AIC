"""
contracts.py
============
API Contracts giữa các Module (theo mục 9 - Kiến trúc hệ thống).

Phase 3 (Indexing Module) chỉ TIÊU THỤ (consume) `MultimodalMetadata`.
Không được sửa đổi contract này một cách cục bộ trong Phase 3 — mọi thay đổi
schema phải được thống nhất với Member 2 (Metadata Module) vì đây là điểm
giao tiếp (interface) giữa hai module, cho phép 5 thành viên code độc lập.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class MultimodalMetadata:
    """Contract Output Phase 2 -> Input Phase 3"""

    keyframe_id: str  # Format: {video_id}_{frame_idx}
    dense_vector: List[float]  # Từ SigLIP2/BEiT-3
    ocr_text: str = ""  # Từ Qwen2.5-VL
    asr_transcript: str = ""  # Từ Whisper
