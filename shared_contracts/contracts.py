from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class KeyframeData:
    """Contract Output Phase 1 -> Input Phase 2"""
    keyframe_id: str          # Format: {video_id}_{frame_idx}
    video_id: str
    frame_idx: int
    timestamp_sec: float
    image_matrix: Any         # Numpy array hoac PIL Image matrix

@dataclass
class MultimodalMetadata:
    """Contract Output Phase 2 -> Input Phase 3"""
    keyframe_id: str
    video_id: str             # Thêm để Phase 3/4 group theo video mà không cần parse keyframe_id
    dense_vector: List[float] # Tu SigLIP2 / BEiT-3
    ocr_text: str             # Tu Qwen2.5-VL

@dataclass
class RetrievalResult:
    """Contract Output Phase 4 -> Input Phase 5"""
    video_id: str
    frame_ids: List[int]      # N phần tử nếu là TRAKE, 1 phần tử nếu KIS/Q&A
    query_type: str           # "KIS", "Q&A", hoặc "TRAKE" — Phase 5 dùng để format output đúng chuẩn
    answer: Optional[str]     # Chỉ dùng cho Q&A
    wrrf_score: float
