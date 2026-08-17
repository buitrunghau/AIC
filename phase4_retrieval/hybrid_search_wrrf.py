"""
Phase 4: Weighted Reciprocal Rank Fusion (WRRF) hybrid search.

Deliverable: hybrid_search_wrrf.py
- Merges ranked results from Milvus (dense/vision) and Elasticsearch (BM25/text).
- Formula: WRRF(q, d) = alpha / (r_text(q,d) + k) + (1 - alpha) / (r_vision(q,d) + k)
  where alpha is the text weight (default 0.6), k is smoothing constant (default 60).
- Returns Top-K merged candidates for KIS and Q&A queries.
"""

from typing import List, Dict, Optional


def calculate_wrrf(
    es_results: List[Dict],
    milvus_results: List[Dict],
    alpha: float = 0.6,
    k: int = 60,
    top_k: Optional[int] = None
) -> List[Dict]:
    """
    Hợp nhất kết quả từ hai luồng tìm kiếm bằng WRRF.

    Args:
        es_results: Danh sách kết quả từ Elasticsearch (đã xếp hạng).
        milvus_results: Danh sách kết quả từ Milvus (đã xếp hạng).
        alpha: Trọng số cho luồng text (Elasticsearch). Mặc định 0.6.
        k: Hằng số làm mượt (smoothing). Mặc định 60.
        top_k: Số lượng kết quả trả về (None = trả về tất cả).

    Returns:
        List các dict có cấu trúc:
        {
            "video_id": str,
            "frame_idx": int,
            "wrrf_score": float,
            "rank": int
        }
        Đã sắp xếp theo wrrf_score giảm dần.
    """
    fused_scores = {}

    def get_key(item: Dict) -> str:
        return f"{item['video_id']}_{item['frame_idx']}"

    # Xử lý kết quả từ Elasticsearch (text)
    for rank, item in enumerate(es_results, start=1):
        # Kiểm tra dữ liệu đầu vào
        if "video_id" not in item or "frame_idx" not in item:
            continue

        key = get_key(item)
        if key not in fused_scores:
            fused_scores[key] = {
                "video_id": item["video_id"],
                "frame_idx": item["frame_idx"],
                "wrrf_score": 0.0
            }
        fused_scores[key]["wrrf_score"] += alpha / (rank + k)

    # Xử lý kết quả từ Milvus (vision)
    for rank, item in enumerate(milvus_results, start=1):
        if "video_id" not in item or "frame_idx" not in item:
            continue

        key = get_key(item)
        if key not in fused_scores:
            fused_scores[key] = {
                "video_id": item["video_id"],
                "frame_idx": item["frame_idx"],
                "wrrf_score": 0.0
            }
        fused_scores[key]["wrrf_score"] += (1 - alpha) / (rank + k)

    # Chuyển sang list và sắp xếp
    ranked_results = list(fused_scores.values())
    ranked_results.sort(key=lambda x: x["wrrf_score"], reverse=True)

    # Gán rank
    for i, res in enumerate(ranked_results, start=1):
        res["rank"] = i

    # Giới hạn top_k nếu có
    if top_k is not None:
        ranked_results = ranked_results[:top_k]

    return ranked_results
