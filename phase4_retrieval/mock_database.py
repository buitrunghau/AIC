"""Phase 4: Mock database layer — giả lập Milvus và Elasticsearch.

Mô phỏng interface của Phase 3 (Indexing Module) để Phase 4 có thể phát triển
và test độc lập mà không cần kết nối database thực.

Interface contract:
    - Input: query (str), top_k (int)
    - Output: List[Dict] với keys: video_id (str), frame_idx (int), score (float)
    - Kết quả đã sắp xếp theo score giảm dần.

Lưu ý:
    Khi tích hợp thực tế, thay thế các hàm này bằng client gọi tới
    Phase 3 MilvusIndexer.search() và ElasticIndexer.search().
"""

import random


def search_elastic_db(query: str, top_k: int = 100) -> list:
    """Giả lập tìm kiếm BM25 trên Elasticsearch (OCR text matching).

    Args:
        query: Chuỗi truy vấn văn bản.
        top_k: Số lượng kết quả trả về.

    Returns:
        List[Dict] với video_id, frame_idx, score — sắp xếp theo score giảm dần.
    """
    random.seed(hash(query) % (2**31))
    results = []
    for _ in range(top_k):
        results.append({
            "video_id": f"L{random.randint(1, 5):02d}_V{random.randint(1, 50):03d}",
            "frame_idx": random.randint(100, 50000),
            "score": round(random.uniform(0.3, 0.99), 3)
        })
    # Sắp xếp theo score giảm dần (mô phỏng BM25 ranking)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def search_milvus_db(query: str, top_k: int = 100) -> list:
    """Giả lập tìm kiếm ANN trên Milvus (dense vector similarity).

    Args:
        query: Chuỗi truy vấn (trong thực tế sẽ được encode thành vector trước).
        top_k: Số lượng kết quả trả về.

    Returns:
        List[Dict] với video_id, frame_idx, score — sắp xếp theo score giảm dần.
    """
    random.seed(hash(query + "_milvus") % (2**31))
    results = []
    for _ in range(top_k):
        results.append({
            "video_id": f"L{random.randint(1, 5):02d}_V{random.randint(1, 50):03d}",
            "frame_idx": random.randint(100, 50000),
            "score": round(random.uniform(0.4, 0.99), 3)
        })
    # Sắp xếp theo score giảm dần (mô phỏng cosine similarity ranking)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
