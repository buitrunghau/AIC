"""
config.py
=========
Cấu hình tập trung cho Phase 3 - Indexing Module.
Tất cả tham số có thể override qua biến môi trường để dễ triển khai trên
nhiều môi trường (local / staging / production) mà không sửa code.
"""
import os
from dataclasses import dataclass, field


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


@dataclass
class MilvusConfig:
    """Cấu hình collection + HNSW index cho Milvus (dense vector search)."""

    host: str = field(default_factory=lambda: _env_str("MILVUS_HOST", "localhost"))
    port: str = field(default_factory=lambda: _env_str("MILVUS_PORT", "19530"))
    collection_name: str = field(
        default_factory=lambda: _env_str("MILVUS_COLLECTION", "video_frames")
    )
    vector_dim: int = field(
        default_factory=lambda: _env_int("MILVUS_VECTOR_DIM", 1024)
    )
    # COSINE phù hợp với embedding đã chuẩn hoá (SigLIP2/BEiT-3 thường L2-normalize)
    metric_type: str = field(
        default_factory=lambda: _env_str("MILVUS_METRIC_TYPE", "COSINE")
    )
    index_type: str = "HNSW"
    # M: số lượng liên kết tối đa mỗi node trong đồ thị HNSW (trade-off:
    # M cao -> recall tốt hơn, tốn RAM/thời gian build hơn).
    hnsw_m: int = field(default_factory=lambda: _env_int("MILVUS_HNSW_M", 16))
    # efConstruction: độ rộng tìm kiếm lúc build index (cao hơn -> index
    # chính xác hơn nhưng build chậm hơn).
    hnsw_ef_construction: int = field(
        default_factory=lambda: _env_int("MILVUS_HNSW_EF_CONSTRUCTION", 200)
    )
    # ef: độ rộng tìm kiếm lúc query (cao hơn -> recall tốt hơn, latency tăng).
    hnsw_ef_search: int = field(
        default_factory=lambda: _env_int("MILVUS_HNSW_EF_SEARCH", 64)
    )


@dataclass
class ElasticConfig:
    """Cấu hình index + BM25 cho Elasticsearch (sparse/text search)."""

    hosts: str = field(
        default_factory=lambda: _env_str("ES_HOSTS", "http://localhost:9200")
    )
    index_name: str = field(
        default_factory=lambda: _env_str("ES_INDEX", "video_text")
    )
    bm25_k1: float = field(default_factory=lambda: _env_float("ES_BM25_K1", 1.2))
    bm25_b: float = field(default_factory=lambda: _env_float("ES_BM25_B", 0.75))
    shards: int = field(default_factory=lambda: _env_int("ES_SHARDS", 1))
    replicas: int = field(default_factory=lambda: _env_int("ES_REPLICAS", 1))
