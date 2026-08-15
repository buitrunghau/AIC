"""
milvus_indexer.py
==================
Phase 3 - Indexing Module (Member 3 - Backend & DB Administrator)

Chịu trách nhiệm:
    - Tạo Milvus collection với `keyframe_id` là khóa chính (VARCHAR).
    - Build HNSW index trên trường dense_vector để ANN search siêu tốc.
    - Nạp dense vectors (SigLIP2/BEiT-3) từ `MultimodalMetadata`.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from config import MilvusConfig
from contracts import MultimodalMetadata

logger = logging.getLogger(__name__)


class MilvusIndexer:
    """Quản lý vòng đời Milvus collection dùng cho dense-vector retrieval."""

    PK_FIELD = "keyframe_id"
    VECTOR_FIELD = "dense_vector"

    def __init__(self, cfg: Optional[MilvusConfig] = None, client=None):
        self.cfg = cfg or MilvusConfig()
        self._collection = None
        self._injected_client = client
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Kết nối & khởi tạo schema
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        if self._injected_client is not None:
            logger.info("MilvusIndexer: dùng injected client (test mode).")
            return

        from pymilvus import connections

        connections.connect(
            alias="default",
            host=self.cfg.host,
            port=self.cfg.port,
        )
        logger.info("Đã kết nối Milvus tại %s:%s", self.cfg.host, self.cfg.port)

    def create_collection(self, recreate: bool = False):
        """Tạo collection với PK = keyframe_id (VARCHAR) + vector field."""
        if self._injected_client is not None:
            self._collection = self._injected_client
            if recreate and hasattr(self._collection, "drop"):
                self._collection.drop()
            return self._collection

        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if utility.has_collection(self.cfg.collection_name):
            if recreate:
                Collection(self.cfg.collection_name).drop()
            else:
                self._collection = Collection(self.cfg.collection_name)
                return self._collection

        fields = [
            FieldSchema(
                name=self.PK_FIELD,
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name=self.VECTOR_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.cfg.vector_dim,
            ),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Dense vectors cho video keyframes",
        )
        self._collection = Collection(
            name=self.cfg.collection_name, schema=schema
        )
        logger.info(
            "Đã tạo collection '%s' (dim=%d).",
            self.cfg.collection_name,
            self.cfg.vector_dim,
        )
        return self._collection

    def build_index(self) -> None:
        """Tạo HNSW index trên trường vector."""
        index_params = {
            "index_type": self.cfg.index_type,
            "metric_type": self.cfg.metric_type,
            "params": {
                "M": self.cfg.hnsw_m,
                "efConstruction": self.cfg.hnsw_ef_construction,
            },
        }
        if self._collection is None:
            raise RuntimeError(
                "Gọi create_collection() trước khi build_index()."
            )
        self._collection.create_index(
            field_name=self.VECTOR_FIELD, index_params=index_params
        )
        logger.info("Đã build HNSW index: %s", index_params)

    # ------------------------------------------------------------------ #
    # Nạp dữ liệu
    # ------------------------------------------------------------------ #
    def insert(self, records: Sequence[MultimodalMetadata]) -> int:
        if not records:
            return 0
        if self._collection is None:
            raise RuntimeError(
                "Collection chưa được khởi tạo. Gọi create_collection() trước."
            )

        self._validate_dim(records)

        ids = [r.keyframe_id for r in records]
        vectors = [r.dense_vector for r in records]

        self._collection.insert([ids, vectors])
        logger.info("Đã insert %d bản ghi vào Milvus (chưa flush).", len(ids))
        return len(ids)

    def flush(self) -> None:
        if self._collection is not None and hasattr(self._collection, "flush"):
            self._collection.flush()
            logger.info("Đã flush collection '%s'.", self.cfg.collection_name)

    def load(self) -> None:
        if self._collection is not None and hasattr(self._collection, "load"):
            self._collection.load()
        self._loaded = True

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.num_entities

    def get_all_ids(self, batch_size: int = 10000) -> List[str]:
        if self._collection is None:
            return []

        all_ids: List[str] = []
        offset = 0
        while True:
            results = self._collection.query(
                expr=f'{self.PK_FIELD} != ""',
                output_fields=[self.PK_FIELD],
                limit=batch_size,
                offset=offset,
            )
            if not results:
                break
            all_ids.extend(r[self.PK_FIELD] for r in results)
            if len(results) < batch_size:
                break
            offset += batch_size
        return all_ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        ef: Optional[int] = None,
    ):
        if not self._loaded:
            logger.warning(
                "Collection '%s' chưa được load() — tự động load trước.",
                self.cfg.collection_name,
            )
            self.load()

        search_params = {
            "metric_type": self.cfg.metric_type,
            "params": {"ef": ef or self.cfg.hnsw_ef_search},
        }
        return self._collection.search(
            data=[query_vector],
            anns_field=self.VECTOR_FIELD,
            param=search_params,
            limit=top_k,
            output_fields=[self.PK_FIELD],
        )

    # ------------------------------------------------------------------ #
    def _validate_dim(self, records: Sequence[MultimodalMetadata]) -> None:
        for r in records:
            if len(r.dense_vector) != self.cfg.vector_dim:
                raise ValueError(
                    f"keyframe_id={r.keyframe_id}: dense_vector có "
                    f"{len(r.dense_vector)} chiều, kỳ vọng "
                    f"{self.cfg.vector_dim} (xem MilvusConfig.vector_dim)."
                )

    def close(self) -> None:
        if self._injected_client is not None:
            return
        from pymilvus import connections

        connections.disconnect("default")
