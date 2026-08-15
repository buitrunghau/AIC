"""Phase 3: Dense/sparse indexing placeholders.
milvus_indexer.py
==================
Phase 3 - Indexing Module (Member 3 - Backend & DB Administrator)

Chịu trách nhiệm:
    - Tạo Milvus collection với `keyframe_id` là khóa chính (VARCHAR).
    - Build HNSW index trên trường dense_vector để ANN search siêu tốc.
    - Nạp dense vectors (SigLIP2/BEiT-3) từ `MultimodalMetadata`.

Thiết kế:
    Client `pymilvus` thật được import LAZY (chỉ khi thực sự kết nối tới
    Milvus), để module này:
      1) import được ngay cả khi chưa cài `pymilvus` trong môi trường dev/CI.
      2) test được bằng cách inject một fake collection object (xem
         tests/fakes.py) mà không cần một Milvus server thật.
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
        """
        Args:
            cfg: MilvusConfig. Mặc định đọc từ biến môi trường.
            client: (test-only) một object giả lập Collection của pymilvus.
                Khi được cung cấp, mọi thao tác sẽ dùng object này thay vì
                kết nối Milvus thật -> cho phép unit test không cần server.
        """
        self.cfg = cfg or MilvusConfig()
        self._collection = None
        self._injected_client = client
        self._loaded = False  # theo dõi trạng thái đã load() lên memory hay chưa

    # ------------------------------------------------------------------ #
    # Kết nối & khởi tạo schema
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        if self._injected_client is not None:
            logger.info("MilvusIndexer: dùng injected client (test mode).")
            return

        from pymilvus import connections

        connections.connect(alias="default", host=self.cfg.host, port=self.cfg.port)
        logger.info("Đã kết nối Milvus tại %s:%s", self.cfg.host, self.cfg.port)

    def create_collection(self, recreate: bool = False):
        """Tạo collection với PK = keyframe_id (VARCHAR) + vector field."""
        if self._injected_client is not None:
            self._collection = self._injected_client
            if recreate and hasattr(self._collection, "drop"):
                self._collection.drop()
            return self._collection

        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

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
            description="Dense vectors (SigLIP2/BEiT-3) cho video keyframes",
        )
        self._collection = Collection(name=self.cfg.collection_name, schema=schema)
        logger.info(
            "Đã tạo collection '%s' (dim=%d).", self.cfg.collection_name, self.cfg.vector_dim
        )
        return self._collection

    def build_index(self) -> None:
        """Tạo HNSW index trên trường vector (M, efConstruction cấu hình được)."""
        index_params = {
            "index_type": self.cfg.index_type,  # "HNSW"
            "metric_type": self.cfg.metric_type,  # "COSINE" | "IP" | "L2"
            "params": {
                "M": self.cfg.hnsw_m,
                "efConstruction": self.cfg.hnsw_ef_construction,
            },
        }
        if self._collection is None:
            raise RuntimeError("Gọi create_collection() trước khi build_index().")
        self._collection.create_index(field_name=self.VECTOR_FIELD, index_params=index_params)
        logger.info("Đã build HNSW index: %s", index_params)

    # ------------------------------------------------------------------ #
    # Nạp dữ liệu
    # ------------------------------------------------------------------ #
    def insert(self, records: Sequence[MultimodalMetadata]) -> int:
        """
        Nạp danh sách MultimodalMetadata vào Milvus. Trả về số bản ghi đã insert.

        LƯU Ý: hàm này KHÔNG tự động gọi flush(). Nếu dữ liệu được nạp theo
        nhiều batch (thường gặp với dataset lớn), gọi flush() sau MỖI batch
        sẽ ép Milvus seal nhiều segment nhỏ (small segments), làm tăng chi
        phí compaction và giảm hiệu năng truy vấn. Hãy gọi `flush()` một
        lần duy nhất sau khi toàn bộ batch đã được insert xong (xem
        `index_data.run_pipeline`).
        """
        if not records:
            return 0
        if self._collection is None:
            raise RuntimeError("Collection chưa được khởi tạo. Gọi create_collection() trước.")

        self._validate_dim(records)

        ids = [r.keyframe_id for r in records]
        vectors = [r.dense_vector for r in records]

        self._collection.insert([ids, vectors])
        logger.info("Đã insert %d bản ghi vào Milvus (chưa flush).", len(ids))
        return len(ids)

    def flush(self) -> None:
        """
        Seal dữ liệu đã insert xuống segment persistent. Chỉ nên gọi MỘT
        LẦN sau khi toàn bộ batch của một lượt nạp đã insert xong, không
        gọi sau mỗi batch để tránh tạo nhiều segment nhỏ.
        """
        if self._collection is not None and hasattr(self._collection, "flush"):
            self._collection.flush()
            logger.info("Đã flush collection '%s'.", self.cfg.collection_name)

    def load(self) -> None:
        """Load collection lên memory để sẵn sàng phục vụ truy vấn (Phase 4)."""
        if self._collection is not None and hasattr(self._collection, "load"):
            self._collection.load()
        self._loaded = True

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.num_entities

    def get_all_ids(self, batch_size: int = 10000) -> List[str]:
        """
        Tiện ích cho test/verify (đối chiếu keyframe_id với Elasticsearch ở
        quy mô nhỏ/vừa): lấy toàn bộ keyframe_id đã index.

        `query()` của Milvus mặc định giới hạn số bản ghi trả về một lần gọi
        (~16384). Với dataset lớn (hàng trăm nghìn/hàng triệu keyframe), hàm
        này phân trang bằng `offset`/`limit` để không bị thiếu dữ liệu. Tuy
        nhiên với dataset thực sự lớn, nên dùng `query_iterator` của pymilvus
        hoặc chỉ chạy verify này trên một mẫu (sample) thay vì toàn bộ.
        """
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

    def search(self, query_vector: List[float], top_k: int = 10, ef: Optional[int] = None):
        """
        ANN search - phục vụ Phase 4 (Retrieval & Alignment Module).

        Tự động gọi load() nếu collection chưa được load lên memory, để
        tránh lỗi "collection not loaded" từ phía Milvus server khi người
        dùng quên gọi load() trước.
        """
        if not self._loaded:
            logger.warning(
                "Collection '%s' chưa được load() — tự động load trước khi search.",
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
                    f"keyframe_id={r.keyframe_id}: dense_vector có {len(r.dense_vector)} "
                    f"chiều, kỳ vọng {self.cfg.vector_dim} (xem MilvusConfig.vector_dim)."
                )

    def close(self) -> None:
        if self._injected_client is not None:
            return
        from pymilvus import connections

        connections.disconnect("default")
        