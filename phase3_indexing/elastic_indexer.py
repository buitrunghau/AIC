"""Phase 3: Elasticsearch indexer for sparse/text (OCR) data.

Deliverable: elastic_indexer.py
- Creates Elasticsearch index with fields: keyframe_id, video_id, ocr_text.
- Configures BM25 analyzer with Vietnamese/English language support.
- Indexes MultimodalMetadata.ocr_text for exact lexical matching.
elastic_indexer.py
===================
Phase 3 - Indexing Module (Member 3 - Backend & DB Administrator)

Chịu trách nhiệm:
    - Tạo Elasticsearch index với `keyframe_id` (keyword) + `ocr_text` /
      `asr_transcript` (text).
    - Cấu hình analyzer song ngữ Việt/Anh + similarity BM25 (k1, b tùy chỉnh).
    - Nạp dữ liệu văn bản thị giác (OCR) và thoại (ASR) qua bulk API.

Ghi chú về analyzer tiếng Việt:
    Elasticsearch KHÔNG có analyzer "vietnamese" built-in (không nằm trong
    danh sách language analyzer mặc định). Vì tiếng Việt dùng ký tự Latin
    có dấu và phân tách bằng khoảng trắng ở cấp âm tiết, ta dùng:
      - `standard` tokenizer (tách theo khoảng trắng/dấu câu, hoạt động ổn
        cho cả tiếng Việt lẫn tiếng Anh).
      - `lowercase` + stopword filter riêng cho vi/en.
      - Một sub-field `.folded` dùng `asciifolding` để hỗ trợ tìm kiếm
        không dấu (UX phổ biến khi người dùng gõ tiếng Việt không dấu).
    Nếu cần tách từ ghép tiếng Việt chính xác hơn (ví dụ "bệnh viện" là 1
    từ chứ không phải 2 âm tiết rời), nên tiền xử lý bằng underthesea/pyvi
    ở Phase 2 trước khi đưa `ocr_text` sang đây, hoặc cài plugin ICU.

Thiết kế:
    Client `elasticsearch` thật và hàm `bulk()` được inject được (dependency
    injection) để unit test không cần một Elasticsearch server thật.
"""
"""
elastic_indexer.py
===================
Phase 3 - Indexing Module (Member 3 - Backend & DB Administrator)

Chịu trách nhiệm:
    - Tạo Elasticsearch index với `keyframe_id` (keyword) + `ocr_text` /
      `asr_transcript` (text).
    - Cấu hình analyzer song ngữ Việt/Anh + similarity BM25 (k1, b tùy chỉnh).
    - Nạp dữ liệu văn bản thị giác (OCR) và thoại (ASR) qua bulk API.

Ghi chú về analyzer tiếng Việt:
    Elasticsearch KHÔNG có analyzer "vietnamese" built-in (không nằm trong
    danh sách language analyzer mặc định). Vì tiếng Việt dùng ký tự Latin
    có dấu và phân tách bằng khoảng trắng ở cấp âm tiết, ta dùng:
      - `standard` tokenizer (tách theo khoảng trắng/dấu câu, hoạt động ổn
        cho cả tiếng Việt lẫn tiếng Anh).
      - `lowercase` + stopword filter riêng cho vi/en.
      - Một sub-field `.folded` dùng `asciifolding` để hỗ trợ tìm kiếm
        không dấu (UX phổ biến khi người dùng gõ tiếng Việt không dấu).
    Nếu cần tách từ ghép tiếng Việt chính xác hơn (ví dụ "bệnh viện" là 1
    từ chứ không phải 2 âm tiết rời), nên tiền xử lý bằng underthesea/pyvi
    ở Phase 2 trước khi đưa `ocr_text` sang đây, hoặc cài plugin ICU.

Thiết kế:
    Client `elasticsearch` thật và hàm `bulk()` được inject được (dependency
    injection) để unit test không cần một Elasticsearch server thật.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from config import ElasticConfig
from contracts import MultimodalMetadata

logger = logging.getLogger(__name__)

# Stopwords tối giản cho tiếng Việt - có thể mở rộng thêm theo domain dữ liệu.
VI_STOPWORDS = [
    "và", "là", "của", "có", "được", "trong", "đã", "cho", "một",
    "những", "các", "này", "đó", "với", "để", "khi", "không", "về",
]


class ElasticIndexer:
    """Quản lý Elasticsearch index cho dữ liệu văn bản thị giác/thoại (OCR/ASR)."""

    def __init__(
        self,
        cfg: Optional[ElasticConfig] = None,
        client: Any = None,
        bulk_fn: Any = None,
    ):
        """
        Args:
            cfg: ElasticConfig. Mặc định đọc từ biến môi trường.
            client: (test-only) fake Elasticsearch client. Khi cung cấp,
                mọi thao tác dùng object này thay vì kết nối ES thật.
            bulk_fn: (test-only) fake thay thế cho `elasticsearch.helpers.bulk`,
                chữ ký `fn(client, actions, refresh=True) -> (success, errors)`.
        """
        self.cfg = cfg or ElasticConfig()
        self._client = client
        self._bulk_fn = bulk_fn

    # ------------------------------------------------------------------ #
    def connect(self):
        if self._client is not None:
            logger.info("ElasticIndexer: dùng injected client (test mode).")
            return self._client

        from elasticsearch import Elasticsearch

        self._client = Elasticsearch(self.cfg.hosts)
        logger.info("Đã kết nối Elasticsearch tại %s", self.cfg.hosts)
        return self._client

    def _get_bulk_fn(self):
        if self._bulk_fn is not None:
            return self._bulk_fn
        from elasticsearch.helpers import bulk

        return bulk

    def create_index(self, recreate: bool = False) -> None:
        if self._client is None:
            self.connect()

        if self._client.indices.exists(index=self.cfg.index_name):
            if recreate:
                self._client.indices.delete(index=self.cfg.index_name)
            else:
                return

        body = {
            "settings": {
                "number_of_shards": self.cfg.shards,
                "number_of_replicas": self.cfg.replicas,
                "similarity": {
                    "vi_bm25": {
                        "type": "BM25",
                        "k1": self.cfg.bm25_k1,
                        "b": self.cfg.bm25_b,
                    }
                },
                "analysis": {
                    "filter": {
                        "vi_stop": {"type": "stop", "stopwords": VI_STOPWORDS},
                        "en_stop": {"type": "stop", "stopwords": "_english_"},
                    },
                    "analyzer": {
                        "vi_en_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "vi_stop", "en_stop"],
                        },
                        "vi_en_folded_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"],
                        },
                    },
                },
            },
            "mappings": {
                "properties": {
                    "keyframe_id": {"type": "keyword"},
                    "ocr_text": {
                        "type": "text",
                        "analyzer": "vi_en_analyzer",
                        "similarity": "vi_bm25",
                        "fields": {
                            "folded": {"type": "text", "analyzer": "vi_en_folded_analyzer"}
                        },
                    },
                    "asr_transcript": {
                        "type": "text",
                        "analyzer": "vi_en_analyzer",
                        "similarity": "vi_bm25",
                        "fields": {
                            "folded": {"type": "text", "analyzer": "vi_en_folded_analyzer"}
                        },
                    },
                }
            },
        }

        self._client.indices.create(index=self.cfg.index_name, body=body)
        logger.info("Đã tạo index '%s'.", self.cfg.index_name)

    # ------------------------------------------------------------------ #
    def insert(self, records: Sequence[MultimodalMetadata]) -> int:
        """
        Nạp danh sách MultimodalMetadata vào Elasticsearch qua bulk API.

        LƯU Ý: hàm này KHÔNG refresh index sau mỗi lần gọi (refresh=False).
        Nếu dữ liệu được nạp theo nhiều batch, gọi refresh sau MỖI batch sẽ
        ép Elasticsearch commit segment liên tục xuống đĩa, làm chậm tốc độ
        nạp đáng kể. Hãy gọi `refresh()` một lần duy nhất sau khi toàn bộ
        batch đã insert xong (xem `index_data.run_pipeline`). Elasticsearch
        vẫn tự refresh theo chu kỳ mặc định (1s) nên dữ liệu sẽ khả kiến
        (searchable) sau đó ít lâu dù không gọi refresh thủ công.
        """
        if not records:
            return 0
        if self._client is None:
            self.connect()

        actions = [
            {
                "_index": self.cfg.index_name,
                # Dùng keyframe_id làm _id -> đồng bộ 1-1 với Milvus PK,
                # tránh sinh ID ngẫu nhiên gây lệch giữa 2 hệ thống.
                "_id": r.keyframe_id,
                "_source": {
                    "keyframe_id": r.keyframe_id,
                    "ocr_text": r.ocr_text or "",
                    "asr_transcript": r.asr_transcript or "",
                },
            }
            for r in records
        ]

        bulk_fn = self._get_bulk_fn()
        success, errors = bulk_fn(self._client, actions, refresh=False)
        if errors:
            logger.warning("Elasticsearch bulk insert có %d lỗi.", len(errors))
        logger.info("Đã insert %d document vào Elasticsearch (chưa refresh).", success)
        return success

    def refresh(self) -> None:
        """
        Refresh index để dữ liệu vừa insert khả kiến (searchable) ngay lập
        tức. Chỉ nên gọi MỘT LẦN sau khi toàn bộ batch của một lượt nạp đã
        insert xong, không gọi sau mỗi batch để tránh commit segment liên tục.
        """
        if self._client is None:
            self.connect()
        if hasattr(self._client, "indices") and hasattr(self._client.indices, "refresh"):
            self._client.indices.refresh(index=self.cfg.index_name)
            logger.info("Đã refresh index '%s'.", self.cfg.index_name)

    def count(self) -> int:
        if self._client is None:
            self.connect()
        result = self._client.count(index=self.cfg.index_name)
        if isinstance(result, dict):
            return result.get("count", 0)
        return getattr(result, "count", 0)

    def get_all_ids(self, page_size: int = 5000, scroll_ttl: str = "2m") -> List[str]:
        """
        Tiện ích cho test/verify (đối chiếu keyframe_id với Milvus): lấy
        toàn bộ keyframe_id đã index.

        Elasticsearch chặn truy vấn `search(size=...)` vượt quá
        `index.max_result_window` (mặc định 10000) — dùng `size=10000` như
        trước đây là chạm ngay ngưỡng này và sẽ lỗi/thiếu dữ liệu khi kho
        video có trên 10.000 keyframe. Vì vậy hàm này dùng Scroll API để
        duyệt hết toàn bộ document theo trang mà không bị giới hạn đó.
        """
        if self._client is None:
            self.connect()
        if hasattr(self._client, "get_all_ids"):
            return self._client.get_all_ids(self.cfg.index_name)

        all_ids: List[str] = []
        resp = self._client.search(
            index=self.cfg.index_name,
            body={"query": {"match_all": {}}, "_source": ["keyframe_id"]},
            scroll=scroll_ttl,
            size=page_size,
        )
        scroll_id = resp.get("_scroll_id")
        hits = resp["hits"]["hits"]
        while hits:
            all_ids.extend(h["_source"]["keyframe_id"] for h in hits)
            resp = self._client.scroll(scroll_id=scroll_id, scroll=scroll_ttl)
            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]

        if scroll_id and hasattr(self._client, "clear_scroll"):
            self._client.clear_scroll(scroll_id=scroll_id)

        return all_ids

    def get_document(self, keyframe_id: str):
        """Lấy 1 document theo keyframe_id (tiện ích cho test/verify)."""
        if self._client is None:
            self.connect()
        if hasattr(self._client, "docs"):
            return self._client.docs.get(self.cfg.index_name, {}).get(keyframe_id)
        resp = self._client.get(index=self.cfg.index_name, id=keyframe_id)
        return resp.get("_source") if isinstance(resp, dict) else resp["_source"]

    # ------------------------------------------------------------------ #
    def search(self, query_text: str, top_k: int = 10):
        """
        BM25 lexical search - phục vụ Phase 4 (Retrieval & Alignment Module),
        cụ thể là nhánh sparse/text của WRRF (Weighted Reciprocal Rank
        Fusion) khi hợp nhất với kết quả dense-vector từ Milvus.

        Truy vấn `multi_match` qua cả `ocr_text`/`asr_transcript` (bản có
        dấu, đã qua stopword filter) lẫn subfield `.folded` (bản không dấu,
        hỗ trợ người dùng gõ tiếng Việt không dấu). Field có dấu được boost
        cao hơn vì mang ngữ nghĩa chính xác hơn field folded.

        Returns:
            List[dict] dạng [{"keyframe_id": ..., "score": ...}, ...],
            sắp xếp giảm dần theo BM25 score.
        """
        if self._client is None:
            self.connect()

        body = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "ocr_text^2",
                        "asr_transcript^2",
                        "ocr_text.folded",
                        "asr_transcript.folded",
                    ],
                    "type": "best_fields",
                }
            },
            "_source": ["keyframe_id"],
        }
        resp = self._client.search(index=self.cfg.index_name, body=body, size=top_k)
        hits = resp["hits"]["hits"] if isinstance(resp, dict) else resp.hits.hits
        return [
            {
                "keyframe_id": h["_source"]["keyframe_id"],
                "score": h.get("_score", 0.0) if isinstance(h, dict) else h._score,
            }
            for h in hits
        ]
