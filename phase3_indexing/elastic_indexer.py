"""
elastic_indexer.py
===================
Phase 3 - Indexing Module (Member 3 - Backend & DB Administrator)

Chịu trách nhiệm:
    - Tạo Elasticsearch index với `keyframe_id` (keyword) + `ocr_text` /
      `asr_transcript` (text).
    - Cấu hình analyzer song ngữ Việt/Anh + similarity BM25 (k1, b tùy chỉnh).
    - Nạp dữ liệu văn bản thị giác (OCR) và thoại (ASR) qua bulk API.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from config import ElasticConfig
from contracts import MultimodalMetadata

logger = logging.getLogger(__name__)

VI_STOPWORDS = [
    "và", "là", "của", "có", "được", "trong", "đã", "cho", "một",
    "những", "các", "này", "đó", "với", "để", "khi", "không", "về",
]


class ElasticIndexer:
    """Quản lý Elasticsearch index cho dữ liệu văn bản thị giác/thoại."""

    def __init__(
        self,
        cfg: Optional[ElasticConfig] = None,
        client: Any = None,
        bulk_fn: Any = None,
    ):
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
                            "folded": {
                                "type": "text",
                                "analyzer": "vi_en_folded_analyzer",
                            }
                        },
                    },
                    "asr_transcript": {
                        "type": "text",
                        "analyzer": "vi_en_analyzer",
                        "similarity": "vi_bm25",
                        "fields": {
                            "folded": {
                                "type": "text",
                                "analyzer": "vi_en_folded_analyzer",
                            }
                        },
                    },
                }
            },
        }

        self._client.indices.create(index=self.cfg.index_name, body=body)
        logger.info("Đã tạo index '%s'.", self.cfg.index_name)

    # ------------------------------------------------------------------ #
    def insert(self, records: Sequence[MultimodalMetadata]) -> int:
        if not records:
            return 0
        if self._client is None:
            self.connect()

        actions = [
            {
                "_index": self.cfg.index_name,
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
        logger.info(
            "Đã insert %d document vào Elasticsearch (chưa refresh).", success
        )
        return success

    def refresh(self) -> None:
        if self._client is None:
            self.connect()
        if hasattr(self._client, "indices") and hasattr(
            self._client.indices, "refresh"
        ):
            self._client.indices.refresh(index=self.cfg.index_name)
            logger.info("Đã refresh index '%s'.", self.cfg.index_name)

    def count(self) -> int:
        if self._client is None:
            self.connect()
        result = self._client.count(index=self.cfg.index_name)
        if isinstance(result, dict):
            return result.get("count", 0)
        return getattr(result, "count", 0)

    def get_all_ids(
        self, page_size: int = 5000, scroll_ttl: str = "2m"
    ) -> List[str]:
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
        if self._client is None:
            self.connect()
        if hasattr(self._client, "docs"):
            return self._client.docs.get(self.cfg.index_name, {}).get(
                keyframe_id
            )
        resp = self._client.get(index=self.cfg.index_name, id=keyframe_id)
        return (
            resp.get("_source") if isinstance(resp, dict) else resp["_source"]
        )

    # ------------------------------------------------------------------ #
    def search(self, query_text: str, top_k: int = 10):
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
        resp = self._client.search(
            index=self.cfg.index_name, body=body, size=top_k
        )
        hits = (
            resp["hits"]["hits"]
            if isinstance(resp, dict)
            else resp.hits.hits
        )
        return [
            {
                "keyframe_id": h["_source"]["keyframe_id"],
                "score": (
                    h.get("_score", 0.0)
                    if isinstance(h, dict)
                    else h._score
                ),
            }
            for h in hits
        ]
