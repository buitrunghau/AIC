"""Phase 3: Main pipeline script for end-to-end indexing into Milvus and Elasticsearch.

Deliverable: index_data.py
Entry point: python -m phase3_indexing.index_data
    --metadata_path   Path to MultimodalMetadata output from Phase 2
    --milvus_host     Milvus server host (default: localhost)
    --milvus_port     Milvus server port (default: 19530)
    --es_host         Elasticsearch host (default: localhost)
    --es_port         Elasticsearch port (default: 9200)

Output: Indexed data in Milvus (dense vectors) and Elasticsearch (OCR text),
        linked by keyframe_id primary key.

        
index_data.py
=============
Phase 3 - Indexing Module - Pipeline chính (end-to-end)

Luồng xử lý:

    MultimodalMetadata[] (nhận từ Phase 2)
            |
            +--> MilvusIndexer   (keyframe_id, dense_vector) --> HNSW index
            +--> ElasticIndexer  (keyframe_id, ocr_text, asr_transcript) --> BM25
            |
            +--> verify_keyframe_sync(): đối chiếu keyframe_id giữa 2 hệ thống
            |
            +--> report (dict) theo đúng format Output ở mục 2

Sử dụng CLI:
    python index_data.py --input sample_data.json [--recreate]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from config import ElasticConfig, MilvusConfig
from contracts import MultimodalMetadata
from elastic_indexer import ElasticIndexer
from milvus_indexer import MilvusIndexer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("index_data")


def load_metadata(path: str) -> List[MultimodalMetadata]:
    """Đọc danh sách MultimodalMetadata từ file JSON (output của Phase 2)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        MultimodalMetadata(
            keyframe_id=item["keyframe_id"],
            dense_vector=item["dense_vector"],
            ocr_text=item.get("ocr_text", ""),
            asr_transcript=item.get("asr_transcript", ""),
        )
        for item in raw
    ]


def verify_keyframe_sync(
    milvus: MilvusIndexer, es: ElasticIndexer, records: List[MultimodalMetadata]
) -> bool:
    """
    Đối chiếu keyframe_id giữa Milvus và Elasticsearch để đảm bảo 2 hệ thống
    liên kết chặt chẽ (yêu cầu cốt lõi của Phase 3), phục vụ việc hợp nhất
    kết quả (WRRF) ở Phase 4.
    """
    expected_ids = {r.keyframe_id for r in records}
    milvus_ids = set(milvus.get_all_ids())
    es_ids = set(es.get_all_ids())

    ok = True
    missing_in_milvus = expected_ids - milvus_ids
    missing_in_es = expected_ids - es_ids
    if missing_in_milvus:
        logger.warning(
            "%d keyframe_id thiếu trong Milvus: %s",
            len(missing_in_milvus),
            list(missing_in_milvus)[:5],
        )
        ok = False
    if missing_in_es:
        logger.warning(
            "%d keyframe_id thiếu trong Elasticsearch: %s",
            len(missing_in_es),
            list(missing_in_es)[:5],
        )
        ok = False
    if milvus_ids != es_ids:
        logger.warning("keyframe_id giữa Milvus và Elasticsearch KHÔNG khớp nhau.")
        ok = False

    if ok:
        logger.info("Đồng bộ keyframe_id OK: %d bản ghi khớp ở cả 2 hệ thống.", len(expected_ids))
    return ok


def run_pipeline(
    records: List[MultimodalMetadata],
    milvus_cfg: Optional[MilvusConfig] = None,
    es_cfg: Optional[ElasticConfig] = None,
    recreate: bool = False,
    milvus_indexer: Optional[MilvusIndexer] = None,
    es_indexer: Optional[ElasticIndexer] = None,
) -> dict:
    """
    Thực thi toàn bộ Phase 3: nạp `records` vào Milvus + Elasticsearch,
    verify đồng bộ, trả về report theo đúng format Output ở mục 2.

    `milvus_indexer` / `es_indexer` cho phép inject sẵn instance (dùng cho
    test hoặc khi caller muốn tái sử dụng kết nối đã mở).
    """
    milvus = milvus_indexer or MilvusIndexer(milvus_cfg)
    es = es_indexer or ElasticIndexer(es_cfg)

    # ---- Milvus: dense vectors ----
    milvus.connect()
    milvus.create_collection(recreate=recreate)
    milvus.build_index()
    milvus.insert(records)
    # flush() chỉ gọi 1 LẦN sau khi toàn bộ batch đã insert xong, tránh tạo
    # nhiều segment nhỏ nếu sau này records được insert theo nhiều batch.
    milvus.flush()
    milvus.load()

    # ---- Elasticsearch: sparse/text ----
    es.connect()
    es.create_index(recreate=recreate)
    es.insert(records)
    # refresh() chỉ gọi 1 LẦN sau khi toàn bộ batch đã insert xong, tránh
    # ép Elasticsearch commit segment liên tục nếu sau này records được
    # insert theo nhiều batch.
    es.refresh()

    # ---- Verify liên kết keyframe_id ----
    verify_keyframe_sync(milvus, es, records)

    report = {
        "milvus_collection": {
            "name": milvus.cfg.collection_name,
            "records": milvus.count(),
            "sample": (
                {
                    "keyframe_id": records[0].keyframe_id,
                    "dense_vector_dim": len(records[0].dense_vector),
                }
                if records
                else None
            ),
        },
        "elasticsearch_index": {
            "name": es.cfg.index_name,
            "documents": es.count(),
            "sample": (
                {"keyframe_id": records[0].keyframe_id, "ocr_text": records[0].ocr_text}
                if records
                else None
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 - Indexing pipeline (Milvus + Elasticsearch)")
    parser.add_argument("--input", required=True, help="Đường dẫn JSON chứa MultimodalMetadata[]")
    parser.add_argument("--recreate", action="store_true", help="Xóa và tạo lại collection/index")
    args = parser.parse_args()

    records = load_metadata(args.input)
    logger.info("Đã load %d bản ghi MultimodalMetadata từ %s", len(records), args.input)

    report = run_pipeline(records, recreate=args.recreate)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()