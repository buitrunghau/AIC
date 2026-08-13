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
"""
