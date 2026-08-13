"""Phase 3: Elasticsearch indexer for sparse/text (OCR) data.

Deliverable: elastic_indexer.py
- Creates Elasticsearch index with fields: keyframe_id, video_id, ocr_text.
- Configures BM25 analyzer with Vietnamese/English language support.
- Indexes MultimodalMetadata.ocr_text for exact lexical matching.
"""
