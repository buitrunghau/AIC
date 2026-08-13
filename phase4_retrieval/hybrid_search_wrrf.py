"""Phase 4: Weighted Reciprocal Rank Fusion (WRRF) hybrid search.

Deliverable: hybrid_search_wrrf.py
- Merges ranked results from Milvus (dense/vision) and Elasticsearch (BM25/text).
- Formula: WRRF(q, d) = alpha_d / (r_text(q,d) + k) + (1 - alpha_d) / (r_vision(q,d) + k)
  where alpha_d is the text weight (default 0.6), k is smoothing constant (default 60).
- Returns Top-K merged candidates for KIS and Q&A queries.
"""
