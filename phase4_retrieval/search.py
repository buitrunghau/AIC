"""Phase 4: Core retriever interface — entry point for Phase 4.

Deliverable: search.py
Entry point: python -m phase4_retrieval.search
    --query       Text query string
    --query_type  Query type: KIS, Q&A, or TRAKE
    --top_k       Number of top results to return (default: 100)

Routes query to:
  - hybrid_search_wrrf.py  for KIS and Q&A
  - dante_trake_solver.py  for TRAKE

Output: List[RetrievalResult] with query_type, video_id, frame_ids, answer, wrrf_score
"""
