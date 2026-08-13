"""Phase 5: CSV formatter for CodaBench submission format.

Deliverable: csv_formatter.py
- Converts List[RetrievalResult] to CodaBench-compliant CSV rows.
- Strips .mp4 suffix from video_id if present.
- Formats frame_ids as space-separated string (e.g., "1050 1080 1120").
- Wraps answer in double-quotes for Q&A entries (e.g., '"59A-123.45"').
- Uses query_type field from RetrievalResult to determine output format.
"""
