"""Phase 5: Submission exporter — packages results into submission.zip.

Deliverable: export_submission.py
Entry point: python -m phase5_qa_ui.export_submission
    --results_path  Path to results.json (List[RetrievalResult])
    --output_dir    Output directory for submission.zip

Steps:
  1. Read results.json
  2. Run csv_formatter to generate results.csv
  3. Validate CSV format (no .mp4 suffix, correct frame_ids format, quoted Q&A answers)
  4. Pack results.csv into submission/ subdirectory
  5. Compress as submission/submission.zip

Output: submission/submission.zip compliant with CodaBench format
"""
