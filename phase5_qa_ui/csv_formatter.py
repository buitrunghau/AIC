"""Phase 5: CSV formatter for CodaBench submission format.

Deliverable: csv_formatter.py
- Converts List[RetrievalResult] to CodaBench-compliant CSV rows.
- Strips .mp4 suffix from video_id if present.
- Formats frame_ids as space-separated string (e.g., "1050 1080 1120").
- Wraps answer in double-quotes for Q&A entries (e.g., '"59A-123.45"').
- Uses query_type field from RetrievalResult to determine output format.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence, Union

from shared_contracts.contracts import RetrievalResult


def _coerce_result(item: Union[RetrievalResult, dict]) -> RetrievalResult:
    """Normalize a retrieval item into a `RetrievalResult` instance."""
    if isinstance(item, RetrievalResult):
        return item

    if isinstance(item, dict):
        return RetrievalResult(
            video_id=str(item.get("video_id", "")),
            frame_ids=list(item.get("frame_ids", []) or []),
            query_type=str(item.get("query_type", "KIS")),
            answer=item.get("answer"),
            wrrf_score=float(item.get("wrrf_score", 0.0)),
        )

    raise TypeError(f"Unsupported result type: {type(item)!r}")


def _normalize_query_type(query_type: str) -> str:
    return str(query_type or "").strip().upper()

def normalize_video_id(video_id: str) -> str:
    """Strip unnecessary video suffixes for submission output."""
    if video_id is None:
        return ""

    normalized = str(video_id).strip()
    if normalized.lower().endswith(".mp4"):
        normalized = normalized[:-4]
    return normalized

def _format_row_for_result(result: RetrievalResult) -> List[str]:
    """Build the exact CSV row according to the query_type rules.

    KIS:     <video_id>,<frame_id>
    Q&A:    <video_id>,<frame_id>,"<answer>"
    TRAKE:  <video_id>,<frame_id_1>,<frame_id_2>,...,<frame_id_N>
    """
    video_id = normalize_video_id(result.video_id)
    frame_ids = [int(frame_id) for frame_id in (result.frame_ids or []) if frame_id is not None]

    if _normalize_query_type(result.query_type) == "Q&A":
        frame_id = str(frame_ids[0]) if frame_ids else ""
        answer = "" if result.answer is None else str(result.answer)
        return [video_id, frame_id, answer]

    if _normalize_query_type(result.query_type) == "TRAKE":
        if not frame_ids:
            return [video_id]
        return [video_id, *[str(frame_id) for frame_id in frame_ids]]

    if frame_ids:
        return [video_id, str(frame_ids[0])]
    return [video_id]


def results_to_rows(results: Sequence[Union[RetrievalResult, dict]]) -> List[List[str]]:
    """Convert a list of retrieval results into raw CSV rows without headers."""
    rows: List[List[str]] = []
    for item in results:
        result = _coerce_result(item)
        rows.append(_format_row_for_result(result))
    return rows


def write_csv(results: Sequence[Union[RetrievalResult, dict]], output_path: Union[str, Path]) -> Path:
    """Write only data rows, without a CSV header row.

    The number of columns is intentionally variable depending on query_type.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = results_to_rows(results)

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for row in rows:
            writer.writerow(row)

    return output


def load_results_from_json(path: Union[str, Path]) -> List[RetrievalResult]:
    """Load a JSON file containing retrieval results."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return [_coerce_result(item) for item in payload]

    if isinstance(payload, dict) and "retrieval_results" in payload:
        return [_coerce_result(item) for item in payload["retrieval_results"]]

    raise ValueError("JSON payload must be a list or contain a 'retrieval_results' key.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CodaBench CSV from retrieval results.")
    parser.add_argument("--results_path", required=True, help="Path to the JSON file with retrieval results.")
    parser.add_argument("--output_path", required=True, help="Destination CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results_from_json(args.results_path)
    output = write_csv(results, args.output_path)
    print(f"CSV written to: {output}")


if __name__ == "__main__":
    main()
