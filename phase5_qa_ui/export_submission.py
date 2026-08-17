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
import argparse
import csv
import zipfile
from pathlib import Path
from typing import Any, List, Sequence, Union

from phase5_qa_ui.csv_formatter import load_results_from_json, write_csv, normalize_video_id
from shared_contracts.contracts import RetrievalResult


def _normalize_query_type(query_type: str) -> str:
    return str(query_type or "").strip().upper()


def _validate_result(result: RetrievalResult) -> None:
    """Validate a retrieval result before packaging."""
    if not result.video_id:
        raise ValueError("video_id cannot be empty.")

    normalized_video_id = normalize_video_id(result.video_id)
    if normalized_video_id == "":
        raise ValueError(f"video_id is invalid: {result.video_id!r}")

    query_type = _normalize_query_type(result.query_type)
    if query_type not in {"KIS", "Q&A", "TRAKE"}:
        raise ValueError(f"Unsupported query_type: {result.query_type!r}")

    frame_ids = [int(frame_id) for frame_id in (result.frame_ids or []) if frame_id is not None]

    if query_type == "Q&A":
        if not frame_ids:
            raise ValueError(f"Q&A result must have at least one frame_id: {result!r}")
        if result.answer is None:
            raise ValueError(f"Q&A result is missing answer: {result!r}")
        # THÊM MỚI: Kiểm tra độ dài câu trả lời không vượt quá 100 ký tự
        if len(str(result.answer)) > 100:
            raise ValueError(f"Q&A answer exceeds 100 characters limit (Length: {len(str(result.answer))}): {result.answer!r}")

    if query_type == "KIS" and len(frame_ids) != 1:
        raise ValueError(f"KIS result must contain exactly one frame_id: {result!r}")

    if query_type == "TRAKE" and len(frame_ids) < 2:
        raise ValueError(f"TRAKE result must contain at least two frame_ids: {result!r}")


def _validate_csv_file(csv_path: Union[str, Path]) -> None:
    """Validate generated CSV rows against the organizer’s format rules."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    if len(rows) > 100:
        raise ValueError(f"CSV file exceeds the maximum limit of 100 rows. Current row count: {len(rows)}")

    for row in rows:
        if not row:
            continue

        if len(row) == 1:
            # For an unexpected single-value row, reject it because data rows must contain video_id and frame id.
            raise ValueError(f"Invalid CSV row: {row!r}")

        video_id = row[0]
        if not video_id or normalize_video_id(video_id) == "":
            raise ValueError(f"Invalid video_id in CSV row: {row!r}")

        if len(row) == 2:
            # KIS row: <video_id>,<frame_id>
            try:
                int(row[1])
            except ValueError as exc:
                raise ValueError(f"Invalid KIS frame_id in CSV row: {row!r}") from exc
        else:
            # Phân biệt Q&A và TRAKE dựa vào định dạng của cột thứ 3 (row[2])
            try:
                int(row[2])
                is_trake = True
            except ValueError:
                is_trake = False

            if not is_trake:
                # Q&A row: <video_id>,<frame_id>,"<answer>"
                if len(row) > 3:
                    raise ValueError(f"Invalid Q&A row (too many columns): {row!r}")
                try:
                    int(row[1])
                except ValueError as exc:
                    raise ValueError(f"Invalid Q&A frame_id in CSV row: {row!r}") from exc
            else:
                # TRAKE row: <video_id>,<frame_1>,...,<frame_n>
                for frame_id in row[1:]:
                    try:
                        int(frame_id)
                    except ValueError as exc:
                        raise ValueError(f"Invalid TRAKE frame_id in CSV row: {row!r}") from exc

def export_submission(results_path: Union[str, Path], output_dir: Union[str, Path]) -> dict:
    """Export a submission directory and its zip archive from retrieval results."""
    results_path = Path(results_path)
    output_dir = Path(output_dir)
# def export_submission(results_or_path: Union[str, Path, Sequence[RetrievalResult], Sequence[dict]], output_dir: Union[str, Path]) -> dict:
#     """Export a submission directory and its zip archive from retrieval results or a JSON path."""
#     output_dir = Path(output_dir)

    # # KIỂM TRA ĐẦU VÀO: 
    # # Nếu là chuỗi (string/Path) -> Đó là đường dẫn, đi đọc file JSON (Dành cho Terminal)
    # if isinstance(results_or_path, (str, Path)):
    #     results = load_results_from_json(results_or_path)
    # # Nếu là List/Sequence -> Đó là dữ liệu RAM (Dành cho Streamlit)
    # else:
    #     # Gọi _coerce_result (từ csv_formatter) để đảm bảo định dạng chuẩn
    #     from phase5_qa_ui.csv_formatter import _coerce_result
    #     results = [_coerce_result(item) for item in results_or_path]

    results = load_results_from_json(results_path)
    # ... (Các đoạn code validate, write_csv và zipfile bên dưới giữ y nguyên) ...
    for result in results:
        _validate_result(result)
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    csv_path = submission_dir / "results.csv"
    write_csv(results, csv_path)
    _validate_csv_file(csv_path)

    zip_path = submission_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)

    return {
        "csv_file": str(csv_path),
        "zip_file": str(zip_path),
        "submission_dir": str(submission_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CodaBench-ready CSV and zip submission package.")
    parser.add_argument("--results_path", required=True, help="Path to the JSON file containing retrieval results.")
    parser.add_argument("--output_dir", required=True, help="Directory where the submission/ folder will be created.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_submission(args.results_path, args.output_dir)
    print(result)


if __name__ == "__main__":
    main()