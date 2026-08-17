"""Placeholder test module for phase 5."""


def test_phase5_placeholder() -> None:
    assert True

import numpy as np
from shared_contracts.contracts import RetrievalResult
# Cập nhật đường dẫn import từ csv_formatter
from phase5_qa_ui.csv_formatter import normalize_video_id, _format_row_for_result
from phase5_qa_ui.rocchio_feedback import apply_rocchio_feedback


def test_normalize_video_id_strips_mp4_suffix() -> None:
    assert normalize_video_id("L01_V025.mp4") == "L01_V025"
    assert normalize_video_id("L02_V030.MP4") == "L02_V030"
    assert normalize_video_id("L03_V001") == "L03_V001"


def test_csv_rows_follow_organizer_format() -> None:
    from phase5_qa_ui.csv_formatter import _format_row_for_result

    kis = RetrievalResult(video_id="L00_V000.mp4", frame_ids=[1234], query_type="KIS", answer=None, wrrf_score=1.0)
    qa = RetrievalResult(video_id="L01_V028.mp4", frame_ids=[3450, 4000], query_type="Q&A", answer='Anh ấy nói "Tuyệt vời"', wrrf_score=1.0)
    trake = RetrievalResult(video_id="L10_V001.mp4", frame_ids=[1200, 1850, 2100, 2450], query_type="TRAKE", answer=None, wrrf_score=1.0)

    assert _format_row_for_result(kis) == ["L00_V000", "1234"]
    assert _format_row_for_result(qa) == ["L01_V028", "3450", 'Anh ấy nói "Tuyệt vời"']
    assert _format_row_for_result(trake) == ["L10_V001", "1200", "1850", "2100", "2450"]


def test_retrieval_result_contract_keeps_required_fields() -> None:
    result = RetrievalResult(
        video_id="L01_V025.mp4",
        frame_ids=[1050, 1080, 1120],
        query_type="TRAKE",
        answer=None,
        wrrf_score=0.942,
    )

    assert result.video_id == "L01_V025.mp4"
    assert result.frame_ids == [1050, 1080, 1120]
    assert result.query_type == "TRAKE"
    assert result.wrrf_score == 0.942