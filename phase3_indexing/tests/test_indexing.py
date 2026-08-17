"""
tests/test_indexing.py
=======================
Kiểm tra Phase 3 - Indexing Module:
    1. Số lượng bản ghi insert vào Milvus / Elasticsearch đúng.
    2. Tính nhất quán `keyframe_id` giữa hai DB (yêu cầu cốt lõi mục 1).
    3. Validate dimension của dense_vector.
    4. Nội dung ocr_text được lưu đúng.
    5. Pipeline end-to-end (`index_data.run_pipeline`) trả về report đúng format.

Không cần Milvus/Elasticsearch server thật — dùng fake client (DI) trong
`tests/fakes.py` để test logic của indexer một cách cô lập (unit test).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ElasticConfig, MilvusConfig  # noqa: E402
from contracts import MultimodalMetadata  # noqa: E402
from elastic_indexer import ElasticIndexer  # noqa: E402
from index_data import run_pipeline, verify_keyframe_sync  # noqa: E402
from milvus_indexer import MilvusIndexer  # noqa: E402
from tests.fakes import (  # noqa: E402
    FakeElasticClient,
    FakeElasticScrollClient,
    FakeMilvusCollection,
    fake_bulk,
)

VECTOR_DIM = 8  # dim nhỏ cho test nhanh; production dùng 768/1024


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def sample_records():
    return [
        MultimodalMetadata(
            keyframe_id=f"L01_V025_{1000 + i}",
            dense_vector=[round(0.1 * (i + 1), 3)] * VECTOR_DIM,
            ocr_text="BỆNH VIỆN CHỢ RẪY" if i == 0 else f"sample ocr text {i}",
            asr_transcript=f"asr transcript {i}",
        )
        for i in range(5)
    ]


@pytest.fixture
def milvus_indexer():
    cfg = MilvusConfig(vector_dim=VECTOR_DIM, collection_name="test_video_frames")
    indexer = MilvusIndexer(cfg=cfg, client=FakeMilvusCollection())
    indexer.connect()
    indexer.create_collection()
    indexer.build_index()
    return indexer


@pytest.fixture
def elastic_indexer():
    cfg = ElasticConfig(index_name="test_video_text")
    indexer = ElasticIndexer(cfg=cfg, client=FakeElasticClient(), bulk_fn=fake_bulk)
    indexer.connect()
    indexer.create_index()
    return indexer


# ---------------------------------------------------------------------- #
# 1. Milvus: số lượng bản ghi & dimension validation
# ---------------------------------------------------------------------- #
def test_milvus_insert_record_count(milvus_indexer, sample_records):
    inserted = milvus_indexer.insert(sample_records)
    assert inserted == len(sample_records)
    assert milvus_indexer.count() == len(sample_records)


def test_milvus_build_index_uses_hnsw_params(milvus_indexer):
    params = milvus_indexer._collection.index_params
    assert params["index_type"] == "HNSW"
    assert params["params"]["M"] == milvus_indexer.cfg.hnsw_m
    assert params["params"]["efConstruction"] == milvus_indexer.cfg.hnsw_ef_construction


def test_milvus_rejects_wrong_vector_dimension(milvus_indexer):
    bad_record = [MultimodalMetadata(keyframe_id="bad_1", dense_vector=[0.1, 0.2])]
    with pytest.raises(ValueError):
        milvus_indexer.insert(bad_record)


def test_milvus_insert_without_collection_raises():
    cfg = MilvusConfig(vector_dim=VECTOR_DIM)
    indexer = MilvusIndexer(cfg=cfg, client=FakeMilvusCollection())
    # Chưa gọi create_collection()
    with pytest.raises(RuntimeError):
        indexer.insert(
            [MultimodalMetadata(keyframe_id="x", dense_vector=[0.0] * VECTOR_DIM)]
        )


def test_flush_is_not_called_automatically_during_insert(milvus_indexer, sample_records):
    """insert() không được tự động flush() -> tránh tạo nhiều segment nhỏ khi
    nạp theo nhiều batch."""
    milvus_indexer.insert(sample_records)
    assert milvus_indexer._collection.flush_calls == 0


def test_flush_can_be_called_explicitly_once_after_all_batches(milvus_indexer, sample_records):
    batch_1 = sample_records[:2]
    batch_2 = sample_records[2:]
    milvus_indexer.insert(batch_1)
    milvus_indexer.insert(batch_2)
    assert milvus_indexer._collection.flush_calls == 0  # vẫn chưa flush

    milvus_indexer.flush()
    assert milvus_indexer._collection.flush_calls == 1  # chỉ 1 lần sau cùng
    assert milvus_indexer.count() == len(sample_records)


def test_get_all_ids_paginates_beyond_single_batch(milvus_indexer):
    """get_all_ids() phải lấy đủ dữ liệu qua nhiều trang, không bị giới hạn
    bởi limit mặc định của query() (mô phỏng bằng batch_size nhỏ)."""
    records = [
        MultimodalMetadata(keyframe_id=f"page_id_{i}", dense_vector=[0.0] * VECTOR_DIM)
        for i in range(25)
    ]
    milvus_indexer.insert(records)

    ids = milvus_indexer.get_all_ids(batch_size=10)  # 25 record, batch=10 -> 3 trang
    assert len(ids) == 25
    assert set(ids) == {r.keyframe_id for r in records}


def test_search_auto_loads_when_not_loaded(milvus_indexer, sample_records):
    """Nếu quên gọi load() trước search(), MilvusIndexer phải tự load thay
    vì để lỗi 'collection not loaded' xảy ra ở tầng Milvus server."""
    milvus_indexer.insert(sample_records)
    assert milvus_indexer._loaded is False
    assert milvus_indexer._collection.load_calls == 0

    milvus_indexer.search(sample_records[0].dense_vector, top_k=3)

    assert milvus_indexer._loaded is True
    assert milvus_indexer._collection.load_calls == 1


def test_search_does_not_reload_if_already_loaded(milvus_indexer, sample_records):
    milvus_indexer.insert(sample_records)
    milvus_indexer.load()
    assert milvus_indexer._collection.load_calls == 1

    milvus_indexer.search(sample_records[0].dense_vector, top_k=3)

    # Đã load rồi -> không load lại lần nữa
    assert milvus_indexer._collection.load_calls == 1


# ---------------------------------------------------------------------- #
# 2. Elasticsearch: số lượng document & nội dung ocr_text
# ---------------------------------------------------------------------- #
def test_elastic_insert_document_count(elastic_indexer, sample_records):
    inserted = elastic_indexer.insert(sample_records)
    assert inserted == len(sample_records)
    assert elastic_indexer.count() == len(sample_records)


def test_elastic_ocr_text_content_preserved(elastic_indexer, sample_records):
    elastic_indexer.insert(sample_records)
    doc = elastic_indexer.get_document(sample_records[0].keyframe_id)
    assert doc is not None
    assert doc["ocr_text"] == "BỆNH VIỆN CHỢ RẪY"
    assert doc["keyframe_id"] == sample_records[0].keyframe_id


def test_elastic_index_created_with_bm25_similarity(elastic_indexer):
    body = elastic_indexer._client.indices._store[elastic_indexer.cfg.index_name]
    assert body["settings"]["similarity"]["vi_bm25"]["type"] == "BM25"
    ocr_mapping = body["mappings"]["properties"]["ocr_text"]
    assert ocr_mapping["similarity"] == "vi_bm25"
    # sub-field không dấu để hỗ trợ tìm kiếm tiếng Việt không dấu
    assert "folded" in ocr_mapping["fields"]


def test_elastic_insert_does_not_refresh_automatically(elastic_indexer, sample_records):
    """insert() phải truyền refresh=False cho bulk, không ép commit segment
    ngay sau mỗi batch."""
    elastic_indexer.insert(sample_records)
    assert elastic_indexer._client.last_bulk_refresh is False
    assert elastic_indexer._client.indices.refresh_calls == 0


def test_elastic_refresh_can_be_called_explicitly_once(elastic_indexer, sample_records):
    batch_1 = sample_records[:2]
    batch_2 = sample_records[2:]
    elastic_indexer.insert(batch_1)
    elastic_indexer.insert(batch_2)
    assert elastic_indexer._client.indices.refresh_calls == 0  # vẫn chưa refresh

    elastic_indexer.refresh()
    assert elastic_indexer._client.indices.refresh_calls == 1  # chỉ 1 lần sau cùng
    assert elastic_indexer.count() == len(sample_records)


def test_elastic_search_multi_match_returns_matching_keyframe(elastic_indexer, sample_records):
    """search() phải tìm được document qua OCR text (phục vụ Phase 4 - WRRF)."""
    elastic_indexer.insert(sample_records)

    results = elastic_indexer.search("bệnh viện", top_k=5)

    assert len(results) >= 1
    assert results[0]["keyframe_id"] == sample_records[0].keyframe_id
    assert results[0]["score"] > 0


def test_elastic_search_no_match_returns_empty(elastic_indexer, sample_records):
    elastic_indexer.insert(sample_records)
    results = elastic_indexer.search("từ khóa không tồn tại xyz123", top_k=5)
    assert results == []


def test_elastic_get_all_ids_paginates_via_scroll_beyond_one_page():
    """Với client KHÔNG có shortcut get_all_ids, ElasticIndexer phải tự
    dùng Scroll API để duyệt hết dữ liệu qua nhiều trang, không bị giới
    hạn bởi index.max_result_window (mô phỏng bằng page_size nhỏ)."""
    cfg = ElasticConfig(index_name="test_scroll_index")
    indexer = ElasticIndexer(cfg=cfg, client=FakeElasticScrollClient(), bulk_fn=fake_bulk)
    indexer.connect()
    indexer.create_index()

    records = [
        MultimodalMetadata(keyframe_id=f"scroll_id_{i}", dense_vector=[0.0] * VECTOR_DIM)
        for i in range(23)
    ]
    indexer.insert(records)

    ids = indexer.get_all_ids(page_size=7)  # 23 record, page=7 -> nhiều trang scroll
    assert len(ids) == 23
    assert set(ids) == {r.keyframe_id for r in records}


# ---------------------------------------------------------------------- #
# 3. Đồng bộ keyframe_id giữa Milvus <-> Elasticsearch (yêu cầu cốt lõi)
# ---------------------------------------------------------------------- #
def test_keyframe_id_consistency_between_milvus_and_es(
    milvus_indexer, elastic_indexer, sample_records
):
    milvus_indexer.insert(sample_records)
    elastic_indexer.insert(sample_records)

    expected_ids = {r.keyframe_id for r in sample_records}
    milvus_ids = set(milvus_indexer.get_all_ids())
    es_ids = set(elastic_indexer.get_all_ids())

    assert milvus_ids == expected_ids
    assert es_ids == expected_ids
    assert milvus_ids == es_ids  # hai hệ thống phải khớp hoàn toàn


def test_verify_keyframe_sync_detects_mismatch(milvus_indexer, elastic_indexer, sample_records):
    # Chỉ insert vào Milvus, "quên" 1 bản ghi ở Elasticsearch -> phải phát hiện lệch
    milvus_indexer.insert(sample_records)
    elastic_indexer.insert(sample_records[:-1])

    ok = verify_keyframe_sync(milvus_indexer, elastic_indexer, sample_records)
    assert ok is False


def test_verify_keyframe_sync_passes_when_synced(milvus_indexer, elastic_indexer, sample_records):
    milvus_indexer.insert(sample_records)
    elastic_indexer.insert(sample_records)

    ok = verify_keyframe_sync(milvus_indexer, elastic_indexer, sample_records)
    assert ok is True


# ---------------------------------------------------------------------- #
# 4. Pipeline end-to-end (index_data.run_pipeline)
# ---------------------------------------------------------------------- #
def test_run_pipeline_end_to_end_report_format(milvus_indexer, elastic_indexer, sample_records):
    report = run_pipeline(
        sample_records,
        milvus_indexer=milvus_indexer,
        es_indexer=elastic_indexer,
    )

    assert report["milvus_collection"]["name"] == milvus_indexer.cfg.collection_name
    assert report["milvus_collection"]["records"] == len(sample_records)
    assert report["milvus_collection"]["sample"]["keyframe_id"] == sample_records[0].keyframe_id
    assert report["milvus_collection"]["sample"]["dense_vector_dim"] == VECTOR_DIM

    assert report["elasticsearch_index"]["name"] == elastic_indexer.cfg.index_name
    assert report["elasticsearch_index"]["documents"] == len(sample_records)
    assert report["elasticsearch_index"]["sample"]["ocr_text"] == sample_records[0].ocr_text


def test_run_pipeline_empty_records_does_not_crash(milvus_indexer, elastic_indexer):
    report = run_pipeline([], milvus_indexer=milvus_indexer, es_indexer=elastic_indexer)
    assert report["milvus_collection"]["records"] == 0
    assert report["elasticsearch_index"]["documents"] == 0
    assert report["milvus_collection"]["sample"] is None
