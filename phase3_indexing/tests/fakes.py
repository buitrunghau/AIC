"""
tests/fakes.py
===============
Fake implementations của Milvus Collection và Elasticsearch client, dùng để
unit-test `milvus_indexer.py` / `elastic_indexer.py` mà KHÔNG cần một Milvus
hay Elasticsearch server thật đang chạy.

Các fake này chỉ implement đúng bề mặt API (interface) mà indexer thật sự
gọi tới, nên nếu `MilvusIndexer`/`ElasticIndexer` gọi sai method/API thật,
test cũng sẽ fail tương tự như khi chạy với client thật.
"""
from typing import Any, Dict, List, Optional


class FakeMilvusCollection:
    """Giả lập `pymilvus.Collection` — lưu dữ liệu trong dict ở memory."""

    def __init__(self):
        self.records: Dict[str, List[float]] = {}
        self.index_params: Optional[dict] = None
        self.flush_calls: int = 0
        self.load_calls: int = 0

    def create_index(self, field_name: str, index_params: dict) -> None:
        self.index_params = index_params

    def insert(self, data) -> None:
        ids, vectors = data
        for kid, vec in zip(ids, vectors):
            self.records[kid] = vec

    def flush(self) -> None:
        self.flush_calls += 1

    def load(self) -> None:
        self.load_calls += 1

    def drop(self) -> None:
        self.records = {}

    def query(self, expr=None, output_fields=None, limit=None, offset=0):
        """Giả lập phân trang offset/limit giống pymilvus.Collection.query."""
        ids = list(self.records.keys())
        sliced = ids[offset:] if limit is None else ids[offset : offset + limit]
        return [{"keyframe_id": kid} for kid in sliced]

    def search(self, data, anns_field, param, limit, output_fields):
        # Trả về danh sách rỗng — không cần logic ANN thật cho test indexing.
        return []

    @property
    def num_entities(self) -> int:
        return len(self.records)


class _FakeIndicesClient:
    """Giả lập `Elasticsearch.indices` (create/exists/delete/refresh index)."""

    def __init__(self, store: Dict[str, Any]):
        self._store = store
        self.refresh_calls = 0

    def exists(self, index: str) -> bool:
        return index in self._store

    def create(self, index: str, body: Optional[dict] = None, **kwargs) -> None:
        self._store[index] = body or {}

    def delete(self, index: str) -> None:
        self._store.pop(index, None)

    def refresh(self, index: str) -> None:
        self.refresh_calls += 1


class FakeElasticClient:
    """Giả lập `elasticsearch.Elasticsearch` — lưu document trong memory."""

    def __init__(self):
        self._index_store: Dict[str, Any] = {}
        self.indices = _FakeIndicesClient(self._index_store)
        self.docs: Dict[str, Dict[str, dict]] = {}  # index_name -> {id: source}
        self.last_bulk_refresh: Optional[bool] = None  # ghi lại refresh= truyền vào bulk gần nhất

    def count(self, index: str) -> dict:
        return {"count": len(self.docs.get(index, {}))}

    def search(self, index: str, body: Optional[dict] = None, size: Optional[int] = None, scroll=None):
        docs = self.docs.get(index, {})
        query = (body or {}).get("query", {})

        if "multi_match" in query:
            # Giả lập BM25 rất đơn giản: khớp substring, score = số lần xuất hiện.
            q = query["multi_match"]["query"].strip().lower()
            hits = []
            for src in docs.values():
                text = f"{src.get('ocr_text', '')} {src.get('asr_transcript', '')}".lower()
                if q and q in text:
                    hits.append(
                        {"_source": {"keyframe_id": src["keyframe_id"]}, "_score": text.count(q) + 1.0}
                    )
            hits.sort(key=lambda h: h["_score"], reverse=True)
        else:
            hits = [{"_source": {"keyframe_id": src["keyframe_id"]}} for src in docs.values()]

        if size is not None:
            hits = hits[:size]
        return {"hits": {"hits": hits}}

    def get(self, index: str, id: str) -> dict:
        return {"_source": self.docs.get(index, {}).get(id)}

    def get_all_ids(self, index: str) -> List[str]:
        """Shortcut tiện lợi cho test thông thường (không đi qua scroll)."""
        return list(self.docs.get(index, {}).keys())


class FakeElasticScrollClient:
    """
    Client giả lập KHÔNG có shortcut `get_all_ids`, buộc
    `ElasticIndexer.get_all_ids()` phải đi qua đường Scroll API thật —
    dùng để test hành vi phân trang vượt quá 1 trang.
    """

    def __init__(self):
        self._index_store: Dict[str, Any] = {}
        self.indices = _FakeIndicesClient(self._index_store)
        self.docs: Dict[str, Dict[str, dict]] = {}
        self.last_bulk_refresh: Optional[bool] = None
        self._scroll_state: Dict[str, Any] = {}
        self._scroll_counter = 0

    def count(self, index: str) -> dict:
        return {"count": len(self.docs.get(index, {}))}

    def search(self, index: str, body: Optional[dict] = None, size: Optional[int] = None, scroll=None):
        all_hits = [
            {"_source": {"keyframe_id": src["keyframe_id"]}}
            for src in self.docs.get(index, {}).values()
        ]
        page_size = size or len(all_hits) or 1
        page, remaining = all_hits[:page_size], all_hits[page_size:]

        resp: Dict[str, Any] = {"hits": {"hits": page}}
        if scroll is not None:
            self._scroll_counter += 1
            scroll_id = f"scroll_{self._scroll_counter}"
            self._scroll_state[scroll_id] = (remaining, page_size)
            resp["_scroll_id"] = scroll_id
        return resp

    def scroll(self, scroll_id: str, scroll=None):
        remaining, page_size = self._scroll_state.get(scroll_id, ([], 0))
        page, rest = remaining[:page_size], remaining[page_size:]
        self._scroll_state[scroll_id] = (rest, page_size)
        return {"hits": {"hits": page}, "_scroll_id": scroll_id}

    def clear_scroll(self, scroll_id: str) -> None:
        self._scroll_state.pop(scroll_id, None)


def fake_bulk(client, actions, refresh: bool = True):
    """
    Giả lập `elasticsearch.helpers.bulk(client, actions, refresh=...)`.
    Trả về (success_count, errors) giống chữ ký hàm thật, đồng thời ghi lại
    `refresh` được truyền vào lần gọi gần nhất trên `client.last_bulk_refresh`
    để test có thể xác nhận `insert()` không tự ép refresh=True.
    """
    if hasattr(client, "last_bulk_refresh"):
        client.last_bulk_refresh = refresh

    errors: List[Any] = []
    count = 0
    for action in actions:
        idx = action["_index"]
        doc_id = action["_id"]
        source = action["_source"]
        client.docs.setdefault(idx, {})[doc_id] = source
        count += 1
    return count, errors
