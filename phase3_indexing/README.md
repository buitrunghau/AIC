# Phase 3 – Indexing Module

**Member 3 – Backend & DB Administrator**
Nạp `MultimodalMetadata` (output Phase 2) vào Milvus (dense vector, HNSW) và
Elasticsearch (OCR/ASR text, BM25), đồng bộ khóa chính `keyframe_id` giữa hai
hệ thống để Phase 4 (Retrieval & Alignment) hợp nhất kết quả.

## Cấu trúc file (đúng Deliverables Checklist)

| File | Vai trò |
|---|---|
| `contracts.py` | `MultimodalMetadata` – contract Input từ Phase 2 (mục 9) |
| `config.py` | Cấu hình Milvus/Elasticsearch qua biến môi trường |
| `milvus_indexer.py` | Tạo collection, build HNSW index, nạp dense vectors |
| `elastic_indexer.py` | Tạo index, analyzer Việt/Anh, BM25, nạp OCR/ASR text |
| `index_data.py` | Pipeline chính end-to-end, verify đồng bộ `keyframe_id` |
| `tests/test_indexing.py` | Kiểm tra số bản ghi & tính nhất quán `keyframe_id` |
| `tests/fakes.py` | Fake Milvus/Elasticsearch client (chạy test không cần server thật) |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Địa chỉ Milvus |
| `MILVUS_COLLECTION` | `video_frames` | Tên collection |
| `MILVUS_VECTOR_DIM` | `1024` | Số chiều vector (768 nếu dùng SigLIP base, 1024 nếu BEiT-3/SigLIP-L) |
| `MILVUS_HNSW_M` | `16` | Số liên kết/node trong đồ thị HNSW |
| `MILVUS_HNSW_EF_CONSTRUCTION` | `200` | Độ rộng tìm kiếm lúc build index |
| `MILVUS_HNSW_EF_SEARCH` | `64` | Độ rộng tìm kiếm lúc query (Phase 4) |
| `ES_HOSTS` | `http://localhost:9200` | Địa chỉ Elasticsearch |
| `ES_INDEX` | `video_text` | Tên index |
| `ES_BM25_K1` / `ES_BM25_B` | `1.2` / `0.75` | Tham số BM25 |

## Chạy pipeline (cần Milvus + Elasticsearch server thật đang chạy)

```bash
export MILVUS_VECTOR_DIM=8   # khớp với sample_data.json (demo); production để mặc định 1024
python index_data.py --input sample_data.json --recreate
```

Output (đúng format mục 2):

```json
{
  "milvus_collection": {
    "name": "video_frames",
    "records": 2,
    "sample": {"keyframe_id": "L01_V025_1050", "dense_vector_dim": 8}
  },
  "elasticsearch_index": {
    "name": "video_text",
    "documents": 2,
    "sample": {"keyframe_id": "L01_V025_1050", "ocr_text": "BỆNH VIỆN CHỢ RẪY"}
  }
}
```

## Chạy test (KHÔNG cần Milvus/Elasticsearch server)

```bash
python -m pytest tests/test_indexing.py -v
```

Test dùng fake client (dependency injection qua tham số `client=` /
`bulk_fn=` của `MilvusIndexer`/`ElasticIndexer`) để kiểm tra logic indexing
độc lập với hạ tầng, cho phép Member 3 code & test song song trong khi
Milvus/Elasticsearch cluster thật chưa sẵn sàng — đúng tinh thần "5 thành
viên code độc lập" của kiến trúc.

## Cập nhật theo review (fix)

| Vấn đề | Fix |
|---|---|
| `flush()` gọi sau mỗi `insert()` (Milvus) → tạo nhiều segment nhỏ, giảm hiệu năng | `insert()` không còn tự flush. Tách `flush()` thành method riêng, `index_data.py` chỉ gọi **1 lần** sau khi toàn bộ batch đã insert xong, trước `load()`. |
| `get_all_ids()` (Milvus) dùng `query()` không phân trang → thiếu dữ liệu khi > ~16384 record | Thêm phân trang `offset`/`limit` (tham số `batch_size`, mặc định 10000), lặp cho đến khi hết dữ liệu. |
| `search()` (Milvus) lỗi "collection not loaded" nếu quên gọi `load()` trước | Thêm cờ nội bộ `_loaded`; `search()` tự động gọi `load()` (kèm log cảnh báo) nếu collection chưa được load. |
| Elasticsearch chưa có `search()` cho Phase 4 (WRRF) | Thêm `search(query_text, top_k)` dùng `multi_match` qua `ocr_text`/`asr_transcript` và subfield `.folded`, boost field có dấu cao hơn, trả về `[{"keyframe_id", "score"}]`. |
| `refresh=True` mỗi batch trong `insert()` (Elasticsearch) → ép commit segment liên tục, chậm khi nạp nhiều batch | `insert()` mặc định `refresh=False`. Tách `refresh()` thành method riêng, `index_data.py` chỉ gọi **1 lần** sau khi toàn bộ batch đã insert xong. |
| `get_all_ids()` (Elasticsearch) dùng `size=10000` → chạm/lỗi `index.max_result_window` khi > 10.000 keyframe | Chuyển sang dùng **Scroll API** (`scroll`/`clear_scroll`), duyệt hết dữ liệu theo trang (`page_size`, mặc định 5000) không bị giới hạn `max_result_window`. |

## Thiết kế đáng chú ý

- **Đồng bộ `keyframe_id`**: dùng trực tiếp làm PK trong Milvus (VARCHAR) và
  `_id` trong Elasticsearch → không cần bảng ánh xạ trung gian, tránh lệch
  dữ liệu giữa 2 hệ thống. `verify_keyframe_sync()` trong `index_data.py`
  đối chiếu tập `keyframe_id` giữa hai DB sau mỗi lần nạp.
- **HNSW cho Milvus**: cân bằng tốc độ/độ chính xác qua `M` và
  `efConstruction` (build-time), `ef` (query-time, dùng ở Phase 4).
- **BM25 + analyzer song ngữ cho Elasticsearch**: `standard` tokenizer +
  stopword filter riêng cho vi/en; thêm sub-field `ocr_text.folded` /
  `asr_transcript.folded` (asciifolding) để hỗ trợ tìm kiếm tiếng Việt
  không dấu.
- **Import lazy `pymilvus`/`elasticsearch`**: cho phép module import và
  unit-test được ngay cả khi chưa có server/thư viện thật, nhưng vẫn dùng
  đúng API thật khi chạy production.