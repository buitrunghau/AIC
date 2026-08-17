Phase 4: Module Truy xuất & Căn chỉnh Chuỗi sự kiện (Retrieval & Alignment)
Module Phase 4 chịu trách nhiệm xử lý các câu truy vấn từ người dùng, thực hiện thuật toán tìm kiếm lai (Hybrid Search) và thuật toán giải bài toán chuỗi thời gian (TRAKE). Kết quả đầu ra là danh sách các khung hình tiềm năng nhất đã được căn chỉnh và xếp hạng để chuyển sang Phase 5.

1. Trách nhiệm cốt lõi
Điều phối tìm kiếm: Gọi các hàm tìm kiếm từ Phase 3 (tìm kiếm theo vector ngữ nghĩa trên Milvus và tìm kiếm theo từ vựng BM25 trên Elasticsearch).
Hợp nhất điểm số (WRRF): Áp dụng thuật toán Weighted Reciprocal Rank Fusion để hợp nhất kết quả từ hai luồng (Milvus và ES) cho các truy vấn dạng KIS (Known-Item Search) và Q&A.
Căn chỉnh thời gian (DANTE): Triển khai thuật toán Dynamic Alignment of Narrative Temporal Events bằng Quy hoạch động (Dynamic Programming) để xử lý truy vấn chuỗi sự kiện TRAKE, đảm bảo tìm ra chuỗi khung hình có trật tự thời gian tăng dần và liền mạch.
2. Cấu trúc thư mục (Deliverables)
text

phase4_retrieval/
├── search.py                 # Core retriever interface (entry point chính)
├── hybrid_search_wrrf.py     # Thuật toán lai WRRF (trộn rank Milvus & ES)
├── dante_trake_solver.py     # Thuật toán DANTE (quy hoạch động căn chỉnh sự kiện)
├── event_segmenter.py        # (Bản nháp PhoBERT - hiện không sử dụng)
├── mock_database.py          # Giả lập Phase 3 (Milvus/ES) cho môi trường dev/test
├── README.md                 # Tài liệu hướng dẫn
└── tests/
    └── test_retrieval.py     # Bộ Unit & Integration test (WRRF, DANTE, End-to-end flows)
3. Yêu cầu hệ thống & Cài đặt
Đảm bảo bạn đang ở thư mục gốc của dự án (PythonProject).

Cài đặt các thư viện cần thiết:

bash

pip install google-genai
# (Nếu Phase 3 có yêu cầu thêm về thư viện vector database, hãy cài đặt theo tài liệu Phase 3)
Cấu hình API Key (BẮT BUỘC): Hệ thống sử dụng LLM qua SDK mới của Google (google-genai) để phân tách sự kiện. Bạn không được hardcode API key vào code. Hãy liên hệ với Member 4 (AI & Retrieval Scientist) để lấy GEMINI_API_KEY. Sau đó cấu hình biến môi trường trước khi chạy:

Trên Windows (PowerShell):

powershell

$env:GEMINI_API_KEY="your_api_key_here"
Trên Linux/macOS:

bash

export GEMINI_API_KEY="your_api_key_here"
Lệnh import lazy: Hệ thống đang sử dụng APIEventSegmenter (dùng model gemini-3.7-flash) được nhúng trực tiếp trong search.py để tách câu. Bạn chưa cần cài đặt các thư viện Deep Learning phức tạp như torch hay mô hình ngôn ngữ lớn (BERT) tại môi trường local.

4. Hướng dẫn sử dụng & Kiểm thử
Chạy CLI Entrypoint
Bạn có thể chạy thử trực tiếp từ thư mục gốc của dự án:

bash

# KIS (Known-Item Search)
python -m phase4_retrieval.search --query "người đàn ông mặc áo đỏ" --query_type KIS --top_k 5
# Q&A (Question & Answering)
python -m phase4_retrieval.search --query "xe cứu thương đỗ trước bệnh viện. Biển số xe là gì?" --query_type "Q&A" --top_k 3
# TRAKE (Chuỗi sự kiện thời gian)
python -m phase4_retrieval.search --query "người đàn ông chạy đà, giậm nhảy qua xà, và tiếp đất trên nệm" --query_type TRAKE --top_k 5
Chạy Unit / Integration Tests
Test suite phủ sóng toàn bộ các kịch bản của DANTE (chọn sequence đúng, loại nhiễu, thứ tự tăng dần) và WRRF (hợp nhất chuẩn, giới hạn top_k). Chạy lệnh sau tại thư mục gốc:

bash

python -m pytest phase4_retrieval/tests/test_retrieval.py -v
5. Hướng dẫn Tích hợp (Module 3 & Module 5)
Phase 4 được thiết kế theo chuẩn data contract RetrievalResult chung của toàn hệ thống (định nghĩa trong shared_contracts/contracts.py).

5.1. Tích hợp với Phase 3 (Indexing Module)
Hiện tại, search.py đang import search_milvus_db và search_elastic_db từ mock_database.py. Khi Phase 3 đã hoàn thiện, bạn cần vào phase4_retrieval/search.py sửa lại import để kết nối với DB thật:

python

# Thay vì import mock:
# from .mock_database import search_milvus_db, search_elastic_db
# Hãy import module từ Phase 3:
from phase3_indexing.milvus_indexer import search_milvus_db
from phase3_indexing.elastic_indexer import search_elastic_db
5.2. Tích hợp với Phase 5 (Q&A / UI / Export)
Phase 5 chỉ cần import hàm search từ package Phase 4 và truyền vào dictionary truy vấn:

python

from phase4_retrieval import search
payload = {
    "query": "người đàn ông đi xe đạp",
    "query_type": "KIS",
    "top_k": 100
}
# results là List[Dict] tuân thủ chuẩn RetrievalResult
results = search(payload)
for r in results:
    print(f"Video: {r['video_id']}, Frame: {r['frame_ids']}, Score: {r['wrrf_score']}, Rank: {r['rank']}")
Lưu ý cho Phase 5 về Q&A: Với query_type = "Q&A", hàm search hiện tại đang sử dụng logic tách query (phần tìm kiếm & phần reasoning) và có sẵn hàm placeholder mock_llm_answer. Nhóm làm Phase 5/LLM cần tích hợp model thật (như Qwen2.5-VL) vào trong hàm mock_llm_answer của search.py để generate câu trả lời từ image frame.
