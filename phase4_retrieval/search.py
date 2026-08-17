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

import os
import json
import logging
import re
from typing import Dict, Any, List

# (Đã chuyển import google.genai vào lazy bên trong APIEventSegmenter để tránh vỡ code khi thiếu thư viện)

# 2. Thêm dấu chấm (.) phía trước để chạy chuẩn Module Package, sửa lỗi ModuleNotFoundError
from .hybrid_search_wrrf import calculate_wrrf
from .dante_trake_solver import solve_dante
from .mock_database import search_milvus_db, search_elastic_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_event_cache: Dict[str, List[str]] = {}

# ══════════════════════════════════════════════════════════════════════
# 1. API EVENT SEGMENTER (NEW GOOGLE GENAI SDK)
# ══════════════════════════════════════════════════════════════════════


class APIEventSegmenter:
    def __init__(self):
        # ⚠️ BẮT BUỘC: Phải thiết lập biến môi trường GEMINI_API_KEY trước khi chạy
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GEMINI_API_KEY! Hãy liên hệ Member 4 (AI & Retrieval Scientist) để lấy API key.")

        # Import lazy để không làm hỏng toàn bộ Phase 4 nếu máy chưa cài `google-genai`
        try:
            from google import genai
        except ImportError:
            logger.error("Thư viện `google-genai` chưa được cài đặt. Vui lòng chạy: pip install google-genai")
            raise

        # Cú pháp khởi tạo Client của thư viện mới
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-3.7-flash'

        self.system_prompt = """
        Bạn là một hệ thống tách sự kiện video. Nhiệm vụ:
        1. Tách câu truy vấn thành các sự kiện nối tiếp nhau theo thời gian.
        2. BẮT BUỘC: Mỗi sự kiện phải là chuỗi con (exact substring) cắt trực tiếp từ câu gốc.
        3. KHÔNG đưa các từ nối (như: sau đó, rồi, tiếp theo, kế tiếp, lập tức) vào đầu chuỗi sự kiện.
        4. CHỈ trả về ĐÚNG 1 mảng JSON, KHÔNG giải thích, KHÔNG bọc trong markdown.

        Ví dụ Input: "Xe máy phanh gấp, sau đó ngã ra đường."
        Output: ["Xe máy phanh gấp,", "ngã ra đường."]
        """

    def segment(self, query: str) -> List[str]:
        if not query.strip():
            return []

        logger.info("Đang gọi Gemini API (New SDK) để phân rã câu TRAKE...")
        prompt = f"{self.system_prompt}\n\nInput: '{query}'\nOutput:"

        try:
            # Cú pháp gọi API mới
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            result_text = response.text.strip()

            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "").replace("```", "").strip()
            if result_text.startswith("```"):
                result_text = result_text.replace("```", "").strip()

            events = json.loads(result_text)

            if isinstance(events, list) and len(events) > 0:
                return events

            return [query]
        except Exception as e:
            logger.error(f"Lỗi khi gọi API: {e}. Đang dùng câu gốc làm fallback.")
            return [query]


_segmenter_instance = None


def get_api_segmenter():
    global _segmenter_instance
    if _segmenter_instance is None:
        _segmenter_instance = APIEventSegmenter()
    return _segmenter_instance

# ══════════════════════════════════════════════════════════════════════
# 2. XỬ LÝ NGÔN NGỮ (TIỀN XỬ LÝ QUERY)
# ══════════════════════════════════════════════════════════════════════


def clean_query_for_dbs(query: str, query_type: str):
    noise_prefixes = [r"^tìm video( về)?", r"^hãy cho tôi biết", r"^tìm kiếm", r"^đoạn clip nào", r"^hình ảnh"]
    clean_q = query.lower()
    for pattern in noise_prefixes:
        clean_q = re.sub(pattern, "", clean_q).strip()

    milvus_query = clean_q
    if query_type == "Q&A":
        question_words = [r"ai là người ", r"tại sao ", r"ở đâu ", r"như thế nào ", r"màu gì ", r"bao nhiêu "]
        for qw in question_words:
            milvus_query = re.sub(qw, "", milvus_query).strip()

    stopwords = ["một", "những", "các", "là", "của", "và", "có", "tại", "trong", "đang"]
    es_words = [word for word in clean_q.split() if word not in stopwords]
    es_query = " ".join(es_words)

    return milvus_query, es_query


def split_qa_query(query: str):
    if "?" in query:
        idx = query.index("?")
        q_search = query[:idx].strip()
        q_reasoning = query[idx + 1:].strip()
        if not q_reasoning:
            q_reasoning = query.strip()
    else:
        q_search = query.strip()
        q_reasoning = query.strip()
    return q_search, q_reasoning

# ══════════════════════════════════════════════════════════════════════
# 3. QUẢN LÝ CACHE CHO TRAKE
# ══════════════════════════════════════════════════════════════════════


def parse_trake_events(query: str, use_cache: bool = True, cache_file: str = "event_cache.json") -> List[str]:
    if use_cache and query in _event_cache:
        logger.info("Memory cache hit (API saved).")
        return _event_cache[query]

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            disk_cache = json.load(f)
        if query in disk_cache:
            logger.info("Disk cache hit (API saved).")
            _event_cache[query] = disk_cache[query]
            return disk_cache[query]

    segmenter = get_api_segmenter()
    sub_queries = segmenter.segment(query)

    _event_cache[query] = sub_queries

    if use_cache:
        disk_cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                disk_cache = json.load(f)

        disk_cache[query] = sub_queries

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(disk_cache, f, ensure_ascii=False, indent=2)

    return sub_queries

# ══════════════════════════════════════════════════════════════════════
# 4. HÀM TÌM KIẾM CỐT LÕI (ENTRY POINT)
# ══════════════════════════════════════════════════════════════════════


def search(query_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    query_text = query_payload.get("query", "")
    query_type = query_payload.get("query_type", "KIS").upper()
    top_k = query_payload.get("top_k", 100)

    logger.info(f"==> Received {query_type} query: {query_text[:80]}...")

    if query_type == "KIS":
        milvus_query, es_query = clean_query_for_dbs(query_text, query_type)

        es_results = search_elastic_db(es_query, top_k=top_k)
        milvus_results = search_milvus_db(milvus_query, top_k=top_k)
        wrrf_ranked = calculate_wrrf(es_results, milvus_results)

        final_results = []
        for rank, item in enumerate(wrrf_ranked[:top_k], start=1):
            final_results.append({
                "video_id": item["video_id"],
                "frame_ids": [item["frame_idx"]],
                "query_type": query_type,
                "answer": None,
                "wrrf_score": round(item["wrrf_score"], 4),
                "rank": rank,
            })
        return final_results

    elif query_type == "Q&A":
        q_search, q_reasoning = split_qa_query(query_text)
        milvus_query, es_query = clean_query_for_dbs(q_search, query_type)

        es_results = search_elastic_db(es_query, top_k=top_k)
        milvus_results = search_milvus_db(milvus_query, top_k=top_k)
        wrrf_ranked = calculate_wrrf(es_results, milvus_results)

        final_results = []
        for rank, item in enumerate(wrrf_ranked[:top_k], start=1):
            final_results.append({
                "video_id": item["video_id"],
                "frame_ids": [item["frame_idx"]],
                "query_type": query_type,
                "answer": "MOCK_ANSWER",
                "wrrf_score": round(item["wrrf_score"], 4),
                "rank": rank,
            })
        return final_results

    elif query_type == "TRAKE":
        sub_queries = parse_trake_events(query_text)
        logger.info(f"Kết quả phân rã ({len(sub_queries)} sự kiện): {sub_queries}")

        candidates_per_event = []
        for idx, sub_q in enumerate(sub_queries):
            milvus_query, es_query = clean_query_for_dbs(sub_q, "KIS")

            sub_es_results = search_elastic_db(es_query, top_k=50)
            sub_milvus_results = search_milvus_db(milvus_query, top_k=50)
            merged = calculate_wrrf(sub_es_results, sub_milvus_results)

            dante_candidates = [
                {"video_id": c["video_id"], "frame_idx": c["frame_idx"], "score": c["wrrf_score"]}
                for c in merged
            ]

            candidates_per_event.append({
                "event_idx": idx,
                "candidates": dante_candidates,
            })

        dante_results = solve_dante(candidates_per_event, lambda_penalty=0.001, top_k=top_k)
        return dante_results

    raise ValueError(f"Unsupported query_type: {query_type}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 4 Retrieval — hybrid search & DANTE alignment"
    )
    parser.add_argument("--query", type=str, default="", help="Text query string")
    parser.add_argument(
        "--query_type",
        type=str,
        default="KIS",
        choices=["KIS", "Q&A", "TRAKE"],
        help="Query type: KIS, Q&A, or TRAKE",
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Number of top results to return"
    )
    args = parser.parse_args()

    payload = {
        "query": args.query,
        "query_type": args.query_type,
        "top_k": args.top_k,
    }

    results = search(payload)
    print(f"\n[KET QUA] Top {min(3, len(results))}:")
    for res in results[:3]:
        print(res)
