"""Phase 4: Unit tests for Retrieval & Alignment module.

Covers:
    - WRRF: formula correctness, rank ordering, edge cases
    - DANTE: temporal alignment, single event, multi-video, max_frame_gap, top_per_video
    - search(): KIS, Q&A, TRAKE end-to-end flows
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybrid_search_wrrf import calculate_wrrf  # noqa: E402
from dante_trake_solver import solve_dante  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# WRRF Tests
# ═══════════════════════════════════════════════════════════════

class TestWRRF(unittest.TestCase):
    """Test suite cho thuật toán Weighted Reciprocal Rank Fusion."""

    def test_basic_formula(self):
        """Kiểm tra công thức WRRF cơ bản: alpha/(rank+k) + (1-alpha)/(rank+k)."""
        es = [{"video_id": "V_01", "frame_idx": 100, "score": 0.9}]
        mv = [{"video_id": "V_01", "frame_idx": 100, "score": 0.8}]
        r = calculate_wrrf(es, mv, alpha=0.6, k=60)
        expected = 0.6 / (1 + 60) + 0.4 / (1 + 60)
        self.assertAlmostEqual(r[0]["wrrf_score"], expected, places=9)

    def test_alpha_weighting(self):
        """Doc chỉ xuất hiện ở ES → chỉ nhận alpha/(1+k)."""
        es = [{"video_id": "V_01", "frame_idx": 100, "score": 0.9}]
        mv = [{"video_id": "V_02", "frame_idx": 200, "score": 0.8}]
        r = calculate_wrrf(es, mv, alpha=0.6, k=60)
        v01 = [x for x in r if x["video_id"] == "V_01"][0]
        v02 = [x for x in r if x["video_id"] == "V_02"][0]
        self.assertAlmostEqual(v01["wrrf_score"], 0.6 / 61, places=9)
        self.assertAlmostEqual(v02["wrrf_score"], 0.4 / 61, places=9)

    def test_both_source_ranked_higher(self):
        """Doc xuất hiện ở cả 2 source phải đứng đầu bảng xếp hạng."""
        es = [{"video_id": "V_01", "frame_idx": 100, "score": 0.9}]
        mv = [{"video_id": "V_01", "frame_idx": 100, "score": 0.8},
              {"video_id": "V_02", "frame_idx": 200, "score": 0.95}]
        r = calculate_wrrf(es, mv, alpha=0.6, k=60)
        self.assertEqual(r[0]["video_id"], "V_01")

    def test_rank_field_assigned(self):
        """Kiểm tra rank được gán chính xác từ 1."""
        es = [{"video_id": f"V_{i}", "frame_idx": i*10, "score": 0.5} for i in range(5)]
        r = calculate_wrrf(es, [], alpha=0.6, k=60)
        ranks = [x["rank"] for x in r]
        self.assertEqual(ranks, [1, 2, 3, 4, 5])

    def test_score_monotonic_decreasing(self):
        """Score phải giảm dần theo rank."""
        es = [{"video_id": f"V_{i}", "frame_idx": i*10, "score": 0.5} for i in range(10)]
        mv = [{"video_id": f"V_{i}", "frame_idx": i*10, "score": 0.5} for i in range(10)]
        r = calculate_wrrf(es, mv, alpha=0.6, k=60)
        for i in range(len(r) - 1):
            self.assertGreaterEqual(r[i]["wrrf_score"], r[i+1]["wrrf_score"])

    def test_empty_inputs(self):
        """Cả 2 nguồn trống → trả về list rỗng."""
        self.assertEqual(calculate_wrrf([], []), [])

    def test_one_side_empty(self):
        """Một nguồn trống vẫn trả về kết quả từ nguồn còn lại."""
        es = [{"video_id": "V_01", "frame_idx": 100, "score": 0.9}]
        r = calculate_wrrf(es, [], alpha=0.6, k=60)
        self.assertEqual(len(r), 1)

    def test_malformed_items_skipped(self):
        """Items thiếu field bắt buộc bị bỏ qua không gây crash."""
        es = [{"video_id": "V_01"}, {"video_id": "V_02", "frame_idx": 200}]
        r = calculate_wrrf(es, [])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["video_id"], "V_02")

    def test_top_k_limits_output(self):
        """top_k giới hạn số lượng output."""
        es = [{"video_id": f"V_{i}", "frame_idx": i*10, "score": 0.5} for i in range(50)]
        r = calculate_wrrf(es, es, top_k=10)
        self.assertEqual(len(r), 10)

    def test_duplicate_key_merges_score(self):
        """Cùng video_id + frame_idx xuất hiện nhiều lần → score cộng dồn."""
        es = [
            {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
            {"video_id": "V_01", "frame_idx": 100, "score": 0.7},
        ]
        r = calculate_wrrf(es, [])
        self.assertEqual(len(r), 1)
        expected = 0.6 / 61 + 0.6 / 62
        self.assertAlmostEqual(r[0]["wrrf_score"], expected, places=9)

    def test_score_field_present(self):
        """Output phải có trường 'score' (alias của wrrf_score cho DANTE)."""
        es = [{"video_id": "V_01", "frame_idx": 100, "score": 0.9}]
        r = calculate_wrrf(es, [], alpha=0.6, k=60)
        self.assertIn("score", r[0])
        self.assertEqual(r[0]["score"], r[0]["wrrf_score"])


# ═══════════════════════════════════════════════════════════════
# DANTE Tests
# ═══════════════════════════════════════════════════════════════

class TestDANTE(unittest.TestCase):
    """Test suite cho thuật toán DANTE (Dynamic Alignment of Narrative Temporal Events)."""

    def test_basic_3event_alignment(self):
        """Test case gốc: 3 sự kiện với candidates lộn xộn → tìm đúng [1050, 1080, 1120]."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 1500, "score": 0.8},
                {"video_id": "V_01", "frame_idx": 1050, "score": 0.9},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 1000, "score": 0.95},
                {"video_id": "V_01", "frame_idx": 1080, "score": 0.89},
            ]},
            {"event_idx": 2, "candidates": [
                {"video_id": "V_01", "frame_idx": 1120, "score": 0.88},
                {"video_id": "V_01", "frame_idx": 1020, "score": 0.92},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001)

        self.assertTrue(len(results) > 0, "DANTE không trả về kết quả.")
        self.assertEqual(results[0]["frame_ids"], [1050, 1080, 1120])
        self.assertEqual(results[0]["video_id"], "V_01")
        self.assertEqual(results[0]["query_type"], "TRAKE")
        self.assertIsNone(results[0]["answer"])
        self.assertEqual(results[0]["rank"], 1)

    def test_ascending_order_guaranteed(self):
        """Mọi kết quả phải có frame_ids tăng dần."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 300, "score": 0.95},
                {"video_id": "V_01", "frame_idx": 100, "score": 0.80},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 200, "score": 0.90},
                {"video_id": "V_01", "frame_idx": 400, "score": 0.70},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001)
        for r in results:
            for i in range(len(r["frame_ids"]) - 1):
                self.assertLess(r["frame_ids"][i], r["frame_ids"][i+1])

    def test_empty_input(self):
        """Input rỗng → output rỗng."""
        self.assertEqual(solve_dante([], lambda_penalty=0.001), [])

    def test_event_with_no_candidates(self):
        """Sự kiện không có candidate nào → không thể tạo chuỗi."""
        cands = [
            {"event_idx": 0, "candidates": [{"video_id": "V_01", "frame_idx": 100, "score": 0.9}]},
            {"event_idx": 1, "candidates": []},
        ]
        self.assertEqual(solve_dante(cands, lambda_penalty=0.001), [])

    def test_single_event(self):
        """N=1 → trả về frame có score cao nhất."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 500, "score": 0.9},
                {"video_id": "V_01", "frame_idx": 300, "score": 0.95},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["frame_ids"], [300])

    def test_multiple_videos(self):
        """Candidates từ nhiều video → trả kết quả cho mỗi video eligible."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
                {"video_id": "V_02", "frame_idx": 200, "score": 0.85},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 150, "score": 0.88},
                {"video_id": "V_02", "frame_idx": 250, "score": 0.87},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001)
        vids = {r["video_id"] for r in results}
        self.assertIn("V_01", vids)
        self.assertIn("V_02", vids)

    def test_max_frame_gap_constraint(self):
        """max_frame_gap loại bỏ frame quá xa, chọn frame gần hơn."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 5000, "score": 0.9},
                {"video_id": "V_01", "frame_idx": 110, "score": 0.5},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001, max_frame_gap=100)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["frame_ids"], [100, 110])

    def test_penalty_prefers_close_frames(self):
        """Lambda penalty cao → ưu tiên frame gần hơn dù score bằng nhau."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 110, "score": 0.9},
                {"video_id": "V_01", "frame_idx": 10000, "score": 0.9},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.01)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["frame_ids"], [100, 110])

    def test_normalize_scores(self):
        """normalize_scores chuyển [-1,1] về [0,1] mà vẫn ra kết quả."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": -0.5},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 200, "score": 0.8},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001, normalize_scores=True)
        self.assertTrue(len(results) > 0)

    def test_top_per_video(self):
        """top_per_video > 1 → nhiều chuỗi khác nhau từ cùng 1 video."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
                {"video_id": "V_01", "frame_idx": 500, "score": 0.85},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 200, "score": 0.88},
                {"video_id": "V_01", "frame_idx": 600, "score": 0.83},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001, top_per_video=3)
        # Ít nhất 2 chuỗi khác nhau nếu có đủ ending points hợp lệ
        if len(results) >= 2:
            self.assertNotEqual(results[0]["frame_ids"], results[1]["frame_ids"])

    def test_invalid_candidates_skipped(self):
        """Candidates thiếu field bắt buộc bị lọc bỏ."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
                {"video_id": "V_01", "frame_idx": 200},  # thiếu score
                {"bad": "data"},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 150, "score": 0.88},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001)
        self.assertTrue(len(results) > 0)

    def test_score_decreasing_by_rank(self):
        """wrrf_score phải giảm dần theo rank."""
        cands = [
            {"event_idx": 0, "candidates": [
                {"video_id": "V_01", "frame_idx": 100, "score": 0.9},
                {"video_id": "V_02", "frame_idx": 200, "score": 0.7},
                {"video_id": "V_03", "frame_idx": 300, "score": 0.5},
            ]},
            {"event_idx": 1, "candidates": [
                {"video_id": "V_01", "frame_idx": 150, "score": 0.88},
                {"video_id": "V_02", "frame_idx": 250, "score": 0.65},
                {"video_id": "V_03", "frame_idx": 350, "score": 0.45},
            ]},
        ]
        results = solve_dante(cands, lambda_penalty=0.001)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]["wrrf_score"], results[i+1]["wrrf_score"])


# ═══════════════════════════════════════════════════════════════
# Integration Tests (search flow)
# ═══════════════════════════════════════════════════════════════

class TestSearchIntegration(unittest.TestCase):
    """Test end-to-end search() function qua mock database."""

    @classmethod
    def setUpClass(cls):
        """Import search qua package-level import."""
        from phase4_retrieval.search import search
        cls.search = staticmethod(search)

    def test_kis_flow(self):
        """KIS: trả về list dict với query_type=KIS, answer=None, rank tăng dần."""
        results = self.search({"query": "nguoi dan ong mac ao do", "query_type": "KIS", "top_k": 5})
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["query_type"], "KIS")
        self.assertIsNone(results[0]["answer"])
        self.assertEqual(results[0]["rank"], 1)
        # frame_ids phải là list 1 phần tử cho KIS
        self.assertEqual(len(results[0]["frame_ids"]), 1)

    def test_qa_flow(self):
        """Q&A: trả về answer = MOCK_ANSWER."""
        results = self.search({
            "query": "Xe cuu thuong do truoc benh vien. Bien so la gi?",
            "query_type": "Q&A",
            "top_k": 3,
        })
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["query_type"], "Q&A")
        self.assertEqual(results[0]["answer"], "MOCK_ANSWER")

    def test_trake_flow(self):
        """TRAKE: trả về frame_ids tăng dần, query_type=TRAKE."""
        results = self.search({
            "query": "nguoi dan ong chay da, giam nhay qua xa, va tiep dat",
            "query_type": "TRAKE",
            "top_k": 5,
        })
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["query_type"], "TRAKE")
        seq = results[0]["frame_ids"]
        for i in range(len(seq) - 1):
            self.assertLess(seq[i], seq[i+1])

    def test_unsupported_query_type_raises(self):
        """Query type không hợp lệ → ValueError."""
        with self.assertRaises(ValueError):
            self.search({"query": "test", "query_type": "INVALID"})

    def test_output_has_required_fields(self):
        """Tất cả kết quả phải có đủ 6 trường theo contract."""
        required = {"video_id", "frame_ids", "query_type", "answer", "wrrf_score", "rank"}
        results = self.search({"query": "test", "query_type": "KIS", "top_k": 3})
        for r in results:
            self.assertTrue(required.issubset(r.keys()), f"Missing fields: {required - set(r.keys())}")


if __name__ == "__main__":
    unittest.main()
