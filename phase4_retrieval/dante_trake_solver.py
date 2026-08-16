"""
Phase 4: DANTE (Dynamic Alignment of Narrative Temporal Events) algorithm for TRAKE queries.

Deliverable: dante_trake_solver.py — nâng cấp v2 (tương thích ngược)

Changes:
  - Two-pointer optimization cho DP (khi penalty tuyến tính)
  - Ràng buộc khoảng cách tối đa (max_frame_gap) để đảm bảo tính liền mạch
  - Chuẩn hóa điểm số (normalize_scores) nếu cần
  - Giới hạn số kết quả trả về (top_k)
  - Kiểm tra dữ liệu đầu vào
  - Output giữ nguyên định dạng cũ: video_id, frame_ids, query_type, answer, wrrf_score, rank
"""

from typing import List, Dict, Any, Optional


def _validate_candidate(cand: Dict[str, Any]) -> bool:
    """Kiểm tra candidate có đủ các trường cần thiết không."""
    return (
        isinstance(cand, dict)
        and "video_id" in cand
        and "frame_idx" in cand
        and "score" in cand
    )


def solve_dante(
    candidates_per_event: List[Dict[str, Any]],
    lambda_penalty: float = 0.001,
    max_frame_gap: Optional[int] = None,
    top_k: int = 10,
    normalize_scores: bool = False,
) -> List[Dict[str, Any]]:
    """
    Giải thuật DANTE tìm chuỗi khung hình tăng dần theo thời gian cho các sự kiện TRAKE.

    Args:
        candidates_per_event: List các dict, mỗi dict có key "candidates" chứa danh sách
                              các candidate (mỗi candidate có video_id, frame_idx, score).
        lambda_penalty: Hệ số phạt tuyến tính theo khoảng cách frame.
        max_frame_gap: Khoảng cách tối đa giữa hai frame liên tiếp (None = không giới hạn).
        top_k: Số lượng kết quả trả về.
        normalize_scores: Nếu True, score được chuẩn hóa về [0,1] bằng (score + 1)/2,
                          giả sử score gốc nằm trong [-1,1] (cosine similarity).

    Returns:
        List các dict có cấu trúc:
        {
            "video_id": str,
            "frame_ids": List[int],   # tăng dần theo thời gian
            "query_type": "TRAKE",
            "answer": None,
            "wrrf_score": float,
            "rank": int               # bắt đầu từ 1
        }
        Đã sắp xếp theo wrrf_score giảm dần và giới hạn top_k.
    """
    # ── 1. Validate input ──────────────────────────────────────
    if not isinstance(candidates_per_event, list) or len(candidates_per_event) == 0:
        return []

    num_events = len(candidates_per_event)

    # Chuẩn hóa dữ liệu: đảm bảo mỗi event có key "candidates"
    normalized_events = []
    for ev in candidates_per_event:
        cands = ev.get("candidates", [])
        if not isinstance(cands, list):
            cands = []
        # Lọc bỏ candidate không hợp lệ
        cands = [c for c in cands if _validate_candidate(c)]
        normalized_events.append(cands)

    if any(len(cands) == 0 for cands in normalized_events):
        # Có sự kiện không có candidate nào → không thể tạo chuỗi
        return []

    # ── 2. Gom nhóm theo video ─────────────────────────────────
    video_groups: Dict[str, List[List[Dict[str, Any]]]] = {}
    for ev_idx, cands in enumerate(normalized_events):
        for cand in cands:
            vid = cand["video_id"]
            if vid not in video_groups:
                video_groups[vid] = [[] for _ in range(num_events)]
            video_groups[vid][ev_idx].append(cand)

    # ── 3. Xử lý từng video ────────────────────────────────────
    all_results = []

    for vid, events in video_groups.items():
        # Sắp xếp candidate mỗi sự kiện theo frame_idx tăng dần
        for i in range(num_events):
            events[i].sort(key=lambda x: x["frame_idx"])

        # Nếu sự kiện nào rỗng thì bỏ qua video
        if any(len(events[i]) == 0 for i in range(num_events)):
            continue

        # Chuẩn hóa score nếu cần
        if normalize_scores:
            for i in range(num_events):
                for cand in events[i]:
                    # Giả sử score gốc thuộc [-1, 1] (cosine similarity)
                    cand["score"] = (cand["score"] + 1.0) / 2.0

        # ── 4. Khởi tạo DP table ───────────────────────────────
        # dp[i][j] = (best_total_score, prev_idx)
        dp = []
        for i in range(num_events):
            dp.append([(-float("inf"), -1)] * len(events[i]))

        # Khởi tạo cho sự kiện đầu tiên
        for j, cand in enumerate(events[0]):
            dp[0][j] = (cand["score"], -1)

        # ── 5. DP qua các sự kiện ──────────────────────────────
        for i in range(1, num_events):
            prev_events = events[i - 1]
            curr_events = events[i]

            if max_frame_gap is None:
                # ── Two-pointer optimization cho penalty tuyến tính ──
                # Ta cần tìm max over k của dp[i-1][k] + lambda_penalty * prev_frame_k
                # vì transition_score = curr_score + dp[i-1][k] - lambda * (curr_frame - prev_frame_k)
                #                    = curr_score + (dp[i-1][k] + lambda * prev_frame_k) - lambda * curr_frame
                best_value = -float("inf")
                best_prev_idx = -1
                best_prev_frame = None

                k = 0  # con trỏ duyệt qua prev_events
                for j, curr_cand in enumerate(curr_events):
                    curr_frame = curr_cand["frame_idx"]
                    # Thêm các prev candidate có frame < curr_frame
                    while k < len(prev_events) and prev_events[k]["frame_idx"] < curr_frame:
                        prev_cand = prev_events[k]
                        value = dp[i - 1][k][0] + lambda_penalty * prev_cand["frame_idx"]
                        if value > best_value:
                            best_value = value
                            best_prev_idx = k
                            best_prev_frame = prev_cand["frame_idx"]
                        k += 1

                    if best_prev_idx != -1:
                        penalty = lambda_penalty * (curr_frame - best_prev_frame)
                        total_score = curr_cand["score"] + best_value - lambda_penalty * curr_frame
                        # Note: best_value - lambda_penalty*curr_frame đã bao gồm cả penalty
                        # Nhưng để rõ ràng, ta tính trực tiếp như trên
                        dp[i][j] = (total_score, best_prev_idx)

            else:
                # ── Có giới hạn khoảng cách: dùng vòng lặp thường (vẫn O(Mi * Mi-1)) ──
                for j, curr_cand in enumerate(curr_events):
                    curr_frame = curr_cand["frame_idx"]
                    max_prev_score = -float("inf")
                    best_prev_idx = -1

                    for k, prev_cand in enumerate(prev_events):
                        prev_frame = prev_cand["frame_idx"]
                        time_gap = curr_frame - prev_frame

                        if 0 < time_gap <= max_frame_gap:
                            penalty = lambda_penalty * time_gap
                            transition_score = dp[i - 1][k][0] - penalty
                            if transition_score > max_prev_score:
                                max_prev_score = transition_score
                                best_prev_idx = k

                    if best_prev_idx != -1:
                        dp[i][j] = (curr_cand["score"] + max_prev_score, best_prev_idx)

        # ── 6. Tìm kết quả tốt nhất ở sự kiện cuối ──────────────
        best_final_score = -float("inf")
        best_final_idx = -1
        last_event = events[num_events - 1]

        for j in range(len(last_event)):
            if dp[num_events - 1][j][0] > best_final_score:
                best_final_score = dp[num_events - 1][j][0]
                best_final_idx = j

        if best_final_idx == -1:
            continue  # Không tìm được chuỗi hợp lệ

        # ── 7. Backtracking ────────────────────────────────────
        sequence_frames = []
        curr_idx = best_final_idx
        for i in range(num_events - 1, -1, -1):
            sequence_frames.append(events[i][curr_idx]["frame_idx"])
            curr_idx = dp[i][curr_idx][1]

        sequence_frames.reverse()  # Đưa về thứ tự tăng dần

        # Kiểm tra tính hợp lệ (tăng dần)
        if all(sequence_frames[k] < sequence_frames[k + 1] for k in range(len(sequence_frames) - 1)):
            all_results.append({
                "video_id": vid,
                "frame_ids": sequence_frames,
                "query_type": "TRAKE",
                "answer": None,
                "wrrf_score": round(best_final_score, 4)
            })

    # ── 8. Sắp xếp, gán rank và giới hạn top_k ─────────────────
    all_results.sort(key=lambda x: x["wrrf_score"], reverse=True)

    final_results = []
    for rank, res in enumerate(all_results[:top_k], start=1):
        res["rank"] = rank
        final_results.append(res)

    return final_results
