"""Phase 5: Rocchio relevance feedback for query vector refinement.

Deliverable: rocchio_feedback.py
- Takes user labels (relevant / not relevant) for retrieved results.
- Shifts query vector q_m closer to relevant centroids (C_r) and away from non-relevant (C_nr).
- Formula: q_m = alpha * q_orig + beta * (1/|C_r|) * sum(C_r) - gamma * (1/|C_nr|) * sum(C_nr)
- Returns updated query vector for re-search.
"""
