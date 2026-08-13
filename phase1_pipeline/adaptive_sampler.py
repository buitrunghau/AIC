"""Phase 1: Adaptive keyframe sampling using L2 distance filtering.

Deliverable: adaptive_sampler.py
- Implements the L2-distance based adaptive filtering formula:
    ||e_current - e_prev||_2 / ||e_prev||_2 > threshold
- Also implements uniform sampling formula within a shot:
    k_extract = {K_{a + floor(i * (b-a)/3)}, for i in {0, 1, 2, 3}}
"""
