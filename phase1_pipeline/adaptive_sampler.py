"""
Phase 1: Adaptive Keyframe Sampling
Implements L2-distance based adaptive filtering and uniform sampling within shots.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for adaptive keyframe sampling."""
    threshold: float = 0.4
    samples_per_shot: int = 4
    use_l2_filtering: bool = True


class AdaptiveSampler:
    """Adaptive keyframe sampling using L2 distance filtering."""

    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()

    @staticmethod
    def compute_pixel_embedding(frame: np.ndarray) -> np.ndarray:
        """Computes low-memory frame statistics for L2 comparison."""
        embedding = []
        for c in range(3):
            channel = frame[:, :, c].astype(np.float32)
            embedding.extend([
                float(channel.mean()),
                float(channel.std()),
                float(np.percentile(channel, 25)),
                float(np.percentile(channel, 75))
            ])
        return np.array(embedding, dtype=np.float32)

    @staticmethod
    def compute_l2_distance(e_current: np.ndarray, e_prev: np.ndarray) -> float:
        diff = e_current - e_prev
        l2_diff = np.linalg.norm(diff)
        l2_prev = np.linalg.norm(e_prev)

        if l2_prev < 1e-8:
            return 0.0

        return float(l2_diff / l2_prev)

    def sample_uniform_in_shot(self, start_frame: int, end_frame: int) -> List[int]:
        n_samples = self.config.samples_per_shot
        length = end_frame - start_frame + 1

        if length <= n_samples:
            return list(range(start_frame, end_frame + 1))

        indices = []
        for i in range(n_samples):
            idx = start_frame + int(i * (length - 1) / (n_samples - 1))
            indices.append(idx)

        return indices

    def filter_candidates_by_l2(
            self,
            frame_dict: Dict[int, np.ndarray],
            candidate_indices: List[int]
    ) -> List[int]:
        """Filters candidate frame indices using pre-fetched candidate frames dictionary."""
        valid_indices = [idx for idx in candidate_indices if idx in frame_dict]
        if not self.config.use_l2_filtering or len(valid_indices) <= 1:
            return valid_indices

        filtered_indices = [valid_indices[0]]
        prev_embedding = self.compute_pixel_embedding(frame_dict[valid_indices[0]])

        for idx in valid_indices[1:]:
            current_frame = frame_dict[idx]
            current_embedding = self.compute_pixel_embedding(current_frame)

            distance = self.compute_l2_distance(current_embedding, prev_embedding)
            if distance > self.config.threshold:
                filtered_indices.append(idx)
                prev_embedding = current_embedding

        return filtered_indices