"""
Phase 1: Adaptive Keyframe Sampling
Implements L2-distance based adaptive filtering and uniform sampling within shots.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for adaptive keyframe sampling."""
    threshold: float = 0.4  # L2 distance threshold
    samples_per_shot: int = 4  # Number of uniform samples per shot
    use_l2_filtering: bool = True  # Enable L2 distance filtering


class AdaptiveSampler:
    """Adaptive keyframe sampling using L2 distance filtering."""

    def __init__(self, config: Optional[SamplingConfig] = None):
        """
        Initialize adaptive sampler.
        
        Args:
            config: Sampling configuration
        """
        self.config = config or SamplingConfig()

    @staticmethod
    def compute_pixel_embedding(frame: np.ndarray) -> np.ndarray:
        """
        Compute pixel-level embedding (simple: flattened RGB values).
        For efficiency on 8GB RAM, use downsampled frame statistics.
        
        Args:
            frame: Frame as numpy array (H, W, 3) uint8
            
        Returns:
            1D embedding vector
        """
        # For 8GB RAM efficiency: use histogram stats instead of full flattening
        # Compute mean, std, and histogram per channel
        embedding = []
        for c in range(3):
            channel = frame[:, :, c].astype(np.float32)
            embedding.extend([
                channel.mean(),
                channel.std(),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        return np.array(embedding, dtype=np.float32)

    @staticmethod
    def compute_l2_distance(e_current: np.ndarray, e_prev: np.ndarray) -> float:
        """
        Compute normalized L2 distance between frame embeddings.
        
        Formula: ||e_current - e_prev||_2 / ||e_prev||_2
        
        Args:
            e_current: Current frame embedding
            e_prev: Previous frame embedding
            
        Returns:
            Normalized L2 distance
        """
        diff = e_current - e_prev
        l2_diff = np.linalg.norm(diff)
        l2_prev = np.linalg.norm(e_prev)
        
        if l2_prev < 1e-8:
            return 0.0
        
        return l2_diff / l2_prev

    def sample_uniform_in_shot(self, start_frame: int, end_frame: int) -> List[int]:
        """
        Sample N evenly-spaced frames within a shot.
        
        Formula: k_extract = {K_a, K_{a + floor((b-a)/3)}, K_{a + floor(2(b-a)/3)}, K_b}
        where a = start_frame, b = end_frame, N = samples_per_shot
        
        Args:
            start_frame: First frame of shot
            end_frame: Last frame of shot
            
        Returns:
            List of sampled frame indices
        """
        n_samples = self.config.samples_per_shot
        length = end_frame - start_frame + 1
        
        if length <= n_samples:
            return list(range(start_frame, end_frame + 1))
        
        # Evenly spaced sampling
        indices = []
        for i in range(n_samples):
            idx = start_frame + int(i * (length - 1) / (n_samples - 1))
            indices.append(idx)
        
        return indices

    def filter_by_l2_distance(self, 
                             frames: np.ndarray, 
                             candidate_indices: List[int]) -> List[int]:
        """
        Filter candidate keyframes using L2 distance threshold.
        
        Keeps frame only if: ||e_current - e_prev||_2 / ||e_prev||_2 > threshold
        
        Args:
            frames: All frames as array (T, H, W, 3)
            candidate_indices: Initial candidate frame indices
            
        Returns:
            Filtered list of frame indices that pass L2 threshold
        """
        if not self.config.use_l2_filtering or len(candidate_indices) <= 1:
            return candidate_indices
        
        filtered_indices = [candidate_indices[0]]
        prev_embedding = self.compute_pixel_embedding(frames[candidate_indices[0]])
        
        for idx in candidate_indices[1:]:
            current_frame = frames[idx]
            current_embedding = self.compute_pixel_embedding(current_frame)
            
            distance = self.compute_l2_distance(current_embedding, prev_embedding)
            
            if distance > self.config.threshold:
                filtered_indices.append(idx)
                prev_embedding = current_embedding
        
        return filtered_indices

    def extract_keyframes_from_shot(self, 
                                   frames: np.ndarray, 
                                   start_frame: int, 
                                   end_frame: int) -> List[int]:
        """
        Extract keyframes from a single shot using uniform sampling + L2 filtering.
        
        Args:
            frames: All frames (T, H, W, 3)
            start_frame: Shot start frame index
            end_frame: Shot end frame index
            
        Returns:
            List of keyframe indices for this shot
        """
        # Step 1: Uniform sampling
        uniform_samples = self.sample_uniform_in_shot(start_frame, end_frame)
        
        # Step 2: L2 distance filtering
        keyframes = self.filter_by_l2_distance(frames, uniform_samples)
        
        return keyframes

    def extract_keyframes_from_segments(self, 
                                       frames: np.ndarray, 
                                       segments: List[Tuple[int, int]]) -> List[int]:
        """
        Extract keyframes from multiple shot segments.
        
        Args:
            frames: All frames (T, H, W, 3)
            segments: List of (start, end) frame indices for each shot
            
        Returns:
            Sorted list of all keyframe indices
        """
        all_keyframes = []
        
        for start_frame, end_frame in segments:
            shot_keyframes = self.extract_keyframes_from_shot(
                frames, start_frame, end_frame
            )
            all_keyframes.extend(shot_keyframes)
        
        return sorted(all_keyframes)

    def get_config_summary(self) -> str:
        """Get human-readable sampling configuration."""
        return (
            f"AdaptiveSampler Config:\n"
            f"  - Threshold: {self.config.threshold}\n"
            f"  - Samples per shot: {self.config.samples_per_shot}\n"
            f"  - L2 filtering enabled: {self.config.use_l2_filtering}"
        )
