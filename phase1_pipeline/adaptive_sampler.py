"""
Phase 1: Adaptive Keyframe Sampling
Implements L2-distance based adaptive filtering and uniform sampling within shots.
"""

import numpy as np
import torch
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
        """Filters candidate frame indices using GPU batch acceleration if available."""
        valid_indices = [idx for idx in candidate_indices if idx in frame_dict]
        if not self.config.use_l2_filtering or len(valid_indices) <= 1:
            return valid_indices

        # TỰ ĐỘNG CHUYỂN MẠCH GPU/CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            # 1. Gộp toàn bộ frames thành 1 Batch và đẩy thẳng lên GPU
            frames_list = [frame_dict[idx] for idx in valid_indices]
            
            # Chuyển NumPy sang Tensor: Shape [N, H, W, 3] -> [N, 3, H, W]
            tensor = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).float().to(device)
            N, C, H, W = tensor.shape
            
            # 2. Xử lý song song siêu tốc toàn bộ đặc trưng
            mean = tensor.mean(dim=[2, 3]) # Kết quả: [N, 3]
            std = tensor.std(dim=[2, 3])   # Kết quả: [N, 3]
            
            # Tính Percentile 25 và 75 trên GPU
            tensor_flat = tensor.reshape(N, C, -1)
            q25 = torch.quantile(tensor_flat, 0.25, dim=2)
            q75 = torch.quantile(tensor_flat, 0.75, dim=2)
            
            # 3. Nối lại thành ma trận Embedding tổng: Shape [N, 12]
            embeddings = torch.cat([mean, std, q25, q75], dim=1)
            
            # 4. Lọc tuyến tính khoảng cách L2
            filtered_indices = [valid_indices[0]]
            prev_embedding = embeddings[0]
            
            for i in range(1, len(valid_indices)):
                current_embedding = embeddings[i]
                
                # Tính L2 Norm trực tiếp trên GPU
                diff = current_embedding - prev_embedding
                l2_diff = torch.norm(diff)
                l2_prev = torch.norm(prev_embedding)
                
                distance = (l2_diff / l2_prev).item() if l2_prev > 1e-8 else 0.0
                
                if distance > self.config.threshold:
                    filtered_indices.append(valid_indices[i])
                    prev_embedding = current_embedding
            
            # Giải phóng VRAM
            del tensor, embeddings
            torch.cuda.empty_cache()
            
            return filtered_indices
            
        else:
            # Dự phòng: Chạy logic CPU cũ nếu không có card đồ họa
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