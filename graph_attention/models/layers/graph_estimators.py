import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

__all__ = ["GraphEstimator", "AttentionStyleEstimator"]


class GraphEstimator(nn.Module, ABC):
    """
    Abstract strategy for constructing the graph structure (Adjacency Matrix)
    from node features.

    This encapsulates the 'Graph Construction' phase (Section 4.1),
    including scoring, sparsification, and normalization.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features (Batch, Seq_Len, Dim)
            mask: Optional attention mask (Batch, Seq_Len, Seq_Len)

        Returns:
            Normalized Adjacency Matrix (Batch, Seq_Len, Seq_Len)
        """
        pass


class AttentionStyleEstimator(GraphEstimator):
    """
    Computes Multi-Head Adjacency Matrices.

    Structure:
        Q, K -> Split Heads -> Scaled Dot Product -> TopK Sparsity -> Softmax
    """

    def __init__(self, dim: int, num_heads: int, k_neighbors: int):
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = k_neighbors
        self.scale = self.head_dim**-0.5

        # Projections to [Batch, Seq, Num_Heads * Head_Dim]
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape

        # 1. Project and Reshape to (B, Heads, N, Head_Dim)
        q = self.W_Q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Compute Scores: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask shape usually (B, 1, 1, N) or (B, 1, N, N) for broadcasting
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 3. Apply Sparsity per head
        if self.k < N:
            top_k_vals, _ = torch.topk(scores, self.k, dim=-1)
            min_k = top_k_vals[..., -1].unsqueeze(-1)
            scores = torch.where(scores < min_k, torch.tensor(float("-inf"), device=x.device), scores)

        return torch.softmax(scores, dim=-1)
