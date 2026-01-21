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

    def __init__(self, dim: int, num_heads: int, dim_head: int, k_neighbors: int):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.k_neighbors = k_neighbors

        inner_dim = dim_head * num_heads

        # Projections to [Batch, Seq, Num_Heads * Head_Dim]
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, num_nodes, _ = x.shape
        query = self.to_q(x).view(bs, num_nodes, self.num_heads, self.dim_head).transpose(1, 2)
        key = self.to_k(x).view(bs, num_nodes, self.num_heads, self.dim_head).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        if self.k_neighbors < num_nodes:
            top_k_vals, _ = torch.topk(scores, self.k_neighbors, dim=-1)
            min_k = top_k_vals[..., -1].unsqueeze(-1)
            scores = torch.where(scores < min_k, torch.tensor(float("-inf"), device=x.device), scores)

        return torch.softmax(scores, dim=-1)
