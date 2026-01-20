import torch
import torch.nn as nn
from abc import ABC, abstractmethod


__all__ = ["GraphFilter", "PolynomialFilter"]


class GraphFilter(nn.Module, ABC):
    """
    Abstract strategy for propagating information across the graph.

    This encapsulates the 'Graph Filtering' phase (Section 4.2),
    handling multi-hop aggregation or spectral filtering.
    """

    @abstractmethod
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: Adjacency Matrix (Batch, Seq_Len, Seq_Len)
            x: Node Features (Batch, Seq_Len, Dim)

        Returns:
            Filtered Features (Batch, Seq_Len, Dim)
        """
        pass


class PolynomialFilter(GraphFilter):
    """
    Applies Graph Filtering independently per head.

    The coefficients alpha are learned per head to allow different
    filtering characteristics (smoothing vs sharpening) in different subspaces.
    """

    def __init__(self, order_K: int, num_heads: int):
        super().__init__()
        self.K = order_K

        # Alphas shape: (K+1, 1, Num_Heads, 1, 1) for broadcasting
        # We broadcast over Batch (dim 1) and Seq/Dim (dim 3, 4)
        self.alphas = nn.Parameter(torch.randn(order_K + 1, 1, num_heads, 1, 1))

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # adj: (B, H, N, N)
        # x (Values): (B, H, N, Head_Dim)

        # Hop 0
        output = self.alphas[0] * x
        current_x = x

        for k in range(1, self.K + 1):
            # A @ X performs batch matmul over (N, N) @ (N, D)
            current_x = torch.matmul(adj, current_x)
            output = output + self.alphas[k] * current_x

        return output
