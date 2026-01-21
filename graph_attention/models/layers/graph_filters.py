import torch
import torch.nn as nn
from abc import ABC, abstractmethod


__all__ = ["GraphFilter", "PolynomialFilter"]


class GraphFilter(nn.Module, ABC):
    """
    Abstract strategy for propagating information across the graph.

    This encapsulates the 'Graph Filtering' phase,
    handling multi-hop aggregation or spectral filtering.
    """

    @abstractmethod
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: Adjacency Matrix (Batch, Heads, Num_Nodes, Num_Nodes)
            x: Node Features (Batch, Heads, Num_Nodes, Dim)

        Returns:
            Filtered Features (Batch, Heads, Num_Nodes, Dim)
        """
        pass


class PolynomialFilter(GraphFilter):
    """
    Applies Graph Filtering independently per head.

    The coefficients alpha are learned per head to allow different
    filtering characteristics (smoothing vs sharpening) in different subspaces.
    """

    def __init__(self, order: int, num_heads: int):
        super().__init__()
        self.order = order

        self.alphas = nn.Parameter(torch.zeros(order + 1, 1, num_heads, 1, 1))
        nn.init.normal_(self.alphas, std=0.01)
        with torch.no_grad():
            self.alphas[0] += 1.0
            self.alphas[1] += 1.0

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # adj: (bs, heads, seq_len, seq_len)
        # x: (bs, heads, seq_len, dim_head)
        with torch.autocast(device_type="cuda", enabled=False):
            adj = adj.float()
            x = x.float()

            alphas = self.alphas.float()
            output = alphas[0] * x
            current_x = x

            for k in range(1, self.order + 1):
                current_x = torch.matmul(adj, current_x)
                output = output + alphas[k] * current_x

            return output
