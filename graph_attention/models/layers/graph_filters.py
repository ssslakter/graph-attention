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

    def __init__(self, order: int, num_heads: int, basis: str = "chebyshev"):
        super().__init__()
        self.order = order
        self.basis = basis.lower()
        if self.basis not in ["monomial", "chebyshev"]:
            raise ValueError(f"Unknown basis: {basis}")

        self.alphas = nn.Parameter(torch.zeros(order + 1, 1, num_heads, 1, 1))
        with torch.no_grad():
            self.alphas[0].fill_(1.0)
            self.alphas[1:].normal_(0, 0.01)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            adj, x, alphas = adj.float(), x.float(), self.alphas.float()
            alphas = torch.tanh(alphas)

            if self.basis == "monomial":
                out, curr = alphas[0] * x, x
                for k in range(1, self.order + 1):
                    curr = torch.matmul(adj, curr)
                    out = out + alphas[k] * curr
                return out

            t_prev = x
            out = alphas[0] * t_prev

            if self.order > 0:
                t_curr = torch.matmul(adj, x)
                out = out + alphas[1] * t_curr

                for k in range(2, self.order + 1):
                    t_next = 2 * torch.matmul(adj, t_curr) - t_prev
                    out = out + alphas[k] * t_next
                    t_prev, t_curr = t_curr, t_next

            return out
