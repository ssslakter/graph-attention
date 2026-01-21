"""Adaptive graph filtering layer"""

import torch.nn as nn, torch
from typing import Optional, Union, Callable
from .graph_estimators import GraphEstimator
from .graph_filters import GraphFilter


class AGFLayer(nn.Module):
    """
    Adaptive Graph Filtering Layer with OV Circuit and Multi-Head support.

    Flow:
    1. Estimator computes Adjacency A from Q, K.
    2. Input projected to Values V.
    3. Filter computes X' = Poly(A, V).
    4. Heads concatenated and projected via Output Linear.
    """

    def __init__(
        self,
        heads: int,
        dim: int,
        graph_estimator: Union[GraphEstimator, Callable],
        graph_filter: Union[GraphFilter, Callable],
        **kwargs,
    ):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads

        # Handle instantiation if factory/partial is passed
        if callable(graph_estimator):
            self.graph_estimator = graph_estimator(dim=dim, num_heads=heads)
        else:
            self.graph_estimator = graph_estimator

        if callable(graph_filter):
            self.graph_filter = graph_filter(num_heads=heads)
        else:
            self.graph_filter = graph_filter

        self.W_V = nn.Linear(dim, dim)
        self.W_O = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape

        # 1. Construct Graph (Multi-Head Adjacency)
        # Returns (B, H, N, N)
        adj_matrix = self.graph_estimator(x, mask)

        self.last_adj = adj_matrix
        self.last_x = x
        # 2. Project Values and Reshape
        # (B, N, D) -> (B, N, H, D_h) -> (B, H, N, D_h)
        v = self.W_V(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Filter to Values
        # Returns (B, H, N, D_h)
        v_filtered = self.graph_filter(adj_matrix, v)

        # 4. Merge Heads
        # (B, H, N, D_h) -> (B, N, H, D_h) -> (B, N, D)
        v_merged = v_filtered.transpose(1, 2).contiguous().view(B, N, -1)

        # 5. Final Output Projection
        return self.W_O(v_merged)

    def get_regularization_loss(self, lambda_smooth: float = 0.01) -> torch.Tensor:
        """
        Computes the spectral smoothness loss for this specific layer.
        (Minimizes Dirichlet Energy).
        """
        if self.last_adj is None or self.last_x is None:
            return torch.tensor(0.0, device=self.W_O.weight.device)

        A = self.last_adj.float()
        X = self.last_x.detach().float()

        X_norm = torch.nn.functional.normalize(X, p=2, dim=-1)

        smoothness = torch.einsum("bnd,bhnm,bmd->", X_norm, A, X_norm)

        B, H, N, _ = A.shape
        normalization = B * H * N

        loss = -1.0 * (smoothness / normalization)

        return loss * lambda_smooth
