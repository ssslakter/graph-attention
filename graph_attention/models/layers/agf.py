"""Adaptive graph filtering layer"""

import torch.nn as nn, torch
from typing import Optional, Union, Callable
from .graph_estimators import GraphEstimator
from .graph_filters import GraphFilter


class AGFAttention(nn.Module):
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
        dim: int,
        heads: int,
        dim_head: int = 64,
        graph_estimator: Union[GraphEstimator, Callable] = None,
        graph_filter: Union[GraphFilter, Callable] = None,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # Factories
        self.attend = graph_estimator(dim=dim, num_heads=heads, dim_head=dim_head) if callable(graph_estimator) else graph_estimator
        self.graph_filter = graph_filter(num_heads=heads) if callable(graph_filter) else graph_filter

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        self.norm_v = nn.LayerNorm(dim_head) 

        self.last_adj = None
        self.last_input = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        
        attn = self.attend(x, mask)
        self.last_adj = attn
        self.last_input = x

        # (B, N, H*D) -> (B, N, H, D)
        values = self.to_v(x).view(batch_size, num_nodes, self.num_heads, self.dim_head)
        values = self.norm_v(values)
        # Transpose for filter: (B, H, N, D)
        values = values.transpose(1, 2)

        values_filtered = self.graph_filter(attn, values)
        values_merged = values_filtered.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        return self.to_out(values_merged)

    def get_regularization_loss(self, lambda_smooth: float = 0.01) -> torch.Tensor:
        """
        Computes the spectral smoothness loss for this specific layer.
        (Minimizes Dirichlet Energy).
        """
        if self.last_adj is None or self.last_input is None:
            return torch.tensor(0.0, device=self.to_out.weight.device)

        adj = self.last_adj.float()
        input_features = self.last_input.detach().float()

        input_norm = torch.nn.functional.normalize(input_features, p=2, dim=-1)

        smoothness = torch.einsum("bnd,bhnm,bmd->", input_norm, adj, input_norm)

        batch_size, num_heads, num_nodes, _ = adj.shape
        normalization = batch_size * num_heads * num_nodes

        loss = -1.0 * (smoothness / normalization)

        return loss * lambda_smooth

AGFLayer = AGFAttention # Alias for backward compatibility