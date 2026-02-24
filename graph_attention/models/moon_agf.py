from __future__ import annotations

import torch
from torch import nn

from .layers import AGFAttention


class MoonsAGFNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        model_dim: int = 64,
        num_heads: int = 4,
        dim_head: int = 16,
        order: int = 3,
        top_k: int | None = None,
        basis: str = "monomial",
        alphas_act: str = "sigmoid",
        mlp_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embed = nn.Linear(1, model_dim)
        self.attn = AGFAttention(
            dim=model_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            order=order,
            top_k=top_k,
            basis=basis,
            alphas_act=alphas_act,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_dim, num_classes),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (B, {self.input_dim}), got {tuple(x.shape)}.")

        x = x.unsqueeze(-1)  # (B, 2, 1)
        x = self.embed(x)  # (B, 2, D)
        x = self.attn(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
