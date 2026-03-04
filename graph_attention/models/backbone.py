import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, attention_layer, dim=384, num_heads=6, mlp_ratio=4.0, **agf_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attention_layer(dim=dim, num_heads=num_heads, **agf_kwargs)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, X):
        X = X + self.attn(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X


class Transformer(nn.Module):
    """A sequence-to-sequence transformer backbone."""

    def __init__(self, depth: int, dim: int, num_heads: int, attention_layer, mlp_ratio=4.0, **agf_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(attention_layer, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, **agf_kwargs)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (batch, tokens, dim)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
