import torch.nn as nn


class ViTBlock(nn.Module):
    def __init__(self, attention_layer, dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attention_layer(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, X):
        # High-order polynomial multi-head attention + MLP
        X = X + self.attn(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X
