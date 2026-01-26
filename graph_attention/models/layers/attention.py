from einops import rearrange
import torch.nn as nn, torch


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        dim_head = dim // num_heads
        self.heads = num_heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # Mask should be broadcastable to (b, h, n, n)
            mask_value = -torch.finfo(attn_scores.dtype).max
            attn_scores = attn_scores.masked_fill(mask == 0, mask_value)

        attn = self.attend(attn_scores)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
