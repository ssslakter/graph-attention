"""Adaptive graph filtering layer"""

from typing import Optional
import torch.nn as nn, torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange


class AGFAttention(nn.Module):
    """
    Adaptive Graph Filtering Layer with OV Circuit and Multi-Head support.

    Flow:
    1. Estimator computes Adjacency A from Q, K.
    2. Input projected to Values V.
    3. Filter computes X' = Poly(A, V).
    4. Heads concatenated and projected via Output Linear.
    """

    _act_map = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "gelu": F.gelu,
        "identity": lambda x: x,
        "none": lambda x: x,
    }

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        order: int = 3,
        top_k: Optional[int] = None,
        basis: str = "monomial",
        alphas_act: str = "sigmoid",
        learn_zero_order: bool = False,
    ):
        super().__init__()
        self.num_heads, self.dim_head, self.order, self.k = num_heads, dim_head, order, top_k
        self.learn_zero_order, self.alphas_act_name = learn_zero_order, alphas_act.lower()
        self.basis, self.scale = basis.lower(), dim_head**-0.5

        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=True)
        self.to_out = nn.Linear(inner, dim, bias=True)

        # Coefficients
        self.alphas_raw = nn.Parameter(torch.randn(order + 1, num_heads) * 0.02)
        self.act = self._act_map.get(alphas_act.lower(), lambda x: x)

        # State keys for regularization (ephemeral dict is risky for DDP/JIT, used properties)
        self.register_buffer("last_adj", None, persistent=False)
        self.register_buffer("last_x", None, persistent=False)
        self.init_as_standard_attention()
    
    @property
    def alphas(self):
        return self.act(self.alphas_raw)

    @torch.no_grad()
    def init_as_standard_attention(self):
        if self.alphas_act_name == "sigmoid":
            zero_val, one_val = -5.0, 5.0
        elif self.alphas_act_name == "tanh":
            zero_val, one_val = 0.0, 5.0
        else:
            zero_val, one_val = 0.0, 1.0
        self.alphas_raw.fill_(zero_val)
        if self.order >= 1:
            self.alphas_raw[1].fill_(one_val)
        return self

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.num_heads
        self.last_x = x
        # Unpack: (B, N, 3*H*D) -> 3 * (B, H, N, D)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        attn_scores: Tensor = (q @ k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask_bc = mask.view(b, 1, 1, n)
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask_bc, float("-inf"))
            else:
                attn_scores = attn_scores.masked_fill(mask_bc == 0, float("-inf"))

        if self.k and self.k < n:
            top_val = attn_scores.topk(self.k, dim=-1)[0][..., -1:]
            attn_scores = attn_scores.masked_fill(attn_scores < top_val, float("-inf"))

        attn = attn_scores.softmax(dim=-1)
        self.last_adj = attn

        with torch.autocast(device_type=x.device.type, enabled=False):
            attn, v = attn.float(), v.float()
            alphas = self.act(self.alphas_raw).view(-1, 1, h, 1, 1).float()

            res = (alphas[0] * v) if self.learn_zero_order else 0
            v_prev, v_curr = None, v

            for i in range(1, self.order + 1):
                if self.basis == "monomial":
                    v_curr = attn @ v_curr
                else:
                    # Chebyshev Recurrence: T_k = 2(2A - I)T_{k-1} - T_{k-2}
                    # l_v represents (2A - I)T_{k-1}
                    l_v = 2 * (attn @ v_curr) - v_curr
                    v_next = (2 * l_v - v_prev) if v_prev is not None else l_v
                    v_prev, v_curr = v_curr, v_next

                res = res + alphas[i] * v_curr

        out = res.to(x.dtype).transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

    def get_reg_loss(self):
        """Computes Spectral Smoothness Loss"""
        if self.last_adj is None:
            return torch.tensor(0.0, device=self.to_out.weight.device)

        # Dirichlet Energy on Normalized Features
        adj = self.last_adj.float()
        x_n = F.normalize(self.last_x.detach().float(), dim=-1)

        # "bnd,bhnm,bmd->" calculates Tr(X^T L X) efficiently
        smooth = torch.einsum("bnd,bhnm,bmd->", x_n, adj, x_n)

        bs, heads, n, _ = adj.shape
        return -smooth / (bs * heads * n)


AGFLayer = AGFAttention  # Alias for backward compatibility
