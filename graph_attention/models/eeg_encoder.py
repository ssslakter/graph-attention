import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import AGFAttention


class DownProjector(nn.Module):
    """Projects raw multichannel EEG into a compact (B, 32, T) feature map.

    Args:
        eeg_channels: Number of EEG electrodes (22 for BCI-2a).
        dropout: Dropout probability.
    """

    def __init__(self, eeg_channels: int = 22, dropout: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1)),
            nn.BatchNorm2d(16),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, eeg_channels)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(7, 1), stride=(7, 1)),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 7, 8)),  # explicit asymmetric padding for kernel_size=16
            nn.Conv2d(32, 32, kernel_size=(16, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(7, 1), stride=(7, 1)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(x)))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, v = self.proj(x).chunk(2, dim=-1)
        return self.out(F.silu(w) * v)


class StableTransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm and SwiGLU FFN."""

    def __init__(self, dim: int = 32, num_heads: int = 2, mlp_ratio: float = 4.0, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = AGFAttention(dim=dim, num_heads=num_heads, **attn_kwargs)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StableTransformer(nn.Module):
    """Stack of StableTransformerBlocks."""

    def __init__(self, dim: int = 32, num_heads: int = 2, depth: int = 4, mlp_ratio: float = 4.0, **attn_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StableTransformerBlock(dim, num_heads, mlp_ratio, **attn_kwargs) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class CausalConv1d(nn.Module):
    """Causal 1-D convolution — left-pads so the time dimension is preserved."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


def _tcn_layer(channels: int, kernel_size: int, dilation: int, activation: nn.Module, dropout: float) -> nn.Sequential:
    """Single TCN layer: CausalConv1d → BatchNorm → Activation → Dropout."""
    return nn.Sequential(
        CausalConv1d(channels, channels, kernel_size, dilation),
        nn.BatchNorm1d(channels),
        activation,
        nn.Dropout(dropout),
    )


class TCN(nn.Module):
    """Temporal Convolutional Network with two dilated residual stages.

    Stage 1 (dilation=1, SiLU) feeds into Stage 2 (dilation=2, ReLU) with a
    skip connection, finishing with a ReLU.

    Args:
        channels: Number of input/output channels (32).
        kernel_size: Convolutional kernel size (3).
        dropout: Dropout probability (0.3).

    Input:  (B, C, T)
    Output: (B, T, C)
    """

    def __init__(self, channels: int = 32, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stage1 = nn.Sequential(
            _tcn_layer(channels, kernel_size, dilation=1, activation=nn.SiLU(), dropout=dropout),
            _tcn_layer(channels, kernel_size, dilation=1, activation=nn.SiLU(), dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            _tcn_layer(channels, kernel_size, dilation=2, activation=nn.ReLU(), dropout=dropout),
            _tcn_layer(channels, kernel_size, dilation=2, activation=nn.ReLU(), dropout=dropout),
        )
        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.final_act(self.stage2(x) + x)
        return x.transpose(1, 2)


class DSTS(nn.Module):
    """Dual-Stream Temporal-Spatial block.

    Fuses local temporal features (TCN) with global spatial features
    (StableTransformer) via element-wise addition of their last hidden states.

    Args:
        in_channels: Channel dimension of the input feature map (32).
        dim: Model dimension shared across both streams (32).
        num_classes: Number of output classes (4 for BCI-2a).
        depth: Number of transformer blocks (4).
        num_heads: Number of attention heads (2).
        dropout: Dropout probability (0.3).
        attn_kwargs: Extra keyword arguments forwarded to AGFAttention.
    """

    def __init__(
        self,
        in_channels: int = 32,
        dim: int = 32,
        num_classes: int = 4,
        depth: int = 4,
        num_heads: int = 2,
        dropout: float = 0.3,
        **attn_kwargs,
    ):
        super().__init__()
        self.tcn = TCN(channels=in_channels, dropout=dropout)
        self.transformer = StableTransformer(dim=dim, num_heads=num_heads, depth=depth, **attn_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h_tcn = self.tcn(x)[:, -1, :]                        # (B, dim) — last time step
        h_tf = self.transformer(x.transpose(1, 2))[:, -1, :] # (B, dim)
        return self.mlp(h_tcn + h_tf)


class EEGEncoder(nn.Module):
    """EEG classification model for motor-imagery BCI.

    Combines a DownProjector front-end with an ensemble of parallel
    Dropout + DSTS branches whose logits are averaged at inference.

    Args:
        eeg_channels: Number of input EEG electrodes (22 for BCI-2a).
        num_classes: Number of MI tasks (4 for BCI-2a).
        num_branches: Number of parallel DSTS branches (5).
        dropout: Dropout probability (0.3).
        attn_kwargs: Extra keyword arguments forwarded to AGFAttention.
    """

    def __init__(
        self,
        eeg_channels: int = 22,
        num_classes: int = 4,
        num_branches: int = 5,
        dropout: float = 0.3,
        **attn_kwargs,
    ):
        super().__init__()
        self.down_projector = DownProjector(eeg_channels=eeg_channels, dropout=dropout)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    DSTS(in_channels=32, dim=32, num_classes=num_classes, dropout=dropout, **attn_kwargs),
                )
                for _ in range(num_branches)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw EEG of shape (B, 1, 1125, 22).
        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.down_projector(x).squeeze(-1)  # (B, 32, T)
        return torch.stack([branch(x) for branch in self.branches]).mean(dim=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 1, 1125, 22)
    model = EEGEncoder()
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(logits.shape)}")