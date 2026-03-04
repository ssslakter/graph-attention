import torch
import torch.nn as nn
from .layers import *
from .backbone import Transformer


class DSTS(nn.Model):
    def __init__(self, backbone: Transformer):
        super().__init__()


class DownProjector(nn.Module):
    def __init__(self, eeg_channels=22, dropout_rate=0.3):
        """
        Args:
            eeg_channels (int): Dataset dependent. Number of EEG electrodes (e.g., 22 for BCI-2a).
            dropout_rate (float): Architecture decision. Regularization factor.
        """
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(1, 1)),
            nn.BatchNorm2d(16)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, eeg_channels)),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(7, 1), stride=(7, 1)),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(16, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(7, 1), stride=(7, 1)),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class EEGEncoder(nn.Module):
    def __init__(self, n_blocks: int = 4):
        super().__init__()
        self.backbone = DSTS
        self.n_blocks = n_blocks 