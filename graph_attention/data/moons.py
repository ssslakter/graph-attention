from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _make_moons(
    n_samples: int,
    noise: float,
    seed: int | None = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    theta_out = rng.random(n_samples_out) * np.pi
    theta_in = rng.random(n_samples_in) * np.pi

    outer = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)
    inner = np.stack([1.0 - np.cos(theta_in), 0.5 - np.sin(theta_in)], axis=1)

    x = np.vstack([outer, inner]).astype(np.float32)
    y = np.concatenate(
        [np.zeros(n_samples_out, dtype=np.int64), np.ones(n_samples_in, dtype=np.int64)],
        axis=0,
    )

    if noise > 0:
        x += rng.normal(scale=noise, size=x.shape).astype(np.float32)

    if shuffle:
        idx = rng.permutation(n_samples)
        x = x[idx]
        y = y[idx]

    return x, y


@dataclass(frozen=True)
class MoonsStats:
    mean: torch.Tensor
    std: torch.Tensor


class MoonsDataset(Dataset):
    classes = ["moon_0", "moon_1"]

    def __init__(self, features: np.ndarray, labels: np.ndarray, stats: MoonsStats):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.stats = stats

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        x = (self.features[idx] - self.stats.mean) / self.stats.std
        y = self.labels[idx]
        return x, y


def _compute_stats(features: np.ndarray) -> MoonsStats:
    tensor = torch.as_tensor(features, dtype=torch.float32)
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0).clamp_min(1e-6)
    return MoonsStats(mean=mean, std=std)


def build_moons_dataloaders(
    n_samples: int = 2000,
    noise: float = 0.1,
    train_ratio: float = 0.8,
    seed: int | None = 42,
    batch_size: int = 128,
    valid_batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, MoonsStats]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1 (got {train_ratio}).")

    features, labels = _make_moons(n_samples=n_samples, noise=noise, seed=seed, shuffle=True)
    split_idx = int(n_samples * train_ratio)

    train_features, valid_features = features[:split_idx], features[split_idx:]
    train_labels, valid_labels = labels[:split_idx], labels[split_idx:]

    stats = _compute_stats(train_features)

    train_ds = MoonsDataset(train_features, train_labels, stats)
    valid_ds = MoonsDataset(valid_features, valid_labels, stats)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dl, valid_dl, stats
