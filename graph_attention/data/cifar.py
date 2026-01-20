from typing import Literal, Optional, List, Union, Callable
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as tfm
from torch.utils.data import Dataset
import torch

DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "cls": CIFAR10
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "cls": CIFAR100
    }
}

def get_transforms(variant: str, train: bool = True, augmentation: str = "standard") -> tfm.Compose:
    stats = DATASET_STATS.get(variant)
    if not stats:
        raise ValueError(f"Unknown variant: {variant}")

    if not train:
        return tfm.Compose([tfm.ToTensor(), tfm.Normalize(stats["mean"], stats["std"])])

    aug_strategies = {
        "none": [],
        "standard": [tfm.RandomCrop(32, padding=4), tfm.RandomHorizontalFlip(), tfm.TrivialAugmentWide()],
        "strong": [tfm.RandomCrop(32, padding=4), tfm.RandomHorizontalFlip(), tfm.RandAugment(num_ops=2, magnitude=9)],
    }

    if augmentation not in aug_strategies:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    transforms = aug_strategies[augmentation] + [tfm.ToTensor(), tfm.Normalize(stats["mean"], stats["std"])]
    return tfm.Compose(transforms)

def get_cifar(
    root: str = "./data",
    train: bool = True,
    transforms: Optional[Union[List, Callable, tfm.Compose]] = None,
    variant: Literal["cifar10", "cifar100"] = "cifar10",
) -> Dataset:
    stats = DATASET_STATS.get(variant)
    if not stats:
        raise ValueError(f"Unknown variant: {variant}")
    
    if transforms is None:
        transforms = get_transforms(variant, train=train)
    elif isinstance(transforms, list):
        transforms = tfm.Compose(transforms + [tfm.Normalize(stats["mean"], stats["std"])])

    return stats["cls"](root=root, train=train, download=True, transform=transforms)
