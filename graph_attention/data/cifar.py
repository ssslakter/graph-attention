from torchvision.transforms import v2
from torchvision.datasets import CIFAR10, CIFAR100
from .utils import register_dataset, get_dataset
import torch

STATS = {
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
}


def _cifar_transforms_factory(train: bool, augmentation: str, part: str):
    if part == "cpu":
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    if part == "gpu":
        if not train:
            return None

        base_aug = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]

        aug_strategies = {
            "none": [],
            "standard": base_aug + [v2.TrivialAugmentWide()],
            "strong": base_aug + [v2.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31)],
        }

        if augmentation not in aug_strategies:
            raise ValueError(f"Unknown augmentation: {augmentation}")

        transform_list = aug_strategies[augmentation]

        if augmentation == "pretrain":
            transform_list.append(v2.RandomErasing(p=0.25, value="random"))

        return v2.Compose(transform_list)

    raise ValueError(f"Unknown part: {part}")


for name, cls in [("cifar10", CIFAR10), ("cifar100", CIFAR100)]:
    register_dataset(
        name=name,
        cls=cls,
        mean=STATS[name]["mean"],
        std=STATS[name]["std"],
        transform_factory=_cifar_transforms_factory,
    )


def get_cifar(variant="cifar10", **kwargs):
    kwargs["variant"] = variant
    return get_dataset(**kwargs)
