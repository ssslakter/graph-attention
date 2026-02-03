from typing import Optional, List, Union, Callable, Dict, Any
from torchvision.transforms import v2
from torch.utils.data import Dataset
import torch

_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_dataset(
    name: str,
    cls: Callable,
    mean: tuple,
    std: tuple,
    transform_factory: Callable[[bool, str, str], Any],
):
    """
    Registers a dataset.

    Args:
        name: Unique key (e.g. 'cifar10').
        cls: The Dataset class (or a wrapper function) accepting (root, train, download, transform).
        mean: Normalization mean.
        std: Normalization std.
        transform_factory: Function (train, augment, part) -> transforms.
    """
    _DATASET_REGISTRY[name] = {
        "cls": cls,
        "mean": mean,
        "std": std,
        "transform_factory": transform_factory,
    }


def get_registry_info(variant: str):
    return _DATASET_REGISTRY[variant]


def get_transforms(variant: str, train: bool = True, augmentation: str = "standard") -> v2.Compose:
    """
    Retrieves the CPU-only transforms (Resize, ToImage, ToDtype uint8).
    """
    info = get_registry_info(variant)
    return info["transform_factory"](train, augmentation, "cpu")


def get_batch_transforms(variant: str, num_classes: int, augmentation: str = "standard", train: bool = True):
    """
    Returns GPU-ready batch transforms (Augmentations, Float conversion, Normalize, MixUp).
    """
    info = get_registry_info(variant)
    transforms = []

    if train:
        gpu_augs = info["transform_factory"](train, augmentation, "gpu")
        if gpu_augs is not None:
            transforms.append(gpu_augs)

    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.Normalize(mean=info["mean"], std=info["std"]))

    if train and augmentation == "strong":
        transforms.append(
            v2.RandomChoice(
                [v2.MixUp(alpha=0.8, num_classes=num_classes), v2.CutMix(alpha=1.0, num_classes=num_classes)]
            )
        )

    return v2.Compose(transforms)


def get_dataset(
    variant: str,
    root: str = "./data",
    train: bool = True,
    transforms: Optional[Union[List, Callable, v2.Compose]] = None,
    augment: str = "standard",
    download: bool = True,
    **kwargs,
) -> Dataset:
    """Factory to retrieve a dataset instance by name.

    By default, this applies CPU-only transforms (e.g., resizing to uint8 tensors)
    via the registered factory, optimized for efficient transfer to GPU.

    Args:
        variant (str): Registered dataset name (e.g., "cifar10", "imagenet").
        root (str): Root directory for data storage.
        train (bool): If True, loads the training split.
        transforms (Optional[Union[List, Callable, v2.Compose]]): Custom transforms.
            If None, defaults to the factory's "cpu" pipeline.
        augment (str): Augmentation strategy ("none", "standard", "strong").
        download (bool): Whether to download the dataset.
        **kwargs: Additional arguments passed to the dataset class.

    Returns:
        Dataset: The initialized dataset instance.
    """
    info = get_registry_info(variant)

    if transforms is None:
        transforms = info["transform_factory"](train, augment, "cpu")

    return info["cls"](root=root, train=train, transform=transforms, download=download, **kwargs)
