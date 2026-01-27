# utils.py
from typing import Optional, List, Union, Callable, Dict, Any, Type
import torchvision.transforms as tfm
from torch.utils.data import Dataset
import torch

_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_dataset(
    name: str, 
    cls: Callable, 
    mean: tuple, 
    std: tuple, 
    transform_factory: Callable[[bool, str, bool], tfm.Compose]
):
    """
    Registers a dataset.
    
    Args:
        name: Unique key (e.g. 'cifar10').
        cls: The Dataset class (or a wrapper function) accepting (root, train, download, transform).
        mean: Normalization mean.
        std: Normalization std.
        transform_factory: Function (train, augment, normalize) -> transforms.
    """
    _DATASET_REGISTRY[name] = {
        "cls": cls,
        "mean": mean,
        "std": std,
        "transform_factory": transform_factory
    }

def get_registry_info(variant: str):
    return _DATASET_REGISTRY[variant]


def get_transforms(
    variant: str, 
    train: bool = True, 
    augmentation: str = "standard", 
    normalize: bool = True
) -> tfm.Compose:
    """
    Retrieves the default transforms for a specific dataset variant.
    """
    info = get_registry_info(variant)
    return info["transform_factory"](train, augmentation, normalize)

def get_dataset(
    variant: str,
    root: str = "./data",
    train: bool = True,
    transforms: Optional[Union[List, Callable, tfm.Compose]] = None,
    augment: str = "standard",
    normalize: bool = True,
    download: bool = True,
    **kwargs
) -> Dataset:
    """
    General factory to get a dataset by name.
    """
    info = get_registry_info(variant)
    
    if transforms is None:
        transforms = info["transform_factory"](train, augment, normalize)
    elif isinstance(transforms, list):
        if not any(isinstance(t, tfm.ToTensor) for t in transforms):
            transforms = transforms + [tfm.ToTensor()]
        if normalize:
            transforms = transforms + [tfm.Normalize(info["mean"], info["std"])]
        transforms = tfm.Compose(transforms)

    return info["cls"](
        root=root, 
        train=train, 
        transform=transforms, 
        download=download, 
        **kwargs
    )