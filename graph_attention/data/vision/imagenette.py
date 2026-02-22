from torchvision.transforms import v2
from torchvision.datasets import Imagenette as TVImagenette
from .utils import register_dataset
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


class ImagenetteAdapter(TVImagenette):
    """
    Wraps torchvision.datasets.Imagenette to accept a generic 'train' boolean arg
    instead of 'split'.
    """

    def __init__(self, root, train=True, download=True, transform=None, **kwargs):
        split = "train" if train else "val"
        super().__init__(root=root, split=split, download=download, transform=transform, **kwargs)


def _imagenette_transform_factory(train: bool, augmentation: str, part: str):
    if part == "cpu":
        if not train or augmentation == "none":
            return v2.Compose([v2.Resize(256), v2.CenterCrop(SIZE), v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

        return v2.Compose(
            [v2.RandomResizedCrop(SIZE, scale=(0.08, 1.0)), v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)]
        )

    if part == "gpu":
        if not train:
            return None

        aug_strategies = {
            "none": [],
            "standard": [v2.RandomHorizontalFlip()],
            "strong": [
                v2.RandomHorizontalFlip(),
                v2.RandAugment(num_ops=2, magnitude=9),
            ],
        }

        if augmentation not in aug_strategies:
            raise ValueError(f"Unknown augmentation: {augmentation}")

        return v2.Compose(aug_strategies[augmentation])

    raise ValueError(f"Unknown part: {part}")


register_dataset(
    name="imagenette", cls=ImagenetteAdapter, mean=MEAN, std=STD, transform_factory=_imagenette_transform_factory
)
