import torchvision.transforms as tfm
from torchvision.datasets import Imagenette as TVImagenette
from .utils import register_dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224  # Standard ViT/ImageNet size


class ImagenetteAdapter(TVImagenette):
    """
    Wraps torchvision.datasets.Imagenette to accept a generic 'train' boolean arg
    instead of 'split'.
    """

    def __init__(self, root, train=True, download=True, transform=None, **kwargs):
        split = "train" if train else "val"
        super().__init__(root=root, split=split, download=download, transform=transform, **kwargs)


def _imagenette_transform_factory(train: bool, augmentation: str, normalize: bool):
    normalization = [tfm.Normalize(MEAN, STD)] if normalize else []

    if not train:
        return tfm.Compose([tfm.Resize(256), tfm.CenterCrop(SIZE), tfm.ToTensor()] + normalization)

    aug_strategies = {
        "none": [tfm.Resize(256), tfm.CenterCrop(SIZE)],
        "standard": [tfm.RandomResizedCrop(SIZE, scale=(0.08, 1.0)), tfm.RandomHorizontalFlip()],
        "strong": [
            tfm.RandomResizedCrop(SIZE, scale=(0.08, 1.0)),
            tfm.RandomHorizontalFlip(),
            tfm.RandAugment(num_ops=2, magnitude=9),
        ],
    }

    if augmentation not in aug_strategies:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    return tfm.Compose(aug_strategies[augmentation] + [tfm.ToTensor()] + normalization)


register_dataset(
    name="imagenette", cls=ImagenetteAdapter, mean=MEAN, std=STD, transform_factory=_imagenette_transform_factory
)
