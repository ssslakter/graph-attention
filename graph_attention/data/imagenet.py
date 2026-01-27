import torchvision.transforms as tfm
from torchvision.datasets import ImageNet
from .utils import register_dataset

# Standard ImageNet-1k Statistics
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


class ImageNetAdapter(ImageNet):
    """
    Wraps torchvision.datasets.ImageNet to accept a generic 'train' boolean arg.
    
    Note: 'download' argument is accepted to satisfy the interface, but ignored.
    ImageNet-1k must be manually downloaded and extracted into the root directory.
    """

    def __init__(self, root, train=True, download=False, transform=None, **kwargs):
        split = "train" if train else "val"
        # We purposely ignore 'download' here as standard ImageNet class 
        # does not support automatic downloading.
        super().__init__(root=root, split=split, transform=transform, **kwargs)


def _imagenet_transform_factory(train: bool, augmentation: str, normalize: bool):
    normalization = [tfm.Normalize(MEAN, STD)] if normalize else []

    if not train:
        return tfm.Compose([
            tfm.Resize(256), 
            tfm.CenterCrop(SIZE), 
            tfm.ToTensor()
        ] + normalization)

    aug_strategies = {
        "none": [
            tfm.Resize(256), 
            tfm.CenterCrop(SIZE)
        ],
        "standard": [
            tfm.RandomResizedCrop(SIZE), 
            tfm.RandomHorizontalFlip()
        ],
        "strong": [
            tfm.RandomResizedCrop(SIZE),
            tfm.RandomHorizontalFlip(),
            tfm.RandAugment(num_ops=2, magnitude=9),
        ],
    }

    if augmentation not in aug_strategies:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    return tfm.Compose(aug_strategies[augmentation] + [tfm.ToTensor()] + normalization)


register_dataset(
    name="imagenet", 
    cls=ImageNetAdapter, 
    mean=MEAN, 
    std=STD, 
    transform_factory=_imagenet_transform_factory
)