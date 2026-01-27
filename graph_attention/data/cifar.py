import torchvision.transforms as tfm
from torchvision.datasets import CIFAR10, CIFAR100
from .utils import register_dataset, get_dataset

STATS = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)}
}

def _cifar_transforms_factory(variant: str):
    """Returns a factory function bound to the specific variant stats."""
    mean, std = STATS[variant]["mean"], STATS[variant]["std"]
    
    def factory(train: bool, augmentation: str, normalize: bool):
        normalization = [tfm.Normalize(mean, std)] if normalize else []
        
        if not train:
            return tfm.Compose([tfm.ToTensor()] + normalization)
            
        base_aug = [tfm.RandomCrop(32, padding=4), tfm.RandomHorizontalFlip()]
        
        aug_strategies = {
            "none": [],
            "standard": base_aug + [tfm.TrivialAugmentWide()],
            "strong": base_aug + [tfm.RandAugment(num_ops=2, magnitude=9)],
        }
        
        if augmentation not in aug_strategies:
            raise ValueError(f"Unknown augmentation: {augmentation}")
            
        return tfm.Compose(aug_strategies[augmentation] + [tfm.ToTensor()] + normalization)
    
    return factory

for name, cls in [("cifar10", CIFAR10), ("cifar100", CIFAR100)]:
    register_dataset(
        name=name,
        cls=cls,
        mean=STATS[name]["mean"],
        std=STATS[name]["std"],
        transform_factory=_cifar_transforms_factory(name)
    )

def get_cifar(variant="cifar10", **kwargs):
    kwargs['variant'] = variant
    return get_dataset(**kwargs)