import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer_tools.all import (
    Trainer,
    ProgressBarHook,
    LRSchedulerHook,
    AMPHook,
    GradClipHook,
    CheckpointHook,
    MetricsHook,
    BatchTransformHook,
    GradientAccumulationHook,
)
from trainer_tools.hooks.accelerate import AccelerateHook
from trainer_tools.hooks.metrics import Loss, Accuracy, LRStats
from graph_attention.data import get_dataset, get_transforms, get_batch_transforms, get_batch_mixup_cutmix
from graph_attention.models.attn_resnet import AttnResNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer_tools.utils import random_seed

SEED = 42
EPOCHS = 5
BATCH_SIZE = 256
LR = 8e-4
WEIGHT_DECAY = 0.05
NUM_CLASSES = 10
DATASET = "imagenette"
AUGMENTATION = "standard"
DEVICE = "cuda"
GRAD_ACUM = 1
PROJECT = "resnet18-attn"
RUN_NAME = "resnet18_base"

random_seed(SEED)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def make_dataloader(dataset, train=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        persistent_workers=train,
        prefetch_factor=1 if train else 2,
    )


def main():
    train_dl = make_dataloader(
        get_dataset(
            DATASET, "data", train=True, transforms=get_transforms(DATASET, train=True, augmentation=AUGMENTATION)
        )
    )
    valid_dl = make_dataloader(get_dataset(DATASET, "data", train=False), train=False)

    model = AttnResNet.load_from_timm(
        model_name="resnet18",
        num_classes=NUM_CLASSES,
        pretrained=True,
        attn_layer_indices=[],
        region_size=16,
    )
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dl) * EPOCHS)

    multi_gpu = torch.cuda.device_count() > 1
    hooks = [
        ProgressBarHook(),
        LRSchedulerHook(scheduler),
        *(
            [AccelerateHook(max_grad_norm=1.0, gradient_accumulation_steps=GRAD_ACUM, mixed_precision = 'bf16')]
            if multi_gpu
            else [
                AMPHook(dtype=torch.float16, device_type=DEVICE),
                GradClipHook(max_norm=1.0),
                GradientAccumulationHook(steps=GRAD_ACUM),
            ]
        ),
        BatchTransformHook(
            x_tfm=get_batch_transforms(DATASET, AUGMENTATION, train=True),
            x_tfms_valid=get_batch_transforms(DATASET, train=False),
            batch_tfms=get_batch_mixup_cutmix(NUM_CLASSES, AUGMENTATION, train=True),
        ),
        CheckpointHook(f"outputs/{RUN_NAME}/checkpoints", save_every_steps=5000),
        MetricsHook(
            verbose=True,
            metrics=[Loss(), Accuracy(), LRStats()],
            tracker_type="trackio",
            project=PROJECT,
            name=RUN_NAME,
        ),
    ]

    Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=optimizer,
        loss_func=nn.CrossEntropyLoss(),
        epochs=EPOCHS,
        hooks=hooks,
    ).fit()


if __name__ == "__main__":
    main()
