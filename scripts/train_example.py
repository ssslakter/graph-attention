import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer_tools.all import (
    Trainer,
    ProgressBarHook,
    LRSchedulerHook,
    AMPHook,
    GradClipHook,
    BatchTransformHook,
    CheckpointHook,
    MetricsHook,
)
from trainer_tools.hooks.metrics import Loss, Accuracy, LRStats
from graph_attention.data import get_dataset, get_transforms, get_batch_transforms
from graph_attention.training.utils import PrefetchLoader
from graph_attention.models.attn_resnet import AttnResNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer_tools.utils import random_seed
import logging

# --- Configuration Constants ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Model Config
MODEL_PRETRAINED = "resnet18"
MODEL_ATTN_LAYER_INDICES = []
MODEL_REGION_SIZE = 16
MODEL_NUM_CLASSES = 10
MODEL_CHANNELS = 3

# Dataset Config
DATASET_VARIANT = "imagenette"
DATASET_ROOT = "data"
DATASET_AUGMENTATION = "standard"

# Dataloader Config
DATALOADER_BATCH_SIZE = 256
DATALOADER_VALID_BATCH_SIZE = 256
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
DATALOADER_PERSISTENT_WORKERS = True
DATALOADER_PREFETCH_FACTOR = 1

# Optimizer Config
OPTIMIZER_LR = 8e-4
OPTIMIZER_WEIGHT_DECAY = 0.05
OPTIMIZER_FUSED = True

# Scheduler Config
SCHEDULER_T_MAX_FACTOR = 1  # Will be multiplied by total_steps

# Training Config
TRAINING_SEED = 42
TRAINING_EPOCHS = 5
TRAINING_USE_AMP = True
TRAINING_AMP_DTYPE = torch.float16
TRAINING_DEVICE = "cuda"
TRAINING_GRAD_CLIP = 1.0
TRAINING_LABEL_SMOOTHING = 0.0

# Torch Config
TORCH_COMPILE = False
TORCH_MATMUL_PRECISION = "high"

# --- Main Training Script ---

# Set random seed
random_seed(TRAINING_SEED)

# Set torch settings
torch.set_float32_matmul_precision(TORCH_MATMUL_PRECISION)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Data Loading ---
train_transforms = get_transforms(DATASET_VARIANT, train=True, augmentation=DATASET_AUGMENTATION)
train_dataset = get_dataset(DATASET_VARIANT, DATASET_ROOT, train=True, transforms=train_transforms)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=DATALOADER_BATCH_SIZE,
    shuffle=True,
    num_workers=DATALOADER_NUM_WORKERS,
    pin_memory=DATALOADER_PIN_MEMORY,
    persistent_workers=DATALOADER_PERSISTENT_WORKERS,
    prefetch_factor=DATALOADER_PREFETCH_FACTOR,
)

valid_dataset = get_dataset(DATASET_VARIANT, DATASET_ROOT, train=False)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=DATALOADER_VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=DATALOADER_PIN_MEMORY,
    persistent_workers=False,
    prefetch_factor=2,
)

# --- Model ---
model = AttnResNet.load_from_timm(
    model_name=MODEL_PRETRAINED,
    num_classes=MODEL_NUM_CLASSES,
    pretrained=True,
    attn_layer_indices=MODEL_ATTN_LAYER_INDICES,
    region_size=MODEL_REGION_SIZE,
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

if TORCH_COMPILE:
    model = torch.compile(model)

optimizer = AdamW(model.parameters(), lr=OPTIMIZER_LR, weight_decay=OPTIMIZER_WEIGHT_DECAY, fused=OPTIMIZER_FUSED)

total_steps = len(train_dataloader) * TRAINING_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=int(total_steps * SCHEDULER_T_MAX_FACTOR))


# --- Hooks ---
hooks = []
hooks.insert(0, ProgressBarHook())
hooks.append(LRSchedulerHook(scheduler))

if TRAINING_USE_AMP:
    hooks.append(AMPHook(dtype=TRAINING_AMP_DTYPE, device_type=TRAINING_DEVICE))
if TRAINING_GRAD_CLIP:
    hooks.append(GradClipHook(max_norm=TRAINING_GRAD_CLIP))

batch_tfms = get_batch_transforms(DATASET_VARIANT, MODEL_NUM_CLASSES, DATASET_AUGMENTATION, train=True)
valid_batch_tfms = get_batch_transforms(DATASET_VARIANT, MODEL_NUM_CLASSES, train=False)
hooks.append(BatchTransformHook(x_tfm=batch_tfms, x_tfms_valid=valid_batch_tfms))

hooks.append(CheckpointHook("outputs/test/checkpoints", save_every_steps=5000))
hooks.append(
    MetricsHook(
        verbose=True,
        metrics=[
            Loss(),
            Accuracy(),
            LRStats(),
        ],
        tracker_type="trackio",
        project="resnet18-attn",
    )
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    train_dl=train_dataloader,
    valid_dl=valid_dataloader,
    optim=optimizer,
    loss_func=torch.nn.CrossEntropyLoss(label_smoothing=TRAINING_LABEL_SMOOTHING),
    epochs=TRAINING_EPOCHS,
    hooks=hooks,
)

# --- Training ---
print("Starting training...")
try:
    trainer.fit()
    print("Training complete!")
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
