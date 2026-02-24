import json
import logging
import os
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer_tools.all import (
    Trainer,
    ProgressBarHook,
    LRSchedulerHook,
    AMPHook,
    GradClipHook,
    MetricsHook,
)
from trainer_tools.hooks.metrics import Loss, Accuracy, LRStats
from trainer_tools.utils import random_seed

from graph_attention.data import build_moons_dataloaders
from graph_attention.models import MoonsAGFNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
SEED = None
DEVICE = "cuda"

# Dataset
N_SAMPLES = 4000
NOISE = 0.1
TRAIN_RATIO = 0.8

# Dataloader
BATCH_SIZE = 128
VALID_BATCH_SIZE = 256
NUM_WORKERS = 0
PIN_MEMORY = False

# Model
MODEL_DIM = 64
NUM_HEADS = 4
DIM_HEAD = 16
AGF_ORDER = 3  # monomial order
AGF_TOP_K = None  # set an int to enable top-k sparsification
ALPHAS_ACT = "sigmoid"
AGF_BASIS = "monomial"
MLP_DIM = 64
DROPOUT = 0.1
NUM_CLASSES = 2

# Optimizer
LR = 2e-3
WEIGHT_DECAY = 1e-2

# Training
EPOCHS = 50
USE_AMP = False
AMP_DTYPE = torch.float16
GRAD_CLIP = 1.0

# Trackio
TRACKIO_PROJECT = "moons-agf"
TRACKIO_NAME = (
    f"moons-agf-o{AGF_ORDER}-k{AGF_TOP_K if AGF_TOP_K is not None else 'all'}-" f"{ALPHAS_ACT}-{int(time.time())}"
)


def _maybe_trackio_login():
    token = os.environ.get("TRACKIO_API_KEY") or os.environ.get("TRACKIO_TOKEN")
    if not token:
        logger.warning("TRACKIO_API_KEY or TRACKIO_TOKEN not set; skipping trackio login.")
        return

    try:
        import trackio
    except Exception as exc:
        logger.warning("Trackio import failed; skipping login. Error: %s", exc)
        return

    for fn_name in ("login", "init", "authenticate"):
        fn = getattr(trackio, fn_name, None)
        if not callable(fn):
            continue
        try:
            fn(token)
            logger.info("Logged into trackio via trackio.%s().", fn_name)
            return
        except TypeError:
            try:
                fn(api_key=token)
                logger.info("Logged into trackio via trackio.%s(api_key=...).", fn_name)
                return
            except Exception:
                continue
        except Exception:
            continue

    logger.warning("No compatible trackio login method found; continuing without explicit login.")


def main():
    random_seed(SEED)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type != DEVICE:
        logger.info("CUDA not available. Falling back to CPU.")

    train_dl, valid_dl, _ = build_moons_dataloaders(
        n_samples=N_SAMPLES,
        noise=NOISE,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        batch_size=BATCH_SIZE,
        valid_batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    _maybe_trackio_login()

    model = MoonsAGFNet(
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        dim_head=DIM_HEAD,
        order=AGF_ORDER,
        top_k=AGF_TOP_K,
        basis=AGF_BASIS,
        alphas_act=ALPHAS_ACT,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: total=%d trainable=%d", total_params, trainable_params)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_dl) * EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    hooks = [ProgressBarHook(), LRSchedulerHook(scheduler)]

    if USE_AMP:
        hooks.append(AMPHook(dtype=AMP_DTYPE, device_type=device.type))
    if GRAD_CLIP:
        hooks.append(GradClipHook(max_norm=GRAD_CLIP))

    run_config = {
        "seed": SEED,
        "device": str(device),
        "dataset": {
            "n_samples": N_SAMPLES,
            "noise": NOISE,
            "train_ratio": TRAIN_RATIO,
            "batch_size": BATCH_SIZE,
            "valid_batch_size": VALID_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
        "model": {
            "model_dim": MODEL_DIM,
            "num_heads": NUM_HEADS,
            "dim_head": DIM_HEAD,
            "order": AGF_ORDER,
            "top_k": AGF_TOP_K,
            "basis": AGF_BASIS,
            "alphas_act": ALPHAS_ACT,
            "mlp_dim": MLP_DIM,
            "dropout": DROPOUT,
            "num_classes": NUM_CLASSES,
            "parameters_total": total_params,
            "parameters_trainable": trainable_params,
        },
        "optimizer": {"lr": LR, "weight_decay": WEIGHT_DECAY},
        "training": {"epochs": EPOCHS, "use_amp": USE_AMP, "grad_clip": GRAD_CLIP},
    }

    hooks.append(
        MetricsHook(
            verbose=True,
            metrics=[Loss(), Accuracy(), LRStats()],
            tracker_type="trackio",
            project=TRACKIO_PROJECT,
            name=TRACKIO_NAME,
            config=json.dumps(run_config),
        )
    )

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=optimizer,
        loss_func=nn.CrossEntropyLoss(),
        epochs=EPOCHS,
        hooks=hooks,
    )

    logger.info("Starting training...")
    try:
        trainer.fit()
        logger.info("Training complete.")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")


if __name__ == "__main__":
    main()
