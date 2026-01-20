import hydra, logging, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from trainer_tools.all import *
from graph_attention.data.cifar import get_cifar, get_transforms

logger = logging.getLogger(__name__)


def build_scheduler(optimizer, cfg, total_steps):
    if not cfg:
        return None
    scheduler_conf = OmegaConf.to_container(cfg, resolve=True)

    # Handle warmup convenience parameters
    if "warmup_ratio" in scheduler_conf:
        scheduler_conf["num_warmup_steps"] = int(total_steps * scheduler_conf.pop("warmup_ratio"))
        scheduler_conf["num_training_steps"] = total_steps

    # Inject dynamic parameters based on scheduler type
    if "OneCycleLR" in scheduler_conf.get("_target_", ""):
        return instantiate(scheduler_conf, optimizer=optimizer, total_steps=total_steps)

    return instantiate(scheduler_conf, optimizer=optimizer)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Data
    dl_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
    bs, valid_bs = dl_cfg.pop("batch_size"), dl_cfg.pop("valid_batch_size")

    train_dl = DataLoader(
        get_cifar(
            cfg.dataset.root,
            True,
            variant=cfg.dataset.variant,
            transforms=get_transforms(cfg.dataset.variant, True, cfg.dataset.augmentation),
        ),
        batch_size=bs,
        shuffle=True,
        **dl_cfg,
    )
    valid_dl = DataLoader(
        get_cifar(cfg.dataset.root, False, variant=cfg.dataset.variant), batch_size=valid_bs, shuffle=False, **dl_cfg
    )

    # Model & Optimizer
    model = instantiate(cfg.model, num_classes=len(train_dl.dataset.classes), channels=cfg.dataset.channels)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Scheduler & Hooks
    total_steps = len(train_dl) * cfg.training.epochs
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps)

    hooks = [
        ProgressBarHook(),
        LRSchedulerHook(scheduler, step_on_batch=True) if scheduler else None,
        (
            AMPHook(
                enabled=cfg.training.use_amp,
                dtype=getattr(torch, cfg.training.get("amp_dtype", "float16")),
                device_type=cfg.training.get("device", "cuda"),
            )
            if cfg.training.get("use_amp")
            else None
        ),
        GradClipHook(max_norm=cfg.training.grad_clip) if cfg.training.get("grad_clip") else None,
    ]

    # Add optional hooks from config
    for name in ["checkpoint", "ema", "metrics"]:
        if (h_cfg := cfg.hooks.get(name)) and h_cfg.get("enabled"):
            hooks.append(
                instantiate({k: v for k, v in OmegaConf.to_container(h_cfg, resolve=True).items() if k != "enabled"})
            )

    # Train
    BaseTrainer(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=optimizer,
        loss_func=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        epochs=cfg.training.epochs,
        hooks=[h for h in hooks if h],
    ).fit()


if __name__ == "__main__":
    main()
