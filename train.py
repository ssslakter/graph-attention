import hydra, logging, json, torch
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from trainer_tools.all import *
from trainer_tools.hooks.utils import remove_disabled_hooks
from graph_attention.data import get_dataset, get_transforms, get_batch_transforms
from graph_attention.training.utils import StepInitHook, load_pretrained, PrefetchLoader
from graph_attention.training.trainer import GraphAttentionTrainer

log = logging.getLogger(__name__)


def _build_datasets_and_loaders(cfg: DictConfig):
    """Build training and validation dataloaders."""
    dl_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
    bs, valid_bs = dl_cfg.pop("batch_size"), dl_cfg.pop("valid_batch_size")

    train_ds = get_dataset(
        cfg.dataset.variant,
        cfg.dataset.root,
        train=True,
        transforms=get_transforms(cfg.dataset.variant, True, cfg.dataset.augmentation),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        **dl_cfg,
    )
    valid_dl = DataLoader(
        get_dataset(cfg.dataset.variant, cfg.dataset.root, train=False),
        batch_size=valid_bs,
        shuffle=False,
        **dl_cfg,
    )

    return train_dl, valid_dl


def _build_optimizer_and_scheduler(cfg: DictConfig, model: torch.nn.Module, train_dl: DataLoader):
    """Build optimizer and learning rate scheduler."""
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    total_steps = len(train_dl) * cfg.training.epochs
    scheduler_cfg = cfg.get("scheduler")

    if not scheduler_cfg:
        return optimizer, None

    scheduler_conf = OmegaConf.to_container(scheduler_cfg, resolve=True)

    if "warmup_ratio" in scheduler_conf:
        scheduler_conf["num_warmup_steps"] = int(total_steps * scheduler_conf.pop("warmup_ratio"))
        scheduler_conf["num_training_steps"] = total_steps

    if "OneCycleLR" in scheduler_conf.get("_target_", ""):
        scheduler = instantiate(scheduler_conf, optimizer=optimizer, total_steps=total_steps)
    elif "CosineAnnealingLR" in scheduler_conf.get("_target_", ""):
        scheduler = instantiate(scheduler_conf, optimizer=optimizer, T_max=total_steps)
    else:
        scheduler = instantiate(scheduler_conf, optimizer=optimizer)

    return optimizer, scheduler


def build_hooks(hook_cfg: DictConfig, config: DictConfig = None):
    """Build list of hooks from configuration."""
    hooks = []
    if not hook_cfg:
        return hooks

    for name, h_cfg in hook_cfg.items():
        if not h_cfg.get("enabled", False):
            continue
        h_cfg = {k: v for k, v in OmegaConf.to_container(h_cfg, resolve=True).items() if k != "enabled"}
        if name == "metrics" and config is not None:
            h_cfg["config"] = json.dumps(OmegaConf.to_container(remove_disabled_hooks(config), resolve=True))
        hooks.append(instantiate(h_cfg, _recursive_=True))
    return hooks


def add_training_hooks(hooks: list, scheduler, cfg: DictConfig):
    """Add standard training hooks to the list."""
    hooks.insert(0, ProgressBarHook())

    if scheduler:
        hooks.append(LRSchedulerHook(scheduler))

    if cfg.training.get("use_amp"):
        hooks.append(
            AMPHook(
                enabled=cfg.training.use_amp,
                dtype=getattr(torch, cfg.training.get("amp_dtype", "float16")),
                device_type=cfg.training.get("device", "cuda"),
            )
        )

    if cfg.training.get("grad_clip"):
        hooks.append(GradClipHook(max_norm=cfg.training.grad_clip))

    if cfg.training.get("init_step"):
        hooks.append(StepInitHook(cfg.training.init_step))

    batch_tfms = get_batch_transforms(
        cfg.dataset.variant, cfg.model.num_classes, cfg.dataset.augmentation, train=True
    )
    valid_batch_tfms = get_batch_transforms(cfg.dataset.variant, cfg.model.num_classes, train=False)
    hooks.append(BatchTransformHook(x_tfm=batch_tfms, x_tfms_valid=valid_batch_tfms))


def build_trainer(model, train_dl, valid_dl, optimizer, hooks, cfg):
    """Build the trainer instance."""
    trainer_cls = Trainer
    if cfg.model.get("attention_layer") and "agf" in cfg.model.attention_layer.get("_target_", "").lower():
        trainer_cls = GraphAttentionTrainer

    return trainer_cls(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=optimizer,
        loss_func=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        epochs=cfg.training.epochs,
        hooks=[h for h in hooks if h],
        config=cfg,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    random_seed(cfg.training.seed)
    log.info(f"Random seed set to {cfg.training.seed}")
    train_dataloader, test_dataloader = _build_datasets_and_loaders(cfg)
    with open_dict(cfg):
        cfg.model.num_classes = len(train_dataloader.dataset.classes)
        cfg.model.channels = cfg.dataset.channels

    # Handle different model loading strategies
    pretrained = cfg.model.get("pretrained", None)
    model_cls = get_class(cfg.model._target_)

    if hasattr(model_cls, "load_from_timm"):
        
        timm_kwargs = {
            "model_name": pretrained,
            "num_classes": cfg.model.num_classes,
            "pretrained": True,
        } if pretrained and isinstance(pretrained, str) else {
            "model_name": cfg.model.get("model_name", "resnet18"),
            "num_classes": cfg.model.num_classes,
            "pretrained": False,
        }
        # Add any extra kwargs from config except reserved keys
        extra_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
            if k not in ["_target_", "pretrained", "num_classes", "channels"]
        }
        timm_kwargs.update(extra_kwargs)
        log.info(f"Loading model from timm: {pretrained}")
        model = model_cls.load_from_timm(**timm_kwargs)
    elif pretrained and hasattr(model_cls, "from_pretrained"):
        # Load using from_pretrained method (e.g., ViT)
        attn = instantiate(cfg.model.attention_layer)
        model = model_cls.from_pretrained(pretrained, attention_layer=attn)
    else:
        # Standard instantiation
        model = instantiate(cfg.model)

    if p := cfg.training.get("pretrained_path"):
        log.info(f"Loading pretrained weights from {p}")
        model = load_pretrained(model, p)

    torch.set_float32_matmul_precision(cfg.torch.get("matmul_precision", "high"))
    model = torch.compile(model) if cfg.torch.get("compile", False) else model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, model, train_dataloader)

    device = torch.device("cuda")
    train_dataloader = PrefetchLoader(train_dataloader, device)
    test_dataloader = PrefetchLoader(test_dataloader, device)

    hooks = build_hooks(cfg.hooks, config=cfg)
    add_training_hooks(hooks, scheduler, cfg)

    log.info(f"Enabled hooks: {[type(h).__name__ for h in hooks]}")

    trainer = build_trainer(
        model=model,
        train_dl=train_dataloader,
        valid_dl=test_dataloader,
        optimizer=optimizer,
        hooks=hooks,
        cfg=cfg,
    )
    log.info(f"Using trainer: {type(trainer).__name__}")

    log.info("Starting training...")
    try:
        trainer.fit()
        log.info("Training complete!")
    except KeyboardInterrupt:
        log.info("\nTraining interrupted by user.")


if __name__ == "__main__":
    main()
