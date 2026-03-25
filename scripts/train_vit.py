from trainer_tools.all import *
from trainer_tools.hooks import BatchTransformHook, ProgressBarHook, MetricsHook, LRSchedulerHook, AMPHook, GradClipHook, CheckpointHook
from trainer_tools.hooks.accelerate import AccelerateHook
from trainer_tools.hooks.metrics import Loss, Accuracy, SamplesPerSecond
from trainer_tools.imports import *
from graph_attention.all import *

configure_logging()

BASE_NUM_HEADS = 8
BASE_TOP_K = 16
BASE_ORDER = 4
BASE_ALPHAS_ACT = "relu"

SEED = 42
EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 0.005
WARMUP_STEPS = 500
GRAD_CLIP_MAX_NORM = 2.0
CHECKPOINT_EVERY = 1000
PROJECT = "vit-attn-norm-abl"
MULTI_GPU = False

random_seed(SEED)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATASET_VARIANT = "imagenette"
IMAGE_SIZE = 224
PATCH_SIZE = 16
DIM = 192
DEPTH = 12
MLP_RATIO = 4


def get_dataset_and_transforms(variant, train):
    ds = get_dataset(variant, root="./data", train=train)
    batch_x_transforms = get_batch_transforms(variant, train=train)
    return ds, batch_x_transforms


def train_model(num_heads: int, top_k: int, order: int, alphas_act: str, run_name: str, config_dict: dict):
    BS = 32

    ds, batch_x_transforms = get_dataset_and_transforms(DATASET_VARIANT, train=True)
    valid_ds, batch_x_transforms_valid = get_dataset_and_transforms(DATASET_VARIANT, train=False)

    train_dl = DataLoader(
        ds,
        batch_size=BS,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )
    valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=False, num_workers=2, persistent_workers=True)

    print(f"\n{'=' * 50}\nStarting Run: {run_name} (Config: {config_dict})\n{'=' * 50}")

    model = SimpleViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=len(ds.classes),
        dim=DIM,
        depth=DEPTH,
        num_heads=num_heads,
        mlp_ratio=MLP_RATIO,
        attention_layer=partial(
            AGFAttention,
            top_k=top_k,
            order=order,
            alphas_act=alphas_act,
            normalization="softmax",
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_func = torch.nn.CrossEntropyLoss()

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * EPOCHS

    if MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using multiple GPUs ({torch.cuda.device_count()}). AccelerateHook will be used.")
        effective_total_steps = total_steps
    else:
        effective_total_steps = total_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=effective_total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    hooks = [
        LRSchedulerHook(scheduler),
        ProgressBarHook(),
        BatchTransformHook(x_tfm=batch_x_transforms, x_tfms_valid=batch_x_transforms_valid),
        MetricsHook(
            metrics=[Loss(), Accuracy(), SamplesPerSecond(), LayerActivationStats((AGFAttention,), freq=100)],
            name=run_name,
            verbose=False,
            tracker_type="trackio",
            config=config_dict,
            project=PROJECT,
        ),
    ]

    if MULTI_GPU and torch.cuda.device_count() > 1:
        hooks.append(AccelerateHook(max_grad_norm=GRAD_CLIP_MAX_NORM, gradient_accumulation_steps=1, mixed_precision="bf16"))
    else:
        hooks.append(AMPHook(dtype=torch.float16, device_type="cuda"))
        hooks.append(GradClipHook(max_norm=GRAD_CLIP_MAX_NORM))

    hooks.append(CheckpointHook(f"outputs/{run_name}/checkpoints", save_every_steps=10000, keep_last=3))

    trainer = Trainer(model=model, train_dl=train_dl, valid_dl=valid_dl, optim=optimizer, loss_func=loss_func, epochs=EPOCHS, hooks=hooks)
    trainer.fit()

    final_accuracy = trainer.get_hook(MetricsHook).epoch_data.get("valid_accuracy", 0.0)

    del model, optimizer, scheduler, trainer
    torch.cuda.empty_cache()

    return final_accuracy


def run_ablation_study():
    base_config = {
        "num_heads": BASE_NUM_HEADS,
        "top_k": BASE_TOP_K,
        "order": BASE_ORDER,
        "alphas_act": BASE_ALPHAS_ACT,
    }

    ablation_space = {
        "top_k": [8, 32, 64, None],
        "order": [1, 2, 3, 5, 6],
        "alphas_act": ["softmax", "none", "sigmoid"],
    }

    configs_to_run = []
    results = []

    baseline_config_dict = base_config.copy()
    baseline_run_name = f"baseline_heads{BASE_NUM_HEADS}_topk{BASE_TOP_K}_order{BASE_ORDER}_act{BASE_ALPHAS_ACT}"
    print(f"\n--- Running BASELINE: {baseline_run_name} ---")
    acc = train_model(
        num_heads=baseline_config_dict["num_heads"],
        top_k=baseline_config_dict["top_k"],
        order=baseline_config_dict["order"],
        alphas_act=baseline_config_dict["alphas_act"],
        run_name=baseline_run_name,
        config_dict=baseline_config_dict,
    )
    results.append(
        {
            "Run Type": "Baseline",
            "Top-K": baseline_config_dict["top_k"],
            "Order": baseline_config_dict["order"],
            "Alphas Act": baseline_config_dict["alphas_act"],
            "Accuracy": acc,
            "Num Heads": baseline_config_dict["num_heads"],
        }
    )

    print(f"\n--- Generating Ablation Configurations ---")
    for param_name, variations in ablation_space.items():
        for val in variations:
            if val == base_config.get(param_name):
                continue

            new_config_dict = base_config.copy()
            new_config_dict[param_name] = val

            run_name_parts = []
            run_name_parts.append(f"ablate_{param_name}_{val}")
            run_name_parts.append(f"h{new_config_dict['num_heads']}")
            run_name_parts.append(f"tk{new_config_dict['top_k']}")
            run_name_parts.append(f"o{new_config_dict['order']}")
            run_name_parts.append(f"a{new_config_dict['alphas_act']}")

            current_run_name = "_".join(run_name_parts)

            configs_to_run.append({"run_name": current_run_name, "config_dict": new_config_dict})

    print(f"Total configurations to evaluate (including baseline): {len(results) + len(configs_to_run)}\n")

    for i, item in enumerate(configs_to_run):
        run_name = item["run_name"]
        cfg_dict = item["config_dict"]

        acc = train_model(
            num_heads=cfg_dict["num_heads"],
            top_k=cfg_dict["top_k"],
            order=cfg_dict["order"],
            alphas_act=cfg_dict["alphas_act"],
            run_name=run_name,
            config_dict=cfg_dict,
        )

        results.append(
            {
                "Run Type": run_name.split("_")[0] + "_" + run_name.split("_")[1],
                "Top-K": cfg_dict.get("top_k"),
                "Order": cfg_dict.get("order"),
                "Alphas Act": cfg_dict.get("alphas_act"),
                "Accuracy": acc,
                "Num Heads": cfg_dict.get("num_heads"),
            }
        )

    print("\n\n" + "=" * 100)
    print(" " * 40 + "ABLATION SUMMARY")
    print("=" * 100)
    all_keys = set()
    for res in results:
        all_keys.update(res.keys())

    column_order = ["Run Type", "Top-K", "Order", "Alphas Act", "Num Heads", "Accuracy"]
    sorted_keys = column_order + sorted([k for k in all_keys if k not in column_order])

    header = " | ".join(f"{col:<15}" for col in sorted_keys)
    print(header)
    print("-" * len(header))

    for res in results:
        row_values = []
        for key in sorted_keys:
            value = res.get(key, "N/A")
            if key == "Accuracy":
                row_values.append(f"{value:<15.4f}")
            else:
                row_values.append(f"{str(value):<15}")
        print(" | ".join(row_values))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    if MULTI_GPU and torch.cuda.device_count() > 1:
        print("Multiple GPUs detected, running ablation with AccelerateHook.")
    else:
        print("Single GPU or no GPU detected, running ablation with AMPHook.")

    run_ablation_study()
