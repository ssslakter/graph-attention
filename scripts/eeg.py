import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import scipy
import random
from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score

from graph_attention.models.eeg_encoder import EEGEncoder
from graph_attention.training.trainer import GraphAttentionTrainer
from trainer_tools.all import (
    ProgressBarHook,
    CheckpointHook,
    MetricsHook,
)
from trainer_tools.hooks.metrics import Loss, Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="EEG BCI-2a training/evaluation")
    parser.add_argument("--data-path", default="data/BCICIV_2a_mat", help="Path to BCICIV_2a_mat directory")
    parser.add_argument("--output-path", default="output", help="Directory to save checkpoints and plots")
    parser.add_argument(
        "--use-agfl",
        action="store_true",
        help="Use AGFL attention (default: True). Use --no-use-agfl to disable.",
    )
    return parser.parse_args()

TARGET_N_TIMES = 1125  # 4.5s * 250Hz
SFREQ = 250.0
SUBJECTS = list(range(1, 10))
USE_AGFL = True
# SEEDS = [0, 1, 2, 3, 4]
SEEDS = [0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

def bandpass_filter(data, lowcut=0.5, highcut=100.0, fs=SFREQ, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)


def notch_filter(data, freq=50.0, fs=SFREQ, q=30.0):
    b, a = iirnotch(freq / (0.5 * fs), q)
    return lfilter(b, a, data, axis=0)


def load_data_from_mat(file_path):
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    all_x = []
    all_y = []

    mi_runs = mat['data'][-6:]

    for run in mi_runs:
        fs = float(run.fs)
        # 1) Фильтрация (0.5-100 Гц) + notch 50 Гц
        x_full = bandpass_filter(run.X[:, :22], fs=fs)
        x_full = notch_filter(x_full, freq=50.0, fs=fs)

        # 2) Нарезка на триалы
        for j, start_sample in enumerate(run.trial):
            end_sample = start_sample + TARGET_N_TIMES
            trial_data = x_full[start_sample:end_sample, :]  # (1125, 22)

            # Проверка на артефакты
            if hasattr(run, 'artifacts') and run.artifacts[j] != 0:
                continue

            all_x.append(trial_data)
            all_y.append(run.y[j] - 1)  # 1-4 -> 0-3

    return np.array(all_x), np.array(all_y)


def standardize_train_test(x_train, x_test):
    # x: (n, t, ch) -> нормируем по тренировке для каждого канала
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std

class EEGDataset(Dataset):
    def __init__(self, x, y):
        # Модель EEGEncoder ожидает (B, 1, T, C) -> (B, 1, 1125, 22)
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 64
    lr: float = 1e-3
    dropout: float = 0.3
    weight_decay: float = 0.0005  # только для Linear
    spectral_loss_lambda: float = 0.01
    checkpoint_every_epochs: int = 10


def _param_groups_linear_decay(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, nn.Linear):
                decay.append(param)
            else:
                no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = (y_true == y_pred).mean() * 100.0
    kappa = cohen_kappa_score(y_true, y_pred) * 100.0
    return acc, kappa


class HistoryHook:
    def before_fit(self, trainer):
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.valid_kappa = []
        self.epoch = []
        self._reset_train()
        self._reset_valid()

    def _reset_train(self):
        self._train_loss_sum = 0.0
        self._train_correct = 0
        self._train_total = 0

    def _reset_valid(self):
        self._valid_loss_sum = 0.0
        self._valid_correct = 0
        self._valid_total = 0
        self._valid_preds = []
        self._valid_targets = []

    def before_epoch(self, trainer):
        self._reset_train()

    def before_valid(self, trainer):
        self._reset_valid()

    def after_loss(self, trainer):
        y = trainer.get_target(trainer.batch)
        if not torch.is_tensor(y):
            return
        batch_size = y.size(0)

        preds = trainer.preds
        if isinstance(preds, dict):
            if "logits" in preds:
                preds = preds["logits"]
            elif "preds" in preds:
                preds = preds["preds"]
        if torch.is_tensor(preds) and preds.ndim >= 2:
            pred_labels = preds.argmax(dim=1)
            correct = (pred_labels == y).sum().item()
        else:
            pred_labels = None
            correct = 0

        loss_val = float(getattr(trainer, "loss", 0.0))

        if trainer.training:
            self._train_loss_sum += loss_val * batch_size
            self._train_correct += correct
            self._train_total += batch_size
        else:
            self._valid_loss_sum += loss_val * batch_size
            self._valid_correct += correct
            self._valid_total += batch_size
            if pred_labels is not None:
                self._valid_preds.append(pred_labels.detach().cpu().numpy())
                self._valid_targets.append(y.detach().cpu().numpy())

    def after_epoch(self, trainer):
        train_loss = self._train_loss_sum / max(self._train_total, 1)
        train_acc = 100.0 * self._train_correct / max(self._train_total, 1)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)

        if self._valid_total > 0:
            valid_loss = self._valid_loss_sum / self._valid_total
            valid_acc = 100.0 * self._valid_correct / self._valid_total
            if self._valid_targets:
                y_true = np.concatenate(self._valid_targets)
                y_pred = np.concatenate(self._valid_preds)
                valid_kappa = cohen_kappa_score(y_true, y_pred) * 100.0
            else:
                valid_kappa = float("nan")
        else:
            valid_loss = float("nan")
            valid_acc = float("nan")
            valid_kappa = float("nan")
        if trainer.epoch % 20 == 0:
            print(valid_acc, valid_loss, valid_kappa)
        self.valid_loss.append(valid_loss)
        self.valid_acc.append(valid_acc)
        self.valid_kappa.append(valid_kappa)
        self.epoch.append(trainer.epoch + 1)


def save_training_plots(history: HistoryHook, save_dir: Path, mode_label: str, subject_id: int, seed: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.epoch

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.train_loss, label="train_loss")
    plt.plot(epochs, history.valid_loss, label="valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({mode_label}) S{subject_id:02d} seed {seed}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.train_acc, label="train_acc")
    plt.plot(epochs, history.valid_acc, label="valid_acc")
    plt.plot(epochs, history.valid_kappa, label="valid_kappa")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title(f"Accuracy/Kappa ({mode_label}) S{subject_id:02d} seed {seed}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_curve.png")
    plt.close()


def run_subject(
    data_path: str,
    output_path: str,
    subject_id: int,
    cfg: TrainConfig,
    seeds: list[int],
    use_agfl: bool,
):
    x_train, y_train = load_data_from_mat(f"{data_path}/A{subject_id:02d}T.mat")
    x_test, y_test = load_data_from_mat(f"{data_path}/A{subject_id:02d}E.mat")

    x_train, x_test = standardize_train_test(x_train, x_test)

    train_loader = DataLoader(EEGDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(EEGDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    steps_per_epoch = len(train_loader)
    save_every_steps = max(steps_per_epoch * cfg.checkpoint_every_epochs, steps_per_epoch)

    seed_metrics = []

    for seed in seeds:
        seed_all(seed)

        # Инициализация модели: AGFL по умолчанию, либо базовый режим через attn_kwargs
        if use_agfl:
            model = EEGEncoder(eeg_channels=22, num_classes=4, dropout=cfg.dropout).to(DEVICE)
        else:
            model = EEGEncoder(
                eeg_channels=22,
                num_classes=4,
                dropout=cfg.dropout,
                order=1,
                alphas_act="softmax",
            ).to(DEVICE)

        optimizer = torch.optim.Adam(
            _param_groups_linear_decay(model, cfg.weight_decay),
            lr=cfg.lr,
        )

        history_hook = HistoryHook()
        trainer = GraphAttentionTrainer(
            model=model,
            train_dl=train_loader,
            valid_dl=test_loader,
            optim=optimizer,
            loss_func=nn.CrossEntropyLoss(),
            epochs=cfg.epochs,
            hooks=[
                history_hook,
                ProgressBarHook(),
                CheckpointHook(
                    f"{output_path}/eeg_bci2a/{'agfl' if use_agfl else 'base'}_subject_{subject_id}/seed_{seed}/checkpoints",
                    save_every_steps=save_every_steps,
                ),
                # verbose=False чтобы не спамить в консоль на 500 эпохах
                MetricsHook(verbose=False, metrics=[Loss(), Accuracy()]),
            ],
            config=OmegaConf.create({"training": {"spectral_loss_lambda": cfg.spectral_loss_lambda}}),
        )

        trainer.fit()
        plot_dir = Path(
            f"{output_path}/eeg_bci2a/{'agfl' if use_agfl else 'base'}_subject_{subject_id}/seed_{seed}/plots"
        )
        mode_label = "AGFL" if use_agfl else "BASE (no AGFL)"
        save_training_plots(history_hook, plot_dir, mode_label, subject_id, seed)
        acc, kappa = evaluate(model, test_loader)
        seed_metrics.append({"seed": seed, "acc": acc, "kappa": kappa})

    return seed_metrics


def aggregate_metrics(seed_metrics):
    accs = np.array([m["acc"] for m in seed_metrics], dtype=np.float32)
    kappas = np.array([m["kappa"] for m in seed_metrics], dtype=np.float32)
    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=0)),
        "kappa_mean": float(kappas.mean()),
        "kappa_std": float(kappas.std(ddof=0)),
    }


def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path
    use_agfl = args.use_agfl

    cfg = TrainConfig()
    results = {}

    mode_label = "AGFL" if use_agfl else "BASE (no AGFL)"
    print(f"\n=== Starting {mode_label} Evaluation ===")
    for subject_id in SUBJECTS:
        print(f"  Running Subject {subject_id:02d} [{mode_label}]...")
        seed_metrics = run_subject(data_path, output_path, subject_id, cfg, SEEDS, use_agfl=use_agfl)
        results[subject_id] = aggregate_metrics(seed_metrics)
        print(f"  Subject {subject_id:02d} Results: {results[subject_id]}")

    # Усреднение по всем субъектам
    our_agfl_mean_acc = np.mean([results[s]["acc_mean"] for s in SUBJECTS])
    our_agfl_mean_kappa = np.mean([results[s]["kappa_mean"] for s in SUBJECTS])

    print(f"\n=== FINAL {mode_label} RESULTS ===")
    print(f"Mean Accuracy across 9 subjects: {our_agfl_mean_acc:.2f}%")
    print(f"Mean Kappa across 9 subjects: {our_agfl_mean_kappa:.2f}%")

    # Данные из статьи для сравнения
    paper_baselines = {
        "ATCNet": 84.1,        # [cite: 344]
        "TCNetFusion": 74.6,   # [cite: 344]
        "EEGTCNet": 67.4,      # [cite: 344]
        "EEGEncoder (Base)": 74.48 # Данные из Table 3 [cite: 363]
    }

    labels = list(paper_baselines.keys()) + [f"Our {mode_label}"]
    values = list(paper_baselines.values()) + [our_agfl_mean_acc]
    colors = ['gray', 'gray', 'gray', 'blue', 'orange']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Mean Accuracy (%)')
    plt.title('Performance Comparison on BCI-2a Dataset (All Subjects)')
    plt.ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/final_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
