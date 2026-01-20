import torch
from trainer_tools.hooks import *


class AccuracyMetricsHook(MetricsHook):
    """Calculats both loss and accuracy metrics"""

    def _get_batch_metrics(self, trainer, prefix: str):
        metrics = {f"{prefix}_loss": trainer.loss}

        logits = trainer.preds

        pred_classes = torch.argmax(logits, dim=1)
        target_classes = trainer.yb

        correct = (pred_classes == target_classes).sum().item()
        total = target_classes.size(0)
        accuracy = correct / total

        metrics[f"{prefix}_accuracy"] = accuracy

        return metrics
