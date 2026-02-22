from trainer_tools.all import *
from .losses import get_total_spectral_loss


class GraphAttentionTrainer(Trainer):
    """Assumes that the model has agf layers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lambda_smooth = self.config.training.get("spectral_loss_lambda", 0.01)

    def get_loss(self, preds, target):
        if isinstance(preds, dict) and "loss" in preds:
            task_loss = preds["loss"]
        else:
            if self.loss_func is None:
                raise ValueError(
                    "No loss function provided. Please implement get_loss, provide loss_func or return loss from the model"
                )
            task_loss = self.loss_func(preds, target)

        reg_loss = get_total_spectral_loss(self.model)
        total_loss = task_loss + reg_loss * self.lambda_smooth
        self.losses = {"task_loss": task_loss, "spectral_loss": reg_loss}
        return total_loss
