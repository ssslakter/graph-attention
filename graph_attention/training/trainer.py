from trainer_tools.all import *
from .losses import get_total_spectral_loss


class GraphAttentionTrainer(Trainer):
    """Assumes that the model has agf layers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lambda_smooth = self.config.training.get("spectral_loss_lambda", 0.01)

    def get_loss(self):
        task_loss = self.loss_func(self.preds, self.yb)
        reg_loss = get_total_spectral_loss(self.model)
        total_loss = task_loss + reg_loss * self.lambda_smooth
        self.losses = {"task_loss": task_loss, "spectral_loss": reg_loss}
        return total_loss
