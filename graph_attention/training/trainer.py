from trainer_tools.all import *
from .losses import get_total_spectral_loss


class GraphAttentionTrainer(Trainer):
    """Assumes that the model has agf layers"""

    def __init__(self, lambda_smooth=0.01, **kwargs):
        super().__init__(**kwargs)
        self.lambda_smooth = lambda_smooth

    def get_loss(self):
        task_loss = self.loss_func(self.preds, self.yb)
        reg_loss = get_total_spectral_loss(self.model, lambda_smooth=self.lambda_smooth)
        total_loss = task_loss + reg_loss
        self.losses = {"task_loss": task_loss, "spectral_loss": reg_loss}
        return total_loss
