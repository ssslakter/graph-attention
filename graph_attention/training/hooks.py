from typing import Callable, Union, Optional
import torch
from trainer_tools.hooks.base import BaseHook
from trainer_tools.trainer import Trainer
from trainer_tools.utils import to_device


class TrainValidBatchTransformHook(BaseHook):
    """Applies batch transforms with separate train/valid handling."""

    def __init__(
        self,
        x_tfm: Optional[Callable] = None,
        y_tfm: Optional[Callable] = None,
        batch_tfms: Union[list, Callable, None] = None,
        x_tfms_valid: Optional[Callable] = None,
        y_tfms_valid: Optional[Callable] = None,
        batch_tfms_valid: Union[list, Callable, None] = None,
    ):
        self.x_tfm, self.y_tfm = x_tfm, y_tfm
        self.x_tfms_valid, self.y_tfms_valid = x_tfms_valid, y_tfms_valid
        self.batch_tfms = batch_tfms or []
        if not isinstance(self.batch_tfms, list):
            self.batch_tfms = [self.batch_tfms]
        self.batch_tfms_valid = batch_tfms_valid or []
        if not isinstance(self.batch_tfms_valid, list):
            self.batch_tfms_valid = [self.batch_tfms_valid]

    @torch.no_grad()
    def before_step(self, trainer: Trainer):
        trainer.batch = to_device(trainer.batch, trainer.device)
        xb = trainer.get_input(trainer.batch)
        yb = trainer.get_target(trainer.batch)

        if trainer.training:
            if self.x_tfm is not None:
                xb = self.x_tfm(xb)
            if self.y_tfm is not None:
                yb = self.y_tfm(yb)
        else:
            if self.x_tfms_valid is not None:
                xb = self.x_tfms_valid(xb)
            if self.y_tfms_valid is not None:
                yb = self.y_tfms_valid(yb)

        if isinstance(trainer.batch, (list, tuple)):
            trainer.batch = (xb, yb)
        elif isinstance(trainer.batch, dict):
            trainer.batch.update(yb)
            # update xb after yb in case they return the same dict
            trainer.batch.update(xb)
        else:
            trainer.batch = xb

        batch_tfms = self.batch_tfms if trainer.training else self.batch_tfms_valid
        for tfm in batch_tfms:
            trainer.batch = tfm(trainer.batch)
