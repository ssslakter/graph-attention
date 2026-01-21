import torch
import torch.nn as nn
from ..models.layers.agf import AGFLayer


def get_total_spectral_loss(model: nn.Module, lambda_smooth: float = 0.01) -> dict:
    """
    Iterates over the model, calls get_regularization_loss on every AGFLayer,
    and returns the total.
    """
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for module in model.modules():
        if isinstance(module, AGFLayer):
            total_loss = total_loss + module.get_regularization_loss(lambda_smooth)

    return total_loss
