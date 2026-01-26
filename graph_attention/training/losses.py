import torch
import torch.nn as nn
from ..models.layers.agf import AGFAttention


def get_total_spectral_loss(model: nn.Module) -> dict:
    """
    Iterates over the model, calls get_regularization_loss on every AGFLayer,
    and returns the total.
    """
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for module in model.modules():
        if isinstance(module, AGFAttention):
            total_loss += module.get_regularization_loss()

    return total_loss
