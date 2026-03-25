import torch
from torch._dynamo import disable
from trainer_tools.hooks.metrics import Metric, Accuracy
from ..models.layers import AGFAttention, Attention

__all__ = ["LayerActivationStats", "AccuracyMixup"]

class LayerActivationStats(Metric):
    """
    Monitors activation norms and graph statistics for specific layers.

    Args:
        layer_cls: The class type of the layers to monitor (e.g., AGFLayer).
        freq: How often to log the stats (default: 1).
    """

    def __init__(self, layer_classes: tuple = (AGFAttention, Attention), freq: int = 1):
        super().__init__("layer_stats", freq, phase="after_loss", use_prefix=False)
        self.layer_classes = tuple(layer_classes)
        self._stats = {}
        self._handles = []

    def _make_hook(self, name):
        @disable
        def hook(module, inp, out):
            if not module.training:
                return

            x_in = inp[0] if isinstance(inp, tuple) else inp

            with torch.no_grad():
                # in_n = x_in.norm(p=2, dim=-1).mean().item()
                # out_n = out.norm(p=2, dim=-1).mean().item()

                # self._stats.update(
                #     {
                #         f"debug/norms/{name}/in_norm": in_n,
                #         f"debug/norms/{name}/out_norm": out_n,
                #     }
                # )

                # Expected shape: (Order + 1, Heads)
                if (alphas := getattr(module, "alphas", None)) is not None:
                    a_vals = alphas.detach().float()

                    for k, k_vals in enumerate(a_vals):  # Iterate over orders
                        prefix = f"debug/alphas/{name}"

                        # Log individual heads
                        for h, val in enumerate(k_vals):
                            self._stats[f"{prefix}/h{h:02d}/alpha_{k}"] = val.item()

        return hook

    def _register(self, model):
        """Lazily register hooks on the first pass."""
        targets = [m for m in model.modules() if isinstance(m, self.layer_classes)]
        for i, layer in enumerate(targets):
            self._handles.append(layer.register_forward_hook(self._make_hook(f"L{i:02d}")))

    def __call__(self, trainer) -> dict:
        if not self._handles:
            self._register(trainer.model)
            return {}

        if not trainer.training:
            self._stats.clear()
            return {}

        data = self._stats.copy()
        self._stats.clear()
        return data


class AccuracyMixup(Accuracy):
    """Like accuracy, but supports mixed targets."""

    def __call__(self, trainer):
        target = trainer.get_target(trainer.batch)
        logits = trainer.preds["logits"] if isinstance(trainer.preds, dict) else trainer.preds
        preds = logits.argmax(dim=1) if logits.ndim > 1 else (logits > 0)

        if target.ndim > 1: # handle mixed targets
            acc = target.gather(1, preds.unsqueeze(1)).squeeze(1)
        else:
            acc = (preds == target).float()
        return {self.name: acc.mean().item()}
