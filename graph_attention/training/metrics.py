import torch
from trainer_tools.metrics import Metric
from ..models.layers.agf import AGFLayer


class LayerActivationStats(Metric):
    """
    Monitors activation norms and graph statistics for specific layers.
    
    Args:
        layer_cls: The class type of the layers to monitor (e.g., AGFLayer).
        freq: How often to log the stats (default: 1).
    """
    def __init__(self, layer_cls = AGFLayer, freq: int = 1):
        # We collect this 'after_loss' to ensure the forward pass is complete
        super().__init__("layer_stats", freq, phase="after_loss", use_prefix=False)
        self.layer_cls = layer_cls
        self._stats = {}
        self._handles = []

    def _make_hook(self, name):
        def hook(module, inp, out):
            if not module.training: return

            # 2. Unpack input (handles (x, mask) tuples common in GNNs/Transformers)
            x_in = inp[0] if isinstance(inp, tuple) else inp
            x_out = out

            with torch.no_grad():
                # Signal Norms & Gain
                in_norm = x_in.norm(p=2, dim=-1).mean().item()
                out_norm = x_out.norm(p=2, dim=-1).mean().item()
                
                self._stats[f"debug/norms/{name}/in_norm"] = in_norm
                self._stats[f"debug/norms/{name}/out_norm"] = out_norm
                self._stats[f"debug/norms/{name}/gain"] = out_norm / (in_norm + 1e-6)

                if (adj := getattr(module, "last_adj", None)) is not None:
                    self._stats[f"debug/details/{name}/adj_max"] = adj.max().item()

                # Alpha coefficients from Graph Filter (e.g., PolynomialFilter)
                graph_filter = getattr(module, "graph_filter", None)
                alphas = getattr(graph_filter, "alphas", None)
                if alphas is not None:
                    # Expected shape: (K+1, 1, H, 1, 1) -> squeeze to (K+1, H)
                    a_vals = alphas.detach().float().squeeze(-1).squeeze(-1).squeeze(1)
                    # Per-degree summaries and per-head values
                    for k in range(a_vals.shape[0]):
                        k_vals = a_vals[k]  # (H,)
                        self._stats[f"debug/alphas/{name}/alpha_{k}/mean"] = k_vals.mean().item()
                        self._stats[f"debug/alphas/{name}/alpha_{k}/max"] = k_vals.max().item()
                        for h in range(k_vals.shape[0]):
                           self._stats[f"debug/alphas/{name}/alpha_{k}/h{h:02d}"] = k_vals[h].item()

                    # If gradients are available, log their summaries too
                    if getattr(alphas, "grad", None) is not None:
                        g_vals = alphas.grad.detach().float().squeeze(-1).squeeze(-1).squeeze(1).abs()
                        for k in range(g_vals.shape[0]):
                            k_grads = g_vals[k]  # (H,)
                            self._stats[f"debug/alphas/{name}/alpha_grad_{k}/mean"] = k_grads.mean().item()
                            self._stats[f"debug/alphas/{name}/alpha_grad_{k}/max"] = k_grads.max().item()
        return hook

    def _register(self, model):
        """Lazily register hooks on the first pass."""
        targets = [m for m in model.modules() if isinstance(m, self.layer_cls)]
        for i, layer in enumerate(targets):
            self._handles.append(
                layer.register_forward_hook(self._make_hook(f"L{i:02d}"))
            )

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