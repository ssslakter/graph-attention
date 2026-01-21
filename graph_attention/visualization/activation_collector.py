import torch


class ActivationCache:
    """
    A simple class to mimic TransformerLens cache behavior.
    It stores activations in a dictionary.
    """

    def __init__(self):
        self.cache = {}

    def __getitem__(self, key):
        return self.cache[key]

    def items(self):
        return self.cache.items()

    def clear(self):
        self.cache = {}


def get_activation_hook(name, cache_dict):
    """Create a hook that saves the output to the cache dict."""

    def hook(module, input, output):
        # Shape: [Batch, Heads, Seq_Len, Seq_Len]
        cache_dict[name] = output.detach().cpu()

    return hook


def run_with_cache(model, x):
    """
    Runs the model and returns (logits, cache).
    Cache keys are formatted like: 'blocks.0.attn.hook_pattern'
    """
    cache = ActivationCache()
    hooks = []

    for i, (attn_layer, _) in enumerate(model.transformer.layers):
        target_module = attn_layer.attend
        name = f"blocks.{i}.attn.hook_pattern"

        handle = target_module.register_forward_hook(get_activation_hook(name, cache.cache))
        hooks.append(handle)
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        for h in hooks:
            h.remove()

    return logits, cache
