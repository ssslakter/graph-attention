import torch
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import re


def get_module_dict(model: torch.nn.Module, prefix: str = "") -> Dict[str, torch.nn.Module]:
    module_dict = {}
    for name, module in model.named_modules():
        if name:  # Skip the root module (empty name)
            full_name = f"{prefix}.{name}" if prefix else name
            module_dict[full_name] = module
    return module_dict


def filter_names(
    names: List[str], filter_pattern: Optional[Union[str, List[str], Callable[[str], bool]]] = None
) -> List[str]:
    if filter_pattern is None:
        return names

    if callable(filter_pattern):
        return [name for name in names if filter_pattern(name)]

    if isinstance(filter_pattern, str):
        pattern = re.compile(filter_pattern)
        return [name for name in names if pattern.search(name)]

    if isinstance(filter_pattern, list):
        filtered = []
        for name in names:
            for pattern in filter_pattern:
                if pattern == name or re.search(pattern, name):
                    filtered.append(name)
                    break
        return filtered

    return names


def run_with_hooks(
    model: torch.nn.Module,
    *model_args,
    fwd_hooks: Optional[List[Tuple[Union[str, List[str]], Callable]]] = None,
    module_dict: Optional[Dict[str, torch.nn.Module]] = None,
    **model_kwargs,
) -> torch.Tensor:
    """
    Run model with custom forward hooks.
    """
    if fwd_hooks is None:
        fwd_hooks = []

    if module_dict is None:
        module_dict = get_module_dict(model)

    handles = []

    try:
        # Register hooks
        for hook_spec, hook_fn in fwd_hooks:
            # Support both single name and list of names
            hook_names = [hook_spec] if isinstance(hook_spec, str) else hook_spec

            for hook_name in hook_names:
                if hook_name not in module_dict:
                    raise ValueError(f"Module '{hook_name}' not found in model. Available: {list(module_dict.keys())}")

                module = module_dict[hook_name]

                def make_hook(fn):
                    def hook(module, input, output):
                        return fn(output)

                    return hook

                handle = module.register_forward_hook(make_hook(hook_fn))
                handles.append(handle)

        # Run model
        return model(*model_args, **model_kwargs)

    finally:
        # Clean up hooks
        for handle in handles:
            handle.remove()


def run_with_cache(
    model: torch.nn.Module,
    *model_args,
    names_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
    module_dict: Optional[Dict[str, torch.nn.Module]] = None,
    device: str = "cpu",
    **model_kwargs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run model and cache activations from specified modules.
    """
    if module_dict is None:
        module_dict = get_module_dict(model)

    filtered_names = filter_names(list(module_dict.keys()), names_filter)
    cache = {}

    def make_cache_hook(name):
        def hook(activation):
            if device == "cpu":
                cache[name] = activation.detach().cpu()
            else:
                cache[name] = activation.detach()
            return activation

        return hook

    # Create hooks for filtered modules
    fwd_hooks = [(name, make_cache_hook(name)) for name in filtered_names]

    with torch.no_grad():
        output = run_with_hooks(model, *model_args, fwd_hooks=fwd_hooks, module_dict=module_dict, **model_kwargs)

    return output, cache


def get_attention_pattern_names(model: torch.nn.Module, prefix: str = "") -> List[str]:
    module_dict = get_module_dict(model, prefix=prefix)
    return [name for name in module_dict.keys() if name.endswith(".attend")]
