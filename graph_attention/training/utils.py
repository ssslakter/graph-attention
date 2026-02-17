from trainer_tools.hooks import BaseHook
from trainer_tools.all import Trainer
import torch

import torch

def load_pretrained(model: torch.nn.Module, pretrained_path: str):
    state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        
        # Strip "torch.compile" prefix
        if name.startswith("_orig_mod."):
            name = name[10:]
            
        new_state_dict[name] = v

    keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded model. Missing: {keys.missing_keys}, Unexpected: {keys.unexpected_keys}")
    
    return model


class PrefetchLoader:
    """
    Prefetches data to GPU using a separate CUDA stream to overlap data transfer with computation.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        xb, yb = None, None

        for next_xb, next_yb in self.loader:
            with torch.cuda.stream(stream):
                next_xb = next_xb.to(self.device, non_blocking=True)
                next_yb = next_yb.to(self.device, non_blocking=True)

            if not first:
                yield xb, yb
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            xb, yb = next_xb, next_yb

        yield xb, yb

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying loader."""
        return getattr(self.loader, name)
