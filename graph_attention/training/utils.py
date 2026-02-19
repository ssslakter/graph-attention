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
