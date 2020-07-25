import numpy as np
import torch
from enum import Enum

def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

# written for pytorch ignite
# fire this on backwards pass
class BackwardsEvents(Enum):
    BACKWARDS_COMPLETED = 'backwards_completed'

def add_autoclip_gradient_handler(engine, model, clip_percentile):
    # Keep track of the history of gradients and select a cutoff
    # to clip values to based on percentile.
    grad_history = []

    @engine.on(BackwardsEvents.BACKWARDS_COMPLETED)
    def autoclip_gradient(engine):
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        clip_value = np.percentile(grad_history, clip_percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
