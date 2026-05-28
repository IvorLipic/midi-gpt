import torch
import torch.nn.functional as F

def compute_remi_loss(logits, targets):
    if isinstance(logits, torch.Tensor) and logits.is_nested:
        return F.cross_entropy(logits.values(), targets, ignore_index=0)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0
    )