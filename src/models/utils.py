import torch

def generate_causal_mask(L, device):
    # (L, L) with -inf above diagonal for TransformerEncoder
    mask = torch.triu(
        torch.full((L, L), float("-inf"), device=device), diagonal=1
    )
    return mask

