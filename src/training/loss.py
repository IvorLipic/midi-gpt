import torch.nn.functional as F

def compute_octuple_loss(logits_per_field, targets):
    losses = []
    # targets shape: (B, L, F)
    for f, logits in enumerate(logits_per_field):
        # logits shape: (B, L, vocab_f) -> reshape to (B*L, vocab_f)
        # targets[:, :, f] shape: (B, L) -> reshape to (B*L)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets[:, :, f].reshape(-1),
            ignore_index=0  # This ignores the pad token for this specific field
        )
        losses.append(loss)
    
    # Return the mean of the losses across all fields
    return sum(losses) / len(losses)

def compute_remi_loss(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0
    )