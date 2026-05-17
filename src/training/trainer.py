import torch
from tqdm import tqdm
from torch.amp import autocast

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []

    for batch in tqdm(dataloader):
        batch = batch.to(device, non_blocking=True)  # REMI: (B, L) | Octuple: (B, L, F)

        # Slice for Next-Token Prediction
        input_ids = batch[:, :-1, ...]
        target_ids = batch[:, 1:, ...]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = criterion(logits, target_ids)
                    
        loss.backward()
        optimizer.step()
            
        losses.append(loss.item())

    return sum(losses) / len(losses)