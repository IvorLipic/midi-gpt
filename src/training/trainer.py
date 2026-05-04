import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []

    for batch in tqdm(dataloader):
        batch = batch.to(device)  # REMI: (B, L) | Octuple: (B, L, F)

        # 1. Slice for Next-Token Prediction
        input_ids = batch[:, :-1, ...]
        target_ids = batch[:, 1:, ...]

        # 2. Generate Causal Mask
        L = input_ids.size(1)
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )

        # 3. Forward Pass
        # logits will be a Tensor for REMI, or a List[Tensor] for Octuple
        logits = model(input_ids, attn_mask=mask)

        # 4. Compute Loss
        loss = criterion(logits, target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)