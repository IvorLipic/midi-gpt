import torch
from tqdm import tqdm
from torch.amp import autocast
from src.utils.logging import log

def train_epoch(model, train_loader, val_loader, optimizer, scheduler, criterion, device, accum_steps=2, eval_interval=4000, nested=False):
    model.train()
    total_loss = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar):
        if nested:
            # NJT path: batch is a list of (L_i,) tensors
            batch = [t.to(device, non_blocking=True) for t in batch]
            input_list = [t[:-1] for t in batch]
            target = torch.cat([t[1:] for t in batch])
        else:
            batch = batch.to(device, non_blocking=True)
            # (B, L)
            input_ids = batch[:, :-1, ...]
            target = batch[:, 1:, ...]
            src_key_padding_mask = (input_ids == 0)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            if nested:
                logits = model(input_list)
                loss = criterion(logits, target)
            else:
                logits = model(input_ids, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(logits, target)
            loss_scaled = loss / accum_steps
                    
        loss_scaled.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            current_lr = scheduler.get_last_lr()[0]
            log({"train/loss": loss.item(), "train/lr": current_lr})
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Periodic Evaluation
        if (i + 1) % eval_interval == 0:
            val_loss = evaluate(model, val_loader, criterion, device, limit=None, nested=nested)
            log({"val/step_loss": val_loss})
            model.train()
            
        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, limit=None, nested=False):
    model.eval()
    losses = []

    pbar = tqdm(dataloader, desc="Evaluating")
    for i, batch in enumerate(pbar):
        if limit and i >= limit: break

        if nested:
            batch = [t.to(device, non_blocking=True) for t in batch]
            input_list = [t[:-1] for t in batch]
            target = torch.cat([t[1:] for t in batch])
        else:
            batch = batch.to(device, non_blocking=True)
            input_ids = batch[:, :-1, ...]
            target = batch[:, 1:, ...]
            src_key_padding_mask = (input_ids == 0)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            if nested:
                logits = model(input_list)
                loss = criterion(logits, target)
            else:
                logits = model(input_ids, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(logits, target)
            
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else 0.0