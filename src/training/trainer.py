import torch
from tqdm import tqdm
from torch.amp import autocast
from src.utils.logging import log

def train_epoch(model, train_loader, val_loader, optimizer, scheduler, criterion, device, accum_steps=2, eval_interval=4000, model_type="remi"):
    model.train()
    
    total_loss = 0
    optimized_steps = 0
    running_accum_loss = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_ids, target = batch['inputs'], batch['targets']

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            if model_type == "hf_remi":
                attention_mask = (input_ids != 0).long()
                logits = model(input_ids, attention_mask=attention_mask)
            else:
                src_key_padding_mask = (input_ids == 0)
                logits = model(input_ids, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits, target)
            loss_scaled = loss / accum_steps
                    
        loss_scaled.backward()

        running_accum_loss += loss.item()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            avg_step_loss = running_accum_loss / accum_steps
            total_loss += avg_step_loss
            optimized_steps += 1
            
            current_lr = scheduler.get_last_lr()[0]
            log({"train/loss": avg_step_loss, "train/lr": current_lr})
            pbar.set_postfix({"loss": f"{avg_step_loss:.4f}"})

            running_accum_loss = 0

        # Periodic Evaluation
        if (i + 1) % eval_interval == 0:
            val_loss = evaluate(model, val_loader, criterion, device, model_type=model_type)
            log({"val/step_loss": val_loss})
            model.train()

    return total_loss / optimized_steps if optimized_steps > 0 else 0.0

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, limit=None, model_type="remi"):
    model.eval()
    losses = []

    pbar = tqdm(dataloader, desc="Evaluating")
    for i, batch in enumerate(pbar):
        if limit and i >= limit: break

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_ids, target = batch['inputs'], batch['targets']

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            if model_type == "hf_remi":
                attention_mask = (input_ids != 0).long()
                logits = model(input_ids, attention_mask=attention_mask)
            else:
                src_key_padding_mask = (input_ids == 0)
                logits = model(input_ids, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits, target)
            
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else 0.0