import tempfile
import traceback
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from torch.amp import autocast
from symusic import Score

from src.data.tokenizer_utils import get_4_bar_prompt
from src.data.detokenize import detokenize
from src.data.utils import silence_cpp
from src.generation.generate import generate
from src.utils.logging import log
from src.utils.visualization import score_to_pianoroll


@torch.no_grad()
def evaluate_handcrafted(model, tokenizer, device, model_type="hf_remi", top_k=None, top_p=0.92, temperature=0.9, step_id=0):
    model.eval()
    midi_dir = Path("data/handcrafted_test_midis")
    midi_files = sorted(midi_dir.glob("*.mid"))

    if not midi_files:
        print("No handcrafted test MIDIs found.")
        return

    for midi_path in midi_files:
        try:
            with silence_cpp():
                score = Score(midi_path)
            tok_seq = tokenizer(score)[0]
            full_tokens = torch.tensor(tok_seq.ids).long()
            prompt = get_4_bar_prompt(full_tokens, tokenizer)
            generated = generate(model, prompt, tokenizer, device, temperature=temperature, top_k=top_k, top_p=top_p)

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                out_path = Path(f.name)
            result_score = detokenize(generated, tokenizer, out_path, prompt_tokens=prompt)
            pianoroll = score_to_pianoroll(result_score)
            wandb.log({
                f"handcrafted/{midi_path.stem}": wandb.Image(
                    pianoroll, caption=f"{midi_path.stem} | gen {len(generated)-len(prompt)} tok"
                )
            })
            import matplotlib.pyplot as plt
            Path("data/test").mkdir(parents=True, exist_ok=True)
            plt.imsave(f"data/test/pianoroll_{midi_path.stem}_step{step_id}.png", pianoroll)
            print(f"Saved pianoroll image to data/test/pianoroll_{midi_path.stem}_step{step_id}.png")
            out_path.unlink(missing_ok=True)
        except Exception as e:
            traceback.print_exc()
            print(f"Error evaluating {midi_path.name}: {e}")


def train_epoch(model, train_loader, val_loader, optimizer, scheduler, criterion, device, accum_steps=2, eval_interval=1000, model_type="remi", tokenizer=None):
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
            if tokenizer is not None:
                evaluate_handcrafted(model, tokenizer, device, model_type=model_type, step_id=optimized_steps + 1)
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