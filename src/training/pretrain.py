import os
import signal
import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.data.dataset import MidiDataset, collate_pad_to_longest
from src.data.tokenizer_utils import get_tokenizer
from src.models.remi_transformer import RemiTransformerLM
from src.models.hf_remi_transformer import HFRemiGPT
from src.training.trainer import train_epoch, evaluate
from src.training.loss import compute_remi_loss
from src.utils.logging import init_wandb, log
from src.utils.checkpoint import save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stop_training = False

def handle_interrupt(sig, frame):
    global stop_training
    print("\n[!] Caught interrupt! Exiting gracefully...")
    stop_training = True

def create_dataloader(folder, seq_len, batch_size, shuffle=True):
    dataset = MidiDataset(folder, max_seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pad_to_longest,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

def main(split="pretrain", checkpoint_path="src/checkpoints/best.pt"):

    config = {
        "batch_size": 24,
        "effective_batch_size": 48,
        "seq_len": 1536,
        "epochs": 1,
        "lr": 1e-4,
        "mode": "remi",
        "model_type": "hf_remi",
        "split": split,
    }

    tokenizer = get_tokenizer(config["mode"])

    base_data_path = f"data/tokens/{split}-{config['mode']}"
    train_folder = f"{base_data_path}/train/4-4"
    val_folder = f"{base_data_path}/validation/4-4"
    test_folder = f"{base_data_path}/test/4-4"

    train_loader = create_dataloader(train_folder, config["seq_len"], config["batch_size"], shuffle=True)

    val_loader = create_dataloader(val_folder, config["seq_len"], config["batch_size"], shuffle=False)

    test_loader = create_dataloader(test_folder, config["seq_len"], config["batch_size"], shuffle=False)

    print(f"Vocab size: {tokenizer.vocab_size}, Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}, Test Batches: {len(test_loader)}")

    if config["model_type"] == "hf_remi":
        model = HFRemiGPT(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"]
        ).to(DEVICE)
    else:
        model = RemiTransformerLM(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"]
        ).to(DEVICE)    
    criterion = compute_remi_loss
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01, fused=True)

    accum_steps = config["effective_batch_size"] // config["batch_size"]
    total_training_steps = (len(train_loader) // accum_steps) * config["epochs"]
    num_warmup_steps = int(0.10 * total_training_steps)

    # 1. Linear Warmup: from 1% of LR up to 100% of LR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps)

    # 2. Cosine Decay: from 100% of LR down to 10% of LR (eta_min)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=(total_training_steps - num_warmup_steps), eta_min=config["lr"] * 0.1)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[num_warmup_steps])

    # Check if a checkpoint exists
    start_epoch = 0
    run_id = None
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        ckpt_model_type = checkpoint["config"].get("model_type", "remi")
        if ckpt_model_type != config["model_type"]:
            print(f"[!] Warning: checkpoint model_type='{ckpt_model_type}' != current model_type='{config["model_type"]}'")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch") + 1
        run_id = checkpoint.get("wandb_run_id")
  
    if run_id:
        init_wandb(config, resume="must", id=run_id)
    else:
        init_wandb(config)
        run_id = wandb.run.id
    
    model = torch.compile(model)

    signal.signal(signal.SIGINT, handle_interrupt)

    for epoch in range(start_epoch, config["epochs"]):
        if stop_training: break
        print(f"\n--- Epoch {epoch} ---")

        avg_train_loss = train_epoch(model=model, train_loader=train_loader, val_loader=val_loader, 
                                     optimizer=optimizer, scheduler=scheduler, criterion=criterion, 
                                     device=DEVICE, accum_steps=accum_steps, eval_interval=2400, model_type=config["model_type"],
                                     tokenizer=tokenizer)
        
        print("Running full test evaluation...")
        test_loss = evaluate(model=model, dataloader=test_loader, criterion=criterion, device=DEVICE, limit=None, model_type=config["model_type"])

        print(f"Epoch {epoch} complete. Test Loss: {test_loss:.4f}")
            
        log({
            "test_loss": test_loss,
            "train/avg_loss": avg_train_loss
        })

        save_checkpoint(model, optimizer, scheduler, epoch, config, avg_train_loss, test_loss, run_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain a MIDI GPT model")
    parser.add_argument("--split", type=str, default="pretrain", help="Dataset split to train on (default: pretrain)")
    args = parser.parse_args()
    main(split=args.split)
