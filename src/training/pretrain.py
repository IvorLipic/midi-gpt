import torch
from torch.utils.data import DataLoader

from src.data.dataset import MidiDataset
from src.data.tokenizer_utils import get_tokenizer
from src.models.remi_transformer import RemiTransformerLM 
from src.models.octuple_transformer import OctupleTransformerLM
from src.training.trainer import train_epoch
from src.training.loss import compute_octuple_loss, compute_remi_loss
from src.utils.logging import init_wandb, log
from src.utils.checkpoint import save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(split="pretrain"):

    config = {
        "batch_size": 32,
        "seq_len": None,
        "epochs": 100,
        "lr": 5e-4,
        "mode": "remi",
        "split": split,
    }

    init_wandb(config)

    tokenizer = get_tokenizer(config["mode"])

    token_folder = f"data/tokens/{split}-{config['mode']}/train/4-4"
    dataset = MidiDataset(token_folder, seq_len=None, mode=config["mode"], max_seq_len=config["seq_len"])

    config["seq_len"] = dataset.seq_len
    print(f"Using seq_len={config['seq_len']}, dataset={len(dataset)} samples")

    loader = DataLoader(dataset, 
                        batch_size=config["batch_size"], 
                        shuffle=True, 
                        pin_memory=True,
                        num_workers=4,
                        persistent_workers=True,
                        prefetch_factor=2)

    print(f"Vocab size: {tokenizer.vocab_size}, Batches per epoch: {len(loader)}")

    if config["mode"] == "remi":
        model = RemiTransformerLM(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"]
        ).to(DEVICE)    
        criterion = compute_remi_loss
    else:
        vocab_sizes = [len(v) for v in tokenizer.vocab]
        model = OctupleTransformerLM(
            vocab_sizes_per_field=vocab_sizes,
            max_len=config["seq_len"]
        ).to(DEVICE)
        criterion = compute_octuple_loss
    
    model = torch.compile(model, mode="max-autotune")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], fused=True)
    best_loss = float("inf")

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        for epoch in range(config["epochs"]):
            loss = train_epoch(model, loader, optimizer, criterion, DEVICE)

            print(f"Epoch {epoch}: {loss:.4f}")
            log({"epoch": epoch, "loss": loss})

            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, config, loss)

            if loss < best_loss:
                best_loss = loss
                print(f"New best loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain a MIDI GPT model")
    parser.add_argument("--split", type=str, default="pretrain", help="Dataset split to train on (default: pretrain)")
    args = parser.parse_args()
    main(split=args.split)
