import torch
from pathlib import Path

CHECKPOINT_DIR = Path("src/checkpoints")

def save_checkpoint(model, optimizer, epoch, config, loss):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "loss": loss,
    }

    epoch_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
    torch.save(state, epoch_path)

    best_path = CHECKPOINT_DIR / "best.pt"
    if not best_path.exists():
        torch.save(state, best_path)
    else:
        best = torch.load(best_path, map_location="cpu")
        if loss < best["loss"]:
            torch.save(state, best_path)

    print(f"Saved checkpoint: {epoch_path}")

def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
