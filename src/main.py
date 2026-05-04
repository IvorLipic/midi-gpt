import torch
from torch.utils.data import DataLoader

from src.data.dataset import MidiDataset
from src.data.tokenize import get_tokenizer
from src.models.remi_transformer import RemiTransformerLM 
from src.models.octuple_transformer import OctupleTransformerLM
from src.training.trainer import train_epoch
from src.training.loss import compute_octuple_loss, compute_remi_loss
from src.utils.logging import init_wandb, log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    config = {
        "batch_size": 8,
        "seq_len": 256,
        "epochs": 50,
        "lr": 1e-4,
        "mode": "remi"
    }

    init_wandb(config)

    tokenizer = get_tokenizer(config["mode"])

    dataset = MidiDataset("data/tokens/" + config["mode"], seq_len=config["seq_len"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    print(tokenizer.vocab_size)

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
        

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        loss = train_epoch(model, loader, optimizer, criterion, DEVICE)

        print(f"Epoch {epoch}: {loss:.4f}")
        log({"epoch": epoch, "loss": loss})


if __name__ == "__main__":
    main()