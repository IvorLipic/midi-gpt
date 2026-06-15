import torch
from pathlib import Path

def extract_model_checkpoint(src="src/checkpoints/best.pt", dst="src/checkpoints/model_best.pt"):
    src, dst = Path(src), Path(dst)
    print(f"Loading: {src}")
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    model_state = {
        "model_state_dict": checkpoint["model_state_dict"],
        "config": checkpoint["config"],
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, dst)
    print(f"Saved: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract model-only checkpoint from full training checkpoint")
    parser.add_argument("--src", default="src/checkpoints/best.pt", help="Source full checkpoint path")
    parser.add_argument("--dst", default="src/checkpoints/model_best.pt", help="Destination model-only checkpoint path")
    args = parser.parse_args()
    extract_model_checkpoint(args.src, args.dst)
