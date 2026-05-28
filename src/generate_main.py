import sys
import random
import torch
import numpy as np
from pathlib import Path

from src.data.tokenizer_utils import get_tokenizer
from src.data.detokenize import detokenize
from src.generation.generate import generate
from src.utils.checkpoint import load_checkpoint
from src.models.remi_transformer import RemiTransformerLM
from src.models.octuple_transformer import OctupleTransformerLM
from src.models.nested_remi_transformer import NestedRemiTransformerLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_4_bar_prompt(tokens, tokenizer):
    bar_token_id = tokenizer.vocab["Bar_None"]
    bar_positions = (tokens == bar_token_id).nonzero(as_tuple=True)[0]

    if len(bar_positions) >= 5:
        return tokens[:bar_positions[4]]
    elif len(bar_positions) >= 4:
        return tokens[:bar_positions[3]]
    return tokens

def main(checkpoint_path=None, top_k=40, n_samples=1, split="pretrain"):
    if checkpoint_path is None:
        checkpoint_path = "src/checkpoints/best.pt"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run 'python -m src.training.pretrain' first to train a model.")
        sys.exit(1)

    checkpoint = load_checkpoint(checkpoint_path, device=DEVICE)
    config = checkpoint["config"]
    mode = config["mode"]
    model_type = config.get("model_type", mode)
    split = config.get("split", split)

    tokenizer = get_tokenizer(mode)

    if model_type == "nested_remi":
        model = NestedRemiTransformerLM(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"],
        ).to(DEVICE)
    elif mode == "remi":
        model = RemiTransformerLM(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"],
        ).to(DEVICE)
    else:
        vocab_sizes = [len(v) for v in tokenizer.vocab]
        model = OctupleTransformerLM(
            vocab_sizes_per_field=vocab_sizes,
            max_len=config["seq_len"],
        ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    token_dir = Path("data/tokens") / f"{split}-{mode}" / "test" / "4-4"
    token_files = sorted(token_dir.glob("*.npz"))
    if not token_files:
        print(f"No token files found in {token_dir}")
        sys.exit(1)

    output_dir = Path("data/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = random.choice(token_files)
    data = np.load(sample_path)
    tokens = torch.from_numpy(data["tokens"]).long()

    if mode == "remi":
        prompt = get_4_bar_prompt(tokens, tokenizer)
    else:
        bar_field = tokens[:, 0]
        bar_mask = bar_field < 4
        prompt = tokens[bar_mask]

    max_new_tokens = len(prompt)
    stem = Path(sample_path).stem

    for i in range(n_samples):
        print(f"Sample {i}: prompt length {len(prompt)} tokens (from {sample_path.name}), generating {max_new_tokens} new tokens")

        generated = generate(model, prompt, max_new_tokens, DEVICE, top_k=top_k)
        print(f"Generated {len(generated)} tokens total")

        out_path = output_dir / f"generated_{stem}_{mode}_sample{i}.mid"
        detokenize(generated, tokenizer, out_path, prompt_tokens=prompt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MIDI from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: src/checkpoints/best.pt)")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--split", type=str, default="pretrain", help="Dataset split for prompt selection (default: pretrain)")
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        n_samples=args.n_samples,
        split=args.split,
    )
