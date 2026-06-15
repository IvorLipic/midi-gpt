import sys

import torch
import numpy as np
from pathlib import Path

from src.data.tokenizer_utils import get_tokenizer, get_4_bar_prompt
from src.data.detokenize import detokenize
from src.generation.generate import generate
from src.utils.checkpoint import load_checkpoint
from src.models.remi_transformer import RemiTransformerLM
from src.models.hf_remi_transformer import HFRemiGPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(checkpoint_path, top_k, top_p, temperature, n_samples, prompt, single_track=False):
    if checkpoint_path is None:
        checkpoint_path = "src/checkpoints/best.pt"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run 'python -m src.training.pretrain' first to train a model.")
        sys.exit(1)

    if prompt is None:
        print("Error: --prompt is required")
        sys.exit(1)

    checkpoint = load_checkpoint(checkpoint_path, device=DEVICE)
    config = checkpoint["config"]
    mode = config["mode"]
    model_type = config.get("model_type", mode)

    tokenizer = get_tokenizer(mode)

    if model_type == "hf_remi":
        model = HFRemiGPT(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"],
        ).to(DEVICE)
    else:
        model = RemiTransformerLM(
            vocab_size=tokenizer.vocab_size,
            max_len=config["seq_len"],
        ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    output_dir = Path("data/generations")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(prompt)
    stem = prompt_path.stem

    if prompt_path.suffix.lower() in (".mid", ".midi"):
        from symusic import Score
        score = Score(str(prompt_path))
        all_tokens = tokenizer(score)
        tokens = torch.tensor(all_tokens[0], dtype=torch.long)
    elif prompt_path.suffix.lower() == ".npz":
        data = np.load(prompt_path)
        tokens = torch.from_numpy(data["tokens"]).long()
    else:
        print(f"Unsupported prompt file: {prompt_path} (use .mid, .midi, or .npz)")
        sys.exit(1)

    prompt_tokens = get_4_bar_prompt(tokens, tokenizer)

    for i in range(n_samples):
        print(f"Sample {i}: prompt length {len(prompt_tokens)} tokens (from {prompt_path.name}), generating up to 8 bars")

        generated = generate(model, prompt_tokens, tokenizer, DEVICE, temperature=temperature, top_k=top_k, top_p=top_p)
        print(f"Generated {len(generated)} tokens total")

        out_path = output_dir / f"generated_{stem}_{mode}_sample{i}.mid"
        detokenize(generated, tokenizer, out_path, prompt_tokens=prompt_tokens, include_prompt_track=not single_track)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MIDI from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: src/checkpoints/best.pt)")
    parser.add_argument("--prompt", type=str, required=True, help="Path to .mid, .midi, or .npz file to use as prompt")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--single-track", action="store_true", help="Output only the generated track (omit the prompt track)")
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        n_samples=args.n_samples,
        single_track=args.single_track,
    )

# python -m src.generation.generate_main --prompt data/handcrafted_test_midis/chords_bass_melody_A#min.mid --single-track