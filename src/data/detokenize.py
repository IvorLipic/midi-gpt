from pathlib import Path
import numpy as np
import torch
from symusic import Score, Track

from src.data.tokenizer_utils import get_tokenizer

def detokenize(tokens, tokenizer, output_path, prompt_tokens=None, include_prompt_track=True):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    if prompt_tokens is not None and isinstance(prompt_tokens, torch.Tensor):
        prompt_tokens = prompt_tokens.cpu().numpy()

    if tokens.ndim == 1:
        tokens_list = [tokens.tolist()]
    else:
        tokens_list = tokens.tolist()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score = tokenizer(tokens_list)
    if isinstance(score, list):
        score = score[0]

    if prompt_tokens is not None and include_prompt_track:
        if prompt_tokens.ndim == 1:
            prompt_list = [prompt_tokens.tolist()]
        else:
            prompt_list = prompt_tokens.tolist()
        prompt_score = tokenizer(prompt_list)
        if isinstance(prompt_score, list):
            prompt_score = prompt_score[0]

        combined = Score(score.tpq)
        combined.tempos = score.tempos
        combined.time_signatures = score.time_signatures
        score.tracks[0].name = "Generated"
        combined.tracks.append(score.tracks[0])
        prompt_score.tracks[0].name = "Prompt"
        combined.tracks.append(prompt_score.tracks[0])
        score = combined
    elif prompt_tokens is not None:
        score.tracks[0].name = "Generated"

    if isinstance(score, Score):
        score.dump_midi(output_path)
    else:
        score.dump(str(output_path))

    print(f"Saved MIDI to {output_path}")
    return score
