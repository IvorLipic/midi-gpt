from pathlib import Path
import numpy as np
import torch
from miditok import REMI, Octuple, TokenizerConfig
from symusic import Score, Track

config_remi = TokenizerConfig(
    use_chords=False,
    use_rests=False,
    use_tempos=False,
    use_time_signatures=False,
    beat_res={(0, 4): 16},
    num_velocities=32,
    use_velocities=True,
    use_programs=False,
)

config_oct = TokenizerConfig(
    use_chords=False,
    use_rests=False,
    use_tempos=False,
    use_time_signatures=False,
    beat_res={(0, 4): 16},
    num_velocities=32,
    use_velocities=True,
    use_programs=False,
)
config_oct.additional_params["max_bar_embedding"] = 64

def get_tokenizer(mode="remi"):
    if mode == "remi":
        return REMI(config_remi)
    elif mode == "octuple":
        return Octuple(config_oct)

def detokenize(tokens, tokenizer, output_path, prompt_tokens=None):
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

    if prompt_tokens is not None:
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

    if isinstance(score, Score):
        score.dump_midi(output_path)
    else:
        score.dump(str(output_path))

    print(f"Saved MIDI to {output_path}")
    return score
