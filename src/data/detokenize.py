from pathlib import Path
import numpy as np
import torch
from miditok import REMI, Octuple, TokenizerConfig
from symusic import Score

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

def detokenize(tokens, tokenizer, output_path):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()

    if tokens.ndim == 1:
        tokens_list = [tokens.tolist()]
    else:
        tokens_list = tokens.tolist()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score = tokenizer(tokens_list)
    if isinstance(score, list):
        score = score[0]

    if isinstance(score, Score):
        score.dump_midi(output_path)
    else:
        score.dump(str(output_path))

    print(f"Saved MIDI to {output_path}")
    return score
