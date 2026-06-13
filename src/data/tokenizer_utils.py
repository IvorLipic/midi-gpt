from pathlib import Path
import pickle
import json
import numpy as np
import torch
from tqdm import tqdm
from miditok import REMI, TokenizerConfig

CONFIG_REMI = TokenizerConfig(
    use_chords=False,
    use_rests=False,
    use_tempos=False,
    use_time_signatures=False,
    beat_res={(0, 4): 16},
    num_velocities=32,
    use_velocities=True,
    use_programs=False,
    use_pitchdrum_tokens=False
)

TOKENIZER_CACHE_DIR = Path("data/tokenizers")


def _cache_path(mode: str) -> Path:
    return TOKENIZER_CACHE_DIR / f"{mode}_tokenizer.pkl"


def get_tokenizer(mode: str = "remi") -> REMI:
    cache = _cache_path(mode)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    if mode == "remi":
        tok = REMI(CONFIG_REMI)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(tok, f)

    return tok


def compute_token_stats(folder: str | Path) -> dict:
    folder = Path(folder)
    npz_files = sorted(folder.rglob("*.npz"))
    lengths = []
    for f in tqdm(npz_files, desc=f"Computing stats for {folder.name}"):
        data = np.load(f)
        lengths.append(data["tokens"].shape[0])

    if not lengths:
        print(f"Warning: no .npz files found in {folder}")
        return {}

    arr = np.array(lengths)
    stats = {
        "count": int(len(arr)),
        "total": int(arr.sum()),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "mean": float(round(arr.mean(), 2)),
        "median": float(round(np.median(arr), 2)),
        "std": float(round(arr.std(), 2)),
        "p95": float(round(np.percentile(arr, 95), 2)),
        "p99": float(round(np.percentile(arr, 99), 2)),
    }

    stats_path = folder / "token_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def get_4_bar_prompt(tokens, tokenizer):
    bar_token_id = tokenizer.vocab["Bar_None"]
    bar_positions = (tokens == bar_token_id).nonzero(as_tuple=True)[0]

    if len(bar_positions) >= 5:
        return tokens[:bar_positions[4] + 1]
    return torch.cat([tokens, tokens.new_tensor([bar_token_id])])
