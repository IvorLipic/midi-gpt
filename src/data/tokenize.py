
from pathlib import Path
from tqdm import tqdm
import numpy as np
from symusic import Score
from miditok import REMI, Octuple, TokenizerConfig

config_remi = TokenizerConfig(
    use_chords=False,
    use_rests=False,
    use_tempos=False,
    use_time_signatures=False,
    beat_res={(0, 4): 16},
    num_velocities=32,
    use_velocities=True,
    use_programs=False
)

config_oct = TokenizerConfig(
    use_chords=False,
    use_rests=False,
    use_tempos=False,
    use_time_signatures=False,
    beat_res={(0, 4): 16},
    num_velocities=32,
    use_velocities=True,
    use_programs=False
)
config_oct.additional_params["max_bar_embedding"] = 64

def get_tokenizer(mode="remi"):
    if mode == "remi":
        return REMI(config_remi)
    elif mode == "octuple":
        return Octuple(config_oct)

def tokenize_and_save_dataset(
    tokenizer,
    midi_folder,
    output_folder,
    suffix
):
    midi_folder = Path(midi_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

     # ---- HARD RESET OUTPUT FOLDER ----
    if output_folder.exists():
        for f in output_folder.glob("*.npz"):
            f.unlink()
    else:
        output_folder.mkdir()

    midi_files = sorted(midi_folder.glob("*.mid"))

    print(f"Tokenizing {len(midi_files)} files - {output_folder}")

    for midi_path in tqdm(midi_files, desc=f"Tokenizing ({suffix})"):
        try:
            score = Score(midi_path)

            # ---- TOKENIZE ----
            tokens = tokenizer(score)

            # miditok returns a list of sequences (one per track)
            tokens = tokens[0]

            tokens = np.asarray(tokens, dtype=np.int32)

            # ---- SAVE ----
            out_path = output_folder / f"{midi_path.stem}_{suffix}.npz"
            np.savez_compressed(
                out_path,
                tokens=tokens,
                length=len(tokens)
            )

        except Exception as e:
            print(f"Error tokenizing {midi_path.name}: {e}")

def tokenize_dataset(mode, midi_folder, output_folder):
    tokenizer = get_tokenizer(mode)
    tokenize_and_save_dataset(
        tokenizer,
        midi_folder,
        output_folder,
        suffix=mode
    )

def slice_into_8bar_chunks(token_path, tokenizer, suffix="remi"):
    tokens = token_path["tokens"]
    mode = "remi" if hasattr(tokenizer, "vocab_size") else "octuple"

    if mode == "remi":
        bar_token_id = tokenizer.vocab["Bar_None"]
        bar_positions = np.where(tokens == bar_token_id)[0]
    else:
        # Octuple: field 0 is bar number, find indices where bar resets
        bar_positions = np.where(tokens[:, 0] == 0)[0]

    if len(bar_positions) == 0:
        return [tokens]

    chunks = []
    bars_per_chunk = 8
    bar_step = np.diff(bar_positions, prepend=-1)

    for i in range(0, len(bar_positions), bars_per_chunk):
        start_bar = bar_positions[i]
        end_bar = bar_positions[min(i + bars_per_chunk, len(bar_positions))] if i + bars_per_chunk < len(bar_positions) else len(tokens)
        chunk = tokens[start_bar:end_bar]
        if len(chunk) > 10:  # skip tiny fragments
            chunks.append(chunk)

    return chunks

def rechunk_tokens(mode, input_folder, output_folder):
    tokenizer = get_tokenizer(mode)
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if output_folder.exists():
        for f in output_folder.glob("*.npz"):
            f.unlink()

    token_files = sorted(input_folder.glob("*.npz"))
    chunk_idx = 0

    print(f"Slicing {len(token_files)} files into 8-bar chunks - {output_folder}")

    for token_path in tqdm(token_files, desc=f"Chunking ({mode})"):
        try:
            data = np.load(token_path)
            chunks = slice_into_8bar_chunks(data, tokenizer, suffix=mode)
            for chunk in chunks:
                out_name = f"{token_path.stem}_chunk{chunk_idx:04d}_{mode}.npz"
                out_path = output_folder / out_name
                np.savez_compressed(out_path, tokens=chunk, length=len(chunk))
                chunk_idx += 1
        except Exception as e:
            print(f"Error chunking {token_path.name}: {e}")

    print(f"Created {chunk_idx} 8-bar chunks in {output_folder}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="remi")
    parser.add_argument("--slice-8bar", action="store_true")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.slice_8bar:
        mode = args.mode
        input_folder = args.input or f"data/tokens/{mode}"
        output_folder = args.output or f"data/tokens/{mode}_8bar"
        rechunk_tokens(mode, input_folder, output_folder)
    else:
        midi_folder = "aligned_piano_tracks"
        output_folder = f"data/tokens/{args.mode}"
        tokenize_dataset(args.mode, midi_folder, output_folder)

