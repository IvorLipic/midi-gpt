from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from symusic import Score

from src.data.tokenizer_utils import get_tokenizer
from src.data.tokenizer_utils import compute_token_stats
from src.data.utils import silence_cpp


def tokenize_and_save_dataset(
    tokenizer,
    midi_folder,
    output_folder,
    suffix
):
    midi_folder = Path(midi_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

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
            tokens = tokenizer(score)
            tokens = tokens[0]
            tokens = np.asarray(tokens, dtype=np.int32)

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


def tokenize_recursive(mode, input_root, output_root, cache_dir=None):
    input_root = Path(input_root)
    output_root = Path(output_root)
    tokenizer = get_tokenizer(mode)

    for split_dir in sorted(input_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        for subset_dir in sorted(split_dir.iterdir()):
            if not subset_dir.is_dir():
                continue
            subset_name = subset_dir.name
            midi_dir = subset_dir / "4-4"
            if not midi_dir.is_dir():
                continue

            midi_files = sorted(midi_dir.glob("*.mid"))
            if not midi_files:
                continue

            out_dir = output_root / f"{split_name}-{mode}" / subset_name / "4-4"
            out_dir.mkdir(parents=True, exist_ok=True)

            lengths = []
            print(f"Tokenizing {split_name}/{subset_name}/4-4/ ({len(midi_files)} files) -> {out_dir}")

            for midi_path in tqdm(midi_files, desc=f"{split_name}-{mode}/{subset_name}"):
                try:
                    with silence_cpp():
                        score = Score(midi_path)
                    tokens = tokenizer(score)
                    tokens = tokens[0]
                    tokens = np.asarray(tokens, dtype=np.int32)

                    out_path = out_dir / f"{midi_path.stem}.npz"
                    np.savez_compressed(out_path, tokens=tokens, length=len(tokens))
                    lengths.append(len(tokens))
                except Exception as e:
                    print(f"Error tokenizing {midi_path.name}: {e}")

            if lengths:
                arr = np.array(lengths)
                stats = {
                    "count": int(len(arr)),
                    "max": int(arr.max()),
                    "min": int(arr.min()),
                    "mean": float(round(arr.mean(), 2)),
                    "median": float(round(np.median(arr), 2)),
                    "std": float(round(arr.std(), 2)),
                    "p95": float(round(np.percentile(arr, 95), 2)),
                    "p99": float(round(np.percentile(arr, 99), 2)),
                }
                stats_path = out_dir / "token_stats.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"  Stats: max={stats['max']}, mean={stats['mean']}, p99={stats['p99']}, count={stats['count']}")


def slice_into_8bar_chunks(token_path, tokenizer, suffix="remi"):
    tokens = token_path["tokens"]
    mode = "remi" if hasattr(tokenizer, "vocab_size") else "octuple"

    if mode == "remi":
        bar_token_id = tokenizer.vocab["Bar_None"]
        bar_positions = np.where(tokens == bar_token_id)[0]
    else:
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
        if len(chunk) > 10:
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
    parser.add_argument("--gigamidi", action="store_true")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--compute-stats", default=None)
    args = parser.parse_args()

    if args.compute_stats:
        stats = compute_token_stats(args.compute_stats)
        print(json.dumps(stats, indent=2))
    elif args.slice_8bar:
        mode = args.mode
        input_folder = args.input or f"data/tokens/{mode}"
        output_folder = args.output or f"data/tokens/{mode}_8bar"
        rechunk_tokens(mode, input_folder, output_folder)
    elif args.gigamidi:
        input_root = args.input or "data/GigaMIDI/filtered_loops_v1"
        output_root = args.output or "data/tokens"
        tokenize_recursive(args.mode, input_root, output_root)
    else:
        midi_folder = args.input or "aligned_piano_tracks"
        output_folder = args.output or f"data/tokens/{args.mode}"
        tokenize_dataset(args.mode, midi_folder, output_folder)
