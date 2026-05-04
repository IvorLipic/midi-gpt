
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
    """
    Tokenize MIDIs using miditok (symusic backend) and save token sequences.

    Args:
        tokenizer: REMI or Octuple tokenizer
        midi_folder: folder with processed MIDIs
        output_folder: where tokens will be saved
        suffix: 'remi' or 'oct'
    """
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

    print(f"Tokenizing {len(midi_files)} files → {output_folder}")

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
            print(f"\nError tokenizing {midi_path.name}: {e}")

def tokenize_dataset(mode, midi_folder, output_folder):
    tokenizer = get_tokenizer(mode)
    tokenize_and_save_dataset(
        tokenizer,
        midi_folder,
        output_folder,
        suffix=mode
    )
