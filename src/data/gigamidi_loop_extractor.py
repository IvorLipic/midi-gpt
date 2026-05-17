import os
import ast
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from symusic import Score, Track, Note, Tempo, TimeSignature

import multiprocessing as mp
from functools import partial

from src.data.utils import silence_cpp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA = os.path.join(SCRIPT_DIR, "../../data/GigaMIDI/GigaMIDI-metadata-filtered-v1.csv")
DATASET_ROOT = os.path.join(SCRIPT_DIR, "../../data/GigaMIDI")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "../../data/GigaMIDI/extracted_loops_v1")


# =========================
# Utils
# =========================

def parse_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return [x] if x != "" else []


def ensure_dir():
    splits = ["train", "test", "validation"]
    ts_types = ["4-4", "2-4", "unknown"]
    for s in splits:
        for t in ts_types:
            os.makedirs(os.path.join(OUTPUT_ROOT, s, t), exist_ok=True)


# =========================
# Metadata
# =========================

def load_metadata(csv_path: str) -> pd.DataFrame:
    print("Loading metadata...")
    df = pd.read_csv(csv_path)

    list_columns = [
        "loop_track_idx",
        "loop_instrument_type",
        "loop_start",
        "loop_end",
        "loop_duration_beats",
        "loop_note_density",
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)

    return df


# =========================
# Path resolution
# =========================

def resolve_midi_path(file_path: str, dataset_root: str) -> Optional[str]:
    """
    Convert CSV file_path to absolute path.
    """
    if pd.isna(file_path):
        return None

    # remove leading "./"
    file_path = file_path.lstrip("./")

    abs_path = os.path.join(dataset_root, file_path)

    if os.path.exists(abs_path):
        return abs_path

    return None


def infer_split(file_path: str) -> str:
    """
    Determine dataset split from path.
    """
    path = file_path.lower()

    if "train" in path:
        return "train"
    elif "test" in path:
        return "test"
    elif "validation" in path:
        return "validation"
    else:
        return "unknown"

# =========================
# Loop augmentation and validation
# =========================

def get_loop_time_signature(score, start_tick, end_tick):
    """
    Returns: "4-4", "2-4", or "unknown"
    """

    if not score.time_signatures:
        return "unknown"

    time_sigs = sorted(score.time_signatures, key=lambda x: x.time)

    overlapping = []

    for i, ts in enumerate(time_sigs):
        ts_start = ts.time
        ts_end = time_sigs[i + 1].time if i + 1 < len(time_sigs) else score.end()

        if not (end_tick <= ts_start or start_tick >= ts_end):
            overlapping.append(ts)

    if len(overlapping) != 1:
        return None  # invalid (multiple TS)

    ts = overlapping[0]

    if ts.denominator != 4:
        return None

    if ts.numerator == 4:
        return "4-4"
    elif ts.numerator == 2:
        return "2-4"
    else:
        return None

def get_bar_starts(score):
    """
    Compute all bar start ticks using time signatures.
    """
    ticks_per_quarter = score.ticks_per_quarter

    if not score.time_signatures:
        # default 4/4
        ticks_per_bar = 4 * ticks_per_quarter
        total_ticks = score.end()
        return list(range(0, int(total_ticks) + ticks_per_bar, int(ticks_per_bar)))

    time_sigs = sorted(score.time_signatures, key=lambda x: x.time)

    bar_starts = []

    for i, ts in enumerate(time_sigs):
        start_tick = ts.time
        end_tick = time_sigs[i + 1].time if i + 1 < len(time_sigs) else score.end()

        ticks_per_bar = (ticks_per_quarter * 4 / ts.denominator) * ts.numerator

        t = start_tick
        while t < end_tick:
            bar_starts.append(int(t))
            t += ticks_per_bar

    return sorted(set(bar_starts))

def get_previous_bar_start(score, tick):
    """
    Find the closest bar start <= tick
    """
    bar_starts = get_bar_starts(score)

    prev = 0
    for b in bar_starts:
        if b > tick:
            break
        prev = b

    return prev

def get_tempo_at_tick(score, tick):
    tempos = sorted(score.tempos, key=lambda x: x.time)

    current = tempos[0] if tempos else None

    for t in tempos:
        if t.time > tick:
            break
        current = t

    return current

# =========================
# Loop extraction
# =========================

def extract_loop_track(
    score,
    track_idx: int,
    start_tick: int,
    end_tick: int,
    ts_type 
):

    if track_idx < 0 or track_idx >= len(score.tracks):
        return None

    track = score.tracks[track_idx]

    if track.is_drum:
        return None

    new_score = Score()
    new_score.ticks_per_quarter = score.ticks_per_quarter

    tempo = get_tempo_at_tick(score, start_tick)

    if tempo is not None:
        new_score.tempos = [Tempo(time=0, qpm=tempo.qpm)]
    
    if ts_type == "4-4":
        new_score.time_signatures = [TimeSignature(time=0, numerator=4, denominator=4)]
    elif ts_type == "2-4":
        new_score.time_signatures = [TimeSignature(time=0, numerator=2, denominator=4)]
    else:
        new_score.time_signatures = [TimeSignature(time=0, numerator=4, denominator=4)]

    new_track = Track(
        name=track.name,
        program=track.program,
        is_drum=False,
    )

    for note in track.notes:
        note_start = note.time
        note_end = note.time + note.duration

        if note_end <= start_tick or note_start >= end_tick:
            continue

        new_start = max(note_start, start_tick) - start_tick
        new_end = min(note_end, end_tick) - start_tick

        if new_end > new_start:
            new_track.notes.append(
                Note(
                    time=new_start,
                    duration=new_end - new_start,
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )

    # remove empty / garbage loops
    if len(new_track.notes) < 10:
        return None

    new_score.tracks.append(new_track)
    return new_score


def is_valid_8bar_loop(duration_beats: float, tol: float = 0.5) -> bool:
    return abs(duration_beats - 32.0) < tol

# =========================
# Main extraction - single thread
# =========================

def process_midi_row(row_tuple, dataset_root: str, output_root: str) -> List[Dict[str, Any]]:
    """
    Unified worker function used by both sequential and parallel processing.
    """
    # row_tuple can be (index, row) from iterrows or just the row depending on caller
    if isinstance(row_tuple, tuple):
        _, row = row_tuple
    else:
        row = row_tuple

    file_path = row.get("file_path", None)
    midi_path = resolve_midi_path(file_path, dataset_root)

    if midi_path is None:
        return []

    split = infer_split(file_path)
    if split not in ["train", "test", "validation"]:
        return []

    results = []
    
    with silence_cpp():
        try:
            score = Score(midi_path)
            
            loop_tracks = row.get("loop_track_idx", [])
            loop_starts = row.get("loop_start", [])
            loop_ends = row.get("loop_end", [])
            loop_beats = row.get("loop_duration_beats", [])
            loop_density = row.get("loop_note_density", [])
            loop_types = row.get("loop_instrument_type", [])

            n_loops = min(len(loop_tracks), len(loop_starts), len(loop_ends))

            for i in range(n_loops):
                track_idx = int(loop_tracks[i])
                start_tick = int(loop_starts[i])
                end_tick = int(loop_ends[i])
                beats = float(loop_beats[i]) if i < len(loop_beats) else None

                # 1. Validation
                if beats is None or not is_valid_8bar_loop(beats):
                    continue

                ts_type = get_loop_time_signature(score, start_tick, end_tick)
                if ts_type is None:
                    continue

                # 2. Alignment
                aligned_start = get_previous_bar_start(score, start_tick)
                aligned_end = aligned_start + (end_tick - start_tick)

                # 3. Extraction
                loop_score = extract_loop_track(score, track_idx, aligned_start, aligned_end, ts_type)
                if loop_score is None:
                    continue
                
                out_name = f"{row['md5']}_loop{i}_track{track_idx}_{aligned_start}.mid"
                out_path = os.path.join(output_root, split, ts_type, out_name)

                loop_score.dump_midi(out_path)

                results.append({
                    "output_midi": out_path,
                    "split": split,
                    "md5": row["md5"],
                    "loop_idx": i,
                    "track_idx": track_idx,
                    "duration_beats": beats,
                    "note_density": float(loop_density[i]) if i < len(loop_density) else None,
                    "instrument_type": loop_types[i] if i < len(loop_types) else None,
                    "music_styles_curated": row["music_styles_curated"],
                    "music_style_scraped": row["music_style_scraped"],
                    "tempo": loop_score.tempos[0].qpm if loop_score.tempos else None
                })
        except Exception:
            return []
    
    return results

def save_manifest(rows, output_root):
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.sort_values(by=["md5", "loop_idx"])
    manifest_path = os.path.join(output_root, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nDone. Extracted {len(manifest)} loops")
    print(f"Saved manifest to {manifest_path}")

def run_sequential():
    df = load_metadata(METADATA)
    ensure_dir()

    extracted_rows = []
    for row_tuple in tqdm(df.iterrows(), total=len(df), desc="Sequential Processing"):
        res = process_midi_row(row_tuple, DATASET_ROOT, OUTPUT_ROOT)
        extracted_rows.extend(res)
    
    save_manifest(extracted_rows, OUTPUT_ROOT)

def run_parallel():
    df = load_metadata(METADATA)
    ensure_dir()

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} workers.")
    worker_func = partial(process_midi_row, dataset_root=DATASET_ROOT, output_root=OUTPUT_ROOT)
    
    extracted_rows = []
    rows = list(df.iterrows())
    
    with mp.Pool(processes=num_workers) as pool:
        for result_list in tqdm(pool.imap_unordered(worker_func, rows), total=len(rows), desc="Parallel Processing"):
            if result_list:
                extracted_rows.extend(result_list)
    
    save_manifest(extracted_rows, OUTPUT_ROOT)

if __name__ == "__main__":
    #run_sequential() # single thread
    run_parallel() # multi threaded
