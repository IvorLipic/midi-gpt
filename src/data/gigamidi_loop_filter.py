import os
import math
import shutil
import argparse
import multiprocessing as mp
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from symusic import Score

from src.data.utils import silence_cpp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "../../data/GigaMIDI/extracted_loops_v1/manifest.csv")
SOURCE_ROOT = os.path.join(SCRIPT_DIR, "../../data/GigaMIDI/extracted_loops_v1")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "../../data/GigaMIDI/filtered_loops_v1")

METRIC_COLUMNS = [
    "n_pitches_used",
    "n_pitch_classes_used",
    "pitch_range",
    "empty_beat_rate",
    "empty_bar_rate",
    "polyphony",
    "polyphony_rate",
    "scale_consistency",
    "pitch_entropy",
    "pitch_class_entropy",
    "groove_consistency_bar",
    "groove_consistency_4bar",
    "is_identical_4bar_loop"
]


# =========================
# Helpers
# =========================


def _extract_notes(score):
    """Extract (times, ends, pitches) numpy arrays from all non-drum tracks.

    Returns None if no notes found.
    """
    notes = []
    for track in score.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            notes.append((note.time, note.time + note.duration, note.pitch))

    if not notes:
        return None

    times = np.array([n[0] for n in notes], dtype=np.int64)
    ends = np.array([n[1] for n in notes], dtype=np.int64)
    pitches = np.array([n[2] for n in notes], dtype=np.int64)
    return times, ends, pitches

def _extract_bar_events(times, ends, pitches, start, end):
    """Extract sorted note events in [start, end) with relative timing."""
    mask = (times >= start) & (times < end)

    t = times[mask] - start
    e = ends[mask] - start
    p = pitches[mask]

    # (onset, duration, pitch) is better than (onset, end, pitch)
    events = list(zip(t.tolist(), (e - t).tolist(), p.tolist()))
    return sorted(events)


def is_identical_4bar_loop(times, ends, pitches, ticks_per_bar):
    """Check if first 4 bars are EXACTLY identical to last 4 bars."""
    split = 4 * ticks_per_bar

    first = _extract_bar_events(times, ends, pitches, 0, split)
    second = _extract_bar_events(times, ends, pitches, split, 2 * split)

    return first == second

def debug_check_identical_loop(midi_path):
    score = Score(midi_path)

    result = _extract_notes(score)
    if result is None:
        print("No notes found")
        return

    times, ends, pitches = result
    ticks_per_bar = 4 * score.ticks_per_quarter

    identical = is_identical_4bar_loop(times, ends, pitches, ticks_per_bar)

    print(f"{midi_path}")
    print(f"Identical 4-bar loop: {identical}")


# =========================
# Individual metric functions (adapted from muspy to symusic)
# =========================


def _n_pitches_used(pitches: np.ndarray) -> int:
    """Return the number of unique pitches used.

    Adapted from muspy.n_pitches_used — drum tracks already excluded upstream.
    """
    return len(np.unique(pitches))


def _n_pitch_classes_used(pitches: np.ndarray) -> int:
    """Return the number of unique pitch classes used.

    Adapted from muspy.n_pitch_classes_used.
    """
    return len(np.unique(pitches % 12))


def _pitch_range(pitches: np.ndarray) -> int:
    """Return the pitch range (highest - lowest).

    Adapted from muspy.pitch_range.
    """
    return int(pitches.max() - pitches.min())


def _empty_beat_rate(times: np.ndarray, ends: np.ndarray, ticks_per_beat: int, length: int) -> float:
    """Return the ratio of empty beats (no note played).

    Adapted from muspy.empty_beat_rate.
    Empty-beat rate = #(empty_beats) / #(beats).
    """
    n_beats = length // ticks_per_beat + 1
    is_empty = np.ones(n_beats, dtype=bool)

    beat_starts = times // ticks_per_beat
    beat_ends = ends // ticks_per_beat
    for bs, be in zip(beat_starts, beat_ends):
        is_empty[bs : be + 1] = False

    return float(np.mean(is_empty))


def _empty_bar_rate(times: np.ndarray, ends: np.ndarray, measure_resolution: int, length: int) -> float:
    """Return the ratio of empty bars (no note played).

    Adapted from muspy.empty_measure_rate.
    Empty-bar rate = #(empty_bars) / #(bars).
    measure_resolution should be ticks_per_bar (4 * ticks_per_beat for 4/4).
    """
    n_measures = length // measure_resolution + 1
    is_empty = np.ones(n_measures, dtype=bool)

    measure_starts = times // measure_resolution
    measure_ends = ends // measure_resolution
    for ms, me in zip(measure_starts, measure_ends):
        is_empty[ms : me + 1] = False

    return float(np.mean(is_empty))


def _get_pianoroll(times: np.ndarray, ends: np.ndarray, pitches: np.ndarray, length: int) -> np.ndarray:
    """Return the binary pianoroll matrix (length × 128).

    Adapted from muspy._get_pianoroll.
    """
    pianoroll = np.zeros((length, 128), dtype=bool)
    for t, e, p in zip(times, ends, pitches):
        pianoroll[t:e, p] = True
    return pianoroll


def _polyphony(times: np.ndarray, ends: np.ndarray, pitches: np.ndarray, length: int) -> float:
    """Return the average number of pitches played concurrently at active time steps.

    Adapted from muspy.polyphony.
    polyphony = #(pitches_when_at_least_one_pitch_is_on) / #(time_steps_where_at_least_one_pitch_is_on).
    """
    pianoroll = _get_pianoroll(times, ends, pitches, length)
    active_mask = pianoroll.sum(axis=1) > 0
    denominator = np.count_nonzero(active_mask)
    if denominator < 1:
        return math.nan
    return pianoroll.sum() / denominator


def _polyphony_rate(times: np.ndarray, ends: np.ndarray, pitches: np.ndarray, length: int, threshold: int = 2) -> float:
    """Return the ratio of time steps where multiple pitches are on.

    Adapted from muspy.polyphony_rate.
    polyphony_rate = #(time_steps_where_>=_threshold_pitches_are_on) / #(time_steps).
    """
    pianoroll = _get_pianoroll(times, ends, pitches, length)
    if len(pianoroll) < 1:
        return math.nan
    return np.count_nonzero(pianoroll.sum(axis=1) >= threshold) / len(pianoroll)


def _get_scale_mask(root: int, mode: str) -> np.ndarray:
    """Return the scale mask for a specific root and mode.

    Adapted from muspy._get_scale.
    """
    if mode == "major":
        c_scale = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    elif mode == "minor":
        c_scale = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=bool)
    else:
        raise ValueError("`mode` must be either 'major' or 'minor'.")
    return np.roll(c_scale, root)


def _scale_consistency(pitches: np.ndarray) -> float:
    """Return the largest pitch-in-scale rate over all major and minor scales.

    Adapted from muspy.scale_consistency.
    scale_consistency = max_{root,mode} pitch_in_scale_rate(root, mode).
    """
    pitch_classes = pitches % 12
    total_notes = len(pitches)
    max_in_scale = 0.0

    for mode in ("major", "minor"):
        for root in range(12):
            scale = _get_scale_mask(root, mode)
            in_scale = np.sum(scale[pitch_classes])
            rate = in_scale / total_notes
            if rate > max_in_scale:
                max_in_scale = rate

    return max_in_scale


def _entropy(prob: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Adapted from muspy._entropy.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = prob > 0
        return -np.sum(prob[mask] * np.log2(prob[mask]))


def _pitch_entropy(pitches: np.ndarray) -> float:
    """Return the entropy of the normalized note pitch histogram.

    Adapted from muspy.pitch_entropy.
    pitch_entropy = -sum_{i=0}^{127} P(pitch=i) * log2(P(pitch=i)).
    """
    counter = np.bincount(pitches, minlength=128).astype(np.float64)
    denominator = counter.sum()
    if denominator < 1:
        return math.nan
    prob = counter / denominator
    return _entropy(prob)


def _pitch_class_entropy(pitches: np.ndarray) -> float:
    """Return the entropy of the normalized note pitch class histogram.

    Adapted from muspy.pitch_class_entropy.
    pitch_class_entropy = -sum_{i=0}^{11} P(pitch_class=i) * log2(P(pitch_class=i)).
    """
    pitch_classes = pitches % 12
    counter = np.bincount(pitch_classes, minlength=12).astype(np.float64)
    denominator = counter.sum()
    if denominator < 1:
        return math.nan
    prob = counter / denominator
    return _entropy(prob)


def _groove_consistency(times: np.ndarray, measure_resolution: int, length: int) -> float:
    """Return the groove consistency (1 - mean Hamming distance of neighbouring measures).

    Adapted from muspy.groove_consistency.
    groove_consistency = 1 - (1 / (T-1)) * sum_{i=1}^{T-1} d(G_i, G_{i+1}).
    Returns NaN if fewer than 2 measures.
    """
    n_measures = length // measure_resolution + 1
    if n_measures < 2:
        return math.nan

    onsets = np.zeros((n_measures, measure_resolution), dtype=bool)
    for t in times:
        m, p = divmod(int(t), measure_resolution)
        if m < n_measures and p < measure_resolution:
            onsets[m, p] = True

    hamming_distance = np.count_nonzero(onsets[:-1] != onsets[1:])
    return 1.0 - hamming_distance / (measure_resolution * (n_measures - 1))


# =========================
# Orchestrator
# =========================


def compute_metrics(midi_path: str) -> Optional[Dict[str, float]]:
    """Load a MIDI file and compute all metrics. Returns None on failure."""
    try:
        with silence_cpp():
            score = Score(midi_path)
    except Exception:
        return None

    if not score.tracks:
        return None

    result = _extract_notes(score)
    if result is None:
        return None

    times, ends, pitches = result
    length = int(ends.max())

    ticks_per_beat = score.ticks_per_quarter
    ticks_per_bar = 4 * ticks_per_beat
    ticks_per_4bar = 16 * ticks_per_beat

    n_pitches = _n_pitches_used(pitches)
    n_pitch_classes = _n_pitch_classes_used(pitches)
    pitch_range_val = _pitch_range(pitches)
    empty_beat_rate_val = _empty_beat_rate(times, ends, ticks_per_beat, length)
    empty_bar_rate_val = _empty_bar_rate(times, ends, ticks_per_bar, length)
    polyphony_val = _polyphony(times, ends, pitches, length)
    polyphony_rate_val = _polyphony_rate(times, ends, pitches, length)
    scale_consistency_val = _scale_consistency(pitches)
    pitch_entropy_val = _pitch_entropy(pitches)
    pitch_class_entropy_val = _pitch_class_entropy(pitches)
    groove_consistency_bar_val = _groove_consistency(times, ticks_per_bar, length)
    groove_consistency_4bar_val = _groove_consistency(times, ticks_per_4bar, length)
    is_identical = is_identical_4bar_loop(times, ends, pitches, ticks_per_bar)

    return {
        "n_pitches_used": n_pitches,
        "n_pitch_classes_used": n_pitch_classes,
        "pitch_range": pitch_range_val,
        "empty_beat_rate": empty_beat_rate_val,
        "empty_bar_rate": empty_bar_rate_val,
        "polyphony": polyphony_val,
        "polyphony_rate": polyphony_rate_val,
        "scale_consistency": scale_consistency_val,
        "pitch_entropy": pitch_entropy_val,
        "pitch_class_entropy": pitch_class_entropy_val,
        "groove_consistency_bar": groove_consistency_bar_val,
        "groove_consistency_4bar": groove_consistency_4bar_val,
        "is_identical_4bar_loop": float(is_identical)
    }


def process_row(row, source_root: str) -> Tuple[int, Optional[Dict]]:
    idx, row_data = row
    midi_path = row_data["output_midi"]

    if not os.path.exists(midi_path):
        alt_path = os.path.join(
            source_root,
            row_data["split"],
            "4-4",
            os.path.basename(midi_path),
        )
        if os.path.exists(alt_path):
            midi_path = alt_path
        else:
            return idx, None

    metrics = compute_metrics(midi_path)
    return idx, metrics


def run_compute_metrics(manifest_path: str, source_root: str, num_workers: int, output_csv: str):
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)

    df_44 = df[
        df["output_midi"].str.contains("4-4", na=False)
        | df["output_midi"].str.contains("4\\\\4", na=False)
    ].copy()
    print(f"Found {len(df_44)} 4/4 loops")

    if df_44.empty:
        print("No 4/4 loops found. Exiting.")
        return

    print(f"Computing metrics with {num_workers} workers...")
    worker_func = partial(process_row, source_root=source_root)
    rows = list(df_44.iterrows())

    results = [None] * len(rows)
    with mp.Pool(processes=num_workers) as pool:
        for i, (_, metrics) in enumerate(tqdm(
            pool.imap_unordered(worker_func, rows),
            total=len(rows),
            desc="Computing metrics",
        )):
            results[i] = metrics

    metrics_list = [r for r in results if r is not None]
    failed = sum(1 for r in results if r is None)
    print(f"Successfully computed metrics for {len(metrics_list)} files ({failed} failed)")

    metrics_df = pd.DataFrame(metrics_list)
    df_result = pd.concat([df_44.reset_index(drop=True), metrics_df], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_result.to_csv(output_csv, index=False)
    print(f"Saved enriched manifest to {output_csv}")

    print("\nMetric statistics:")
    for col in METRIC_COLUMNS:
        if col in df_result.columns:
            s = df_result[col].describe()
            print(f"  {col}:")
            for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                if stat in s.index:
                    print(f"    {stat}: {s[stat]:.4f}")


def apply_filters(
    enriched_csv: str,
    output_root: str,
    source_root: str,
    filters: Dict[str, Tuple[Optional[float], Optional[float]]],
    keep_na: bool = False,
):
    print(f"Loading enriched manifest from {enriched_csv}...")
    df = pd.read_csv(enriched_csv)
    print(f"Loaded {len(df)} rows")

    mask = pd.Series(True, index=df.index)

    for metric, (min_val, max_val) in filters.items():
        if metric not in df.columns:
            print(f"Warning: {metric} not found in manifest, skipping")
            continue

        col = df[metric]
        if min_val is not None:
            mask &= col >= min_val
        if max_val is not None:
            mask &= col <= max_val

    if not keep_na:
        for metric in filters:
            if metric in df.columns:
                mask &= df[metric].notna()

    df_filtered = df[mask].copy()
    print(f"Filtered to {len(df_filtered)} rows ({len(df) - len(df_filtered)} removed)")

    if df_filtered.empty:
        print("No files match the filters. Exiting.")
        return

    splits = ["train", "test", "validation"]
    for split in splits:
        os.makedirs(os.path.join(output_root, split, "4-4"), exist_ok=True)

    print("Copying filtered files...")
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Copying files"):
        midi_path = row["output_midi"]
        split = row["split"]

        if not os.path.exists(midi_path):
            alt_path = os.path.join(source_root, split, "4-4", os.path.basename(midi_path))
            if os.path.exists(alt_path):
                midi_path = alt_path
            else:
                continue

        dest = os.path.join(output_root, split, "4-4", os.path.basename(midi_path))
        shutil.copy2(midi_path, dest)

    filtered_csv = os.path.join(output_root, "filtered_manifest.csv")
    df_filtered.to_csv(filtered_csv, index=False)
    print(f"Saved filtered manifest to {filtered_csv}")

    print("\nFiltered dataset statistics by split:")
    for split in splits:
        n = len(df_filtered[df_filtered["split"] == split])
        print(f"  {split}: {n} files")


def parse_filter_arg(arg_str: str) -> Tuple[str, Optional[float], Optional[float]]:
    parts = arg_str.split(":")
    metric = parts[0]
    min_val = None
    max_val = None
    if len(parts) >= 2 and parts[1]:
        min_val = float(parts[1])
    if len(parts) >= 3 and parts[2]:
        max_val = float(parts[2])
    return metric, min_val, max_val


def main():
    parser = argparse.ArgumentParser(description="Compute metrics and filter GigaMIDI loops")
    parser.add_argument(
        "--mode",
        choices=["compute", "filter", "all"],
        default="all",
        help="Mode: compute metrics, filter existing, or both",
    )
    parser.add_argument(
        "--manifest",
        default=MANIFEST_PATH,
        help="Path to extracted_loops manifest.csv",
    )
    parser.add_argument(
        "--enriched-csv",
        default=None,
        help="Path to save/load enriched manifest with metrics",
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help="Root directory for filtered output",
    )
    parser.add_argument(
        "--source-root",
        default=SOURCE_ROOT,
        help="Root directory of extracted loops",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter as metric:min:max (e.g., --filter empty_beat_rate::0.3 --filter n_pitches_used:10:)",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with NaN metric values during filtering",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show metric statistics and exit (requires enriched CSV)",
    )

    args = parser.parse_args()

    if args.workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = args.workers

    enriched_csv = args.enriched_csv or os.path.join(args.output_root, "enriched_manifest.csv")

    if args.mode in ("compute", "all"):
        run_compute_metrics(args.manifest, args.source_root, num_workers, enriched_csv)

    if args.show_stats:
        if not os.path.exists(enriched_csv):
            print(f"Enriched CSV not found at {enriched_csv}. Run --mode compute first.")
            return
        df = pd.read_csv(enriched_csv)
        print("\nMetric statistics:")
        for col in METRIC_COLUMNS:
            if col in df.columns:
                s = df[col].describe()
                print(f"\n  {col}:")
                for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                    if stat in s.index:
                        print(f"    {stat}: {s[stat]:.4f}")
        return

    if args.mode in ("filter", "all"):
        if not os.path.exists(enriched_csv):
            print(f"Enriched CSV not found at {enriched_csv}. Run --mode compute first.")
            return

        filters = {}
        for f in args.filter:
            metric, min_val, max_val = parse_filter_arg(f)
            filters[metric] = (min_val, max_val)

        if not filters:
            print("No filters specified. Copying all files.")
            print("Use --filter metric:min:max to filter (e.g., --filter empty_beat_rate::0.3)")

        apply_filters(enriched_csv, args.output_root, args.source_root, filters, args.keep_na)


if __name__ == "__main__":
    main()

'''
# Compute metrics (1245574 files)
python -m src.data.gigamidi_loop_filter --mode compute --enriched-csv ./data/GigaMIDI/filtered_loops_v1/enriched_manifest.csv

# Pre-training Dataset (Filtered to 732258 files, train: 585013 files, test: 74145 files, validation: 73100 files)
python -m src.data.gigamidi_loop_filter --mode filter --enriched-csv ./data/GigaMIDI/filtered_loops_v1/enriched_manifest.csv --output-root ./data/GigaMIDI/filtered_loops_v1/pretrain --filter empty_bar_rate:0.0:0.0 --filter empty_beat_rate::0.4 --filter polyphony:1.0:5.5 --filter pitch_range:4.0:48.0 --filter n_pitches_used:4.0:35.0 --filter scale_consistency:0.90: --filter groove_consistency_bar:0.90: --filter is_identical_4bar_loop:0:0

# SFT Monophonic Source (Filtered to 338496 files, train: 270581 files, test: 34392 files, validation: 33523 files)
python -m src.data.gigamidi_loop_filter --mode filter --enriched-csv ./data/GigaMIDI/filtered_loops_v1/enriched_manifest.csv --output-root ./data/GigaMIDI/filtered_loops_v1/sft_mono --filter empty_bar_rate:0.0:0.0 --filter empty_beat_rate::0.4 --filter polyphony:1.0:1.05 --filter pitch_range:4.0:48.0 --filter n_pitches_used:4.0:35.0 --filter scale_consistency:0.95: --filter groove_consistency_bar:0.90: --filter is_identical_4bar_loop:0:0 --filter polyphony_rate:0.0:0.02 

# SFT Polyphonic Source (Filtered to 151174 files, train: 120802 files, test: 15260 files, validation: 15112 files)
python -m src.data.gigamidi_loop_filter --mode filter --enriched-csv ./data/GigaMIDI/filtered_loops_v1/enriched_manifest.csv --output-root ./data/GigaMIDI/filtered_loops_v1/sft_poly --filter empty_bar_rate:0.0:0.0 --filter empty_beat_rate::0.4 --filter polyphony:2.0:5.0 --filter pitch_range:4.0:48.0 --filter n_pitches_used:4.0:35.0 --filter scale_consistency:0.95: --filter groove_consistency_4bar:0.90: --filter is_identical_4bar_loop:0:0 --filter polyphony_rate:0.50:1.0
'''