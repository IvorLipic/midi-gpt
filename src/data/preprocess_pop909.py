from pathlib import Path
from tqdm import tqdm
from symusic import Score, Note, Track
import symusic as sm

def calculate_bars_with_time_signatures(score):
    """
    Calculate number of bars considering time signature changes
    """
    if not score.time_signatures:
        # Default to 4/4 if no time signature specified
        ticks_per_quarter = score.ticks_per_quarter
        total_ticks = score.end()
        bars = total_ticks / (4 * ticks_per_quarter)
        return bars
    
    # Sort time signatures by tick
    time_sigs = sorted(score.time_signatures, key=lambda x: x.time)
    
    total_bars = 0
    current_tick = 0
    
    for i, ts in enumerate(time_sigs):
        start_tick = ts.time
        # Get end tick (next time signature change or end of score)
        end_tick = time_sigs[i + 1].time if i + 1 < len(time_sigs) else score.end()
        
        # Calculate ticks in this time signature section
        section_ticks = end_tick - start_tick
        
        # Convert to bars
        # ts.numerator = beats per bar, ts.denominator = note value (4 = quarter note)
        # ticks_per_bar = (ticks_per_quarter * 4 / ts.denominator) * ts.numerator
        ticks_per_bar = (score.ticks_per_quarter * 4 / ts.denominator) * ts.numerator
        
        bars_in_section = section_ticks / ticks_per_bar
        total_bars += bars_in_section
        
        current_tick = end_tick
    
    return total_bars

def align_piano_track_to_start(score):
    """
    Extract only the piano track named 'PIANO' and align it to tick 0
    """
    piano_tracks = [track for track in score.tracks if track.name.upper() == "PIANO"]
    
    if not piano_tracks:
        return None
    
    piano_track = piano_tracks[0]
    
    if not piano_track.notes:
        return None
    
    offset = min(note.time for note in piano_track.notes)
    
    new_score = Score(score.tpq)
    new_score.tempos = score.tempos
    new_score.time_signatures = score.time_signatures

    if offset == 0:
        new_score.tracks = [piano_track]
        return new_score
    
    new_score.tracks.append(piano_track.shift_time(-offset))
    
    return new_score

def tick_at_bar(score, target_bar):
    """
    Return the tick position corresponding to the start of `target_bar`
    considering time signature changes.
    """
    ticks_per_quarter = score.ticks_per_quarter

    if not score.time_signatures:
        # Default 4/4
        ticks_per_bar = 4 * ticks_per_quarter
        return int(target_bar * ticks_per_bar)

    time_sigs = sorted(score.time_signatures, key=lambda x: x.time)

    current_bar = 0
    current_tick = 0

    for i, ts in enumerate(time_sigs):
        start_tick = ts.time
        end_tick = time_sigs[i + 1].time if i + 1 < len(time_sigs) else score.end()

        ticks_per_bar = (ticks_per_quarter * 4 / ts.denominator) * ts.numerator
        section_ticks = end_tick - start_tick
        bars_in_section = section_ticks / ticks_per_bar

        if current_bar + bars_in_section >= target_bar:
            remaining_bars = target_bar - current_bar
            return int(start_tick + remaining_bars * ticks_per_bar)

        current_bar += bars_in_section
        current_tick = end_tick

    return score.end()

def truncate_score_to_bars(score, max_bars=64):
    """
    Truncate all tracks in score to first `max_bars` bars.
    """
    cutoff_tick = tick_at_bar(score, max_bars)

    new_score = Score(score.tpq)

    # ---- tempos ----
    new_score.tempos = [
        t for t in score.tempos if t.time < cutoff_tick
    ]

    # ---- force 4/4 time signature ----
    new_score.time_signatures = [
        sm.TimeSignature(time=0, numerator=4, denominator=4)
    ]

    # ---- tracks ----
    for track in score.tracks:
        new_track = Track(
            name=track.name,
            program=track.program,
            is_drum=track.is_drum
        )

        for note in track.notes:
            if note.time >= cutoff_tick:
                continue

            end_time = min(note.time + note.duration, cutoff_tick)
            if end_time > note.time:
                new_track.notes.append(
                    Note(
                        time=note.time,
                        duration=end_time - note.time,
                        pitch=note.pitch,
                        velocity=note.velocity
                    )
                )

        if new_track.notes:
            new_score.tracks.append(new_track)

    return new_score

def is_4_4_compatible(score):
    """
    Accept:
      - no time signatures (assume 4/4)
      - 1/4, 2/4, 4/4 ONLY
    Reject:
      - any other numerator or denominator
      - time signature changes to invalid meters
    """
    if not score.time_signatures:
        return True

    for ts in score.time_signatures:
        if ts.denominator != 4:
            return False
        if ts.numerator not in (1, 2, 4):
            return False

    return True


def batch_process_midi_files(midi_folder, max_bars=64):
    """
    Reprocess MIDI files into clean piano-only, aligned, truncated (64-bar) MIDIs.

    Output:
      - folder: aligned_piano_tracks/
      - files: <songFolder>_<originalFilename>.mid
    """

    midi_folder = Path(midi_folder)
    output_folder = Path("aligned_piano_tracks")

    # ---- HARD RESET OUTPUT FOLDER ----
    if output_folder.exists():
        for f in output_folder.glob("*.mid"):
            f.unlink()
    else:
        output_folder.mkdir()

    # ---- FIND SONG FOLDERS ----
    song_folders = []
    for i in range(1, 910):
        if i == 555: 
            continue
        folder = midi_folder / f"{i:03d}"
        if folder.is_dir():
            song_folders.append(folder)

    print(f"Found {len(song_folders)} song folders")

    # ---- COLLECT MIDI FILES ----
    midi_files = []
    for folder in song_folders:
        for midi_file in list(folder.glob("*.mid")) + list(folder.glob("*.midi")):
            if "versions" not in str(midi_file.parent).lower():
                midi_files.append(midi_file)

    print(f"Found {len(midi_files)} MIDI files")

    processed = 0
    skipped = 0

    with tqdm(midi_files, desc="Reprocessing MIDIs") as pbar:
        for midi_file in pbar:
            try:
                score = Score(midi_file)

                # ---- TIME SIGNATURE FILTER ----
                if not is_4_4_compatible(score):
                    skipped += 1
                    continue

                # ---- ALIGN PIANO TRACK ----
                aligned = align_piano_track_to_start(score)
                if aligned is None:
                    skipped += 1
                    continue

                # ---- TRUNCATE TO FIRST N BARS (FROM TICK 0) ----
                processed_score = truncate_score_to_bars(
                    aligned,
                    max_bars=max_bars
                )

                if not processed_score.tracks:
                    skipped += 1
                    continue

                # ---- SAVE PROCESSED SCORE ----
                folder_name = midi_file.parent.name
                output_path = output_folder / f"{folder_name}_{midi_file.name}"
                processed_score.dump_midi(output_path)

                processed += 1
                pbar.set_postfix({
                    "saved": processed,
                    "skipped": skipped
                })

            except Exception as e:
                print(f"\nError processing {midi_file}: {e}")
                skipped += 1

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Saved MIDIs:   {processed}")
    print(f"Skipped MIDIs: {skipped}")

def preprocess_dataset(input_dir, output_dir, max_bars=64):
    batch_process_midi_files(input_dir, max_bars=max_bars)