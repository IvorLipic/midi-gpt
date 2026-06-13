import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from symusic import Score


def score_to_pianoroll(score, max_time_beats=32, pitch_range=(21, 109), figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4A90D9", "#C0C0C0"]
    labels = ["Generated", "Prompt"]

    for i, track in enumerate(score.tracks):
        color = colors[i % len(colors)]
        for note in track.notes:
            start = note.time / score.tpq
            dur = note.duration / score.tpq
            ax.barh(note.pitch, dur, left=start, height=0.8,
                    color=color, edgecolor="none", alpha=0.85)

    for i in range(min(len(score.tracks), 2)):
        ax.barh(0, 0, color=colors[i], label=labels[i])

    ax.set_xlim(0, max_time_beats)

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    used_pitches = {note.pitch for track in score.tracks for note in track.notes}
    if used_pitches:
        padding = 4
        y_min = min(used_pitches) - padding
        y_max = max(used_pitches) + padding
    else:
        y_min, y_max = pitch_range
    ax.set_ylim(y_min - 1, y_max + 1)

    full_range = list(range(y_min, y_max + 1))
    ax.set_yticks(full_range)
    ax.set_yticklabels(
        [f"{NOTES[p % 12]}{p // 12 - 1}" if p in used_pitches else "" for p in full_range],
        fontsize=9,
    )
    ax.set_ylabel("Pitch")

    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img
