# midi-gpt — Agent Guide

## Project
GPT-style MIDI generator: takes a 4-bar prompt, generates a 4-bar continuation (8 bars total, single track).
Two tokenization modes: **REMI** (single-token) and **Octuple** (multi-field per timestep).

## Pipeline & Commands
Run from repo root. Python 3.13 confirmed.

1. **Preprocess MIDIs** (POP909 → piano-only, 4/4, aligned, 64-bar truncated):
   `python -m src.data.preprocess`
   Output: `aligned_piano_tracks/`. Reads POP909 from `data/raw/` (not tracked; see `.gitignore`).
   NOTE: Will require changes — tracks are truncated to 64 bars, not fixed 8-bar songs.

2. **Tokenize** preprocessed MIDIs → .npz files:
   `python -m src.data.tokenize --mode remi` (or `--mode octuple`)
   Output: `data/tokens/{remi|octuple}/*.npz`. Tokenizer uses symusic backend.

3. **Slice into 8-bar chunks** (for proper sequential training):
   `python -c "from src.data.tokenize import rechunk_tokens; rechunk_tokens('remi', 'data/tokens/remi', 'data/tokens/remi_8bar')"`
   Output: `data/tokens/{remi|octuple}_8bar/*.npz`. Each file = exactly 8 bars.

4. **Train** (hardcoded config in `src/main.py`):
   `python -m src.main`
   Default: mode=remi, batch=64, seq_len=512 (dynamic, capped), lr=1e-4, 3 epochs. Logs to W&B project `midi-gpt`.
   Checkpoints saved to `src/checkpoints/` every epoch + `best.pt` for lowest loss.

5. **Generate MIDI** from checkpoint:
   `python -m src.generate_main --checkpoint src/checkpoints/best.pt --max-tokens 128 --top-k 40`
   Output: `data/test/generated_*.mid`. Uses 4-bar priming from random training sample.

## Architecture
- `src/data/tokenize.py` — REMI / Octuple tokenizer creation (miditok). Config hardcodes: no chords/rests/tempos/time-sigs, 16 ticks/beat, 32 velocities, no programs. Also has `rechunk_tokens()` for 8-bar slicing.
- `src/data/dataset.py` — `MidiDataset`: **one .npz file = one 8-bar sample**. No random cropping — uses fixed 8-bar chunks. Sequences < seq_len are zero-padded. Long sequences capped at `max_seq_len` (default 512). `seq_len` is computed dynamically from data.
- `src/data/detokenize.py` — Converts generated tokens back to MIDI files using miditok.
- `src/models/` — Both are `TransformerEncoder`-based (causal mask applied manually). Octuple sums per-field embeddings; REMI uses a single embedding. Positional encoding is standard sinusoidal.
- `src/generation/generate.py` — Autoregressive sampling with **top-k filtering** (default k=40). No KV-cache yet.
- `src/utils/checkpoint.py` — `save_checkpoint()` saves every epoch + updates `best.pt` when loss improves. `load_checkpoint()` for inference.
- `src/generate_main.py` — Full generation pipeline: load checkpoint → 4-bar prime → generate → detokenize → save MIDI.

## Key Conventions & Gotchas
- **Pad token = 0** everywhere (loss `ignore_index=0`, dataset `value=0`).
- `src/main.py` config is hardcoded — no `configs/` directory exists despite `project_structure.txt` referencing it.
- `MidiDataset` scans all .npz files to compute `seq_len` dynamically — this takes a few seconds on startup.
- Octuple mode: `tokenizer.vocab` is a **list** of vocab sizes per field. REMI: single `tokenizer.vocab_size`.
- `.gitignore` excludes `/data/` and `wandb/` — dataset and logs are never committed.
- Checkpoints (.pt files) are excluded from git but the directory is tracked via `.gitkeep`.

## Pending Work
- Train, test, eval split
- Implement eval metrics and log to W&B:
  - Pitch histogram distance between 4-bar prompt and 4-bar continuation, how similar they are (muspy)
  - Rhythm distance — does the continuation hold the same rhythm as prompt (Inter-Onset Interval: time between starting points of each 2 neighboring notes, compare distributions)
  - See [MGEval](https://github.com/RichardYang40148/mgeval) for implementations
  - Additionally: Information Rate (IR) based on Variable Markov Oracle (VMO)
- Add more data (GigaMIDI dataset)
- Take loop detector from [gigamidi-dataset](https://github.com/metacreation-lab/gigamidi-dataset) to filter new dataset into 8-bar loops
- Enforce bar token as first generated token (or keep 5th bar token from prompt as the first token in continuation)
- Optionally: add top-p sampling and KV-cache
- Verify `pad_token_id` is indeed 0

## Dependencies
miditok==3.0.6.post1, numpy==2.4.4, symusic==0.6.0, torch==2.11.0, tqdm==4.67.1, wandb==0.26.1
No test framework, no linter, no formatter, no type checker configured.
