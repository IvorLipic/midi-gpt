# midi-gpt — Agent Guide

## Project
GPT-style MIDI generator: takes a 4-bar prompt, generates a 4-bar continuation (8 bars total). Output MIDI has 2 tracks: "Generated" (continuation) and "Prompt" (prompt only).
Two tokenization modes: **REMI** (single-token) and **Octuple** (multi-field per timestep).

## Pipeline & Commands
Run from repo root. Python 3.13 confirmed.

1. **Preprocess MIDIs**
   - `src/data/GigaMIDI_Metadata_Filtering.ipynb` — filters GigaMIDI metadata CSV to 8-bar loops, excludes drum-only/classical/jazz → `data/GigaMIDI/GigaMIDI-metadata-filtered-v1.csv`
   - `src/data/gigamidi_loop_extractor.py` — extracts validated 8-bar loops from filtered metadata (parallel processing). Command: `python -m src.data.gigamidi_loop_extractor`. **Do NOT run** — long execution and a lot of data.
   - `src/data/preprocess_pop909.py` — Legacy (different pipeline): POP909 dataset → piano-only, 4/4 only, aligned to tick 0, truncated to 64 bars. Tokens generated externally, currently in `data/tokens`, not runnable here.

2. **Tokenize** preprocessed MIDIs → .npz files:
   `python -m src.data.tokenize --mode remi` (or `--mode octuple`)
   Output: `data/tokens/{remi|octuple}/*.npz`. Tokenizer uses symusic `Score` backend.

3. **Slice into 8-bar chunks** (for proper sequential training):
   `python -m src.data.tokenize --mode remi --slice-8bar`
   (or specify custom paths: `--input data/tokens/remi --output data/tokens/remi_8bar`)
   Output: `data/tokens/{remi|octuple}_8bar/*.npz`. Each file = exactly 8 bars.

4. **Train** (hardcoded config in `src/main.py`):
   `python -m src.main`
   Default: batch_size=32, lr=5e-4, epochs=100, mode=remi. Logs to W&B project `midi-gpt` (W&B required).
   Checkpoints saved to `src/checkpoints/` every **10 epochs** + `best.pt` for lowest loss.

5. **Generate MIDI** from checkpoint:
   `python -m src.generate_main --checkpoint src/checkpoints/best.pt --top-k 1 --n-samples 1`
   Output: `data/test/generated_{stem}_{mode}_sample{i}.mid`. Auto-generates `len(prompt)` tokens. `n_samples` generates multiple continuations from the same prompt file.

## Architecture
- `src/data/preprocess_pop909.py` — POP909 dataset preprocessor (see Pipeline Step 1). Tokens generated externally, not runnable here.
- `src/data/gigamidi_loop_extractor.py` — GigaMIDI preprocessor (see Pipeline Step 1). Reads filtered metadata CSV, validates 8-bar loops (32-beat ±0.5 duration, single TS with denominator=4, non-drum tracks, ≥10 notes), extracts to `data/GigaMIDI/extracted_loops_v1/{split}/{ts_type}/`. Runs in parallel via `mp.Pool`. Produces `manifest.csv`.
- `src/data/tokenize.py` — REMI / Octuple tokenizer creation (miditok). Config hardcodes: no chords/rests/tempos/time-sigs, 16 ticks/beat, 32 velocities, no programs. Also has `rechunk_tokens()` for 8-bar slicing (CLI: `--slice-8bar`).
- `src/data/dataset.py` — `MidiDataset`: one .npz file = one 8-bar sample. No random cropping. Pad token = 0. `seq_len` computed dynamically from data. Long sequences capped at `max_seq_len`.
- `src/data/detokenize.py` — Converts generated tokens back to MIDI files using miditok. **Tokenizer config duplicated from tokenize.py — keep in sync.** Optional `prompt_tokens` param builds a 2-track Score (Generated + Prompt).
- `src/models/` — Both are `TransformerEncoder`-based (causal mask applied manually as `triu(-inf, diag=1)`). Octuple sums per-field embeddings; REMI uses a single embedding. Sinusoidal positional encoding.
- `src/generation/generate.py` — Autoregressive sampling with **top-k filtering**. No KV-cache.
- `src/utils/checkpoint.py` — `save_checkpoint()` saves every 10 epochs + updates `best.pt`. `load_checkpoint()` uses `weights_only=False`.
- `src/generate_main.py` — Full generation pipeline: load checkpoint → 4-bar prime → generate → detokenize → save MIDI. Output has 2 tracks (Generated + Prompt). `n_samples` produces multiple continuations from the same prompt file. `max_new_tokens` is auto-set to `len(prompt)`.

## Key Conventions & Gotchas
- **Pad token = 0** everywhere (loss `ignore_index=0`, dataset pad `value=0`).
- `src/main.py` config is hardcoded — `configs/` dir exists but is empty (unused).
- Tokenizer config (`beat_res`, `num_velocities`, etc.) is **duplicated** in both `tokenize.py` and `detokenize.py` — edits must stay in sync.
- `MidiDataset` scans all .npz files to compute `seq_len` dynamically — takes a few seconds on startup.
- Octuple mode: `tokenizer.vocab` is a **list** of vocab sizes per field. REMI: single `tokenizer.vocab_size`.
- Octuple bar detection for prompt: `tokens[:, 0] < 4` (not `tokenizer.vocab["Bar_None"]`). REMI uses `tokenizer.vocab["Bar_None"]`.
- symusic `Score` object is the required backend for miditok.
- W&B required: `wandb.init()` called unconditionally — train will fail if not logged in.
- `.gitignore` excludes `/data/` and `wandb/` — dataset and logs never committed.
- Checkpoints (.pt files) excluded from git; directory tracked via `.gitkeep`.

## Pending Work
- Train/test/eval split — integrate with GigaMIDI's existing train/test/validation split (`gigamidi_loop_extractor.py`)
- Eval metrics logged to W&B: pitch histogram distance (muspy), IOI distribution (rhythm distance), IR/VMO
  - See [MGEval](https://github.com/RichardYang40148/mgeval) for implementations
- Enforce bar token as first generated token (or keep 5th bar token from prompt as the first token in continuation)
- Optionally: top-p sampling and KV-cache

## Dependencies
miditok==3.0.6.post1, numpy==2.4.4, symusic==0.6.0, torch==2.11.0, tqdm==4.67.1, wandb==0.26.1, pandas==2.2.3
