# midi-gpt — Agent Guide

## Project
GPT-style MIDI generator: takes a 4-bar prompt, generates a 4-bar continuation (8 bars total). Output MIDI has 2 tracks: "Generated" and "Prompt".
Two tokenization modes: **REMI** (single-token) and **Octuple** (multi-field per timestep).
Data lives in `data/GigaMIDI/filtered_loops_v1/{pretrain|sft_mono|sft_poly}/{train|test|validation}/4-4/*.mid`.

## Pipeline & Commands
Run from repo root. Python 3.13 confirmed.

1. **Preprocess MIDIs** (already done — **do not re-run**, takes hours on 1.3M files)
   - `src/data/gigamidi_loop_extractor.py` → extracts validated 8-bar loops
   - `src/data/gigamidi_loop_filter.py` → computes 12 quality metrics, filters by threshold, copies to `filtered_loops_v1/`

2. **Tokenize** preprocessed MIDIs → .npz files (preserves directory structure):
   ```
   python -m src.data.tokenize --mode remi --gigamidi
   python -m src.data.tokenize --mode octuple --gigamidi
   ```
   Output: `data/tokens/{split}-{mode}/{subset}/4-4/{stem}.npz`. Each file = exactly one 8-bar loop.
   Tokenizer auto-cached to `data/tokenizers/{mode}_tokenizer.pkl`. Stats (`token_stats.json`) written per leaf folder.
   Legacy (POP909 flat): `python -m src.data.tokenize --mode remi`

3. **Pretrain** on the pretrain split:
   ```
   python -m src.training.pretrain [--split pretrain]
   ```
   Config hardcoded in code: batch_size=32, lr=5e-4, epochs=100, mode=remi. Logs to W&B project `midi-gpt` (W&B required).
   Checkpoints saved to `src/checkpoints/` every 10 epochs + `best.pt` for lowest loss.
   Token path: `data/tokens/{split}-{mode}/train/4-4/`.

4. **Generate MIDI** from checkpoint (picks a random test file as prompt):
   ```
   python -m src.generate_main --checkpoint src/checkpoints/best.pt --top-k 1 --n-samples 1 [--split pretrain]
   ```
   Output: `data/test/generated_{stem}_{mode}_sample{i}.mid`. Auto-generates `len(prompt)` tokens.

## Architecture
- `src/data/gigamidi_loop_extractor.py` — Reads filtered metadata CSV, validates 8-bar loops (32-beat ±0.5, single TS with denom=4, non-drum, ≥10 notes), extracts to `data/GigaMIDI/extracted_loops_v1/`. Parallel via `mp.Pool`.
- `src/data/gigamidi_loop_filter.py` — 12 quality metrics (adapted from muspy) on extracted 4/4 loops. Two-phase: `compute` writes enriched CSV, `filter` applies thresholds and copies to `filtered_loops_v1/`.
- `src/data/tokenizer_utils.py` — Shared configs + cached `get_tokenizer()` (pickled to `data/tokenizers/`). Also `compute_token_stats()`.
- `src/data/tokenize.py` — REMI/Octuple tokenizer. `tokenize_recursive()` walks the GigaMIDI tree. `rechunk_tokens()` for 8-bar slicing of legacy flat tokens.
- `src/data/detokenize.py` — Converts tokens back to MIDI via miditok. Optional `prompt_tokens` builds a 2-track Score.
- `src/data/dataset.py` — `MidiDataset`: one .npz = one sample. Pad token = 0. `seq_len` computed by scanning all files on startup.
- `src/models/remi_transformer.py` — `TransformerEncoder`-based (causal via `is_causal=True`). d_model=512, 8 layers, 8 heads, FF=2048.
- `src/models/octuple_transformer.py` — Same pattern but sums per-field embeddings. d_model=256, 4 layers, 4 heads, FF=512.
- `src/generation/generate.py` — Autoregressive top-k sampling. No KV-cache.
- `src/utils/checkpoint.py` — `save_checkpoint()` saves every 10 epochs + updates `best.pt` on lower loss. `load_checkpoint()` uses `weights_only=False`.
- `src/training/trainer.py` — Slices `batch[:, :-1]` / `batch[:, 1:]` for next-token prediction. Uses bfloat16 AMP.

## Key Conventions & Gotchas
- **Pad token = 0** everywhere (loss `ignore_index=0`, dataset pad `value=0`).
- Octuple mode: `tokenizer.vocab` is a **list** of vocab sizes per field. REMI: single `tokenizer.vocab_size`.
- Octuple bar detection for prompt: `tokens[:, 0] < 4`. REMI uses `tokenizer.vocab["Bar_None"]`.
- symusic `Score` object is the required backend for miditok.
- W&B required: `wandb.init()` called unconditionally — train will fail if not logged in.
- bfloat16 AMP in trainer — requires compatible GPU (CUDA). Falls back otherwise.
- `.gitignore` excludes `/data/`, `wandb/`, and `src/checkpoints/*.pt` (directory tracked via `.gitkeep`).
- `configs/` directory does not exist; all training config is hardcoded in `pretrain.py`.

## Dependencies
miditok==3.0.6.post1, numpy==2.4.4, symusic==0.6.0, torch==2.11.0, tqdm==4.67.1, wandb==0.26.1, pandas==2.2.3
