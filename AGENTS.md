# midi-gpt — Agent Guide

## Project
GPT-style MIDI generator: takes a 4-bar prompt, generates a 4-bar continuation (8 bars total). Output MIDI has 2 tracks: "Generated" and "Prompt".
Tokenization mode: **REMI** (single-token).
Data lives in `data/GigaMIDI/filtered_loops_v1/{pretrain|sft_mono|sft_poly}/{train|test|validation}/4-4/*.mid`.

## Pipeline & Commands
Run from repo root. Python 3.13 confirmed. Install deps first:
```
pip install -r requirements.txt
```
`requirements.txt` references `requirements_torch.txt` (`-r`), which includes `--extra-index-url https://download.pytorch.org/whl/cu128` for CUDA 12.8 wheels, `torch==2.11.0`, and `triton-windows==3.6.0.post26` (use `triton` on Linux).

1. **Preprocess MIDIs** (already done — **do not re-run**, takes hours on 1.3M files)
   - `src/data/gigamidi_loop_extractor.py` → extracts validated 8-bar loops
   - `src/data/gigamidi_loop_filter.py` → computes 12 quality metrics, filters by threshold, copies to `filtered_loops_v1/`

2. **Tokenize** preprocessed MIDIs → .npz files (preserves directory structure):
   ```
   python -m src.data.tokenize --mode remi --gigamidi [--subsets pretrain,sft_mono --max-tokens 1536]
   ```
   `--subsets` filters which of `pretrain,sft_mono,sft_poly` to process (default: all). `--max-tokens` skips sequences exceeding the limit (default: 1536).
   Output: `data/tokens/{split}-{mode}/{subset}/4-4/{stem}.npz`. Each file = one 8-bar loop.
   Tokenizer auto-cached to `data/tokenizers/{mode}_tokenizer.pkl`. Stats (`token_stats.json`) written per leaf folder.
   Legacy (POP909 flat): `python -m src.data.tokenize --mode remi`

3. **Pretrain** on the pretrain split:
   ```
   python -m src.training.pretrain [--split pretrain]
   ```
   Config hardcoded in code: batch_size=24, effective_batch_size=48 (gradient accumulation ×2), lr=1e-4, epochs=5, mode=remi. W&B project `midi-gpt` (required — `wandb.init()` called unconditionally).
   Uses `torch.compile` (full model for dense, just encoder for NJT) + AdamW `fused=True`.
   LR schedule: linear warmup (10% of steps) → cosine decay to 10% of lr.
   Checkpoints saved to `src/checkpoints/` every 10 epochs + `best.pt` for lowest loss.

4. **Generate MIDI** from checkpoint (picks a random test file as prompt):
   ```
   python -m src.generation.generate_main --checkpoint src/checkpoints/best.pt --top-k 1 --n-samples 1 [--split pretrain]
   ```
   Output: `data/test/generated_{stem}_{mode}_sample{i}.mid`. Auto-generates `len(prompt)` tokens.

## Architecture
- `src/data/gigamidi_loop_extractor.py` — Reads filtered metadata CSV, validates 8-bar loops (32-beat ±0.5, single TS with denom=4, non-drum, ≥10 notes), extracts to `data/GigaMIDI/extracted_loops_v1/`. Parallel via `mp.Pool`.
- `src/data/gigamidi_loop_filter.py` — 12 quality metrics (adapted from muspy) on extracted 4/4 loops. Two-phase: `compute` writes enriched CSV, `filter` applies thresholds and copies to `filtered_loops_v1/`.
- `src/data/tokenizer_utils.py` — Shared configs + cached `get_tokenizer()` (pickled to `data/tokenizers/`). Also `compute_token_stats()` with `total` field.
- `src/data/tokenize.py` — REMI tokenizer. `tokenize_recursive()` walks the GigaMIDI tree. Supports `--subsets` and `--max-tokens`. `rechunk_tokens()` for 8-bar slicing of legacy flat tokens.
- `src/data/detokenize.py` — Converts tokens back to MIDI via miditok. Optional `prompt_tokens` builds a 2-track Score.
- `src/data/dataset.py` — Single `MidiDataset`: returns raw unpadded (L_i,) tensors (truncated if > `max_seq_len`). Two collate functions: `collate_pad_to_longest` (pads each batch to its longest sequence, returns dict with pre-split `inputs`/`targets`) for dense, `nested_collate` (identity, returns list) for NJT. Pad token = 0 everywhere.
- `src/models/remi_transformer.py` — `TransformerEncoder`-based (causal via `is_causal=True`). d_model=512, 8 layers, 8 heads, FF=2048.
- `src/models/nested_remi_transformer.py` — NJT variant of REMI. Training path: accepts list of (L_i,) tokens, embeds each + adds PE eagerly, creates contiguous NJT via `as_nested_tensor`, then encoder. Generation path: dense tensor (same as remi). Encoder is `torch.compile`'d separately; PE step runs in eager to avoid PyTorch 2.11 compile bugs with NJT creation. SDPA backward on contiguous NJT works in PyTorch 2.11.
- `src/generation/generate.py` — Autoregressive top-k sampling. No KV-cache.
- `src/training/trainer.py` — NJT path (nested=True): batch is list of (L_i,) tensors, slices per-item, creates flat `target = torch.cat([t[1:]...])`, passes `input_list` to model. Dense path: receives dict `{'inputs', 'targets'}` from `collate_pad_to_longest`, uses `inputs` directly, `src_key_padding_mask=(inputs == 0)`.
- `src/training/loss.py` — `compute_remi_loss`: NJT-aware — uses `.values()` + flat target with `ignore_index=0` when logits are nested.
- `src/training/pretrain.py` — Training entrypoint.
- `src/generation/generate_main.py` — Generation entrypoint.
- `src/utils/checkpoint.py` — `save_checkpoint()` saves every 10 epochs + updates `best.pt` on lower loss. `load_checkpoint()` uses `weights_only=False`.
- `src/utils/logging.py` — Thin wrappers around `wandb.init` / `wandb.log`.
- `src/data/utils.py` — `silence_cpp()` context manager to suppress symusic C-level stdout/stderr.

## Key Conventions & Gotchas
- **Pad token = 0** everywhere (loss `ignore_index=0`, collate `pad_id=0`).
- **Vocab**: `tokenizer.vocab_size`, bar token `Bar_None`.
- **`weights_only=False`** in `load_checkpoint()` — allows pickle, intentional.
- **Pre-LN architecture**: Both dense and nested models use Pre-LN (`norm_first=True`) with a final `nn.LayerNorm` before the LM head.
- **Key padding mask**: Dense model uses `src_key_padding_mask=(input_ids == 0)` to skip attention over padding. Both `mask` (causal) and `src_key_padding_mask` are bool tensors.
- **Weight init alignment**: `NestedMultiHeadAttention._reset_parameters()` matches `nn.MultiheadAttention._reset_parameters()`: Xavier Uniform for in-projection weights, zero for biases. `out_proj.weight` keeps default Kaiming Uniform.
- **bfloat16 AMP** in trainer — requires compatible GPU (CUDA). Falls back otherwise.
- **AdamW `fused=True`** in optimizer — requires CUDA.
- **`silence_cpp()`** wraps `Score()` calls in tokenization to suppress symusic C-level output.
- **NJT contiguous requirement**: `torch.nested.narrow` creates non-contiguous NJT → fails with `F.linear`. Always use `torch.nested.as_nested_tensor(list, layout=torch.jagged)` for contiguous NJT that supports linear/SDPA backward.
- **`torch.compile` + NJT**: creating NJT inside a compiled function crashes PyTorch 2.11 (InternalTorchDynamoError with shape env guards). Workaround: embed/PE/NJT creation runs in eager, only `model.encoder` is compiled separately.
- **NJT CUDA non-determinism**: NJT SDPA forward pass on CUDA has inherent non-determinism (~0.04 diff between identical calls). Checkpoint save/load round-trip diff is within this noise range — not a save/load bug.
- **No `__init__.py` files in `src/`** — Python 3.13 implicit namespace packages work fine.
- **No test/lint/typecheck infrastructure** — no CI, no pytest, no mypy, no ruff. Only validation is running scripts directly.
- `.gitignore` excludes `/data/`, `wandb/`, and `src/checkpoints/*.pt` (directory tracked via `.gitkeep`).
- `configs/` directory does not exist; all training config is hardcoded in `pretrain.py`.

## Dependencies (pinned)
`pip install -r requirements.txt` pulls from both files:

**`requirements.txt`**: miditok==3.0.6.post1, numpy==2.4.4, symusic==0.6.0, tqdm==4.67.1, wandb==0.26.1, pandas==2.2.3

**`requirements_torch.txt`** (via `-r`): torch==2.11.0, triton-windows==3.6.0.post26 (use `triton` on Linux). `--extra-index-url https://download.pytorch.org/whl/cu128` for CUDA 12.8 wheels.
