# midi-gpt

GPT-style MIDI continuation model. Takes a 4-bar MIDI prompt and generates a 4-bar continuation (8 bars total). Uses REMI tokenization and a decoder-only GPT-2 transformer.

Created as a Master's thesis project at FER (Faculty of Electrical Engineering and Computing, University of Zagreb).

Output MIDI has 2 tracks ("Generated" + "Prompt") by default, or 1 track ("Generated") with `--single-track`.

## Requirements

- Python 3.12 (recommended; `triton-windows` is not available on 3.13)
- `pip install -r requirements.txt`

## Dataset Setup

Download from [huggingface.co/datasets/Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI). You need **two things**:

1. The raw MIDI folder `e.g. Final_GigaMIDI_V1.1_Final/`
2. The metadata `e.g. Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv`

Place them under `data/GigaMIDI/` so the structure is:

```
data/GigaMIDI/
  Final_GigaMIDI_V1.1_Final/
    training-V1.1-80%/{all-instruments-with-drums,no-drums}/{0..54}/*.mid
    test-V1.1-10%/{all-instruments-with-drums,no-drums}/{0..6}/*.mid
    validation-V1.1-10%/{all-instruments-with-drums,no-drums}/{0..6}/*.mid
  Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv
```

The name `Final_GigaMIDI_V1.1_Final` must match what the CSV's `file_path` column references (the extractor joins this path with `data/GigaMIDI/`).

## Pipeline

### a) Generate filtered metadata

Run `src/data/GigaMIDI_Analysis_and_Filtering.ipynb`. It reads `data/GigaMIDI/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv` and outputs `data/GigaMIDI/GigaMIDI-metadata-filtered-v1.csv`.

### b) Extract 8-bar loops

```bash
python -m src.data.gigamidi_loop_extractor
```

Parallel extraction (can take hours on 1.3M files). Validates loops (32 beats ±0.5, single 4/4 time signature, non-drum tracks, ≥10 notes) and outputs to `data/GigaMIDI/extracted_loops_v1/{train,test,validation}/{4-4,2-4,unknown}/*.mid`.

### c) Filter by quality metrics

Computes 12 quality metrics (adapted from muspy) then applies thresholds:

```bash
# Compute metrics on all 4/4 extracted loops
python -m src.data.gigamidi_loop_filter --mode compute --enriched-csv ./data/GigaMIDI/filtered_loops_v1/enriched_manifest.csv

# Filter with thresholds (produces pretrain subset)
python -m src.data.gigamidi_loop_filter --mode filter \
    --enriched-csv ./data/GigaMIDI/filtered_loops_v1/enriched_manifest.csv \
    --output-root ./data/GigaMIDI/filtered_loops_v1/pretrain \
    --filter empty_bar_rate:0.0:0.0 \
    --filter empty_beat_rate::0.4 \
    --filter polyphony:1.0:5.5 \
    --filter pitch_range:4.0:48.0 \
    --filter n_pitches_used:4.0:35.0 \
    --filter scale_consistency:0.90: \
    --filter groove_consistency_bar:0.90: \
    --filter is_identical_4bar_loop:0:0
```

Output: `data/GigaMIDI/filtered_loops_v1/pretrain/{train,test,validation}/4-4/*.mid`

### d) Tokenize

```bash
python -m src.data.tokenize --mode remi --gigamidi --subsets pretrain --max-tokens 1536
```

Produces `data/tokens/pretrain-remi/{train,test,validation}/4-4/*.npz` (one `.npz` file per 8-bar loop). The tokenizer is auto-cached to `data/tokenizers/remi_tokenizer.pkl`.

### e) Pretrain

```bash
python -m src.training.pretrain --split pretrain
```

- Model: `hf_remi` (GPT2LMHeadModel wrapper), d_model=512, 8 layers, 8 heads, FF=2048
- LR 1e-4, linear warmup (10% of steps) → cosine decay to 10% of LR
- Batch size 24, gradient accumulation ×2 (effective batch 48)
- bfloat16 AMP, AdamW fused, `torch.compile`
- Full checkpoints saved to `src/checkpoints/` (includes optimizer/scheduler state for resumption)
- W&B project `midi-gpt` required

### f) Generate

```bash
python -m src.generation.generate_main \
    --prompt data/handcrafted_test_midis/chords_bass_melody_Amin.mid \
    --checkpoint src/checkpoints/best.pt \
    --top-k 1 \
    --top-p 0.92 \
    --temperature 0.9
```

Options: `--top-k`, `--top-p`, `--temperature`, `--n-samples` (default 5), `--single-track` (omit prompt track).

Output: `data/generations/generated_{stem}_remi_sample{i}.mid`

Generation stops at 9 bar tokens (≈4-bar continuation, max 1536 total).

## Pre-trained Checkpoint

Download manually from [huggingface.co/Origamay/midi-gpt/tree/main](https://huggingface.co/Origamay/midi-gpt/tree/main). Place `best.pt` in `src/checkpoints/`. It is already in minimal format (only `model_state_dict` + `config`), ready for generation.

If you train locally, `src/utils/extract_model_checkpoint.py` can strip full checkpoints (which include optimizer/scheduler state for resumption) down to the minimal format.

## Architecture Notes

- **Tokenization**: REMI, config: beat_res 16th notes, 32 velocities, no chords/rests/tempos/time-signatures/programs/drums
- **Model**: HuggingFace GPT2LMHeadModel with learned positional embeddings (`GPTStyleEmbedding`), Pre-LN (`norm_first=True`), final LayerNorm before LM head
- **Pad token = 0** everywhere: loss `ignore_index=0`, collate `pad_id=0`, key padding mask `(input_ids == 0)`
- **Bucketed padding**: sequence lengths rounded up to `{64, 128, 256, 512, 1024, 1536}` to limit `torch.compile` recompilations
