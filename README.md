## Indic Modern BERT

Hindi-first port of [ModernBERT](https://arxiv.org/abs/2412.13663) (22L encoder, 8192 context, retrieval-focused). Goal: scale to 15 Indic languages once the Hindi recipe works.

Run everything from the **repo root**. Python package lives in `indic-modernBERT/` (flat imports, not pip-installable).

## Quick start

```bash
uv sync
uv run python -m dataset.sangrah_dataset --count 20 --eval-count 1
make train-bpe
make eval-bpe
```

- **Train Hindi BPE** (all vocab sizes): `make train-bpe` or `make train-bpe-nohup`
- **Tokenizer eval:** `make eval-bpe` writes the comparison table to `logs/eval_bpe.log`
- **LR sweep (Optuna, full pretrain):** `uv sync --extra pretrain --extra sweep && make lr-sweep`
- **Full pretrain (Composer):** `uv sync --extra pretrain && make train-pretrain`
- **Phase 1 / context extension:** see [Training](#training) below
- **Model smoke test:** `make train-smoke-50ba`
- **HF export:** `make export-hf ARGS="ckpt.pt out/dir --tokenizer artifacts/tokenizer/bpe_vs50368"`

Artifacts: `artifacts/tokenizer/bpe_vs{V}/`, `artifacts/model/modernbert/`.

## Tokenizer

- **BPE** on Sangrah Hindi: script norm (`indic-nlp`) → NFKC + regex pre-tokenize → BPE (`tokenizers` lib).
- **Target vocab:** **50,368** (same as upstream: BPE + specials + `[unused*]`, padded to ÷64 for tensor cores).
- Use `preprocess_for_tokenizer()` at inference so train/eval/inference match.
- Config: `configs/tokenizer.yaml`. Intrinsic eval holdout: `data/eval/hi/`.

Hardware vocab notes (tensor vs tile vs wave): `configs/model/README.md`.

### Eval Summary

| Metric | Better |
|--------|--------|
| Fertility | lower |
| Bytes/token | higher |
| NSL | lower |
| Rényi efficiency | higher |

Latest Hindi holdout run (174k rows, `data/eval/hi/`):

| Tokenizer | Fertility | Bytes/token | NSL | Rényi eff | Vocab |
|-----------|-----------|-------------|-----|-----------|-------|
| IndicBERTv2 (reference) | 1.233 | 10.534 | 0.000 | 0.380 | 250k |
| BPE 32k | 1.260 | 10.310 | 1.022 | 0.469 | 32k |
| **BPE 50k** | **1.224** | **10.608** | **0.993** | **0.447** | **50k** |
| BPE 65k | 1.208 | 10.751 | 0.980 | 0.434 | 65k |
| BPE 98k | 1.188 | 10.927 | 0.964 | 0.417 | 98k |
| BPE 131k | 1.178 | 11.025 | 0.955 | 0.405 | 131k |
| BPE 196k | 1.166 | 11.135 | 0.946 | 0.390 | 196k |
| sarvam-1 | 1.471 | 8.829 | 0.000 | 0.452 | 68k |
| gemma-4 | 1.389 | 9.348 | 0.000 | 0.396 | 262k |
| Llama-4 Scout | 1.730 | 7.507 | 0.000 | 0.434 | 200k |

**50k** is the production target: fertility matches IndicBERT (1.224 vs 1.233) with slightly higher bytes/token; larger BPE vocabs trade Rényi efficiency for lower fertility and longer tokens.

## Model

Encoder ported from `_support_repo/ModernBERT/` → `indic-modernBERT/model/modernbert/`.

| Config | Use |
|--------|-----|
| `configs/model/modernbert_base.yaml` | 22L production |
| `configs/model/modernbert_tiny.yaml` | 4L GPU smoke |

Key settings: RoPE, unpadded + sequence packing, FA3/FA2 global + FA2 local attention, `init_method: full_megatron`, `vocab_size: 50368`.

Pipeline walkthrough: `notebook/pipeline_map.ipynb`.

## Training

ModernBERT trains in **three phases** (~2T tokens total). Published ratios from the paper — **we will rescale for Hindi** (corpus size, batch, duration).

1. **MLM pretrain** — seq 1024, RoPE 10k / 10k, **1.719T** tokens (upstream); Hindi phase 1 uses **1e-4** LR (see [LR sweep](#learning-rate-sweep)), WSD warmup → flat
2. **Context extension** — seq 8192, RoPE 160k / 10k, **250B** tokens, LR **3e-4**, flat
3. **LR decay @ 8k** — seq 8192, RoPE 160k / 10k, **50B** tokens, LR **3e-4**, 1−√ decay

Also: **30%** MLM mask (train), **15%** (eval); **StableAdamW** (β=(0.9,0.98), ε=1e-6, WD=1e-5); batch ramp **768→4608** over **50B** tokens (base).

**Our configs:**

```bash
# Smoke defaults
uv run python scripts/run_pretrain.py --config-name hindi_mlm

# Paper phase-1 ratios (LR + batch warmup; override max_duration for prod)
uv run python scripts/run_pretrain.py --config-name hindi_mlm_phase1

# Phase 2 — needs phase-1 checkpoint
uv run python scripts/run_pretrain.py --config-name hindi_mlm_context_extension
```

| Yaml | Role |
|------|------|
| `configs/pretrain/hindi_mlm.yaml` | dev / smoke |
| `configs/pretrain/hindi_mlm_phase1.yaml` | phase 1 targets |
| `configs/pretrain/hindi_mlm_context_extension.yaml` | 8192 + resume from phase 1 |

Full tables (large model, microbatches, rollback): `docs/learning.md`. Parity notes (packing, FA, parquet vs MDS): same file.

### Learning rate (sweep)

Upstream ModernBERT pretrain uses peak LR **8e-4** at global batch **4608**. Run an Optuna sweep on the **same phase-1 stack** as production (`hindi_mlm_phase1` → `modernbert_base`, micro=8, 500M packer warmup, `fa_cross_entropy`):

```bash
uv sync --extra pretrain --extra sweep
make lr-sweep
```

8 trials, log-uniform **3e-5–3e-4**, **1000 batches/trial** (~524M tokens — enough to finish packer warmup). Artifacts: `artifacts/model/modernbert/lr_sweep/`.
Each trial writes `sweep_summary.json` with `eval_loss` and `lr`. Optuna minimizes best eval MLM loss from `save_best_checkpoints`.

### DataLoader (workers / prefetch)

Historical local tuning on **RTX 4090** found the packed-path settings below to be fastest without worker OOMs.

Production (`hindi_mlm.yaml`, `sequence_packing=true`) uses the **packed-path** winner from a partial sweep (2 shards; full 8-shard grid OOM'd workers at w≥4):

| Setting | Value |
|---------|-------|
| Train `num_workers` | **2** |
| Train `prefetch_factor` | **4** |
| Eval `num_workers` | **3** |
| `packing_prefetch_factor` | **5** |

**5.64 ms/batch** (177.3 batches/s) vs **14.12 ms** with `workers=0`.

Padded eval path (seq 512, no packing) was faster at w=8/pf=2 (2.98 ms) in an older profile, but production follows packed-path RAM limits. Re-run dataloader profiling after changing batch size, shard count, or model depth.

## Data

**Training:** `data/sangrah_dataset/verified/hin/*.parquet` (~12.6B Hindi tokens).

**Eval holdout:** `data/eval/hi/` — create with `dataset.sangrah_dataset --eval-count N`.

**Candidates for scale-up:** Sangrah unverified Hindi, IndicCorp V2/V1.

## Layout

```
indic-modernBERT/     # config/, model/, pretrain/, tokenizer/, dataset/
configs/              # tokenizer, model, pretrain, sweep yamls
scripts/              # run_pretrain.py, compare_bpe_vocabs.py, export_hf.py, …
docs/                 # learning notes and port differences
artifacts/            # tokenizers, checkpoints, ablations
_support_repo/        # upstream ModernBERT reference (do not commit blindly)
```

## Reference

- Paper: [arxiv:2412.13663](https://arxiv.org/abs/2412.13663)
- Deep dives: `docs/learning.md`, `docs/difference.md`, `configs/model/README.md`
