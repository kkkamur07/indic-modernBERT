# Learning notes

Background on data formats and streaming choices for this project.

## Project decision: parquet is enough

We train on **parquet** (Sangraha `verified/hin/*.parquet`, ~100 GB scale). No MDS conversion, no Mosaic `StreamingDataset`. `ParquetMLMMapDataset` is the only dataset-layer glue; packing, unpadding, optimizer, and scheduler match upstream ModernBERT.

Revisit MDS only if profiling shows the dataloader is the bottleneck after tuning `num_workers` and disk layout — not as a default prerequisite.

## What is MDS?

**MDS (Mosaic Streaming Dataset)** is MosaicML/Databricks’ on-disk training format: binary shards plus `index.json` for fast sample lookup. Upstream ModernBERT stores pretokenized (or raw `text`) samples in MDS and reads them via `NoStreamingDataset` (`streaming: false`) or `StreamingDataset` (`streaming: true`) in `src/text_data.py`.

| | **MDS (upstream)** | **Parquet (ours)** |
|---|---|---|
| Layout | `index.json` + raw shard files under e.g. `data_folder/train/` | `verified/hin/*.parquet` (Sangraha) |
| Access | mmap / shard index; optional remote streaming | `pyarrow` row reads via `(path, row_idx)` index |
| Pretokenization | Can store `input_ids` in shards | Tokenize `text` on the fly in the dataloader |
| Upstream code | `text_data.py` works as-is | Thin adapter only: `ParquetMLMMapDataset` in `pretrain/parquet_mlm.py` |

With parquet, we only need that thin adapter — **`GreedyBestFitSequencePacker`**, unpadding (`bert_padding.py`), optimizer, scheduler, and callbacks are copied verbatim from `_support_repo/ModernBERT/`. The training stack is the same; only the dataset layer differs.

## Do we need streaming for ~100 GB?

**No — not Mosaic `StreamingDataset`.** At ~100 GB (Sangraha verified + unverified Hindi is in this ballpark), data lives on disk; you do **not** load the full corpus into RAM.

- **Full pretrain (`sequence_packing: true`):** `ParquetMLMMapDataset` builds a lightweight index of `(parquet_file, row)` pairs. Workers read and cache one shard at a time. Shuffle is via `DistributedSamplerPCG64DXSM`, same idea as upstream.
- **Probe / padded path:** `ParquetMLMDataset` (`IterableDataset`) walks shards sequentially — enough for LR probes and eval.

Mosaic streaming is mainly for **remote/object-store** corpora or **multi-node** setups where you want sequential shard streaming without a local copy. For a **local ~100 GB parquet tree on NVMe**, map-style parquet + `num_workers` is sufficient for this project.

## Flash Attention: global FA3/FA2 + local FA2 (recommended)

Upstream ModernBERT (and our port) uses a **layer-dependent** path:

| Layer type | `sliding_window` | Kernel |
|------------|------------------|--------|
| Global (every 3rd layer) | `(-1, -1)` | FA3 if `flash_attn_interface` is installed, else FA2 |
| Local (other layers) | `(64, 64)` for window 128 | **FA2 only** with `window_size=` |

**Recommendation: keep this mix.** FA3’s hopper API now supports `window_size`, but upstream deliberately uses FA3 only on global layers — local layers stay on FA2 varlen + sliding window, which is battle-tested in their training stack. Switching local layers to FA3 would be an ablation, not the default parity path.

Code: `indic-modernBERT/model/modernbert/attention.py` (`use_fa3 = ... sliding_window == (-1, -1)`).

### GPU support (FA2 vs FA3)

| GPU | Architecture | Compute cap | Production path |
|-----|--------------|-------------|-----------------|
| RTX 4090 / 3090 | Ada / Ampere | 8.9 / 8.6 | **FA2 on all layers** (global layers use FA2 fallback) |
| H100 / H800 | Hopper | 9.0 | FA3 on global + FA2 on local (if `hopper/` installed) |

Official FA3 (`flash-attention/hopper` → `import flash_attn_interface`) targets **Hopper only** (Dao-AILab docs: H100/H800, CUDA ≥ 12.3). Upstream ModernBERT: install FA3 only “if using H100s”; otherwise FA2 everywhere is correct.

Verify routing on your machine: `make verify-fa-routing` (uses `configs/model/modernbert_base.yaml`).

## Training phases (RoPE)

| Phase | Seq len | Global RoPE θ | Local RoPE θ | Upstream duration |
|-------|---------|---------------|--------------|-----------------|
| Pretrain | 1024 | 10_000 | 10_000 | ~1.719T tok |
| Context extension (stable) | 8192 | 160_000 | 10_000 | 250B tok |
| Context extension (decay) | 8192 | 160_000 | 10_000 | 50B tok |

**Total ~2T tokens** (paper Table 3). Large model: phase-1 rollback at 900B (5e-4) → 800B (5e-5, WD 1e-6) before context extension.

Configs: `modernbert_base.yaml` + `hindi_mlm_phase1.yaml`; extension: `modernbert_context_extension.yaml` + `hindi_mlm_context_extension.yaml`.

### Paper training ratios (reference)

Treat as upstream targets; Hindi runs will rescale duration, batch, and data mix.

| Phase | Base tokens | Large tokens | Peak LR (base / large) | Scheduler |
|-------|-------------|--------------|------------------------|-----------|
| 1 — pretrain @ 1024 | 1.719T | 900B + 800B (rollback) | 8e-4 / 5e-4 → 5e-5 | WSD (`warmup_stable_decay`, `t_decay: 0tok`) |
| 2 — ext stable @ 8192 | 250B | 250B | 3e-4 / 5e-5 | `constant_with_warmup`, 0 warmup |
| 3 — ext decay @ 8192 | 50B | 50B | 3e-4 / 5e-5 | `one_minus_sqrt`, `alpha_f=0.001` |

**LR warmup:** 3B tokens (base), 2B (large). **Batch warmup:** 768→4608 over 50B (base); 448→4928 over 10B (large).

**At target batch (Table 3):**

| Phase | Base global / micro | Large global / micro |
|-------|---------------------|----------------------|
| Pretrain @ 1024 | 4608 / 96 | 4928 / 56 |
| Ext stable @ 8192 | 576 / 12 | 616 / 7 |
| Ext decay @ 8192 | 576 / 12 | 624 / 6 |

**Optimizer (all phases):** StableAdamW / `decoupled_stableadamw`, β=(0.9, 0.98), ε=1e-6, WD=1e-5 (1e-6 on large rollback + ext decay). MLM **30%** train, **15%** eval. Init: Megatron (base); tile-from-base (large).

Upstream yamls: `_support_repo/ModernBERT/yamls/modernbert/modernbert-base-pretrain.yaml`, `modernbert-base-context-extension.yaml`, `modernbert-base-learning-rate-decay.yaml`.

## Context extension (upstream recipe)

Reference: `_support_repo/ModernBERT/yamls/modernbert/modernbert-base-context-extension.yaml`.

Upstream does **not** use a gradual RoPE schedule algorithm in production yaml — they **swap static bases** and **resume from the phase-1 checkpoint**:

1. **Load** phase-1 weights: `load_path: checkpoints/{pretrain_run_name}/latest-rank0.pt`
2. **Raise** `max_seq_len` to **8192** (packing still on for train)
3. **Change RoPE** in model config only:
   - `rotary_emb_base: 160000` (global layers)
   - `local_attn_rotary_emb_base: 10000` (sliding-window layers — unchanged from pretrain)
4. **Optimizer reset** via Composer flags:
   - `reset_time: true` — scheduler/dataloader step counters restart
   - `restart_override: true` — LR, weight decay, microbatch taken from extension yaml (not checkpoint)
5. **Scheduler:** `constant_with_warmup` with `t_warmup: 0tok` (flat LR for extension)
6. **Train** `250B` tokens at lower peak LR (`3e-4` base, `5e-5` large)
7. **Eval loader:** `sequence_packing: false`, `mlm_probability: 0.15`, data from pretrain pool (validation split)

Same 22-layer / 768-dim arch as phase 1; only seq length, RoPE bases, LR schedule, and duration change.

## Hindi corpus & training budget

Measured 2026-06-14 (`make measure-corpus-tokens`): **23.617 B** tokens across verified + unverified + synthetic (274 shards, ~89 GB parquet). Details: `difference.md` § corpus estimate, `artifacts/corpus_stats/`.

**Phase 1 production (`hindi_mlm_phase1.yaml`):** `global_batch=512`, `microbatch=8`. Eval every **450ba** (~100×/epoch). **`save_best_checkpoints`** keeps top **3** by eval MLM loss under `checkpoints/phase1/best/`; TensorBoard logs `eval/loss` and `eval/masked_accuracy`. Composer also keeps `latest-rank0.pt` + 1 rolling interval save.

| Knob | Value | % of 1 epoch |
|------|-------|--------------|
| `max_duration` | `23617204995tok` | 100% |
| `t_warmup` | `50000000tok` | 0.21% |
| `batch_size_warmup_tokens` | `500000000tok` | 2.12% |

**Context extension:** deferred until more Hindi data or a task needs 8192 context.

**Eval:** `global_eval_batch_size: 64`, `device_eval_microbatch_size: 8`. Holdout is one parquet shard — eval every batch (`1ba`) would be ~45k full passes/epoch and unusable; **`450ba`** (~100 evals/epoch at `global_batch=512`) is the phase-1 default on a 4090.

## Eval MLM metrics (verified 2026-06-14)

Upstream parity path for `loss_function: fa_cross_entropy` in `modernbert_base.yaml`:

| Metric | Implementation | Source |
|--------|----------------|--------|
| Eval loss | `EfficientCrossEntropy` | Reads `outputs["loss"]` — FlashAttention CE, `reduction: mean` over **masked tokens in that forward** |
| Eval accuracy | `MaskedAccuracy(ignore_index=-100)` | `outputs["logits"]` + labels from `eval_forward` (labels popped from batch; filtered when `masked_prediction: true`) |

**Not a bug — intentional upstream behavior:**

- **`disable_train_metrics: true`** (phase-1 yaml) only clears **train** metrics. `eval_metrics` is a separate `deepcopy` in `create_modernbert_mlm()` — eval still runs.
- **Microbatch averaging:** Composer splits each eval device batch into `device_eval_microbatch_size` chunks. `EfficientCrossEntropy` does `sum(loss) / num_microbatches` — equal weight per microbatch, **not** token-weighted across the full device batch. Same as upstream ModernBERT + Composer.
- **MLM randomness:** `DataCollatorForLanguageModeling` re-masks on every collate. Two passes over the eval loader will not match bit-for-bit; compare paths on **frozen batches** only.
- **Probe vs Composer:** `evaluate_mlm()` (probe) and `EfficientHuggingFaceModel.eval_forward` + `update_metric` match to float noise on identical batches.

**`SaveBestCheckpoints`:** after each eval, reads `state.eval_metrics['eval']` (`EfficientCrossEntropy` + `MaskedAccuracy`), logs TensorBoard `eval/loss` + `eval/masked_accuracy`, saves top-N to `checkpoints/phase1/best/manifest.json`.

**Composer device placement:** Trainer calls `_ensure_metrics_device_and_dtype()` before eval — metrics must be on the same device as `outputs["loss"]`. `EfficientCrossEntropy.update` also `.detach().to(metric.device)` as a safety net.

**Eval dataloader:** padded MLM (`eval_sequence_packing: false`), `eval_mlm_probability: 0.15`, `shuffle: false` — mirrors `modernbert-base-pretrain.yaml` `eval_loader`.

**Eval holdout gotcha:** `data/eval/hi/holdout_manifest.txt` lists shards (e.g. `verified/hin/data-84.parquet`) but eval only works once those files exist under `data/eval/hi/`. Create with `dataset.sangrah_dataset --holdout-eval`; otherwise `iter_parquet_paths` raises `FileNotFoundError`.

**Config smoke test (Hydra):** `initialize_config_dir()` requires an **absolute** path:

```python
from pathlib import Path
from hydra import compose, initialize_config_dir
from config import load_pretrain_config

with initialize_config_dir(config_dir=str(Path("configs/pretrain").resolve()), version_base=None):
    cfg = compose(config_name="hindi_mlm_phase1")
p = load_pretrain_config(cfg)
```

**Env note:** when `/tmp` is full, `uv run` can fail — use `.venv/bin/python` with `TMPDIR=$PWD/.tmp` or free `/tmp` first.

## Hardware alignment ablations

**Not hard-enforced in Pydantic** (defaults match upstream; use `hardware_alignment.enforce: true` only if you opt in).

Recorded run: `artifacts/ablations/hardware_alignment/README.md` (RTX, 128 SMs).

| `vocab_size` | Tensor ÷64 | Tile ÷128 | Notes |
|--------------|------------|-----------|-------|
| **50368** | pass | **fail** | **Upstream ModernBERT** — intentional |
| 50304 | pass | pass | −64 from upstream |
| 50432 | pass | pass | +64 from upstream |

Upstream uses **50368**, not 50432. That size is BPE + specials + `[unused*]` padding to **÷64** (tensor cores), not **÷128** (full LM-head tile). Our Hindi tokenizer target should stay **50368** unless profiling shows a real gain from a tile-aligned neighbor.

```bash
make ablate-hardware
```

Results: `artifacts/ablations/hardware_alignment/latest.json`. Wave check uses **vocab tiles only** (`vocab/128 % sm_count`); none of the ±512 sweep hit wave-zero on 128 SMs — treat wave as experimental.
