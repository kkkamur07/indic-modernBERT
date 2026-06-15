# Upstream ModernBERT vs indic-modernBERT

Reference upstream: `_support_repo/ModernBERT/` (`main.py`, `src/text_data.py`, `yamls/modernbert/modernbert-base-pretrain.yaml`).

Our implementation: `indic-modernBERT/pretrain/`, `configs/pretrain/hindi_mlm.yaml`, `scripts/run_pretrain.py`.

**Bottom line:** Model architecture, sequence packing, optimizer, scheduler, callbacks, and Composer `DataSpec` wiring are largely ported. The main intentional fork is **Parquet + Hindi pretokenization** instead of MDS. **Single-GPU production** is the target — no FSDP or multi-GPU algorithms. Production Trainer wiring (checkpoints, resume, eval cadence, progress bar, TensorBoard) is in place via `hindi_mlm_phase1.yaml`. Corpus token budget is measured (~23.6B); remaining work: production batch size and eval batch size on our GPU.

---

## Hindi corpus token estimate (2026-06-14)

**Method:** local sampling of one parquet per split × shard count.
**Tokenizer:** `bpe_vs50368` (vocab 50368). **Pretokenization:** `preprocess_for_tokenizer()` with script norm on.  
**Raw JSON:** `artifacts/corpus_stats/hindi_bpe_vs50368_estimate.json`  
**Derived schedule:** `artifacts/corpus_stats/hindi_training_schedule.json`

### Corpus by split

| Split | Shards | Rows | Parquet | Sample shard | Tokens/shard | Est. tokens | tok/row | B/token |
|-------|--------|------|---------|--------------|--------------|-------------|---------|---------|
| verified | 97 | 16.95M | 34.2 GB | `data-0.parquet` | 92.5 M | **8.97 B** | 529 | 10.61 |
| unverified | 113 | 16.93M | 39.9 GB | `data-0.parquet` | 91.8 M | **10.38 B** | 613 | 10.76 |
| synthetic | 64 | 5.78M | 15.3 GB | `wiki_hin_Deva_0000_of_0063.parquet` | 66.8 M | **4.27 B** | 740 | 10.02 |
| **Total** | **274** | **39.65 M** | **89.3 GB** | | | **23.617 B** | **596** | **10.57** |

- vs upstream ModernBERT pretrain corpus exposure (**1.719 T** tok trained): Hindi raw corpus is **1.37%** of that budget; matching upstream token-steps would be **~73 epochs** over this data.
- UTF-8 text in corpus: **~233 GB** (uncompressed text bytes before BPE).

### Production schedule (phase 1 @ `global_batch=512`)

Practical warmups (round numbers, close to upstream %): **50M** LR warmup, **500M** batch ramp. Context extension **deferred** until more data or 8192 is needed.

| Knob | Hindi value | % of 1 epoch |
|------|-------------|--------------|
| `max_duration` | `23617204995tok` | 100% |
| `t_warmup` | `50000000tok` | 0.21% |
| `batch_size_warmup_tokens` | `500000000tok` | 2.12% |
| `global_train_batch_size` | `512` | ~45k batches/epoch |
| `eval_interval` | `450ba` | ~100 evals/epoch (~236M tok apart) |
| Best checkpoints | `save_best_checkpoints` ×3 | by eval loss → `phase1/best/` |

---

## Glossary

| Term | Meaning |
|------|---------|
| **`ba`** | **Batch** — one optimizer step at the configured **global** batch size (after Composer microbatching + gradient accumulation). `4000ba` = save/eval every 4000 optimizer steps. |
| **`tok`** | **Tokens** — duration counted in tokens seen (unpadded when `count_padding_tokens: false`). `3_000_000_000tok` = 3B tokens. |
| **`grad_accum`** | Composer splits each dataloader batch into `device_train_microbatch_size` chunks and accumulates until one optimizer step. We do **not** manually divide loss; `PretrainConfig.grad_accum_steps` is **informational/logging only** (`global_batch // microbatch`). |

---

## Quick parity checklist

| Area | Status | Decision / notes |
|------|--------|------------------|
| Sequence packer + `DataSpec` | ✅ Aligned | Verbatim port; production train path is unpadded via `cu_seqlens` |
| `decoupled_stableadamw` + `warmup_stable_decay` | ✅ Aligned | Production schedule in `hindi_mlm_phase1.yaml`; drop `OneMinusSqrtScheduler` (not needed) |
| Unpadded + FA2 + `amp_bf16` + `compile_model` | ✅ Target | Production precision/attention must match upstream; `torch.compile` + FA2 should work on single GPU |
| MLM 0.3 train / 0.15 eval | ✅ Aligned | |
| `count_padding_tokens: false` | ✅ Aligned | |
| Parquet (not MDS) | ✅ Accepted | Keep parquet; no MDS conversion |
| Hindi BPE + pretokenization | ✅ Accepted | Fork-only; `preprocess_for_tokenizer()` before BPE |
| `batch_size_warmup_tokens` (~50B) | ✅ Scaled | `686946044tok` (2.91% of ph1, same as upstream) |
| `global_train_batch_size` | 🔲 Scale | Smoke default 8; set production global batch for our GPU; Composer handles accum |
| DataLoader workers/prefetch | ✅ Tuned | RTX 4090 sweep → train **6/4**, eval **3** |
| `eval_interval` | ✅ Wired | `450ba` in `hindi_mlm_phase1.yaml` (~100 evals/epoch); smoke yaml omits it (eval off) |
| `save_interval` + retention | ✅ Wired | `450ba`, `save_num_checkpoints_to_keep: 1` + `save_best_checkpoints` top-3 |
| Resume / autoresume | ✅ Wired | `load_path` on Trainer; `autoresume: true` in phase1; context extension uses explicit `load_path` |
| `progress_bar` | ✅ Wired | `progress_bar: true` in phase1 yaml → Composer Trainer |
| TensorBoard logging | ✅ Wired | `build_logger()` + `loggers.tensorboard` in phase1; `tensorboard` in `pretrain` extra |
| FSDP / RoPE schedule / EMA | ⏭️ Skip | Single GPU — not needed for our setup |
| Multi-GPU / `update_batch_size_info()` | ⏭️ Skip | Single GPU; eval batch size can be set explicitly in yaml |
| W&B / downstream eval harness | ⏭️ Skip | TensorBoard instead; no GLUE/BEIR port for now |
| `ConcatenatedSequenceCollatorWrapper` | ⏭️ Low priority | See [§ collator vs packing](#concatenatedsequencecollatorwrapper-vs-packing) — not used in upstream production pretrain |

---

## Decisions (agreed)

1. **Parquet + Hindi BPE** — keep; no MDS path.
2. **Single GPU** — skip FSDP, EMA, RoPE schedule algorithms, multi-GPU batch math.
3. **Production train** — packed, unpadded FA path (`cu_seqlens` from sequence packer); precision/attention match upstream yaml.
4. **Production checkpoints** — `eval_interval`/`save_interval: 450ba`, `save_best_checkpoints` (top 3 by eval loss), `autoresume`, TensorBoard.
5. **Schedulers** — `warmup_stable_decay` only; do not use `OneMinusSqrtScheduler`.
6. **Gradient accumulation** — Composer owns it via `global_train_batch_size` + `device_train_microbatch_size`; no manual `loss / accum` in our code.

## Open action items (production)

| Priority | Item | Notes |
|----------|------|-------|
| P0 | Production yaml: token-based `max_duration` / `t_warmup` | ✅ Set in `hindi_mlm_phase1.yaml` (1 epoch ≈ 23.617B tok) |
| P1 | Estimate Hindi corpus token count | ✅ `artifacts/corpus_stats/hindi_bpe_vs50368_estimate.json` |
| P1 | Set production `global_train_batch_size` / `device_train_microbatch_size` | Fit VRAM with FA2 + compile; verify Composer accum in a short run |
| P1 | Eval batch size in production yaml | Set `global_eval_batch_size` / `device_eval_microbatch_size` explicitly (single GPU) |
| P2 | Unpadded eval collator (optional) | Only if we want eval to match train attention path; upstream eval is padded by default |
| P2 | Context extension production knobs | Mirror phase1 checkpoint/eval/TensorBoard settings in `hindi_mlm_context_extension.yaml` |

---

## 1. Data format

### Upstream

| Item | Detail |
|------|--------|
| Format | MDS shards (`index.json` + raw shards) |
| Conversion | `src/convert_dataset.py`, `src/data/hf_to_mds.py` |
| Loaders | `NoStreamingDataset` (local), `StreamingTextDataset` (remote) in `src/text_data.py` |
| Layout | `data_folder/train`, `data_folder/validation` |
| Pretokenized option | MDS with `input_ids` bytes + optional `len` |

**Defaults** (`modernbert-base-pretrain.yaml`): `streaming: false`, `split: train` / `validation`.

### Ours

| Item | Detail |
|------|--------|
| Format | Parquet (`text` column) under `data/sangrah_dataset/verified/hin/*.parquet` |
| Download | `indic-modernBERT/dataset/sangrah_dataset.py` (HF `ai4bharat/sangraha`) |
| Loaders | `ParquetMLMDataset` + `TokenizeCollator` / `MLMCollator` in `pretrain/parquet_mlm.py`; packed train via `build_parquet_train_dataloader` |
| Eval holdout | `data/eval/hi/` via `holdout_manifest.txt` |
| Tokenization | Always on-the-fly from `text` (no pretokenized shards) |

### Differences

- **No MDS path** — no conversion step; works directly with Sangraha parquet.
- **No streaming / remote object store** — local parquet only.
- **Sangraha splits** — `verified/hin`, `unverified/hin`, `synthetic/hin_*` (same globs as `measure_corpus_tokens.py`).
- **Separate eval tree** vs upstream validation split from same MDS root.
- **Parquet read path (fixed 2026-06-14):** `ParquetMLMDataset` uses mmap'd row-group reads and per-row `.as_py()` — **not** `Column.to_pylist()` or full-shard `pq.read_table()` on training shards (OOM with workers). Details: `LEARNINGS.md` § Parquet DataLoader memory fixes.
- Parquet row reads may be slower than mmap'd MDS at very large scale (mitigated by row-group reads and a small LRU cache).

**File pairs to diff:** `src/text_data.py` ↔ `pretrain/dataloader.py` + `pretrain/parquet_mlm.py`

---

## 2. Tokenization and MLM masking

### Upstream

- Tokenize in `StreamingTextDataset._tokenize()` / `NoStreamingDataset._tokenize()`.
- **Packed train:** MLM masking inside `SequencePacker.mlm_masking()` (`src/sequence_packer.py`).
- **Eval / unpacked:** `DataCollatorForLanguageModeling` in `build_text_dataloader()` collate_fn.
- Tokenizer: `answerdotai/ModernBERT-base`.

### Ours

- Hindi text through `preprocess_for_tokenizer()` (script norm) before BPE.
- Tokenizer: custom `artifacts/tokenizer/bpe_vs50368`.
- **Packed train:** same `SequencePacker.mlm_masking()` (`pretrain/sequence_packer.py`, ported).
- **Eval / probe:** `MLMCollator` wraps HF `DataCollatorForLanguageModeling` with `padding="max_length"` (`pretrain/parquet_mlm.py`).

### Differences

- Hindi-specific pretokenization pipeline (no upstream equivalent).
- **Train (packed):** `TokenizeCollator` tokenizes with `padding=False`; packer emits `cu_seqlens` for unpadded FA — **this is the production path** and matches upstream packing behavior.
- **Eval / probe:** `MLMCollator` uses `padding="max_length"` — padded batches. Upstream eval is also padded (`sequence_packing: false`); acceptable for held-out loss/accuracy, but compute is less efficient than packed train.

### `ConcatenatedSequenceCollatorWrapper` vs packing

Upstream `ConcatenatedSequenceCollatorWrapper` (`text_data.py`) adds `sequence_id` from EOS/BOS boundaries so **unpadded** attention knows doc boundaries when multiple sequences share one padded row. It is only used in the **non-packing** branch **and** only when `eos_token_id` or `bos_token_id` is set in dataset config.

**`modernbert-base-pretrain.yaml` does not set those IDs** — production upstream pretrain does not use this wrapper. It uses **sequence packing** instead, which produces `cu_seqlens` directly (same as our ported packer).

**Recommendation:** Do **not** block production on `ConcatenatedSequenceCollatorWrapper`. Train is already correct via packing. Implement the wrapper only if we later want an **unpacked** multi-doc-per-row path (e.g. unpadded eval without packing). For now, padded eval + packed train matches upstream defaults.

---

## 3. Sequence packing

### Upstream & Ours (ported)

Both use:

- `GreedyBestFitSequencePacker` — greedy best-fit into `micro_batch_size × max_seq_len` windows
- `BufferedIterable` — background prefetch thread
- `BatchSizeWarmupScheduler` — ramps packed outgoing batch size
- `split_packed_batch()`, `get_num_samples_in_packed_batch()` — Composer microbatch splitting
- `DataSpec` in `main.py` / `pretrain/wiring.py`

**Activation:** `streaming: false` + `sequence_packing: true` → raw `DataLoader` → packer → `BufferedIterable`.

### Differences

| Setting | Upstream | Ours (`hindi_mlm.yaml`) |
|---------|----------|-------------------------|
| `sequence_packing` | `true` | `true` |
| `eval_sequence_packing` | `false` | `false` |
| `packing_prefetch_factor` | default 5 | `5` |
| `batch_size_warmup_min_size` | `${device_train_microbatch_size}` | same |
| `batch_size_warmup_tokens` | `50_000_000_000tok` | `686946044tok` in `hindi_mlm_phase1.yaml` |

Probe path does not use packing.

### Batch warmup tokens — pretokenize vs on-the-fly?

`batch_size_warmup_tokens` controls how long the packer ramps from `device_train_microbatch_size` to full device batch. It is a **schedule knob**, not a requirement to pretokenize the corpus.

| Approach | Pros | Cons |
|----------|------|------|
| **On-the-fly** (current) | No extra disk; pretokenization stays in DataLoader workers | Need to **estimate** total training tokens (sample shards × count) to set `50B`-style values |
| **Pretokenized parquet/MDS** | Faster reads at scale; exact per-shard token counts | Large disk footprint — **measure storage first** before committing |

**Action:** Corpus measured (see [§ Hindi corpus token estimate](#hindi-corpus-token-estimate-2026-06-14)). `batch_size_warmup_tokens` and `max_duration` set in `hindi_mlm_phase1.yaml`. Pretokenize only if profiling shows tokenization is the bottleneck after worker tuning.

**File pairs to diff:** `src/sequence_packer.py` ↔ `pretrain/sequence_packer.py`

---

## 4. DataLoader and worker prefetch

### Upstream (`text_data.py`)

| Setting | Value |
|---------|-------|
| Train `num_workers` | 6 |
| Eval `num_workers` | 3 |
| `prefetch_factor` | 2 (default) |
| `pin_memory` | true |
| `persistent_workers` | true |
| Sampler | `DistributedSamplerPCG64DXSM` |

### Ours (`pretrain/dataloader.py`)

| Setting | Value |
|---------|-------|
| Train `num_workers` | `pretrain.num_workers` or fallback `probe.dataloader_num_workers` (**6** in `hindi_mlm.yaml`) |
| Eval `num_workers` | `pretrain.eval_num_workers` or fallback `probe.eval_dataloader_num_workers` (**3**) |
| `prefetch_factor` | `pretrain.dataloader_prefetch_factor` or probe default (**4** in `hindi_mlm.yaml`) |
| `pin_memory` | true |
| `persistent_workers` | true when workers > 0 |
| Sampler | `DistributedSamplerPCG64DXSM` (ported) |
| Packing buffer | `5 × device_batch_size` (same as upstream default) |

### Differences

- Worker/prefetch tuned on RTX 4090 — train **6/4**, eval **3**; set in `hindi_mlm.yaml`.
- Multi-GPU sampler exists but **not needed** for single-GPU production.
- `DataloaderSpeedMonitor` callback available but not in default yaml — use during tuning.

Measured on RTX 4090 (probe): workers ≥ 2 hides tokenize/collate overhead; production yaml uses 6 train / 3 eval workers. Re-profile when moving to production model depth and batch size.

---

## 5. Batch size and gradient accumulation

### Upstream

```yaml
global_train_batch_size: 4608
device_train_microbatch_size: 96
```

Composer splits each dataloader batch into microbatches and accumulates — no manual `loss / accum` in user code. `update_batch_size_info()` in `main.py` validates divisibility and sets per-device / eval batch sizes.

### Ours

```yaml
global_train_batch_size: 8          # smoke
device_train_microbatch_size: 2
# informational only: grad_accum_steps = 8 / 2 = 4
```

Composer `Trainer` receives `device_train_microbatch_size` and splits each global batch into microbatches, accumulating gradients until one optimizer step. **We do not manually divide loss.** `PretrainConfig.grad_accum_steps` exists for logging in `train.py` / `run_pretrain.py` only.

### Differences

- **4608 vs 8** — smoke default; set production global batch to fit VRAM (FA2 + `compile_model`).
- **Verify in production:** short run logging `global_batch`, `microbatch`, and Composer step count confirms accum = `global / microbatch`.
- No `update_batch_size_info()` — set `global_eval_batch_size` / `device_eval_microbatch_size` explicitly in production yaml (single GPU).

---

## 6. Batch size warmup

### Upstream

```yaml
batch_size_warmup_min_size: ${device_train_microbatch_size}  # 96
batch_size_warmup_tokens: 50_000_000_000tok
```

`BatchSizeWarmupScheduler` ramps packed outgoing batch from microbatch to full device batch over warmup tokens (divided by `world_size`).

### Ours

- `batch_size_warmup_min_size` wired in `hindi_mlm.yaml`.
- `batch_size_warmup_tokens` only in `hindi_mlm_phase1.yaml` (`686946044tok`).
- Smoke `hindi_mlm.yaml` keeps `max_duration: 100ba`; use `hindi_mlm_phase1.yaml` for full token budget.

---

## 7. Training framework and loop

### Upstream

- Entry: `composer main.py yamls/modernbert/modernbert-base-pretrain.yaml`
- `main.py` → `Trainer.fit()` with token/batch duration units (`ba`, `tok`).
- Algorithms: `gradient_clipping`, `ema`, `rope_schedule` (`FlexBertRopeSchedule`).
- FSDP, `compile_config`, `autoresume`, `init_from_checkpoint()` for context extension.

### Ours

| Path | Framework |
|------|-----------|
| Full pretrain | Composer `Trainer` in `pretrain/train.py` |
| Probe | Custom PyTorch loop in `pretrain/probe.py` |
| Wiring | `pretrain/wiring.py` (ported subset of `main.py`) |

### Differences

| Feature | Upstream | Ours (decision) |
|---------|----------|-----------------|
| `algorithms` (RoPE schedule, EMA, grad clip) | yes | **Skip** — single GPU |
| `fsdp_config` | supported | **Skip** — single GPU |
| `compile_model` | `true` in arch yaml | **Keep** — `torch.compile` + FA2 on single GPU |
| `autoresume` | supported | **Wired** — `autoresume: true` in `hindi_mlm_phase1.yaml` |
| `progress_bar` | yes | **Wired** — `progress_bar: true` in phase1 yaml |
| `save_interval` / retention | yes | **Wired** — phase1 `450ba`, keep 1 rolling + `best/` top-3 |
| Context extension | `load_path` + `restart_override` | supported via `hindi_mlm_context_extension.yaml` |

**File pairs to diff:** `main.py` ↔ `pretrain/wiring.py` + `pretrain/train.py`

---

## 8. Optimizer

### Upstream & Ours (aligned)

Both use `decoupled_stableadamw` → `StableAdamW(..., decouple_lr=True)` with `filter_bias_norm_wd: true`.

```yaml
lr: 8e-4
betas: [0.9, 0.98]
eps: 1e-6
weight_decay: 1e-5
log_grad_norm: true
```

**Files:** upstream `main.py:build_optimizer()` + `src/optimizer.py` ↔ `pretrain/wiring.py:build_optimizer()` + `pretrain/optimizer.py`.

### Difference

- Probe uses plain `torch.optim.AdamW` (intentional for LR sweep simplicity).

---

## 9. Learning rate schedule

### Upstream

```yaml
scheduler:
  name: warmup_stable_decay
  t_warmup: 3_000_000_000tok
  alpha_f: 0.0
  t_decay: 0tok
max_duration: 1_719_000_000_000tok
```

### Ours

| Config | `t_warmup` | `max_duration` | Other production wiring |
|--------|------------|----------------|-------------------------|
| `hindi_mlm.yaml` (smoke) | `100ba` | `100ba` | no eval interval; schema defaults for save (`1000ba`) |
| `hindi_mlm_phase1.yaml` | `50000000tok` | `23617204995tok` | `450ba` eval/save, `save_best_checkpoints` ×3, TensorBoard |

Scheduler classes ported: `WarmupStableDecayScheduler`, `CosineInverseSqrtScheduler` in `pretrain/scheduler.py`.

**Decision:** Use **`warmup_stable_decay` only** for production. `OneMinusSqrtScheduler` is upstream machinery for LR decay experiments / data upsampling — **not planned for Hindi pretrain**.

Context extension (both): `constant_with_warmup`, `lr: 3e-4`, `reset_time: true`, `restart_override: true`.

---

## 10. Precision and attention path

### Upstream & Ours (largely aligned)

| Setting | Value |
|---------|-------|
| `precision` | `amp_bf16` |
| `padding` | `unpadded` |
| `unpad_embeddings` | `true` |
| `loss_function` | `fa_cross_entropy` |
| `sliding_window` | `128` |
| `global_attn_every_n_layers` | `3` |
| `compile_model` | `true` |
| `masked_prediction` | `true` |

Fork adds `use_fa2: true` explicitly in `configs/model/modernbert_base.yaml` (upstream defaults to true in model config).

### Differences

- Probe uses `modernbert_probe.yaml` (6 layers, `loss_function: cross_entropy`, `global_attn_every_n_layers: -1`) — not full production loss path.
- Fork `hardware_alignment` block in model yaml (fork-only).
- TF32 enabled in `train.py` on CUDA.

---

## 11. Eval during training

### Upstream

- Always builds `eval_loader` from `validation` MDS split.
- `eval_interval: 450ba` (~100 evals/epoch at `global_batch=512`)
- `global_eval_batch_size: 1024`, `device_eval_batch_size: 128`
- Padded MLM at 0.15 mask, no packing.

### Ours

| Path | Status |
|------|--------|
| **Probe** | Eval every `metrics_every_steps` on `data/eval/hi/` via `evaluate_mlm()` |
| **Full pretrain** | `build_eval_evaluator()` returns `None` unless **both** `eval_data_root` and `eval_interval` are set |

Default `hindi_mlm.yaml` sets `eval_data_root: data/eval/hi` but **no `eval_interval`** → Composer eval disabled (smoke).

`hindi_mlm_phase1.yaml` sets `eval_interval: 450ba` → eval runs on holdout when using phase1 config. Metrics: `eval/loss`, `eval/masked_accuracy` (TensorBoard).

Eval batch falls back to `probe.eval_batch_size: 4` unless `global_eval_batch_size` is set.

**Gaps vs upstream (remaining)**

| | Upstream | Ours |
|---|---|---|
| Eval interval | `450ba` | ✅ in `hindi_mlm_phase1.yaml` |
| Eval batch | `device_eval_batch_size: 128` | set explicitly (single GPU) — still open |
| Downstream eval | GLUE, BEIR, etc. | **skip** for now |

---

## 12. Checkpointing and resume

### Upstream

```yaml
save_folder: checkpoints/{run_name}
save_interval: 450ba
save_num_checkpoints_to_keep: 1
save_num_checkpoints_to_keep: -1
# load_path: null
```

`restart_override` in `main.py` resets LR/WD/scheduler after resume. Context extension loads from `latest-rank0.pt`.

### Ours

**Implemented:**

- `save_folder`, `save_interval`, `save_num_checkpoints_to_keep` passed to Trainer
- `autoresume` passed when set in yaml (`true` in `hindi_mlm_phase1.yaml`)
- `load_path`, `reset_time`, `restart_override` on `fit()`
- `apply_restart_override()` in `wiring.py`
- `hindi_mlm_context_extension.yaml` mirrors upstream resume pattern (explicit `load_path`, `autoresume` not set)

**Not wired (optional / low priority):**

- `save_overwrite`, `load_weights_only`

Probe writes `metrics.jsonl` / `probe_summary.json`, not Composer checkpoints.

---

## 13. Logging and monitoring

### Upstream callbacks (`main.py:build_callback()`)

| Callback | Default |
|----------|---------|
| `speed_monitor` | `window_size: 100` |
| `lr_monitor` | `{}` |
| `scheduled_gc` | `{}` |
| `log_grad_norm` | `batch_log_interval: 10` |
| `packing_efficiency` | `log_interval: 10` |

W&B commented in pretrain yaml; `WandBLogger` via `build_logger()`.

### Ours

- Same callback set in `pretrain/callbacks/` + `pretrain/wiring.py` (skips `packing_efficiency` when packing off).
- Default `hindi_mlm.yaml` includes all five callbacks.
- Entry: `scripts/run_pretrain.py` — loguru file logs, writes `resolved_config.json`.
- **TensorBoard** — `build_logger()` in `wiring.py`; phase1 yaml:

  ```yaml
  loggers:
    tensorboard:
      log_dir: artifacts/model/modernbert/tensorboard/phase1
  ```

  View: `tensorboard --logdir artifacts/model/modernbert/tensorboard/phase1` (requires `uv sync --extra pretrain`).
- Probe: local JSONL only.
- `make profile-probe` for dataloader vs compute regime (scripts only).

### Differences

- We use **TensorBoard** instead of W&B for local experiment tracking.
- Probe logging stays local JSONL.

Upstream W&B is commented out in pretrain yaml; we prefer TensorBoard for our workflow.

---

## 14. Entry points and config

| | Upstream | Ours |
|---|----------|------|
| Launch | `composer main.py <yaml> [overrides]` | `python scripts/run_pretrain.py` (Hydra) |
| Config | OmegaConf yaml tree | Pydantic `PretrainConfig` + Hydra yaml |
| Pretrain yaml | `yamls/modernbert/modernbert-base-pretrain.yaml` | `configs/pretrain/hindi_mlm.yaml` |
| Phase configs | separate yamls under `yamls/modernbert/` | `hindi_mlm_phase1.yaml`, `hindi_mlm_context_extension.yaml` |
| Probe | N/A | `configs/sweep/hindi_mlm_probe.yaml` |
| Model arch | inline under `model:` | `configs/model/modernbert_base.yaml` |
| Tokenizer | `answerdotai/ModernBERT-base` | `artifacts/tokenizer/bpe_vs50368` |

---

## Recommended next steps (updated)

1. **Batch size** — pick `global_train_batch_size` / `device_train_microbatch_size` for RTX 4090; confirm Composer grad accum; retune `eval_interval`.
2. **Eval batch size** — set `global_eval_batch_size` / `device_eval_microbatch_size` in production yaml.
3. **Context extension decay yaml** — `hindi_mlm_lr_decay.yaml` @ `686946044tok` when starting phase 3.

**Explicitly out of scope (single GPU):** FSDP, EMA, RoPE schedule algorithms, multi-GPU, W&B, downstream GLUE/BEIR, `ConcatenatedSequenceCollatorWrapper` (unless unpadded eval becomes a requirement).

---

## Config files by intent

| Goal | Config |
|------|--------|
| Smoke pretrain | `configs/pretrain/hindi_mlm.yaml` |
| Production scheduler / checkpoints / eval / TensorBoard | `configs/pretrain/hindi_mlm_phase1.yaml` |
| Context extension (1024 → 8192) | `configs/pretrain/hindi_mlm_context_extension.yaml` |
| LR / pipeline probe | `configs/sweep/hindi_mlm_probe.yaml` |

---

## File index

| Topic | Upstream | Ours |
|-------|----------|------|
| Data loading | `src/text_data.py` | `pretrain/dataloader.py`, `pretrain/parquet_mlm.py` |
| Sequence packing | `src/sequence_packer.py` | `pretrain/sequence_packer.py` |
| Trainer wiring | `main.py` | `pretrain/train.py`, `pretrain/wiring.py` |
| Probe loop | — | `pretrain/probe.py` |
| Optimizer | `src/optimizer.py` | `pretrain/optimizer.py` |
| Schedulers | `src/scheduler.py` | `pretrain/scheduler.py` |
| MLM eval metrics | — | `pretrain/evals/mlm.py` |
| Config schema | yaml only | `indic-modernBERT/config/schema.py` |
| Sangraha download | — | `indic-modernBERT/dataset/sangrah_dataset.py` |
| Entry scripts | `main.py` | `scripts/run_pretrain.py`, `scripts/run_mlm_probe.py` |
`