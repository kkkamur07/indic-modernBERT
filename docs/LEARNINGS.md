# Learnings & Engineering Notes

Running notes on every non-obvious decision, fix, and "why does this work this way" moment in the project. Add new entries at the top of the relevant section so the most recent knowledge is easiest to find.

---

## Table of Contents

1. [GPU / Hardware fundamentals](#1-gpu--hardware-fundamentals)
2. [Flash Attention: FA2 vs FA3 and when to use each](#2-flash-attention-fa2-vs-fa3-and-when-to-use-each)
3. [Tokenizer design](#3-tokenizer-design)
4. [Data: why Parquet instead of MDS](#4-data-why-parquet-instead-of-mds)
5. [Parquet DataLoader memory fixes](#5-parquet-dataloader-memory-fixes)
6. [Training phases and RoPE](#6-training-phases-and-rope)
7. [Eval MLM metrics](#7-eval-mlm-metrics)
8. [DataLoader worker leak across Optuna trials](#8-dataloader-worker-leak-across-optuna-trials)
9. [Smoke pretrain launcher fixes](#9-smoke-pretrain-launcher-fixes)
10. [Code-review correctness fixes](#10-code-review-correctness-fixes)
11. [Vendored SuperBPE patch](#11-vendored-superbpe-patch)
12. [IndicCorp V2 ingestion + phase-2 setup](#12-indiccorp-v2-ingestion--phase-2-setup)

---

## 1. GPU / Hardware fundamentals

Understanding these three concepts saves hours of mysterious slowdowns.

### Tensors (128-byte alignment)

Modern GPU memory controllers work most efficiently when tensors start on 128-byte boundaries and their innermost dimension is a multiple of a certain size. In FP16/BF16 that means **64 elements** (64 × 2 bytes = 128 bytes). This is why `vocab_size` targets like 50368 are chosen — they are divisible by 64.

Violating alignment doesn't crash anything; it just silently de-rates the memory bus. Always keep `vocab_size`, `hidden_size`, and `intermediate_size` divisible by 64. For maximum LM-head efficiency, 128 is even better (a full tile), but upstream ModernBERT deliberately uses 50368 (÷64, not ÷128) — the token-count overhead of going to the nearest ÷128 isn't worth the minor gain.

### Waves (SM occupancy)

A "wave" is one full fill of all Streaming Multiprocessors (SMs) on the GPU. An RTX 4090 has 128 SMs. If your vocabulary matrix needs exactly 393 tiles and you have 128 SMs, you get 3 full waves + 9 leftover tiles — those last 9 tiles still use a full wave, so you pay for 4 waves of work. The theoretical optimum is for `vocab_size / 128` to divide evenly by SM count, but in practice no vocabulary size in a ±512 sweep hits wave-zero on 128 SMs. Treat this as experimental; don't optimize for it at the expense of other alignment requirements.

### Tiling (128×256 tile blocks)

GEMM kernels (matrix multiplications) operate on tiles, typically 128×256 elements. When a dimension is not a multiple of the tile size, the last tile is padded with zeros, doing useless work. This matters most for the LM head (vocab×hidden), which runs on every single token. Keep vocab and hidden dimensions tile-friendly.

**Recorded ablation on RTX-class hardware (128 SMs):**

| `vocab_size` | Tensor ÷64 | Tile ÷128 | Notes |
|---|---|---|---|
| **50368** | ✓ | ✗ | **Upstream ModernBERT** — intentional |
| 50304 | ✓ | ✓ | −64 from upstream |
| 50432 | ✓ | ✓ | +64 from upstream |

Our Hindi tokenizer target stays **50368** to match upstream exactly unless profiling shows a real gain.

---

## 2. Flash Attention: FA2 vs FA3 and when to use each

ModernBERT uses a **layer-dependent** attention kernel strategy:

| Layer type | `sliding_window` setting | Kernel used |
|---|---|---|
| Global (every 3rd layer) | `(-1, -1)` — full context | FA3 if available, else FA2 |
| Local (remaining layers) | `(64, 64)` — window 128 | FA2 only (`flash_attn_varlen_func` with `window_size=`) |

**Why not FA3 on local layers too?** FA3's Hopper API now supports `window_size`, but upstream deliberately keeps local layers on FA2 — it's battle-tested in their training stack. Switching local to FA3 would be an ablation, not a parity path.

### GPU support matrix

| GPU | Architecture | Compute cap | What runs |
|---|---|---|---|
| RTX 4090 / 3090 | Ada / Ampere | 8.9 / 8.6 | **FA2 on all layers** (global layers fall back from FA3 to FA2) |
| H100 / H800 | Hopper | 9.0 | FA3 on global + FA2 on local |

FA3 (`flash_attn_interface`, from `flash-attention/hopper/`) targets Hopper only (CUDA ≥ 12.3). If you're on an RTX card, FA3 is simply not installed and the code falls back to FA2 everywhere — this is correct and expected.

### Important correctness note (fixed 2026-06-15)

`FlexBertPaddedRopeAttention.forward` was checking the global import flag `IMPL_USE_FLASH2` instead of the per-instance `self.use_fa2`. This meant `config.use_fa2=False` was silently ignored when FlashAttention was installed, always routing through FA2 even when you explicitly disabled it (e.g. for CPU/SDPA testing). Fixed to use `self.use_fa2`.

Similarly, `dropout_p=self.p_dropout` was passed unconditionally to all FA and SDPA kernel calls — attention dropout was applied during `eval()`. Fixed to `self.p_dropout if self.training else 0.0` across all 26 call sites.

---

## 3. Tokenizer design

### Why BPE on Hindi specifically

Devanagari script has a rich morphology — words combine prefixes, roots, suffixes, and case endings into long strings. A tokenizer that hasn't seen Hindi specifically will shatter common words into many subpieces, inflating sequence lengths and diluting contextual signal. Training BPE directly on Sangrah Hindi text gives the model a Hindi-native vocabulary.

### Pipeline

```
raw text → ScriptNormalization (indic-nlp) → NFKC → regex pre-tokenize → BPE (HuggingFace tokenizers lib)
```

`ScriptNormalization` maps visually equivalent Devanagari characters to a canonical form (e.g. normalizes nukta placement). NFKC is standard Unicode normalization. The regex pre-tokenizer splits on whitespace and punctuation boundaries before BPE merges — same pattern as GPT-2.

**Always run `preprocess_for_tokenizer()` at inference.** Train, eval, and inference must go through the same normalization pipeline or subword boundaries will differ.

### Vocab size choice: 50,368

The target is **50,368** — same as upstream ModernBERT. This is BPE vocab + special tokens + `[unused*]` padding, rounded up to the nearest multiple of 64 for tensor-core alignment. It's *not* a multiple of 128 (which would be tile-aligned for the LM head), but matching upstream exactly avoids weight-tiling complexity when loading pretrained embeddings.

### What the intrinsic eval metrics mean

| Metric | What it measures | Better direction |
|---|---|---|
| **Fertility** | Average tokens per word (lower = more whole words in vocab) | Lower |
| **Bytes/token** | Average bytes encoded per token (higher = denser encoding) | Higher |
| **NSL** (Normalized Sequence Length) | How much longer sequences are vs. the reference tokenizer, normalized | Lower |
| **Rényi efficiency** | Information-theoretic measure of how uniformly the vocab is used | Higher |

A good tokenizer has low fertility (few splits per word), high bytes/token (long, meaningful tokens), low NSL (sequences not much longer than reference), and high Rényi efficiency (vocab not dominated by a few tokens).

**50k BPE** hits parity with IndicBERTv2 on fertility (1.224 vs 1.233) while improving bytes/token — a good operating point. Larger vocabs improve fertility but trade Rényi efficiency (vocab usage becomes sparser).

---

## 4. Data: why Parquet instead of MDS

### Retrieval mMARCO TSV encoding (fixed 2026-06-26)

When preparing local Hindi mMARCO retrieval splits, do not use `requests.iter_lines(decode_unicode=True)`. Hugging Face serves UTF-8 TSV bytes, but `requests` can infer a Latin-1-style response encoding and persist mojibake such as `à¤à¤¿...` instead of Devanagari. Stream raw bytes and decode each line explicitly with UTF-8 before writing JSONL.

The full DPR training split should be prepared once as local JSONL with `make retrieval-prepare-full-subset`; it writes 1,250,000 train rows plus the 1,000-row held-out triplet dev split expected by `configs/retrieval_finetune/hindi_dpr.yaml`. The Optuna LR sweep keeps its smaller fixed 100k+1k JSONL split.

**MDS (Mosaic Streaming Dataset)** is MosaicML's binary shard format with an `index.json` for fast lookup. Upstream ModernBERT stores pretokenized samples in MDS. We use **Parquet instead**, for these reasons:

- Sangraha data is already in Parquet — converting adds a multi-hour preprocessing step with no benefit at ~100 GB scale
- MDS streaming is designed for remote/object-store corpora or multi-node setups without a full local copy; on a local NVMe SSD, Parquet + `num_workers` is just as fast
- `ParquetMLMDataset` is a thin adapter (~200 lines) that builds a lightweight `(parquet_file, row_idx)` index and reads one row group at a time — the rest of the training stack (packing, unpadding, optimizer, scheduler) is copied verbatim from upstream

Revisit MDS only if profiling shows the DataLoader is the actual bottleneck after tuning `num_workers`, prefetch, and disk layout.

### Corpus

Training data lives under `data/sangrah_dataset/verified/hin/*.parquet` (~12.6B Hindi tokens from 19 shards). Full corpus estimate (verified + unverified + synthetic): **23.617 B tokens** across 274 shards, ~89 GB.

Eval holdout: `data/eval/hi/` — one shard, created by `dataset.sangrah_dataset --eval-count 1`.

---

## 5. Parquet DataLoader memory fixes

*(2026-06-14)*

**Symptom:** pretrain or notebook DataLoader OOMs, swap-thrashes, or stalls for minutes when reading Sangrah Parquet shards. With `num_workers=2`, each worker could hold multi-GB Python lists in RAM.

### Root cause 1 — `to_pylist()` materializes the whole column

Old code did:
```python
pq.read_table(path, columns=[text_column])[text_column].to_pylist()
```
`to_pylist()` converts the entire Arrow column into Python `str` objects — several GB per shard. With multiple workers each caching one shard, plus a separate eval loader, RAM explodes.

### Root cause 2 — Arrow memory-map still too large for forked workers

Replacing `to_pylist()` with:
```python
pq.read_table(path, columns=[text_column], memory_map=True)
```
avoids materializing Python strings, but the full-shard Arrow table is still too large for forked workers to carry in parallel. Workers were killed by the OOM killer:
```
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
```

### Fix

`ParquetMLMDataset` now builds a shard + row-group offset index from Parquet *metadata* (no data read), then reads only the specific row group needed for each batch using `ParquetFile.read_row_group(..., columns=[text_column], use_threads=False)`. A small LRU cache keeps recently-used row-group tables warm.

**Do not** use `to_pylist()` or full-shard `pq.read_table()` on pretrain shards. The tokenizer training code (`tokenizer/trainer/`) still uses `to_pylist()` on small curated slices — that path reads much less data and is fine.

---

## 6. Training phases and RoPE

ModernBERT trains in three sequential phases. We rescale durations for Hindi corpus size; the upstream phase ratios are reference points.

### RoPE (Rotary Position Embeddings) — why the base matters

RoPE encodes position by rotating query/key vectors by angles that depend on the position index. The `base` parameter controls the frequency range — a higher base gives the model a "longer ruler" and lets it generalize to longer sequences than it saw during training.

Phase 1 uses `base=10000` (the standard). Context extension bumps the **global** layer base to `160000` — this is what allows 8192-token contexts without re-training from scratch. Local (sliding-window) layers keep `base=10000` because they only ever attend within a 128-token window, so they don't need the extended range.

### Phase table

| Phase | Seq len | Global RoPE θ | Local RoPE θ | Upstream tokens |
|---|---|---|---|---|
| 1 — Pretrain | 1024 | 10,000 | 10,000 | ~1.719T |
| 2 — Context extension (stable) | 8192 | 160,000 | 10,000 | 250B |
| 3 — Context extension (decay) | 8192 | 160,000 | 10,000 | 50B |

### Scheduler: WarmupStableDecay (WSD)

Phase 1 uses WSD: LR warms up linearly, holds flat at peak, then decays. Upstream peak is 8e-4 at global batch 4608. Our Hindi LR sweep explores 3e-5–3e-4 at batch 512 (RTX 4090 scale) with 1000 batches per Optuna trial.

### Context extension procedure (upstream recipe)

1. Load phase-1 weights (`load_path: latest-rank0.pt`)
2. Raise `max_seq_len` to 8192
3. Change RoPE in model config only: `rotary_emb_base: 160000`, keep local at 10000
4. Reset Composer step counters (`reset_time: true`, `restart_override: true`) so the LR/WD schedule restarts from the extension yaml
5. Train with `constant_with_warmup` scheduler at lower peak LR (3e-4 base, 5e-5 large)

### Optimizer: StableAdamW

StableAdamW divides the per-parameter learning rate by `max(1, RMS(gradient))` — this "stabilizes" the effective step size when gradient magnitudes blow up, which happens frequently early in training for large models. Params: β=(0.9, 0.98), ε=1e-6, WD=1e-5.

**`decoupled_stableadamw`** mode multiplies weight decay by `lr/max_lr` each step, decoupling it from the learning rate schedule. This requires `max_lr` to be set — a validation guard was added (2026-06-15) so it fails loudly rather than crashing at line 303 deep in the optimizer step.

### Paper training targets (upstream reference)

| Phase | Base global / micro | Large global / micro | Peak LR |
|---|---|---|---|
| Pretrain @ 1024 | 4608 / 96 | 4928 / 56 | 8e-4 / 5e-4 |
| Ext stable @ 8192 | 576 / 12 | 616 / 7 | 3e-4 / 5e-5 |
| Ext decay @ 8192 | 576 / 12 | 624 / 6 | 3e-4 / 5e-5 |

LR warmup: 3B tokens (base), 2B (large). Batch warmup: 768→4608 over 50B tokens (base).

---

## 7. Eval MLM metrics

*(Verified 2026-06-14)*

### How eval loss and accuracy are computed

| Metric | Implementation | Notes |
|---|---|---|
| Eval loss | `EfficientCrossEntropy` | FA-fused cross-entropy, `reduction=mean` over masked tokens in that forward pass |
| Eval accuracy | `MaskedAccuracy(ignore_index=-100)` | Logits vs labels, masked positions only |

**Microbatch averaging:** Composer splits each eval device batch into `device_eval_microbatch_size` chunks. `EfficientCrossEntropy` gives equal weight to each microbatch — *not* token-weighted across the full device batch. This matches upstream ModernBERT + Composer behavior.

**`disable_train_metrics: true`** in the phase-1 yaml only silences train metrics. Eval metrics are a separate `deepcopy` in `create_modernbert_mlm()` and always run.

**MLM randomness:** `DataCollatorForLanguageModeling` re-masks on every collate call. Two passes over the same eval loader will not match bit-for-bit. Compare only on frozen batches.

### Eval holdout gotcha

`data/eval/hi/holdout_manifest.txt` lists shards but eval only works once those files physically exist under `data/eval/hi/`. Create the holdout with:
```bash
uv run python -m dataset.sangrah_dataset --eval-count 1
```
Otherwise `iter_parquet_paths` raises `FileNotFoundError`.

### Labels mask: `!= -100`, not `> 0`

The MLM collator marks unmasked (non-predicted) positions with `labels = -100` (the PyTorch `ignore_index` convention). The original code used `labels > 0` as the mask, which *silently excluded token id 0* from the loss — any vocabulary entry at position 0 was never learned. Fixed (2026-06-15) to `labels != -100`.

---

## 8. DataLoader worker leak across Optuna trials

*(2026-06-15)*

**Symptom:** the LR sweep (`make lr-sweep`) ran the first trial fine but each subsequent Optuna trial consumed more RAM, eventually OOMing and being killed by the OS.

### Root cause 1 — `persistent_workers=True` with no teardown

`_dataloader_kwargs` hardcoded `persistent_workers=True`. This keeps PyTorch DataLoader workers alive across `__iter__` calls — good for epoch-looped single-run training (avoids re-spawning workers every epoch), but fatal for in-process Optuna multirun: each trial creates a new `Trainer` but the old DataLoader workers kept running in the background, accumulating RAM until OOM.

### Root cause 2 — teardown didn't reach the actual workers

The original `_release_training_resources` called `train_loader.close()`, which stopped the packer's `BufferedIterable` fill thread but *never reached the underlying `DataLoader` or its worker iterator*. The complication: with `persistent_workers=False`, PyTorch does **not** store the live iterator on `DataLoader._iterator` — instead it's held inside the packer as `src_iterator`. You have to walk the wrapper chain to find it:

```
DataSpec  →  BufferedIterable  →  packer  →  src_iterator  (_MultiProcessingDataLoaderIter)
```

### Fix

**`config/schema.py`:** added `dataloader_persistent_workers: bool = True` config field.

**`pretrain/dataloader.py`:** `_dataloader_kwargs` now reads `pretrain_cfg.dataloader_persistent_workers` instead of hardcoding `True`.

**`pretrain/train.py`:** replaced `_release_training_resources` with a full loader-graph walker:

```python
_LOADER_GRAPH_ATTRS = ("dataloader", "iterable", "src_iterable", "src_iterator",
                       "_active_iterator", "iterator", "_iterator")

def _walk_loader_graph(root) -> list:
    # BFS through wrapper chain, collecting every object
    ...

def _release_loader(loader):
    graph = _walk_loader_graph(loader)
    # 1. Stop fill threads first (packer, buffer) so they stop pulling from workers
    for obj in graph:
        if not isinstance(obj, DataLoader):
            close = getattr(obj, "close", None)
            if callable(close): close()
    # 2. Then reap the DataLoader worker subprocesses
    for obj in graph:
        if isinstance(obj, DataLoader):
            obj.persistent_workers = False
            _shutdown_worker_iterator(getattr(obj, "_iterator", None))
        else:
            _shutdown_worker_iterator(obj)  # catches src_iterator in packer
```

**`configs/sweep/hindi_mlm_lr_sweep.yaml`:** explicitly sets `dataloader_persistent_workers: true` so workers persist *within* a trial (performance) while the new teardown reaps them *between* trials.

**Rule of thumb:** when `persistent_workers=False`, the live `_MultiProcessingDataLoaderIter` is owned by whoever consumed the DataLoader (the packer, a buffer), not by `DataLoader._iterator`. You must reach it through the consumer's `src_iterator` attribute to shut it down.

---

## 9. Smoke pretrain launcher fixes

*(2026-06-14)*

**Symptom:** `make train-smoke-50ba-nohup` appeared stuck or failed silently before training even started.

| Area | Bug | Fix |
|---|---|---|
| Shell | `set -o pipefail` without forcing bash, no pipeline anyway | Makefile sets `SHELL := /bin/bash`; smoke command uses `script -e` for child exit status |
| Hydra paths | `hydra.run.dir: logs/smoke_50ba` changed cwd; relative config paths resolved to `logs/smoke_50ba/artifacts/...` (wrong) | `utils.paths.resolve_from_cwd()` resolves relative paths from repo root when cwd is inside the project |
| DataLoader worker | First train batch waited forever — a worker was OOM-killed reading a full parquet shard | `ParquetMLMDataset` now reads row groups instead of full shard columns (see §5) |
| Debug callback | `TrainStepLogger` crashed on multi-element `state.loss` with `float(tensor)` | `_loss_scalar()` averages multi-element tensors before logging |
| Log noise | Per-microbatch and packer logs made `nohup.log` enormous | Smoke Makefile sets `TRAIN_STEP_LOG=0`; smoke config sets `log_to_console: false` |

**Current quiet smoke command:**
```bash
make train-smoke-50ba-nohup
# logs at logs/smoke_50ba/nohup.log
# checkpoints at artifacts/model/modernbert/checkpoints/smoke_50ba/
```

---

## 10. Code-review correctness fixes

*(2026-06-15)*

A CodeRabbit review of the architecture-implemented PR caught several real bugs. Each was verified against the actual code before applying; hallucinations were discarded.

### Critical — active runtime crashes

| Location | Bug | Fix |
|---|---|---|
| `model/modernbert/model.py` L377, L403 | `labels > 0` mask excluded token-id 0 from MLM loss — any vocab entry at position 0 was silently skipped | `labels != -100` (the ignore-index convention) |
| `model/modernbert/model.py` L1544 | Layer-remapping used `round(i * pretrained / new)`, which can equal `pretrained_layers` — off-by-one IndexError when depth-expanding | `min((i * pretrained) // new, pretrained - 1)` |
| `model/modernbert/model.py` L1352, L1474 | Two Flex classifier heads did `(logits,) + output` where `output` is a bare tensor → `TypeError` when `return_dict=False` | `(logits,) + (output,)` |
| `pretrain/optimizer.py` L303 | `decoupled_stableadamw` never passed `max_lr`; division by `None` on first step with `weight_decay != 0` | Added `__init__` guard: `ValueError` when `decouple_lr=True` and `max_lr` is `None` or ≤ 0 |
| `pretrain/optimizer.py` L333 | `torch.stack(l1_norms)` raised `RuntimeError: stack expects non-empty TensorList` when a param group has no gradients | Added empty-list guard before stacking |
| `pretrain/optimizer.py` L177–178 | `self.grad_norms` was overwritten each loop iteration — only the last param group's norms were kept | Accumulate across groups; combine after the loop |
| `pretrain/scheduler.py` L81–82 | `_get_scheduler` returned `_inverse_sqrt_schedule` for warmup/cooldown slots; those call the function with `start_y`/`finish_y` but the inverse-sqrt function only accepts `alpha`/`beta` → `TypeError` | Added `for_bounded=True` flag; raises `ValueError` eagerly at scheduler construction |
| `pretrain/scheduler.py` L250–255 | When `t_cooldown=0`, a warning was issued but the cooldown branch still ran, dividing by `v_cooldown=0` at the final training step | Guard with `v_cooldown > 0` |
| `pretrain/sequence_packer.py` L528–540 | `_background_fill` only caught `StopIteration`; any other exception killed the fill thread silently, and `__next__` looped forever on an empty buffer — silent deadlock | Catch all exceptions, store in `self._error`, re-raise from `__next__` |
| `model/modernbert/attention.py` L1052 | `FlexBertPaddedRopeAttention.forward` checked global `IMPL_USE_FLASH2` flag instead of `self.use_fa2` — `config.use_fa2=False` was bypassed when FlashAttention was installed | `if self.use_fa2:` |
| `model/modernbert/initialization.py` L552 | `tile_embedding` declared `-> nn.Embedding` but had no `return` — callers got `None` | Added `return new_embedding` |
| `scripts/pipeline_trace.py` L132 | `max(norms)` raises `ValueError` on an empty list when no parameters have gradients | `max(norms) if norms else "N/A"` |

### Correctness — wrong behaviour, not a hard crash

| Location | Bug | Fix |
|---|---|---|
| `model/modernbert/attention.py` (26 sites) | `dropout_p=self.p_dropout` passed to FA/SDPA kernels unconditionally — dropout applied at eval time | `self.p_dropout if self.training else 0.0` everywhere |
| `model/modernbert/layers.py` L296, L343, L431, L468 | `_init_weights` called `.reset_parameters()` on `attn_norm`/`norm` which can be `nn.Identity` → `AttributeError` | `if hasattr(..., "reset_parameters")` guard |
| `model/modernbert/layers.py` L541–545 | `FlexBertPaddedPostNormLayer._init_weights` reset `mlp_norm` but not `attn_norm` (both are real norms in post-norm) | Added `self.attn_norm.reset_parameters()` |
| `attention.py`, `layers.py`, `mlp.py` | `get_attention_layer`, `get_bert_layer`, `get_mlp_layer` compared `layer_id < config.num_initial_layers` without checking for `None` → `TypeError` on default call | Added `layer_id is not None and …` guard in all three |

### Minor / lint

| Location | Fix |
|---|---|
| `pretrain/scheduler.py` | `stacklevel=2` on `warnings.warn` (Ruff B028); renamed `type` param to `schedule_type` (Ruff A002 — shadowed builtin) |
| `pretrain/wiring.py` | Both `WarmupStableDecayScheduler` and `CosineInverseSqrtScheduler` were missing `t_max=cfg.t_max` — silently fell back to `"1dur"` |
| `pretrain/gpu_batch.py` | `device.index` is `None` for `torch.device("cuda")` → log printed `cuda:None`; resolved via `torch.cuda.current_device()` fallback |
| `pretrain/callbacks/log_grad_norm.py` | Added `batch_log_interval > 0` validation |
| `pretrain/callbacks/packing_efficiency.py` | Added `log_interval > 0` validation |
| `model/modernbert/rotary.py` | `scale_base: Optional[bool]` → `Optional[float]` (used in numeric division) |
| `tokenizer/evals/common.py` | Parquet discovery changed from `verified/hin/*.parquet or *.parquet` to `**/*.parquet` — was missing `unverified/hin/` and `synthetic/hin_*/` shards |
| `scripts/diagnose_smoke_stages.py` | `device_batch_size = global // 1` → `global // dist.get_world_size()` |
| `configs/tokenizer.yaml` | Moved gated `meta-llama/Llama-4-Scout-17B-16E-Instruct` out of `baseline_tokenizer_names` — fails without HF access |
| `docs/difference.md` | `num_workers` documented as 6 but `hindi_mlm.yaml` says 2 — corrected |
| `notebook/README.md` | Broken link `../LEARNINGS.md` → `../docs/LEARNINGS.md` |

---

## 11. Vendored SuperBPE patch

SuperBPE extends a stage-1 BPE by reloading `merges.txt` and continuing training from those merges. Our Hindi BPE checkpoints use the HuggingFace/GPT-2 convention of a **leading space** on word-initial tokens (e.g. `" क"` is "क" at the start of a word). `BPE::save` writes those lines as **two leading spaces** in `merges.txt` — `"  क"` means merge `(" ", "क")`.

**Upstream bug:** `do_train_extend` in the forked `tokenizers` library used `line.split(" ")` (splitting on all spaces), which mis-parses those double-space lines. This caused panics (`Option::unwrap() on None`) or noisy ` not found in word_to_id` errors during stage 2.

**Our fix** (applied to the nested submodule under `_support_repo/superbpe/`, not upstream):

| File | Change |
|---|---|
| `tokenizers/src/models/bpe/model.rs` | `parse_bpe_merge_line()` — handles `"  …"` via `strip_prefix("  ")` |
| `tokenizers/src/models/bpe/trainer.rs` | `do_train_extend` uses `parse_bpe_merge_line` instead of raw `split(" ")` |

**If you update the `tokenizers_superbpe` submodule**, re-check this fix is still present; re-apply if it was reset. Rebuild the editable wheel after any Rust changes:

```bash
uv pip install -e _support_repo/superbpe/tokenizers_superbpe/bindings/python --force-reinstall --no-deps
```

Run a smoke check after rebuilding: tokenize a short Hindi phrase and verify the word-initial space handling is correct before retraining.

---

## 12. IndicCorp V2 ingestion + phase-2 setup

Notes from adding IndicCorp V2 (Hindi) as the phase-2 context-extension corpus.

### Source format — one document per line, blank-line separated

IndicCorp V2 (`ai4bharat/IndicCorpV2`) ships Hindi as plain `.txt` (`hi-1.txt`, `hi-3.txt`, ~26.66 GB each — download `hi-1` + `hi-3`, `hi-2` was not used). **Measured fact:** every document is exactly one non-empty line, and blank lines are pure separators (the max run of consecutive non-empty lines across 2M lines is 1). So there's no multi-line document grouping to do — iterate non-empty lines, skip empties. This killed an earlier `doc_mode`/`buffer` design: it was dead weight on this corpus.

### Converter — `dataset/indiccorp_dataset.py`

Streams the `.txt` line by line (a 25 GB file never lands in RAM) and writes parquet shards with the **exact Sangraha `verified/hin` schema** so all downstream readers work unchanged:

```
doc_id: large_string   # sha1(text), same convention as Sangraha
text:   large_string   # the document
type:   large_string   # "indiccorp_v2"
```

- Config is a constants block at the top of the file (no argparse) — edit + run `make convert-indiccorp`.
- The only bounded buffer is the current shard's rows (`ROWS_PER_SHARD`), flushed + cleared per shard. **Shards never split a document** (flush happens on whole-doc count), so words/text are never broken.
- `tqdm` is byte-based (`total = file size`, update by raw line bytes) so a 25 GB file shows a real %/ETA bar, not just a counter.

### Shard sizing

Hindi docs average ~752 utf-8 bytes (median 549, p99 4334). `ROWS_PER_SHARD = 200_000` → ~150 MB uncompressed text/shard, ~43 MB on disk after **zstd** — in line with Sangraha's ~175k-row shards.

### Token count — `one_shard_extrapolation`

Mirror the existing `artifacts/corpus_stats/hindi_bpe_vs50368_estimate.json` method: tokenize one full shard with `bpe_vs50368` (apply `preprocess_for_tokenizer` first), get tokens/row, multiply by shard count.

| metric | IndicCorp V2 shard | Sangraha verified (ref) |
|---|---|---|
| tokens/shard (200k rows) | 13,679,809 | 92,454,616 |
| tokens/row | 68.4 | 529.0 |
| utf-8 bytes/token | 11.0 | 10.6 |

Low tokens/row (68 vs 529) is expected: IndicCorp rows are single lines (~752 B), Sangraha rows are multi-paragraph docs. **IndicCorp V2 (hi-1+hi-3) = 70,923,978 docs across 356 shards = 4.851 B tokens.** Sangraha (phase-1) = 23.617 B.

### Duplicates are expected, not a bug

`doc_id = sha1(text)`, so identical text → identical id. IndicCorp has ~0.08% duplicate docs — short repeated web/Wikipedia boilerplate headers like `इतिहास.` (History.), `बाहरी कड़ियाँ.` (External links.). Inherent to a web crawl; no dedup was applied (deferred).

### Phase-2 config (`configs/pretrain/hindi_mlm_context_extension.yaml`) gotchas

- **Batch size is NOT safely inherited.** Base `hindi_mlm` sets `global_train_batch_size: 8`; phase-1 overrides to 512 but phase-2 originally did **not** → it silently ran at batch 8. Always set `global_train_batch_size` explicitly in phase-2.
- **`autoresume: true` needs `run_name`.** `train.py` only passes `run_name` to Composer when set; without it autoresume can't find prior checkpoints across launches. Phase-2 was missing it (phase-1 had it). Added `run_name: hindi_modernbert_context_extension`.
- **Batch-scoped intervals must be re-derived per phase.** At seq 8192, `tokens/step = global_batch × 8192`. With all IndicCorp (4.851 B tok) and `global_batch=512` → only ~1,156 batches total. Phase-1's `eval_interval: 450ba` would give ~2 evals; use `12ba` for ~96 evals.
- **100 evals cost ~fixed compute, independent of batch size.** Eval work = `n_evals × eval_subset_num_batches × eval_batch × seq`; training work = fixed tokens. At `eval_subset_num_batches=50`, 100 evals ≈ +54% compute. Dropped to 10 (~+11%).
- **`grad_accum_steps = global_train_batch_size // device_train_microbatch_size`.** Effective optimization batch is always `global_train_batch_size`; raising the microbatch only changes speed/VRAM, not the math. Schema enforces `global % microbatch == 0`, so use divisors of 512 (1, 2, 4, 8…).
- **Scheduler:** upstream phase-2 is `constant_with_warmup` with `t_warmup: 0tok` (no warmup; the 3B-tok warmup is phase-1 only). The wiring's `constant_with_warmup` path uses only `t_warmup` — `alpha_f`/`t_decay` are no-ops there.
- **LR:** upstream used a *lower* phase-2 LR (8e-4 → 3e-4). We instead reuse phase-1's Optuna lr `4.88e-4` (validated on this model+data); `restart_override: true` makes the yaml's LR/WD/microbatch win over the checkpoint's.
- **Arch:** `configs/model/modernbert_context_extension.yaml` extends base and bumps `rotary_emb_base` 10000 → **160000** for the 8192 context (local-attn RoPE base stays 10000).
- **Cross-corpus eval:** phase-2 trains on IndicCorp but `eval_data_root` inherits `data/eval/hi` (Sangraha holdout). Deliberate — keeps eval loss comparable to phase-1; just know it measures Sangraha-domain generalization.

### VRAM probing

Box is an RTX 4090 (24 GB). `make train-smoke-phase2` runs a 20-batch slice of the real phase-2 config (loads the phase-1 checkpoint, evals too, throwaway save folder, `autoresume=false`). Override microbatch to probe: `ARGS="pretrain.device_train_microbatch_size=4"`, watch `nvidia-smi`. Eval at 8192 is often the real memory peak, so the smoke evals on purpose.
