### Guidelines 

Beware of the hardware requirements for wave, tiling and tensor quantization of the GPU. 

1. Tensors 128 Bytes : So around 64 size in FP16
2. Wave : SM Processors
3. Tiling : Tiles are in blocks of 128*256 blocks. 


### Tokenizations Approaches

1. BPE + SuperBPE
2. Sentence Piece

We need to do `ScriptNormalization` and how are we going to do that is the question mostly with `indicNLP` library and sampling strategy as well to build a very good tokenizer. 

Need to inspect the folder structure of the `sangrah` dataset and then download random datasets to ensure that we have fair representation, we can calculate the size of the dataset as proxy and then only download the required.

**Potential / evaluated Hindi sources:** Sangraha verified+unverified, IndicCorp V2 (`hi-1/2/3.txt`), IndicCorp V1 (`hi.txt` via IndicNLPSuite). For encoder MLM, verified+unverified Sangraha (~25B Hindi tokens) is sufficient for round 2; current 19-shard sample is probe-only for round 1.

### Vendored SuperBPE patch: `merges.txt` leading-space parsing

SuperBPE stage 2 reloads stage-1 `merges.txt` and extends it. Our Hindi BPE checkpoints
include word-initial tokens with a **leading space** (HF/GPT-2 convention). `BPE::save`
writes those lines as two leading spaces, e.g. `"  क"` means merge `(" ", "क")`.

**Upstream bug:** `do_train_extend` in the fork used `line.split(" ")` (all spaces),
which mis-parses those lines and can panic (`Option::unwrap() on None`) or spam
` not found in word_to_id` during stage 2.

**Our local fix** (in the nested submodule, not upstream yet):

| File | Change |
|------|--------|
| `tokenizers/src/models/bpe/model.rs` | `parse_bpe_merge_line()` — handles `"  …"` via `strip_prefix("  ")` |
| `tokenizers/src/models/bpe/trainer.rs` | `do_train_extend` uses `parse_bpe_merge_line` instead of `split(" ")` |

**If you update `tokenizers_superbpe`:** re-check this fix is still present; re-apply if
the submodule was reset. Rebuild the editable wheel after Rust changes:

```bash
uv pip install -e _support_repo/superbpe/tokenizers_superbpe/bindings/python --force-reinstall --no-deps
```

**Smoke check:** rebuild the editable wheel and run a tiny Hindi phrase tokenization before retraining.

### Parquet DataLoader memory fixes (2026-06-14)

**Symptom:** Pretrain or notebook DataLoader steps OOM, swap thrash, or stall for minutes when
reading Sangrah parquet shards. With `num_workers=2`, each worker could hold full-shard Python
lists or full-shard Arrow text columns in RAM.

**Root cause 1:** `ParquetMLMDataset._rows()` (old) and `load_eval_texts()` (old) did:

```python
pq.read_table(path, columns=[text_column])[text_column].to_pylist()
```

`to_pylist()` materializes the **entire text column** as Python `str` objects (~multi‑GB per
shard). Caching one shard per path × multiple workers × train + eval loaders → RAM explosion.

**Root cause 2:** Replacing `to_pylist()` with full-shard Arrow tables was still too large for
forked DataLoader workers:

```python
pq.read_table(path, columns=[text_column], memory_map=True)
```

Even though this avoids Python `str` lists, each worker can still map/cache large text columns
while prefetching raw batches for the sequence packer. The smoke run showed this as:

```text
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
```

**Fix (in `indic-modernBERT/pretrain/parquet_mlm.py`):**

| Component | Approach |
|-----------|----------|
| `ParquetMLMDataset` | Build shard + row-group offsets from parquet metadata; read only the needed row group with `ParquetFile.read_row_group(..., columns=[text_column], use_threads=False)`; LRU-cache a small number of row-group tables; `__getitem__` uses `table.column(col)[i].as_py()` per row |
| `load_eval_texts` | Arrow column iteration with `.as_py()` (notebook helper only; do not use for full training shards) |

**Do not** use `to_pylist()` or full-shard `pq.read_table()` on full pretrain parquet shards.
Tokenizer **training** code under `tokenizer/trainer/` still uses `to_pylist()` on smaller curated
slices — that path is separate.

**Verify:** `notebook/pipeline_map.ipynb` Step 2 (production DataLoaders) or:

```bash
PYTHONPATH=indic-modernBERT uv run python -c "
from pathlib import Path
from pretrain.parquet_mlm import ParquetMLMDataset
ds = ParquetMLMDataset(Path('data/sangrah_dataset'), 'text', max_shards=1)
assert len(ds[0]) > 0 and len(ds[1]) > 0
print('ok', len(ds), 'rows')
"
```

**Not a data bug:** Garbled decoded eval text (`ą`, split English subwords) in the notebook is
from MLM `[MASK]` replacements + BPE — not bad parquet reads. Check `labels` and
`eval_masked_preview` in Step 2 for the real batch fields.

### Smoke pretrain launcher fixes (2026-06-14)

**Symptom:** `make train-smoke-50ba-nohup` looked stuck or failed before real training, and logs
were noisy enough to hide the actual error.

**Fixes:**

| Area | Bug | Fix |
|------|-----|-----|
| Shell | `set -o pipefail` was used without forcing bash, and the command had no pipeline anyway. | `Makefile` sets `SHELL := /bin/bash`; the smoke command now relies on `script -e` for child exit status and no longer uses redundant `pipefail`. |
| Hydra paths | `hydra.run.dir: logs/smoke_50ba` changed cwd; missing artifact paths could resolve under `logs/smoke_50ba/artifacts/...`. | `utils.paths.resolve_from_cwd()` now resolves relative config paths from the repo root whenever the cwd is inside this project. |
| DataLoader worker | First train batch waited forever because a worker was OOM-killed while reading parquet. | `ParquetMLMDataset` now reads row groups instead of full shard columns. |
| Debug callback | `TrainStepLogger` crashed on multi-element `state.loss` with `float(tensor)`. | `_loss_scalar()` averages multi-element tensors before logging. |
| Log noise | Per-microbatch, packer, collator, and Composer console logs made `nohup.log` huge. | Smoke Makefile sets `TRAIN_STEP_LOG=0`; smoke config sets `log_to_console: false` and `train_step_logger.log_microbatches: false`. |

**Current quiet smoke command:**

```bash
make train-smoke-50ba-nohup
```

Expected useful artifacts/logs:

| Path | Purpose |
|------|---------|
| `logs/smoke_50ba/nohup.log` | Outer `make` + minimal training output |
| `logs/smoke_50ba/train.log` | `script` TTY capture |
| `artifacts/model/modernbert/checkpoints/smoke_50ba` | Smoke checkpoints |
| `artifacts/model/modernbert/tensorboard/smoke_50ba` | TensorBoard logs |

If debug detail is needed again, temporarily remove `TRAIN_STEP_LOG=0` from the Makefile command
or run the Python command manually with `TRAIN_STEP_LOG=1`.