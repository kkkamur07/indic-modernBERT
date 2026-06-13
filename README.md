## Indic Modern BERT

The goal of the project is to extend the ModernBERT paper ([arxiv:2412.13663](https://arxiv.org/abs/2412.13663)) to 15 Indic languages so that people can benefit from higher retrieval quality as a consequence of better sequence length.

The first approach is to train a good tokenizer for Hindi only. Once that recipe works, we can extend it to other Indic languages and combine tokenizers if required.

## Vocabulary size design

Tokenizer vocabulary size is not chosen only for linguistic coverage — it must also align with how the embedding and softmax layers map onto GPU execution. ModernBERT uses **50,368** tokens for this reason: a base BPE vocabulary plus special tokens and reserved `[unused*]` slots, rounded to a hardware-friendly size.

When picking or padding `vocab_size`, aim for alignment with three GPU-level constraints:

- **Tensor** — divisible by **64** (clean FP16 tensor-core loads; no wasted tail in embedding/softmax matmuls)
- **Tile** — friendly to **128 × 256** GEMM blocks (better occupancy on the vocab projection)
- **Wave** — divisible across **SMs** (fewer idle lanes at the end of a wave)

**Practical rule:** choose `vocab_size % 64 == 0` (enforced in Pydantic config validation). If BPE + special tokens land slightly below a good target — e.g. 50,285 merges + 5 specials = 50,290 — pad with reserved `[unused0]` … `[unusedN]` tokens (ModernBERT uses 83 unused slots, IDs 50285–50367) until you hit the aligned size (50,368).

### ModernBERT special tokens (reference)

Released ModernBERT tokenizers (`answerdotai/ModernBERT-base`) append these after the base BPE vocabulary:

- `[UNK]` — 50280
- `[CLS]` — 50281
- `[SEP]` — 50282
- `[PAD]` — 50283
- `[MASK]` — 50284
- `[unused0]` … `[unused82]` — 50285–50367

`[CLS]` / `[SEP]` also serve as BOS/EOS in the model config. Our Hindi tokenizer trainers use the same five functional specials and can pad with `[unused*]` to reach an aligned `vocab_size`.


### BPE + SuperBPE

BPE should only merge inside the pertoken chunck and super, one stage is with pretok and one stage is without pretok. So that tokens can be learned beyond the boundaries as well, which are created by **pre-tok**.

## Running

Run from the **repo root** (`indic-modernBERT/`). Source code is in `indic-modernBert/` — flat imports, not an installable package.

```bash
uv sync
make train-superbpe
make validate-superbpe
```

Or without Make:

```bash
PYTHONPATH=indic-modernBert uv run python -m tokenizer.trainer.superbpe_trainer
cd indic-modernBert && uv run python scripts/validate_superbpe.py
```

Other targets: `train-bpe`, `eval-intrinsic`, `eval-parity`, `pretokenization`.

SuperBPE uses a vendored `tokenizers` patch (merge extension in Rust, under `_support_repo/`). `uv sync` installs it automatically. Requires Rust if no prebuilt wheel matches your platform.

```bash
uv sync
make train-superbpe
make validate-superbpe
```
