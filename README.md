# Indic ModernBERT

A Hindi-first extension of [ModernBERT](https://arxiv.org/abs/2412.13663) — a state-of-the-art masked language model with 22 transformer layers, 8192-token context, and retrieval-focused design. The goal is to build a strong Hindi encoder first, then scale to all 15 major Indic languages once the recipe is proven.

Why only 15 languages ? Because for them only substantial dataset ( > 5B tokens ) are available.

---

## What is ModernBERT?

BERT-style encoders read text bidirectionally (every token attends to every other token) and are trained with Masked Language Modelling (MLM): randomly mask ~30% of tokens, predict them. This makes them excellent at understanding tasks — classification, NER, retrieval, question answering — but they don't generate text.

ModernBERT is a 2024 redesign of BERT that incorporates everything learned from the GPT generation: rotary position embeddings (RoPE) for long contexts, Flash Attention for memory-efficient training, alternating global+local attention layers for efficiency, and a much larger training corpus. It achieves state-of-the-art results on retrieval benchmarks while remaining fast to fine-tune.

We port it to Hindi by replacing the tokenizer and training from scratch on Sangraha Hindi text (~23B tokens).

ModernBERT focuses on two things : 

1. Efficiency : Designing the model as per the hardware
2. Long context retrieval : Due to extended context length because of **ROPE** and **Alternating Attention**

---

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Download ~20 training shards + 1 eval shard from Sangraha
# We donwloaded around 270 shards. 
uv run python -m dataset.sangrah_dataset --count 20 --eval-count 1

# 3. Train a Hindi BPE tokenizer (vocab size 50k) - after the abalations. 
make train-bpe

# 4. Evaluate the tokenizer against baselines
make eval-bpe
```

After that, the main commands are:


| What you want to do                                               | Command                                                                          |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Smoke-test the full training pipeline (50 batches)                | `make train-smoke-50ba`                                                          |
| Run an Optuna LR sweep to find the best learning rate             | `make lr-sweep`                                                                  |
| Full phase-1 pretrain                                             | `uv sync --extra pretrain && make train-pretrain`                                |
| Export a checkpoint to HuggingFace format                         | `make export-hf ARGS="ckpt.pt out/ --tokenizer artifacts/tokenizer/bpe_vs50368"` |
| Evaluate an HF export / Hub model                                 | `make run-evals ARGS="eval.model.model_name_or_path=<hf-id-or-path>"`            |
| Explore the pipeline interactively ( Debugging purposes as well ) | open `notebook/pipeline_map.ipynb`                                               |


Artifacts are written to `artifacts/tokenizer/bpe_vs{V}/` and `artifacts/model/modernbert/`.

---

## Tokenizer

We train a BPE tokenizer directly on Hindi text. The pipeline is:

```
raw text → script normalisation (indic-nlp) → NFKC → regex pre-tokenise → BPE
```

Yes, each step is necessary — skip one and the tokenizer silently degrades.

**Script normalisation** is the most critical step for Devanagari. The same word can be encoded in multiple byte-level ways that look pixel-identical on screen but are different strings to a computer. For example, the Hindi word for "Kumar" (कुमार) can be written with the vowel sign `ु` composed into the consonant, or as a separate combining character — same glyph, different bytes. Without normalisation, BPE treats these as two different words and wastes two vocabulary slots on the same meaning. Multiply this across thousands of common words and you lose a huge chunk of your 50k vocabulary to duplicates.

```
# Same word, two byte sequences — both look like "कुमार" on screen:
"क\u0941म\u093Eर"   ← composed form  (4 codepoints)
"क\u0941म\u093Eर"   ← decomposed form (5 codepoints, nukta separate)

# After ScriptNormalization → both become the same canonical string
# → BPE sees one word, not two
```

**NFKC** handles Unicode compatibility characters that aren't Devanagari-specific. Examples:

```
"１२३"  (fullwidth digits, common in web-scraped text)  →  "123"
"ﬁ"    (fi ligature)                                    →  "fi"
"²"    (superscript 2)                                  →  "2"
```

Without NFKC, the model learns `123` and `１２３` as completely separate sequences even though they mean the same thing.

**Regex pre-tokeniser** draws hard boundaries before BPE merges begin. See the image at the top of this page for a side-by-side example. It splits on whitespace and punctuation, so BPE can never create a token that straddles a word boundary. Without it you'd get tokens like `"है।"` (word + full-stop fused together) — useful in some contexts but inconsistent, and it bloats the vocabulary with punctuation-suffixed variants of every common word.

**Target vocab size: 50,368** — same as upstream ModernBERT. This is divisible by 64 (required for GPU tensor-core alignment) and matches the upstream embedding weight dimensions exactly, which simplifies initialising from pretrained weights later.

Always use `preprocess_for_tokenizer()` at inference time. Training and inference must go through the same normalisation pipeline or the same Hindi word will tokenize differently at inference than it did during training.

### Tokenizer eval results

Latest run on the Hindi holdout (174k rows):


| Tokenizer               | Fertility ↓ | Bytes/token ↑ | NSL ↓     | Rényi eff ↑ | Vocab   |
| ----------------------- | ----------- | ------------- | --------- | ----------- | ------- |
| IndicBERTv2 (reference) | 1.233       | 10.534        | 0.000     | 0.380       | 250k    |
| BPE 32k                 | 1.260       | 10.310        | 1.022     | 0.469       | 32k     |
| **BPE 50k**             | **1.224**   | **10.608**    | **0.993** | **0.447**   | **50k** |
| BPE 65k                 | 1.208       | 10.751        | 0.980     | 0.434       | 65k     |
| sarvam-1                | 1.471       | 8.829         | 0.000     | 0.452       | 68k`    |
| gemma-4                 | 1.389       | 9.348         | 0.000     | 0.396       | 262k    |


BPE vocab comparison

**50k is the production target.** It matches IndicBERTv2's fertility (fewer splits per Hindi word) while encoding more bytes per token than smaller vocabs. Larger vocabs improve fertility further but at the cost of Rényi efficiency — the vocab becomes dominated by rare tokens.

Config: `configs/tokenizer.yaml`. Eval holdout: `data/eval/hi/`.

---

## Model

The encoder is ported from `_support_repo/ModernBERT/` into `indic-modernBERT/model/modernbert/`.

Key architectural choices:

- **RoPE** (rotary position embeddings) instead of learned absolute positions — generalises better to longer sequences
- **Alternating attention:** every 3rd layer attends to the full sequence (global), other layers use a 128-token sliding window (local). Local attention is O(n) in sequence length instead of O(n²), making 8192-token training feasible
- **Flash Attention:** FA3 on global layers (H100), FA2 on local layers and as fallback on consumer GPUs. Both are memory-fused implementations that avoid materialising the full attention matrix
- **Sequence packing:** training sequences are packed end-to-end into fixed-length tensors with no padding between them, then unpadded inside the model. This wastes almost no compute on padding tokens
- `**init_method: full_megatron`** weight initialisation, scaled by layer depth — prevents activations from exploding in deep networks


| Config file                          | Use                            |
| ------------------------------------ | ------------------------------ |
| `configs/model/modernbert_base.yaml` | 22 layers, production          |
| `configs/model/modernbert_tiny.yaml` | 4 layers, fast GPU smoke tests |


> Note for alternating attention : We are only using FA2 because FA3 support is not available on RTX4090 ( Ampere class ) only available on Hoppers i.e. H100s and also the H100s we have don't have the storage to support 100GBs of memory. 

---

## Training

ModernBERT is trained in **three phases**, progressively extending context length. We rescale durations for our Hindi corpus size.

### Phase 1 — Pretrain at 1024 tokens

The model learns language from scratch. 30% of tokens are masked and the model must predict them. We use:

- **StableAdamW** optimiser: like AdamW but divides the per-parameter learning rate by the gradient RMS, preventing unstable steps when gradients spike early in training
- **WSD (WarmupStableDecay) scheduler:** LR warms up linearly → holds flat → decays at the end
- **Sequence packing:** Hindi sentences are packed into 1024-token windows with no padding waste
- **Global batch 512, microbatch 8** on a single RTX 4090; gradient accumulation handles the rest for much more stable training.

The LR sweep runs 8 Optuna trials over the range 1e-2–1e-6, each for 1000 batches (~524M tokens), and selects the LR minimising eval MLM loss.

### Phase 2 — Context extension to 8192 tokens

Load phase-1 weights and extend the context window by raising `max_seq_len` to 8192 and increasing the global RoPE base from 10,000 to 160,000. The higher RoPE base gives the model a "longer ruler" for position encoding — without it, the model would have no signal for positions it never saw during phase 1. Local (sliding-window) layers keep their original RoPE base since they never attend beyond 128 tokens anyway.

### Phase 3 — LR decay at 8192

Continue from phase 2 with a `1−√` learning rate decay. The model converges its long-context representations.

### Our configs

```bash
# Smoke test (any GPU, quick)
make train-smoke-50ba

# Phase 1 with paper-ratio targets (override max_duration for production run)
uv run python scripts/run_pretrain.py --config-name hindi_mlm_phase1

# Phase 2 (needs a phase-1 checkpoint)
uv run python scripts/run_pretrain.py --config-name hindi_mlm_context_extension
```

### Learning rate sweep

```bash
uv sync --extra pretrain --extra sweep
make lr-sweep          # runs in foreground
make lr-sweep-nohup    # runs in background, logs to logs/lr_sweep/nohup.log
```

8 Optuna trials, log-uniform 3e-5–3e-4, 1000 batches each. Results in `artifacts/model/modernbert/lr_sweep/`; each trial writes `sweep_summary.json` with `eval_loss` and `lr`.

---

## Evaluation

The first checkpoint gate is Hindi-only and runs from a Hugging Face model ID or a local HF export directory. It combines:

- **MLM holdout:** loss and masked accuracy on `data/eval/hi/`
- **Supervised gate:** IndicSentiment, Naamapadam NER, IndicQA, and IndicCOPA
- **Efficiency sweep:** ModernBERT-style inference latency, examples/sec, tokens/sec, tokens/sec per million parameters, CUDA allocated/reserved memory, and optional NVML power readings at 128, 256, 512, and 1024 tokens

Run a full configured suite:

```bash
uv sync --extra evals
make run-evals ARGS="eval.model.model_name_or_path=artifacts/model/modernbert/hf_export"
```

Run a tiny smoke path that caps data and benchmark steps:

```bash
make run-evals-smoke ARGS="eval.model.model_name_or_path=ai4bharat/IndicBERTv2-MLM-only"
```

Hydra config lives at `configs/evals/hindi_phase1.yaml`. Prefer command-line overrides like `eval.model.model_name_or_path=...`, `eval.tasks='[sentiment,ner]'`, `eval.efficiency.sequence_lengths='[128,512]'`, or `eval.efficiency.measure_power=true` rather than editing code. Reports are written under `artifacts/evals/<checkpoint>/` as JSON, CSV, and Markdown.

Retrieval is a separate benchmark because fair ModernBERT-style retrieval numbers require retrieval fine-tuning. Upstream compares backbones by applying the same retrieval recipe, selecting the best checkpoint/hyperparameters, then reporting `nDCG@10`. Use `configs/evals/hindi_retrieval.yaml` for that path:

```bash
uv sync --extra evals
make run-evals-retrieval ARGS="eval.models.0.model_name_or_path=<retrieval-finetuned-checkpoint>"
```

The Hindi retrieval suite separates two signals:

- **Retrieval quality:** `AIhnIndicRag/mmarco_hindi`, a Hindi MS-MARCO-style benchmark.
- **8192-token retrieval capacity:** `Shitao/MLDR` with `language=hi`, a long-document retrieval benchmark aligned with ModernBERT's MLDR long-context framing.

Running retrieval on a raw MLM export is useful only as a smoke test; the score mostly reflects the pooling/indexing choice rather than a fair retriever.

---

## Data

**Training:** `data/sangrah_dataset/verified/hin/*.parquet` (~12.6B Hindi tokens from 19 shards).

**Eval holdout:** `data/eval/hi/` — one shard withheld from training. Create with:

```bash
uv run python -m dataset.sangrah_dataset --count 20 --eval-count 1
```

> We are using `hindi` datasets from sangrah, where we expect the gains from modernBERT also came from new datasets, this might be challenging here. 

**Full corpus available:** Sangraha verified + unverified + synthetic Hindi — ~~23.6B tokens across 274 shards (~~89 GB). Scale up by downloading more shards.

**Scale-up candidates:** Sangraha unverified Hindi, IndicCorp V2/V1.

We use Parquet directly (no conversion to MDS/Mosaic format) — at ~100 GB on local NVMe, Parquet + DataLoader workers is just as fast without the preprocessing overhead. See `docs/LEARNINGS.md §4` for details on garbage collection and dataloader design. 

### DataLoader settings (tuned on RTX 4090)


| Setting                   | Value | Why                                            |
| ------------------------- | ----- | ---------------------------------------------- |
| Train `num_workers`       | **2** | Higher counts OOM'd workers with 8-shard loads |
| Train `prefetch_factor`   | **4** | Best latency/memory tradeoff                   |
| Eval `num_workers`        | **3** | Padded eval path is lighter; 3 workers is safe |
| `packing_prefetch_factor` | **5** | Keeps the sequence packer buffer full          |


These are tuned for the packed training path on a 4090. Re-profile after changing batch size, shard count, or model depth.

---

## Repository layout

```
indic-modernBERT/     main Python package
  config/             Pydantic config schemas
  model/modernbert/   encoder (attention, layers, RoPE, MLP, …)
  pretrain/           training loop, DataLoader, optimizer, scheduler, callbacks
  tokenizer/          BPE trainer, eval metrics
  dataset/            Sangraha downloader

configs/              YAML configs (tokenizer, model, pretrain phases, LR sweep)
scripts/              run_pretrain.py, compare_bpe_vocabs.py, export_hf.py, …
docs/                 LEARNINGS.md (this project's engineering journal), difference.md
artifacts/            tokenizers and checkpoints (gitignored)
notebook/             interactive walkthrough of the full pipeline
_support_repo/        upstream ModernBERT reference (read-only; do not commit blindly)
```

---

## Notes

> **Run everything from the repo root.** The Python package lives in `indic-modernBERT/` and uses flat imports (not pip-installable). Always `cd` to the repo root before running any command.

## Further reading


| Document                      | What's in it                                                                            |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `docs/LEARNINGS.md`           | Engineering journal: every non-obvious decision, bug fix, and "why does this work" note |
| `docs/difference.md`          | Line-by-line comparison of our implementation vs. upstream ModernBERT                   |
| `configs/model/README.md`     | GPU hardware alignment notes (tensor cores, vocab sizing)                               |
| `notebook/pipeline_map.ipynb` | Interactive walkthrough of the full training pipeline                                   |


Paper: [ModernBERT (arxiv:2412.13663)](https://arxiv.org/abs/2412.13663)

### Mistakes :

One of the mistakes I think I did is essentially that only had the holdout / eval set with sangrah but from next time onwards need to mix them so that a good data mix can be constructed. 