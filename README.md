# Indic ModernBERT

[![Model Size](https://img.shields.io/badge/model_size-188M-blue)](#model-summary)
[![Language](https://img.shields.io/badge/language-Hindi-orange)](#hindi)
[![License](https://img.shields.io/badge/license-Apache_2.0-lightgrey)](LICENSE)
[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/raw/main/model-on-hf-md.svg)](https://huggingface.co/kkkamur07/hindi-modernbert)

**Goal:** ModernBERT extension for Indic languages. Prove the recipe on Hindi first, then scale to ModernBERT architectures for more than 5 Indic languages that each have substantial (>5B token) corpora available.

## Hindi

**TL;DR:** hindi-modernBERT is a **pretrained base MLM checkpoint**: a Hindi extension of the ModernBERT architecture, trained from scratch on Hindi text. The base model is competitive with other models across tasks, and it outperforms them on retrieval after DPR fine-tuning. Checkpoint **ba1157**, 22 layers, 8192 context, ~188M params, trained on 1× RTX 4090 in **5 days**.

This release uses the [ModernBERT](https://arxiv.org/abs/2412.13663) architecture and training recipe ([View on alphaXiv](https://www.alphaxiv.org/abs/2412.13663)), adapted for Hindi with a new tokenizer and ~28B tokens of Hindi pretraining.

**Checkpoint folders on the Hub:**

| Hub path | What it contains |
| --- | --- |
| `.` | Main release, ba1157, 8192 context |
| `checkpoints/phase1` | Phase 1 Sangraha checkpoint, 1024 context |
| `checkpoints/phase2_ba135` | Phase 2 lowest MLM-loss checkpoint, 8192 context |

## Model summary

| | |
| --- | --- |
| Type | **Base MLM checkpoint** (pretrained, not task fine-tuned) |
| Architecture | ModernBERT (`ModernBertForMaskedLM`) |
| Initialization | Megatron init (`full_megatron`); pretrained from scratch on Hindi |
| Parameters | ~188M |
| Layers | 22 |
| Hidden size | 768 |
| Attention heads | 12 |
| Vocab size | 50,368 |
| Max sequence length | 8192 |
| Languages | Hindi (`hi`) |
| Pretraining tokens | ~23.6B (Sangraha) + ~4.85B (IndicCorp V2) |
| Hardware | 1× NVIDIA RTX 4090 (24 GB) |
| Training time | **5 days** |
| Transformers | `>=5.12.0` |

---

## Why 8192 tokens

ModernBERT can encode whole documents in one pass instead of truncating to 512 tokens. That matters for Hindi RAG: benchmarks like [MLDR](https://huggingface.co/datasets/Shitao/MLDR) (`language=hi`) rank long passages, not short snippets. Existing Hindi encoders such as IndicBERTv2 and mBERT-family models usually cap at 512 tokens and use older architectures without RoPE or alternating attention.

We port the ModernBERT architecture to Hindi and pretrain it with a new Hindi BPE tokenizer.

The Hindi extension required several pieces to work together:

- **Hindi BPE from scratch:** Devanagari script normalisation, NFKC, regex pre-tokenisation, 50,368-token vocab
- **Two-phase pretraining on a single RTX 4090:** phase 1 at 1024 tokens, phase 2 extends context to 8192 tokens with global RoPE base scaling (10k → 160k)
- **~28.85B Hindi tokens** ingested via a custom Parquet pipeline during the 8192 context pass
- **Downstream + retrieval eval gates:** NER, intent, DPR fine-tune on 1.25M mMARCO Hindi triplets

| Phase | Corpus | Max seq | What happened |
| --- | --- | ---: | --- |
| **Phase 1** | Sangraha Hindi (~23.6B tokens) | 1024 | Language modelling from scratch |
| **Phase 2** | + IndicCorp V2 Hindi (~4.85B tokens) | **8192** | Load phase 1 weights, extend context, continue pretraining |

Production checkpoint: **ba1157**. Full eval write-up: [`artifacts/results/hi/eval_summary_report.md`](artifacts/results/hi/eval_summary_report.md).

Benchmark numbers below fine-tune this **base checkpoint** on [Naamapadam Hindi NER](https://huggingface.co/datasets/ai4bharat/naamapadam) and [MASSIVE Hindi intent](https://huggingface.co/datasets/AmazonScience/massive). Retrieval numbers use a **DPR fine-tuned checkpoint** trained on Hindi mMARCO triples and evaluated on [mmarco_hindi](https://huggingface.co/datasets/AIhnIndicRag/mmarco_hindi) + [MLDR hi](https://huggingface.co/datasets/Shitao/MLDR).

### Phase 1 → phase 2

| Stage | Pretraining | Max seq | NER F1 | MASSIVE Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| Phase 1 | Sangraha Hindi | 1024 | 0.7963 | 0.3451 |
| **hindi-modernBERT** | + IndicCorp V2, 8192 context | **8192** | **0.8001** | **0.4731** |
| Δ | | | +0.0038 | +0.1279 |

### Baseline comparison

| Model | Max seq | NER F1 | MASSIVE Macro-F1 |
| --- | ---: | ---: | ---: |
| mmBERT-small | 8192 | 0.8347 | 0.5462 |
| xlm-roberta-base | 512 | 0.8214 | 0.0743 |
| muril-base-cased | 512 | 0.8148 | 0.0382 |
| IndicBERTv2-MLM-only | 128 | 0.8053 | 0.0821 |
| **hindi-modernBERT** | **8192** | **0.8001** | **0.4731** |

On NER and intent, hindi-modernBERT is competitive with IndicBERTv2 and within reach of larger multilingual encoders. The phase 1 to phase 2 jump (+0.13 macro-F1 on intent) shows the value of the long-context IndicCorp pass; with more pretraining data, we expect this base model to close the remaining supervised-task gap.

### Retrieval

Retrieval uses a DPR fine-tuned checkpoint built from this base checkpoint.

After DPR fine-tuning on 1.25M mMARCO Hindi triplets, hindi-modernBERT outperforms the other Hindi baselines on both full mMARCO Hindi and long-document MLDR hi.

| Model | Max seq | mMARCO nDCG@10 | MLDR hi nDCG@10 |
| --- | ---: | ---: | ---: |
| **hindi-modernBERT** | 8192 | **0.2825** | **0.2635** |
| mmBERT-small | 8192 | 0.2714 | 0.2337 |
| IndicBERTv2-MLM-only | 512 | 0.2821 | 0.1707 |

The 8192-token backbone is what separates hindi-modernBERT on long-document MLDR: IndicBERTv2 is close on short mMARCO but falls behind on MLDR hi because it cannot encode full long documents. These are DPR numbers, one vector per document. Upstream ModernBERT's largest MLDR gains use ColBERT; see `docs/retrieval.md` for the DPR vs ColBERT details.

Build a retriever from this base checkpoint with [`scripts/run_retrieval_finetune.py`](scripts/run_retrieval_finetune.py):

```bash
make retrieval-finetune ARGS="retrieval_ft.backbone=kkkamur07/hindi-modernbert retrieval_ft.max_seq_length=8192"
```

**Full retrieval benchmarks (DPR, 8192 context):**

| Benchmark | What it measures | nDCG@10 | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: |
| Selection (1k mMARCO Hindi) | Small held-out Hindi mMARCO validation split used to select the DPR learning rate/checkpoint. Not the final headline benchmark. | **0.8191** | 0.8987 | 0.7980 |
| [mmarco_hindi](https://huggingface.co/datasets/AIhnIndicRag/mmarco_hindi) (full) | Full Hindi MS MARCO-style passage retrieval benchmark. Tests standard dense retrieval quality on Hindi queries and passages. | **0.2825** | 0.4398 | 0.2368 |
| [MLDR hi](https://huggingface.co/datasets/Shitao/MLDR) (8192 ctx) | Hindi long-document retrieval benchmark. Tests whether the 8192-token context helps retrieve long documents. | **0.2635** | 0.3900 | 0.2252 |

Metric glossary: `nDCG@10` measures ranking quality in the top 10, `Recall@10` measures whether relevant passages appear in the top 10, and `MRR@10` measures how high the first relevant result appears.

---

## Quick start

Main release:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "kkkamur07/hindi-modernbert"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

text = "भारत [MASK] विशाल देश है।"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
print(tokenizer.decode([logits[0, mask_idx].argmax(dim=-1).item()]))
```

Checkpoint subfolder:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

repo_id = "kkkamur07/hindi-modernbert"
subfolder = "checkpoints/phase1"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
model = AutoModelForMaskedLM.from_pretrained(repo_id, subfolder=subfolder)
```

Apply Devanagari script normalisation + NFKC before tokenization for best results. See [Tokenizer](#tokenizer) for `preprocess_for_tokenizer()`.

For dense retrieval, fine-tune this base checkpoint with the DPR recipe on Hindi mMARCO triples.

---

## What is ModernBERT?

BERT-style encoders read text bidirectionally (every token attends to every other token) and are trained with Masked Language Modelling (MLM): randomly mask ~30% of tokens, predict them. This makes them excellent at understanding tasks: classification, NER, retrieval, question answering. They do not generate text.

ModernBERT is a 2024 redesign of BERT that incorporates everything learned from the GPT generation: rotary position embeddings (RoPE) for long contexts, Flash Attention for memory-efficient training, alternating global+local attention layers for efficiency, and a much larger training corpus. It achieves state-of-the-art results on retrieval benchmarks while remaining fast to fine-tune.

We port it to Hindi by training a new tokenizer and running the full pretraining recipe above, not a lightweight fine-tune of the English checkpoint.

ModernBERT focuses on two things:

1. **Efficiency:** designing the model for the hardware
2. **Long-context retrieval:** extended context length from RoPE and alternating attention

---

## Tokenizer

We train a BPE tokenizer directly on Hindi text. The pipeline is:

```
raw text → script normalisation (indic-nlp) → NFKC → regex pre-tokenise → BPE
```

Yes, each step is necessary: skip one and the tokenizer silently degrades.

**Script normalisation** is the most critical step for Devanagari. The same word can be encoded in multiple byte-level ways that look pixel-identical on screen but are different strings to a computer. For example, the Hindi word for "Kumar" (कुमार) can be written with the vowel sign `ु` composed into the consonant, or as a separate combining character: same glyph, different bytes. Without normalisation, BPE treats these as two different words and wastes two vocabulary slots on the same meaning. Multiply this across thousands of common words and you lose a huge chunk of your 50k vocabulary to duplicates.

```
# Same word, two byte sequences: both look like "कुमार" on screen
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

**Regex pre-tokeniser** draws hard boundaries before BPE merges begin. It splits on whitespace and punctuation, so BPE can never create a token that straddles a word boundary. Without it you'd get tokens like `"है।"` (word + full-stop fused together): useful in some contexts but inconsistent, and it bloats the vocabulary with punctuation-suffixed variants of every common word.

**Target vocab size: 50,368**, same as upstream ModernBERT. This is divisible by 64 (required for GPU tensor-core alignment) and matches the upstream embedding weight dimensions exactly, which simplifies initialising from pretrained weights later.

Always use `preprocess_for_tokenizer()` at inference time. Training and inference must go through the same normalisation pipeline or the same Hindi word will tokenize differently at inference than it did during training.

### Tokenizer eval results

Latest run on the Hindi holdout (174k rows):

| Tokenizer | Fertility ↓ | Bytes/token ↑ | NSL ↓ | Rényi eff ↑ | Vocab |
| --- | ---: | ---: | ---: | ---: | ---: |
| IndicBERTv2 (reference) | 1.233 | 10.534 | 0.000 | 0.380 | 250k |
| BPE 32k | 1.260 | 10.310 | 1.022 | 0.469 | 32k |
| **BPE 50k** | **1.224** | **10.608** | **0.993** | **0.447** | **50k** |
| BPE 65k | 1.208 | 10.751 | 0.980 | 0.434 | 65k |
| sarvam-1 | 1.471 | 8.829 | 0.000 | 0.452 | 68k |
| gemma-4 | 1.389 | 9.348 | 0.000 | 0.396 | 262k |

**50k is the production target.** It matches IndicBERTv2's fertility (fewer splits per Hindi word) while encoding more bytes per token than smaller vocabs. Larger vocabs improve fertility further but at the cost of Rényi efficiency: the vocab becomes dominated by rare tokens.

Config: `configs/hi/tokenizer.yaml`. Eval holdout: `data/eval/hi/`. **SuperBPE extension pending.**

---

## Model

Key architectural choices:

- **RoPE** (rotary position embeddings) instead of learned absolute positions: generalises better to longer sequences
- **Alternating attention:** every 3rd layer attends to the full sequence (global), other layers use a 128-token sliding window (local). Local attention is O(n) in sequence length instead of O(n²), making 8192-token training feasible
- **Flash Attention:** FA3 on global layers (H100), FA2 on local layers and as fallback on consumer GPUs. Both are memory-fused implementations that avoid materialising the full attention matrix
- **Sequence packing:** training sequences are packed end-to-end into fixed-length tensors with no padding between them, then unpadded inside the model. This wastes almost no compute on padding tokens
- **`init_method: full_megatron`** weight initialisation, scaled by layer depth: prevents activations from exploding in deep networks

| Config file | Use |
| --- | --- |
| `configs/model/modernbert_base.yaml` | 22 layers, production |
| `configs/model/modernbert_tiny.yaml` | 4 layers, fast GPU smoke tests |

> **Note on alternating attention:** we only use FA2 in production because FA3 is not available on RTX 4090 (Ampere). FA3 requires Hopper-class GPUs (H100). The H100s we have access to also lack the storage headroom for 100GB-scale training runs.

### Eval summary

hindi-modernBERT is competitive on supervised Hindi understanding tasks and is the strongest model in this comparison on retrieval after DPR fine-tuning.

| Area | Benchmark | Score |
| --- | --- | ---: |
| NER | Naamapadam Hindi F1 | 0.8001 |
| Intent | MASSIVE Hindi Macro-F1 | 0.4731 |
| Retrieval | mMARCO Hindi nDCG@10 | **0.2825** |
| Retrieval | MLDR hi nDCG@10 | **0.2635** |

---

## Training

ModernBERT is trained in phases that progressively extend context length. We rescale durations for our Hindi corpus size.

### Phase 1: pretrain at 1024 tokens

The model learns language from scratch. 30% of tokens are masked and the model must predict them. We use:

- **StableAdamW** optimiser: like AdamW but divides the per-parameter learning rate by the gradient RMS, preventing unstable steps when gradients spike early in training
- **WSD (WarmupStableDecay) scheduler:** LR warms up linearly, holds flat, then decays at the end
- **Sequence packing:** Hindi sentences are packed into 1024-token windows with no padding waste
- **Global batch 512, microbatch 8** on a single RTX 4090; gradient accumulation handles the rest for much more stable training

The LR sweep runs 8 Optuna trials over the range 1e-2 to 1e-6, each for 1000 batches (~524M tokens), and selects the LR minimising eval MLM loss (`4.884×10⁻⁴` for production).

### Phase 2: context extension to 8192 tokens

Load phase 1 weights and extend the context window by raising `max_seq_len` to 8192 and increasing the global RoPE base from 10,000 to 160,000. The higher RoPE base gives the model a longer ruler for position encoding: without it, the model would have no signal for positions it never saw during phase 1. Local (sliding-window) layers keep their original RoPE base since they never attend beyond 128 tokens anyway. Continue pretraining on IndicCorp V2 Hindi.

Configs live under `configs/hi/pretrain/` (`hindi_mlm_phase1.yaml`, `hindi_mlm_context_extension.yaml`). LR sweep config: `configs/hi/sweep/hindi_mlm_lr_sweep.yaml`.

```bash
# Smoke test (any GPU, quick)
make train-smoke-50ba

# Phase 1
uv sync --extra pretrain && make train-phase1

# Phase 2 (needs a phase 1 checkpoint)
make train-phase2
```

---

## Evaluation

Supervised numbers fine-tune this **base checkpoint** on Naamapadam NER and MASSIVE Hindi intent. Retrieval numbers use a **DPR fine-tuned checkpoint** on mMARCO Hindi triples, evaluated on `mmarco_hindi` + `mldr_hi`.

The downstream gate also covers IndicSentiment, IndicQA, and IndicCOPA. Configs live under `configs/hi/evals/`. See `docs/retrieval.md` for DPR vs ColBERT details.

---

## Data

**Training:** `data/sangrah_dataset/verified/hin/*.parquet` (~12.6B Hindi tokens from 19 shards).

**Eval holdout:** `data/eval/hi/`, one shard withheld from training.

**Full corpus:** Sangraha verified + unverified + synthetic Hindi, ~23.6B tokens across 274 shards. Scale-up candidates: IndicCorp V2/V1.

**Evaluation + retrieval:** [Naamapadam Hindi NER](https://huggingface.co/datasets/ai4bharat/naamapadam), [MASSIVE Hindi intent](https://huggingface.co/datasets/AmazonScience/massive), [mmarco_hindi](https://huggingface.co/datasets/AIhnIndicRag/mmarco_hindi), and [MLDR hi](https://huggingface.co/datasets/Shitao/MLDR).

We use Parquet directly (no MDS conversion). See `docs/learnings.md` for dataloader tuning notes.

---

## Repository layout

```
indic-modernBERT/     main Python package
configs/hi/           Hindi tokenizer, pretrain, eval, retrieval configs
scripts/              run_pretrain.py, export_hf.py, run_evals.py, …
artifacts/            tokenizers and checkpoints (gitignored)
docs/                 learnings.md, retrieval.md, repo_layout.md
notebook/             interactive pipeline walkthrough
```

> Run everything from the repo root. The package uses flat imports, so `cd` to the repo root before running commands.

---

## Commands

| What | Command |
| --- | --- |
| Train BPE tokenizer | `make train-bpe` |
| Eval tokenizer | `make eval-bpe` |
| Download Sangraha shards | `uv run python -m dataset.sangrah_dataset --count 20 --eval-count 1` |
| LR sweep | `uv sync --extra pretrain --extra sweep && make lr-sweep` |
| Phase 1 pretrain | `uv sync --extra pretrain && make train-phase1` |
| Phase 2 context extension | `make train-phase2` |
| Phase 2 VRAM smoke | `make train-smoke-phase2` |
| Export checkpoint to HF format | `make export-hf ARGS="ckpt.pt out/ --tokenizer artifacts/tokenizer/hi/bpe_vs50368"` |
| Upload to Hub | `make upload-hf-mlm HF_REPO_ID_MLM=kkkamur07/hindi-modernbert` |
| Run downstream evals | `uv sync --extra evals && make run-evals-phase2 ARGS="eval.model.model_name_or_path=<path-or-hub-id>"` |
| Run retrieval evals | `make run-evals-retrieval ARGS="eval.models.0.model_name_or_path=<retriever-checkpoint>"` |
| DPR fine-tune | `make retrieval-finetune ARGS="retrieval_ft.backbone=<backbone>"` |
| Eval comparison reports | `make eval-comparison-reports` |
| Pipeline notebook | open `notebook/pipeline_map.ipynb` |

Artifacts: `artifacts/tokenizer/hi/bpe_vs{V}/`, `artifacts/model/modernbert/hi/`, `logs/hi/`.

---

## Further reading

| Document | Contents |
| --- | --- |
| `docs/learnings.md` | Engineering journal |
| `docs/retrieval.md` | DPR vs ColBERT, retrieval eval setup |
| `docs/repo_layout.md` | Language-scoped artifacts and logs |
| `artifacts/results/hi/eval_summary_report.md` | Combined baseline + phase 1/2 + retrieval results |

Paper: [ModernBERT (arxiv:2412.13663)](https://arxiv.org/abs/2412.13663)

---

## Limitations

- **Hindi-only pretraining:** other Indic languages are not in scope for this release.
- **Tokenizer preprocessing:** Devanagari script normalisation is required for best results and is not applied automatically by `AutoTokenizer`.
- **Not a retriever by itself:** dense retrieval requires DPR fine-tuning on top of this base checkpoint.
- **Single-seed evals:** downstream and retrieval benchmarks use one fine-tuning seed; multi-seed averaging may shift scores slightly.
- **Phase 1 holdout:** MLM eval holdout was Sangraha-only; future runs may mix holdout sources for a more representative training signal.

## Acknowledgments

- **[AI4Bharat](https://ai4bharat.org/):** Sangraha and IndicCorp V2 corpora; Naamapadam NER eval
- **Amazon Science:** [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) Hindi intent
- **AIhnIndicRag:** [mmarco_hindi](https://huggingface.co/datasets/AIhnIndicRag/mmarco_hindi)
- **Answer.AI / ModernBERT team:** architecture ([arxiv:2412.13663](https://arxiv.org/abs/2412.13663))

## Citation

```bibtex
@misc{hindi-modernbert2026,
  title  = {hindi-modernBERT: A Hindi ModernBERT Encoder with 8192 Context},
  author = {Krrish Agarwalla},
  year   = {2026},
  note   = {Checkpoint ba1157. Base MLM; trained from scratch on Hindi.}
}
```

```bibtex
@article{modernbert2024,
  title  = {Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
  author = {Warner, Benjamin and Chizhov, Anton and Ermolaev, Alexander and others},
  journal = {arXiv preprint arXiv:2412.13663},
  year   = {2024}
}
```
