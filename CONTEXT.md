# Hindi Tokenizer

Building and evaluating a Hindi text tokenizer (BPE and SuperBPE) for a ModernBERT-style model.

## Language

**Normalization**:
Unicode-level cleanup applied to the full input string before it is split.
_Avoid_: Pre-tokenization, regex splitting

**Script normalization**:
Devanagari-specific cleanup that collapses script encoding variants (e.g. ZWJ conjuncts) before Unicode normalization.
_Avoid_: NFKC, pre-tokenization

**Pre-tokenization**:
Regex-based splitting that defines the chunks BPE merge operations cannot cross.
_Avoid_: Normalization, tokenization

**Subword stage**:
The strict pre-tokenization mode used for standard BPE and for SuperBPE phase 1.
_Avoid_: Stage 1 (use "subword stage" when talking about pre-tokenization specifically)

**Superword stage**:
A relaxed pre-tokenization mode used in SuperBPE phase 2, allowing merges across former word boundaries.
_Avoid_: Stage 2 (ambiguous outside SuperBPE context)

**Semantic unit**:
A pre-tokenization chunk that corresponds to a meaningful text fragment — a word, punctuation mark, digit group, or whitespace run.
_Avoid_: Token (tokens are produced later by BPE)

**Downstream evaluation**:
Fine-tuning or probing an exported encoder on task datasets after MLM pretraining.
_Avoid_: Pretraining eval, tokenizer eval

**Supervised gate**:
The first Hindi-only downstream checkpoint check: IndicSentiment, Naamapadam NER, IndicQA, and IndicCOPA.
_Avoid_: Final benchmark, retrieval benchmark

**MLM holdout**:
Masked-language-model loss and masked-token accuracy on the withheld Hindi parquet shard under `data/eval/hi/`.
_Avoid_: Training loss

**Efficiency sweep**:
Inference-only length sweep over fixed Hindi inputs that reports latency, throughput, and CUDA peak memory.
_Avoid_: Training speed, dataloader speed

**Retrieval evaluation**:
Ranking/query-document evaluation for retrieval-tuned encoders. It is deferred until context-extension or retrieval-specific fine-tuning exists.
_Avoid_: QA, MLM holdout

## Relationships

- **Script normalization** runs first, then **Normalization** (NFKC), then **Pre-tokenization**
- **Pre-tokenization** produces **Semantic units** that constrain BPE merging
- **Subword stage** pre-tokenization is used by BPE and by the first phase of SuperBPE
- **Superword stage** pre-tokenization replaces **Subword stage** during the second phase of SuperBPE
- **MLM holdout**, **Supervised gate**, and **Efficiency sweep** form the phase-1 checkpoint evaluation suite
- **Retrieval evaluation** is a later phase and should not be used as the first Hindi checkpoint gate

## Example dialogue

> **Dev:** "Should NFKC run before or after the regex split?"
> **Domain expert:** "After **Script normalization**, before **Pre-tokenization** — script rules stabilize Devanagari forms, NFKC handles Unicode compatibility, then regex splits into **Semantic units**."

## Flagged ambiguities

- "Regex normalization" was used to mean **Pre-tokenization** — resolved: NFKC is **Normalization**; regex splitting is **Pre-tokenization**.
- For any problems you can evaluate modernBERT repo in `_support_repo` to understand better. 

