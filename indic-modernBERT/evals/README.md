# Hindi Evaluation Suite

This package contains the first Hindi-only evaluation suite for phase-1 Indic ModernBERT checkpoints. It is meant to answer a practical checkpoint question: is the model learning useful Hindi representations yet?

The suite is Hydra-driven through `configs/evals/hindi_phase1.yaml`. Change `eval.model.model_name_or_path` to point at any Hugging Face Hub model ID or local HF export directory, then run the same pipeline against that model.

## What It Runs

The runner combines three evaluation layers:

- **MLM holdout**: evaluates masked-language-model loss and masked accuracy on `data/eval/hi`.
- **Supervised Hindi gate**: fine-tunes and evaluates one representative task per major downstream task type.
- **Efficiency sweep**: measures inference behavior over Hindi inputs at `128`, `256`, `512`, and `1024` tokens.

The supervised gate starts with:

- **IndicSentiment** for sentence classification.
- **Naamapadam NER** for token classification / named entity recognition.
- **IndicQA** for question answering.
- **IndicCOPA** for multiple-choice commonsense reasoning.

Retrieval is intentionally deferred for now. During phase 1, the model has only MLM pretraining, so retrieval scores would mostly reflect an untuned pooling/indexing choice rather than the mature retrieval strength ModernBERT is known for.

## Why This Shape

The suite is intentionally small and repeatable. A phase-1 checkpoint may change often, so the first gate should be cheap enough to run repeatedly but broad enough to catch different failure modes:

- MLM holdout tracks whether pretraining itself is improving.
- Sentiment checks sentence-level semantics.
- NER checks token-level Hindi representations and subword boundaries.
- QA checks whether the model can connect a question to evidence in context.
- COPA checks sentence-pair reasoning and commonsense plausibility.
- Efficiency sweep checks whether the architecture is preserving ModernBERT's practical speed and memory advantages as sequence length grows.

Everything is config-first because model comparison should not require code edits. The intended workflow is to swap `eval.model.model_name_or_path`, choose task subsets with Hydra overrides, and compare outputs under `artifacts/evals/`.

## Inspirations

This package borrows from two local references:

- `_support_repo/IndicBERT`: task selection, Hindi/Indic benchmark framing, dataset choices, and metrics. Its `fine-tuning/` scripts and `eval.sh` show how IndicBERT evaluates IndicSentiment, Naamapadam, IndicQA, IndicCOPA, XNLI, paraphrase, and retrieval.
- `_support_repo/ModernBERT`: evaluation orchestration and efficiency measurement style. `RunEvals.md` motivates config-driven fine-tuning runs across tasks/seeds, while `benchmark.py` motivates reporting latency, tokens/sec, tokens/sec per million parameters, memory, and optional GPU power.

The result is not a direct copy of either repo. It keeps the IndicBERT task families, but wraps them in this repo's Hydra workflow and adds ModernBERT-style efficiency reporting.

## How To Run

Full configured suite:

```bash
uv sync --extra evals
make run-evals ARGS="eval.model.model_name_or_path=<hf-id-or-local-hf-export>"
```

Tiny smoke path:

```bash
make run-evals-smoke ARGS="eval.model.model_name_or_path=hf-internal-testing/tiny-random-bert"
```

Useful overrides:

```bash
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.tasks='[sentiment,ner]'"
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.efficiency.sequence_lengths='[128,1024]'"
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.efficiency.measure_power=true"
```

Outputs are written under `artifacts/evals/<model-slug>/`:

- `suite_summary.json`
- `suite_metrics.csv`
- `suite_report.md`
- per-task supervised metrics
- `efficiency_metrics.json`

