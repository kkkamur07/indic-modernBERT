# Multilingual Run Layout

The project started as Hindi-first, but generated outputs should now be grouped by
language code so additional languages do not mix artifacts, eval reports, or logs.

Use ISO language codes as the first language-specific directory level:

```text
configs/
  hi/
    tokenizer.yaml
    pretrain/
    evals/
    retrieval_finetune/
    sweep/
  model/              # shared architecture configs

artifacts/
  tokenizer/<lang>/
    bpe_vs50368/
  model/modernbert/<lang>/
    checkpoints/
    hf_export/
    tensorboard/
    lr_sweep/
  evals/<lang>/
    phase1/
    phase2/
    retrieval/
    transfer/
  results/<lang>/
    eval_summary_report.md
  retrieval_finetune/<lang>/
    raw/
    subsets/
    full_local_jsonl_train_eval_runs/
  corpus_stats/<lang>/

logs/<lang>/
  tokenizer/
  pretrain/
  evals/
  retrieval/
```

For Hindi, `<lang>` is `hi`. Historical runs may still exist under the older
top-level paths such as `artifacts/evals_phase2/` or `logs/2026-..._evals/`.
Treat those as archived outputs; new configs and Make targets should write to
the language-scoped paths above.

When adding another language, create configs by copying the closest Hindi config,
then update dataset inputs, model names, and every output/log path to use the new
language code. Shared model architecture YAMLs stay under `configs/model/`.

Per-model eval artifacts stay under `artifacts/evals/<lang>/`. Combined
cross-model comparison reports are written to `artifacts/results/<lang>/` via
`make eval-comparison-reports`.
