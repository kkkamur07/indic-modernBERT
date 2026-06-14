# Pipeline notebook

**[`pipeline_map.ipynb`](pipeline_map.ipynb)** is the interactive pipeline guide.

Open with kernel cwd = **repo root** or **`notebook/`** — repo is auto-detected. Run cells **top to bottom**.

## Structure

Each pipeline stage is two cells:

1. **Markdown** — what the stage does in production, which configs/files apply
2. **Code** — runs `validate_*()` for that stage and prints ✓ with asserted outputs

**Pipeline steps** (each step = markdown cell + code cell):

- **0 — Environment** — CUDA + flash-attn
- **1 — Configs** — YAML extends, Hydra job, Pydantic validation, dataloader settings
- **2 — Data** — pretokenization + parquet + production DataLoaders (raw / packed / eval)
- **3 — Tokenizer** — `make train-bpe`
- **4 — GPU batch** — `position_ids`, bf16 autocast
- **5 — Model smoke** — 22L `modernbert_base` forward + MLM loss
- **6 — Attention** — 22L layer pattern + backward at production seq len
- **7 — Production training step** — 22L base + production eval + packed dataloaders, forward/eval/backward
- **8 — E2E trace** — `make pipeline-trace` (22L, `max_seq_len=1024`)
- **9 — Full pretrain** — 3-step packed train loop (forward/backward/optim) + eval; production: `make train-smoke-50ba`

Helpers: [`pipeline_steps.py`](pipeline_steps.py) (`detect_repo`, `ensure_src_on_path`, `validate_*`).

## Requirements

- Steps 3–9 need `artifacts/tokenizer/bpe_vs50368/`
- Steps 2, 7–9 need `data/sangrah_dataset/` parquet and tokenizer artifact
- Steps 5–9 need GPU + `flash-attn` for sliding-window attention
- Step 9 needs `uv sync --extra pretrain` (Composer)

Missing optional artifacts print **⊘ SKIPPED** instead of failing the notebook.
