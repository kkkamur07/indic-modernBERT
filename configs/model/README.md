# Model architecture configs

Only **`modernbert_base.yaml`** is canonical (22 layers, upstream ModernBERT phase-1).

Other files are **thin overrides** via `extends:` — not duplicate full configs.

| File | Purpose |
|------|---------|
| `modernbert_base.yaml` | Phase-1 pretrain (`hindi_mlm.yaml`) — RoPE θ=10k, `compile_model: true` |
| `modernbert_context_extension.yaml` | Phase-2 @ 8192 — global RoPE θ=160k, local θ=10k |
| `modernbert_tiny.yaml` | 4-layer smoke (`compile_model: false`) |

Pretrain Hydra configs:

| File | Phase |
|------|-------|
| `configs/pretrain/hindi_mlm.yaml` | Default smoke / dev |
| `configs/pretrain/hindi_mlm_phase1.yaml` | Seq 1024, base arch |
| `configs/pretrain/hindi_mlm_context_extension.yaml` | Seq 8192, extension arch |

Hardware alignment is validated by `compute_hardware_alignment()` / `validate_hardware_alignment()` when enabled in the model config.

Kernel/runtime checks are covered by the 50-batch smoke train (`make train-smoke-50ba`) with CUDA + `flash-attn` from `uv sync --extra pretrain`:

| Target | What it checks |
|--------|----------------|
| Attention | FA2 varlen attention + unpadded RoPE forward |
| Routing | Per-layer FA3/FA2 routing vs GPU (4090 → FA2 all layers; H100+hopper → FA3 global) |
| Compile | `torch.compile` on embeddings, MLP, LM head (+ backward) |

**Smoke train:** `make train-smoke-50ba` — 50 batches, inherits `hindi_mlm_phase1` (FA2, `compile_model`, packing, TensorBoard, production callbacks).

Loader: `load_modernbert_arch_config()` deep-merges `model_config` over the parent.
