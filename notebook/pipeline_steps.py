"""Helpers for notebook/pipeline_map.ipynb — one validate_* function per pipeline stage."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


class StageSkip(Exception):
    """Raised when a stage cannot run (missing artifact); notebook should print and continue."""


def detect_repo(start: Path | None = None) -> Path:
    """Repo root = directory that contains both indic-modernBERT/ (src) and configs/."""
    cwd = Path(start) if start is not None else Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "indic-modernBERT").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return cwd.resolve()


def ensure_src_on_path(root: Path | None = None) -> Path:
    import os

    repo = detect_repo(root)
    os.chdir(repo)
    src = str((repo / "indic-modernBERT").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return repo


@dataclass
class PipelineContext:
    repo: Path
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    tokenizer: PreTrainedTokenizerFast | None = None
    pretrain_cfg: Any = None
    production_cfg: Any = None
    mlm_model: Any = None

    @classmethod
    def detect(cls, repo: Path | None = None) -> PipelineContext:
        root = ensure_src_on_path(repo)
        from pretrain.gpu_batch import resolve_device

        return cls(repo=root, device=resolve_device())

    @property
    def tokenizer_path(self) -> Path:
        return self.repo / "artifacts/tokenizer/bpe_vs50368"

    @property
    def train_data(self) -> Path:
        return self.repo / "data/sangrah_dataset"

    @property
    def eval_data(self) -> Path:
        return self.repo / "data/eval/hi"

    @property
    def arch_base(self) -> Path:
        return self.repo / "configs/model/modernbert_base.yaml"

    @property
    def pretrain_config_dir(self) -> Path:
        return self.repo / "configs/pretrain"

    def has_tokenizer(self) -> bool:
        return (self.tokenizer_path / "tokenizer.json").is_file()

    def has_train_parquet(self) -> bool:
        return self.train_data.exists() and bool(list(self.train_data.rglob("*.parquet")))

    def has_eval_parquet(self) -> bool:
        return self.eval_data.exists() and bool(list(self.eval_data.rglob("*.parquet")))


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_tokenizer(ctx: PipelineContext) -> PreTrainedTokenizerFast:
    if ctx.tokenizer is not None:
        return ctx.tokenizer
    if not ctx.has_tokenizer():
        raise StageSkip(f"tokenizer not found at {ctx.tokenizer_path} — run: make train-bpe")
    from utils.paths import resolve_hf_tokenizer_dir

    tok_dir = resolve_hf_tokenizer_dir(ctx.tokenizer_path)
    ctx.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    return ctx.tokenizer


def _sync_device(ctx: PipelineContext) -> torch.device:
    """Refresh device (notebook kernels can desync after DataLoader workers)."""
    from pretrain.gpu_batch import resolve_device

    ctx.device = resolve_device()
    return ctx.device


def _prepare_batch(batch: dict, device: torch.device) -> dict:
    from pretrain.gpu_batch import move_batch_to_device

    prepared = move_batch_to_device(batch, device)
    for key, value in prepared.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device, non_blocking=False)
    return prepared


def _decode_ids_sample(
    tokenizer: PreTrainedTokenizerFast,
    ids: list[int] | torch.Tensor,
    *,
    max_tokens: int = 32,
) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    text = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    return text[:140].replace("\n", " ").strip()


def _ids_list(ids: list[int] | torch.Tensor) -> list[int]:
    return ids.tolist() if isinstance(ids, torch.Tensor) else ids


def _mlm_mask_preview(
    tokenizer: PreTrainedTokenizerFast,
    input_ids: list[int],
    labels: list[int],
    *,
    max_pairs: int = 4,
) -> str:
    pairs: list[str] = []
    for idx, label in enumerate(labels):
        if label == -100:
            continue
        inp = tokenizer.decode([input_ids[idx]])
        tgt = tokenizer.decode([label])
        pairs.append(f"{inp!r}->{tgt!r}")
        if len(pairs) >= max_pairs:
            break
    return ", ".join(pairs) or "(none — MLM is stochastic; rerun cell)"


def _summarize_raw_row(
    tokenizer: PreTrainedTokenizerFast,
    item: dict[str, list[int]],
) -> dict[str, str]:
    input_ids = item["input_ids"]
    attention_mask = item.get("attention_mask")
    real = (
        int(sum(attention_mask))
        if attention_mask is not None
        else len(input_ids) - int(sum(1 for i in input_ids if i == tokenizer.pad_token_id))
    )
    return {
        "keys": ", ".join(sorted(item.keys())),
        "seq_len": str(len(input_ids)),
        "real_tokens": str(real),
        "decoded": _decode_ids_sample(tokenizer, input_ids),
        "note": "no labels yet — TokenizeCollator only; MLM masking happens in packer",
    }


def _summarize_mlm_row(
    tokenizer: PreTrainedTokenizerFast,
    batch: dict[str, torch.Tensor],
    row: int = 0,
) -> dict[str, str]:
    input_ids = _ids_list(batch["input_ids"][row])
    labels = _ids_list(batch["labels"][row])
    attention_mask = batch.get("attention_mask")
    mask_row = _ids_list(attention_mask[row]) if attention_mask is not None else None

    if mask_row is not None:
        real_tokens = int(sum(mask_row))
        pad_tokens = len(mask_row) - real_tokens
    else:
        pad_tokens = int(sum(1 for i in input_ids if i == tokenizer.pad_token_id))
        real_tokens = len(input_ids) - pad_tokens

    masked = int(sum(1 for label in labels if label != -100))
    return {
        "keys": ", ".join(sorted(batch.keys())),
        "seq_len": str(len(input_ids)),
        "real_tokens": str(real_tokens),
        "pad_tokens": str(pad_tokens),
        "masked_tokens": str(masked),
        "decoded_input": _decode_ids_sample(tokenizer, input_ids),
        "masked_preview": _mlm_mask_preview(tokenizer, input_ids, labels),
    }


def _mlm_batch(
    ctx: PipelineContext,
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
    *,
    max_seq_len: int,
    mlm_probability: float = 0.3,
) -> dict[str, torch.Tensor]:
    from pretrain.gpu_batch import move_batch_to_device

    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_len)
    examples = [{k: enc[k][i] for k in enc} for i in range(len(texts))]
    batch = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )(examples)
    return _prepare_batch(batch, _sync_device(ctx))


def _mlm_model(ctx: PipelineContext, arch_path: Path):
    """Build production FlexBERT MLM (modernbert_base) with tokenizer vocab."""
    from config import load_modernbert_arch_config
    from model.factory import create_modernbert_mlm

    arch = load_modernbert_arch_config(arch_path)
    tokenizer = _ensure_tokenizer(ctx)
    if arch.vocab_size != tokenizer.vocab_size:
        arch = arch.model_copy(update={"vocab_size": tokenizer.vocab_size})
    model = create_modernbert_mlm(
        pretrained_model_name="bert-base-uncased",
        model_config=arch,
        tokenizer_path=str(ctx.tokenizer_path),
        gradient_checkpointing=False,
        disable_train_metrics=True,
    ).model
    device = _sync_device(ctx)
    model.to(device)
    return model, tokenizer, arch


def _load_production_cfg(ctx: PipelineContext):
    """Production notebook path: hindi_mlm_smoke_50ba (same stack as phase-1, 50-batch duration)."""
    from config import load_pretrain_config

    if ctx.production_cfg is not None:
        return ctx.production_cfg

    with initialize_config_dir(config_dir=str(ctx.pretrain_config_dir), version_base=None):
        cfg = load_pretrain_config(compose(config_name="hindi_mlm_smoke_50ba"))
    if cfg.max_train_shards is None:
        cfg = cfg.model_copy(update={"max_train_shards": 2})
    ctx.production_cfg = cfg
    return cfg


def _production_microbatch(batch: dict[str, torch.Tensor], micro: int) -> dict[str, torch.Tensor]:
    """Slice a production dataloader batch to device microbatch size (Composer-style)."""
    return {key: value[:micro] for key, value in batch.items()}


def _validate_production_dataloaders(ctx: PipelineContext, cfg) -> dict[str, str]:
    """Raw, packed-train, and eval DataLoaders with production num_workers / prefetch."""
    from composer.utils import dist
    from pretrain.dataloader import (
        DistributedSamplerPCG64DXSM,
        _dataloader_kwargs,
        build_eval_dataloader,
        build_parquet_train_dataloader,
    )
    from pretrain.parquet_mlm import ParquetMLMDataset, TokenizeCollator
    from torch.utils.data import DataLoader

    tokenizer = _ensure_tokenizer(ctx)
    device = _sync_device(ctx)
    device_batch_size = cfg.global_train_batch_size // dist.get_world_size()

    dataset = ParquetMLMDataset(
        cfg.data_root,
        cfg.text_column,
        max_shards=cfg.max_train_shards,
    )
    collator = TokenizeCollator(tokenizer, max_seq_len=cfg.max_seq_len)
    sampler = DistributedSamplerPCG64DXSM(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=True,
        seed=cfg.shuffle_seed,
        drop_last=cfg.drop_last,
    )
    raw_kwargs = _dataloader_kwargs(
        cfg,
        device,
        batch_size=device_batch_size,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )
    raw_loader = DataLoader(dataset, collate_fn=collator, **raw_kwargs)
    raw_batch = next(iter(raw_loader))
    assert raw_batch and "input_ids" in raw_batch[0]

    packed = build_parquet_train_dataloader(
        cfg,
        tokenizer,
        device,
        device_batch_size=device_batch_size,
    )
    packed_batch = next(iter(packed))
    assert "input_ids" in packed_batch

    eval_loader = build_eval_dataloader(cfg, tokenizer, device)
    assert eval_loader.num_workers == cfg.eval_num_workers
    eval_batch = next(iter(eval_loader))
    assert "input_ids" in eval_batch and "labels" in eval_batch and "attention_mask" in eval_batch

    raw_row = _summarize_raw_row(tokenizer, raw_batch[0])
    eval_row = _summarize_mlm_row(tokenizer, eval_batch, row=0)
    packed_row = _summarize_mlm_row(tokenizer, packed_batch, row=0)

    return {
        "pretrain_config": "hindi_mlm_smoke_50ba",
        "train_workers": str(raw_loader.num_workers),
        "train_prefetch": str(raw_kwargs.get("prefetch_factor", "n/a")),
        "persistent_workers": str(raw_kwargs.get("persistent_workers", False)),
        "eval_workers": str(eval_loader.num_workers),
        "packing_prefetch": str(cfg.packing_prefetch_factor),
        "raw_batch": f"n={len(raw_batch)} first_len={len(raw_batch[0]['input_ids'])}",
        "packed_batch": f"shape={tuple(packed_batch['input_ids'].shape)}",
        "eval_batch": f"shape={tuple(eval_batch['input_ids'].shape)}",
        "raw_keys": raw_row["keys"],
        "raw_tokens": f"len={raw_row['seq_len']} real={raw_row['real_tokens']}",
        "raw_sample": raw_row["decoded"],
        "eval_keys": eval_row["keys"],
        "eval_tokens": (
            f"len={eval_row['seq_len']} real={eval_row['real_tokens']} "
            f"pad={eval_row['pad_tokens']} masked={eval_row['masked_tokens']}"
        ),
        "eval_sample": eval_row["decoded_input"],
        "eval_masked_preview": eval_row["masked_preview"],
        "packed_keys": packed_row["keys"],
        "packed_tokens": (
            f"len={packed_row['seq_len']} real={packed_row['real_tokens']} "
            f"masked={packed_row['masked_tokens']}"
        ),
        "packed_masked_preview": packed_row["masked_preview"],
    }


# ---------------------------------------------------------------------------
# Stage validators — raise AssertionError on failure, StageSkip if optional dep missing
# ---------------------------------------------------------------------------


def validate_environment(ctx: PipelineContext) -> dict[str, str]:
    from model.modernbert.attention import IMPL_USE_FLASH2
    from pretrain.gpu_batch import log_device_summary, resolve_device

    device = resolve_device()
    assert device.type in ("cuda", "cpu"), f"unexpected device type: {device.type}"
    summary = log_device_summary(device)
    assert "device=" in summary
    if ctx.device.type == "cuda" and not IMPL_USE_FLASH2:
        print("WARNING: flash-attn not installed — sliding window needs FA2 on GPU")
    return {
        "device": str(device),
        "summary": summary,
        "flash_attn": str(IMPL_USE_FLASH2),
    }


def validate_config(ctx: PipelineContext) -> dict[str, str]:
    from config import load_modernbert_arch_config, load_pretrain_config
    from model.factory import build_modernbert_config

    base = load_modernbert_arch_config(ctx.arch_base)
    assert base.num_hidden_layers == 22 and base.use_fa2 and base.padding == "unpadded"
    assert base.sliding_window == 128 and base.global_attn_every_n_layers == 3

    base_cfg = build_modernbert_config(model_config=base)
    assert base_cfg.vocab_size == base.vocab_size == 50368

    with initialize_config_dir(config_dir=str(ctx.pretrain_config_dir), version_base=None):
        hydra_cfg = compose(config_name="hindi_mlm_phase1")
    p = load_pretrain_config(hydra_cfg)
    ctx.pretrain_cfg = p
    assert str(p.arch_config_path).endswith("modernbert_base.yaml")

    prod = _load_production_cfg(ctx)
    assert prod.num_workers == 2
    assert prod.dataloader_prefetch_factor == 4
    assert prod.eval_num_workers == 3
    assert prod.packing_prefetch_factor == 5
    assert prod.sequence_packing is True
    assert prod.max_seq_len == 1024

    return {
        "arch": f"22L hidden={base.hidden_size} window={base.sliding_window} global_every=3",
        "vocab": str(base_cfg.vocab_size),
        "phase1": f"lr={p.optimizer.lr} global_batch={p.global_train_batch_size} max_seq={p.max_seq_len}",
        "production": "hindi_mlm_smoke_50ba",
        "workers": f"train={prod.num_workers} prefetch={prod.dataloader_prefetch_factor} eval={prod.eval_num_workers}",
        "packing": f"enabled prefetch={prod.packing_prefetch_factor} micro={prod.device_train_microbatch_size}",
    }


def validate_data(ctx: PipelineContext) -> dict[str, str]:
    from pretrain.parquet_mlm import describe_data_root
    from pretrain.parquet_mlm import (
        ListMLMDataset,
        ParquetMLMDataset,
        iter_parquet_paths,
        load_eval_texts,
    )
    from tokenizer.pretokenization import preprocess_for_tokenizer

    normed = preprocess_for_tokenizer("  भारत   ", use_script_norm=True)
    assert len(normed) > 0

    ds = ListMLMDataset(["नमस्ते", "परीक्षण"])
    assert len(ds) == 2 and ds[0] == "नमस्ते"

    out: dict[str, str] = {
        "preprocess": repr(normed),
        "list_dataset": "len=2",
        "describe": describe_data_root(
            ctx.train_data if ctx.has_train_parquet() else ctx.repo / "data"
        ),
    }

    if not ctx.has_train_parquet():
        out["parquet"] = "SKIP — no train parquet under data/sangrah_dataset/"
        return out

    paths = iter_parquet_paths(ctx.train_data)
    assert len(paths) >= 1
    row = ParquetMLMDataset(ctx.train_data, "text", max_shards=1)[0]
    assert isinstance(row, str) and len(row) > 0
    root = ctx.eval_data if ctx.has_eval_parquet() else ctx.train_data
    texts = load_eval_texts(root, "text", max_rows=4)
    assert len(texts) >= 1
    out["parquet"] = f"shard={paths[0].name} sample_len={len(row)} eval_rows={len(texts)}"

    if ctx.has_tokenizer():
        cfg = _load_production_cfg(ctx)
        dl_out = _validate_production_dataloaders(ctx, cfg)
        out.update(dl_out)
        print(f"    ↳ raw [{dl_out['raw_keys']}]: {dl_out['raw_sample']!r}")
        print(f"    ↳ eval [{dl_out['eval_keys']}]: {dl_out['eval_tokens']}")
        print(f"    ↳ eval decoded: {dl_out['eval_sample']!r}")
        print(f"    ↳ eval MLM masks: {dl_out['eval_masked_preview']}")
        print(f"    ↳ packed [{dl_out['packed_keys']}]: {dl_out['packed_tokens']}")
        print(f"    ↳ packed MLM masks: {dl_out['packed_masked_preview']}")
    else:
        out["dataloaders"] = "SKIP — tokenizer artifact missing (make train-bpe)"

    return out


def validate_dataloaders(ctx: PipelineContext) -> dict[str, str]:
    """Backward-compatible alias — prefer validate_data which includes this."""
    if not ctx.has_train_parquet():
        raise StageSkip("train parquet required under data/sangrah_dataset/")
    if not ctx.has_tokenizer():
        raise StageSkip("tokenizer required — run: make train-bpe")
    return _validate_production_dataloaders(ctx, _load_production_cfg(ctx))


def validate_tokenizer(ctx: PipelineContext) -> dict[str, str]:
    from pretrain.parquet_mlm import tokenize_batch
    from utils.paths import resolve_hf_tokenizer_dir

    tokenizer = _ensure_tokenizer(ctx)
    tok_dir = resolve_hf_tokenizer_dir(ctx.tokenizer_path)
    assert (tok_dir / "tokenizer.json").is_file()

    cfg = _load_production_cfg(ctx)
    max_seq_len = cfg.max_seq_len

    sample_text = "भारत एक विशाल देश है और हिंदी इसकी प्रमुख भाषाओं में से एक है।"
    shape = tuple(tokenize_batch(tokenizer, [sample_text], max_seq_len=max_seq_len)["input_ids"].shape)
    assert shape == (1, max_seq_len)

    enc = tokenizer([sample_text], padding="max_length", truncation=True, max_length=max_seq_len)
    ex = [{k: enc[k][0] for k in enc}]
    train_c = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=cfg.mlm_probability)
    eval_c = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=cfg.eval_mlm_probability
    )
    train_batch = train_c(ex)
    eval_batch = eval_c(ex)
    n_train = int((train_batch["labels"] != -100).sum())
    n_eval = int((eval_batch["labels"] != -100).sum())

    tokenized_sample = _decode_ids_sample(tokenizer, enc["input_ids"][0])
    masked_parts: list[str] = []
    preview_len = min(32, max_seq_len)
    for i in range(preview_len):
        label = int(train_batch["labels"][0, i])
        if label == -100:
            continue
        tid = int(train_batch["input_ids"][0, i])
        masked_parts.append(f"{tokenizer.decode([tid])!r}->{tokenizer.decode([label])!r}")
    masked_preview = ", ".join(masked_parts[:4]) or "(none — rerun cell; MLM is stochastic)"

    out = {
        "tokenizer_dir": str(tok_dir),
        "vocab": str(tokenizer.vocab_size),
        "max_seq_len": str(max_seq_len),
        "tokenize_batch_shape": str(shape),
        "mlm_masks": (
            f"train={n_train} eval={n_eval} "
            f"({cfg.mlm_probability} / {cfg.eval_mlm_probability} prob)"
        ),
        "sample_text": sample_text[:120],
        "tokenized_sample": tokenized_sample,
        "masked_preview": masked_preview,
    }
    print(f"    ↳ sample: {out['sample_text']!r}")
    print(f"    ↳ tokenized: {out['tokenized_sample']!r}")
    print(f"    ↳ mlm masks (train 0.3): {out['masked_preview']}")
    return out


def validate_gpu(ctx: PipelineContext) -> dict[str, str]:
    from pretrain.gpu_batch import add_position_ids, move_batch_to_device, training_autocast

    am = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    pid = add_position_ids({"attention_mask": am})["position_ids"]
    assert pid[0, 2].item() == 2 and pid[0, 3].item() == 0

    batch = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    moved = move_batch_to_device(batch, ctx.device)
    assert moved["input_ids"].device.type == ctx.device.type
    assert "position_ids" in moved

    x = torch.randn(4, 4, device=ctx.device)
    with training_autocast(ctx.device):
        y = x @ x
    assert y.shape == (4, 4)

    return {
        "position_ids_row0": str(pid[0].tolist()),
        "batch_device": str(moved["input_ids"].device),
        "autocast_dtype": str(y.dtype),
    }


def validate_model_smoke(ctx: PipelineContext) -> dict[str, str]:
    from pretrain.gpu_batch import training_autocast

    cfg = _load_production_cfg(ctx)
    model, tokenizer, arch = _mlm_model(ctx, ctx.arch_base)
    ctx.mlm_model = model
    trace = _load_script_module("pipeline_trace", ctx.repo / "scripts/pipeline_trace.py")
    device = next(model.parameters()).device
    batch = _mlm_batch(
        ctx,
        tokenizer,
        trace._sample_hindi_texts()[:1],
        max_seq_len=cfg.max_seq_len,
        mlm_probability=cfg.mlm_probability,
    )

    model.eval()
    with torch.no_grad(), training_autocast(device):
        out = model(**batch)
    assert out.loss is not None

    return {
        "arch": f"{arch.num_hidden_layers}L hidden={arch.hidden_size} padding={arch.padding}",
        "vocab": str(arch.vocab_size),
        "max_seq_len": str(cfg.max_seq_len),
        "attention": f"fa2={arch.use_fa2} window={arch.sliding_window} global_every={arch.global_attn_every_n_layers}",
        "forward": f"loss={float(out.loss):.4f} logits={tuple(out.logits.shape)}",
    }


def validate_attention(ctx: PipelineContext) -> dict[str, str]:
    from config import load_modernbert_arch_config
    from model.factory import build_modernbert_config
    from model.modernbert.attention import get_attention_layer
    from pretrain.gpu_batch import training_autocast

    cfg = _load_production_cfg(ctx)
    arch_yaml = load_modernbert_arch_config(ctx.arch_base)
    tokenizer = _ensure_tokenizer(ctx)
    if arch_yaml.vocab_size != tokenizer.vocab_size:
        arch_yaml = arch_yaml.model_copy(update={"vocab_size": tokenizer.vocab_size})
    model_cfg = build_modernbert_config(model_config=arch_yaml)

    patterns: list[str] = []
    for layer_id in range(arch_yaml.num_hidden_layers):
        layer = get_attention_layer(model_cfg, layer_id=layer_id)
        window = getattr(layer, "sliding_window", None)
        kind = "global" if window == (-1, -1) else "sliding"
        patterns.append(f"L{layer_id}:{kind}")
    assert patterns[:4] == ["L0:global", "L1:sliding", "L2:sliding", "L3:global"]
    n_global = sum(1 for p in patterns if p.endswith("global"))
    assert n_global == (arch_yaml.num_hidden_layers + 2) // 3

    layer0 = get_attention_layer(model_cfg, layer_id=0)
    assert "Rope" in layer0.__class__.__name__

    model, tokenizer, _ = _mlm_model(ctx, ctx.arch_base)
    trace = _load_script_module("pipeline_trace", ctx.repo / "scripts/pipeline_trace.py")
    batch = _mlm_batch(
        ctx,
        tokenizer,
        trace._sample_hindi_texts()[:2],
        max_seq_len=cfg.max_seq_len,
        mlm_probability=cfg.mlm_probability,
    )
    micro = _production_microbatch(batch, cfg.device_train_microbatch_size)

    model.eval()
    with torch.no_grad(), training_autocast(ctx.device):
        out = model(**micro)
    assert out.loss is not None

    model.train()
    model.zero_grad(set_to_none=True)
    with training_autocast(ctx.device):
        loss = model(**micro).loss
    assert loss is not None
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    return {
        "layers": f"{arch_yaml.num_hidden_layers}L global={n_global} sliding={len(patterns) - n_global}",
        "pattern_head": ", ".join(patterns[:4]),
        "layer0": f"{layer0.__class__.__name__} window={getattr(layer0, 'sliding_window', None)}",
        "microbatch": f"n={micro['input_ids'].shape[0]} seq={micro['input_ids'].shape[1]}",
        "mlm_backward": f"loss={float(loss):.4f} grad_tensors={len(grads)}",
    }


def validate_mlm_smoke(ctx: PipelineContext) -> dict[str, str]:
    if not ctx.has_train_parquet():
        raise StageSkip("train parquet required for production eval DataLoader")
    from pretrain.dataloader import build_eval_dataloader, build_parquet_train_dataloader
    from pretrain.evals import evaluate_mlm, masked_accuracy
    from pretrain.gpu_batch import training_autocast
    from composer.utils import dist

    tokenizer = _ensure_tokenizer(ctx)
    cfg = _load_production_cfg(ctx)
    model, _, arch = _mlm_model(ctx, ctx.arch_base)
    ctx.mlm_model = model
    device = next(model.parameters()).device
    assert device.type == _sync_device(ctx).type

    device_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    eval_loader = build_eval_dataloader(cfg, tokenizer, device)
    assert eval_loader.num_workers == cfg.eval_num_workers
    full_batch = next(iter(eval_loader))
    assert tuple(full_batch["input_ids"].shape) == (
        eval_loader.batch_size,
        cfg.max_seq_len,
    )
    micro_n = cfg.device_eval_microbatch_size or cfg.device_train_microbatch_size
    batch = _prepare_batch(_production_microbatch(full_batch, micro_n), device)
    labels = batch["labels"]

    model.eval()
    with torch.no_grad(), training_autocast(device):
        out = model(**batch)
    masked_n = int((labels != -100).sum())
    acc = masked_accuracy(out.logits, labels)
    assert out.loss is not None
    assert tuple(out.logits.shape) == (masked_n, model.config.vocab_size)
    assert 0.0 <= acc <= 1.0

    metrics = evaluate_mlm(
        model, eval_loader, device, max_batches=2, microbatch_size=micro_n
    )
    assert metrics.steps > 0 and metrics.tokens > 0

    packed = build_parquet_train_dataloader(
        cfg,
        tokenizer,
        device,
        device_batch_size=device_batch_size,
    )
    packed_batch = next(iter(packed))
    assert "input_ids" in packed_batch

    model.train()
    model.zero_grad(set_to_none=True)
    with training_autocast(device):
        loss = model(**batch).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    return {
        "arch": f"{arch.num_hidden_layers}L vocab={tokenizer.vocab_size}",
        "device": str(device),
        "eval_loader": (
            f"workers={eval_loader.num_workers} "
            f"batch={tuple(full_batch['input_ids'].shape)} micro={micro_n}"
        ),
        "eval_keys": ", ".join(sorted(full_batch.keys())),
        "eval_tokens": (
            f"real={int(full_batch['attention_mask'][0].sum())} "
            f"masked={int((full_batch['labels'][0] != -100).sum())}"
        ),
        "eval_sample": _decode_ids_sample(tokenizer, batch["input_ids"][0]),
        "eval_masked_preview": _mlm_mask_preview(
            tokenizer,
            _ids_list(batch["input_ids"][0]),
            _ids_list(batch["labels"][0]),
        ),
        "forward": f"loss={float(out.loss):.4f} logits={tuple(out.logits.shape)} acc={acc:.4f}",
        "evaluate": f"loss={metrics.loss:.4f} acc={metrics.masked_accuracy:.4f} tokens={metrics.tokens}",
        "packed_batch": f"shape={tuple(packed_batch['input_ids'].shape)}",
        "backward": f"loss={float(loss):.4f} grad_tensors={len(grads)}",
    }


def validate_probe(ctx: PipelineContext) -> dict[str, str]:
    """Backward-compatible alias for notebook step 7."""
    return validate_mlm_smoke(ctx)


def validate_trace(ctx: PipelineContext) -> dict[str, str]:
    if not ctx.has_tokenizer():
        raise StageSkip("tokenizer required — run Step 3 first")

    cfg = _load_production_cfg(ctx)
    trace = _load_script_module("pipeline_trace", ctx.repo / "scripts/pipeline_trace.py")
    results = trace.run_pipeline_trace(
        arch_config=ctx.arch_base,
        tokenizer_path=ctx.tokenizer_path,
        max_seq_len=cfg.max_seq_len,
        microbatch_size=cfg.device_train_microbatch_size,
    )
    failed = [r for r in results if not r.ok]
    assert not failed, "; ".join(f"{r.name}: {r.detail}" for r in failed)
    out = {r.name: r.detail for r in results}
    out["arch"] = "modernbert_base"
    out["max_seq_len"] = str(cfg.max_seq_len)
    return out


_NOTEBOOK_E2E_STEPS = 3


def _load_notebook_e2e_cfg(ctx: PipelineContext):
    """Notebook mini-run: production stack from smoke yaml, tiny batch + 1 shard."""
    from config import load_pretrain_config

    if not ctx.has_tokenizer():
        raise StageSkip("tokenizer required — run Step 3 first")
    if not ctx.has_train_parquet():
        raise StageSkip("train parquet required")
    if _sync_device(ctx).type != "cuda":
        raise StageSkip("CUDA required for packed pretrain e2e (FA2 + torch.compile)")

    micro = 8
    global_batch = 8

    with initialize_config_dir(config_dir=str(ctx.pretrain_config_dir), version_base=None):
        cfg = load_pretrain_config(compose(config_name="hindi_mlm_smoke_50ba"))

    return cfg.model_copy(
        update={
            "global_train_batch_size": global_batch,
            "device_train_microbatch_size": micro,
            "device_eval_microbatch_size": micro,
            "global_eval_batch_size": global_batch,
            "max_train_shards": 1,
            "num_workers": 0,
            "eval_num_workers": 0,
            "batch_size_warmup_tokens": None,
            "batch_size_warmup_min_size": None,
        }
    )


def _run_notebook_train_loop(ctx: PipelineContext, cfg, *, num_steps: int) -> dict[str, str]:
    """Sample packed train batches, forward/backward microbatches, optimizer steps, then eval."""
    from composer.utils import dist
    from model.factory import create_modernbert_mlm
    from pretrain.dataloader import build_eval_dataloader, build_parquet_train_dataloader
    from pretrain.evals import evaluate_mlm, masked_accuracy
    from pretrain.gpu_batch import move_batch_to_device, training_autocast
    from pretrain.sequence_packer import get_num_samples_in_packed_batch, split_packed_batch
    from pretrain.train import _validate_production_kernels
    from pretrain.wiring import build_optimizer

    device = _sync_device(ctx)
    tokenizer = _ensure_tokenizer(ctx)
    composer_model = create_modernbert_mlm(
        pretrained_model_name=cfg.pretrained_model_name,
        model_config=cfg.load_arch(),
        tokenizer_path=str(cfg.tokenizer_path),
        gradient_checkpointing=cfg.gradient_checkpointing,
        disable_train_metrics=cfg.disable_train_metrics,
    )
    _validate_production_kernels(cfg, composer_model)
    model = composer_model.model
    model.to(device)
    model.train()

    optimizer = build_optimizer(cfg.optimizer, model)
    device_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    train_loader = build_parquet_train_dataloader(
        cfg,
        tokenizer,
        device,
        device_batch_size=device_batch_size,
    )
    eval_loader = build_eval_dataloader(cfg, tokenizer, device)
    train_iter = iter(train_loader)

    step_summaries: list[str] = []
    last_packed_shape = "n/a"
    last_micro_logits = "n/a"
    last_masked_n = 0

    for step in range(num_steps):
        packed_batch = next(train_iter)
        last_packed_shape = str(tuple(packed_batch["input_ids"].shape))
        microbatches = split_packed_batch(packed_batch, cfg.device_train_microbatch_size)
        assert microbatches, f"step {step}: empty microbatch split"

        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for micro_idx, micro in enumerate(microbatches):
            micro = move_batch_to_device(micro, device)
            with training_autocast(device):
                out = model(**micro)
            assert out.loss is not None
            out.loss.backward()
            micro_losses.append(float(out.loss))
            if step == num_steps - 1 and micro_idx == 0:
                labels = micro["labels"]
                last_masked_n = int((labels != -100).sum())
                last_micro_logits = str(tuple(out.logits.shape))

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()

        n_seqs = get_num_samples_in_packed_batch(packed_batch)
        step_summaries.append(
            f"step{step + 1}: micros={len(microbatches)} seqs={n_seqs} "
            f"loss={sum(micro_losses) / len(micro_losses):.4f} grads={len(grads)}"
        )

    eval_micro = cfg.device_eval_microbatch_size or cfg.device_train_microbatch_size
    eval_batch = _prepare_batch(_production_microbatch(next(iter(eval_loader)), eval_micro), device)
    model.eval()
    with torch.no_grad(), training_autocast(device):
        eval_out = model(**eval_batch)
    eval_masked_n = int((eval_batch["labels"] != -100).sum())
    eval_acc = masked_accuracy(eval_out.logits, eval_batch["labels"])
    assert eval_out.loss is not None
    assert tuple(eval_out.logits.shape) == (eval_masked_n, model.config.vocab_size)

    metrics = evaluate_mlm(
        model,
        eval_loader,
        device,
        max_batches=2,
        microbatch_size=eval_micro,
    )
    assert metrics.steps > 0 and metrics.tokens > 0

    return {
        "steps": " | ".join(step_summaries),
        "packed_batch": last_packed_shape,
        "last_micro": f"masked={last_masked_n} logits={last_micro_logits}",
        "eval_forward": (
            f"loss={float(eval_out.loss):.4f} masked={eval_masked_n} "
            f"logits={tuple(eval_out.logits.shape)} acc={eval_acc:.4f}"
        ),
        "eval_loop": (
            f"loss={metrics.loss:.4f} acc={metrics.masked_accuracy:.4f} "
            f"batches={metrics.steps} tokens={metrics.tokens}"
        ),
    }


def validate_full_pretrain(ctx: PipelineContext) -> dict[str, str]:
    try:
        from composer.utils import dist  # noqa: F401
    except ImportError as exc:
        raise StageSkip("Composer not installed — run: uv sync --extra pretrain") from exc

    from config import load_modernbert_arch_config

    cfg = _load_notebook_e2e_cfg(ctx)
    arch = load_modernbert_arch_config(cfg.arch_config_path)
    assert cfg.sequence_packing and arch.num_hidden_layers == 22

    loop = _run_notebook_train_loop(ctx, cfg, num_steps=_NOTEBOOK_E2E_STEPS)
    entrypoint = ctx.repo / "scripts/run_pretrain.py"
    assert entrypoint.is_file()

    return {
        "path": f"packed train → {_NOTEBOOK_E2E_STEPS}× (forward+backward+optim.step) → eval",
        "config": "hindi_mlm_smoke_50ba, 1 shard, workers=0, no packer warmup",
        "arch": f"{arch.num_hidden_layers}L loss={arch.loss_function}",
        "batch": (
            f"global={cfg.global_train_batch_size} micro={cfg.device_train_microbatch_size} "
            f"grad_accum={cfg.grad_accum_steps}"
        ),
        **loop,
        "production_entry": str(entrypoint.relative_to(ctx.repo)),
        "production_smoke": "make train-smoke-50ba",
        "production_full": "run_pretrain.py --config-name hindi_mlm_phase1",
    }


def print_stage_result(step: int, title: str, result: dict[str, str]) -> None:
    print(f"✓ Step {step} — {title}")
    for key, value in result.items():
        print(f"    {key}: {value}")
