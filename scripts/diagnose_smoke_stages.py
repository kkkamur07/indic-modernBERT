"""Stage-by-stage smoke diagnostics — isolate LR-sweep regressions in the data path."""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "indic-modernBERT"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("TMPDIR", str(_REPO / ".tmp"))

import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from transformers import PreTrainedTokenizerFast

from config import load_pretrain_config
from pretrain.dataloader import build_parquet_train_dataloader
from pretrain.parquet_mlm import ParquetMLMDataset, TokenizeCollator
from utils.paths import resolve_hf_tokenizer_dir


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float
    rss_mb: float
    detail: str


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load_smoke_cfg():
    config_dir = str((_REPO / "configs" / "pretrain").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return load_pretrain_config(compose(config_name="hindi_mlm_smoke_50ba"))


def stage_config() -> StageResult:
    t0 = time.perf_counter()
    cfg = _load_smoke_cfg()
    detail = (
        f"shards=all max_train_shards={cfg.max_train_shards} "
        f"global_batch={cfg.global_train_batch_size} micro={cfg.device_train_microbatch_size} "
        f"workers={cfg.num_workers} packing={cfg.sequence_packing}"
    )
    return StageResult("1_config", True, time.perf_counter() - t0, _rss_mb(), detail)


def stage_dataset_init(cfg, tokenizer) -> StageResult:
    t0 = time.perf_counter()
    ds = ParquetMLMDataset(
        cfg.data_root,
        cfg.text_column,
        max_shards=cfg.max_train_shards,
    )
    detail = f"len={len(ds):,}"
    return StageResult("2_dataset_init", True, time.perf_counter() - t0, _rss_mb(), detail)


def stage_single_getitem(cfg, tokenizer) -> StageResult:
    t0 = time.perf_counter()
    ds = ParquetMLMDataset(
        cfg.data_root,
        cfg.text_column,
        max_shards=cfg.max_train_shards,
    )
    sample = ds[0]
    mid = ds[len(ds) // 2]
    detail = f"idx0_len={len(sample)} mid_len={len(mid)}"
    del ds, sample, mid
    gc.collect()
    return StageResult("3_single_getitem", True, time.perf_counter() - t0, _rss_mb(), detail)


def stage_raw_dataloader_batch(cfg, tokenizer, *, num_workers: int) -> StageResult:
    from torch.utils.data import DataLoader

    from pretrain.dataloader import DistributedSamplerPCG64DXSM, _dataloader_kwargs
    from composer.utils import dist

    t0 = time.perf_counter()
    cfg = cfg.model_copy(update={"num_workers": num_workers})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    ds = ParquetMLMDataset(
        cfg.data_root,
        cfg.text_column,
        max_shards=cfg.max_train_shards,
    )
    collator = TokenizeCollator(tokenizer, max_seq_len=cfg.max_seq_len)
    sampler = DistributedSamplerPCG64DXSM(
        ds,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=True,
        seed=cfg.shuffle_seed,
        drop_last=cfg.drop_last,
    )
    loader = DataLoader(
        ds,
        collate_fn=collator,
        **_dataloader_kwargs(
            cfg,
            device,
            batch_size=device_batch_size,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        ),
    )
    batch = next(iter(loader))
    detail = f"workers={num_workers} batch_len={len(batch)} first_ids={len(batch[0]['input_ids'])}"
    del loader, ds, batch
    gc.collect()
    ok = True
    return StageResult(f"4_raw_dataloader_w{num_workers}", ok, time.perf_counter() - t0, _rss_mb(), detail)


def stage_packed_batch(cfg, tokenizer) -> StageResult:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_batch_size = cfg.global_train_batch_size // 1
    packed = build_parquet_train_dataloader(
        cfg,
        tokenizer,
        device,
        device_batch_size=device_batch_size,
    )
    batch = next(iter(packed))
    detail = f"keys={list(batch.keys())} input_shape={tuple(batch['input_ids'].shape)}"
    del packed, batch
    gc.collect()
    return StageResult("5_packed_batch", True, time.perf_counter() - t0, _rss_mb(), detail)


def stage_model_init(cfg) -> StageResult:
    from model.factory import create_modernbert_mlm

    t0 = time.perf_counter()
    model = create_modernbert_mlm(
        pretrained_model_name=cfg.pretrained_model_name,
        model_config=cfg.load_arch(),
        tokenizer_path=str(cfg.tokenizer_path),
        gradient_checkpointing=cfg.gradient_checkpointing,
        disable_train_metrics=cfg.disable_train_metrics,
    )
    detail = f"params={sum(p.numel() for p in model.parameters()):,}"
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageResult("6_model_init", True, time.perf_counter() - t0, _rss_mb(), detail)


def run_all() -> list[StageResult]:
    results: list[StageResult] = []
    cfg = _load_smoke_cfg()
    tok_dir = resolve_hf_tokenizer_dir(cfg.tokenizer_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))

    stages = [
        lambda: stage_config(),
        lambda: stage_dataset_init(cfg, tokenizer),
        lambda: stage_single_getitem(cfg, tokenizer),
        lambda: stage_raw_dataloader_batch(cfg, tokenizer, num_workers=0),
        lambda: stage_raw_dataloader_batch(cfg, tokenizer, num_workers=cfg.num_workers),
        lambda: stage_packed_batch(cfg, tokenizer),
        lambda: stage_model_init(cfg),
    ]
    for fn in stages:
        try:
            results.append(fn())
        except Exception as exc:
            results.append(
                StageResult(
                    getattr(fn, "__name__", "stage"),
                    False,
                    0.0,
                    _rss_mb(),
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}",
                )
            )
            break
    return results


def stage_broken_lr_sweep_regressions() -> list[StageResult]:
    """Reproduce the two bugs introduced during LR-sweep dataloader work."""
    results: list[StageResult] = []
    data_root = _REPO / "data" / "sangrah_dataset"

    # Bug A (LR sweep v1): materialize 39M (path, row) tuples at init.
    t0 = time.perf_counter()
    try:
        from pretrain.parquet_mlm import iter_parquet_paths

        rows: list[tuple[Path, int]] = []
        for path in iter_parquet_paths(data_root):
            n = pq.read_metadata(path).num_rows
            rows.extend((path, i) for i in range(n))
        results.append(
            StageResult(
                "X_bugA_materialized_rows",
                True,
                time.perf_counter() - t0,
                _rss_mb(),
                f"rows={len(rows):,} rss_after_build={_rss_mb():.0f}MB (OLD ParquetMLMMapDataset)",
            )
        )
        del rows
        gc.collect()
    except MemoryError:
        results.append(
            StageResult(
                "X_bugA_materialized_rows",
                False,
                time.perf_counter() - t0,
                _rss_mb(),
                "MemoryError building 39M tuple index",
            )
        )

    # Bug B (LR sweep v2): invalid pq.read_table(row_groups=...) API.
    t0 = time.perf_counter()
    try:
        from pretrain.parquet_mlm import iter_parquet_paths

        path = iter_parquet_paths(data_root)[0]
        pq.read_table(path, row_groups=[0], columns=["text"])
        results.append(
            StageResult(
                "X_bugB_invalid_row_groups_kwarg",
                True,
                time.perf_counter() - t0,
                _rss_mb(),
                "unexpectedly succeeded",
            )
        )
    except TypeError as exc:
        results.append(
            StageResult(
                "X_bugB_invalid_row_groups_kwarg",
                False,
                time.perf_counter() - t0,
                _rss_mb(),
                f"TypeError: {exc}",
            )
        )
    return results


def main() -> None:
    print("=== LR-sweep regression repro ===")
    for r in stage_broken_lr_sweep_regressions():
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name} | {r.seconds:.2f}s | RSS={r.rss_mb:.0f}MB | {r.detail}")

    print("\n=== smoke stage diagnostics (current code) ===")
    for r in run_all():
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name} | {r.seconds:.2f}s | RSS={r.rss_mb:.0f}MB | {r.detail}")


if __name__ == "__main__":
    main()
