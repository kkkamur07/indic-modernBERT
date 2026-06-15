"""Parquet dataloader glue — mirrors upstream text_data.py packing / padded branches."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from composer.utils import dist

from config import PretrainConfig
from pretrain.parquet_mlm import (
    MLMCollator,
    ParquetMLMDataset,
    TokenizeCollator,
)
from pretrain.sequence_packer import BufferedIterable, GreedyBestFitSequencePacker


class DistributedSamplerPCG64DXSM(DistributedSampler):
    """PCG64DXSM shuffle with parquet shard locality.

    Global row permutations make each DataLoader batch jump across many parquet
    shards. Since ``ParquetMLMDataset`` mmaps/caches Arrow tables per shard, that
    pattern can blow up worker RAM before the first packed batch is ready.
    """

    def __iter__(self) -> Iterator[int]:
        shard_ends = getattr(self.dataset, "_ends", None)
        if shard_ends is not None and self.shuffle:
            return self._iter_shard_local(shard_ends)

        if self.shuffle:
            rng = np.random.Generator(np.random.PCG64DXSM(self.seed + self.epoch))
            indices = rng.permutation(len(self.dataset)).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def _iter_shard_local(self, shard_ends: tuple[int, ...]) -> Iterator[int]:
        rng = np.random.Generator(np.random.PCG64DXSM(self.seed + self.epoch))
        shard_order = rng.permutation(len(shard_ends))
        emitted = 0
        seen = 0
        prefix: list[int] = []
        padding_size = self.total_size - len(self.dataset)  # type: ignore[arg-type]

        def maybe_yield(index: int) -> Iterator[int]:
            nonlocal emitted, seen
            if len(prefix) < padding_size:
                prefix.append(index)
            if seen < self.total_size and seen % self.num_replicas == self.rank:
                emitted += 1
                yield index
            seen += 1

        for shard_idx in shard_order:
            start = 0 if shard_idx == 0 else shard_ends[shard_idx - 1]
            end = shard_ends[shard_idx]
            for index in rng.permutation(np.arange(start, end)):
                yield from maybe_yield(int(index))

        if not self.drop_last:
            for index in prefix:
                yield from maybe_yield(index)
                if emitted >= self.num_samples:
                    break

        assert emitted == self.num_samples


def _dataloader_kwargs(
    pretrain_cfg: PretrainConfig,
    device: torch.device,
    *,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    sampler: DistributedSampler | None = None,
) -> dict[str, object]:
    workers = pretrain_cfg.num_workers
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pretrain_cfg.dataloader_pin_memory and device.type == "cuda",
        "drop_last": drop_last,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = pretrain_cfg.dataloader_prefetch_factor
    return kwargs


def build_padded_mlm_dataloader(
    pretrain_cfg: PretrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    *,
    device_batch_size: int,
) -> DataLoader:

    collator = MLMCollator(
        tokenizer,
        max_seq_len=pretrain_cfg.max_seq_len,
        mlm_probability=pretrain_cfg.mlm_probability,
    )
    
    return DataLoader(
        ParquetMLMDataset(pretrain_cfg.data_root, pretrain_cfg.text_column),
        collate_fn=collator,
        **_dataloader_kwargs(
            pretrain_cfg,
            device,
            batch_size=device_batch_size,
            drop_last=pretrain_cfg.drop_last,
            shuffle=False,
        ),
    )


def build_parquet_train_dataloader(
    pretrain_cfg: PretrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    *,
    device_batch_size: int,
) -> BufferedIterable:
    """Upstream text_data.py packing branch — parquet replaces NoStreamingDataset."""
    dataset = ParquetMLMDataset(
        pretrain_cfg.data_root,
        pretrain_cfg.text_column,
        max_shards=pretrain_cfg.max_train_shards,
    )
    collator = TokenizeCollator(
        tokenizer,
        max_seq_len=pretrain_cfg.max_seq_len,
    )

    sampler = DistributedSamplerPCG64DXSM(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=True,
        seed=pretrain_cfg.shuffle_seed,
        drop_last=pretrain_cfg.drop_last,
    )

    buffer_size = pretrain_cfg.packing_buffer_size
    if buffer_size is None:
        buffer_size = 5 * device_batch_size

    raw_loader = DataLoader(
        dataset,
        collate_fn=collator,
        **_dataloader_kwargs(
            pretrain_cfg,
            device,
            batch_size=device_batch_size,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        ),
    )

    sequence_packer = GreedyBestFitSequencePacker.from_composer(
        raw_loader,
        batch_size=device_batch_size,
        micro_batch_size=pretrain_cfg.device_train_microbatch_size,
        max_seq_len=pretrain_cfg.max_seq_len,
        buffer_size=buffer_size,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_prob=pretrain_cfg.mlm_probability,
        seed=pretrain_cfg.shuffle_seed,
        batch_size_warmup_min_size=pretrain_cfg.batch_size_warmup_min_size,
        batch_size_warmup_tokens=pretrain_cfg.batch_size_warmup_tokens,
        world_size=dist.get_world_size(),
    )
    
    return BufferedIterable(sequence_packer, buffer_size=pretrain_cfg.packing_prefetch_factor)


def _eval_device_batch_size(pretrain_cfg: PretrainConfig) -> int:
    if pretrain_cfg.global_eval_batch_size is not None:
        batch = pretrain_cfg.global_eval_batch_size // dist.get_world_size()
        if batch < 1:
            raise ValueError(
                f"global_eval_batch_size={pretrain_cfg.global_eval_batch_size} is too small "
                f"for world_size={dist.get_world_size()}"
            )
        return batch
    if pretrain_cfg.device_eval_microbatch_size is not None:
        return pretrain_cfg.device_eval_microbatch_size
    return pretrain_cfg.device_train_microbatch_size


def build_eval_dataloader(
    pretrain_cfg: PretrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> DataLoader:
    """Upstream eval_loader — padded MLM, shuffle=false, mlm_probability=0.15 by default."""
    if pretrain_cfg.eval_sequence_packing:
        device_batch_size = _eval_device_batch_size(pretrain_cfg)
        return build_parquet_train_dataloader(
            pretrain_cfg.model_copy(
                update={
                    "mlm_probability": pretrain_cfg.eval_mlm_probability,
                    "drop_last": False,
                }
            ),
            tokenizer,
            device,
            device_batch_size=device_batch_size,
        )

    eval_root = pretrain_cfg.eval_data_root or pretrain_cfg.data_root
    collator = MLMCollator(
        tokenizer,
        max_seq_len=pretrain_cfg.max_seq_len,
        mlm_probability=pretrain_cfg.eval_mlm_probability,
    )
    eval_kwargs = _dataloader_kwargs(
        pretrain_cfg,
        device,
        batch_size=_eval_device_batch_size(pretrain_cfg),
        drop_last=False,
        shuffle=False,
    )
    eval_kwargs["num_workers"] = pretrain_cfg.eval_num_workers
    return DataLoader(
        ParquetMLMDataset(eval_root, pretrain_cfg.text_column),
        collate_fn=collator,
        **eval_kwargs,
    )
