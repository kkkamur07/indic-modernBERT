"""Parquet text dataset for MLM pretraining.

Reading parquet shards for training/eval DataLoaders must **not** materialize a
full text column in each worker: that turns large shards into multi-GB Python or
Arrow objects and multiplies RAM across DataLoader workers.

Use mmap-backed row-group reads instead:

- ``ParquetFile.read_row_group(..., columns=[text])`` — read only one row group
- ``table.column(name)[i].as_py()`` — decode one row at a time
- LRU-cache a small number of row-group tables

See ``LEARNINGS.md`` § Parquet DataLoader memory fixes.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from constants import HINDI_LANG3
from tokenizer.pretokenization import preprocess_for_tokenizer


def describe_data_root(data_root: Path) -> str:
    shards = sorted(data_root.rglob("*.parquet"))
    if not shards:
        return f"no parquet shards under {data_root}"
    return f"{len(shards)} parquet shard(s) under {data_root}"


def iter_parquet_paths(data_root: Path) -> list[Path]:
    patterns = (
        f"verified/{HINDI_LANG3}/*.parquet",
        f"unverified/{HINDI_LANG3}/*.parquet",
        "synthetic/hin_*/**/*.parquet",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(data_root.glob(pattern)))
    if not paths:
        paths = sorted(data_root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {data_root}")
    return paths


def _text_from_value(value: object, *, use_script_norm: bool) -> str:
    if value is None:
        return ""
    return preprocess_for_tokenizer(str(value), use_script_norm=use_script_norm).strip()


class ParquetMLMDataset(Dataset):
    """PyTorch Dataset over parquet text rows (mmap row groups, per-row ``as_py()``)."""

    def __init__(
        self,
        data_root: Path,
        text_column: str,
        *,
        max_shards: int | None = None,
        max_cached_shards: int = 2,
        use_script_norm: bool = True,
    ) -> None:

        paths = iter_parquet_paths(data_root)
        if max_shards is not None:
            paths = paths[:max_shards]

        self.text_column = text_column
        self.use_script_norm = use_script_norm
        self.paths = tuple(paths)
        self._max_cached_shards = max(1, max_cached_shards)
        # mmap-backed row-group tables; never materialize full shard columns in workers.
        self._row_group_tables: OrderedDict[tuple[Path, int], pa.Table] = OrderedDict()

        ends: list[int] = []
        row_group_ends_by_shard: list[tuple[int, ...]] = []
        total = 0
        for path in self.paths:
            metadata = pq.read_metadata(path)
            shard_total = 0
            row_group_ends: list[int] = []
            for row_group_idx in range(metadata.num_row_groups):
                shard_total += metadata.row_group(row_group_idx).num_rows
                row_group_ends.append(shard_total)
            total += shard_total
            ends.append(total)
            row_group_ends_by_shard.append(tuple(row_group_ends))
        if total == 0:
            raise FileNotFoundError(f"No parquet rows under {data_root}")
        self._ends = tuple(ends)
        self._row_group_ends_by_shard = tuple(row_group_ends_by_shard)

    def __len__(self) -> int:
        return self._ends[-1]

    def _row_group_table(self, path: Path, row_group_idx: int) -> pa.Table:
        key = (path, row_group_idx)
        cached = self._row_group_tables.get(key)
        if cached is not None:
            self._row_group_tables.move_to_end(key)
            return cached

        parquet_file = pq.ParquetFile(path, memory_map=True)
        cached = parquet_file.read_row_group(row_group_idx, columns=[self.text_column], use_threads=False)
        self._row_group_tables[key] = cached
        while len(self._row_group_tables) > self._max_cached_shards:
            self._row_group_tables.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += len(self)
        shard_idx = bisect.bisect_right(self._ends, index)
        start = 0 if shard_idx == 0 else self._ends[shard_idx - 1]
        path = self.paths[shard_idx]
        shard_row = index - start
        row_group_ends = self._row_group_ends_by_shard[shard_idx]
        row_group_idx = bisect.bisect_right(row_group_ends, shard_row)
        row_group_start = 0 if row_group_idx == 0 else row_group_ends[row_group_idx - 1]
        value = self._row_group_table(path, row_group_idx).column(self.text_column)[
            shard_row - row_group_start
        ].as_py()
        return _text_from_value(value, use_script_norm=self.use_script_norm)


class ListMLMDataset(Dataset):
    """In-memory text list for eval."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def load_eval_texts(
    data_root: Path,
    text_column: str,
    *,
    max_rows: int,
    max_shards: int | None = None,
    use_script_norm: bool = True,
) -> list[str]:
    texts: list[str] = []

    paths = iter_parquet_paths(data_root)
    if max_shards is not None:
        paths = paths[:max_shards]

    for path in paths:
        table = pq.read_table(path, columns=[text_column], memory_map=True)
        for value in table.column(text_column):
            text = _text_from_value(value.as_py(), use_script_norm=use_script_norm)

            if text:
                texts.append(text)
                if len(texts) >= max_rows:
                    return texts
    return texts


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:

    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )


class TokenizeCollator:
    """DataLoader collate_fn for unpadded tokenization (sequence packing)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, texts: list[str]) -> list[dict[str, list[int]]]:
        import time

        from pretrain.step_log import bump, should_log_detail, step_log

        n = bump("tokenize_collate")
        log = should_log_detail("tokenize_collate", first_n=10)
        if log:
            step_log("data", f"tokenize start | #{n} | n_texts={len(texts)} | max_seq_len={self.max_seq_len}")
            t0 = time.perf_counter()

        result = [
            self.tokenizer(text, truncation=True, padding=False, max_length=self.max_seq_len)
            if text
            else {"input_ids": []}
            for text in texts
        ]

        if log:
            lens = [len(row["input_ids"]) for row in result if row["input_ids"]]
            if lens:
                step_log(
                    "data",
                    f"tokenize done | #{n} | nonempty={len(lens)} | "
                    f"len min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.0f} | "
                    f"{time.perf_counter() - t0:.1f}s",
                )
            else:
                step_log("data", f"tokenize done | #{n} | all empty | {time.perf_counter() - t0:.1f}s")
        return result


class MLMCollator:
    """DataLoader collate_fn for padded MLM batches."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_len: int,
        mlm_probability: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        import time

        from pretrain.step_log import bump, should_log_detail, step_log

        n = bump("mlm_collate")
        log = should_log_detail("mlm_collate", first_n=10)
        if log:
            step_log(
                "data",
                f"eval tokenize+mlm start | #{n} | n_texts={len(texts)} | "
                f"max_seq_len={self.max_seq_len} prob={self.collator.mlm_probability}",
            )
            t0 = time.perf_counter()

        features = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        examples = [{key: features[key][i] for key in features} for i in range(len(texts))]
        batch = self.collator(examples)

        if log:
            masked = int((batch["labels"] != -100).sum())
            step_log(
                "data",
                f"eval tokenize+mlm done | #{n} | shape={tuple(batch['input_ids'].shape)} "
                f"masked={masked} | {time.perf_counter() - t0:.1f}s",
            )
        return batch
