"""Parquet text dataset for MLM pretraining."""

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
    """PyTorch Dataset over parquet text rows."""

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
        # mmap-backed Arrow tables; never materialize full-shard Python lists (≈4GB/shard).
        self._tables: OrderedDict[Path, pa.Table] = OrderedDict()

        ends: list[int] = []
        total = 0
        for path in self.paths:
            total += pq.read_metadata(path).num_rows
            ends.append(total)
        if total == 0:
            raise FileNotFoundError(f"No parquet rows under {data_root}")
        self._ends = tuple(ends)

    def __len__(self) -> int:
        return self._ends[-1]

    def _table(self, path: Path) -> pa.Table:
        cached = self._tables.get(path)
        if cached is not None:
            self._tables.move_to_end(path)
            return cached

        cached = pq.read_table(path, columns=[self.text_column], memory_map=True)
        self._tables[path] = cached
        while len(self._tables) > self._max_cached_shards:
            self._tables.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += len(self)
        shard_idx = bisect.bisect_right(self._ends, index)
        start = 0 if shard_idx == 0 else self._ends[shard_idx - 1]
        path = self.paths[shard_idx]
        value = self._table(path).column(self.text_column)[index - start].as_py()
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
    *,``
    max_rows: int,
    use_script_norm: bool = True,
) -> list[str]:
    texts: list[str] = []

    for path in iter_parquet_paths(data_root):
        #! Milgaya bug, always use the arrow format. 
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
        return [
            self.tokenizer(text, truncation=True, padding=False, max_length=self.max_seq_len)
            if text
            else {"input_ids": []}
            for text in texts
        ]


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
        features = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        examples = [{key: features[key][i] for key in features} for i in range(len(texts))]
        return self.collator(examples)
