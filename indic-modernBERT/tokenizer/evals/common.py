"""Shared utilities for tokenizer evaluation."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import HINDI_LANG3

from utils.progress import iter_with_progress

from .metrics import (
    aggregate_parity_ratio,
    bytes_per_token,
    fertility,
    normalized_sequence_length,
    parity_ratio_cross_lingual,
    renyi_efficiency,
)

IntrinsicMetrics = dict[str, float | int]
EncodeLenFn = Callable[[str], int]
EncodeTokensFn = Callable[[str], list[str]]


def load_candidate_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path.resolve()}. "
            "Train first (make train-superbpe) or run make eval-intrinsic-smoke."
        )
    return Tokenizer.from_file(str(path))


def load_hf_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def fast_encode_fns(tokenizer: Tokenizer) -> tuple[EncodeLenFn, EncodeTokensFn]:

    def tokenize_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def tokenize_tokens(text: str) -> list[str]:
        return tokenizer.encode(text, add_special_tokens=False).tokens

    return tokenize_len, tokenize_tokens


def hf_encode_fns(tokenizer: PreTrainedTokenizerBase) -> tuple[EncodeLenFn, EncodeTokensFn]:

    def tokenize_len(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def tokenize_tokens(text: str) -> list[str]:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return tokenizer.convert_ids_to_tokens(token_ids)

    return tokenize_len, tokenize_tokens


def iter_hindi_lines(
    data_root: Path,
    text_column: str,
    *,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> Iterator[tuple[str, list[str]]]:
    sangrah_layout = sorted(data_root.glob(f"verified/{HINDI_LANG3}/*.parquet"))
    eval_layout = sorted(data_root.glob("*.parquet"))
    parquet_files = sangrah_layout or eval_layout

    if not parquet_files:
        raise FileNotFoundError(
            "No Hindi eval parquet files found. Expected either "
            f"{data_root}/verified/{HINDI_LANG3}/*.parquet (Sangrah layout) "
            f"or {data_root}/*.parquet (eval holdout layout)."
        )

    label = progress_desc or "Eval"
    shard_total = len(parquet_files)

    for shard_idx, parquet_path in enumerate(parquet_files, start=1):
        table = pq.read_table(parquet_path, columns=[text_column])
        column = table[text_column]
        values = column.to_pylist()
        desc = f"{label} | {shard_idx}/{shard_total} {parquet_path.name}"
        row_iter = iter_with_progress(
            values,
            total=len(column),
            desc=desc,
            show_progress=show_progress,
        )

        for value in row_iter:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            words = text.split()
            if not words:
                continue
            yield text, words


def iter_parallel_lines(
    parallel_path: Path,
    hindi_column: str,
    reference_column: str,
    *,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> Iterator[tuple[str, str]]:

    table = pq.read_table(parallel_path, columns=[hindi_column, reference_column])
    hindi_values = table[hindi_column].to_pylist()
    reference_values = table[reference_column].to_pylist()

    label = progress_desc or "Parity"
    desc = f"{label} | {parallel_path.name}"
    pair_iter = zip(hindi_values, reference_values, strict=True)
    if show_progress and sys.stderr.isatty():
        from tqdm import tqdm

        pair_iter = tqdm(pair_iter, total=len(hindi_values), desc=desc, unit="rows")
    elif show_progress:
        logger.info("{} | {} rows", desc, len(hindi_values))

    for hindi_value, reference_value in pair_iter:
        if hindi_value is None or reference_value is None:
            continue

        hindi_text = str(hindi_value).strip()
        reference_text = str(reference_value).strip()

        if not hindi_text or not reference_text:
            continue

        yield hindi_text, reference_text


def collect_intrinsic_metrics(
    *,
    tokenize_len: EncodeLenFn,
    tokenize_tokens: EncodeTokensFn,
    data_root: Path,
    text_column: str,
    reference_tokenize_len: EncodeLenFn | None = None,
    vocab_size: int | None = None,
    renyi_alpha: float = 2.5,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> IntrinsicMetrics:

    totals = {
        "rows": 0,
        "words": 0,
        "tokens": 0,
        "reference_tokens": 0,
        "bytes": 0,
    }

    token_counts: Counter[str] = Counter()

    for text, words in iter_hindi_lines(
        data_root,
        text_column,
        show_progress=show_progress,
        progress_desc=progress_desc,
    ):
        tokens = tokenize_len(text)
        totals["rows"] += 1
        totals["words"] += len(words)
        totals["tokens"] += tokens
        totals["bytes"] += len(text.encode("utf-8"))
        token_counts.update(tokenize_tokens(text))

        if reference_tokenize_len is not None:
            totals["reference_tokens"] += reference_tokenize_len(text)

    renyi_entropy_value = 0.0
    renyi_efficiency_value = 0.0

    if vocab_size is not None and vocab_size > 0:
        renyi_entropy_value, renyi_efficiency_value = renyi_efficiency(
            token_counts,
            vocab_size=vocab_size,
            alpha=renyi_alpha,
        )

    return {
        **totals,
        "fertility": fertility(totals["tokens"], totals["words"]),
        "bytes_per_token": bytes_per_token(totals["bytes"], totals["tokens"]),
        "nsl": (
            normalized_sequence_length(totals["tokens"], totals["reference_tokens"])
            if reference_tokenize_len is not None
            else 0.0
        ),
        "renyi_entropy": renyi_entropy_value,
        "renyi_efficiency": renyi_efficiency_value,
        "unique_tokens_observed": len(token_counts),
    }


def collect_cross_lingual_parity(
    *,
    hindi_tokenize_len: EncodeLenFn,
    reference_lang_tokenize_len: EncodeLenFn,
    parallel_path: Path,
    hindi_column: str,
    reference_column: str,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> IntrinsicMetrics:

    ratios: list[float] = []
    hindi_tokens = 0
    reference_tokens = 0
    rows = 0

    for hindi_text, reference_text in iter_parallel_lines(
        parallel_path,
        hindi_column,
        reference_column,
        show_progress=show_progress,
        progress_desc=progress_desc,
    ):
        hi_count = hindi_tokenize_len(hindi_text)
        ref_count = reference_lang_tokenize_len(reference_text)
        ratios.append(parity_ratio_cross_lingual(hi_count, ref_count))
        hindi_tokens += hi_count
        reference_tokens += ref_count
        rows += 1

    return {
        "rows": rows,
        "parity_ratio": aggregate_parity_ratio(ratios),
        "parity_ratio_micro": parity_ratio_cross_lingual(hindi_tokens, reference_tokens),
        "hindi_tokens": hindi_tokens,
        "reference_lang_tokens": reference_tokens,
    }
