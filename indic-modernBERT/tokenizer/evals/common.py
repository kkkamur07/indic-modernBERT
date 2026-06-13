"""Shared utilities for tokenizer evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import HINDI_LANG3

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
    return Tokenizer.from_file(str(tokenizer_path))


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
) -> Iterator[tuple[str, list[str]]]:
    parquet_files = sorted(data_root.glob(f"verified/{HINDI_LANG3}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Hindi parquet files found under: {data_root}/verified/{HINDI_LANG3}"
        )

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=[text_column])

        for value in table[text_column].to_pylist():
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
) -> Iterator[tuple[str, str]]:

    table = pq.read_table(parallel_path, columns=[hindi_column, reference_column])

    for hindi_value, reference_value in zip(
        table[hindi_column].to_pylist(),
        table[reference_column].to_pylist(),
        strict=True,
    ):
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
) -> IntrinsicMetrics:

    totals = {
        "rows": 0,
        "words": 0,
        "tokens": 0,
        "reference_tokens": 0,
        "bytes": 0,
    }

    token_counts: Counter[str] = Counter()

    for text, words in iter_hindi_lines(data_root, text_column):
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
) -> IntrinsicMetrics:

    ratios: list[float] = []
    hindi_tokens = 0
    reference_tokens = 0
    rows = 0

    for hindi_text, reference_text in iter_parallel_lines(
        parallel_path,
        hindi_column,
        reference_column,
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
