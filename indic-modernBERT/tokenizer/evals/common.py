"""Shared utilities for tokenizer evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import HINDI_LANG3
from tokenizer.pretokenization import preprocess_for_eval

from utils.progress import iter_with_progress

from .metrics import (
    bytes_per_token,
    fertility,
    normalized_sequence_length,
    renyi_efficiency,
)

IntrinsicMetrics = dict[str, float | int]
EncodeLenFn = Callable[[str], int]
EncodeTokensFn = Callable[[str], list[str]]


def short_tokenizer_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def log_intrinsic_metrics(label: str, metrics: IntrinsicMetrics) -> None:
    logger.info(
        "{} | fertility={:.6f} | bytes/token={:.6f} | NSL={:.6f} | "
        "Rényi entropy={:.6f} | Rényi efficiency={:.6f}",
        label,
        metrics["fertility"],
        metrics["bytes_per_token"],
        metrics["nsl"],
        metrics["renyi_entropy"],
        metrics["renyi_efficiency"],
    )


def log_fertility_comparison(fertility_by_label: dict[str, float]) -> None:
    parts = [f"{label}={value:.4f}" for label, value in fertility_by_label.items()]
    logger.info("Fertility comparison | {} | lower is better", " | ".join(parts))


def load_candidate_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path.resolve()}. "
            "Train first with make train-bpe."
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
    use_script_norm: bool = False,
    use_nfkc: bool = False,
    max_shards: int | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> Iterator[tuple[str, list[str]]]:

    # Just get all the parquet files under the data root.
    parquet_files = sorted(data_root.glob("**/*.parquet"))

    if max_shards is not None:
        parquet_files = parquet_files[:max_shards]

    if not parquet_files:
        raise FileNotFoundError(
            f"No Hindi eval parquet files found under {data_root}."
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

            text = preprocess_for_eval(
                str(value),
                use_script_norm=use_script_norm,
                use_nfkc=use_nfkc,
            ).strip()

            if not text:
                continue

            words = text.split()

            if not words:
                continue

            yield text, words


def collect_intrinsic_metrics(
    *,
    tokenize_len: EncodeLenFn,
    tokenize_tokens: EncodeTokensFn,
    data_root: Path,
    text_column: str,
    reference_tokenize_len: EncodeLenFn | None = None,
    vocab_size: int | None = None,
    renyi_alpha: float = 2.5,
    use_script_norm: bool = False,
    use_nfkc: bool = False,
    max_shards: int | None = None,
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
        use_script_norm=use_script_norm,
        use_nfkc=use_nfkc,
        max_shards=max_shards,
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
