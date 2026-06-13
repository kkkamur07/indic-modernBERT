"""Shared utilities for tokenizer evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
from omegaconf import DictConfig

from ... import HINDI_LANG3
from ...utils.log_helpers import setup_run_log, slug


def setup_eval_run_log(eval_cfg: DictConfig, prefix: str) -> Path:
    cand = slug(Path(eval_cfg.tokenizer_path).parent.name or Path(eval_cfg.tokenizer_path).stem)
    base_names = get_baseline_names(eval_cfg)
    base = "multi" if len(base_names) > 1 else slug(str(base_names[0]).split("/")[-1])
    data = slug(Path(eval_cfg.data_root).name)
    log_name = f"{prefix}__cand-{cand}__base-{base}__data-{data}.log"
    return setup_run_log(log_name)


def get_baseline_names(eval_cfg: DictConfig) -> list[str]:
    names = eval_cfg.get("baseline_tokenizer_names")

    if names is not None:
        return [str(name) for name in names]

    fallback = eval_cfg.get("baseline_tokenizer_name")

    if fallback is not None:
        return [str(fallback)]

    raise ValueError(
        "Expected one of `baseline_tokenizer_names` or `baseline_tokenizer_name` in eval config."
    )


def collect_stats(
    tokenize_len: Callable[[str], int],
    data_root: Path,
    text_column: str,
) -> dict[str, float]:
    parquet_files = sorted(data_root.glob(f"verified/{HINDI_LANG3}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Hindi parquet files found under: {data_root}/verified/{HINDI_LANG3}"
        )

    stats = {"rows": 0, "words": 0, "tokens": 0, "chars": 0, "bytes": 0}

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

            stats["rows"] += 1
            stats["words"] += len(words)
            stats["tokens"] += tokenize_len(text)
            stats["chars"] += len(text)
            stats["bytes"] += len(text.encode("utf-8"))

    tokens = stats["tokens"]
    words = stats["words"]

    return {
        **stats,
        "fertility": (tokens / words) if words else 0.0,
        "bytes_per_token": (stats["bytes"] / tokens) if tokens else 0.0,
    }
