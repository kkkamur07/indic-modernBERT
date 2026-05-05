"""Shared utilities for tokenizer evaluation scripts."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig
import pyarrow.parquet as pq

try:
    from ..utils.log_helpers import setup_run_log, slug
except ImportError:
    from tokenizer.utils.log_helpers import setup_run_log, slug

# Helper functions

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


LANG_RE = re.compile(r"verified/(?P<lang>[^/]+)/")


def collect_stats(
    tokenize_len: Callable[[str], int],
    data_root: Path,
    text_column: str,
) -> dict[str, object]:

    parquet_files = sorted(data_root.glob("verified/*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_root}")

    stats = defaultdict(
        lambda: {"words": 0, "tokens": 0, "rows": 0, "chars": 0, "bytes": 0}
    )

    for parquet_path in parquet_files:
        match = LANG_RE.search(parquet_path.as_posix())

        if match is None:
            continue

        lang = match["lang"]
        
        # TODO : Pyarrow has better support for faster reading, so need to check this. 
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

            stats[lang]["rows"] += 1
            stats[lang]["words"] += len(words)
            stats[lang]["tokens"] += tokenize_len(text)
            stats[lang]["chars"] += len(text)
            stats[lang]["bytes"] += len(text.encode("utf-8"))

    per_language: dict[str, dict[str, float]] = {}
    totals = {"rows": 0, "words": 0, "tokens": 0, "chars": 0, "bytes": 0}

    for lang in sorted(stats):
        row = stats[lang]
        tokens = row["tokens"]
        words = row["words"]

        per_language[lang] = {
            "rows": row["rows"],
            "words": words,
            "tokens": tokens,
            "chars": row["chars"],
            "bytes": row["bytes"],
            "fertility": (tokens / words) if words else 0.0,
            "bytes_per_token": (row["bytes"] / tokens) if tokens else 0.0,
        }
        
        for k in totals:
            totals[k] += row[k]

    overall_tokens = totals["tokens"]
    overall_words = totals["words"]

    overall = {
        **totals,
        "fertility": (overall_tokens / overall_words) if overall_words else 0.0,
        "bytes_per_token": (totals["bytes"] / overall_tokens) if overall_tokens else 0.0,
    }
    
    return {"overall": overall, "per_language": per_language}


def parity_from_metric(per_language: dict[str, dict[str, float]], metric: str) -> dict[str, float]:
    values = [float(row[metric]) for row in per_language.values()]

    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "range": 0.0,
            "parity_ratio": 0.0,
            "coefficient_of_variation": 0.0,
        }
        
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    min_v = min(values)
    max_v = max(values)

    return {
        "mean": mean,
        "min": min_v,
        "max": max_v,
        "std": std,
        "range": max_v - min_v,
        "parity_ratio": (min_v / max_v) if max_v else 0.0,
        "coefficient_of_variation": (std / mean) if mean else 0.0,
    }

