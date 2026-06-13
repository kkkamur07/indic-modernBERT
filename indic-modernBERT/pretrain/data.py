"""Text dataset loaders for Hindi MLM pretraining.

Port target: _support_repo/ModernBERT/src/text_data.py
"""

from __future__ import annotations

from pathlib import Path


def describe_data_root(data_root: Path) -> str:
    """Return a short summary of parquet shards available for training."""
    shards = sorted(data_root.rglob("*.parquet"))
    if not shards:
        return f"no parquet shards under {data_root}"
    return f"{len(shards)} parquet shard(s) under {data_root}"
