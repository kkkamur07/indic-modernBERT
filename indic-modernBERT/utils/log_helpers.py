"""Shared logging helpers for tokenizer training and evaluations."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from loguru import logger


def slug(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")[:80]


def setup_run_log(log_name: str) -> Path:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log_path = out_dir / log_name
    logger.remove()
    logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )
    return log_path
