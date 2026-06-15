"""Flushed progress lines for training (data pipeline + model + Composer callbacks)."""

from __future__ import annotations

import os
import sys
import threading

from loguru import logger

_lock = threading.Lock()
_counters: dict[str, int] = {}


def step_log(category: str, message: str, *, always: bool = False) -> None:
    """Log one progress line to loguru + stderr (visible in nohup / train.log)."""
    if not always and os.environ.get("TRAIN_STEP_LOG", "1") == "0":
        return
    logger.info("{} | {}", category, message)
    sys.stderr.flush()


def bump(name: str) -> int:
    with _lock:
        _counters[name] = _counters.get(name, 0) + 1
        return _counters[name]


def should_log_detail(name: str, *, first_n: int = 10, every_n: int = 0) -> bool:
    """Log the next event for ``name`` if within the first N or every Nth."""
    with _lock:
        n = _counters.get(name, 0) + 1
    if n <= first_n:
        return True
    return bool(every_n) and n % every_n == 0
