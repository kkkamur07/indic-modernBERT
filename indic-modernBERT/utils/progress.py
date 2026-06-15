"""Progress reporting for long parquet iteration (TTY vs log file)."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from loguru import logger
from tqdm import tqdm

T = TypeVar("T")


def iter_with_progress(
    values: Iterable[T],
    *,
    total: int,
    desc: str,
    unit: str = "rows",
    show_progress: bool = True,
) -> Iterator[T]:

    if not show_progress:
        yield from values
        return

    if sys.stderr.isatty():
        yield from tqdm(values, total=total, desc=desc, unit=unit, mininterval=0.5)
        return

    logger.info("{} | {} {}", desc, total, unit)
    yield from values
