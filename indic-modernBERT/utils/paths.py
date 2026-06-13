"""Path helpers for training runs."""

from __future__ import annotations

from pathlib import Path


def resolve_vocab_output_dir(
    base_output_dir: Path,
    vocab_sizes: list[int],
    vocab_size: int,
) -> Path:
    if len(vocab_sizes) == 1:
        return base_output_dir

    return base_output_dir.parent / f"{base_output_dir.name}_vs{vocab_size}"
