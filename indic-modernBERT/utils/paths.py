"""Path helpers for training runs."""

from __future__ import annotations

from pathlib import Path


def resolve_from_cwd(path: Path | str) -> Path:
    """Resolve a config path against the process cwd (Hydra runs from repo root)."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (Path.cwd() / resolved).resolve()


def resolve_vocab_output_dir(
    base_output_dir: Path,
    vocab_sizes: list[int],
    vocab_size: int,
) -> Path:
    if len(vocab_sizes) == 1:
        return base_output_dir

    return base_output_dir.parent / f"{base_output_dir.name}_vs{vocab_size}"
