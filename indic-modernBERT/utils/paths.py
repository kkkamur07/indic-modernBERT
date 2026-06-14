"""Path helpers for training runs."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    """Return repo root (has indic-modernBERT/ + configs/), or None."""
    cwd = Path.cwd() if start is None else Path(start)
    for candidate in (cwd, *cwd.parents):
        if (candidate / "indic-modernBERT").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return None


def resolve_from_cwd(path: Path | str) -> Path:
    """Resolve a relative config path from cwd, then from detected repo root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved

    cwd = Path.cwd()
    for base in (cwd, find_repo_root(cwd)):
        if base is None:
            continue
        candidate = (base / resolved).resolve()
        if candidate.exists():
            return candidate
    return (cwd / resolved).resolve()


def resolve_vocab_output_dir(
    base_output_dir: Path,
    vocab_sizes: list[int],
    vocab_size: int,
) -> Path:
    if len(vocab_sizes) == 1:
        return base_output_dir

    return base_output_dir.parent / f"{base_output_dir.name}_vs{vocab_size}"


def resolve_hf_tokenizer_dir(tokenizer_path: Path | str) -> Path:
    """Return a directory suitable for ``PreTrainedTokenizerFast.from_pretrained``."""
    path = Path(tokenizer_path)
    if path.is_file() and path.name == "tokenizer.json":
        return path.parent
    return path
