"""Tokenizer training entry points."""

from .bpe_trainer import train_bpe
from .superbpe_trainer import train_superbpe

__all__ = ["train_bpe", "train_superbpe"]
