"""Pre-tokenization and normalization pipeline for Hindi tokenizer training."""

from .patterns import SUBWORD_SPLIT_PATTERN
from .pipeline import (
    PretokenizationStage,
    apply_nfkc,
    apply_script_normalization,
    build_pre_tokenizer,
    describe_splits,
    normalize_text,
)

__all__ = [
    "PretokenizationStage",
    "SUBWORD_SPLIT_PATTERN",
    "apply_nfkc",
    "apply_script_normalization",
    "build_pre_tokenizer",
    "describe_splits",
    "normalize_text",
]
