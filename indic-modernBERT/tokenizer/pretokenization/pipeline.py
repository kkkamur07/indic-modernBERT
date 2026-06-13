"""Build normalization + regex pre-tokenization for HuggingFace ``tokenizers``."""

from __future__ import annotations

import unicodedata
from typing import Literal

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from tokenizers import Regex, pre_tokenizers

from ... import HINDI_LANG2

from .patterns import SUBWORD_SPLIT_PATTERN, SUPERWORD_SPLIT_PATTERN

PretokenizationStage = Literal["subword", "superword"]

_STAGE_TO_PATTERN: dict[PretokenizationStage, str] = {
    "subword": SUBWORD_SPLIT_PATTERN,
    "superword": SUPERWORD_SPLIT_PATTERN,
}

_hindi_normalizer = IndicNormalizerFactory().get_normalizer(HINDI_LANG2, remove_nuktas=False)


def apply_script_normalization(text: str) -> str:
    return _hindi_normalizer.normalize(text)


def apply_nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_text(
    text: str,
    *,
    use_script_norm: bool = True,
    use_nfkc: bool = True,
) -> str:

    if use_script_norm:
        text = apply_script_normalization(text)
    if use_nfkc:
        text = apply_nfkc(text)
    return text


def build_pre_tokenizer(stage: PretokenizationStage = "subword") -> pre_tokenizers.PreTokenizer:

    pattern = _STAGE_TO_PATTERN[stage]

    return pre_tokenizers.Split(Regex(pattern), behavior="isolated")


def describe_splits(
    text: str,
    *,
    stage: PretokenizationStage = "subword",
    use_script_norm: bool = True,
    use_nfkc: bool = True,
) -> list[tuple[str, tuple[int, int]]]:

    text = normalize_text(text, use_script_norm=use_script_norm, use_nfkc=use_nfkc)

    # Using hugging face tokenizers to build the pre-tokenizer
    return build_pre_tokenizer(stage).pre_tokenize_str(text)
