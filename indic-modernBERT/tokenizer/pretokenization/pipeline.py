"""Build normalization + regex pre-tokenization for HuggingFace ``tokenizers``."""

from __future__ import annotations

import unicodedata
from typing import Literal

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from tokenizers import Regex, pre_tokenizers

from constants import HINDI_LANG2

from .patterns import SUBWORD_SPLIT_PATTERN

PretokenizationStage = Literal["subword", "superword"]


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


def preprocess_for_tokenizer(text: str, *, use_script_norm: bool = True) -> str:
    """Prepare raw text for our BPE tokenizer (NFKC is applied inside ``tokenizer.json``)."""
    return normalize_text(text, use_script_norm=use_script_norm, use_nfkc=False)


def preprocess_for_eval(
    text: str,
    *,
    use_script_norm: bool = True,
    use_nfkc: bool = True,
) -> str:
    """Prepare raw text for intrinsic eval (same normalization for every tokenizer)."""
    return normalize_text(text, use_script_norm=use_script_norm, use_nfkc=use_nfkc)


def build_pre_tokenizer(
    stage: PretokenizationStage = "subword",
) -> pre_tokenizers.PreTokenizer | None:
    if stage == "superword":
        return None

    return pre_tokenizers.Split(Regex(SUBWORD_SPLIT_PATTERN), behavior="isolated")


def describe_splits(
    text: str,
    *,
    stage: PretokenizationStage = "subword",
    use_script_norm: bool = True,
    use_nfkc: bool = True,
) -> list[tuple[str, tuple[int, int]]]:

    text = normalize_text(text, use_script_norm=use_script_norm, use_nfkc=use_nfkc)

    pre_tokenizer = build_pre_tokenizer(stage)
    if pre_tokenizer is None:
        return [(text, (0, len(text)))]

    return pre_tokenizer.pre_tokenize_str(text)
