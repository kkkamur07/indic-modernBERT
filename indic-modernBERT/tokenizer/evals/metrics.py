"""Intrinsic tokenizer metric definitions (MUTANT / Zouhar et al. / Petrov et al.)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def fertility(tokens: int, words: int) -> float:
    """Tokens per whitespace-delimited word (lower is better)."""
    return (tokens / words) if words else 0.0


def bytes_per_token(total_bytes: int, tokens: int) -> float:
    """UTF-8 bytes per token (higher is better)."""
    return (total_bytes / tokens) if tokens else 0.0


def normalized_sequence_length(candidate_tokens: int, reference_tokens: int) -> float:
    """NSL relative to a reference tokenizer (lower is better).

    c_{λ/β} = Σ|t_λ(x_i)| / Σ|t_β(x_i)|  (Dagan et al., 2024; MUTANT §3.4)
    """
    return (candidate_tokens / reference_tokens) if reference_tokens else 0.0


def parity_ratio_cross_lingual(hindi_tokens: int, reference_lang_tokens: int) -> float:
    """Per-sentence |t(hi)| / |t(ref)| (Petrov et al., 2023). Closer to 1 is better."""
    return (hindi_tokens / reference_lang_tokens) if reference_lang_tokens else 0.0


def aggregate_parity_ratio(per_line_ratios: Iterable[float]) -> float:
    ratios = list(per_line_ratios)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def renyi_entropy(probabilities: list[float], *, alpha: float) -> float:
    """Rényi entropy of a unigram token distribution (Zouhar et al., 2023)."""
    positive = [prob for prob in probabilities if prob > 0.0]
    if not positive:
        return 0.0

    if math.isclose(alpha, 1.0):
        return -sum(prob * math.log2(prob) for prob in positive)

    scale = 1.0 / (1.0 - alpha)
    return scale * math.log2(sum(prob**alpha for prob in positive))


def renyi_efficiency(
    token_counts: Counter[str],
    *,
    vocab_size: int,
    alpha: float = 2.5,
) -> tuple[float, float]:
    """Rényi efficiency = H_alpha / log2(|V|). Higher is better (MUTANT Table 7)."""

    if vocab_size <= 1:
        return 0.0, 0.0

    total = sum(token_counts.values())

    if total == 0:
        return 0.0, 0.0

    probabilities = [count / total for count in token_counts.values()]
    entropy = renyi_entropy(probabilities, alpha=alpha)
    max_entropy = math.log2(vocab_size)
    efficiency = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return entropy, efficiency
